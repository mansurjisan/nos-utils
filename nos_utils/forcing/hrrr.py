"""
HRRR (High-Resolution Rapid Refresh) forcing processor.

Processes HRRR 3km GRIB2 data as a secondary atmospheric forcing source.
HRRR provides higher resolution over CONUS but has limited domain and forecast length.

Input: HRRR GRIB2 files from COMINhrrr
  Pattern: hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfHH.grib2
  Resolution: 3km (Lambert Conformal projection)
  Max forecast: 48 hours

Key differences from GFS:
  - Uses MSLMA instead of PRMSL for sea level pressure
  - Lambert Conformal grid — MUST regrid to regular lat/lon (lesson #14)
  - Optional source — failure is non-fatal
  - Writes to source_index=2 (secondary) for SCHISM sflux blending

Output:
  sflux mode: sflux_air_2.N.nc, sflux_rad_2.N.nc, sflux_prc_2.N.nc
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ForcingConfig
from ..io.grib_extract import GRIBExtractor, get_extractor
from .base import ForcingProcessor, ForcingResult
from .sflux_writer import SfluxWriter

log = logging.getLogger(__name__)


class HRRRProcessor(ForcingProcessor):
    """
    HRRR atmospheric forcing processor (secondary source).

    HRRR failure is always non-fatal — returns success=True with warnings.
    GFS provides primary coverage; HRRR is an enhancement where available.
    """

    SOURCE_NAME = "HRRR"
    MIN_FILE_SIZE = 0  # HRRR is optional, no size enforcement
    MAX_FORECAST_HOURS = 48

    # GRIB2 variable mapping (note: MSLMA not PRMSL)
    GRIB2_VARIABLES = {
        "uwind": ("UGRD", "10 m above ground"),
        "vwind": ("VGRD", "10 m above ground"),
        "prmsl": ("MSLMA", "mean sea level"),
        "stmp": ("TMP", "2 m above ground"),
        "spfh": ("SPFH", "2 m above ground"),
        "dlwrf": ("DLWRF", "surface"),
        "dswrf": ("DSWRF", "surface"),
        "prate": ("PRATE", "surface"),
    }

    DEFAULT_VARIABLES = ["uwind", "vwind", "prmsl", "stmp", "spfh", "dlwrf", "dswrf", "prate"]

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        variables: Optional[List[str]] = None,
        regrid_dx: float = 0.03,
        extractor: Optional[GRIBExtractor] = None,
        phase: str = "nowcast",
        time_hotstart: Optional[datetime] = None,
    ):
        """
        Args:
            config: ForcingConfig with domain, cycle, and run window
            input_path: Root HRRR data directory (COMINhrrr)
            output_path: Output directory for sflux files
            variables: Variables to extract (default: 8 core sflux vars)
            regrid_dx: Target resolution after LCC->latlon regrid (degrees, ~3.3km)
            extractor: GRIB2 extractor (auto-detected if None)
            phase: "nowcast" or "forecast"
            time_hotstart: Hotstart datetime
        """
        super().__init__(config, input_path, output_path)
        self.variables = variables or self.DEFAULT_VARIABLES
        self.regrid_dx = regrid_dx
        self._extractor = extractor
        self.phase = phase
        self.time_hotstart = time_hotstart

    @property
    def extractor(self) -> GRIBExtractor:
        if self._extractor is None:
            self._extractor = get_extractor()
        return self._extractor

    def process(self) -> ForcingResult:
        """
        Process HRRR forcing data.

        HRRR is optional — all failures return success=True with warnings.
        """
        log.info(f"HRRR processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z phase={self.phase}")

        if not self.input_path.exists():
            return ForcingResult(
                success=True, source=self.SOURCE_NAME,
                warnings=[f"HRRR input path not found: {self.input_path}"],
            )

        self.create_output_dir()

        try:
            # Step 1: Find files
            hrrr_files = self.find_input_files()
            if not hrrr_files:
                return ForcingResult(
                    success=True, source=self.SOURCE_NAME,
                    warnings=["No HRRR files found — using GFS only"],
                )
            log.info(f"Found {len(hrrr_files)} HRRR files")

            # Step 2: Extract data
            # igrd_met=0: pass native LCC grid (matches Fortran behavior)
            # igrd_met>0: regrid to regular lat/lon
            if self.config.igrd_met == 0:
                extracted = self._extract_native(hrrr_files)
            else:
                extracted = self._extract_all(hrrr_files)
            if not extracted["times"]:
                return ForcingResult(
                    success=True, source=self.SOURCE_NAME,
                    warnings=["Could not extract HRRR data"],
                )

            # Step 3: Filter to phase-specific time window
            extracted = self._filter_to_time_window(extracted)

            # Step 4: Write sflux (source_index=2 for secondary)
            writer = SfluxWriter(self.output_path, source_index=2)
            base_date = self._compute_base_date()
            files = writer.write_all(
                extracted["data"], extracted["times"],
                extracted["lons"], extracted["lats"], base_date,
            )

            return ForcingResult(
                success=True, source=self.SOURCE_NAME,
                output_files=files,
                metadata={
                    "num_input_files": len(hrrr_files),
                    "num_timesteps": len(extracted["times"]),
                    "regrid_dx": self.regrid_dx,
                },
            )

        except Exception as e:
            log.error(f"HRRR processing failed (non-fatal): {e}")
            return ForcingResult(
                success=True, source=self.SOURCE_NAME,
                warnings=[f"HRRR processing failed: {e}"],
            )

    def find_input_files(self) -> List[Path]:
        """
        Find HRRR GRIB2 files for nowcast + forecast window.

        Nowcast: hourly analysis files (f01) from multiple cycles
        Forecast: hourly forecast files (f01-f48) from current cycle
        """
        hrrr_files = []
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        prev_date = base_date - timedelta(days=1)

        prev_path = self._resolve_path(prev_date)
        today_path = self._resolve_path(base_date)

        # Nowcast: hourly f01 from previous hours
        # E.g., for 12z cycle with 24h nowcast: yesterday t12z-t23z + today t00z-t11z
        nowcast_start_hour = self.config.cyc - self.config.nowcast_hours
        if nowcast_start_hour < 0:
            # Spans previous day
            start_hour_prev = 24 + nowcast_start_hour
            for hr in range(start_hour_prev, 24):
                f = self._find_file(prev_path, hr, 1)
                if f:
                    hrrr_files.append(f)
            for hr in range(0, self.config.cyc):
                f = self._find_file(today_path, hr, 1)
                if f:
                    hrrr_files.append(f)
        else:
            for hr in range(nowcast_start_hour, self.config.cyc):
                f = self._find_file(today_path, hr, 1)
                if f:
                    hrrr_files.append(f)

        # Forecast: f01-f48 from current cycle
        max_fhr = min(self.config.forecast_hours, self.MAX_FORECAST_HOURS)
        for fhr in range(1, max_fhr + 1):
            f = self._find_file(today_path, self.config.cyc, fhr)
            if f:
                hrrr_files.append(f)

        return hrrr_files

    def _resolve_path(self, date: datetime) -> Path:
        """Resolve HRRR directory path for a date."""
        date_str = date.strftime("%Y%m%d")
        conus_path = self.input_path / f"hrrr.{date_str}" / "conus"
        if conus_path.exists():
            return conus_path
        return self.input_path

    def _find_file(self, base_path: Path, cycle_hour: int, fhr: int) -> Optional[Path]:
        """Find a single HRRR file by cycle and forecast hour."""
        patterns = [
            f"hrrr.t{cycle_hour:02d}z.wrfsfcf{fhr:02d}.grib2",
            f"hrrr.t{cycle_hour:02d}z.wrfsfcf{fhr:02d}.*.grib2",
        ]
        for pattern in patterns:
            found = sorted(base_path.glob(pattern))
            if found:
                return found[0]
        return None

    def _extract_all(self, hrrr_files: List[Path]) -> dict:
        """Extract variables from HRRR files, regridding LCC -> regular lat/lon.

        Two regrid strategies:
        1. wgrib2 -new_grid (fast, requires IPOLATES)
        2. Python scipy interpolation (fallback, always works)
        """
        result = {"times": [], "lons": None, "lats": None, "data": {}}
        for var in self.variables:
            result["data"][var] = []

        domain = self.config.domain
        lon_min, lon_max, lat_min, lat_max = domain
        # Compute nx/ny exactly as wgrib2 -new_grid does: nx = round((max-min)/dx) + 1
        nx = int(round((lon_max - lon_min) / self.regrid_dx)) + 1
        ny = int(round((lat_max - lat_min) / self.regrid_dx)) + 1
        target_lons = np.linspace(lon_min, lon_min + (nx - 1) * self.regrid_dx, nx)
        target_lats = np.linspace(lat_min, lat_min + (ny - 1) * self.regrid_dx, ny)
        result["lons"] = target_lons
        result["lats"] = target_lats

        import tempfile
        tmpdir = Path(tempfile.mkdtemp(prefix="hrrr_regrid_"))

        # Interpolation index (built once from first file)
        interp_index = None

        try:
            for hrrr_file in hrrr_files:
                valid_time = self._parse_valid_time(hrrr_file)
                if valid_time is None:
                    continue

                # Try wgrib2 -new_grid first (needs IPOLATES; U+V must be together)
                regridded = tmpdir / f"regrid_{hrrr_file.stem}.grb2"
                # Build match pattern that includes all vars WITH levels (so U+V pair works)
                match_parts = []
                for var in self.variables:
                    if var in self.GRIB2_VARIABLES:
                        grib_var, level = self.GRIB2_VARIABLES[var]
                        match_parts.append(f"{grib_var}:{level}")
                match_pattern = ":(" + "|".join(match_parts) + "):"

                regridded_path = self.extractor.regrid_to_latlon(
                    hrrr_file, domain, self.regrid_dx, regridded,
                    match_pattern=match_pattern,
                )

                if regridded_path is not None:
                    # wgrib2 regrid worked — extract from regridded file
                    # skip_subset=True: file already has exact target grid dimensions,
                    # re-subsetting would trim rows due to float boundary matching
                    result["times"].append(valid_time)
                    for var in self.variables:
                        if var not in self.GRIB2_VARIABLES:
                            continue
                        grib_var, level = self.GRIB2_VARIABLES[var]
                        data = self.extractor.extract(
                            regridded_path, grib_var, level, domain, skip_subset=True
                        )
                        if data is not None:
                            # Mask wgrib2 undefined values (9.999e20) as NaN
                            data = np.where(np.abs(data) > 1e10, np.nan, data)
                            result["data"][var].append(data)
                        else:
                            ny, nx = len(target_lats), len(target_lons)
                            result["data"][var].append(np.full((ny, nx), np.nan, dtype=np.float32))
                    regridded_path.unlink(missing_ok=True)
                else:
                    # Fallback: Python interpolation via wgrib2 -spread
                    file_data = self._extract_spread_interpolate(
                        hrrr_file, domain, target_lons, target_lats, interp_index, tmpdir
                    )
                    if file_data is None:
                        continue

                    interp_index = file_data.get("_interp_index", interp_index)
                    result["times"].append(valid_time)
                    for var in self.variables:
                        if var in file_data:
                            result["data"][var].append(file_data[var])
                        else:
                            ny, nx = len(target_lats), len(target_lons)
                            result["data"][var].append(np.full((ny, nx), np.nan, dtype=np.float32))

        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

        # Sort by time
        if result["times"]:
            sorted_idx = sorted(range(len(result["times"])), key=lambda i: result["times"][i])
            result["times"] = [result["times"][i] for i in sorted_idx]
            for var in result["data"]:
                if result["data"][var]:
                    n_data = len(result["data"][var])
                    if n_data != len(sorted_idx):
                        log.warning(f"Sort: {var} has {n_data} entries but "
                                    f"expected {len(sorted_idx)}")
                    result["data"][var] = [result["data"][var][i] for i in sorted_idx
                                           if i < n_data]

        return result

    def _extract_native(self, hrrr_files: List[Path]) -> dict:
        """Extract HRRR data on native Lambert Conformal grid (no regrid).

        Matches Fortran nos_ofs_create_forcing_met behavior with IGRD=0:
        pass the native HRRR grid through to sflux, storing 2D lon/lat.
        """
        import subprocess
        import tempfile

        result = {"times": [], "lons": None, "lats": None, "data": {}}
        for var in self.variables:
            result["data"][var] = []

        tmpdir = Path(tempfile.mkdtemp(prefix="hrrr_native_"))
        native_lons = None
        native_lats = None

        try:
            for hrrr_file in hrrr_files:
                valid_time = self._parse_valid_time(hrrr_file)
                if valid_time is None:
                    continue

                file_data = {}
                for var in self.variables:
                    if var not in self.GRIB2_VARIABLES:
                        continue
                    grib_var, level = self.GRIB2_VARIABLES[var]
                    match_str = f":{grib_var}:{level}:"

                    # Extract to binary on native grid (no regrid, no subset)
                    subset_file = tmpdir / f"match_{hrrr_file.stem}_{var}.grb2"
                    bin_file = tmpdir / f"data_{hrrr_file.stem}_{var}.bin"

                    cmd1 = [
                        self.extractor.wgrib2, str(hrrr_file),
                        "-match", match_str,
                        "-grib", str(subset_file),
                    ]
                    r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)
                    if r1.returncode != 0 or not subset_file.exists():
                        continue

                    # Dump first record to binary
                    cmd2 = [
                        self.extractor.wgrib2, str(subset_file),
                        "-d", "1",
                        "-no_header", "-bin", str(bin_file),
                    ]
                    r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
                    if r2.returncode != 0 or not bin_file.exists():
                        subset_file.unlink(missing_ok=True)
                        continue

                    data = np.fromfile(bin_file, dtype=np.float32)
                    nx, ny = self.extractor._get_nxny(subset_file)

                    if nx and ny and data.size == nx * ny:
                        field = data.reshape((ny, nx))
                        # Mask fill values
                        field = np.where(np.abs(field) > 1e10, np.nan, field)
                        file_data[var] = field
                    else:
                        log.debug(f"Grid dim mismatch for {var}: size={data.size}, nx={nx}, ny={ny}")

                    # Get native lon/lat from first successful extraction
                    if native_lons is None and nx and ny:
                        lon_file = tmpdir / "lon.bin"
                        lat_file = tmpdir / "lat.bin"
                        cmd_lon = [
                            self.extractor.wgrib2, str(subset_file),
                            "-d", "1", "-no_header",
                            "-lon_grib", str(lon_file),
                        ]
                        cmd_lat = [
                            self.extractor.wgrib2, str(subset_file),
                            "-d", "1", "-no_header",
                            "-lat_grib", str(lat_file),
                        ]
                        # Use rpn to dump lon/lat
                        cmd_lonlat = [
                            self.extractor.wgrib2, str(subset_file),
                            "-d", "1",
                            "-rpn", "rcl_lon", "-no_header", "-bin", str(lon_file),
                        ]
                        r = subprocess.run(cmd_lonlat, capture_output=True, text=True, timeout=60)
                        if r.returncode == 0 and lon_file.exists():
                            lons = np.fromfile(lon_file, dtype=np.float32).reshape((ny, nx))
                            # Now get lat
                            cmd_lat2 = [
                                self.extractor.wgrib2, str(subset_file),
                                "-d", "1",
                                "-rpn", "rcl_lat", "-no_header", "-bin", str(lat_file),
                            ]
                            r2 = subprocess.run(cmd_lat2, capture_output=True, text=True, timeout=60)
                            if r2.returncode == 0 and lat_file.exists():
                                lats = np.fromfile(lat_file, dtype=np.float32).reshape((ny, nx))
                                # Convert lon from 0-360 to -180..180
                                lons = np.where(lons > 180, lons - 360, lons)
                                native_lons = lons
                                native_lats = lats
                                log.info(f"Native HRRR grid: {ny}x{nx}, "
                                         f"lon=[{np.min(lons):.3f}, {np.max(lons):.3f}], "
                                         f"lat=[{np.min(lats):.3f}, {np.max(lats):.3f}]")

                    # Cleanup
                    subset_file.unlink(missing_ok=True)
                    bin_file.unlink(missing_ok=True)

                # Fill land/NaN cells from nearest valid point
                # Matches Fortran behavior at line 2043 of nos_ofs_create_forcing_met.f
                for var in list(file_data.keys()):
                    file_data[var] = self._fill_land_nearest(file_data[var])

                # Rotate grid-relative winds to earth coordinates
                # HRRR Lambert Conformal stores winds relative to the grid,
                # not earth. Fortran does this via w3fc07 subroutine.
                if "uwind" in file_data and "vwind" in file_data and native_lons is not None:
                    u_grid = file_data["uwind"]
                    v_grid = file_data["vwind"]
                    u_earth, v_earth = self._rotate_winds_lcc(
                        u_grid, v_grid, native_lons,
                    )
                    file_data["uwind"] = u_earth
                    file_data["vwind"] = v_earth

                if file_data:
                    result["times"].append(valid_time)
                    for var in self.variables:
                        if var in file_data:
                            result["data"][var].append(file_data[var])
                        elif result["data"][var]:
                            result["data"][var].append(
                                np.full_like(result["data"][var][-1], np.nan))

        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

        if native_lons is not None:
            result["lons"] = native_lons
            result["lats"] = native_lats

        # Sort by time
        if result["times"]:
            sorted_idx = sorted(range(len(result["times"])),
                                key=lambda i: result["times"][i])
            result["times"] = [result["times"][i] for i in sorted_idx]
            for var in result["data"]:
                if result["data"][var]:
                    n_data = len(result["data"][var])
                    result["data"][var] = [result["data"][var][i]
                                           for i in sorted_idx if i < n_data]

        return result

    def _extract_spread_interpolate(
        self,
        hrrr_file: Path,
        domain: tuple,
        target_lons: np.ndarray,
        target_lats: np.ndarray,
        interp_index: Optional[dict],
        tmpdir: Path,
    ) -> Optional[dict]:
        """
        Extract HRRR data via wgrib2 -spread and interpolate LCC -> regular lat/lon.

        Uses scipy Delaunay triangulation (built once, reused for all files/variables).
        """
        import shutil
        import subprocess

        wgrib2 = shutil.which("wgrib2")
        if not wgrib2:
            log.warning("wgrib2 not found — cannot extract HRRR")
            return None

        try:
            from scipy.interpolate import griddata
        except ImportError:
            log.warning("scipy required for HRRR Python interpolation")
            return None

        lon_min, lon_max, lat_min, lat_max = domain
        ny_target, nx_target = len(target_lats), len(target_lons)
        target_lon_2d, target_lat_2d = np.meshgrid(target_lons, target_lats)

        file_data = {}

        for var in self.variables:
            if var not in self.GRIB2_VARIABLES:
                continue
            grib_var, level = self.GRIB2_VARIABLES[var]
            match_str = f":{grib_var}:{level}:"

            # Step 1: Subset to domain
            subset = tmpdir / f"sub_{var}_{hrrr_file.stem}.grb2"
            cmd = [wgrib2, str(hrrr_file), "-match", match_str,
                   "-small_grib", f"{lon_min}:{lon_max}", f"{lat_min}:{lat_max}",
                   str(subset)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0 or not subset.exists():
                continue

            # Step 2: Dump lon/lat/value via -spread
            spread_file = tmpdir / f"spread_{var}_{hrrr_file.stem}.txt"
            cmd2 = [wgrib2, str(subset), "-d", "1", "-spread", str(spread_file)]
            r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
            if r2.returncode != 0 or not spread_file.exists():
                subset.unlink(missing_ok=True)
                continue

            # Step 3: Parse spread output (lon,lat,value per line, skip header)
            lons_lcc = []
            lats_lcc = []
            vals = []
            with open(spread_file) as f:
                for i, line in enumerate(f):
                    if i == 0:  # skip header
                        continue
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        try:
                            lo = float(parts[0])
                            la = float(parts[1])
                            va = float(parts[2])
                            # Convert 0-360 to -180-180 if needed
                            if lo > 180:
                                lo -= 360.0
                            lons_lcc.append(lo)
                            lats_lcc.append(la)
                            vals.append(va)
                        except ValueError:
                            continue

            subset.unlink(missing_ok=True)
            spread_file.unlink(missing_ok=True)

            if not vals:
                continue

            lons_lcc = np.array(lons_lcc, dtype=np.float32)
            lats_lcc = np.array(lats_lcc, dtype=np.float32)
            vals = np.array(vals, dtype=np.float32)

            # Step 4: Interpolate LCC -> regular lat/lon
            points = np.column_stack([lons_lcc, lats_lcc])
            regridded = griddata(
                points, vals,
                (target_lon_2d, target_lat_2d),
                method="linear", fill_value=np.nan,
            ).astype(np.float32)

            file_data[var] = regridded
            log.debug(f"Interpolated {var}: {np.nanmin(regridded):.2f}..{np.nanmax(regridded):.2f}")

        if not file_data:
            return None

        return file_data

    def _parse_valid_time(self, hrrr_file: Path) -> Optional[datetime]:
        """Parse valid time from HRRR filename and parent directory."""
        try:
            # Extract cycle hour: hrrr.tHHz.wrfsfcfHH.grib2
            name = hrrr_file.name
            cyc_str = name.split(".t")[1][:2]
            cyc_hour = int(cyc_str)

            fhr_str = name.split("wrfsfcf")[1][:2]
            fhr = int(fhr_str)

            # Get date from parent dir: hrrr.YYYYMMDD/conus/
            for parent in [hrrr_file.parent, hrrr_file.parent.parent]:
                if parent.name.startswith("hrrr."):
                    date_str = parent.name.split("hrrr.")[1][:8]
                    break
            else:
                date_str = self.config.pdy

            base = datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=cyc_hour)
            return base + timedelta(hours=fhr)

        except (ValueError, IndexError):
            log.warning(f"Cannot parse time from {hrrr_file.name}")
            return None

    def _get_time_window(self) -> Tuple[datetime, datetime]:
        """Compute time window for this phase (same logic as GFS)."""
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)

        if self.phase == "nowcast":
            if self.time_hotstart:
                t_start = self.time_hotstart - timedelta(hours=3)
            else:
                t_start = cycle_dt - timedelta(hours=self.config.nowcast_hours) - timedelta(hours=3)
            t_end = cycle_dt + timedelta(hours=3)
        elif self.phase == "forecast":
            t_start = cycle_dt - timedelta(hours=3)
            t_end = cycle_dt + timedelta(hours=self.config.forecast_hours) + timedelta(hours=3)
        else:
            if self.time_hotstart:
                t_start = self.time_hotstart - timedelta(hours=3)
            else:
                t_start = cycle_dt - timedelta(hours=self.config.nowcast_hours) - timedelta(hours=3)
            t_end = cycle_dt + timedelta(hours=self.config.forecast_hours) + timedelta(hours=3)

        return t_start, t_end

    def _filter_to_time_window(self, extracted: dict) -> dict:
        """Filter extracted data to phase-specific time window."""
        t_start, t_end = self._get_time_window()
        times = extracted["times"]

        keep = [i for i, t in enumerate(times) if t_start <= t <= t_end]

        if len(keep) == len(times):
            return extracted

        n_dropped = len(times) - len(keep)
        log.info(f"Time window [{t_start} to {t_end}]: kept {len(keep)}/{len(times)}")

        filtered = {
            "times": [times[i] for i in keep],
            "lons": extracted["lons"],
            "lats": extracted["lats"],
            "data": {},
        }
        for var, arrays in extracted["data"].items():
            if len(arrays) != len(times):
                log.warning(f"Filter: {var} has {len(arrays)} entries but "
                            f"expected {len(times)}")
            filtered["data"][var] = [arrays[i] for i in keep if i < len(arrays)]

        return filtered

    @staticmethod
    def _fill_land_nearest(field: np.ndarray) -> np.ndarray:
        """Fill NaN (land) cells from nearest valid (ocean) point.

        Matches Fortran nos_ofs_create_forcing_met behavior where land
        cells are filled from nearest coastal points before writing sflux.
        """
        nan_mask = np.isnan(field)
        if not np.any(nan_mask) or np.all(nan_mask):
            return field

        try:
            from scipy.ndimage import distance_transform_edt

            # Use distance transform to find nearest valid pixel for each NaN
            valid_mask = ~nan_mask
            _, nearest_idx = distance_transform_edt(
                nan_mask, return_distances=True, return_indices=True,
            )
            filled = field.copy()
            filled[nan_mask] = field[tuple(nearest_idx[:, nan_mask])]
            return filled
        except ImportError:
            return field

    @staticmethod
    def _rotate_winds_lcc(
        u_grid: np.ndarray, v_grid: np.ndarray, lons: np.ndarray,
        lov: float = -97.5, latin1: float = 38.5, latin2: float = 38.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate Lambert Conformal grid-relative winds to earth coordinates.

        Port of NCEP w3fc07 subroutine used by Fortran nos_ofs_create_forcing_met.
        HRRR standard parameters: lov=-97.5, latin1=38.5, latin2=38.5.

        Args:
            u_grid: U-wind on grid (grid-relative)
            v_grid: V-wind on grid (grid-relative)
            lons: 2D longitude array (degrees, -180..180)
            lov: Longitude of orientation (center longitude of projection)
            latin1, latin2: Standard parallels

        Returns:
            (u_earth, v_earth) — winds rotated to earth coordinates
        """
        # Cone factor for Lambert Conformal
        if abs(latin1 - latin2) < 1e-6:
            cone = np.sin(np.radians(abs(latin1)))
        else:
            cone = (np.log(np.cos(np.radians(latin1))) -
                    np.log(np.cos(np.radians(latin2)))) / \
                   (np.log(np.tan(np.radians(45 - latin1 / 2))) -
                    np.log(np.tan(np.radians(45 - latin2 / 2))))

        # Rotation angle at each grid point
        angle = cone * np.radians(lons - lov)

        sin_a = np.sin(angle)
        cos_a = np.cos(angle)

        u_earth = cos_a * u_grid + sin_a * v_grid
        v_earth = -sin_a * u_grid + cos_a * v_grid

        return u_earth.astype(np.float32), v_earth.astype(np.float32)

    def _compute_base_date(self) -> datetime:
        """Compute sflux base date (start of model simulation).

        Uses time_hotstart if available, otherwise cycle - nowcast_hours.
        Phase-independent: sflux uses a continuous time axis across
        nowcast and forecast, so both must share the same base_date.
        """
        if self.time_hotstart:
            return self.time_hotstart
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + timedelta(hours=self.config.cyc)
        return cycle_dt - timedelta(hours=self.config.nowcast_hours)
