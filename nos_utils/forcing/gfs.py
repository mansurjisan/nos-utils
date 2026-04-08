"""
GFS (Global Forecast System) forcing processor.

Processes GFS 0.25° GRIB2 data and creates SCHISM sflux or DATM forcing files.

Input: GFS GRIB2 files from COMINgfs
  Pattern: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fHHH
  Resolution: 0.25° (hourly) or 0.50° (3-hourly)
  Min file size: 500 MB

Output:
  sflux mode (nws=2): sflux_air_1.N.nc, sflux_rad_1.N.nc, sflux_prc_1.N.nc
  DATM mode (nws=4):  datm_forcing.nc
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
from .datm_writer import DATMWriter

log = logging.getLogger(__name__)


class GFSProcessor(ForcingProcessor):
    """
    GFS atmospheric forcing processor.

    Extracts meteorological variables from GFS GRIB2 files and writes
    SCHISM-compatible sflux NetCDF files or DATM forcing for UFS-Coastal.
    """

    SOURCE_NAME = "GFS"
    # Minimum file size by resolution (bytes).
    # GFS 0.25°: ~500 MB, GFS 0.50°: ~60 MB per file.
    MIN_FILE_SIZE_BY_RES = {
        "0p25": 500_000_000,  # 500 MB (matches shell FILESIZE=500000000)
        "0p50": 40_000_000,   # 40 MB
    }
    MIN_FILE_SIZE = 40_000_000  # fallback: 40 MB

    # GRIB2 variable mapping: internal name -> (GRIB2 name, level)
    GRIB2_VARIABLES = {
        # Core 8 variables for sflux
        "uwind": ("UGRD", "10 m above ground"),
        "vwind": ("VGRD", "10 m above ground"),
        "prmsl": ("PRMSL", "mean sea level"),
        "stmp": ("TMP", "2 m above ground"),
        "spfh": ("SPFH", "2 m above ground"),
        "dlwrf": ("DLWRF", "surface"),
        "dswrf": ("DSWRF", "surface"),
        "prate": ("PRATE", "surface"),
        # Extended variables (for COMF Fortran parity)
        "tdair": ("DPT", "2 m above ground"),
        "rh": ("RH", "2 m above ground"),
        "ulwrf": ("ULWRF", "surface"),
        "uswrf": ("USWRF", "surface"),
        "lhtfl": ("LHTFL", "surface"),
        "shtfl": ("SHTFL", "surface"),
        "tcdc": ("TCDC", "entire atmosphere"),
        "apcp": ("APCP", "surface"),
        "evp": ("EVP", "surface"),
        "wtmp": ("TMP", "surface"),
    }

    # Default: extract only the 8 core sflux variables
    DEFAULT_VARIABLES = ["uwind", "vwind", "prmsl", "stmp", "spfh", "dlwrf", "dswrf", "prate"]

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        variables: Optional[List[str]] = None,
        resolution: str = "0p25",
        extractor: Optional[GRIBExtractor] = None,
        phase: str = "nowcast",
        time_hotstart: Optional[datetime] = None,
    ):
        """
        Args:
            config: ForcingConfig with domain, cycle, and run window
            input_path: Root GFS data directory (COMINgfs)
            output_path: Output directory for sflux/DATM files
            variables: Variables to extract (default: 8 core sflux vars)
            resolution: GFS resolution ("0p25" or "0p50")
            extractor: GRIB2 extractor (auto-detected if None)
            phase: "nowcast" or "forecast" — determines time window
            time_hotstart: Hotstart datetime (nowcast starts from here)
        """
        super().__init__(config, input_path, output_path)
        self.variables = variables or (config.variables if config.variables else self.DEFAULT_VARIABLES)
        self.resolution = resolution
        self._extractor = extractor
        self.phase = phase
        self.time_hotstart = time_hotstart
        # Set resolution-appropriate file size threshold
        self.MIN_FILE_SIZE = self.MIN_FILE_SIZE_BY_RES.get(resolution, 40_000_000)

    @property
    def extractor(self) -> GRIBExtractor:
        if self._extractor is None:
            self._extractor = get_extractor()
        return self._extractor

    def _get_time_window(self) -> Tuple[datetime, datetime]:
        """
        Compute the time window for this phase.

        Nowcast:  time_hotstart (or cycle-nowcast_hours) → cycle + 3h buffer
        Forecast: cycle - 3h buffer → cycle + forecast_hours + 3h buffer

        The 3h buffer ensures overlap between nowcast and forecast,
        matching ORG Fortran behavior.
        """
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + \
                   timedelta(hours=self.config.cyc)

        if self.phase == "nowcast":
            if self.time_hotstart:
                t_start = self.time_hotstart - timedelta(hours=3)  # buffer before hotstart
            else:
                t_start = cycle_dt - timedelta(hours=self.config.nowcast_hours) - timedelta(hours=3)
            t_end = cycle_dt + timedelta(hours=3)  # buffer past nowcast end
        elif self.phase == "forecast":
            t_start = cycle_dt - timedelta(hours=3)  # buffer before forecast start
            t_end = cycle_dt + timedelta(hours=self.config.forecast_hours) + timedelta(hours=3)
        else:
            # Full
            if self.time_hotstart:
                t_start = self.time_hotstart - timedelta(hours=3)
            else:
                t_start = cycle_dt - timedelta(hours=self.config.nowcast_hours) - timedelta(hours=3)
            t_end = cycle_dt + timedelta(hours=self.config.forecast_hours) + timedelta(hours=3)

        return t_start, t_end

    def _filter_to_time_window(self, extracted: dict) -> dict:
        """Filter extracted data to the phase-specific time window."""
        t_start, t_end = self._get_time_window()
        times = extracted["times"]

        # Find indices within window
        keep = [i for i, t in enumerate(times) if t_start <= t <= t_end]

        if len(keep) == len(times):
            log.info(f"Time window [{t_start} to {t_end}]: all {len(times)} steps kept")
            return extracted

        n_dropped = len(times) - len(keep)
        log.info(f"Time window [{t_start} to {t_end}]: kept {len(keep)}/{len(times)} "
                 f"(dropped {n_dropped} outside window)")

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

    def process(self) -> ForcingResult:
        """
        Process GFS forcing data.

        Pipeline: discover files -> extract GRIB2 -> filter to time window -> write sflux or DATM
        """
        log.info(f"GFS processor: pdy={self.config.pdy} cyc={self.config.cyc:02d}z "
                 f"phase={self.phase} domain={self.config.domain} res={self.resolution}")

        self.create_output_dir()

        # Step 1: Find input files
        gfs_files = self.find_input_files()
        if not gfs_files:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No GFS input files found"],
            )
        log.info(f"Found {len(gfs_files)} GFS files")

        # Write met_files_used log for traceability
        self.write_files_used(gfs_files, self.output_path.parent, "GFS", self.phase)

        # Step 2: Extract variables from GRIB2
        extracted = self._extract_all(gfs_files)
        if not extracted["times"]:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["Failed to extract data from GFS files"],
            )

        # Step 3: Filter to phase-specific time window
        extracted = self._filter_to_time_window(extracted)

        # Step 4: Write output
        output_files = []
        warnings = []

        if self.config.nws == 4:
            # DATM output
            writer = DATMWriter()
            datm_path = self.output_path / "datm_forcing.nc"
            writer.write(
                extracted["data"], extracted["times"],
                extracted["lons"], extracted["lats"], datm_path,
            )
            output_files.append(datm_path)
        else:
            # sflux output (default)
            writer = SfluxWriter(self.output_path, source_index=1)
            base_date = self._compute_base_date()
            files = writer.write_all(
                extracted["data"], extracted["times"],
                extracted["lons"], extracted["lats"], base_date,
            )
            output_files.extend(files)

            # Write sflux_inputs.txt
            inputs_file = writer.write_sflux_inputs(met_num=self.config.met_num)
            output_files.append(inputs_file)

        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=output_files, warnings=warnings,
            metadata={
                "num_input_files": len(gfs_files),
                "num_timesteps": len(extracted["times"]),
                "variables": self.variables,
                "resolution": self.resolution,
                "grid_shape": (len(extracted["lats"]), len(extracted["lons"])),
            },
        )

    def find_input_files(self) -> List[Path]:
        """
        Config-driven GFS file discovery with multi-cycle fallback.

        Searches multiple GFS cycles to cover the full nowcast+forecast window.
        If the primary list is incomplete, supplements with backup files.
        """
        primary = self._build_file_list()

        # Target: enough files for nowcast + forecast (hourly)
        n_target = self.config.nowcast_hours + self.config.forecast_hours + 1

        if len(primary) >= n_target:
            return primary

        log.warning(f"Primary GFS list incomplete ({len(primary)}/{n_target}), checking backup")
        backup = self._build_backup_list()

        if backup and len(backup) > len(primary):
            n_supplement = min(n_target - len(primary), len(backup) - len(primary))
            merged = primary + backup[len(primary):len(primary) + n_supplement]
            log.info(f"Merged {n_supplement} backup files (total: {len(merged)})")
            return merged

        return primary

    def _build_file_list(self) -> List[Path]:
        """
        Build primary file list from GFS cycles covering the run window.

        For forecast phase: uses single cycle with extended leads (matching COMF).
        For nowcast phase: uses multiple cycles with short leads.
        """
        cycles = self._compute_search_cycles()
        gfs_files = []

        # Compute max forecast hour needed from a single cycle
        # For forecast: cycle may be 6h before forecast end, so need longer leads
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        cycle_dt = base_date + timedelta(hours=self.config.cyc)
        forecast_end = cycle_dt + timedelta(hours=self.config.forecast_hours)

        for date, cyc in cycles:
            date_str = date.strftime("%Y%m%d")
            cycle_start = date + timedelta(hours=cyc)

            # Max lead = hours from this cycle to the end of the forecast window
            max_fhr = int((forecast_end - cycle_start).total_seconds() / 3600)
            max_fhr = max(max_fhr, self.config.forecast_hours)

            # Try standard path structures
            for path_fmt in [
                self.input_path / f"gfs.{date_str}" / f"{cyc:02d}" / "atmos",
                self.input_path / f"gfs.{date_str}" / f"{cyc:02d}",
                self.input_path,
            ]:
                if not path_fmt.exists():
                    continue

                pattern = f"gfs.t{cyc:02d}z.pgrb2.{self.resolution}.f*"
                found = sorted(path_fmt.glob(pattern))

                for f in found:
                    try:
                        fhr = int(f.name.split(".f")[-1])
                    except (ValueError, IndexError):
                        continue

                    if fhr > max_fhr:
                        continue

                    if self.MIN_FILE_SIZE and not self.validate_file_size(f, self.MIN_FILE_SIZE):
                        log.warning(f"Skipping undersized file: {f.name}")
                        continue

                    gfs_files.append(f)

                if found:
                    break  # Found files at this path level

        # Deduplicate preserving order
        seen = set()
        unique = []
        for f in gfs_files:
            if f not in seen:
                seen.add(f)
                unique.append(f)

        return unique

    def _build_backup_list(self) -> List[Path]:
        """Build backup file list from previous day's t12z cycle."""
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        prev_date = base_date - timedelta(days=1)
        date_str = prev_date.strftime("%Y%m%d")

        for path_fmt in [
            self.input_path / f"gfs.{date_str}" / "12" / "atmos",
            self.input_path / f"gfs.{date_str}" / "12",
        ]:
            if not path_fmt.exists():
                continue

            pattern = f"gfs.t12z.pgrb2.{self.resolution}.f*"
            found = sorted(path_fmt.glob(pattern))
            files = []
            for f in found:
                try:
                    fhr = int(f.name.split(".f")[-1])
                    if fhr <= self.config.forecast_hours:
                        files.append(f)
                except (ValueError, IndexError):
                    continue
            return files

        return []

    def _compute_search_cycles(self) -> List[Tuple[datetime, int]]:
        """
        Determine which GFS cycles to search.

        Production-realistic strategy matching COMF behavior:
        - Nowcast: walk backward from current cycle to cover the nowcast window
          (past cycles are available at runtime)
        - Forecast: use ONLY the latest available cycle before cycle time
          (future cycles don't exist yet at runtime — extend with longer leads)

        Returns cycles in chronological order (oldest first).
        """
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        cycle_dt = base_date + timedelta(hours=self.config.cyc)

        if self.phase == "nowcast":
            # Nowcast: multi-cycle, walk backward to cover nowcast window
            nowcast_start = cycle_dt - timedelta(hours=self.config.nowcast_hours)
            cycles = []
            t = cycle_dt
            while t >= nowcast_start - timedelta(hours=6):
                cyc_hour = t.hour - (t.hour % 6)  # Snap to 0, 6, 12, 18
                cyc_date = t.replace(hour=0, minute=0, second=0, microsecond=0)
                cycles.append((cyc_date, cyc_hour))
                t -= timedelta(hours=6)
            cycles.reverse()
        else:
            # Forecast: single cycle — the latest GFS cycle at or before cycle_dt.
            # In production at t00z, this is typically the previous cycle (t18z)
            # since t00z GFS may not be fully available yet.
            # Use the same cycle as the last nowcast cycle for continuity.
            cyc_hour = cycle_dt.hour - (cycle_dt.hour % 6)
            cyc_date = cycle_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            cycles = [(cyc_date, cyc_hour)]

            # Also include the previous cycle as fallback
            prev = cycle_dt - timedelta(hours=6)
            prev_hour = prev.hour - (prev.hour % 6)
            prev_date = prev.replace(hour=0, minute=0, second=0, microsecond=0)
            # Put previous cycle first so it gets searched first;
            # if current cycle exists, its files will also be found
            cycles.insert(0, (prev_date, prev_hour))

        # Deduplicate
        seen = set()
        unique = []
        for entry in cycles:
            if entry not in seen:
                seen.add(entry)
                unique.append(entry)

        return unique

    def _extract_all(self, gfs_files: List[Path]) -> dict:
        """Extract all variables from GFS GRIB2 files."""
        result = {"times": [], "lons": None, "lats": None, "data": {}}
        for var in self.variables:
            result["data"][var] = []

        domain = self.config.domain

        # Get grid coordinates from first file
        result["lons"], result["lats"] = self.extractor.get_grid(gfs_files[0], domain)

        for gfs_file in gfs_files:
            # Compute valid time from filename
            try:
                fhr = int(gfs_file.name.split(".f")[-1])
                # Determine cycle from filename: gfs.tHHz...
                cyc_str = gfs_file.name.split(".t")[1][:2]
                cyc_hour = int(cyc_str)
                # Determine date from parent path
                for parent in [gfs_file.parent, gfs_file.parent.parent, gfs_file.parent.parent.parent]:
                    if parent.name.startswith("gfs."):
                        date_str = parent.name.split("gfs.")[1][:8]
                        break
                else:
                    date_str = self.config.pdy

                base_time = datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=cyc_hour)
                valid_time = base_time + timedelta(hours=fhr)
            except (ValueError, IndexError):
                log.warning(f"Cannot parse time from {gfs_file.name}, skipping")
                continue

            result["times"].append(valid_time)

            # Extract each variable — append fill array if missing to keep aligned with times
            for var in self.variables:
                if var not in self.GRIB2_VARIABLES:
                    continue
                grib_var, level = self.GRIB2_VARIABLES[var]
                data = self.extractor.extract(gfs_file, grib_var, level, domain)
                if data is not None:
                    result["data"][var].append(data)
                else:
                    # Variable missing (e.g., dlwrf/dswrf absent in f000 analysis)
                    # Append NaN-filled array to keep time alignment
                    if result["lons"] is not None and result["lats"] is not None:
                        ny, nx = len(result["lats"]), len(result["lons"])
                        result["data"][var].append(
                            np.full((ny, nx), np.nan, dtype=np.float32)
                        )
                    log.debug(f"Missing {var} in {gfs_file.name}, filled with NaN")

        # Sort by time and deduplicate (multi-cycle overlap produces duplicate valid times)
        if result["times"]:
            sorted_idx = sorted(range(len(result["times"])), key=lambda i: result["times"][i])
            result["times"] = [result["times"][i] for i in sorted_idx]
            for var in result["data"]:
                if result["data"][var]:
                    # All data arrays must be same length as times (NaN-filled for missing)
                    n_data = len(result["data"][var])
                    n_times = len(sorted_idx)
                    if n_data == n_times:
                        result["data"][var] = [result["data"][var][i] for i in sorted_idx]
                    else:
                        log.warning(f"{var}: data length {n_data} != times {n_times}, skipping sort")

            # Deduplicate: keep first occurrence for each valid time (shortest lead preferred)
            seen_times = {}
            for i, t in enumerate(result["times"]):
                if t not in seen_times:
                    seen_times[t] = i
            unique_idx = sorted(seen_times.values())

            if len(unique_idx) < len(result["times"]):
                n_dups = len(result["times"]) - len(unique_idx)
                log.info(f"Removed {n_dups} duplicate valid times from multi-cycle overlap")
                result["times"] = [result["times"][i] for i in unique_idx]
                for var in result["data"]:
                    if result["data"][var]:
                        n_data = len(result["data"][var])
                        if n_data < max(unique_idx) + 1:
                            log.warning(f"Dedup: {var} has {n_data} entries but "
                                        f"max index is {max(unique_idx)}, truncating")
                        result["data"][var] = [result["data"][var][i]
                                               for i in unique_idx if i < n_data]

        return result

    def _compute_base_date(self) -> datetime:
        """Compute sflux base date (start of model simulation).

        Returns the START OF DAY (00Z) of the hotstart date.
        Fortran convention: base_date is always day-start, and the sflux
        time axis is "days since YYYY-MM-DD 00:00:00". The base_date
        attribute [Y,M,D,0] must match the actual reference used for
        computing time values, otherwise SCHISM reads wrong absolute times.
        """
        if self.time_hotstart:
            return self.time_hotstart.replace(hour=0, minute=0, second=0, microsecond=0)
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + timedelta(hours=self.config.cyc)
        base = cycle_dt - timedelta(hours=self.config.nowcast_hours)
        return base.replace(hour=0, minute=0, second=0, microsecond=0)
