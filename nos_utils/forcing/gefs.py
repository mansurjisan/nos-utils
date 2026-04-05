"""
GEFS (Global Ensemble Forecast System) forcing processor.

Processes GEFS GRIB2 data for ensemble atmospheric forcing. Each ensemble
member is processed independently, producing its own set of sflux files.

Input: GEFS GRIB2 files from COMINgefs
  Pattern: gefs.YYYYMMDD/HH/atmos/{product}/{prefix}.tHHz.{file_product}.{res}.fHHH
  prefix: gec00 (control), gep01-gep30 (perturbation)
  Resolution: 0.25° (pgrb2sp25) or 0.50° (pgrb2ap5)
  Temporal: 3-hourly (f003, f006, ..., f132)
  Min file size: 5 MB

Key differences from GFS:
  - 3-hourly (not hourly)
  - Has RH instead of SPFH -> requires RH->SPFH conversion (Tetens formula)
  - Has APCP instead of PRATE -> requires APCP/10800 conversion
  - Needs PRES:surface for RH->SPFH conversion
  - Ensemble: 30 perturbation + 1 control member
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


class GEFSProcessor(ForcingProcessor):
    """
    GEFS ensemble atmospheric forcing processor.

    Processes one ensemble member at a time. For a full ensemble run,
    create multiple GEFSProcessor instances with different member IDs.
    """

    SOURCE_NAME = "GEFS"
    MIN_FILE_SIZE = 5_000_000  # 5 MB (GEFS files are ~15 MB at 0.50°)

    # GEFS raw GRIB2 variables (different from GFS for humidity and precip)
    GRIB2_VARIABLES = {
        "uwind": ("UGRD", "10 m above ground"),
        "vwind": ("VGRD", "10 m above ground"),
        "prmsl": ("PRMSL", "mean sea level"),
        "stmp": ("TMP", "2 m above ground"),
        # GEFS-specific: extract raw RH, APCP, PRES — convert later
        "rh": ("RH", "2 m above ground"),
        "apcp": ("APCP", "surface"),
        "pres": ("PRES", "surface"),
        # Radiation (available in GEFS)
        "dlwrf": ("DLWRF", "surface"),
        "dswrf": ("DSWRF", "surface"),
    }

    # Variables to extract from GRIB2 (includes conversion inputs)
    EXTRACT_VARIABLES = ["uwind", "vwind", "prmsl", "stmp", "rh", "apcp", "pres",
                         "dlwrf", "dswrf"]

    # Final output variables (after conversion)
    OUTPUT_VARIABLES = ["uwind", "vwind", "prmsl", "stmp", "spfh", "prate",
                        "dlwrf", "dswrf"]

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        member: str = "c00",
        product: str = "pgrb2sp25",
        resolution: str = "0p25",
        extractor: Optional[GRIBExtractor] = None,
    ):
        """
        Args:
            config: ForcingConfig with domain, cycle, and run window
            input_path: Root GEFS data directory (COMINgefs)
            output_path: Output directory for sflux files
            member: Ensemble member ID: "c00" for control, "01"-"30" for perturbation
            product: GEFS product directory (pgrb2sp25 or pgrb2ap5)
            resolution: Grid resolution (0p25 or 0p50)
            extractor: GRIB2 extractor (auto-detected if None)
        """
        super().__init__(config, input_path, output_path)
        self.member = member
        self.product = product
        self.resolution = resolution
        self._extractor = extractor

        # Derive file naming components
        self.file_prefix = "gec00" if member == "c00" else f"gep{member}"
        # pgrb2sp25 -> pgrb2s, pgrb2ap5 -> pgrb2a
        self.file_product = product[:6]

    @property
    def extractor(self) -> GRIBExtractor:
        if self._extractor is None:
            self._extractor = get_extractor()
        return self._extractor

    def process(self) -> ForcingResult:
        """
        Process GEFS forcing data for one ensemble member.

        Pipeline: discover -> extract GRIB2 -> convert RH->SPFH, APCP->PRATE -> write sflux
        """
        log.info(f"GEFS processor: member={self.member} ({self.file_prefix}) "
                 f"pdy={self.config.pdy} cyc={self.config.cyc:02d}z")

        self.create_output_dir()

        # Step 1: Find input files
        gefs_files = self.find_input_files()
        if not gefs_files:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[f"No GEFS files found for member {self.member}"],
            )
        log.info(f"Found {len(gefs_files)} GEFS files for member {self.member}")

        # Step 2: Extract raw variables from GRIB2
        extracted = self._extract_all(gefs_files)
        if not extracted["times"]:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=[f"Failed to extract GEFS data for member {self.member}"],
            )

        # Step 3: Convert RH->SPFH, APCP->PRATE
        converted = self._convert_variables(extracted)

        # Step 4: Write sflux
        writer = SfluxWriter(self.output_path, source_index=1)
        base_date = self._compute_base_date()
        files = writer.write_all(
            converted["data"], converted["times"],
            converted["lons"], converted["lats"], base_date,
        )

        return ForcingResult(
            success=True, source=self.SOURCE_NAME,
            output_files=files,
            metadata={
                "member": self.member,
                "file_prefix": self.file_prefix,
                "num_input_files": len(gefs_files),
                "num_timesteps": len(converted["times"]),
                "resolution": self.resolution,
                "product": self.product,
            },
        )

    def find_input_files(self) -> List[Path]:
        """
        3-hourly GEFS file discovery for a specific ensemble member.

        Primary list (covering nowcast + forecast):
          yest t06z f006 + yest t12z f003-f006 + yest t18z f003-f006
          + today t00z f003-f006 + today t06z f003-f006
          + today t12z f003-f{max_fhr}

        Backup: yest t12z f003-f{max_fhr}

        Target: ~53 files for 5.5-day run, minimum ~35.
        """
        primary = self._build_file_list()

        n_target = 35  # Minimum for ~4.4 days coverage at 3-hourly
        if len(primary) >= n_target:
            return primary

        log.warning(f"Primary GEFS list incomplete ({len(primary)}/{n_target}), checking backup")
        backup = self._build_backup_list()

        if backup and len(backup) > len(primary):
            n_supplement = len(backup) - len(primary)
            merged = primary + backup[len(primary):len(primary) + n_supplement]
            log.info(f"Merged {n_supplement} backup files (total: {len(merged)})")
            return merged

        return primary

    def _build_file_list(self) -> List[Path]:
        """Build primary GEFS file list from multiple 3-hourly cycles."""
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        prev_date = base_date - timedelta(days=1)
        max_fhr = min(self.config.nowcast_hours + self.config.forecast_hours, 384)

        files = []

        # Yesterday t06z f006
        f = self._find_gefs_file(prev_date, 6, 6)
        if f:
            files.append(f)

        # Yesterday t12z, t18z: f003-f006 (3-hourly, 2 files each)
        for cyc in [12, 18]:
            for fhr in range(3, 7, 3):
                f = self._find_gefs_file(prev_date, cyc, fhr)
                if f:
                    files.append(f)

        # Today t00z, t06z: f003-f006
        for cyc in [0, 6]:
            for fhr in range(3, 7, 3):
                f = self._find_gefs_file(base_date, cyc, fhr)
                if f:
                    files.append(f)

        # Today t12z: f003 through max_fhr (3-hourly, main forecast period)
        for fhr in range(3, max_fhr + 1, 3):
            f = self._find_gefs_file(base_date, 12, fhr)
            if f:
                files.append(f)

        return files

    def _build_backup_list(self) -> List[Path]:
        """Build backup list from yesterday's t12z extended forecast."""
        base_date = datetime.strptime(self.config.pdy, "%Y%m%d")
        prev_date = base_date - timedelta(days=1)
        max_fhr = min(self.config.nowcast_hours + self.config.forecast_hours, 384)

        files = []

        # Yesterday t06z f006
        f = self._find_gefs_file(prev_date, 6, 6)
        if f:
            files.append(f)

        # Yesterday t12z: f003 through max_fhr
        for fhr in range(3, max_fhr + 1, 3):
            f = self._find_gefs_file(prev_date, 12, fhr)
            if f:
                files.append(f)

        return files

    def _find_gefs_file(self, date: datetime, cyc: int, fhr: int) -> Optional[Path]:
        """Find a single GEFS file by date, cycle, and forecast hour."""
        date_str = date.strftime("%Y%m%d")
        fhr_str = f"{fhr:03d}"

        # Standard path: gefs.YYYYMMDD/HH/atmos/{product}/{prefix}.tHHz.{fp}.{res}.fHHH
        gefs_dir = self.input_path / f"gefs.{date_str}" / f"{cyc:02d}" / "atmos" / self.product
        filename = f"{self.file_prefix}.t{cyc:02d}z.{self.file_product}.{self.resolution}.f{fhr_str}"

        filepath = gefs_dir / filename
        if filepath.exists():
            if self.validate_file_size(filepath, self.MIN_FILE_SIZE):
                return filepath
            else:
                log.warning(f"Undersized GEFS file: {filepath.name}")

        return None

    def _extract_all(self, gefs_files: List[Path]) -> dict:
        """Extract raw variables from GEFS GRIB2 files."""
        result = {"times": [], "lons": None, "lats": None, "data": {}}
        for var in self.EXTRACT_VARIABLES:
            result["data"][var] = []

        domain = self.config.domain
        result["lons"], result["lats"] = self.extractor.get_grid(gefs_files[0], domain)

        for gefs_file in gefs_files:
            valid_time = self._parse_valid_time(gefs_file)
            if valid_time is None:
                continue

            result["times"].append(valid_time)

            for var in self.EXTRACT_VARIABLES:
                if var not in self.GRIB2_VARIABLES:
                    continue
                grib_var, level = self.GRIB2_VARIABLES[var]
                data = self.extractor.extract(gefs_file, grib_var, level, domain)
                if data is not None:
                    result["data"][var].append(data)
                else:
                    log.debug(f"Missing {var} in {gefs_file.name}")

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

    def _convert_variables(self, extracted: dict) -> dict:
        """
        Convert GEFS-specific variables to sflux-compatible format.

        RH + TMP + PRES -> SPFH (Tetens formula)
        APCP -> PRATE (÷ 10800s)
        """
        result = {
            "times": extracted["times"],
            "lons": extracted["lons"],
            "lats": extracted["lats"],
            "data": {},
        }

        raw = extracted["data"]

        # Pass through variables that don't need conversion
        for var in ["uwind", "vwind", "prmsl", "stmp", "dlwrf", "dswrf"]:
            if var in raw and raw[var]:
                result["data"][var] = raw[var]

        # RH -> SPFH conversion
        if "rh" in raw and "stmp" in raw and "pres" in raw:
            n = min(len(raw["rh"]), len(raw["stmp"]), len(raw["pres"]))
            spfh_list = []
            for i in range(n):
                spfh = self.convert_rh_to_spfh(raw["rh"][i], raw["stmp"][i], raw["pres"][i])
                spfh_list.append(spfh)
            result["data"]["spfh"] = spfh_list
            log.info(f"Converted RH->SPFH for {n} timesteps")
        else:
            log.warning("Cannot convert RH->SPFH: missing rh, stmp, or pres data")

        # APCP -> PRATE conversion
        if "apcp" in raw and raw["apcp"]:
            prate_list = [self.convert_apcp_to_prate(a) for a in raw["apcp"]]
            result["data"]["prate"] = prate_list
            log.info(f"Converted APCP->PRATE for {len(prate_list)} timesteps")
        else:
            log.warning("Cannot convert APCP->PRATE: missing apcp data")

        return result

    @staticmethod
    def convert_rh_to_spfh(
        rh: np.ndarray, temp: np.ndarray, pres: np.ndarray
    ) -> np.ndarray:
        """
        Convert relative humidity to specific humidity using Tetens formula.

        es = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65))  [saturation vapor pressure, Pa]
        e  = (RH / 100.0) * es                                 [actual vapor pressure, Pa]
        SPFH = 0.622 * e / (P - 0.378 * e)                    [specific humidity, kg/kg]

        Args:
            rh: Relative humidity (%)
            temp: Temperature (K)
            pres: Surface pressure (Pa)

        Returns:
            Specific humidity (kg/kg)
        """
        rh = np.asarray(rh, dtype=np.float32)
        temp = np.asarray(temp, dtype=np.float32)
        pres = np.asarray(pres, dtype=np.float32)

        es = 611.2 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65))
        e_vap = (rh / 100.0) * es
        spfh = 0.622 * e_vap / (pres - 0.378 * e_vap)

        # Clamp to physical range
        spfh = np.clip(spfh, 0.0, 0.1)

        return spfh

    @staticmethod
    def convert_apcp_to_prate(
        apcp: np.ndarray, dt_seconds: float = 10800.0
    ) -> np.ndarray:
        """
        Convert accumulated precipitation to precipitation rate.

        PRATE = APCP / dt_seconds

        Args:
            apcp: Accumulated precipitation (kg/m²) over dt_seconds
            dt_seconds: Accumulation period (default: 10800s = 3 hours for GEFS)

        Returns:
            Precipitation rate (kg/m²/s)
        """
        apcp = np.asarray(apcp, dtype=np.float32)
        prate = apcp / dt_seconds
        prate = np.clip(prate, 0.0, None)
        return prate

    def _parse_valid_time(self, gefs_file: Path) -> Optional[datetime]:
        """Parse valid time from GEFS filename."""
        try:
            name = gefs_file.name
            # {prefix}.t{HH}z.{product}.{res}.f{HHH}
            parts = name.split(".")
            cyc_str = parts[1].replace("t", "").replace("z", "")
            cyc_hour = int(cyc_str)
            fhr_str = parts[-1].replace("f", "")
            fhr = int(fhr_str)

            # Get date from parent: gefs.YYYYMMDD/HH/atmos/...
            for parent in gefs_file.parents:
                if parent.name.startswith("gefs."):
                    date_str = parent.name.split("gefs.")[1][:8]
                    break
            else:
                date_str = self.config.pdy

            base = datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=cyc_hour)
            return base + timedelta(hours=fhr)

        except (ValueError, IndexError):
            log.warning(f"Cannot parse time from {gefs_file.name}")
            return None

    def _compute_base_date(self) -> datetime:
        """Compute sflux base date (start of nowcast period)."""
        cycle_dt = datetime.strptime(self.config.pdy, "%Y%m%d") + timedelta(hours=self.config.cyc)
        return cycle_dt - timedelta(hours=self.config.nowcast_hours)
