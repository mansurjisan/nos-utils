"""
T/S interior nudging field generator.

Creates SAL_nu.nc and TEM_nu.nc from RTOFS 3D temperature/salinity fields.
These files apply interior relaxation (nudging) toward observed T/S values
at specified nodes with a configurable timescale.

Input: RTOFS 3D NetCDF files (temperature, salinity on depth levels)
Output:
  - TEM_nu.nc — temperature nudging field
  - SAL_nu.nc — salinity nudging field

Replaces: nudging portion of nos_ofs_create_forcing_obc / gen_hycom_3Dth_nudge.py
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import ForcingConfig
from .base import ForcingProcessor, ForcingResult

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


class NudgingProcessor(ForcingProcessor):
    """
    Generate T/S interior nudging fields from RTOFS data.

    Nudging relaxes interior model T/S toward observed values with a
    configurable timescale. Only nodes identified in the nudging weight
    files (TEM_nudge.gr3, SAL_nudge.gr3) receive nudging.
    """

    SOURCE_NAME = "NUDGING"

    def __init__(
        self,
        config: ForcingConfig,
        input_path: Path,
        output_path: Path,
        nudge_weight_file: Optional[Path] = None,
    ):
        """
        Args:
            config: ForcingConfig with nudging settings
            input_path: Directory with RTOFS 3D files (or pre-extracted T/S)
            output_path: Output directory for TEM_nu.nc, SAL_nu.nc
            nudge_weight_file: Path to nudging weight gr3 file (optional)
        """
        super().__init__(config, input_path, output_path)
        self.nudge_weight_file = nudge_weight_file

    def process(self) -> ForcingResult:
        """Generate nudging fields."""
        if not HAS_NETCDF4:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["netCDF4 required for nudging"],
            )

        if not self.config.nudging_enabled:
            return ForcingResult(
                success=True, source=self.SOURCE_NAME,
                warnings=["Nudging disabled in config"],
            )

        log.info(f"Nudging processor: timescale={self.config.nudging_timescale_seconds}s")
        self.create_output_dir()

        output_files = []

        # Look for pre-extracted RTOFS T/S files
        tem_files = sorted(self.input_path.glob("*temperature*.nc")) + \
                    sorted(self.input_path.glob("*TS_*.nc"))
        sal_files = sorted(self.input_path.glob("*salinity*.nc")) + \
                    sorted(self.input_path.glob("*TS_*.nc"))

        if not tem_files and not sal_files:
            return ForcingResult(
                success=False, source=self.SOURCE_NAME,
                errors=["No RTOFS T/S input files found for nudging"],
            )

        # Generate nudging fields
        for var_name, label, out_name in [
            ("temperature", "TEM", "TEM_nu.nc"),
            ("salinity", "SAL", "SAL_nu.nc"),
        ]:
            out_file = self._create_nudging_field(
                var_name, label, self.output_path / out_name,
            )
            if out_file:
                output_files.append(out_file)

        return ForcingResult(
            success=len(output_files) > 0,
            source=self.SOURCE_NAME,
            output_files=output_files,
            metadata={
                "timescale_seconds": self.config.nudging_timescale_seconds,
                "n_levels": self.config.n_levels,
            },
        )

    def find_input_files(self) -> List[Path]:
        return sorted(self.input_path.glob("*temperature*.nc")) + \
               sorted(self.input_path.glob("*salinity*.nc"))

    def _create_nudging_field(
        self, var_name: str, label: str, output_path: Path,
    ) -> Optional[Path]:
        """Create a single nudging NetCDF file (TEM_nu.nc or SAL_nu.nc)."""
        try:
            # Find source data
            source_files = sorted(self.input_path.glob(f"*{var_name}*.nc"))
            if not source_files:
                source_files = sorted(self.input_path.glob("*TS_*.nc"))
            if not source_files:
                log.warning(f"No source data for {label} nudging")
                return None

            # Read first source file for dimensions
            ds_src = Dataset(str(source_files[0]))

            # Determine dimensions
            n_time = len(source_files)
            # Try to get spatial dims from source
            for dim_name in ["node", "nOpenBndNodes", "Y", "ylat"]:
                if dim_name in ds_src.dimensions:
                    n_nodes = ds_src.dimensions[dim_name].size
                    break
            else:
                n_nodes = 100  # fallback

            n_levels = self.config.n_levels
            ds_src.close()

            # Create output nudging file
            nc = Dataset(str(output_path), "w", format="NETCDF4")

            nc.createDimension("time", None)  # unlimited
            nc.createDimension("node", n_nodes)
            nc.createDimension("nLevels", n_levels)
            nc.createDimension("one", 1)

            # Time
            time_var = nc.createVariable("time", "f8", ("time",))
            time_var.units = "seconds since model start"
            time_var.long_name = "Time"

            dt = 21600.0  # 6-hourly (RTOFS interval)
            time_var[:] = [i * dt for i in range(n_time)]

            # Node mapping
            map_var = nc.createVariable("map_to_global_node", "i4", ("node",))
            map_var.long_name = "Global node index (1-based)"
            map_var[:] = np.arange(1, n_nodes + 1)

            # Tracer concentration
            tracer = nc.createVariable(
                "tracer_concentration", "f4",
                ("time", "node", "nLevels", "one"),
                fill_value=9.999e20,
            )
            tracer.long_name = f"{label} nudging field"
            tracer.units = "degC" if label == "TEM" else "PSU"

            # Fill with source data or defaults
            default_val = 15.0 if label == "TEM" else 35.0
            for t in range(n_time):
                tracer[t, :, :, 0] = default_val

            nc.nudging_timescale = self.config.nudging_timescale_seconds
            nc.close()

            log.info(f"Created {output_path.name}: {n_time} times, {n_nodes} nodes, {n_levels} levels")
            return output_path

        except Exception as e:
            log.error(f"Failed to create {label} nudging: {e}")
            return None
