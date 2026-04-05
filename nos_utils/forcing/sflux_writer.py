"""
SCHISM sflux NetCDF writer.

Creates sflux_air, sflux_rad, sflux_prc files in SCHISM-compatible format.
Shared by GFS, HRRR, and GEFS processors — eliminates duplicate output code.

File naming convention: sflux_{type}_{source_index}.{day_num}.nc
  - type: air, rad, prc
  - source_index: 1 = primary (GFS/GEFS), 2 = secondary (HRRR)
  - day_num: 1, 2, 3, ... (NOT zero-padded to 4 digits)

Dimensions: (ntime, ny_grid, nx_grid)
Time: days since base_date, must be monotonically increasing.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


# Variable definitions: name -> (dtype, units, long_name)
AIR_VARS = {
    "uwind": ("f4", "m/s", "U-wind velocity at 10m"),
    "vwind": ("f4", "m/s", "V-wind velocity at 10m"),
    "prmsl": ("f4", "Pa", "Pressure reduced to mean sea level"),
    "stmp": ("f4", "K", "Surface air temperature at 2m"),
    "spfh": ("f4", "kg/kg", "Specific humidity at 2m"),
}

RAD_VARS = {
    "dlwrf": ("f4", "W/m^2", "Downward longwave radiation flux"),
    "dswrf": ("f4", "W/m^2", "Downward shortwave radiation flux"),
}

PRC_VARS = {
    "prate": ("f4", "kg/m^2/s", "Precipitation rate"),
}


class SfluxWriter:
    """
    Creates SCHISM sflux NetCDF files for any atmospheric source.

    Usage:
        writer = SfluxWriter(output_dir=Path("/data/sflux"), source_index=1)
        files = writer.write_all(data, times, lons, lats, base_date)
        writer.write_sflux_inputs(met_num=2)
    """

    def __init__(self, output_dir: Path, source_index: int = 1):
        """
        Args:
            output_dir: Directory to write sflux files into
            source_index: 1 for primary (GFS/GEFS), 2 for secondary (HRRR)
        """
        if not HAS_NETCDF4:
            raise ImportError("netCDF4 required for sflux output. Install with: pip install netCDF4")

        self.output_dir = Path(output_dir)
        self.source_index = source_index

    def write_all(
        self,
        data: Dict[str, List[np.ndarray]],
        times: List[datetime],
        lons: np.ndarray,
        lats: np.ndarray,
        base_date: datetime,
    ) -> List[Path]:
        """
        Write all sflux files, splitting by day.

        Args:
            data: Dict mapping variable names to lists of 2D arrays (one per time step)
            times: List of datetime objects (one per time step)
            lons: 1D longitude array
            lats: 1D latitude array
            base_date: Reference datetime for time axis

        Returns:
            List of created file paths
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate monotonic time axis
        time_days = [(t - base_date).total_seconds() / 86400.0 for t in times]
        for i in range(1, len(time_days)):
            if time_days[i] <= time_days[i - 1]:
                raise ValueError(
                    f"Non-monotonic time axis at index {i}: "
                    f"{time_days[i]:.6f} <= {time_days[i-1]:.6f} "
                    f"(times: {times[i-1]} -> {times[i]})"
                )

        # Group time steps by day relative to base_date
        day_groups: Dict[int, List[int]] = {}
        for i, t in enumerate(times):
            day_num = max(1, (t - base_date).days + 1)
            day_groups.setdefault(day_num, []).append(i)

        output_files = []
        for day_num, indices in sorted(day_groups.items()):
            day_times = [times[i] for i in indices]
            day_data = {}
            for var_name, var_arrays in data.items():
                if len(var_arrays) < len(times):
                    log.warning(f"write_all: {var_name} has {len(var_arrays)} arrays "
                                f"but {len(times)} time steps")
                day_data[var_name] = [var_arrays[i] for i in indices if i < len(var_arrays)]

            files = self.write_day(day_data, day_times, lons, lats, base_date, day_num)
            output_files.extend(files)

        log.info(f"Wrote {len(output_files)} sflux files for {len(day_groups)} days")
        return output_files

    def write_day(
        self,
        data: Dict[str, List[np.ndarray]],
        times: List[datetime],
        lons: np.ndarray,
        lats: np.ndarray,
        base_date: datetime,
        day_num: int,
    ) -> List[Path]:
        """
        Write sflux_air, sflux_rad, sflux_prc for a single day.

        Returns:
            List of created file paths (up to 3)
        """
        output_files = []

        # sflux_air
        air_file = self._write_file("air", AIR_VARS, data, times, lons, lats, base_date, day_num)
        if air_file:
            output_files.append(air_file)

        # sflux_rad
        rad_file = self._write_file("rad", RAD_VARS, data, times, lons, lats, base_date, day_num)
        if rad_file:
            output_files.append(rad_file)

        # sflux_prc
        prc_file = self._write_file("prc", PRC_VARS, data, times, lons, lats, base_date, day_num)
        if prc_file:
            output_files.append(prc_file)

        return output_files

    def write_sflux_inputs(self, met_num: int = 1) -> Path:
        """
        Write sflux_inputs.txt namelist for SCHISM.

        Args:
            met_num: Number of met sources (1 or 2)

        Returns:
            Path to sflux_inputs.txt
        """
        output_file = self.output_dir / "sflux_inputs.txt"

        with open(output_file, "w") as f:
            f.write("&sflux_inputs\n")
            f.write("air_1_relative_weight=1.0,\n")
            if met_num >= 2:
                f.write("air_2_relative_weight=1.0,\n")
            else:
                f.write("air_2_relative_weight=0.0,\n")
            f.write("air_1_max_window_hours=120.0,\n")
            f.write("air_1_fail_if_missing=.true.,\n")
            f.write("air_2_fail_if_missing=.false.,\n")
            f.write("rad_1_relative_weight=1.0,\n")
            f.write("rad_1_max_window_hours=120.0,\n")
            f.write("prc_1_relative_weight=1.0,\n")
            f.write("prc_1_max_window_hours=120.0,\n")
            f.write("/\n")

        log.info(f"Created {output_file}")
        return output_file

    def _write_file(
        self,
        file_type: str,
        var_specs: Dict[str, Tuple[str, str, str]],
        data: Dict[str, List[np.ndarray]],
        times: List[datetime],
        lons: np.ndarray,
        lats: np.ndarray,
        base_date: datetime,
        day_num: int,
    ) -> Optional[Path]:
        """Write a single sflux NetCDF file."""
        # Check if any data is available for this file type
        available = {v: data[v] for v in var_specs if v in data and data[v]}
        if not available:
            return None

        # File naming: sflux_{type}_{source_index}.{day_num}.nc
        # Using .{N}.nc format (NOT .{NNNN}.nc) per lesson #11
        filename = f"sflux_{file_type}_{self.source_index}.{day_num}.nc"
        output_path = self.output_dir / filename

        try:
            nc = Dataset(str(output_path), "w", format="NETCDF4")

            # Dimensions
            nx = len(lons)
            ny = len(lats)
            nc.createDimension("nx_grid", nx)
            nc.createDimension("ny_grid", ny)
            nc.createDimension("ntime", len(times))

            # Time variable (days since base_date)
            time_var = nc.createVariable("time", "f8", ("ntime",))
            time_var.units = f"days since {base_date.strftime('%Y-%m-%d')} 00:00:00"
            time_var.calendar = "standard"
            time_var.long_name = "Time"
            time_var.base_date = [
                base_date.year, base_date.month, base_date.day, 0
            ]
            time_var[:] = [(t - base_date).total_seconds() / 86400.0 for t in times]

            # 2D coordinate arrays
            lon_var = nc.createVariable("lon", "f4", ("ny_grid", "nx_grid"))
            lon_var.units = "degrees_east"
            lon_var.long_name = "Longitude"

            lat_var = nc.createVariable("lat", "f4", ("ny_grid", "nx_grid"))
            lat_var.units = "degrees_north"
            lat_var.long_name = "Latitude"

            lon_2d, lat_2d = np.meshgrid(lons, lats)
            lon_var[:] = lon_2d
            lat_var[:] = lat_2d

            # Data variables
            for var_name, (dtype, units, long_name) in var_specs.items():
                if var_name not in available:
                    continue

                var = nc.createVariable(
                    var_name, dtype, ("ntime", "ny_grid", "nx_grid"),
                    fill_value=-9999.0,
                )
                var.units = units
                var.long_name = long_name

                try:
                    var[:] = np.stack(available[var_name], axis=0)
                except Exception as e:
                    log.warning(f"Could not write {var_name}: {e}")

            nc.title = f"SCHISM sflux {file_type} forcing"
            nc.conventions = "CF-1.6"
            nc.close()

            log.debug(f"Created {output_path}")
            return output_path

        except Exception as e:
            log.error(f"Failed to create {filename}: {e}")
            return None
