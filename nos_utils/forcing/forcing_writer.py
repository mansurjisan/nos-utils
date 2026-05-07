"""
Single-file forcing NetCDF writer for UFS-Coastal DATM blending.

Produces ``gfs_forcing.nc`` and ``hrrr_forcing.nc`` in the format the
shell pipeline (``ush/nosofs/nos_ofs_create_datm_forcing.sh``) writes,
so the BlenderProcessor can read them with the same logic as
``ush/python/nos_ofs/datm/blend_hrrr_gfs.py``.

Format:
  - Dimensions: time, latitude, longitude (1D coords for regular lat/lon
    GFS grids) OR time, y, x (2D coords for HRRR Lambert Conformal).
  - Time: float64, "seconds since 1970-01-01 00:00:00", calendar=standard
    — required by CDEPS/ESMF time manager.
  - Coordinates: float32, units degrees_east/degrees_north, axis attrs.
  - Variables: float32, DATM names (UGRD_10maboveground, etc.),
    fill_value=9.999e+20.

This is the single intermediate format consumed by the BlenderProcessor
in nws=4 mode. Distinct from SfluxWriter (3 split files, sflux variable
names) which is used for nws=2 standalone SCHISM.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


# Internal sflux name -> DATM variable name (mirrors blend_hrrr_gfs.py / sflux_to_datm.py)
SFLUX_TO_DATM = {
    "uwind": "UGRD_10maboveground",
    "vwind": "VGRD_10maboveground",
    "stmp":  "TMP_2maboveground",
    "spfh":  "SPFH_2maboveground",
    "prmsl": "MSLMA_meansealevel",
    "prate": "PRATE_surface",
    "dswrf": "DSWRF_surface",
    "dlwrf": "DLWRF_surface",
}

# Long names + units (shell pipeline matches these)
DATM_META = {
    "UGRD_10maboveground": ("U-Component of Wind", "m s-1"),
    "VGRD_10maboveground": ("V-Component of Wind", "m s-1"),
    "TMP_2maboveground":   ("Temperature", "K"),
    "SPFH_2maboveground":  ("Specific Humidity", "kg kg-1"),
    "MSLMA_meansealevel":  ("Pressure Reduced to MSL", "Pa"),
    "PRATE_surface":       ("Precipitation Rate", "kg m-2 s-1"),
    "DSWRF_surface":       ("Downward Short-Wave Radiation Flux", "W m-2"),
    "DLWRF_surface":       ("Downward Long-Wave Radiation Flux", "W m-2"),
}

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
FILL_VALUE = np.float32(9.999e20)


def _to_epoch_seconds(times: List[datetime]) -> np.ndarray:
    """Convert list of datetimes to seconds-since-1970 (CDEPS convention)."""
    out = np.empty(len(times), dtype=np.float64)
    for i, t in enumerate(times):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        out[i] = (t - EPOCH).total_seconds()
    return out


class ForcingNcWriter:
    """Write a single ``<source>_forcing.nc`` file consumable by BlenderProcessor.

    Two coordinate layouts supported:

    - **1D coords** (``write_1d``): regular lat/lon grids like GFS 0.25°.
      lats[ny], lons[nx], data[time, ny, nx].
    - **2D coords** (``write_2d``): native projected grids like HRRR LCC.
      lats[ny, nx], lons[ny, nx], data[time, ny, nx].

    The blender side reads either layout transparently.
    """

    def __init__(self):
        if not HAS_NETCDF4:
            raise ImportError("netCDF4 required for forcing.nc output.")

    def _write_common(self, ncout, times, source_name):
        """Time variable + global attributes shared between 1D/2D writers."""
        ncout.createDimension("time", None)  # unlimited
        time_var = ncout.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01 00:00:00"
        time_var.calendar = "standard"
        time_var.standard_name = "time"
        time_var.axis = "T"
        time_var[:] = _to_epoch_seconds(times)

        ncout.Conventions = "CF-1.6"
        ncout.title = f"{source_name} forcing data for UFS-Coastal DATM"
        ncout.source = f"NCEP {source_name}"
        ncout.institution = "NOAA/NOS/OCS"
        ncout.history = f"Created by nos-utils ForcingNcWriter on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

    def _write_data_vars(self, ncout, data, dim_tuple, source_name):
        """Write each DATM variable with metadata."""
        for sflux_name, datm_name in SFLUX_TO_DATM.items():
            if sflux_name not in data:
                continue
            arr = np.asarray(data[sflux_name], dtype=np.float32)
            long_name, units = DATM_META[datm_name]
            v = ncout.createVariable(
                datm_name, "f4", dim_tuple, fill_value=FILL_VALUE, zlib=False,
            )
            v.long_name = long_name
            v.units = units
            v.standard_name = datm_name
            v[:] = arr

    def write_1d(
        self,
        data: Dict[str, np.ndarray],
        times: List[datetime],
        lons: np.ndarray,
        lats: np.ndarray,
        output_path: Path,
        source_name: str = "GFS",
    ) -> Path:
        """Write forcing file with 1D ``latitude(latitude)``/``longitude(longitude)``.

        Use for regular lat/lon grids (GFS 0.25°/0.5°). Variables are
        ``var(time, latitude, longitude)``.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Dataset(str(output_path), "w", format="NETCDF4_CLASSIC") as ncout:
            ncout.createDimension("latitude", len(lats))
            ncout.createDimension("longitude", len(lons))
            self._write_common(ncout, times, source_name)

            lat_v = ncout.createVariable("latitude", "f4", ("latitude",))
            lat_v.units = "degrees_north"
            lat_v.standard_name = "latitude"
            lat_v.axis = "Y"
            lat_v[:] = np.asarray(lats, dtype=np.float32)

            lon_v = ncout.createVariable("longitude", "f4", ("longitude",))
            lon_v.units = "degrees_east"
            lon_v.standard_name = "longitude"
            lon_v.axis = "X"
            lon_v[:] = np.asarray(lons, dtype=np.float32)

            self._write_data_vars(
                ncout, data, ("time", "latitude", "longitude"), source_name,
            )

        log.info(
            f"Wrote {output_path.name}: {len(times)} times, "
            f"{len(lats)}x{len(lons)} 1D grid, source={source_name}"
        )
        return output_path

    def write_2d(
        self,
        data: Dict[str, np.ndarray],
        times: List[datetime],
        lons2d: np.ndarray,
        lats2d: np.ndarray,
        output_path: Path,
        source_name: str = "HRRR",
    ) -> Path:
        """Write forcing file with 2D ``latitude(y,x)``/``longitude(y,x)``.

        Use for projected grids (HRRR Lambert Conformal). Variables are
        ``var(time, y, x)``.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ny, nx = lats2d.shape
        with Dataset(str(output_path), "w", format="NETCDF4_CLASSIC") as ncout:
            ncout.createDimension("y", ny)
            ncout.createDimension("x", nx)
            self._write_common(ncout, times, source_name)

            lat_v = ncout.createVariable("latitude", "f4", ("y", "x"))
            lat_v.units = "degrees_north"
            lat_v.standard_name = "latitude"
            lat_v[:] = np.asarray(lats2d, dtype=np.float32)

            lon_v = ncout.createVariable("longitude", "f4", ("y", "x"))
            lon_v.units = "degrees_east"
            lon_v.standard_name = "longitude"
            lon_v[:] = np.asarray(lons2d, dtype=np.float32)

            self._write_data_vars(ncout, data, ("time", "y", "x"), source_name)

        log.info(
            f"Wrote {output_path.name}: {len(times)} times, "
            f"{ny}x{nx} 2D grid, source={source_name}"
        )
        return output_path
