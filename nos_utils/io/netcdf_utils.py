"""
Shared NetCDF utilities for nos-utils.

Common helpers for time axis creation, fill value handling,
variable subsetting, and dimension naming conventions.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

# Standard fill values
FILL_FLOAT32 = -9999.0
FILL_FLOAT64 = 9.999e20


def read_time_axis(filepath: Path) -> Tuple[List[float], str]:
    """
    Read time values and units from a NetCDF file.

    Returns:
        (time_values, units_string)
    """
    ds = Dataset(str(filepath))
    time_var = ds.variables.get("time") or ds.variables.get("MT")
    if time_var is None:
        ds.close()
        return [], ""

    values = time_var[:].tolist()
    units = getattr(time_var, "units", "")
    ds.close()
    return values, units


def validate_monotonic(time_values: List[float], label: str = "time") -> bool:
    """
    Check that a time axis is strictly monotonically increasing.

    Raises ValueError if not monotonic.
    """
    for i in range(1, len(time_values)):
        if time_values[i] <= time_values[i - 1]:
            raise ValueError(
                f"Non-monotonic {label} at index {i}: "
                f"{time_values[i]} <= {time_values[i-1]}"
            )
    return True


def replace_fill_values(
    data: np.ndarray,
    threshold: float = 10000.0,
    fill_value: float = FILL_FLOAT32,
) -> np.ndarray:
    """
    Replace extreme values (abs > threshold) with fill_value.

    RTOFS data sometimes has values > 10000 for missing data.
    """
    result = data.copy()
    mask = np.abs(result) > threshold
    result[mask] = fill_value
    return result


def get_grid_dims(filepath: Path) -> Tuple[int, int]:
    """
    Read (nx, ny) from a NetCDF file's coordinate dimensions.

    Checks common dimension names: longitude/latitude, lon/lat, nx_grid/ny_grid, x/y.
    """
    ds = Dataset(str(filepath))

    nx = ny = 0
    for x_name in ["longitude", "lon", "nx_grid", "x", "nx"]:
        if x_name in ds.dimensions:
            nx = ds.dimensions[x_name].size
            break

    for y_name in ["latitude", "lat", "ny_grid", "y", "ny"]:
        if y_name in ds.dimensions:
            ny = ds.dimensions[y_name].size
            break

    ds.close()
    return nx, ny


def subset_domain(
    data: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    domain: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subset a 2D field to a lon/lat bounding box.

    Args:
        data: 2D array (ny, nx)
        lons: 1D longitude array
        lats: 1D latitude array
        domain: (lon_min, lon_max, lat_min, lat_max)

    Returns:
        (subset_data, subset_lons, subset_lats)
    """
    lon_min, lon_max, lat_min, lat_max = domain

    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)

    subset_lons = lons[lon_mask]
    subset_lats = lats[lat_mask]

    if data.ndim == 2:
        subset_data = data[np.ix_(lat_mask, lon_mask)]
    elif data.ndim == 3:
        subset_data = data[:, np.ix_(lat_mask, lon_mask)[0], np.ix_(lat_mask, lon_mask)[1]]
    else:
        subset_data = data

    return subset_data, subset_lons, subset_lats


def copy_variable(
    src_ds: "Dataset",
    dst_ds: "Dataset",
    var_name: str,
    new_name: Optional[str] = None,
) -> None:
    """Copy a variable from one NetCDF dataset to another."""
    src_var = src_ds.variables[var_name]
    out_name = new_name or var_name

    # Create dimensions if needed
    for dim_name in src_var.dimensions:
        if dim_name not in dst_ds.dimensions:
            dim = src_ds.dimensions[dim_name]
            dst_ds.createDimension(dim_name, None if dim.isunlimited() else dim.size)

    # Create variable
    dst_var = dst_ds.createVariable(
        out_name, src_var.dtype, src_var.dimensions,
        fill_value=getattr(src_var, "_FillValue", None),
    )

    # Copy attributes
    for attr in src_var.ncattrs():
        if attr != "_FillValue":
            setattr(dst_var, attr, getattr(src_var, attr))

    # Copy data
    dst_var[:] = src_var[:]
