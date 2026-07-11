"""Post-processing product writers for the unified NOS workflow.

Domain-parameterized ports of the operational STOFS-3D-ATL post
generators (current lineage: ``STOFS-operational-main``; the IT-STOFS
copies are older for adcirc/geopkg/slab2d): every module takes explicit
paths/arrays instead of CWD-relative FIX reads, carries no hardcoded
cycle/mesh dimensions, and is shared by every SCHISM system (secofs,
stofs_3d_atl, ...). Orchestration (product selection, COMOUT naming,
subprocess isolation) lives in nos_workflow; these modules only compute
and write.

Heavy dependencies (netCDF4, scipy, matplotlib, geopandas) are imported
inside the functions that need them, so importing this package needs
only numpy.
"""
from .adcirc import city_mask_from_polygons, write_adcirc
from .geopkg import (
    disturbance_field,
    nowcast_forecast_namer,
    write_disturbance_gpkg,
    write_disturbance_series,
)
from .maxele import write_maxele
from .mesh import split_quads
from .profiles import (
    compute_area_coords,
    read_station_in,
    stack_inputs,
    write_station_profiles,
)
from .slab2d import write_slab2d
from .stations import load_station_csv, write_station_timeseries

__all__ = [
    "city_mask_from_polygons",
    "compute_area_coords",
    "disturbance_field",
    "load_station_csv",
    "nowcast_forecast_namer",
    "read_station_in",
    "split_quads",
    "stack_inputs",
    "write_adcirc",
    "write_disturbance_gpkg",
    "write_disturbance_series",
    "write_maxele",
    "write_slab2d",
    "write_station_profiles",
    "write_station_timeseries",
]
