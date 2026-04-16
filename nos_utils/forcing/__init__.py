"""
Forcing processors for NOS-OFS models.

Atmospheric:
    GFSProcessor  - Global Forecast System (0.25°, hourly)
    HRRRProcessor - High-Resolution Rapid Refresh (3km, hourly, CONUS)
    GEFSProcessor - Global Ensemble Forecast System (0.25/0.50°, 3-hourly)

Ocean Boundary:
    RTOFSProcessor         - Real-Time Ocean Forecast System (SSH, T/S/UV boundaries)
    DynamicAdjustProcessor - NOAA tide-gauge bias correction for SSH boundary

River:
    NWMProcessor        - National Water Model (streamflow → vsource/msource)
    RiverClimProcessor  - USGS daily climatology (when NWM/BUFR unavailable)
    StLawrenceProcessor - St. Lawrence River (Canadian hydrometric + GFS-rad temp)

Tidal:
    TidalProcessor - Tidal constituents (bctides.in generation)

Nudging:
    NudgingProcessor - T/S interior nudging (TEM_nu.nc, SAL_nu.nc)

Model Config:
    ParamNmlProcessor - Generate/patch SCHISM param.nml
    HotstartProcessor - Find and validate restart files
    PartitionProcessor - Generate partition.prop for MPI decomposition

UFS-Coastal:
    ESMFMeshProcessor - ESMF mesh file for DATM coupling
    BlenderProcessor  - HRRR+GFS Delaunay blending for DATM

Writers:
    SfluxWriter    - SCHISM sflux NetCDF output (nws=2)
    DATMWriter     - UFS-Coastal DATM NetCDF output (nws=4)
"""

from .base import ForcingProcessor, ForcingResult
from .gfs import GFSProcessor
from .hrrr import HRRRProcessor
from .gefs import GEFSProcessor
from .rtofs import RTOFSProcessor
from .dynamic_adjust import DynamicAdjustProcessor
from .nwm import NWMProcessor, RiverConfig
from .river_clim import RiverClimProcessor
from .st_lawrence import StLawrenceProcessor
from .tidal import TidalProcessor, compute_nodal_corrections
from .nudging import NudgingProcessor
from .param_nml import ParamNmlProcessor
from .hotstart import HotstartProcessor, HotstartInfo
from .partition import PartitionProcessor
from .esmf_mesh import ESMFMeshProcessor
from .blender import BlenderProcessor
from .sflux_writer import SfluxWriter
from .datm_writer import DATMWriter

__all__ = [
    "ForcingProcessor",
    "ForcingResult",
    # Atmospheric
    "GFSProcessor",
    "HRRRProcessor",
    "GEFSProcessor",
    # Ocean boundary
    "RTOFSProcessor",
    "DynamicAdjustProcessor",
    # River
    "NWMProcessor",
    "RiverConfig",
    "RiverClimProcessor",
    "StLawrenceProcessor",
    # Tidal
    "TidalProcessor",
    "compute_nodal_corrections",
    # Nudging
    "NudgingProcessor",
    # Model config
    "ParamNmlProcessor",
    "HotstartProcessor",
    "HotstartInfo",
    "PartitionProcessor",
    # UFS-Coastal
    "ESMFMeshProcessor",
    "BlenderProcessor",
    # Writers
    "SfluxWriter",
    "DATMWriter",
]
