"""
Forcing processors for NOS-OFS models.

Atmospheric:
    GFSProcessor  - Global Forecast System (0.25°, hourly)
    HRRRProcessor - High-Resolution Rapid Refresh (3km, hourly, CONUS)
    GEFSProcessor - Global Ensemble Forecast System (0.25/0.50°, 3-hourly)

Ocean Boundary:
    RTOFSProcessor - Real-Time Ocean Forecast System (SSH, T/S/UV boundaries)

River:
    NWMProcessor   - National Water Model (streamflow → vsource/msource)

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
from .nwm import NWMProcessor, RiverConfig
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
    # River
    "NWMProcessor",
    "RiverConfig",
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
