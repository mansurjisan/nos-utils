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
    # Writers
    "SfluxWriter",
    "DATMWriter",
]
