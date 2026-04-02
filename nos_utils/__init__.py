"""
nos-utils: NOAA NOS-OFS forcing generation utilities.

Standalone package for generating atmospheric, ocean, river, and tidal
forcing files for SCHISM-based ocean forecast systems.

Usage:
    from nos_utils.config import ForcingConfig
    from nos_utils.forcing import GFSProcessor, HRRRProcessor, GEFSProcessor

    config = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
    gfs = GFSProcessor(config, input_path="/path/to/gfs", output_path="/path/to/sflux")
    result = gfs.process()
"""

__version__ = "0.1.0"
