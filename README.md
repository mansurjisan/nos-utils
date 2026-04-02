# nos-utils

Python forcing generation utilities for NOS Operational Forecast Systems.

Standalone package that generates atmospheric, ocean boundary, river, and tidal forcing files for SCHISM-based ocean forecast systems (SECOFS, STOFS-3D-ATL, CREOFS, etc.).

## Processors

| Processor | Source | Resolution | Output |
|-----------|--------|------------|--------|
| **GFSProcessor** | GFS 0.25° | Hourly | sflux or DATM |
| **HRRRProcessor** | HRRR 3km CONUS | Hourly | sflux (secondary) |
| **GEFSProcessor** | GEFS ensemble | 3-hourly | sflux per member |
| **RTOFSProcessor** | RTOFS global ocean | 6-hourly | elev2D/TEM_3D/SAL_3D/uv3D |
| **NWMProcessor** | National Water Model | Hourly | vsource/msource/source_sink |
| **TidalProcessor** | TPXO9 harmonics | Static | bctides.in |

## Install

```bash
pip install -e .            # Core (numpy only)
pip install -e ".[full]"    # With netCDF4, scipy, cfgrib
pip install -e ".[dev]"     # With pytest
```

## Usage

```python
from nos_utils.config import ForcingConfig
from nos_utils.forcing import GFSProcessor, HRRRProcessor

# SECOFS: 6h nowcast + 48h forecast
config = ForcingConfig.for_secofs(pdy="20260324", cyc=12)

# GFS (primary atmospheric)
gfs = GFSProcessor(config, input_path="/data/gfs/v16.3", output_path="/data/sflux")
result = gfs.process()
print(result.output_files)  # [sflux_air_1.1.nc, sflux_rad_1.1.nc, ...]

# HRRR (secondary, optional)
hrrr = HRRRProcessor(config, input_path="/data/hrrr/v4.1", output_path="/data/sflux")
hrrr.process()  # Non-fatal if unavailable
```

## Test

```bash
pytest -v                    # Unit tests (no data needed)
pytest tests/test_integration_gfs.py  # Integration (requires GFS GRIB2 data + wgrib2)
```

## GRIB2 Backends

- **wgrib2** (production): Fast, handles all projections. Required for HRRR LCC regrid.
- **cfgrib** (development): No external binary needed. `pip install cfgrib xarray`.
- Auto-detected at runtime — prefers wgrib2 if available.
