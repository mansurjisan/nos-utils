# nos-utils

[![Docs](https://readthedocs.org/projects/nos-utils/badge/?version=latest)](https://nos-utils.readthedocs.io/en/latest/)

Python forcing generators for NOAA NOS-OFS ocean forecast systems.

**Documentation**: [nos-utils.readthedocs.io](https://nos-utils.readthedocs.io/en/latest/)

## Install

```bash
pip install -e ".[full]"    # numpy, netCDF4, scipy
```

## Usage

```python
from nos_utils.config import ForcingConfig
from nos_utils.forcing import GFSProcessor

config = ForcingConfig.for_secofs(pdy="20260324", cyc=12)
gfs = GFSProcessor(config, input_path="/data/gfs/v16.3", output_path="/data/sflux")
result = gfs.process()
```

CLI:
```bash
nos-utils prep --ofs secofs --pdy 20260324 --cyc 12 --gfs /data/gfs --output /work/
```

## Processors

| Processor | Input | Output |
|-----------|-------|--------|
| GFSProcessor | GFS 0.25° GRIB2 | sflux or DATM |
| HRRRProcessor | HRRR 3km GRIB2 | sflux (secondary) |
| GEFSProcessor | GEFS ensemble GRIB2 | sflux per member |
| RTOFSProcessor | RTOFS NetCDF | elev2D, TEM/SAL_3D, uv3D |
| NWMProcessor | NWM channel_rt | vsource.th, msource.th |
| TidalProcessor | TPXO9 template | bctides.in |
| ParamNmlProcessor | param.nml template | param.nml |
| HotstartProcessor | restart archive | hotstart.nc |
| PartitionProcessor | hgrid.gr3 | partition.prop |
| ESMFMeshProcessor | forcing grid | esmf_mesh.nc |
| BlenderProcessor | GFS+HRRR sflux | datm_forcing.nc |

## Test

```bash
pytest -v    # 169 unit tests, no data needed
```
