# nos-utils

[![Docs](https://readthedocs.org/projects/nos-utils/badge/?version=latest)](https://nos-utils.readthedocs.io/en/latest/)

nos-utils is a Python package that generates the forcing and run-configuration files for NOAA National Ocean Service coastal forecast systems, and some of their post-processing products. It replaces parts of the COMF and STOFS Fortran/shell prep chains with testable Python modules. [nos-workflow](https://github.com/mansurjisan/nos-workflow) uses it as a submodule (`ush/python/nos-utils`) to run the prep and post stages.

## Systems

| System | Model | What nos-utils provides |
|---|---|---|
| SECOFS | SCHISM, DATM+SCHISM, DATM+SCHISM+WW3 (UFS-Coastal) | sflux or DATM atmospheric forcing, RTOFS open boundary, river, tides, nudging, param.nml, UFS-Coastal configuration |
| STOFS-3D-ATL | SCHISM standalone, DATM+SCHISM; ported to match operational v3.1 | the same set plus ADT and tide-gauge SSH adjustment, NWM and St. Lawrence rivers, and the post products |
| STOFS-2D-Global | ADCIRC | GFS meteorology in ADCIRC OWI NetCDF form (NWS=14), tidal factors |

## Install, test, docs

```bash
pip install -e ".[full]"    # adds netCDF4, scipy, cfgrib, pyyaml
pytest                      # unit tests; synthetic fixtures, no data downloads
```

Documentation: [nos-utils.readthedocs.io](https://nos-utils.readthedocs.io/en/latest/). Example:

```python
from nos_utils.config import ForcingConfig
from nos_utils.forcing import GFSProcessor

config = ForcingConfig.for_secofs(pdy="20260324", cyc=12)
result = GFSProcessor(config, input_path="/data/gfs", output_path="/work/sflux").process()
```

A full SCHISM prep cycle runs through `PrepOrchestrator` or `nos-utils prep --ofs secofs --pdy 20260324 --cyc 12 --output /work/prep` (see `nos-utils --help` and `nos-utils list`).

## Main modules

| Purpose | Modules in `nos_utils.forcing` |
|---|---|
| Atmospheric | `gfs`, `hrrr`, `gefs`, `blender` (HRRR+GFS), `sflux_writer`, `datm_writer`, `forcing_writer` |
| Ocean boundary | `rtofs`, `adt`, `dynamic_adjust`, `wl_bias`, `obc_ops_interp`, `nudging`, `nudge_ops` |
| River | `nwm`, `river_clim`, `st_lawrence` |
| Tides | `tidal` (bctides.in), `adcirc_tides` (ADCIRC nodal factors) |
| Restart, partition, UFS | `hotstart`, `partition`, `param_nml`, `esmf_mesh`, `ufs_config` |
| Waves | `wave_boundary` (GFS-Wave spectra to WW3 `nest.ww3`) |
| ADCIRC | `adcirc_met` (GFS to OWI NetCDF, fort.22), `adcirc_tides` |
| Post-processing | `nos_utils.post`: `adcirc`, `geopkg`, `maxele`, `profiles`, `slab2d`, `stations` |

Other packages: `nos_utils.interp` (structured and precomputed-weight interpolation), `nos_utils.io` (GRIB2, NetCDF and SCHISM grid readers), `nos_utils.config` (`ForcingConfig`), `nos_utils.orchestrator`, `nos_utils.nco_bridge` (NCO environment variables), `nos_utils.cli`.

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
