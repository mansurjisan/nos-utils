# nos-utils

[![Docs](https://readthedocs.org/projects/nos-utils/badge/?version=latest)](https://nos-utils.readthedocs.io/en/latest/)

nos-utils is a Python package that generates the forcing and run-configuration files for NOAA National Ocean Service coastal forecast systems, and some of their post-processing products. It replaces parts of the COMF and STOFS Fortran/shell prep chains with testable Python modules. [nos-workflow](https://github.com/mansurjisan/nos-workflow) uses it as a submodule (`ush/python/nos-utils`) to run the prep and post stages.

## Systems

| System | Model | What nos-utils provides |
|---|---|---|
| SECOFS | SCHISM, DATM+SCHISM, DATM+SCHISM+WW3 (UFS-Coastal) | sflux or DATM atmospheric forcing, RTOFS open boundary, river, tides, nudging, param.nml, UFS-Coastal configuration; for WW3, the GFS-Wave `nest.ww3` boundary and WAV PET bounds / coupling interval |
| STOFS-3D-ATL | SCHISM standalone, DATM+SCHISM; ported to match operational v3.1 | the same set plus ADT and tide-gauge SSH adjustment, NWM and St. Lawrence rivers, and the post products |
| STOFS-2D-Global | ADCIRC | GFS meteorological forcing in ADCIRC OWI NetCDF form (NWS=14) |

## Install, test, docs

```bash
pip install -e ".[full,dev]"    # full: netCDF4, scipy, cfgrib, pyyaml; dev: pytest
pytest                          # unit tests; synthetic fixtures, no data downloads
```

Documentation: [nos-utils.readthedocs.io](https://nos-utils.readthedocs.io/en/latest/). Example:

```python
from nos_utils.config import ForcingConfig
from nos_utils.forcing import GFSProcessor

config = ForcingConfig.for_secofs(pdy="20260324", cyc=12)
result = GFSProcessor(config, input_path="/data/gfs", output_path="/work/sflux").process()
```

A prep cycle runs through `PrepOrchestrator(config, paths, skip_legacy=False)`, which nos-workflow reaches via `nco_bridge.run_prep(skip_legacy=False)`. The `nos-utils prep` CLI (see `nos-utils --help`) keeps the default `skip_legacy=True`, which skips the SECOFS RTOFS and NWM steps.

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

## UFS-Coastal files

For DATM-coupled (`nws=4`) configurations, prep also renders the per-cycle UFS-Coastal inputs from the templates and fix files staged in `$FIXofs`:

| File | Module | What it holds |
|---|---|---|
| `datm_forcing.nc` | `blender` | GFS, or HRRR blended over GFS, on a regular lat/lon grid for CDEPS DATM |
| `datm_esmf_mesh.nc` | `esmf_mesh` | ESMF unstructured mesh of that grid |
| `model_configure`, `datm_in`, `datm.streams` | `ufs_config` | From the `*.template` files: start time (cycle minus nowcast length), length of the phase being prepped (nowcast, forecast, or both), `dt_atmos`, DATM grid size and file names |
| `ufs.configure` | `ufs_config` | From `$FIXofs`, with the PET bounds set from the resource layout (MED, ATM, OCN, and WAV when wave tasks are set) and the `runSeq` coupling interval set to the SCHISM time step or a configured multiple of it |
| `fd_ufs.yaml`, `noahmptable.tbl` | `ufs_config` | Copied from `$FIXofs` |
| `nest.ww3` | `wave_boundary` | WW3 boundary spectra from the GFS-Wave point spectra, through `ww3_bound`, when waves are enabled and `$COMINgfswave` is set |

`model_configure` uses the same start time as `param.nml` and `bctides.in` (cycle minus nowcast length), which keeps the CMEPS clock in step with SCHISM.

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
