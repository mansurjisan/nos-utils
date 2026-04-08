nos-utils
=========

**Python forcing generators for NOAA NOS-OFS ocean forecast systems.**

nos-utils replaces the legacy Fortran preprocessing scripts (COMF/STOFS) with a
modular, testable Python package.  It generates atmospheric, ocean-boundary,
river, and tidal forcing files for `SCHISM <https://schism-dev.github.io/schism/>`_-based
Operational Forecast Systems (OFS) including **SECOFS** (SE Coastal),
**STOFS-3D-ATL** (Atlantic Storm Surge), and their UFS-Coastal counterparts.

Supported OFS
-------------

* **SECOFS** -- SE Coastal Ocean Forecast System (6 h nowcast, 48 h forecast)
* **STOFS-3D-ATL** -- Storm Surge 3-D Atlantic (24 h nowcast, 108 h forecast,
  ADT satellite SSH blending, T/S nudging, NWM medium-range river forcing)
* **UFS-Coastal variants** -- DATM-coupled versions of both (``nws=4``)
* **Ensemble** -- GEFS-driven perturbation members

Key capabilities
----------------

* **Atmospheric forcing** -- GFS (0.25/0.50 deg), HRRR (3 km, CONUS),
  GEFS (ensemble) via GRIB2 extraction and regridding.  STOFS uses separate
  HRRR domain bounds and GFS 0.25 deg resolution.
* **Ocean boundary conditions** -- RTOFS interpolation to SCHISM open-boundary
  nodes (SSH, temperature, salinity, velocity).  STOFS-3D-ATL adds ROI-based
  subsetting and shell-script OBC generation for production parity.
* **ADT satellite SSH blending** -- CMEMS Absolute Dynamic Topography corrects
  RTOFS boundary SSH for STOFS-3D-ATL (``adt_enabled=True``).
* **River forcing** -- NWM streamflow mapped to SCHISM source/sink nodes.
  STOFS uses ``medium_range_mem1`` product with 121-file assembly.
  Climatological fallback via RiverClimProcessor when NWM is unavailable.
* **Interior nudging** -- Temperature and salinity nudging fields from RTOFS
  with ROI subsetting and shell-script generation for STOFS.
* **Tidal constituents** -- Eight major constituents with 18.6-year nodal
  corrections for ``bctides.in``.
* **Model configuration** -- ``param.nml`` generation, hotstart discovery,
  MPI domain partitioning.
* **UFS-Coastal support** -- ESMF mesh generation, HRRR+GFS Delaunay blending,
  and DATM NetCDF output (``nws=4``).

Quick start
-----------

.. code-block:: bash

   pip install -e ".[full]"

.. code-block:: python

   from nos_utils.config import ForcingConfig
   from nos_utils.forcing import GFSProcessor

   # SECOFS
   config = ForcingConfig.for_secofs(pdy="20260324", cyc=12)

   # STOFS-3D-ATL (includes ADT, nudging, medium-range NWM)
   config = ForcingConfig.for_stofs_3d_atl(pdy="20260324", cyc=12)

   gfs = GFSProcessor(config,
                       input_path="/data/gfs",
                       output_path="/data/sflux")
   result = gfs.process()

CLI usage:

.. code-block:: bash

   nos-utils prep --ofs secofs --pdy 20260324 --cyc 12 \
       --gfs /data/gfs --hrrr /data/hrrr --output /work/prep/

   nos-utils prep --ofs stofs_3d_atl --pdy 20260324 --cyc 12 \
       --gfs /data/gfs --hrrr /data/hrrr --rtofs /data/rtofs --output /work/prep/

   nos-utils list   # show available OFS factories and processors

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   notebooks
   api/index
   contributing
