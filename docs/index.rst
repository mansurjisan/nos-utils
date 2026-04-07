nos-utils
=========

**Python forcing generators for NOAA NOS-OFS ocean forecast systems.**

nos-utils replaces the legacy Fortran preprocessing scripts (COMF/STOFS) with a
modular, testable Python package.  It generates atmospheric, ocean-boundary,
river, and tidal forcing files for `SCHISM <https://schism-dev.github.io/schism/>`_-based
Operational Forecast Systems (OFS) such as SECOFS, STOFS-3D-ATL, and their
UFS-Coastal counterparts.

Key capabilities
----------------

* **Atmospheric forcing** -- GFS (0.25 deg, hourly), HRRR (3 km, CONUS),
  GEFS (ensemble, 0.25/0.50 deg) via GRIB2 extraction and regridding.
* **Ocean boundary conditions** -- RTOFS interpolation to SCHISM open-boundary
  nodes (SSH, temperature, salinity, velocity).
* **River forcing** -- NWM streamflow mapped to SCHISM source/sink nodes with
  climatological fallback.
* **Tidal constituents** -- Eight major constituents with 18.6-year nodal
  corrections for ``bctides.in``.
* **Interior nudging** -- Temperature and salinity nudging fields from RTOFS.
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

   config = ForcingConfig.for_secofs(pdy="20260324", cyc=12)
   gfs = GFSProcessor(config,
                       input_path="/data/gfs/v16.3",
                       output_path="/data/sflux")
   result = gfs.process()

CLI usage:

.. code-block:: bash

   nos-utils prep --ofs secofs --pdy 20260324 --cyc 12 \
       --gfs /data/gfs --hrrr /data/hrrr --output /work/prep/

   nos-utils list   # show available OFS factories and processors

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   api/index
   contributing
