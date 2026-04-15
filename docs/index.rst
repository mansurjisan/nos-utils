nos-utils
=========

**Python forcing generators for NOAA NOS-OFS ocean forecast systems.**

``nos-utils`` is a standalone Python package that replaces the legacy Fortran
and shell preprocessing toolchain (COMF/STOFS) with a modular, testable
pipeline for generating atmospheric, ocean-boundary, river, and tidal forcing
files for `SCHISM <https://schism-dev.github.io/schism/>`_-based Operational
Forecast Systems (OFS).

Supported systems
-----------------

* **SECOFS** -- SE Coastal Ocean Forecast System. Production-ready on WCOSS2:
  validated against the operational COMF Fortran pipeline
  (SSH RMSE = 0.5 mm, R = 0.999998; prep time ~23 min vs 40 min sequential).
* **STOFS-3D-ATL** -- Storm Surge 3-D Atlantic. Work in progress: atmospheric,
  NWM medium-range river, ADT satellite SSH blending, and interior T/S nudging
  are wired up; operational validation ongoing.
* **UFS-Coastal variants** -- DATM-coupled (``nws=4``) configurations of both
  SECOFS and STOFS-3D-ATL (ESMF mesh + HRRR/GFS Delaunay blender).
* **Ensemble** -- GEFS-driven perturbation members via
  :meth:`ForcingConfig.for_ensemble`.

Key features
------------

* **Byte-identical COMF output structure** -- ``obc.tar`` (6 files),
  ``sflux_*.nc``, ``bctides.in``, ``param.nml``, ``vsource.th``, and
  ``source_sink.in`` match the layout expected by the operational model.
* **Precomputed INTERP_REMESH weights** -- The Fortran triangulation is
  exported once and replayed every cycle as pure NumPy gather operations,
  giving Fortran-equivalent accuracy (SSH RMSE 0.5 mm) at a fraction of the
  runtime.
* **ROI subsetting for RTOFS** -- 3D temperature and salinity reads are
  restricted to the SECOFS subdomain, yielding ~92% I/O reduction.
* **Parallel Phase 1 prep** -- GFS, HRRR, NWM, and tidal processors run
  concurrently via ``ThreadPoolExecutor``; heavy NetCDF+Delaunay work for
  RTOFS and nudging runs sequentially in Phase 2.
* **Full-cycle speedup** -- Total SECOFS prep: ~23 min with nos-utils vs
  ~40 min for the sequential COMF Fortran pipeline (1.7x faster).
* **Interior T/S nudging** -- Python generates ``TEM_nu.nc`` and ``SAL_nu.nc``
  for all 32,613 SECOFS interior nodes using precomputed nudge weights.
* **Unified API across OFS** -- the same ``PrepOrchestrator`` runs SECOFS or
  STOFS just by swapping the :class:`ForcingConfig` factory.
* **NCO environment bridge** -- ``nco_bridge.run_prep()`` reads standard
  ``PDY``/``cyc``/``COMINgfs``/``FIXofs`` variables and drives the
  orchestrator from inside a J-job.
* **206 unit tests passing** -- all inputs are synthetic fixtures; no data
  downloads required.

Quick start
-----------

.. code-block:: bash

   pip install -e ".[full]"

.. code-block:: python

   from nos_utils.config import ForcingConfig
   from nos_utils.orchestrator import PrepOrchestrator

   config = ForcingConfig.for_secofs(pdy="20260406", cyc=0)
   paths = {
       "gfs":    "/lfs/h1/ops/prod/com/gfs/v16.3",
       "hrrr":   "/lfs/h1/ops/prod/com/hrrr/v4.1",
       "rtofs":  "/lfs/h1/ops/prod/com/rtofs/v2.5",
       "nwm":    "/lfs/h1/ops/prod/com/nwm/v3.0",
       "fix":    "/path/to/fix/secofs",
       "output": "/tmp/secofs_prep",
   }
   result = PrepOrchestrator(config, paths).run(phase="nowcast")
   print(result.summary())

CLI:

.. code-block:: bash

   nos-utils prep --ofs secofs --pdy 20260406 --cyc 0 \
       --gfs /data/gfs --hrrr /data/hrrr --rtofs /data/rtofs \
       --nwm /data/nwm --fix /data/fix/secofs --output /work/prep/

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
