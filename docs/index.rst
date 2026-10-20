nos-utils
=========

**Python forcing generators for NOAA NOS-OFS ocean forecast systems.**

``nos-utils`` is a standalone Python package that generates forcing and
run-configuration files for NOAA NOS coastal forecast systems, and some of
their post-processing products. It replaces parts of the legacy Fortran and
shell preprocessing toolchain (COMF/STOFS) with modular, testable Python.
`nos-workflow <https://github.com/mansurjisan/nos-workflow>`_ uses it as a
submodule to run the prep and post stages.

Supported systems
-----------------

* **SECOFS** -- SE Coastal Ocean Forecast System. SCHISM (``nws=2`` sflux),
  DATM+SCHISM (``nws=4``) and DATM+SCHISM+WW3 under UFS-Coastal. The SCHISM
  prep was validated against the operational COMF Fortran pipeline (SSH RMSE
  0.5 mm, measured on SECOFS RTOFS open-boundary output).
* **STOFS-3D-ATL** -- Storm Surge 3-D Atlantic. Standalone SCHISM and
  DATM+SCHISM, ported to match operational v3.1: GFS/HRRR atmospheric forcing,
  RTOFS open boundary with ADT and tide-gauge SSH adjustment, NWM and
  St. Lawrence rivers, interior T/S nudging, and the post products.
* **STOFS-2D-Global** -- ADCIRC. GFS meteorology in ADCIRC OWI NetCDF form
  (``NWS=14``) and tidal nodal factors.
* **Ensemble** -- GEFS-driven perturbation members via
  :meth:`ForcingConfig.for_ensemble`.

Key features
------------

* **COMF output structure** -- ``obc.tar``, ``sflux_*.nc``, ``bctides.in``,
  ``param.nml``, ``vsource.th`` and ``source_sink.in`` follow the layout
  expected by the operational SCHISM model.
* **Precomputed INTERP_REMESH weights** -- The Fortran triangulation is
  exported once and replayed every cycle as NumPy gather operations.
* **Parallel Phase 1 prep** -- GFS, HRRR, NWM and tidal processors run
  concurrently via ``ThreadPoolExecutor``; the heavy NetCDF work for RTOFS
  and nudging runs sequentially in Phase 2.
* **Interior T/S nudging** -- ``TEM_nu.nc`` and ``SAL_nu.nc`` from precomputed
  nudge weights.
* **UFS-Coastal support** -- ESMF mesh, HRRR/GFS blender, DATM writer, and
  generation of the UFS configuration files, including the WW3 boundary
  spectra (``nest.ww3``) from GFS-Wave.
* **ADCIRC support** -- GFS to OWI NetCDF meteorology and ADCIRC tidal
  parameters for STOFS-2D-Global.
* **Post-processing** -- the ``nos_utils.post`` subpackage writes the
  ADCIRC-format water-level field, hourly disturbance GeoPackage, maximum
  water level, station time series and profiles, and 2-D slab products.
* **NCO environment bridge** -- ``nco_bridge.run_prep()`` reads standard
  ``PDY``/``cyc``/``COMINgfs``/``FIXofs`` variables and drives the
  orchestrator from inside a J-job.
* **Synthetic test fixtures** -- the unit tests need no data downloads.

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
