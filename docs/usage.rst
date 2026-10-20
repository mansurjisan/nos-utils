Usage
=====

Quick start: PrepOrchestrator
-----------------------------

The simplest way to run a SCHISM prep cycle is via
:class:`~nos_utils.orchestrator.PrepOrchestrator`. It chains every processor
in the correct order and runs the lightweight ones (GFS, HRRR, NWM, tidal)
concurrently via a ``ThreadPoolExecutor``; the heavy RTOFS and nudging steps
run sequentially in Phase 2 because the netCDF4 C library can stall under
concurrent thread access on large reads.

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
       "comout": "/tmp/secofs_comout",
   }

   orch = PrepOrchestrator(config, paths, run_name="secofs", skip_legacy=False)
   result = orch.run(phase="nowcast")
   print(result.summary())

With the default ``skip_legacy=True`` the RTOFS, NWM and nudging steps are
skipped for SECOFS (they were handled by the legacy shell scripts). Pass
``skip_legacy=False`` to run them in Python; the same
flag is available as :func:`nos_utils.nco_bridge.run_prep`\ ``(skip_legacy=False)``,
which nos-workflow calls. The ``nos-utils prep`` CLI
does not set it, so for SECOFS it skips those steps.

``PrepResult`` exposes ``success`` (all critical steps passed),
``results`` (one :class:`~nos_utils.forcing.base.ForcingResult` per step),
``all_output_files``, ``all_errors``, and ``elapsed_seconds``.

Switching OFS
~~~~~~~~~~~~~

Running STOFS-3D-ATL is a one-line config change; the orchestrator
auto-detects it from ``run_name`` or from the presence of RTOFS ROI indices
in the config:

.. code-block:: python

   config = ForcingConfig.for_stofs_3d_atl(pdy="20260406", cyc=0)
   orch = PrepOrchestrator(config, paths, run_name="stofs_3d_atl")

The STOFS factory pre-configures GFS 0.25 deg resolution, a separate HRRR
domain, RTOFS ROI indices for both OBC and nudging, ADT satellite SSH
blending, dynamic SSH adjustment, St. Lawrence River forcing, interior T/S
nudging, and the NWM ``medium_range_mem1`` product with 121 target files.

Configuration
-------------

:class:`~nos_utils.config.ForcingConfig` is a dataclass that defines the
domain, cycle, time windows, and per-processor options. Factory methods
create preconfigured instances:

.. code-block:: python

   from nos_utils.config import ForcingConfig

   # SE Coastal OFS (6h nowcast / 48h forecast, GFS 0.5 deg, GFS+HRRR blend)
   config = ForcingConfig.for_secofs(pdy="20260406", cyc=0)

   # SECOFS with UFS-Coastal DATM coupling (nws=4)
   config = ForcingConfig.for_secofs_ufs(pdy="20260406", cyc=0)

   # STOFS-3D-ATL (24h nowcast / 96h forecast, ADT blending, nudging)
   config = ForcingConfig.for_stofs_3d_atl(pdy="20260406", cyc=0)

   # STOFS-3D-ATL + UFS-Coastal DATM
   config = ForcingConfig.for_stofs_3d_atl_ufs(pdy="20260406", cyc=0)

   # STOFS-2D-Global (ADCIRC; GFS 0.25 deg, nws=14)
   config = ForcingConfig.for_stofs_2d_glo(pdy="20260406", cyc=0)

   # Ensemble member (GEFS-driven; member 0 = control, met_num=2)
   config = ForcingConfig.for_ensemble(pdy="20260406", cyc=0, member=3)

   # Load from YAML (requires pyyaml)
   config = ForcingConfig.from_yaml("/path/to/secofs.yaml",
                                    pdy="20260406", cyc=0)

Individual processors
---------------------

Each processor follows a consistent pattern:

.. code-block:: python

   from nos_utils.forcing import GFSProcessor

   gfs = GFSProcessor(
       config=config,
       input_path="/lfs/h1/ops/prod/com/gfs/v16.3",
       output_path="/work/secofs/sflux",
       resolution="0p50",        # SECOFS uses 0.50 deg; STOFS uses 0.25
       phase="nowcast",
       time_hotstart=datetime(2026, 4, 5, 18, 0),
   )
   result = gfs.process()

   print(result.success)        # bool
   print(result.output_files)   # list[Path]
   print(result.errors)         # list[str]
   print(result.metadata)       # dict (grid_shape, num_timesteps, ...)

Available processors (imported from :mod:`nos_utils.forcing`):

============================  ==========================================================
Processor                     Responsibility
============================  ==========================================================
``GFSProcessor``              GFS GRIB2 -> ``sflux_air_1/rad_1/prc_1.*.nc``
``HRRRProcessor``             HRRR GRIB2 -> ``sflux_air_2/rad_2/prc_2.*.nc``
``GEFSProcessor``             GEFS ensemble GRIB2 -> sflux per member
``RTOFSProcessor``            RTOFS NetCDF -> ``elev2D``, ``TEM_3D``, ``SAL_3D``, ``uv3D`` + ``obc.tar``
``NWMProcessor``              NWM channel_rt -> ``vsource.th``, ``msource.th``, ``source_sink.in``
``RiverClimProcessor``        USGS climatology fallback for rivers
``StLawrenceProcessor``       St. Lawrence River ``flux.th`` (STOFS-3D-ATL)
``DynamicAdjustProcessor``    SSH boundary correction from tide-gauge observations
``TidalProcessor``            bctides.in generation (calls Fortran ``tide_fac`` exe when available)
``NudgingProcessor``          ``TEM_nu.nc``, ``SAL_nu.nc`` interior nudging
``ParamNmlProcessor``         ``param.nml`` from template with runtime substitution
``HotstartProcessor``         Locate and validate restart file; extract ``time_hotstart``
``PartitionProcessor``        ``partition.prop`` for MPI domain decomposition
``ESMFMeshProcessor``         ESMF mesh for UFS-Coastal DATM coupling
``BlenderProcessor``          HRRR+GFS Delaunay blending -> ``datm_forcing.nc``
``UFSConfigProcessor``        UFS-Coastal ``model_configure``, ``datm_in``, ``ufs.configure``, ...
``SfluxWriter``               SCHISM sflux NetCDF writer (``nws=2``)
``DATMWriter``                UFS-Coastal DATM NetCDF writer (``nws=4``)
============================  ==========================================================

``WaveBoundaryProcessor`` (GFS-Wave spectra to the WW3 ``nest.ww3`` boundary) is
imported from :mod:`nos_utils.forcing.wave_boundary`. The ADCIRC modules
(:mod:`nos_utils.forcing.adcirc_met`, :mod:`nos_utils.forcing.adcirc_tides`)
are function-based and are listed under the API reference.

RTOFS with precomputed weights and ADT blending
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the SSH open boundary, :class:`~nos_utils.forcing.rtofs.RTOFSProcessor`
looks for precomputed INTERP_REMESH weights in ``$FIXofs`` or ``$FIXstofs3d``
(environment variables), then in its input directory, under the exact names
``secofs.obc_ssh_weights.npz`` or ``obc_ssh_weights.npz``. If neither is
found it falls back to on-the-fly Delaunay interpolation. For SECOFS the weights reproduce the
operational Fortran SSH boundary to a 0.5 mm RMSE.

.. code-block:: python

   from nos_utils.forcing import RTOFSProcessor

   rtofs = RTOFSProcessor(
       config=config,
       input_path="/data/rtofs",
       output_path="/work/secofs/obc",
       grid_file="/path/to/fix/secofs/secofs.hgrid.ll",
       obc_ctl_file="/path/to/fix/secofs/secofs.obc.ctl",
       vgrid_file="/path/to/fix/secofs/secofs.vgrid.in",
   )
   result = rtofs.process()
   # ADT blending runs when config.adt_enabled is True and the ADT data
   # and weight file are available.

NWM river forcing
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nos_utils.forcing import NWMProcessor

   nwm = NWMProcessor(
       config=config,
       input_path="/data/nwm",
       output_path="/work/secofs",
       phase="nowcast",
   )
   result = nwm.process()

For the STOFS domain this assembles 121 ``medium_range_mem1`` files into
``vsource.th`` / ``msource.th`` / ``source_sink.in``. SECOFS has no matching
NWM reaches; it runs ``NWMProcessor`` with the ``analysis_assim`` product.
:class:`~nos_utils.forcing.river_clim.RiverClimProcessor` (USGS daily
climatology) is available as a separate processor but the orchestrator does
not call it.

Interior nudging
~~~~~~~~~~~~~~~~

.. code-block:: python

   from nos_utils.forcing import NudgingProcessor

   nudge = NudgingProcessor(
       config=config,
       input_path="/data/rtofs",
       output_path="/work/secofs",
       nudge_weight_file="/path/to/fix/secofs/secofs.nudge.gr3",
   )
   result = nudge.process()

The nudging processor reads the interior-node nudge weights and writes ``TEM_nu.nc`` and
``SAL_nu.nc`` with a configurable relaxation timescale
(``config.nudging_timescale_seconds``, default 86400 s).

NCO / ecFlow integration
------------------------

On WCOSS2, :func:`nos_utils.nco_bridge.run_prep` is the canonical entrypoint
called from a J-job. It reads the standard NCO environment variables and
drives the orchestrator:

.. code-block:: bash

   # In the J-job
   export PYTHONPATH=$HOMEnos/ush/python/nos-utils:$PYTHONPATH
   python3 -c "from nos_utils.nco_bridge import run_prep; \
               import sys; sys.exit(0 if run_prep(phase='nowcast') else 1)"

Environment variables read by the bridge:

=================  ====================================================
Variable           Purpose
=================  ====================================================
``PDY``            Production date (YYYYMMDD) -- required
``cyc``            Cycle hour (0, 6, 12, 18) -- required
``RUN``            OFS name (``secofs``, ``stofs_3d_atl``, ...)
``OFS_CONFIG``     Optional path to a YAML config file
``USE_DATM``       Set to ``true`` for UFS-Coastal mode (``nws=4``)
``COMINgfs``       GFS input root
``COMINhrrr``      HRRR input root
``COMINrtofs``     RTOFS input root
``COMINnwm``       NWM input root
``FIXofs``         FIX directory (grid, templates, ``.npz`` weights)
``DATA``           Working directory
``COMOUT``         Output archive directory
``COMIN``          Previous cycle output (for hotstart search)
``RESTART_DIR``    Explicit hotstart search override
=================  ====================================================

CLI
---

``nos-utils`` is also installed as a console script:

.. code-block:: bash

   # SECOFS prep (nowcast)
   nos-utils prep --ofs secofs --pdy 20260406 --cyc 0 \
       --gfs /data/gfs --hrrr /data/hrrr --rtofs /data/rtofs \
       --nwm /data/nwm --fix /data/fix/secofs --output /work/prep/

   # STOFS-3D-ATL prep (nowcast)
   nos-utils prep --ofs stofs_3d_atl --pdy 20260406 --cyc 0 \
       --gfs /data/gfs --hrrr /data/hrrr --rtofs /data/rtofs \
       --nwm /data/nwm --fix /data/fix/stofs_3d_atl --output /work/prep/

   # UFS-Coastal mode (nws=4)
   nos-utils prep --ofs stofs_3d_atl --ufs --pdy 20260406 --cyc 0 \
       --fix /data/fix/stofs_3d_atl --output /work/prep/

   # Custom YAML config
   nos-utils prep --yaml /path/to/myofs.yaml --pdy 20260406 --cyc 0 \
       --fix /data/fix --output /work/prep/

   # List available OFS factories and processors
   nos-utils list

   # Show help
   nos-utils --help
