Usage
=====

Configuration
-------------

All processors share a :class:`~nos_utils.config.ForcingConfig` dataclass that
defines the model domain, cycle time, and processing options.

Factory methods create pre-configured instances for supported OFS:

.. code-block:: python

   from nos_utils.config import ForcingConfig

   # SE Coastal OFS (sflux output, 6h nowcast / 48h forecast)
   config = ForcingConfig.for_secofs(pdy="20260324", cyc=12)

   # STOFS-3D-ATL (24h nowcast / 108h forecast, ADT blending, nudging)
   config = ForcingConfig.for_stofs_3d_atl(pdy="20260324", cyc=12)

   # STOFS-3D-ATL with UFS-Coastal DATM output
   config = ForcingConfig.for_stofs_3d_atl_ufs(pdy="20260324", cyc=12)

   # Ensemble member (GEFS-driven)
   config = ForcingConfig.for_ensemble(pdy="20260324", cyc=12, member=3)

   # From YAML
   config = ForcingConfig.from_yaml("/path/to/ofs.yaml")

STOFS-3D-ATL specifics
~~~~~~~~~~~~~~~~~~~~~~

The STOFS factory pre-configures several features not used by SECOFS:

* **GFS 0.25 deg** resolution (SECOFS uses 0.50 deg)
* **Separate HRRR domain** bounds (``hrrr_lon_min``/``hrrr_lat_max``)
* **RTOFS ROI indices** for 2D and 3D subsetting (``obc_roi_2d``, ``obc_roi_3d``)
* **ADT satellite SSH blending** (``adt_enabled=True``)
* **Interior T/S nudging** (``nudging_enabled=True``)
* **NWM medium-range product** (``nwm_product="medium_range_mem1"``, 121 target files)

Running individual processors
-----------------------------

Each processor follows the same pattern:

.. code-block:: python

   from nos_utils.forcing import GFSProcessor

   gfs = GFSProcessor(config, input_path="/data/gfs", output_path="/work/sflux")
   result = gfs.process()

   print(result.success)        # True/False
   print(result.output_files)   # list of Paths created
   print(result.errors)         # any error messages

RTOFS with ADT blending (STOFS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nos_utils.forcing import RTOFSProcessor
   from nos_utils.forcing.adt import ADTBlender

   config = ForcingConfig.for_stofs_3d_atl(pdy="20260324", cyc=12)

   rtofs = RTOFSProcessor(config,
                           input_path="/data/rtofs",
                           output_path="/work/obc",
                           fix_path="/data/fix/stofs_3d_atl")
   result = rtofs.process()

   # ADT blending runs automatically when config.adt_enabled is True
   # and ADT data + weight files are available in fix_path

NWM river forcing (STOFS)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nos_utils.forcing import NWMProcessor

   config = ForcingConfig.for_stofs_3d_atl(pdy="20260324", cyc=12)

   nwm = NWMProcessor(config,
                       input_path="/data/nwm",
                       output_path="/work/river",
                       fix_path="/data/fix/stofs_3d_atl")
   result = nwm.process()

Orchestrator
------------

The :class:`~nos_utils.orchestrator.PrepOrchestrator` chains all processors in
the correct order:

.. code-block:: python

   from nos_utils.orchestrator import PrepOrchestrator

   orch = PrepOrchestrator(config, output_root="/work/prep/",
                           gfs_path="/data/gfs", hrrr_path="/data/hrrr",
                           rtofs_path="/data/rtofs", nwm_path="/data/nwm",
                           fix_path="/data/fix/secofs")
   prep_result = orch.run()
   print(prep_result.summary())

CLI
---

.. code-block:: bash

   # SECOFS prep run
   nos-utils prep --ofs secofs --pdy 20260324 --cyc 12 \
       --gfs /data/gfs --hrrr /data/hrrr --rtofs /data/rtofs \
       --nwm /data/nwm --fix /data/fix/secofs --output /work/prep/

   # STOFS-3D-ATL prep run
   nos-utils prep --ofs stofs_3d_atl --pdy 20260324 --cyc 12 \
       --gfs /data/gfs --hrrr /data/hrrr --rtofs /data/rtofs \
       --nwm /data/nwm --fix /data/fix/stofs_3d_atl --output /work/prep/

   # UFS-Coastal mode
   nos-utils prep --ofs stofs_3d_atl --pdy 20260324 --cyc 12 --ufs --output /work/prep/

   # List available OFS and processors
   nos-utils list

NCO environment bridge
----------------------

In WCOSS2/ecFlow workflows, ``nco_bridge`` reads standard environment variables
(``PDY``, ``cyc``, ``COMINgfs``, etc.) and runs the orchestrator:

.. code-block:: python

   from nos_utils.nco_bridge import run_prep

   run_prep(phase="nowcast")
