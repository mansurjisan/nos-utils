UFS-Coastal files
=================

For the DATM-coupled configurations (``nws=4``: SECOFS DATM+SCHISM and
DATM+SCHISM+WW3, STOFS-3D-ATL DATM+SCHISM), prep also writes the inputs that
UFS-Coastal needs, so a coupled run needs no hand-made configuration files.
:class:`~nos_utils.orchestrator.PrepOrchestrator` runs these as its UFS-only
steps: DATM blending and the ESMF mesh, then the configuration files.

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - File
     - Module
     - What it holds
   * - ``datm_forcing.nc``
     - :mod:`~nos_utils.forcing.blender`, :mod:`~nos_utils.forcing.datm_writer`
     - GFS, or HRRR blended over GFS, on a regular lat/lon grid for CDEPS DATM.
       ``nx_global``/``ny_global`` in ``datm_in`` are taken from its dimensions.
   * - ``datm_esmf_mesh.nc``
     - :mod:`~nos_utils.forcing.esmf_mesh`
     - ESMF unstructured mesh of the DATM grid (all elements active).
   * - ``model_configure``, ``datm_in``, ``datm.streams``
     - :mod:`~nos_utils.forcing.ufs_config`
     - Filled from the ``*.template`` files in ``$FIXofs``: run start
       (cycle minus nowcast length), run length (nowcast plus forecast),
       DATM time step, grid size and file names.
   * - ``ufs.configure``
     - :mod:`~nos_utils.forcing.ufs_config`
     - Copied from ``$FIXofs`` with the PET bounds set from the resource
       layout (ATM, OCN, and WAV when wave tasks are configured) and the
       ``runSeq`` coupling interval set to the SCHISM time step (for WW3, a
       configured integer multiple of it). The WW3 component stanza comes
       from the system's own ``ufs.configure`` fix file.
   * - ``fd_ufs.yaml``, ``noahmptable.tbl``
     - :mod:`~nos_utils.forcing.ufs_config`
     - Copied from ``$FIXofs`` as they are.
   * - ``nest.ww3``
     - :mod:`~nos_utils.forcing.wave_boundary`
     - WW3 boundary spectra for DATM+SCHISM+WW3, built from the GFS-Wave
       point spectra with ``ww3_bound`` (needs ``mod_def.ww3``).

All components share one start time, the cycle minus the nowcast length (the
same anchor :mod:`~nos_utils.forcing.param_nml` and :mod:`~nos_utils.forcing.tidal`
use), which keeps the CMEPS clock of DATM, SCHISM and WW3 in step.

WW3 runtime files (``ww3_shel.nml``, ``mod_def.ww3``, ``ww3_grid.nml``) are
static fix files and are not generated here.
