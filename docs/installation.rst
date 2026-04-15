Installation
============

Requirements
------------

Core:

* Python >= 3.9
* NumPy >= 1.20

Optional (installed via the ``full`` extra):

* netCDF4 >= 1.5 -- NetCDF I/O for sflux, OBC, nudging, DATM writers
* SciPy >= 1.7 -- KDTree for precomputed-weight replay and Delaunay blending
* cfgrib >= 0.9 -- pure-Python GRIB2 reader (fallback when wgrib2 is missing)
* PyYAML >= 5.4 -- :meth:`ForcingConfig.from_yaml` and the NCO bridge

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/mansurjisan/nos-utils.git
   cd nos-utils

   # Minimal install (numpy only)
   pip install -e .

   # Full runtime (netCDF4, scipy, cfgrib, pyyaml) -- recommended
   pip install -e ".[full]"

   # Development extras (pytest, pytest-cov, pyyaml)
   pip install -e ".[dev]"

   # Documentation build (sphinx, sphinx-rtd-theme)
   pip install -e ".[docs]"

On WCOSS2
---------

On the operational WCOSS2 systems ``nos-utils`` is distributed as a checkout
next to the NOS-OFS workflow and is **imported directly** rather than
pip-installed:

.. code-block:: bash

   cd $HOMEnos/ush/python/nos-utils
   git pull

Add the checkout to ``PYTHONPATH`` in the calling J-job or shell script:

.. code-block:: bash

   export PYTHONPATH=$HOMEnos/ush/python/nos-utils:$PYTHONPATH

Running ``pip install`` on WCOSS2 is not supported: operational jobs must run
from the checked-out tree so that changes land via ``git pull`` alone.

External tools
--------------

* **wgrib2** -- preferred for GRIB2 extraction in the GFS, HRRR, and GEFS
  processors. If wgrib2 is not on ``$PATH``, the processors fall back to
  ``cfgrib``.
* **NCO / CDO** -- used by the legacy STOFS-3D-ATL shell-script OBC/nudging
  wrappers. Available as standard modules on WCOSS2 and Hera.

Precomputed interpolation weights
---------------------------------

Several processors replay Fortran ``INTERP_REMESH`` triangulation via
``.npz`` weight files stored in the FIX directory. These are generated once
and then reused every cycle.

Expected files (SECOFS, in ``$FIXofs``):

* ``secofs.obc_ssh_weights.npz`` -- 1488 boundary nodes, global RTOFS 2D
  (3298x4500). Used by :class:`~nos_utils.forcing.RTOFSProcessor` for SSH.
* ``secofs.obc_3d_weights.npz`` -- 1488 boundary nodes, regional RTOFS 3D
  (e.g. US_east 1710x742). Used for T, S, and UV boundary interpolation.
* ``secofs.obc_nudge_weights.npz`` -- ~32K interior nodes. Used by
  :class:`~nos_utils.forcing.NudgingProcessor` for ``TEM_nu.nc`` and
  ``SAL_nu.nc``.

Generating the weight files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The weights come from a one-time instrumented Fortran run of
``nos_ofs_create_forcing_obc_schism``. The INTERP_REMESH subroutine writes a
text export when ``NOS_EXPORT_WEIGHTS=YES`` is set in the environment. The
export is then converted to ``.npz`` using the helpers in
:mod:`nos_utils.interp.precomputed_weights`:

.. code-block:: python

   from pathlib import Path
   from nos_utils.interp.precomputed_weights import (
       build_npz, build_3d_npz, build_nudge_npz,
   )

   # SSH weights
   build_npz(Path("remesh_export_ssh.txt"), Path("secofs.obc_ssh_weights.npz"))

   # 3D T/S/UV weights
   build_3d_npz(Path("remesh_export_3d.txt"), Path("secofs.obc_3d_weights.npz"))

   # Nudging weights
   build_nudge_npz(Path("remesh_export_nudge.txt"),
                   Path("secofs.obc_nudge_weights.npz"))

Copy the resulting ``.npz`` files to ``$FIXofs`` before running ``nos-utils``
on WCOSS2. If the weights are not found the processors will fall back to
on-the-fly Delaunay triangulation, which is slower and not bit-for-bit
equivalent to the Fortran.
