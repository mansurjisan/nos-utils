Interpolation
=============

.. automodule:: nos_utils.interp
   :members:
   :undoc-members:

Overview
--------

The ``nos_utils.interp`` subpackage provides grid-to-grid interpolation
utilities used by the forcing processors. Two flavors are available:

* **Structured bilinear** (:mod:`nos_utils.interp.structured_interp`) — on-the-fly
  bilinear weights built from a source lon/lat mesh. Useful for ad-hoc
  interpolation from a regular RTOFS/GFS grid to scattered target points.

* **Precomputed REMESH weights** (:mod:`nos_utils.interp.precomputed_weights`) —
  weight replay from one-time Fortran ``INTERP_REMESH`` exports. This is the
  hot path used every cycle for SSH, 3D T/S, and T/S nudging on SECOFS and
  STOFS-3D-ATL. By reusing the weights that the original Fortran prep
  computed, the Python path is numerically equivalent to the Fortran
  baseline and roughly 20x faster than rebuilding a Delaunay triangulation
  every cycle.

Precomputed weight workflow
---------------------------

There are two steps:

1. **One-time build** (offline, after any grid change): run the Fortran
   prep once with ``NOS_EXPORT_WEIGHTS=YES`` to dump ``*_remesh_export.txt``
   text files, then convert them to NPZ using :func:`build_ssh_npz`,
   :func:`build_nudge_npz`, or :func:`build_3d_npz`. The NPZ is committed
   to the FIX directory and keyed to the RTOFS grid via an MD5 checksum.

2. **Per-cycle apply**: each operational run loads the NPZ with
   :func:`numpy.load`, calls :func:`apply_precomputed_ssh` or
   :func:`apply_precomputed_nudge` on the current RTOFS field, and writes
   the result to ``elev2D.th.nc``, ``TEM_3D.th.nc``, ``SAL_3D.th.nc``,
   ``TEM_nu.nc``, or ``SAL_nu.nc``.

Example
~~~~~~~

.. code-block:: python

   import numpy as np
   from pathlib import Path
   from nos_utils.interp.precomputed_weights import (
       build_ssh_npz,
       apply_precomputed_ssh,
   )

   # --- One-time build (offline) ---
   build_ssh_npz(
       export_txt=Path("obc_ssh_remesh_export.txt"),
       rtofs_lon_2d=rtofs_lon,  # (ny, nx) from rtofs_glo_2ds
       rtofs_lat_2d=rtofs_lat,
       out_npz=Path("secofs.obc_ssh_weights.npz"),
   )

   # --- Per-cycle apply ---
   npz = dict(np.load("secofs.obc_ssh_weights.npz"))
   ssh_at_obc = apply_precomputed_ssh(npz, ssh_2d=current_rtofs_ssh)
   # ssh_at_obc has shape (n_boundary_nodes,)

Structured bilinear interpolator
--------------------------------

.. automodule:: nos_utils.interp.structured_interp
   :members:
   :undoc-members:
   :show-inheritance:

Precomputed REMESH weights
--------------------------

.. automodule:: nos_utils.interp.precomputed_weights
   :members:
   :undoc-members:
   :show-inheritance:
