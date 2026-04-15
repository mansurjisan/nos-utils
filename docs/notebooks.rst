Notebooks
=========

Jupyter notebooks that walk through a complete nos-utils prep cycle live in
the ``notebooks/`` directory of the repository.

SECOFS prep walkthrough
-----------------------

``notebooks/secofs_prep_walkthrough.ipynb`` runs through every step of a
SECOFS nowcast + forecast cycle, calling each processor in turn and
inspecting the outputs:

1. Configure the run with :meth:`ForcingConfig.for_secofs`
2. GFS atmospheric forcing (nowcast and forecast)
3. HRRR atmospheric forcing (nowcast and forecast)
4. Tidal boundary conditions (``bctides.in``)
5. USGS climatological river forcing (``vsource.th``, ``msource.th``,
   ``source_sink.in``)
6. ``param.nml`` generation from template
7. OBC / nudging notes (Fortran exe today; Python ``RTOFSProcessor`` +
   ``NudgingProcessor`` are shown in the orchestrator usage)
8. A short visualization of the resulting sflux wind field and river
   discharge table

Run it locally:

.. code-block:: bash

   pip install -e ".[full]" jupyter matplotlib
   jupyter notebook notebooks/secofs_prep_walkthrough.ipynb

The notebook expects ``COMINgfs``, ``COMINhrrr``, ``FIXofs``, ``DATA``,
and ``COMOUT`` environment variables to point at real data (e.g. WCOSS2
prod paths or a local mirror). Cells that touch heavy inputs will no-op
cleanly when the data is absent so you can read the notebook end-to-end
without running everything.
