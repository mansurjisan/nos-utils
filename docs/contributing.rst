Contributing
============

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/mansurjisan/nos-utils.git
   cd nos-utils
   pip install -e ".[full,dev]"

Running tests
-------------

The test suite uses only synthetic fixtures defined in
``tests/conftest.py`` -- no external data downloads are needed and the
whole suite runs in a few seconds.

.. code-block:: bash

   pytest tests/ -q                    # full suite (206 tests)
   pytest tests/ -q -k gfs             # only GFS tests
   pytest tests/ -q -k stofs           # only STOFS-3D-ATL tests
   pytest --cov=nos_utils --cov-report=html

Code style
----------

* Follow PEP 8.
* Type-annotate public APIs.
* Use Google-style docstrings on new classes and public methods (Napoleon
  is configured in ``docs/conf.py``).
* Run the formatters / linters before opening a PR:

  .. code-block:: bash

     black nos_utils/ tests/
     flake8 nos_utils/ tests/

Adding a new processor
----------------------

1. Create ``nos_utils/forcing/my_processor.py`` subclassing
   :class:`~nos_utils.forcing.ForcingProcessor`.
2. Implement ``process()`` and return a
   :class:`~nos_utils.forcing.ForcingResult` -- populate ``success``,
   ``output_files``, ``metadata``, and ``errors``.
3. Export it from ``nos_utils/forcing/__init__.py`` and add it to
   ``__all__``.
4. Wire it into :class:`~nos_utils.orchestrator.PrepOrchestrator` if it
   should run as part of the standard prep pipeline.
5. Add unit tests in ``tests/test_my_processor.py``. Reuse the synthetic
   GRIB2 / NetCDF fixtures from ``conftest.py`` where possible.
6. Add an RST stub under ``docs/api/`` and include it in the toctree.

PR workflow
-----------

* Open PRs against ``main``.
* Include a short summary of the change, the testing that was done, and
  any validation results (e.g. RMSE vs Fortran reference for numerical
  changes).
* CI runs the full pytest suite on every PR. Please make sure
  ``pytest tests/ -q`` passes locally before requesting review.
* For WCOSS2-only changes (paths, modules, etc.) note it explicitly in the
  PR description so reviewers know the change cannot be validated in CI.
