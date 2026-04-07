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

.. code-block:: bash

   pytest -v              # all tests
   pytest -v -k gfs       # only GFS tests
   pytest --cov=nos_utils # with coverage

The test suite uses no external data -- all inputs are synthetic fixtures defined
in ``tests/conftest.py``.

Code style
----------

* Follow PEP 8.
* Use type annotations for public APIs.
* Add docstrings (Google style) to new classes and public methods.

Adding a new processor
----------------------

1. Create ``nos_utils/forcing/my_processor.py`` subclassing
   :class:`~nos_utils.forcing.base.ForcingProcessor`.
2. Implement ``process()`` returning a
   :class:`~nos_utils.forcing.base.ForcingResult`.
3. Register in ``nos_utils/forcing/__init__.py``.
4. Add tests in ``tests/test_my_processor.py``.
5. Add an RST file under ``docs/api/`` and include it in the toctree.
