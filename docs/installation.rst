Installation
============

Requirements
------------

* Python >= 3.9
* NumPy >= 1.20

Optional dependencies (installed with the ``full`` extra):

* netCDF4 >= 1.5
* SciPy >= 1.7
* cfgrib >= 0.9
* PyYAML >= 5.4

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/mansurjisan/nos-utils.git
   cd nos-utils
   pip install -e ".[full]"

For development:

.. code-block:: bash

   pip install -e ".[full,dev]"

External tools
--------------

Some processors optionally use **wgrib2** for GRIB2 extraction.  If wgrib2 is
not on ``$PATH``, the package falls back to ``cfgrib`` (pure Python).
