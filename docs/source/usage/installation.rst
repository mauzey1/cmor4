Installation
============

CMOR requires Python 3.11 or newer. Install it from the source repository:

.. code-block:: bash

   git clone https://github.com/mauzey1/cmor4.git
   cd cmor4
   pip install -e .

The usage examples use the CMIP7 project tables included as a Git submodule.
Initialize them from the repository root:

.. code-block:: bash

   git submodule update --init project_tables/cmip7-cmor-tables
