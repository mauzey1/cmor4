Installation
============

CMOR requires Python 3.11 or newer. Install it from the source repository:

.. code-block:: bash

   git clone https://github.com/mauzey1/cmor4.git
   cd cmor4
   pip install -e .

The examples in this guide use the CMIP7 project tables. These are a Git
submodule, so they are not downloaded by ``git clone``. To fetch them, run
this from the repository root:

.. code-block:: bash

   git submodule update --init project_tables/cmip7-cmor-tables
