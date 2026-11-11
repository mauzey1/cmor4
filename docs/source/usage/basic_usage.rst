Basic Usage
===========

Use :func:`cmor.cmorize` when the complete variable fits in memory. CMOR uses
project tables to validate metadata and coordinates, then writes a NetCDF file
at the project's output path. For variables too large to hold in memory, see
:doc:`incremental_writing`.

Write a variable
----------------

This example uses the CMIP7 tables included as a repository submodule. Run it
from the repository root after initializing that submodule. Replace the sample
values and output directory with your own data and metadata.

.. code-block:: python

   import cmor
   import numpy as np

   # 1. Load project tables that define metadata and coordinate rules.
   project = cmor.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_ocean.json"],
   )

   # 2. Describe the source, experiment, and output location.
   dataset = project.dataset_info({
       "activity_id": "CMIP",
       "experiment_id": "amip",
       "forcing_index": "f1",
       "grid_label": "g101",
       "initialization_index": "i1",
       "institution_id": "CNRM-CERFACS",
       "license_id": "CC-BY-4.0",
       "nominal_resolution": "100 km",
       "physics_index": "p1",
       "realization_index": "r1",
       "region": "glb",
       "source_id": "CNRM-ESM2-1e",
       "outpath": "./output",
   })

   # 3. Select the variable definition from the table.
   variable = project.variable("tos_tavg-u-hxy-sea", table_id="ocean")

   # 4. Define coordinates and provide data in the variable's dimension order.
   axes = [
       project.axis(
           "time", calendar="360_day", frequency="mon",
           values=[15.5, 45.5], bounds=[[0.0, 31.0], [31.0, 60.0]],
           units="days since 1979-01-01",
       ),
       project.axis(
           "latitude", values=[10.0, 20.0, 30.0],
           bounds=[[5.0, 15.0], [15.0, 25.0], [25.0, 35.0]],
       ),
       project.axis(
           "longitude", values=[0.0, 90.0, 180.0, 270.0],
           bounds=[[-45.0, 45.0], [45.0, 135.0],
                   [135.0, 225.0], [225.0, 315.0]],
       ),
   ]

   # Shape: (time, lat, lon).
   data = np.full((2, 3, 4), 20.0, dtype="f4")  # Degrees Celsius

   # 5. Validate metadata and data, then write the NetCDF file.
   path = cmor.cmorize(dataset, variable, axes, data)
   print(path)

The variable and axis tables supply output metadata. CMOR can also add
fixed scalar axes required by the variable table; they need not be supplied
in ``axes``. :doc:`controlled_vocabularies` explains how dataset attributes
are defaulted and validated.

Output options
--------------

Pass ``attrs`` for additional global attributes and ``encoding`` for NetCDF
encoding. The encoding keys are output variable names, which can differ from
the variable's table entry name:

.. code-block:: python

   path = cmor.cmorize(
       dataset, variable, axes, data,
       attrs={"comment": "Example sea surface temperature"},
       encoding={"tos": {"zlib": True, "complevel": 4}},
       path="./output/tos_encoded.nc",
   )

CMIP7 chunk sizes are chosen automatically unless you supply them in
``encoding``. CMOR validates user-supplied CMIP7 chunk sizes.

Missing values
--------------

Set the variable's missing value when the data uses a sentinel. The sentinel
and data type should agree:

.. code-block:: python

   missing = np.float32(1.0e20)
   variable = project.variable(
       "tos_tavg-u-hxy-sea", table_id="ocean", missing_value=missing,
   )
   data[0, 0, 0] = missing
   path = cmor.cmorize(
       dataset, variable, axes, data, path="./output/tos_missing.nc",
   )

For other coordinate layouts, see :doc:`grids`,
:doc:`vertical_coordinates`, and :doc:`time_coordinates`.
