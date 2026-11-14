Basic Usage
===========

CMOR has two ways to write a variable:

* :func:`cmor.cmorize` writes a whole variable in one call. Use it when the
  data fits in memory.
* :class:`cmor.DatasetWriter` writes a variable in pieces. Use it for data
  too large to hold in memory (see :doc:`incremental_writing`).

Both check your metadata and coordinates against the project tables, then
write a NetCDF file under the project's output path.

Write a variable
----------------

This example uses the CMIP7 tables (see :doc:`installation`). Run it from
the repository root.

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

Most of the metadata in the output file comes from the tables, not from
your code:

* The variable and axis tables supply variable and coordinate attributes
  such as units and standard names.
* If the variable table requires a fixed scalar axis (for example, a single
  height level), CMOR adds it for you. You do not need to include it in
  ``axes``.
* The controlled vocabulary fills in and checks the global attributes. See
  :doc:`controlled_vocabularies` for how CMOR applies defaults and
  validates them.

Output options
--------------

Use ``attrs`` to add global attributes and ``encoding`` to set NetCDF
encoding options such as compression. Keys in ``encoding`` are the names of
variables *in the output file*. These can differ from the table entry name:
here the entry ``tos_tavg-u-hxy-sea`` is written as ``tos``.

.. code-block:: python

   path = cmor.cmorize(
       dataset, variable, axes, data,
       attrs={"comment": "Example sea surface temperature"},
       encoding={"tos": {"zlib": True, "complevel": 4}},
       path="./output/tos_encoded.nc",
   )

For CMIP7, CMOR chooses chunk sizes for you. If you set chunk sizes in
``encoding`` instead, CMOR checks that they meet the CMIP7 rules, such as a
minimum data chunk size of 4 MiB.

Missing values
--------------

If your data marks missing points with a special fill value, pass that
value as ``missing_value``. Use the same data type for the value as for the
data (here ``float32``):

.. code-block:: python

   missing = np.float32(1.0e20)
   variable = project.variable(
       "tos_tavg-u-hxy-sea", table_id="ocean", missing_value=missing,
   )
   data[0, 0, 0] = missing
   path = cmor.cmorize(
       dataset, variable, axes, data, path="./output/tos_missing.nc",
   )

Next steps
----------

* :doc:`grids`: curvilinear grids and map projections.
* :doc:`vertical_coordinates`: pressure levels and hybrid coordinates.
* :doc:`time_coordinates`: calendars and forecast times.
* :doc:`incremental_writing`: writing large variables in pieces.
