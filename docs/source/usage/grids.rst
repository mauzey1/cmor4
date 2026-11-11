Grids
=====

Use ordinary latitude and longitude axes for a rectilinear grid. Use
:meth:`cmor.ProjectTables.grid` when geographic coordinates are auxiliary
arrays or the variable has a CF grid mapping. The examples build on the
project and dataset metadata in :doc:`basic_usage`.

Rectilinear grids
-----------------

For independent latitude and longitude coordinates, pass both axes to
:func:`cmor.cmorize`. The values need not be evenly spaced; the axes and their
bounds still describe the grid:

.. code-block:: python

   latitude = project.axis(
       "latitude", values=[-60.0, -20.0, 30.0, 70.0],
       bounds=[[-80.0, -40.0], [-40.0, 5.0],
               [5.0, 50.0], [50.0, 90.0]],
   )
   longitude = project.axis(
       "longitude", values=[45.0, 135.0, 225.0, 315.0],
       bounds=[[0.0, 90.0], [90.0, 180.0],
               [180.0, 270.0], [270.0, 360.0]],
   )
   rectilinear_axes = [axes[0], latitude, longitude]

Curvilinear grids
-----------------

For a two-dimensional grid, create its indexing axes with the project's grid
table, then give ``grid()`` the geographic coordinates. The grid owns its
indexing axes; pass only the other axes to ``cmorize()``. This example uses
the ``dataset`` created in Basic Usage to reuse its dataset metadata:

.. code-block:: python

   import cmor
   import numpy as np

   grid_project = cmor.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_atmos.json"],
       grid_table="tables/CMIP7_grids.json",
   )
   grid_dataset = grid_project.dataset_info(dataset.user_info)
   grid_variable = grid_project.variable("hfls_tavg-u-hxy-u", table_id="atmos")

   y = grid_project.axis("y", values=[0.0, 10000.0, 20000.0])
   x = grid_project.axis("x", values=[0.0, 10000.0, 20000.0, 30000.0])
   latitude, longitude = np.meshgrid(
       [10.0, 20.0, 30.0], [280.0, 290.0, 300.0, 310.0],
       indexing="ij",
   )
   grid = grid_project.grid(
       axes=[y, x], latitude=latitude, longitude=longitude,
   )
   time_axis = grid_project.axis(
       "time", calendar="360_day", frequency="mon",
       values=[15.5, 45.5], bounds=[[0.0, 31.0], [31.0, 60.0]],
       units="days since 1979-01-01",
   )
   data = np.full((2, 3, 4), 80.0, dtype="f4")
   path = cmor.cmorize(
       grid_dataset, grid_variable, [time_axis], data, grid=grid,
       path="./output/hfls_curvilinear.nc",
   )

Latitude and longitude arrays must have the shape of the grid axes, here
``(y, x)``. To write cell vertices, supply ``latitude_vertices`` and
``longitude_vertices`` arrays with shape ``(y, x, n_vertices)``.

CF grid mappings
----------------

Add mapping metadata to the grid when the coordinates use a projection. CMOR
validates the mapping name and parameters against CF conventions and writes
the mapping variable and reference from the data variable:

.. code-block:: python

   grid = grid_project.grid(
       axes=[y, x], latitude=latitude, longitude=longitude,
       mapping_name="lambert_conformal_conic",
       params={
           "standard_parallel1": [-20.0, ""],
           "standard_parallel2": [20.0, ""],
           "longitude_of_central_meridian": [175.0, ""],
           "latitude_of_projection_origin": [13.0, ""],
           "false_easting": [8.0, ""],
           "false_northing": [0.0, ""],
       },
   )
   path = cmor.cmorize(
       grid_dataset, grid_variable, [time_axis], data, grid=grid,
       path="./output/hfls_mapped.nc",
   )

For a single spatial index, the same grid API accepts one index axis and
one-dimensional latitude and longitude arrays. See :class:`cmor.Grid` for
the complete grid metadata fields.
