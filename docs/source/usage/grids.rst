Grids
=====

How you describe a grid depends on its shape:

* **Rectilinear**: latitude and longitude are each one-dimensional. Use
  ordinary latitude and longitude axes.
* **Curvilinear**: latitude and longitude are two-dimensional arrays over
  ``(y, x)`` index axes. Use :meth:`cmor.ProjectTables.grid`.
* **Projected**: the grid is defined by a map projection. Use
  :meth:`cmor.ProjectTables.grid` with a CF grid mapping.

Rectilinear grids
-----------------

Pass a latitude axis and a longitude axis to :func:`cmor.cmorize`. The
values do not have to be evenly spaced; the values and bounds describe the
grid fully. This code builds unevenly spaced latitude and longitude
axes and lists them after the time axis:

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

For a two-dimensional grid:

1. Create the ``y`` and ``x`` index axes from the project's grid table.
2. Pass them, with the latitude and longitude arrays, to ``grid()``.
3. Pass the grid to ``cmorize()`` with ``grid=``. List only the other axes
   (here, time) in ``axes``; do not list ``y`` and ``x`` again.

Load the grid table along with the variable table. This example writes
latent heat flux on a 3 x 4 curvilinear grid:

.. code-block:: python

   import cmor
   import numpy as np

   grid_project = cmor.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_atmos.json"],
       grid_table="tables/CMIP7_grids.json",
   )
   grid_dataset = grid_project.dataset_info({
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
   print(path)

The latitude and longitude arrays must have the shape of the index axes,
here ``(y, x)``. To also write cell corners, pass ``latitude_vertices`` and
``longitude_vertices`` with shape ``(y, x, n_vertices)``.

CF grid mappings
----------------

If the grid comes from a map projection, give ``grid()`` the CF mapping
name and its parameters. Each parameter value is a ``[value, units]`` pair;
``""`` means no units. CMOR checks the name and parameters against the CF
conventions. It then writes a grid-mapping variable and links the data
variable to it. This code adds a Lambert conformal conic projection to
the curvilinear grid:

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

Data on a list of points or sites, with one spatial index, uses the same
``grid()`` call with one index axis and one-dimensional latitude and
longitude arrays. See :class:`cmor.Grid` for all grid options.
