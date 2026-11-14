Vertical Coordinates and Formula Terms
======================================

If the variable table lists a vertical dimension, add a vertical axis to
``axes``.

Some vertical coordinates, such as hybrid sigma-pressure levels, have no
units of their own. To turn them into pressure or height, you also supply
*formula terms*: extra arrays such as ``a``, ``b``, ``p0`` and ``ps``. CMOR
calls these *z-factors*, and their definitions come from the project's
formula table.

Pressure levels
---------------

For a pressure-level variable, create the pressure axis the variable
table asks for (here ``plev``). List the axes in the same order as the
table's dimensions. This code creates a three-level pressure axis and
places it between the time and horizontal axes:

.. code-block:: python

   pressure = project.axis("plev", values=[92500.0, 50000.0, 10000.0])
   pressure_axes = [axes[0], pressure, axes[1], axes[2]]
   # The data shape follows (time, pressure, latitude, longitude).

Some variables are defined at a single fixed level, such as 850 hPa. The
variable table says so, and CMOR adds that scalar axis for you.

Hybrid sigma-pressure coordinates
---------------------------------

This example writes cloud fraction on hybrid sigma-pressure levels. Load
the formula table along with the variable table. The variable has shape
``(time, lev, lat, lon)``.

The formula for this coordinate needs four terms. The ``a`` and ``b``
coefficients and the reference pressure ``p0`` are constant. The surface
pressure ``ps`` changes with time and location.

.. code-block:: python

   import cmor
   import numpy as np

   project = cmor.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_atmos.json"],
       formula_table="tables/CMIP7_formula_terms.json",
   )
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
   cloud_variable = project.variable("cl_tavg-al-hxy-u", table_id="atmos")

   time_axis = project.axis(
       "time", calendar="360_day", frequency="mon",
       values=[15.5, 45.5], bounds=[[0.0, 31.0], [31.0, 60.0]],
       units="days since 1979-01-01",
   )
   lev = project.axis(
       "standard_hybrid_sigma",
       values=[0.92, 0.72, 0.50, 0.30, 0.10],
       bounds=[[1.00, 0.83], [0.83, 0.61], [0.61, 0.40],
               [0.40, 0.20], [0.20, 0.00]],
   )
   latitude = project.axis(
       "latitude", values=[10.0, 20.0, 30.0],
       bounds=[[5.0, 15.0], [15.0, 25.0], [25.0, 35.0]],
   )
   longitude = project.axis(
       "longitude", values=[0.0, 90.0, 180.0, 270.0],
       bounds=[[-45.0, 45.0], [45.0, 135.0],
               [135.0, 225.0], [225.0, 315.0]],
   )
   hybrid_axes = [time_axis, lev, latitude, longitude]

   a = [0.12, 0.22, 0.30, 0.20, 0.10]
   a_bounds = [[0.06, 0.18], [0.18, 0.26], [0.26, 0.25],
               [0.25, 0.15], [0.15, 0.00]]
   b = [0.80, 0.50, 0.20, 0.10, 0.00]
   b_bounds = [[0.94, 0.65], [0.65, 0.35], [0.35, 0.15],
               [0.15, 0.05], [0.05, 0.00]]
   ps = np.full((2, 3, 4), 100000.0, dtype="f4")
   zfactors = [
       project.zfactor("a", values=a, bounds=a_bounds),
       project.zfactor("b", values=b, bounds=b_bounds),
       project.zfactor("p0", values=100000.0),
       project.zfactor("ps", values=ps),
   ]

   data = np.full((2, 5, 3, 4), 50.0, dtype="f4")
   path = cmor.cmorize(
       dataset, cloud_variable, hybrid_axes, data, zfactors=zfactors,
       path="./output/cl_hybrid.nc",
   )
   print(path)

To write this variable in pieces, create ``ps`` without values. Then pass
the surface pressure for each piece with
``DatasetWriter.write(..., zfactors={"ps": ps_chunk})``. See
:doc:`incremental_writing`.
