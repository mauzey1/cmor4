Vertical Coordinates and Formula Terms
======================================

Add a vertical axis to ``axes`` when the variable table declares one. For
dimensionless hybrid coordinates, also supply the formula terms that define
physical height or pressure. The project loads coordinate and formula tables
alongside its variable table. Start with :doc:`basic_usage` for dataset
metadata and ordinary time and horizontal axes.

Pressure levels
---------------

A pressure-level variable takes a pressure axis in the same position as its
table dimensions. Use the axis entry required by that variable:

.. code-block:: python

   pressure = project.axis("plev", values=[92500.0, 50000.0, 10000.0])
   pressure_axes = [axes[0], pressure, axes[1], axes[2]]
   # The data shape follows (time, pressure, latitude, longitude).

The variable table determines whether a fixed scalar level is needed. CMOR
adds table-defined fixed scalar axes automatically when possible.

Hybrid sigma-pressure coordinates
---------------------------------

The following uses the CMIP7 atmosphere table and reuses the dataset metadata
from Basic Usage. The cloud variable has shape ``(time, lev, lat, lon)``.
The ``a`` and ``b`` coefficients and reference pressure are static; surface
pressure varies over time and space.

.. code-block:: python

   import cmor
   import numpy as np

   project = cmor.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_atmos.json"],
       formula_table="tables/CMIP7_formula_terms.json",
   )
   dataset = project.dataset_info(dataset.user_info)
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
   )

For time-varying formula terms written in chunks, leave ``ps`` without
values in the initial definition and pass each surface-pressure chunk to
``DatasetWriter.write(..., zfactors={"ps": ps_chunk})``. See
:doc:`incremental_writing`.
