Quick Start
===========

This guide will get you started with CMOR4 in minutes.

Installation
------------

Install CMOR4 from PyPI:

.. code-block:: bash

   pip install cmor4

Or install from source:

.. code-block:: bash

   git clone https://github.com/mauzey1/cmor4.git
   cd cmor4
   pip install -e .

Basic Example
-------------

Here's a minimal example of creating a CMOR-compliant NetCDF file:

.. code-block:: python

   import cmor4
   import numpy as np

   # Load project tables
   project = cmor4.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_ocean.json"],
   )

   # Define dataset metadata
   dataset = project.dataset_info({
       "mip_era": "CMIP7",
       "activity_id": "CMIP",
       "institution_id": "CCCma",
       "source_id": "DUMMY-MODEL",
       "experiment_id": "amip",
       "license_id": "CC-BY-4.0",
       "variant_label": "r1i1p1f1",
       "grid_label": "gn",
       "outpath": "/path/to/output",
   })

   # Get variable definition
   variable = project.variable("tos_tavg-u-hxy-sea", table_id="ocean")

   # Define axes
   time_values = [15.0, 45.0, 74.0]  # Days since base time
   time_bounds = [[0, 30], [30, 60], [60, 90]]

   axes = [
       project.axis("time", values=time_values, bounds=time_bounds,
                    units="days since 1850-01-01"),
       project.axis("latitude", values=[-45.0, 45.0],
                    bounds=[[-90, 0], [0, 90]]),
       project.axis("longitude", values=[90.0, 270.0],
                    bounds=[[0, 180], [180, 360]]),
   ]

   # Create data array (time, lat, lon)
   data = np.random.randn(3, 2, 2) * 5 + 290.0  # Ocean temperature in K

   # Write to NetCDF
   ds, output_path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=axes,
   )

   print(f"Created: {output_path}")
   ds.close()

What Just Happened?
-------------------

1. **Loaded project tables** that define CMIP7 metadata standards
2. **Created dataset metadata** specifying your model, experiment, etc.
3. **Selected a variable** from the project tables (sea surface temperature)
4. **Defined coordinate axes** with values and bounds
5. **Created sample data** matching the axis dimensions
6. **Wrote a CMOR-compliant NetCDF file** with automatic validation

The output file includes:

- ✅ CF-compliant metadata and attributes
- ✅ Validated against CMIP7 tables
- ✅ Standardized filename following DRS conventions
- ✅ Complete coordinate bounds and metadata
- ✅ Standard units and calendar

Next Steps
----------

- See :doc:`basic_usage` for more complete examples
- Learn about :doc:`incremental_writing` for large datasets
- Explore :doc:`advanced_features` for customization options
