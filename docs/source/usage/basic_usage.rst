Basic Usage
===========

This guide covers common CMOR4 usage patterns for creating climate model output.

One-Shot Writing with cmorize()
--------------------------------

The :func:`cmor4.cmorize` function is the simplest way to write CMOR-compliant NetCDF files when your complete dataset fits in memory.

Simple 3D Dataset (time, lat, lon)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cmor4
   import numpy as np

   # Load project tables
   project = cmor4.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_atmos.json"],
   )

   # Define dataset metadata
   dataset = project.dataset_info({
       "mip_era": "CMIP7",
       "activity_id": "CMIP",
       "institution_id": "PCMDI",
       "source_id": "PCMDI-MODEL",
       "experiment_id": "historical",
       "license_id": "CC-BY-4.0",
       "variant_label": "r1i1p1f1",
       "grid_label": "gn",
       "outpath": "./output",
   })

   # Select variable
   variable = project.variable("tas", table_id="atmos")

   # Create coordinate axes
   time_values = np.arange(0, 3650, 30.0)  # 10 years, monthly
   time_bounds = np.stack([time_values, time_values + 30.0], axis=1)

   lat_values = np.linspace(-90, 90, 180)
   lat_bounds = np.column_stack([
       lat_values - 0.5,
       lat_values + 0.5,
   ])

   lon_values = np.linspace(0, 360, 360, endpoint=False)
   lon_bounds = np.column_stack([
       lon_values - 0.5,
       lon_values + 0.5,
   ])

   axes = [
       project.axis("time", values=time_values.tolist(),
                    bounds=time_bounds.tolist(),
                    units="days since 1850-01-01"),
       project.axis("latitude", values=lat_values.tolist(),
                    bounds=lat_bounds.tolist()),
       project.axis("longitude", values=lon_values.tolist(),
                    bounds=lon_bounds.tolist()),
   ]

   # Generate sample data (time, lat, lon)
   data = np.random.randn(len(time_values), 180, 360) * 10 + 288.0

   # Write to NetCDF
   ds, output_path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=axes,
   )

   print(f"Wrote {output_path}")
   print(f"File size: {ds.nbytes / 1e6:.1f} MB")
   ds.close()

4D Dataset with Height Levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cmor4
   import numpy as np

   project = cmor4.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_atmos.json"],
   )

   dataset = project.dataset_info({
       "mip_era": "CMIP7",
       "activity_id": "CMIP",
       "institution_id": "PCMDI",
       "source_id": "PCMDI-MODEL",
       "experiment_id": "historical",
       "license_id": "CC-BY-4.0",
       "variant_label": "r1i1p1f1",
       "grid_label": "gn",
       "outpath": "./output",
   })

   variable = project.variable("ta", table_id="atmos")  # Air temperature

   # Define pressure levels
   plev_values = [100000, 92500, 85000, 70000, 60000, 50000,
                  40000, 30000, 25000, 20000, 15000, 10000,
                  7000, 5000, 3000, 2000, 1000]  # Pa

   axes = [
       project.axis("time", values=[15.0], bounds=[[0, 30]],
                    units="days since 2000-01-01"),
       project.axis("plev", values=plev_values),
       project.axis("latitude", values=np.linspace(-90, 90, 90).tolist()),
       project.axis("longitude", values=np.linspace(0, 360, 180, endpoint=False).tolist()),
   ]

   # Create 4D data (time, plev, lat, lon)
   data = np.random.randn(1, len(plev_values), 90, 180) * 15 + 250.0

   ds, output_path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=axes,
   )

   print(f"Wrote {output_path}")
   ds.close()

Custom Compression Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control NetCDF encoding and compression:

.. code-block:: python

   # Define custom encoding
   encoding = {
       "tas": {
           "chunksizes": (1, 90, 180),  # time, lat, lon
           "zlib": True,
           "complevel": 5,  # Compression level (1-9)
           "shuffle": True,
           "_FillValue": 1.0e20,
       }
   }

   ds, output_path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=axes,
       encoding=encoding,
   )

Additional Global Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add custom metadata to your output files:

.. code-block:: python

   attrs = {
       "comment": "Simulation with increased CO2 forcing",
       "contact": "scientist@institution.edu",
       "references": "doi:10.1234/example",
   }

   ds, output_path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=axes,
       attrs=attrs,
   )

Working with Masked Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handle missing values properly:

.. code-block:: python

   import numpy.ma as ma

   # Create masked array with missing values
   data = np.random.randn(12, 180, 360) * 10 + 288.0
   mask = np.random.random((12, 180, 360)) < 0.1  # 10% missing
   masked_data = ma.masked_array(data, mask=mask)

   ds, output_path = cmor4.cmorize(
       data=masked_data,
       dataset=dataset,
       variable=variable,
       axes=axes,
   )

Creating Multiple Files
~~~~~~~~~~~~~~~~~~~~~~~~

Process multiple variables or experiments:

.. code-block:: python

   variables = ["tas", "pr", "hus"]  # Temperature, precipitation, humidity

   for var_name in variables:
       variable = project.variable(var_name, table_id="atmos")

       # Generate appropriate data for this variable
       data = generate_data_for_variable(var_name)

       ds, output_path = cmor4.cmorize(
           data=data,
           dataset=dataset,
           variable=variable,
           axes=axes,
       )

       print(f"Wrote {var_name}: {output_path}")
       ds.close()

Error Handling
--------------

Handle validation errors gracefully:

.. code-block:: python

   try:
       ds, output_path = cmor4.cmorize(
           data=data,
           dataset=dataset,
           variable=variable,
           axes=axes,
       )
   except cmor4.AxisValidationError as e:
       print(f"Axis validation failed: {e}")
   except cmor4.VariableValidationError as e:
       print(f"Variable validation failed: {e}")
   except cmor4.CVValidationError as e:
       print(f"Controlled vocabulary validation failed: {e}")
   except ValueError as e:
       print(f"Data validation failed: {e}")

Common Issues
~~~~~~~~~~~~~

**Shape mismatch:**

.. code-block:: python

   # Wrong: data shape doesn't match axes
   # axes define (time=12, lat=180, lon=360)
   # but data is (12, 90, 180)

   # Fix: ensure data shape matches axes
   assert data.shape == (12, 180, 360)

**Unit conversion:**

.. code-block:: python

   # CMOR expects specific units
   # Check variable.units to see what's required

   print(f"Variable expects units: {variable.units}")

   # Convert if needed
   if variable.units == "K" and your_units == "degC":
       data = data + 273.15

**Time axis issues:**

.. code-block:: python

   # Ensure time bounds are contiguous
   time_values = [15.0, 45.0, 74.0]
   time_bounds = [[0, 30], [30, 60], [60, 90]]  # ✓ Contiguous

   # Not this:
   time_bounds = [[0, 30], [35, 60], [65, 90]]  # ✗ Gaps!

When to Use cmorize()
---------------------

**Best for:**

- Complete datasets that fit in memory
- Simple one-shot writes
- Prototyping and small datasets
- When all data is available at once

**Not ideal for:**

- Datasets larger than RAM
- Streaming/incremental data
- Data generated over time by a running model

For these cases, use :doc:`incremental_writing` instead.

See Also
--------

- :doc:`incremental_writing` - For large datasets
- :doc:`advanced_features` - Custom grids, formula terms, etc.
- :doc:`/api/functions` - Complete function documentation
