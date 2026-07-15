Incremental Writing with DatasetWriter
=======================================

:class:`cmor4.DatasetWriter` enables memory-bounded incremental writing of CMOR-compliant NetCDF files. Write data one time slice at a time as it becomes available, perfect for streaming model output or datasets larger than RAM.

When to Use DatasetWriter
--------------------------

**✅ Use DatasetWriter when:**

- Writing data incrementally as a climate model runs
- Dataset is larger than available RAM
- Data becomes available over time
- Splitting long time series across multiple files
- Processing streaming data where complete time range isn't known upfront

**❌ Use cmorize() instead when:**

- Complete dataset fits in memory
- Simple one-shot write where all data is available
- Prototyping or small datasets

Memory Characteristics
----------------------

DatasetWriter is designed for bounded memory usage:

- **Per-write memory:** O(chunk size) - only current chunk in memory
- **Total memory:** Independent of total dataset size
- **Example:** Write a 10 GB dataset using only ~50 MB RAM

.. note::
   DatasetWriter uses Zarr for staging and Dask for lazy loading during finalization. This means the complete dataset is never loaded into memory - data is streamed directly from the Zarr store to NetCDF. You don't need to use Dask directly; it's handled automatically.

Basic Usage
-----------

Incremental Time Slices
~~~~~~~~~~~~~~~~~~~~~~~~

Write data one time slice at a time:

.. code-block:: python

   import cmor4
   import numpy as np

   # Load project tables
   project = cmor4.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_ocean.json"],
   )

   # Define metadata
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

   variable = project.variable("tos_tavg-u-hxy-sea", table_id="ocean")

   # Define axes (time axis is empty - values provided per write)
   axes = [
       project.axis("time", units="days since 1850-01-01"),
       project.axis("latitude", values=[-45.0, 45.0],
                    bounds=[[-90, 0], [0, 90]]),
       project.axis("longitude", values=[90.0, 270.0],
                    bounds=[[0, 180], [180, 360]]),
   ]

   # Write incrementally
   with cmor4.DatasetWriter(dataset, variable, axes) as writer:
       for year in range(1850, 2015):
           # Load one year's data (12 months)
           time_values = np.arange(12) * 30.0 + 15.0 + (year - 1850) * 365.0
           time_bounds = np.stack([
               np.arange(12) * 30.0 + (year - 1850) * 365.0,
               np.arange(12) * 30.0 + 30.0 + (year - 1850) * 365.0,
           ], axis=1)

           # Your function that loads one year (returns 12, 2, 2 array)
           data = load_monthly_data(year)

           # Write this year
           writer.write(data, time_values=time_values, time_bounds=time_bounds)
           print(f"Wrote {year}: {len(time_values)} time steps")

   print("Complete dataset written")
   # File automatically finalized by context manager

Pre-Specified Time Values
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you know the complete time range upfront:

.. code-block:: python

   # Define complete time axis
   time_values = np.arange(0, 3650, 30.0)  # 10 years, monthly
   time_bounds = np.stack([time_values, time_values + 30.0], axis=1)

   time_axis = project.axis(
       "time",
       values=time_values.tolist(),
       bounds=time_bounds.tolist(),
       units="days since 1850-01-01",
   )

   axes = [time_axis, lat, lon]

   with cmor4.DatasetWriter(dataset, variable, axes) as writer:
       # Write in chunks - time values taken from axis
       for i in range(0, len(time_values), 12):
           chunk = data[i:i+12]
           writer.write(chunk)  # No time_values argument needed

Explicit Control
~~~~~~~~~~~~~~~~

For more control over finalization:

.. code-block:: python

   writer = cmor4.DatasetWriter(
       dataset,
       variable,
       axes,
       path="./output/custom_filename.nc",
   )

   # Write data
   writer.write(data1, time_values=[15.0], time_bounds=[[0.0, 30.0]])
   writer.write(data2, time_values=[45.0], time_bounds=[[30.0, 60.0]])

   # Finalize and get result
   ds, output_path = writer.close()

   print(f"Wrote {output_path}")
   print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

   # Close the dataset when done
   ds.close()

Advanced Features
-----------------

Segmented Files with Time Gaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create separate file segments with gaps between them:

.. code-block:: python

   from pathlib import Path

   # First segment: 1850-1900
   writer = cmor4.DatasetWriter(
       dataset,
       variable,
       axes,
       path="segment_1850-1900.nc",
   )

   for year in range(1850, 1900):
       data, time_vals, time_bnds = load_year(year)
       writer.write(data, time_values=time_vals, time_bounds=time_bnds)

   # Close with preserve_definition=True to reuse definition
   ds1, path1 = writer.close(preserve_definition=True)
   ds1.close()

   # Second segment: 1950-2000 (gap from 1900-1950)
   writer.path = Path("segment_1950-2000.nc")

   for year in range(1950, 2000):
       data, time_vals, time_bnds = load_year(year)
       writer.write(data, time_values=time_vals, time_bounds=time_bnds)

   ds2, path2 = writer.close()
   ds2.close()

Custom Output Path
~~~~~~~~~~~~~~~~~~

Control where files are written:

.. code-block:: python

   # Explicit path
   writer = cmor4.DatasetWriter(
       dataset,
       variable,
       axes,
       path="/custom/path/output.nc",
   )

   # Auto-generated from metadata (CMIP DRS)
   writer = cmor4.DatasetWriter(dataset, variable, axes)

Incremental Formula Terms (Hybrid Sigma Coordinates)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For variables on hybrid sigma coordinates, time-varying formula terms (e.g., surface pressure) can be written incrementally:

.. code-block:: python

   # Define hybrid vertical axis
   hybrid_axis = project.axis(
       "standard_hybrid_sigma",
       values=[0.9, 0.1],
       bounds=[[1.0, 0.5], [0.5, 0.0]],
   )

   # Define zfactors - omit values for time-varying terms
   zfactors = [
       project.zfactor("a", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
       project.zfactor("b", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
       project.zfactor("p0", values=100000.0),
       project.zfactor("ps"),  # Time-varying - no values here
   ]

   axes = [time_axis, hybrid_axis, lat_axis, lon_axis]

   with cmor4.DatasetWriter(dataset, variable, axes, zfactors=zfactors) as writer:
       for year in range(1850, 2015):
           # Load model data and surface pressure for this year
           data = load_model_data(year)  # Shape: (12, 2, 180, 360)
           ps = load_surface_pressure(year)  # Shape: (12, 180, 360)

           # Write data and surface pressure together
           writer.write(
               data,
               time_values=time_values,
               time_bounds=time_bounds,
               zfactors={"ps": ps}  # Provide ps values per chunk
           )

   # Result: ps variable is accumulated incrementally along with data

**Requirements:**

- Static zfactors (``a``, ``b``, ``p0``) must include complete values in the zfactor definitions
- Time-varying zfactors (``ps``) can omit values initially and provide them via ``write()``
- If you provide a time-varying zfactor in any write, you must provide it in all subsequent writes
- Zfactor chunk shape must match the dimensions for that write chunk

Encoding and Compression
~~~~~~~~~~~~~~~~~~~~~~~~~

Specify NetCDF encoding:

.. code-block:: python

   encoding = {
       "tos": {
           "chunksizes": (1, 2, 2),  # time, lat, lon
           "zlib": True,
           "complevel": 4,
           "shuffle": True,
           "_FillValue": 1.0e20,
       }
   }

   writer = cmor4.DatasetWriter(
       dataset,
       variable,
       axes,
       encoding=encoding,
   )

Custom Staging Directory
~~~~~~~~~~~~~~~~~~~~~~~~~

Use fast local storage for temporary files:

.. code-block:: python

   writer = cmor4.DatasetWriter(
       dataset,
       variable,
       axes,
       staging_dir="/fast/scratch/staging",
   )

Useful when system `/tmp` is small or on a slow filesystem.

Additional Global Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   attrs = {
       "comment": "Experimental run",
       "contact": "user@institution.edu",
   }

   writer = cmor4.DatasetWriter(
       dataset,
       variable,
       axes,
       attrs=attrs,
   )

Handling Existing Files
~~~~~~~~~~~~~~~~~~~~~~~~

Control what happens when the output file already exists:

.. code-block:: python

   # Default: raise error if exists
   writer = cmor4.DatasetWriter(dataset, variable, axes, existing="error")

   # Overwrite existing
   writer = cmor4.DatasetWriter(dataset, variable, axes, existing="replace")

   # Append new time records to existing file
   writer = cmor4.DatasetWriter(dataset, variable, axes, existing="append")

Best Practices
--------------

Use Context Managers
~~~~~~~~~~~~~~~~~~~~

Always use the ``with`` statement:

.. code-block:: python

   # ✓ Good - automatic cleanup
   with cmor4.DatasetWriter(dataset, variable, axes) as writer:
       writer.write(data, time_values=times, time_bounds=bounds)
   # File automatically finalized

   # ✗ Avoid - manual cleanup required
   writer = cmor4.DatasetWriter(dataset, variable, axes)
   writer.write(data, time_values=times, time_bounds=bounds)
   writer.close()  # Must remember to call close()

Validate Early
~~~~~~~~~~~~~~

Test with small data first:

.. code-block:: python

   # Test with one time slice
   test_data = data[:1]
   test_time = time_values[:1]
   test_bounds = time_bounds[:1]

   with cmor4.DatasetWriter(dataset, variable, axes) as writer:
       writer.write(test_data, time_values=test_time, time_bounds=test_bounds)

   print("Validation successful! Proceeding with full write...")

Choose Appropriate Chunk Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Balance memory and I/O efficiency:

.. code-block:: python

   # ✗ Too small - many writes, high overhead
   for i in range(1000):
       writer.write(data[i:i+1], ...)  # 1 time slice per write

   # ✓ Good - monthly or annual chunks
   for year in range(start_year, end_year):
       year_data = load_year(year)  # 12 time slices
       writer.write(year_data, ...)

   # ✗ Too large - defeats memory-bounded design
   writer.write(entire_dataset, ...)  # Just use cmorize() instead

Monitor Progress
~~~~~~~~~~~~~~~~

.. code-block:: python

   with cmor4.DatasetWriter(dataset, variable, axes) as writer:
       for year in range(1850, 2015):
           time_vals, data = load_year(year)
           writer.write(data, time_values=time_vals, time_bounds=bounds)

           # Progress indicator
           total_slices = writer._time_offset
           print(f"Year {year}: {total_slices} time slices written")

Error Handling
--------------

Handle Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       writer = cmor4.DatasetWriter(dataset, variable, axes)
       writer.write(data, time_values=[15.0], time_bounds=[[0.0, 30.0]])
       ds, path = writer.close()
   except cmor4.AxisValidationError as e:
       print(f"Time axis validation failed: {e}")
   except cmor4.VariableValidationError as e:
       print(f"Variable validation failed: {e}")
   except ValueError as e:
       print(f"Data validation failed: {e}")

Inspect Staging on Error
~~~~~~~~~~~~~~~~~~~~~~~~~

The staging directory is preserved for debugging:

.. code-block:: python

   writer = cmor4.DatasetWriter(dataset, variable, axes)
   print(f"Staging directory: {writer.staging_root}")

   try:
       writer.write(bad_data, time_values=[15.0], time_bounds=[[0.0, 30.0]])
       writer.close()
   except Exception as e:
       print(f"Error: {e}")
       print(f"Inspect staging at: {writer.staging_path}")

Common Errors
~~~~~~~~~~~~~

**Shape mismatch:**

.. code-block:: python

   # Error: data shape doesn't match axes
   writer.write(np.ones((2, 3, 3)), time_values=[15.0, 45.0])
   # ValueError: Data chunk shape (2, 3, 3) does not match expected (2, 2, 2)

**Non-monotonic time:**

.. code-block:: python

   writer.write(data1, time_values=[45.0])
   writer.write(data2, time_values=[15.0])  # Goes backward!
   # ValueError: Time values must be strictly monotonic

**Non-contiguous bounds:**

.. code-block:: python

   writer.write(data1, time_values=[15.0], time_bounds=[[0.0, 30.0]])
   writer.write(data2, time_values=[45.0], time_bounds=[[35.0, 60.0]])  # Gap!
   # ValueError: Time bounds must be contiguous

**Inconsistent bounds:**

.. code-block:: python

   writer.write(data1, time_values=[15.0], time_bounds=[[0.0, 30.0]])
   writer.write(data2, time_values=[45.0])  # Forgot bounds
   # ValueError: time_bounds must be supplied for every write or omitted

**Append to non-existent file:**

.. code-block:: python

   writer = cmor4.DatasetWriter(dataset, variable, axes,
                                 path="missing.nc", existing="append")
   writer.write(data, time_values=[15.0])
   # FileNotFoundError: Cannot append because output file 'missing.nc' does not exist

**Incompatible append:**

.. code-block:: python

   # Original file has different grid
   writer = cmor4.DatasetWriter(dataset, variable, axes,
                                 path="output.nc", existing="append")
   writer.write(data, time_values=[75.0])
   # ValueError: dimension 'lat' has size 180 in existing file and 91 in new dataset

Troubleshooting
---------------

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

Reduce chunk size:

.. code-block:: python

   # Instead of writing 10 years at once
   writer.write(data_10years, ...)

   # Write 1 year at a time
   for year in range(10):
       writer.write(data_1year[year], ...)

Slow Writes
~~~~~~~~~~~

**Possible causes:**

1. Staging directory on slow storage
2. Very small chunks with high overhead
3. Complex validation

**Solutions:**

.. code-block:: python

   # Use fast local storage
   writer = cmor4.DatasetWriter(
       dataset, variable, axes,
       staging_dir="/fast/local/scratch"
   )

   # Increase chunk size (write monthly/annual instead of daily)

Disk Full During Write
~~~~~~~~~~~~~~~~~~~~~~

Zarr staging requires ~2× final file size:

.. code-block:: python

   # Check available space
   import shutil
   stat = shutil.disk_usage("/tmp")
   print(f"Free space: {stat.free / 1e9:.1f} GB")

   # Use different staging location
   writer = cmor4.DatasetWriter(
       dataset, variable, axes,
       staging_dir="/larger/disk/staging"
   )

Performance Comparison
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - ``cmorize()``
     - ``DatasetWriter``
   * - Memory usage
     - O(dataset size)
     - O(chunk size)
   * - Time specification
     - Complete upfront
     - Incremental or complete
   * - Use case
     - One-shot write
     - Streaming/incremental
   * - Complexity
     - Simple
     - Moderate
   * - Performance
     - Fast
     - Comparable
   * - Max dataset size
     - Limited by RAM
     - Unlimited

**Rule of thumb:** If your dataset fits comfortably in memory (<50% of RAM), use :func:`cmorize`. If larger or arrives incrementally, use :class:`DatasetWriter`.

Append Mode: Extending Existing Files
--------------------------------------

Use ``existing="append"`` to add new time records to an existing CMOR file:

.. code-block:: python

   # Initial write
   with cmor4.DatasetWriter(
       dataset,
       variable,
       axes,
       path="output.nc",
   ) as writer:
       writer.write(data_2000, time_values=[15.0, 45.0],
                    time_bounds=[[0.0, 30.0], [30.0, 60.0]])

   # Later, append more data
   with cmor4.DatasetWriter(
       dataset,
       variable,
       axes,
       path="output.nc",
       existing="append",
   ) as writer:
       writer.write(data_2001, time_values=[75.0],
                    time_bounds=[[60.0, 90.0]])

Append Mode Validation
~~~~~~~~~~~~~~~~~~~~~~~

When appending, DatasetWriter validates compatibility:

✓ **Checks that pass:**

- All non-time dimensions match exactly (size and values)
- All variables present in both files
- Coordinate attributes are identical
- Grid mappings match (for curvilinear grids)
- Formula terms match (for hybrid coordinates)
- Time values and bounds are strictly monotonic and contiguous

✗ **Errors raised:**

.. code-block:: python

   # Different spatial grid
   # ValueError: dimension 'lat' has size 90 in existing file and 91 in new dataset

   # Non-monotonic time
   # ValueError: Time values must be strictly monotonic

   # Gap in time bounds
   # ValueError: Time bounds must be contiguous across append boundary

   # Attribute mismatch
   # ValueError: global attribute 'experiment_id' differs

Sequential Appends
~~~~~~~~~~~~~~~~~~

You can append multiple times to the same file:

.. code-block:: python

   output_path = Path("accumulated.nc")

   # Initial write
   with cmor4.DatasetWriter(dataset, variable, axes, path=output_path) as writer:
       writer.write(chunk1, time_values=times1, time_bounds=bounds1)

   # First append
   with cmor4.DatasetWriter(
       dataset, variable, axes, path=output_path, existing="append"
   ) as writer:
       writer.write(chunk2, time_values=times2, time_bounds=bounds2)

   # Second append
   with cmor4.DatasetWriter(
       dataset, variable, axes, path=output_path, existing="append"
   ) as writer:
       writer.write(chunk3, time_values=times3, time_bounds=bounds3)

   # Result: all three chunks in one file

Incremental Formula Terms (Hybrid Coordinates)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DatasetWriter supports incremental writes for time-varying formula terms (e.g., surface pressure for hybrid coordinates):

**Method 1: Per-chunk zfactor values**

Provide time-varying zfactor values incrementally via the ``zfactors`` parameter of :meth:`write`:

.. code-block:: python

   # Define zfactors without values for time-varying terms
   zfactors = [
       project.zfactor("a", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
       project.zfactor("b", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
       project.zfactor("p0", values=100000.0),
       project.zfactor("ps"),  # No values - will provide per chunk
   ]

   with cmor4.DatasetWriter(
       dataset, variable, axes, path="hybrid.nc", zfactors=zfactors
   ) as writer:
       # First write - provide ps values for this chunk
       ps_chunk1 = np.full((12, 2, 2), 99000.0, dtype="f4")
       writer.write(
           data_2000,
           time_values=times_2000,
           time_bounds=bounds_2000,
           zfactors={"ps": ps_chunk1}
       )

       # Second write - provide ps values for next chunk
       ps_chunk2 = np.full((12, 2, 2), 99100.0, dtype="f4")
       writer.write(
           data_2001,
           time_values=times_2001,
           time_bounds=bounds_2001,
           zfactors={"ps": ps_chunk2}
       )

   # Result: ps variable has 24 time slices accumulated from both writes

**Method 2: Append mode with complete zfactor values**

Alternatively, use append mode with complete zfactor values for each write:

.. code-block:: python

   # Initial write with surface pressure
   ps_2000 = np.full((12, 2, 2), 99000.0, dtype="f4")
   zfactors_2000 = [
       project.zfactor("a", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
       project.zfactor("b", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
       project.zfactor("p0", values=100000.0),
       project.zfactor("ps", values=ps_2000),
   ]

   with cmor4.DatasetWriter(
       dataset, variable, axes, path="hybrid.nc", zfactors=zfactors_2000
   ) as writer:
       writer.write(data_2000, time_values=times_2000, time_bounds=bounds_2000)

   # Append next year with new surface pressure
   ps_2001 = np.full((12, 2, 2), 99100.0, dtype="f4")
   zfactors_2001 = [
       project.zfactor("a", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
       project.zfactor("b", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
       project.zfactor("p0", values=100000.0),
       project.zfactor("ps", values=ps_2001),
   ]

   with cmor4.DatasetWriter(
       dataset, variable, axes, path="hybrid.nc",
       zfactors=zfactors_2001, existing="append"
   ) as writer:
       writer.write(data_2001, time_values=times_2001, time_bounds=bounds_2001)

   # Result: ps variable has 24 time slices (2000 + 2001)

.. note::
   - Static zfactors (e.g., ``a``, ``b``, ``p0``) should include complete values in the initial zfactor definitions
   - Time-varying zfactors (e.g., ``ps``) can omit values and provide them per chunk via ``write()``
   - Once you start providing a time-varying zfactor in a write, you must continue providing it in all subsequent writes
   - Zfactor chunk shapes must match the variable's dimensions for that write

Provenance Tracking in Append Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When appending:

- ``tracking_id`` is regenerated (new UUID)
- ``creation_date`` is updated to current timestamp
- ``history`` attribute is preserved from the original file
- All other metadata attributes must match exactly

.. code-block:: python

   # After append, the file has:
   # - New tracking_id
   # - New creation_date
   # - Original history entry preserved
   # - All other attributes unchanged

Error Recovery
~~~~~~~~~~~~~~

Append operations are atomic - if validation fails, the original file is unchanged:

.. code-block:: python

   try:
       with cmor4.DatasetWriter(
           dataset, variable, axes,
           path="output.nc", existing="append"
       ) as writer:
           writer.write(incompatible_data, time_values=[75.0])
   except ValueError as e:
       print(f"Append failed: {e}")
       # output.nc is unchanged - original data is intact

Future Features
---------------

The following features are planned for upcoming releases:

- **Enhanced preserve mode** - Full support for reusing metadata definitions across multiple files
- **Multi-dimensional incremental writes** - Support incremental writes along spatial dimensions, not just time

Current workarounds:

- **For multiple files with gaps**: Use ``preserve_definition=True`` when closing (see Segmented Files example above)

Complete Example
----------------

Here's a full example with error handling and progress monitoring:

.. code-block:: python

   import cmor4
   import numpy as np
   from pathlib import Path

   def write_model_output(start_year, end_year, output_dir):
       """Write model output incrementally."""

       # Setup
       project = cmor4.ProjectTables.from_directory(
           "project_tables/cmip7-cmor-tables",
           cv_file="tables-cvs/cmor-cvs.json",
           variable_tables=["tables/CMIP7_ocean.json"],
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
           "outpath": str(output_dir),
       })

       variable = project.variable("tos_tavg-u-hxy-sea", table_id="ocean")

       axes = [
           project.axis("time", units="days since 1850-01-01"),
           project.axis("latitude", values=np.linspace(-90, 90, 180).tolist()),
           project.axis("longitude", values=np.linspace(0, 360, 360, endpoint=False).tolist()),
       ]

       # Write incrementally
       writer = cmor4.DatasetWriter(
           dataset,
           variable,
           axes,
           staging_dir="/tmp/cmor_staging",
       )

       try:
           for year in range(start_year, end_year):
               # Load yearly data
               time_values, time_bounds, data = load_model_year(year)

               # Write
               writer.write(data, time_values=time_values, time_bounds=time_bounds)

               # Progress
               print(f"✓ {year}: {writer._time_offset} time slices total")

       except Exception as e:
           print(f"✗ Error on year {year}: {e}")
           print(f"Partial data in: {writer.staging_path}")
           raise

       else:
           ds, output_path = writer.close()
           print(f"✓ Success: {output_path}")
           print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
           print(f"  File size: {Path(output_path).stat().st_size / 1e6:.1f} MB")
           ds.close()

   # Run it
   write_model_output(1850, 2015, Path("./output"))

See Also
--------

- :doc:`basic_usage` - Simple one-shot writes with :func:`cmorize`
- :doc:`advanced_features` - Custom grids, formula terms
- :class:`API Reference <cmor4.DatasetWriter>` - Complete class documentation
