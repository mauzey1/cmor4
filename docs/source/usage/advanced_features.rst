Advanced Features
=================

This guide covers advanced CMOR4 features for complex datasets.

Custom Grids
------------

Non-Standard Latitude/Longitude Grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define custom grid structures:

.. code-block:: python

   import cmor4
   import numpy as np

   project = cmor4.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_ocean.json"],
   )

   # Irregular latitude spacing (higher resolution at poles)
   lat_values = np.concatenate([
       np.linspace(-90, -60, 30),
       np.linspace(-60, 60, 60),
       np.linspace(60, 90, 30),
   ])

   # Compute bounds
   lat_bounds = np.column_stack([
       np.concatenate([[lat_values[0] - 1], (lat_values[:-1] + lat_values[1:]) / 2]),
       np.concatenate([(lat_values[:-1] + lat_values[1:]) / 2, [lat_values[-1] + 1]]),
   ])

   axes = [
       project.axis("time", values=[15.0], bounds=[[0, 30]],
                    units="days since 2000-01-01"),
       project.axis("latitude", values=lat_values.tolist(),
                    bounds=lat_bounds.tolist()),
       project.axis("longitude", values=np.linspace(0, 360, 180, endpoint=False).tolist()),
   ]

Curvilinear Grids
~~~~~~~~~~~~~~~~~

For non-rectangular grids (e.g., tripolar ocean grids):

.. code-block:: python

   # Grid with auxiliary coordinates
   grid = project.grid(
       "curvilinear_grid",
       latitude=lat_2d,  # 2D latitude array
       longitude=lon_2d,  # 2D longitude array
   )

   # Variable on curvilinear grid
   variable = project.variable("tos", table_id="ocean")

   # Create dataset with 2D spatial grid
   ds, path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=[time_axis],
       grid=grid,
   )

Unstructured Grids
~~~~~~~~~~~~~~~~~~

For meshes and unstructured grids:

.. code-block:: python

   # Define grid cells
   grid = project.grid(
       "unstructured_grid",
       ncells=12000,
       latitude=cell_lat,
       longitude=cell_lon,
       bounds_latitude=cell_lat_bounds,
       bounds_longitude=cell_lon_bounds,
   )

Grid Mapping Validation
~~~~~~~~~~~~~~~~~~~~~~~

CMOR4 validates grid mappings against the CF-1.12 convention's registered grid mapping names and parameters. This ensures that projected coordinate systems are properly specified:

.. code-block:: python

   # Lambert Conformal Conic projection (automatically validated)
   grid = project.grid(
       "lambert_conformal",
       grid_mapping_name="lambert_conformal_conic",
       standard_parallel=[33.0, 45.0],
       longitude_of_central_meridian=-97.0,
       latitude_of_projection_origin=40.0,
       false_easting=0.0,
       false_northing=0.0,
   )

   # CMOR4 validates:
   # 1. grid_mapping_name is a valid CF-1.12 registered name
   # 2. All required parameters for that projection are present
   # 3. Parameter names and values follow CF conventions

**Supported Grid Mappings:**

CMOR4 validates against all CF-1.12 registered grid mappings, including:

- ``albers_conical_equal_area``
- ``azimuthal_equidistant``
- ``geostationary``
- ``lambert_azimuthal_equal_area``
- ``lambert_conformal_conic``
- ``lambert_cylindrical_equal_area``
- ``latitude_longitude`` (unprojected)
- ``mercator``
- ``orthographic``
- ``polar_stereographic``
- ``rotated_latitude_longitude``
- ``sinusoidal``
- ``stereographic``
- ``transverse_mercator``
- ``vertical_perspective``
- and more...

**Grid Table Validation:**

When loading project tables, CMOR4 now performs enhanced validation on grid definitions:

.. code-block:: python

   # This will raise an error if the grid table contains:
   # - Invalid grid_mapping_name values
   # - Missing required parameters for grid mappings
   # - Inconsistent coordinate specifications
   project = cmor4.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       grid_table="tables/CMIP7_grids.json",  # Validated on load
   )

Formula Terms and Vertical Coordinates
---------------------------------------

Hybrid Sigma-Pressure Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For atmospheric model levels:

.. code-block:: python

   import cmor4
   import numpy as np

   project = cmor4.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_atmos.json"],
   )

   # Define hybrid levels
   nlev = 40
   a = np.linspace(0, 100000, nlev)  # Pressure offset
   b = np.linspace(1.0, 0.0, nlev)   # Sigma coefficient
   a_bounds = np.column_stack([a[:-1], a[1:]])
   b_bounds = np.column_stack([b[:-1], b[1:]])

   # Create hybrid axis
   lev_axis = project.axis(
       "hybrid_sigma_pressure",
       values=list(range(nlev)),
   )

   # Define formula terms
   zfactors = [
       project.zfactor("a", values=a.tolist(), bounds=a_bounds.tolist()),
       project.zfactor("b", values=b.tolist(), bounds=b_bounds.tolist()),
       project.zfactor("ps", depends_on=["time", "latitude", "longitude"]),
       project.zfactor("p0", value=100000.0),  # Reference pressure
   ]

   axes = [
       project.axis("time", values=[15.0], bounds=[[0, 30]],
                    units="days since 2000-01-01"),
       lev_axis,
       project.axis("latitude", values=np.linspace(-90, 90, 90).tolist()),
       project.axis("longitude", values=np.linspace(0, 360, 180, endpoint=False).tolist()),
   ]

   # Surface pressure (formula term variable)
   ps_data = np.random.randn(1, 90, 180) * 5000 + 101325

   # 4D atmospheric data
   data = np.random.randn(1, nlev, 90, 180) * 10 + 250.0

   variable = project.variable("ta", table_id="atmos")

   ds, path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=axes,
       zfactors=zfactors,
       zfactor_values={"ps": ps_data},
   )

Ocean Depth Coordinates
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define depth levels (meters, positive down)
   depth_values = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

   depth_axis = project.axis(
       "depth",
       values=depth_values,
       units="m",
   )

   axes = [
       project.axis("time", values=[15.0], bounds=[[0, 30]],
                    units="days since 2000-01-01"),
       depth_axis,
       project.axis("latitude", values=np.linspace(-90, 90, 180).tolist()),
       project.axis("longitude", values=np.linspace(0, 360, 360, endpoint=False).tolist()),
   ]

Time Handling
-------------

Different Calendar Types
~~~~~~~~~~~~~~~~~~~~~~~~

CMOR4 supports various calendar types:

.. code-block:: python

   # Standard Gregorian calendar
   time_axis = project.axis(
       "time",
       values=[15.0, 45.0, 74.0],
       bounds=[[0, 30], [30, 60], [60, 90]],
       units="days since 1850-01-01",
       calendar="gregorian",
   )

   # No-leap calendar (365-day years)
   time_axis = project.axis(
       "time",
       values=[15.0, 45.0, 74.0],
       bounds=[[0, 30], [30, 60], [60, 90]],
       units="days since 1850-01-01",
       calendar="noleap",
   )

   # 360-day calendar
   time_axis = project.axis(
       "time",
       values=[15.0, 45.0, 74.0],
       bounds=[[0, 30], [30, 60], [60, 90]],
       units="days since 1850-01-01",
       calendar="360_day",
   )

Forecast Coordinates (Reference Time and Lead Time)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CMOR4 automatically derives forecast coordinates for forecast/initialized model output. When you provide a ``forecast_reference_time`` (reftime) coordinate, CMOR4 automatically generates the corresponding ``forecast_period`` (leadtime) coordinate:

.. code-block:: python

   import cmor4
   import numpy as np

   project = cmor4.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_atmos.json"],
   )

   # Define dataset with forecast metadata
   dataset = project.dataset_info({
       "mip_era": "CMIP7",
       "activity_id": "DCPP",  # Decadal Climate Prediction Project
       "institution_id": "PCMDI",
       "source_id": "PCMDI-MODEL",
       "experiment_id": "dcppA-hindcast",
       "license_id": "CC-BY-4.0",
       "variant_label": "r1i1p1f1",
       "grid_label": "gn",
       "outpath": "./output",
   })

   # Define forecast reference time (initialization time)
   reftime_axis = project.axis(
       "reftime",
       out_name="reftime",
       values=[0.0],  # Single initialization time
       units="days since 2020-01-01",
       standard_name="forecast_reference_time",
       scalar=True,
   )

   # Define forecast time axis
   time_axis = project.axis(
       "time",
       values=[30.0, 60.0, 90.0],  # Monthly forecasts
       bounds=[[0, 30], [30, 60], [60, 90]],
       units="days since 2020-01-01",
   )

   axes = [time_axis, reftime_axis, lat_axis, lon_axis]

   # CMOR4 automatically generates leadtime coordinate
   # leadtime[i] = time[i] - reftime
   ds, path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=axes,
   )

   # Output contains:
   # - time: [30.0, 60.0, 90.0] days since 2020-01-01
   # - reftime: 0.0 days since 2020-01-01 (scalar coordinate)
   # - leadtime: [30.0, 60.0, 90.0] days (automatically derived)

**Multiple Initialization Times:**

For ensemble forecasts with multiple initialization times:

.. code-block:: python

   # Non-scalar reftime for multiple initializations
   reftime_axis = project.axis(
       "reftime",
       out_name="reftime",
       values=[0.0, 365.0],  # Two annual initializations
       units="days since 2020-01-01",
       standard_name="forecast_reference_time",
       scalar=False,
   )

   # Time axis spans both forecasts
   time_axis = project.axis(
       "time",
       values=[30.0, 60.0, 395.0, 425.0],
       units="days since 2020-01-01",
   )

   # Data has shape (time, reftime, lat, lon)
   # CMOR4 derives leadtime for each time-reftime pair

**Incremental Writing with Forecasts:**

DatasetWriter also supports incremental writing of forecast data:

.. code-block:: python

   with cmor4.DatasetWriter(dataset, variable, axes) as writer:
       for init_time in initialization_times:
           # Load forecast data for this initialization
           reftime = init_time
           time_values = np.arange(reftime + 30, reftime + 365, 30)
           data = load_forecast_data(init_time)

           writer.write(
               data,
               time_values=time_values.tolist(),
           )

   # Leadtime automatically computed and written for each chunk

Climatological Time
~~~~~~~~~~~~~~~~~~~

For climatology data (multi-year means):

.. code-block:: python

   # Monthly climatology from 1981-2010
   time_values = [15.0, 45.0, 74.0, 105.0, 135.0, 165.0,
                  195.0, 225.0, 255.0, 285.0, 315.0, 345.0]

   # Climatology bounds span 30 years for each month
   climatology_bounds = [
       [cftime.datetime(1981, 1, 1), cftime.datetime(2010, 1, 31)],
       [cftime.datetime(1981, 2, 1), cftime.datetime(2010, 2, 28)],
       # ... etc for all months
   ]

   time_axis = project.axis(
       "time",
       values=time_values,
       bounds=time_values,  # Regular time bounds
       climatology_bounds=climatology_bounds,
       units="days since 1850-01-01",
   )

Working with Different Variable Types
--------------------------------------

Scalar Variables
~~~~~~~~~~~~~~~~

Variables with no spatial dimensions:

.. code-block:: python

   # Global mean surface temperature (time series only)
   time_axis = project.axis(
       "time",
       values=np.arange(0, 3650, 30.0).tolist(),
       units="days since 1850-01-01",
   )

   data = np.random.randn(len(time_axis.values)) * 0.5 + 288.0

   ds, path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=[time_axis],
   )

Site-Level Data
~~~~~~~~~~~~~~~

Data at specific locations:

.. code-block:: python

   # Define sites
   site_axis = project.axis(
       "site",
       values=["station_1", "station_2", "station_3"],
   )

   # Site coordinates
   site_lat = [45.0, 50.0, 55.0]
   site_lon = [-120.0, -110.0, -100.0]

   axes = [
       project.axis("time", values=[15.0], bounds=[[0, 30]],
                    units="days since 2000-01-01"),
       site_axis,
   ]

   data = np.random.randn(1, 3) * 5 + 288.0  # (time, site)

   ds, path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=axes,
       coords={"latitude": (["site"], site_lat),
               "longitude": (["site"], site_lon)},
   )

Multiple Experiments and Variants
----------------------------------

Processing Multiple Variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   variants = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]

   for variant in variants:
       # Update dataset metadata
       dataset_info = project.dataset_info({
           "mip_era": "CMIP7",
           "activity_id": "CMIP",
           "institution_id": "PCMDI",
           "source_id": "PCMDI-MODEL",
           "experiment_id": "historical",
           "license_id": "CC-BY-4.0",
           "variant_label": variant,
           "grid_label": "gn",
           "outpath": f"./output/{variant}",
       })

       # Load ensemble member data
       data = load_ensemble_data(variant)

       ds, path = cmor4.cmorize(
           data=data,
           dataset=dataset_info,
           variable=variable,
           axes=axes,
       )

       print(f"Wrote {variant}: {path}")
       ds.close()

Multiple Experiments
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   experiments = ["historical", "ssp245", "ssp585"]

   for experiment in experiments:
       dataset_info = project.dataset_info({
           "mip_era": "CMIP7",
           "activity_id": "ScenarioMIP" if "ssp" in experiment else "CMIP",
           "institution_id": "PCMDI",
           "source_id": "PCMDI-MODEL",
           "experiment_id": experiment,
           "license_id": "CC-BY-4.0",
           "variant_label": "r1i1p1f1",
           "grid_label": "gn",
           "outpath": f"./output/{experiment}",
       })

       # Load experiment-specific data
       data = load_experiment_data(experiment)

       ds, path = cmor4.cmorize(
           data=data,
           dataset=dataset_info,
           variable=variable,
           axes=axes,
       )

       print(f"Wrote {experiment}: {path}")
       ds.close()

Custom Validation
-----------------

Relaxed Validation
~~~~~~~~~~~~~~~~~~

For debugging or non-standard use cases:

.. code-block:: python

   # Disable certain validations (use with caution!)
   dataset_info = project.dataset_info(
       metadata_dict,
       validate_cv=False,  # Skip CV validation
   )

Custom Quality Control
~~~~~~~~~~~~~~~~~~~~~~

Add your own validation logic:

.. code-block:: python

   def validate_data(data, variable):
       """Custom validation checks."""
       # Check for reasonable values
       if variable.name == "tas":
           if np.any(data < 180) or np.any(data > 330):
               raise ValueError(f"Temperature out of range: [{data.min()}, {data.max()}]")

       # Check for suspicious patterns
       if np.all(data == data[0]):
           raise ValueError("Data appears to be constant")

       # Check for NaN/Inf
       if np.any(~np.isfinite(data)):
           raise ValueError("Data contains NaN or Inf")

   # Use before cmorize
   validate_data(data, variable)
   ds, path = cmor4.cmorize(data, dataset, variable, axes)

Parallel Processing
-------------------

Process Multiple Variables in Parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from multiprocessing import Pool
   import cmor4

   def process_variable(var_name):
       """Process a single variable."""
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

       variable = project.variable(var_name, table_id="atmos")
       data = load_data(var_name)

       ds, path = cmor4.cmorize(data, dataset, variable, axes)
       ds.close()

       return path

   # Process in parallel
   variables = ["tas", "pr", "hus", "ua", "va"]

   with Pool(processes=4) as pool:
       paths = pool.map(process_variable, variables)

   print(f"Wrote {len(paths)} files")

Integration with Xarray
------------------------

Starting from Xarray Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import xarray as xr
   import cmor4

   # Load existing dataset
   ds_in = xr.open_dataset("model_output.nc")

   # Extract data and coordinates
   data = ds_in["temperature"].values
   time_values = ds_in["time"].values
   lat_values = ds_in["lat"].values
   lon_values = ds_in["lon"].values

   # Convert to CMOR-compliant
   axes = [
       project.axis("time", values=time_values.tolist(),
                    units="days since 1850-01-01"),
       project.axis("latitude", values=lat_values.tolist()),
       project.axis("longitude", values=lon_values.tolist()),
   ]

   ds_out, path = cmor4.cmorize(
       data=data,
       dataset=dataset,
       variable=variable,
       axes=axes,
   )

Post-Processing CMOR Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create CMOR file
   ds, path = cmor4.cmorize(data, dataset, variable, axes)
   ds.close()

   # Open and post-process
   ds = xr.open_dataset(path)

   # Compute derived quantities
   annual_mean = ds.groupby("time.year").mean()

   # Save derived product
   annual_mean.to_netcdf("annual_mean.nc")

Working with obs4MIPs
---------------------

Observational Data
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load obs4MIPs tables
   project = cmor4.ProjectTables.from_directory(
       "project_tables/obs4MIPs-cmor-tables",
       cv_file="obs4MIPs_CVs.json",
       variable_tables=["obs4MIPs_satellite.json"],
   )

   # obs4MIPs metadata
   dataset = project.dataset_info({
       "mip_era": "obs4MIPs",
       "activity_id": "obs4MIPs",
       "institution_id": "NASA-JPL",
       "source_id": "AIRS",
       "variant_label": "v20200101",
       "outpath": "./output",
   })

   # Process observational data
   variable = project.variable("ta", table_id="satellite")

   ds, path = cmor4.cmorize(
       data=obs_data,
       dataset=dataset,
       variable=variable,
       axes=axes,
   )

Large Dataset Strategies
-------------------------

Chunked Processing
~~~~~~~~~~~~~~~~~~

For datasets too large for memory:

.. code-block:: python

   # Use DatasetWriter for incremental writes
   with cmor4.DatasetWriter(dataset, variable, axes) as writer:
       # Process in time chunks
       for year in range(1850, 2015):
           # Load one year at a time
           data_chunk = load_year_data(year)
           time_chunk = generate_time_coords(year)

           writer.write(data_chunk, time_values=time_chunk)

           # Free memory
           del data_chunk

Working with Dask Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~

DatasetWriter already uses Dask internally for memory-efficient finalization, but you can also use Dask arrays as input for lazy loading:

.. code-block:: python

   import dask.array as da
   import cmor4

   # Load data lazily from existing Zarr store
   data = da.from_zarr("large_dataset.zarr")

   # Write in chunks - only compute what's needed
   with cmor4.DatasetWriter(dataset, variable, axes) as writer:
       for i in range(0, len(data), chunk_size):
           # Compute only this chunk (lazy evaluation)
           chunk = data[i:i+chunk_size].compute()
           writer.write(chunk, time_values=time_values[i:i+chunk_size])

This is useful when your source data is already in Zarr format or when combining DatasetWriter with Dask-based processing pipelines.

Memory Profiling
~~~~~~~~~~~~~~~~

Monitor memory usage:

.. code-block:: python

   import tracemalloc
   import cmor4

   tracemalloc.start()

   # Your CMOR processing
   with cmor4.DatasetWriter(dataset, variable, axes) as writer:
       for chunk in data_chunks:
           writer.write(chunk, time_values=times)

           # Check memory
           current, peak = tracemalloc.get_traced_memory()
           print(f"Current: {current / 1e6:.1f} MB, Peak: {peak / 1e6:.1f} MB")

   tracemalloc.stop()

See Also
--------

- :doc:`basic_usage` - Simple examples
- :doc:`incremental_writing` - DatasetWriter details
- :doc:`/api/classes` - Complete API documentation
