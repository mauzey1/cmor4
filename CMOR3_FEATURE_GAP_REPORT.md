# CMOR3 Feature Gap Report for CMOR4

Date: 2026-06-30

## Scope

This report compares the local CMOR4 repository with a shallow checkout of
`https://github.com/PCMDI/cmor` into `/private/tmp/cmor3-pcmdi-cmor`.

CMOR4 already covers many CMOR3 behaviors around CMIP6/CMIP7 controlled
vocabulary validation, DRS templates, scalar axes, hybrid z-factors, grids,
climatology bounds, external variables, history/tracking attributes, and CMIP7
chunking. The gaps below are features present in CMOR3 that are not supported,
or are only partially represented, in CMOR4.

## Summary

CMOR4 is currently a Python/xarray whole-dataset builder. CMOR3 is a stateful
C library with Python and Fortran bindings, numeric object IDs, mutable current
dataset state, incremental writes, file append modes, explicit NetCDF packing
controls, and a larger compatibility API.

Not every CMOR3 behavior is a CMOR4 product goal. CMOR4 should not attempt to
recreate CMOR3's legacy stateful API, compiled interfaces, JSON-driver entry
point, mutable attribute APIs, packing helper functions, cdms2/MV2 adapters,
logging/exit-control system, or NetCDF mode constants. The features worth
carrying forward are the ones that support modern CMOR4 production workflows:
incremental writes, append/preserve-style output, companion variable writes,
enhanced grid mapping/CRS support, and forecast leadtime handling.

## CMOR4 Scope Decision

- Legacy compatibility, do not implement: gaps 1, 2, 3, and 4.
- Target CMOR4 features: gaps 5, 6, 7, 10, and 11.
- Ignore for now: gaps 8, 9, 12, 13, and 14.

## Feature Gaps

### 1. Drop-in CMOR3 Python API

Status: legacy CMOR3 compatibility; not a CMOR4 implementation target.

CMOR3 exposes a procedural Python API from `Lib/__init__.py` and
`Lib/pywrapper.py`, including:

- `setup`, `dataset_json`, `load_table`, `set_table`
- `axis`, `variable`, `zfactor`, `grid`, `set_grid_mapping`, `set_crs`
- `write`, `close`, `get_final_filename`
- dataset and variable attribute mutation/query helpers
- compression, quantization, chunking, climatology, and terminate-signal helpers

CMOR4 exports object-oriented helpers such as `ProjectTables`, `DatasetInfo`,
`Axis`, `Variable`, `Grid`, `ZFactor`, `create_dataset`, `write_netcdf`, and
`cmorize`. It does not provide CMOR3-compatible function names, numeric IDs, or
the same stateful workflow.

Impact: existing CMOR3 Python drivers cannot run against CMOR4 without being
rewritten.

### 2. C and Fortran Interfaces

Status: legacy CMOR3 compatibility; not a CMOR4 implementation target.

CMOR3 includes a C API in `include/cmor.h` / `Src/*.c` and Fortran bindings in
`Src/cmor_fortran_interface.f90` and `Src/cmor_cfortran_interface.c`.

CMOR4 is a Python package only. There is no C ABI, Fortran module, or compiled
library interface.

Impact: C and Fortran CMOR3 workflows, including compiled model post-processing
drivers, are not supported by CMOR4.

### 3. Stateful Session Management

Status: legacy CMOR3 compatibility; not a CMOR4 implementation target.

CMOR3 maintains global state after `cmor.setup()`: current dataset attributes,
loaded tables, current table, axes, variables, grids, and open NetCDF files.
APIs such as `set_table()`, `set_cur_dataset_attribute()`,
`get_cur_dataset_attribute()`, and `has_cur_dataset_attribute()` operate on that
state.

CMOR4 uses immutable-ish metadata objects and explicit function arguments. It
does not have a global current dataset, table registry, current table pointer,
or ID registry.

Impact: CMOR3 code that depends on late mutation or implicit current-state
resolution has no direct CMOR4 equivalent.

### 4. `dataset_json()` Compatibility

Status: legacy CMOR3 compatibility; not a CMOR4 implementation target.

CMOR3 reads dataset metadata from a JSON file via `dataset_json()`. The JSON can
also name side tables through keys such as `_controlled_vocabulary_file`,
`_AXIS_ENTRY_FILE`, and `_FORMULA_VAR_FILE`, which CMOR3 uses during table
loading.

CMOR4 expects the caller to construct `ProjectTables` with explicit table paths
and pass dataset metadata as a mapping to `ProjectTables.dataset_info()`.

Impact: CMOR3 input JSON files are not directly executable CMOR4 inputs. A
translation layer is needed.

### 5. Incremental and Streaming Writes

Status: target CMOR4 feature.

CMOR3 supports repeated `write()` calls for the same variable, with
`ntimes_passed`, `time_vals`, and `time_bnds` supplied at write time. Several
CMOR3 tests exercise this, including `test_python_forecast_coordinates.py`,
`test_time_gap_multi_write.py`, and `test_cmor_python_not_enough_times_written.py`.

CMOR4 builds an `xarray.Dataset` from complete arrays and writes it once through
`write_netcdf()` or `cmorize()`. Axis values and bounds must be known when the
dataset is created.

Impact: streaming workflows that write one time slice or block at a time are
not supported.

### 6. Append, Preserve, Replace, and Close Semantics

Status: target CMOR4 feature, using CMOR4-native names and behavior rather
than CMOR3 constants.

CMOR3 supports `CMOR_PRESERVE`, `CMOR_APPEND`, `CMOR_REPLACE`, and NetCDF3/4
variants via `setup(netcdf_file_action=...)`. It also supports
`close(var_id, file_name=True, preserve=True)` and `get_final_filename()`.

CMOR4 writes to a path using xarray. It can overwrite if the backend allows it,
but it does not implement CMOR3's file action modes, append-to-existing-file
workflow, close-preserve behavior, or final filename state.

Impact: CMOR3 workflows that split writes across runs or append later time
ranges to an existing file are not supported.

### 7. Associated-Variable Writes and `store_with`

Status: target CMOR4 feature, expressed as companion variables/zfactors rather
than CMOR3 numeric-variable IDs.

CMOR3 `write()` accepts `store_with`, allowing z-factor or associated-variable
data to be written into the same output file as another variable. For example,
`Test/test_cmor_associated_variable.py` writes `ps` with `store_with=cmorVar`.

CMOR4 can include z-factors and grid-associated coordinate variables when all
metadata and data are provided up front, but it does not support a separate
`store_with` write operation or delayed associated-variable data injection.

Impact: associated variables must be known before dataset creation in CMOR4.

### 8. Per-Variable and Current-Dataset Attribute Mutation

Status: ignore. CMOR4 should continue using explicit metadata objects and
object updates rather than mutable global/current state.

CMOR3 supports late mutation/query APIs:

- `set_cur_dataset_attribute`, `get_cur_dataset_attribute`,
  `has_cur_dataset_attribute`
- `set_variable_attribute`, `get_variable_attribute`, `has_variable_attribute`
- `set_furtherinfourl`

CMOR4 supports attributes through object construction (`DatasetInfo`,
`Variable.attrs`, `Axis.attrs`, `Grid.attrs`, `ZFactor.attrs`) and the `attrs=`
argument to `create_dataset()` / `cmorize()`. It does not support CMOR3-style
ID-based mutation after variable creation.

Impact: CMOR3 code that patches metadata after objects are defined needs to be
rewritten to build new metadata objects before dataset creation.

### 9. Explicit NetCDF Packing Controls

Status: ignore as CMOR3 API parity. CMOR4 already accepts xarray-style
`encoding=` and validates CMIP7 chunking.

CMOR3 provides validated per-variable controls:

- `set_deflate(var_id, shuffle, deflate, deflate_level)`
- `set_zstandard(var_id, zstandard_level)`
- `set_quantize(var_id, quantize_mode, quantize_nsd)`
- `set_chunking(var_id, chunking_dimensions)`

CMOR4 has an `encoding=` argument and CMIP7 auto-chunking/validation, but it
does not expose CMOR3-compatible packing functions or CMOR3's validation rules
for zstandard and quantization.

Impact: users can sometimes pass equivalent xarray/netCDF encodings manually,
but CMOR3 packing workflows are not directly supported.

### 10. Enhance `Grid` for CMOR3-like Grid-Mapping/CRS Behavior

Status: target CMOR4 feature. The goal is not to add CMOR3's `set_crs()` API;
it is to enhance the `Grid` class and project-table validation so CMOR4 can
produce equivalent CF grid-mapping/CRS metadata where useful.

CMOR3 has both `set_grid_mapping()` and `set_crs()`. The CMOR3 implementation
validates parameter names against grid mapping definitions, checks required
projection axes, warns about missing required parameters, supports text
parameters such as `crs_wkt` and `GeoTransform`, and normalizes some projection
parameters such as paired `standard_parallel1` / `standard_parallel2` into the
CF `standard_parallel` attribute.

CMOR4 can write a scalar grid mapping variable through `Grid`, including custom
attributes, latitude/longitude arrays, and vertices. It does not provide the
CMOR3 `set_crs()` API, does not implement the same required-parameter warning
logic, and does not fully emulate CMOR3's projection-parameter normalization.

Impact: projected-grid output is partially supported, but exact CMOR3 CRS/grid
mapping behavior is not.

### 11. Forecast Reference Time and Lead Time Derivation

Status: target CMOR4 feature.

CMOR3 tests such as `test_python_forecast_time.py` and
`test_python_forecast_coordinates.py` exercise `reftime` axes and automatic
lead-time handling. CMOR3 can write reference-time coordinates and derive
`leadtime` from forecast time metadata.

CMOR4 has generic axis support, so users can manually create extra coordinates,
but there is no dedicated forecast-reference-time or automatic leadtime
derivation logic.

Impact: forecast products using CMOR3 leadtime semantics require manual
coordinate construction or new CMOR4 support.

### 12. cdms2/MV2 Compatibility

Status: ignore. CMOR4 should remain numpy/xarray oriented.

CMOR3's Python wrapper accepts `cdms2` axes/variables, `MV2`, and masked arrays.
It extracts units, bounds, intervals, and data from those objects before calling
the C API.

CMOR4 is based on numpy and xarray. It can consume array-like values, but it has
no explicit cdms2/MV2 integration layer.

Impact: older CMOR3 drivers using cdms2 objects need data-conversion glue.

### 13. CMOR3 Logging, Verbosity, Exit Control, and Terminate Signal

Status: ignore. CMOR4 should continue using Python exceptions and warnings.

CMOR3 supports `set_verbosity`, `exit_control`, `logfile`,
`set_terminate_signal()`, and `get_terminate_signal()`. These affect how CMOR3
logs warnings/errors and whether it exits on warnings or major errors.

CMOR4 uses Python exceptions and warnings. It does not implement CMOR3's
logging file, verbosity constants, exit-control modes, or terminate-signal
behavior.

Impact: scripts that depend on CMOR3 process-control or log-file behavior need
different error handling under CMOR4.

### 14. NetCDF3 / NetCDF4 Mode Selection Constants

Status: ignore. CMOR4 should continue forwarding backend choices through
`xarray.Dataset.to_netcdf()` arguments.

CMOR3 exposes constants such as `CMOR_REPLACE_3`, `CMOR_REPLACE_4`,
`CMOR_APPEND_3`, and `CMOR_APPEND_4` to select output format and file action.

CMOR4 delegates writing to `xarray.Dataset.to_netcdf()` and does not expose
CMOR3 constants or a CMOR-specific file-mode abstraction.

Impact: equivalent output format choices, where possible, must be expressed
through xarray backend arguments rather than CMOR3 constants.

## Implementation Exploration for Target Features

### Gaps 5 and 6: Incremental Writes and Append/Preserve-Style Output

CMOR4 should add a CMOR4-native writer object, not a CMOR3 global session. The
core idea is to keep `create_dataset()` as the authoritative validation and
dataset-construction path, then add a layer that accumulates or appends data
chunks and calls the existing machinery at safe boundaries.

Recommended public shape:

```python
writer = cmor4.DatasetWriter(
    dataset=dataset,
    variable=variable,
    axes=axes,
    zfactors=zfactors,
    grid=grid,
    path=path,
    existing="error",  # "error", "replace", or "append"
)

writer.write(data, time_values=[...], time_bounds=[...])
writer.write(data2, time_values=[...], time_bounds=[...])
result = writer.close(preserve_definition=False)
```

Stage 1 should be a chunk-collector implementation:

- Store each `write()` call as a chunk with data, time values, time bounds, and
  time-dependent zfactor/companion data.
- On `close()`, concatenate chunks along the time dimension, build a complete
  set of axes, call `create_dataset()`, and write through `write_netcdf()`.
- For `existing="append"`, open the existing output file with xarray,
  concatenate the existing dataset and the new dataset along `time`, validate
  metadata compatibility, and write a replacement file through a temporary
  path.
- For `preserve_definition=True`, clear accumulated chunks after close but keep
  the metadata object reusable for a new output segment.

This gives users the repeated-write workflow they need without immediately
taking on direct NetCDF unlimited-dimension mutation. It also keeps all current
validation and path-generation behavior centralized.

Stage 2 should add a Zarr-based disk-backed writer for large datasets:

- Create a temporary Zarr staging store containing the main variable, time
  coordinates and bounds, zfactor/companion arrays, grid coordinate arrays, and
  a small CMOR4 metadata document with dimensions, attributes, encodings, and
  write state.
- Append each `write()` call directly to the staging arrays, using Zarr chunks
  as the persistence boundary. This keeps memory bounded to the incoming chunk
  plus metadata instead of allocating the whole dataset.
- On `close()`, stream chunks from the Zarr store into the final NetCDF file.
  The finalizer can use netCDF4 or h5netcdf directly, defining dimensions,
  variables, attributes, fill values, compression/chunking, and bounds
  variables from CMOR4 metadata.
- Do not require reopening the staged data as an `xarray.Dataset`. Xarray can
  still be useful for the existing in-memory path and for Zarr operations, but
  the disk-backed writer should be able to finalize Zarr -> NetCDF under
  CMOR4's direct control.
- Keep the same `DatasetWriter` public API so users do not need to care which
  backend is selected.

Xarray note: xarray does not provide a strong API for incrementally building a
NetCDF file on disk slice by slice. It can write Zarr incrementally with
`to_zarr(append_dim=...)` or `to_zarr(region=...)`, and it can write lazy/dask
arrays to NetCDF, but the current CMOR4 API still requires the full logical
data object up front. A Zarr staging backend is therefore the best fit for true
incremental CMOR4 production while staying close to the xarray ecosystem.

Current API limitation:

- `create_dataset()` and `cmorize()` can avoid some peak memory if callers pass
  dask-backed or otherwise lazy arrays, but they are not streaming APIs.
- With ordinary numpy input, the whole data array must already exist in memory.
- Coordinates and bounds are also expected up front.
- A Zarr-backed `DatasetWriter` would be the first CMOR4 API that accepts data
  as it arrives and persists it without whole-dataset allocation.

Validation rules needed for both stages:

- The write dimension must be explicit, usually `time`.
- Every chunk must match the non-time shape implied by `variable`, `axes`, and
  `grid`.
- Time values and bounds must be monotonic and contiguous unless the table
  explicitly allows gaps.
- Appended files must match dataset global attributes, variable metadata,
  non-time axes, grid mapping metadata, zfactor definitions, units, dtype, and
  encoding-relevant fields.
- The final time range should be recomputed after concatenation, then used for
  output filename/path rendering.

### Gap 7: Companion Variables and `store_with` Equivalent

CMOR4 should not expose CMOR3's `store_with` integer-ID mechanism. The CMOR4
equivalent should be explicit companion data supplied to `create_dataset()` or
to the proposed `DatasetWriter`.

Recommended model:

```python
companion = cmor4.CompanionVariable(
    variable=cmor4.Variable(...),
    data=values,
    role="cell_measure",  # or "coordinate", "formula_term", "ancillary"
)
```

Near-term implementation can focus on zfactors because CMOR4 already has a
`ZFactor` model and `_add_zfactor()` path. The writer can allow zfactor values
to be supplied per chunk:

```python
writer.write(
    data,
    time_values=[...],
    zfactors={"ps": surface_pressure_slice},
)
```

Longer term, add a generic `companions=` argument:

- `create_dataset(..., companions=[...])`
- `DatasetWriter(..., companions=[...])`
- companion variables are included in the same dataset only when referenced by
  `formula_terms`, `coordinates`, `cell_measures`, `ancillary_variables`, or an
  explicit role.
- companion dimensions are resolved through the same axis/grid dimension map as
  the primary variable.
- companion data undergoes the same NaN/range validation where table metadata
  is available.

This preserves CMOR4's explicit data model while covering the practical CMOR3
case where a variable such as `ps` is written into the same file as a hybrid
coordinate variable.

### Gap 10: Enhanced Grid Mapping and CRS Support

The `Grid` class already writes a scalar mapping variable and handles
lat/lon/vertex arrays. The enhancement should be a stronger `Grid`
normalization and validation layer, driven by the grid table when available.

Recommended additions:

- Add a table-derived `GridMappingSpec` internal representation with:
  allowed parameter names, required parameter names, optional parameter names,
  expected projection axes, text parameters, and alternate-required groups.
- Normalize paired parameters where CMOR3 produced CF attributes, especially
  `standard_parallel1` and `standard_parallel2` into a numeric
  `standard_parallel` vector.
- Support text CRS attributes directly through `Grid.attrs` or a new
  `crs_attrs` field, including `crs_wkt`, `GeoTransform`, `long_name`, and
  other CF/GDAL-style metadata.
- Validate grid mapping axes using the existing `Grid.axes` and
  `ProjectTables.validate_dataset()` path.
- Warn, rather than fail, for missing recommended parameters where CMOR3 only
  warned.
- Fail for table-defined conflicts, invalid axis count/order, and values that
  cannot be represented in CF-compliant output.

Potential API:

```python
grid = project.grid(
    "lambert_conformal_conic",
    axes=[y_axis, x_axis],
    latitude=lat,
    longitude=lon,
    params={
        "standard_parallel1": (-20.0, "degrees_north"),
        "standard_parallel2": (20.0, "degrees_north"),
        "longitude_of_central_meridian": (175.0, "degrees_east"),
        "latitude_of_projection_origin": (13.0, "degrees_north"),
        "false_easting": (8.0, "m"),
        "false_northing": (0.0, "m"),
    },
    attrs={"crs_wkt": "..."},
)
```

The generated scalar mapping variable should contain CF-native attributes such
as:

- `grid_mapping_name = "lambert_conformal_conic"`
- `standard_parallel = [-20.0, 20.0]`
- numeric projection attributes without auxiliary `*_units` attributes unless
  the project explicitly requires them
- text CRS attributes such as `crs_wkt` when supplied

### Gap 11: Forecast Reference Time and Leadtime

CMOR4 should add forecast-coordinate support as a normal coordinate-generation
feature, not as CMOR3 API compatibility.

Recommended behavior:

- If a dataset includes a forecast time axis and a scalar or size-one
  `reftime` axis, CMOR4 can derive `leadtime = time - reftime`.
- The derived `leadtime` coordinate should use metadata from the coordinate
  table when available.
- If the user explicitly supplies a `leadtime` axis, CMOR4 should validate it
  against the derived values rather than overwrite it silently.
- If no `reftime` is present, no leadtime derivation should happen.

Recommended API options:

1. Automatic derivation in `create_dataset()`:
   detect axes named `reftime`, `reftime1`, or with standard/table metadata
   indicating reference time; derive `leadtime` when the variable/table expects
   it.
2. Explicit helper for clarity:

```python
forecast = cmor4.ForecastCoordinates(
    reference_time=reftime_axis,
    leadtime_name="leadtime",
)
ds = cmor4.create_dataset(..., forecast=forecast)
```

Option 1 is friendlier for project-table-driven workflows. Option 2 is easier
to reason about and test. A practical path is to implement the internal helper
first and expose the explicit `ForecastCoordinates` object only if automatic
detection becomes ambiguous.

Implementation details:

- Use `cftime.num2date()` and existing helpers in `_time_utils.py` to handle
  calendar-aware differences.
- Keep leadtime units explicit, probably matching the time axis unit unless the
  coordinate table declares otherwise.
- Ensure filename time ranges continue to use forecast-valid time, not leadtime.
- Add tests based on CMOR3's `test_python_forecast_time.py` and
  `test_python_forecast_coordinates.py`, rewritten in CMOR4 style.

## Suggested Implementation Order

1. Add a chunk-collector `DatasetWriter` for repeated writes and close-time
   dataset creation.
2. Extend that writer to support `existing="append"` by concatenating an
   existing CMOR4 file and rewriting atomically.
3. Add a Zarr-backed `DatasetWriter` backend that persists chunks as they
   arrive and finalizes by streaming Zarr chunks into NetCDF.
4. Add per-chunk zfactor data and then a generic companion-variable model.
5. Enhance `Grid.mapping_attributes()` and project-table grid validation for
   CRS normalization and richer warnings.
6. Add forecast leadtime derivation and validation.
7. Revisit direct NetCDF unlimited-dimension appends only if Zarr staging is
   not sufficient for performance, dependency, or operational reasons.

## Lower-Priority or Partial Gaps

- CMOR3's compiled-library limit constants (`CMOR_MAX_*`, `CMOR_VERSION_*`,
  `CMOR_CF_VERSION_*`) are not exported by CMOR4.
- CMOR3's `get_original_shape()` helper, used by its wrapper to validate write
  shape before passing flattened arrays into C, has no CMOR4 equivalent.
- CMOR3's `set_climatology()` / `get_climatology()` process-global flag is not
  present. CMOR4 supports climatology axes through axis metadata instead.
- CMOR3's exact warning/error text and log formatting are not generally
  preserved, even where CMOR4 validates the same condition.

## Features That Appear Covered or Mostly Covered

These were found in CMOR3 and already have CMOR4 support or parity tests:

- CMIP6/CMIP7 CV validation for institution, source, experiment, parent,
  sub-experiment, activity, grid label, variant/RIPF, forcing terms, and
  required global attributes.
- DRS path and filename templates from CV tables.
- Tracking ID and history attribute generation.
- External variables derived from `cell_measures`.
- Scalar axes and singleton coordinates.
- Hybrid-coordinate z-factors when all values are supplied up front.
- Curvilinear/unstructured latitude/longitude grids and vertices.
- Basic grid mapping variables.
- Climatology bounds.
- Time-interval, bounds, monotonicity, longitude normalization, NaN, and
  variable range checks.
- CMIP7 auto-chunking and chunk validation.

## Recommendation

CMOR4 should stay intentionally different from CMOR3 where CMOR3's behavior is
mainly legacy API compatibility. The implementation focus should be:

1. A CMOR4-native `DatasetWriter` for repeated writes, append-style workflows,
   and reusable metadata definitions.
2. Companion-variable support, starting with per-chunk zfactor data and then
   extending to explicitly referenced cell-measure/coordinate/ancillary
   variables.
3. `Grid` enhancements for CF grid-mapping/CRS behavior comparable to CMOR3
   output, without adding CMOR3's `set_crs()` procedural API.
4. Forecast reference-time and leadtime derivation.

The legacy or ignored gaps should remain documented only as non-goals unless a
future user requirement changes CMOR4's scope.
