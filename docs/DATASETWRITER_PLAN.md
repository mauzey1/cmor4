# DatasetWriter Implementation Plan

## Context

CMOR4 currently operates as a single-pass, whole-dataset builder: users provide complete metadata, axes with all coordinate values, and full data arrays to `create_dataset()` or `cmorize()`, which validates everything and writes a complete NetCDF file in one operation. This workflow works well for datasets that fit in memory but does not support:

1. **Incremental writes** — writing one time slice at a time as data becomes available
2. **Append mode** — adding new time records to an existing output file
3. **Preserve mode** — reusing metadata definitions across multiple output segments
4. **Large datasets** — datasets where holding all time slices in memory simultaneously is impractical

Many climate modeling workflows produce data incrementally over time (e.g., monthly or annual output from long simulations) and would benefit from being able to write CMOR-compliant output as the data becomes available rather than waiting for the complete time series. Similarly, users may want to extend previously written datasets with new time records or split large time series across multiple files while reusing the same metadata definition.

The goal is to add CMOR4-native incremental write capability that integrates with CMOR4's existing validation, axis construction, and path-generation logic while providing a streaming interface for time-series data. This should use modern Python patterns (context managers, explicit method calls) rather than replicating legacy stateful global-session designs.

## Implementation Strategy: Zarr-Backed Staging

Two potential approaches were considered for implementing incremental writes:

- **Approach 1**: In-memory chunk collector that accumulates write() calls and builds the complete dataset at close() time
- **Approach 2**: Zarr-backed disk staging that persists each chunk immediately and streams to NetCDF at close() time

### Decision: Use Zarr-Backed Staging (Approach 2)

**Rationale for Zarr-backed approach:**

1. **Memory is the core motivation** — If users can hold the full dataset in memory, they can already use `create_dataset()` with lazy dask arrays or manual chunking. The value of a `DatasetWriter` is enabling workflows where the complete dataset does *not* fit in memory. An in-memory collector does not address this.

2. **Zarr is already in xarray's ecosystem** — xarray has excellent Zarr support via `to_zarr()` with `append_dim` and `region` arguments. Zarr is designed for incremental writes and cloud-native workflows. It's a natural fit for CMOR4's xarray foundation.

3. **Avoid API churn** — Releasing an in-memory implementation first means users adopt an interface that later switches backends. If the initial approach has memory limitations, we'd need to document when to use it vs. when not to, creating confusion. Going directly to disk-backed staging delivers the full capability immediately.

4. **Zarr is not a heavy dependency** — Adding `zarr` as an optional dependency is lightweight. It's widely used in scientific Python and has minimal transitive dependencies that don't overlap with CMOR4's existing stack (xarray, numpy, netCDF4, cftime).

5. **Validation and path generation are already modular** — `create_dataset()` performs validation and constructs the xarray dataset; `write_netcdf()` writes the file. A Zarr staging layer can reuse the validation machinery without needing a transitional in-memory implementation.

**Potential concerns about going straight to Zarr:**

- **Dependency management**: Zarr would be an optional dependency. Users who don't need incremental writes don't need to install it. Those who do can `pip install cmor4[streaming]` or similar.
- **NetCDF4 finalization complexity**: Writing a Zarr store to NetCDF requires careful handling of chunking, compression, bounds variables, and fill values. However, this is well-trodden ground — xarray, h5netcdf, and netCDF4-python all have robust patterns for this.
- **Testing overhead**: We'll need tests for incremental writes, append mode, metadata validation across chunks, and NetCDF finalization. These tests would be needed for *any* incremental-write implementation.

**Conclusion**: Implement Zarr-backed disk-staging as the primary incremental-write API. This delivers the full capability immediately, avoids API churn, and aligns with the scientific Python ecosystem's direction toward cloud-native data formats.

## Architecture

### Validation Pipeline Refactoring

**Status:** ✅ **COMPLETED** - The validation pipeline refactoring (Phase 0) has been completed and merged.

The refactored architecture now has:
- **`utils/validation.py`**: Reusable validation functions and `ValidationContext` ✅
- **`utils/construction.py`**: Pure construction functions (no validation mixed in) ✅
- **Refactored `dataset.py`** (renamed from `core.py`): Orchestrates validation → construction → finalization ✅
- **Removed files**: `_axis_validation.py` and `_variable_validation.py` have been deleted and replaced by utils modules ✅

### Key Classes and Modules

Current CMOR4 architecture (from `src/cmor4/dataset.py`):

- **`create_dataset()`** (lines 50-385): Validates metadata, constructs axes, adds grid coords and zfactors, validates data, applies chunking, returns `xr.Dataset`
- **`write_netcdf()`** (lines 388-449): Renders output path, creates directories, calls `ds.to_netcdf()`
- **`cmorize()`** (lines 452-514): Convenience wrapper that calls `create_dataset()` + `write_netcdf()`

Dependencies:
- **`DatasetInfo`**: Dataset-level metadata (from `datasetinfo.py`)
- **`Variable`**: Variable metadata (from `variable.py`)  
- **`Axis`**: Coordinate axis metadata and values (from `axis.py`)
- **`ZFactor`**: Formula-term metadata (from `zfactor.py`)
- **`Grid`**: Grid mapping and spatial coordinate metadata (from `grid.py`)
- **`validate_metadata()`**: Metadata validation (from `utils/validation.py`)
- **`validate_data_chunk()`**: Data value validation (from `utils/validation.py`)
- **`validate_final_dataset()`**: Final dataset validation (from `utils/validation.py`)
- **`build_axis_mappings()`**: Axis construction (from `utils/construction.py`)
- **Time utilities**: `utils/time_utils.py` for time range computation, calendar handling
- **Chunking**: `utils/chunking.py` for CMIP7 auto-chunking and validation

### New Components

**`DatasetWriter`** (new class in `src/cmor4/writer.py`):

Main user-facing API for incremental writes:

```python
writer = cmor4.DatasetWriter(
    dataset=dataset,
    variable=variable,
    axes=axes,
    zfactors=zfactors,
    grid=grid,
    path=path,
    existing="error",  # "error", "replace", or "append"
    encoding=encoding,
    attrs=attrs,
)

writer.write(data, time_values=[...], time_bounds=[...], zfactors={"ps": ps_slice})
writer.write(data2, time_values=[...], time_bounds=[...], zfactors={"ps": ps_slice2})
result = writer.close(preserve_definition=False)
```

**Internal implementation using Zarr staging**:

The writer operates in three phases:

1. **Initialize** (`__init__`):
   - Validate non-time axes and metadata (same validation as `create_dataset()`)
   - Create a temporary Zarr store (in system temp or user-specified staging dir)
   - Initialize Zarr arrays for:
     - Main variable (with placeholder time dimension, size 0 initially)
     - Time coordinate and time bounds
     - Zfactor arrays (if provided)
     - Grid coordinate arrays (if grid is provided)
   - Store metadata document (JSON/YAML) with dimensions, attributes, encodings, write state

2. **Write** (`write()` method):
   - Validate chunk shape matches expected non-time dimensions
   - Validate time values are monotonic and contiguous with previous chunks
   - Validate data values (NaN, range checks)
   - Append data to Zarr arrays using `zarr.resize()` and `zarr[start:end] = chunk`
   - Update metadata document with new time range

3. **Close** (`close()` method):
   - Read final metadata from staging store
   - Stream chunks from Zarr store to final NetCDF file using netCDF4-python or h5netcdf
   - Apply compression, chunking, fill values, and CF attributes
   - Generate final output path from metadata and final time range
   - For `existing="append"`: open existing file with xarray, concatenate, validate metadata compatibility, write merged file
   - For `preserve_definition=True`: keep metadata object reusable, clear Zarr store for next segment
   - Clean up temporary Zarr store (unless preserve_definition=True and user wants to inspect)

## Implementation Details

### Zarr Store Layout

Temporary Zarr directory structure:

```
/tmp/cmor4-writer-{uuid}/
  ├── .zmetadata           # Zarr consolidated metadata
  ├── {var_name}/          # Main variable array
  │   ├── 0.0.0            # Zarr chunks
  │   ├── 0.0.1
  │   └── .zarray          # Array metadata (shape, dtype, chunks, compressor)
  ├── time/                # Time coordinate
  │   ├── 0
  │   └── .zarray
  ├── time_bnds/           # Time bounds
  │   ├── 0
  │   └── .zarray
  ├── ps/                  # Example zfactor
  │   ├── 0.0.0
  │   └── .zarray
  ├── latitude/            # Grid coords (if applicable)
  ├── longitude/
  └── _cmor4_metadata.json # CMOR4-specific metadata
```

The `_cmor4_metadata.json` file stores:
- Dataset global attributes
- Variable attributes  
- Dimension names and sizes
- NetCDF encoding parameters (compression, chunksizes, fill values)
- Write state (current time index, expected time units/calendar)

### Time Dimension Handling

The time dimension is the write dimension (concatenation axis). Each `write()` call:

1. Checks that `time_values` shape matches the incoming data's time dimension
2. Validates monotonicity: `time_values[0]` must be > last written time value (or configurable tolerance for gaps)
3. Resizes the time coordinate array: `time_coord.resize((current_size + len(time_values),))`
4. Appends values: `time_coord[current_size:current_size+len(time_values)] = time_values`
5. Does the same for time_bounds and the main variable

### Validation Strategy

**At initialization:**
- All `Axis` metadata validation (same as `create_dataset()`)
- Non-time axes must have complete values up front (no incremental spatial axes)
- Variable metadata validation
- Grid and zfactor metadata validation
- Encoding validation (CMIP7 chunking rules)

**At each write:**
- Chunk shape validation: non-time dimensions must match expected shape
- Time monotonicity and contiguity checks
- Data value validation (NaN, valid_min, valid_max per variable metadata)
- Zfactor data validation (if provided)

**At close:**
- Global attribute validation via `project.validate_global_attributes()`
- Final dataset structure validation via `project.validate_dataset()`
- Time range computation for filename/path rendering

### Append Mode Implementation

For `existing="append"`:

1. Check that target output file exists
2. Open with xarray: `existing_ds = xr.open_dataset(path)`
3. Validate compatibility:
   - Same variable name, dimensions, dtype
   - Same global attributes (except `history`, `tracking_id`, `creation_date`)
   - Same non-time axes (values and bounds)
   - Same grid mapping metadata
   - Same zfactor definitions
4. Stream new data from Zarr store into an xarray dataset
5. Concatenate: `merged_ds = xr.concat([existing_ds, new_ds], dim="time")`
6. Validate merged time coordinate is monotonic and contiguous
7. Write merged dataset to a temporary file
8. Atomically replace original file (or write to new version directory)

### Preserve Mode Implementation

For `preserve_definition=True`:

1. After close, do NOT delete the Zarr staging store metadata
2. Keep `DatasetWriter` object in a "reset" state where metadata is intact but data arrays are cleared
3. On next `write()`, resize arrays from size 0 again
4. Allows writing multiple output files (e.g., annual chunks) with the same metadata definition

### NetCDF Finalization

Converting Zarr store to NetCDF file using xarray:

```python
import xarray as xr

# Open Zarr store as xarray Dataset with lazy loading
ds = xr.open_zarr(staging_path)

# Apply encoding parameters from metadata
for var_name, encoding in variable_encodings.items():
    if var_name in ds:
        ds[var_name].encoding.update(encoding)

# Write to NetCDF - xarray handles streaming automatically
ds.to_netcdf(
    output_path,
    encoding={var_name: var_encoding for var_name, var_encoding in variable_encodings.items()},
)
```

This approach:
- Uses xarray's `open_zarr()` for lazy loading - data stays on disk
- Main variable and large arrays remain dask-backed and are streamed chunk-by-chunk
- Coordinate arrays (time, lat, lon) are loaded into memory but are typically small
- xarray's `to_netcdf()` is battle-tested and handles encoding, compression, and CF conventions correctly
- Memory overhead: ~20-70 MB for typical datasets (dominated by 2D curvilinear coordinate arrays if present)
- Complexity is much lower than direct netCDF4-python approach

### Error Handling

**Common error scenarios:**

1. **Shape mismatch**: User provides data with wrong non-time dimensions → raise `ValueError` with clear message
2. **Time gap**: User provides time values that don't follow previous chunk → warn or error based on tolerance parameter
3. **Metadata incompatibility** (append mode): Existing file doesn't match new metadata → raise `ValueError` with detailed diff
4. **Disk space**: Zarr write fails due to insufficient space → clean up partial writes, raise `OSError`
5. **NetCDF finalization failure**: Handle corrupted Zarr store, encoding errors → provide diagnostic info and preserve Zarr store for inspection

## File Structure

### Phase 0 (Validation Refactoring) - ✅ COMPLETED

Files created:
- ✅ **`src/cmor4/utils/__init__.py`**: Utils package initialization
- ✅ **`src/cmor4/utils/validation.py`**: Validation pipeline with `ValidationContext`, `validate_metadata()`, `validate_data_chunk()`, `validate_final_dataset()`
- ✅ **`src/cmor4/utils/construction.py`**: Pure construction functions - `build_axis_mappings()`, `build_grid_mappings()`, `build_zfactor_mappings()`
- ✅ **`tests/test_validation.py`**: Unit tests for validation functions
- ✅ **`tests/test_construction.py`**: Unit tests for construction functions

Files renamed/refactored:
- ✅ **`src/cmor4/core.py`** → **`src/cmor4/dataset.py`**: Renamed and refactored to use new validation/construction pipeline (internal changes only, API unchanged)

Files removed:
- ✅ **`src/cmor4/_axis_validation.py`**: Replaced by `utils/validation.py`
- ✅ **`src/cmor4/_variable_validation.py`**: Replaced by `utils/validation.py`

### Phases 1-4 (DatasetWriter Implementation) - READY TO START

New files:
- **`src/cmor4/writer.py`**: `DatasetWriter` class and staging/finalization logic
- **`src/cmor4/utils/writer_helpers.py`**: Helper functions for Zarr operations, chunk iteration, time validation
- **`tests/test_incremental_writes.py`**: Tests for `DatasetWriter` with single and multiple writes
- **`tests/test_append_mode.py`**: Tests for append functionality
- **`tests/test_large_datasets.py`**: Memory-bounded tests with Zarr staging

Modified files:
- **`src/cmor4/__init__.py`**: Export `DatasetWriter` and `ValidationContext`
- **`pyproject.toml`**: Add `zarr` as an optional dependency under `[project.optional-dependencies]` as `streaming = ["zarr>=2.16"]`
- **`README.md`**: Document incremental write workflow (separate section or new guide)
- **`docs/VALIDATION_REFACTOR_PLAN.md`**: Already created, documents validation pipeline refactoring

## API Design

### DatasetWriter Class

```python
class DatasetWriter:
    """Incremental dataset writer for time-series data.
    
    Supports writing data in chunks (typically by time slice), appending to
    existing files, and reusing metadata definitions across output segments.
    Uses Zarr for temporary staging to bound memory usage.
    
    Parameters
    ----------
    dataset : DatasetInfo
        Dataset-level metadata.
    variable : Variable
        Variable metadata.
    axes : Sequence[Axis]
        Coordinate axes. Time axis values can be incomplete; other axes must
        have complete values.
    path : str | Path | None
        Output path. If None, rendered from dataset/variable metadata at close time.
    zfactors : Sequence[ZFactor] | None
        Formula-term variables.
    grid : Grid | None
        Grid mapping and spatial coordinates.
    existing : Literal["error", "replace", "append"]
        How to handle existing output file. Default "error".
    encoding : Mapping[str, Any] | None
        Encoding parameters (chunksizes, compression, etc.).
    attrs : Mapping[str, Any] | None
        Extra global attributes.
    staging_dir : str | Path | None
        Directory for temporary Zarr store. If None, uses system temp directory.
    time_axis_name : str
        Name of the time axis. Default "time".
    allow_time_gaps : bool
        Whether to allow non-contiguous time values. Default False.
    """
    
    def __init__(
        self,
        dataset: DatasetInfo,
        variable: Variable,
        axes: Sequence[Axis],
        path: str | Path | None = None,
        *,
        zfactors: Sequence[ZFactor] | None = None,
        grid: Grid | None = None,
        existing: Literal["error", "replace", "append"] = "error",
        encoding: Mapping[str, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        staging_dir: str | Path | None = None,
        time_axis_name: str = "time",
        allow_time_gaps: bool = False,
    ) -> None:
        ...
    
    def write(
        self,
        data: Any,
        *,
        time_values: Any | None = None,
        time_bounds: Any | None = None,
        zfactors: Mapping[str, Any] | None = None,
    ) -> None:
        """Write a data chunk.
        
        Parameters
        ----------
        data : array-like
            Data values for this chunk. Shape must match (len(time_values), *spatial_shape).
        time_values : array-like | None
            Time coordinate values for this chunk. If None, must have been fully
            specified in the time axis at initialization.
        time_bounds : array-like | None
            Time bounds for this chunk. Shape must be (len(time_values), 2).
        zfactors : Mapping[str, array-like] | None
            Zfactor data for this chunk. Keys are zfactor names, values are arrays.
        """
        ...
    
    def close(
        self,
        *,
        preserve_definition: bool = False,
    ) -> Cmor4Result:
        """Finalize and write the output file.
        
        Parameters
        ----------
        preserve_definition : bool
            If True, keep metadata definition for writing another segment.
            Useful for splitting output across multiple files (e.g., annual chunks).
        
        Returns
        -------
        Cmor4Result
            The written dataset (loaded from output file) and output path.
        """
        ...
    
    def __enter__(self) -> DatasetWriter:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit. Calls close() unless an exception occurred."""
        if exc_type is None:
            self.close()
```

### Usage Examples

**Basic incremental write:**

```python
import cmor4
import numpy as np

project = cmor4.ProjectTables("CMIP7", data_tables_path)
dataset = project.dataset_info({...})
variable = project.variable("tas")
lat = project.axis("latitude", values=lats)
lon = project.axis("longitude", values=lons)
time = project.axis("time", values=[], units="days since 1850-01-01")

with cmor4.DatasetWriter(
    dataset=dataset,
    variable=variable,
    axes=[time, lat, lon],
) as writer:
    for year in range(1850, 2015):
        time_vals, data = load_year(year)  # User function
        writer.write(data, time_values=time_vals, time_bounds=compute_bounds(time_vals))
    
    result = writer.close()
    print(f"Wrote {result.path}")
```

**Append to existing file:**

```python
# Extend an existing dataset with new time records
with cmor4.DatasetWriter(
    dataset=dataset,
    variable=variable,
    axes=[time, lat, lon],
    path="/path/to/existing.nc",
    existing="append",
) as writer:
    time_vals, data = load_new_data()
    writer.write(data, time_values=time_vals, time_bounds=compute_bounds(time_vals))
```

**Preserve definition for multiple files:**

```python
# Write annual files with same metadata
writer = cmor4.DatasetWriter(dataset, variable, axes=[time, lat, lon])

for year in [2010, 2011, 2012]:
    time_vals, data = load_year(year)
    writer.write(data, time_values=time_vals)
    result = writer.close(preserve_definition=True)
    print(f"Wrote {result.path}")
```

## Testing Strategy

### Unit Tests

1. **Basic write** (`tests/test_incremental_writes.py`):
   - Write single chunk, verify output
   - Write multiple chunks, verify concatenation
   - Verify time range in filename

2. **Validation** (`tests/test_incremental_writes.py`):
   - Shape mismatch errors
   - Time monotonicity errors
   - Data value range violations

3. **Append mode** (`tests/test_append_mode.py`):
   - Append compatible data to existing file
   - Detect metadata incompatibilities
   - Verify merged time coordinate

4. **Preserve mode** (`tests/test_incremental_writes.py`):
   - Write multiple files with same writer
   - Verify metadata consistency across files

5. **Encoding** (`tests/test_incremental_writes.py`):
   - Verify chunking applied correctly
   - Verify compression parameters
   - Verify CMIP7 auto-chunking

6. **Zfactors** (`tests/test_incremental_writes.py`):
   - Write with per-chunk zfactor data
   - Verify formula_terms in output

### Integration Tests

1. **Large dataset** (`tests/test_large_datasets.py`):
   - Write dataset larger than available memory (mock with chunked writes)
   - Verify memory usage stays bounded

2. **CMIP7 example** (`tests/test_cmip7_examples.py`):
   - Rewrite an existing CMIP7 example using `DatasetWriter`
   - Verify output matches single-pass `cmorize()` result

3. **Error recovery** (`tests/test_incremental_writes.py`):
   - Simulate disk full during write
   - Verify cleanup of staging store

### Performance Tests

Optional benchmarks:

1. Compare `DatasetWriter` vs. `cmorize()` for equivalent dataset
2. Measure memory usage for incremental vs. whole-dataset workflows
3. Measure Zarr→NetCDF finalization overhead

## Migration Path

### Backward Compatibility

`DatasetWriter` is a new API. Existing `cmorize()` and `create_dataset()` workflows are unchanged.

### Deprecation

No deprecations. `cmorize()` remains the recommended API for whole-dataset workflows.

### Documentation

Update user guide with new "Incremental Writes" section covering:

- When to use `DatasetWriter` vs. `cmorize()`
- Example workflows (basic, append, preserve)
- Memory considerations
- Zarr dependency installation

## Design Decisions

1. **Time axis identification**: Auto-detect by checking for `axis="T"` attribute or name matching "time*", with `time_axis_name` as override for edge cases. This covers 99% of use cases without requiring extra parameters.

2. **Partial zfactor writes**: Require zfactor data in every `write()` call if the zfactor dimensions include time. Raise error if omitted. This prevents silent correctness issues.

3. **Staging directory cleanup**: Auto-delete on success, keep on error. Add optional `keep_staging=True` parameter for debugging. Prevents temp directory clutter while preserving debugging capability.

4. **NetCDF finalization**: Use xarray's `open_zarr()` + `to_netcdf()`. The memory overhead is minimal (20-70 MB) for real-world datasets, and xarray's finalization logic is battle-tested and handles edge cases correctly. The complexity reduction outweighs the small memory cost.

5. **Concurrent writes**: No. Each writer gets a unique staging directory (UUID-based). Concurrent writes to the *same logical dataset* are not supported in initial implementation.

## Implementation Order

### Phase 0: Validation Pipeline Refactoring - ✅ COMPLETED AND MERGED

**Status:** All Phase 0 work has been completed and merged into the main branch.

**Completed deliverables:**
- ✅ `src/cmor4/utils/validation.py` with reusable validation functions
- ✅ `src/cmor4/utils/construction.py` with reusable construction functions
- ✅ Renamed and refactored `src/cmor4/dataset.py` (formerly `core.py`) using new functions
- ✅ Unit tests for validation and construction functions
- ✅ Regression tests confirming no behavior changes
- ✅ All existing tests pass without modification
- ✅ No changes to public API
- ✅ `_axis_validation.py` and `_variable_validation.py` removed and replaced

---

### Phase 1: Core DatasetWriter Implementation

**1.1. Core `DatasetWriter` with Zarr staging** (`src/cmor4/writer.py`):
- `__init__`: Use `validate_metadata()` from Phase 0, create Zarr store
- `write()`: Use `validate_data_chunk()` from Phase 0, append to Zarr
- `close()`: Load from Zarr, use `validate_final_dataset()`, write NetCDF via xarray

**1.2. Basic tests** (`tests/test_incremental_writes.py`):
- Single write, multiple writes, shape validation, time monotonicity

**Estimated effort:** 3-4 days

---

### Phase 2: Append Mode

**2.1. Extend `DatasetWriter.close()`**:
- File compatibility checking, concatenation, atomic replacement

**2.2. Append mode tests** (`tests/test_append_mode.py`):
- Append compatible data, detect incompatibilities, verify merged time coordinate

**Estimated effort:** 2-3 days

---

### Phase 3: Preserve Mode and Zfactors

**3.1. Preserve mode** (extend `DatasetWriter.close()` and internal state):
- Metadata retention, Zarr store reset for multiple output files

**3.2. Zfactor support** (extend `DatasetWriter.write()`):
- Per-chunk zfactor data, validation using Phase 0 functions

**3.3. Tests** (extend `tests/test_incremental_writes.py`):
- Write multiple files with same writer
- Write with per-chunk zfactor data
- Verify formula_terms in output

**Estimated effort:** 2-3 days

---

### Phase 4: Integration and Documentation

**4.1. Integration tests** (`tests/test_cmip7_examples.py`, `tests/test_large_datasets.py`):
- Rewrite CMIP7 example using DatasetWriter, verify output matches
- Test memory-bounded large dataset writes

**4.2. Encoding tests** (extend `tests/test_incremental_writes.py`):
- Verify chunking, compression, CMIP7 auto-chunking

**4.3. Documentation**:
- User guide section on incremental writes
- API reference for DatasetWriter
- Usage examples (basic, append, preserve)
- Update README with installation instructions for zarr

**Estimated effort:** 2-3 days

---

### Total Estimated Effort

- **Phase 0 (Validation refactoring):** ~~4-6 days~~ ✅ COMPLETED
- **Phase 1 (Core DatasetWriter):** 3-4 days
- **Phase 2 (Append mode):** 2-3 days
- **Phase 3 (Preserve/Zfactors):** 2-3 days
- **Phase 4 (Integration/Docs):** 2-3 days

**Total remaining:** 10-13 days (Phase 0 complete)

**Benefits achieved from Phase 0:**
- ✅ Eliminated ~400 lines of code duplication
- ✅ Ensures validation consistency between APIs
- ✅ Created reusable validation infrastructure
- ✅ Makes both `cmorize()` and `DatasetWriter` more maintainable
- ✅ Provides better testability and clearer control flow

## Success Criteria

### Phase 0 (Validation Refactoring) - ✅ COMPLETED
1. ✅ All existing tests pass without modification after refactoring
2. ✅ `create_dataset()` output is byte-for-byte identical before and after refactoring
3. ✅ No performance regression (within 5% of baseline)
4. ✅ Validation functions have ≥ 90% test coverage
5. ✅ `ValidationContext` correctly tracks all validation state

### Phases 1-4 (DatasetWriter Implementation) - IN PROGRESS
1. Users can write CMIP7 datasets incrementally without loading full time series into memory
2. `DatasetWriter` output is byte-for-byte identical (or equivalent) to `cmorize()` output for the same data
3. `DatasetWriter` and `cmorize()` use identical validation logic (via shared functions)
4. Append mode successfully extends existing CMOR4-generated files
5. Memory usage remains bounded regardless of total dataset size
6. All validation rules (axis, variable, global attributes, CMIP7 chunking) apply consistently
7. Test coverage ≥ 90% for new code
8. Documentation includes clear usage examples and migration guidance

## Out of Scope (Future Work)

- **Companion variables**: Support for writing associated variables (e.g., `ps` with `ta`) into the same file with explicit companion-variable write operations. This will be addressed in a separate feature after `DatasetWriter` is stable.
- **Enhanced Grid mapping/CRS**: Advanced CRS parameter normalization and projection validation beyond current `Grid` capabilities. Orthogonal to incremental writes.
- **Forecast leadtime derivation**: Automatic leadtime coordinate computation from reference time. Orthogonal to incremental writes.
- **Parallel writes**: Support for multiple concurrent `DatasetWriter` instances writing different variables to the same output directory.
- **Dask integration**: Using dask for distributed Zarr writes. Initial implementation is single-threaded.
- **Cloud-native output**: Direct write to S3/GCS without local staging. Future optimization.

## Verification

After implementation:

1. Run existing CMOR4 test suite — all tests must pass (no regressions)
2. Run new `DatasetWriter` tests — all must pass
3. Rewrite a subset of CMIP7 examples using `DatasetWriter` — output must match `cmorize()` results
4. Manually test append mode with real CMIP7 output from a climate model
5. Profile memory usage for incremental write of 10-year daily dataset — verify it stays under threshold (e.g., 1GB regardless of dataset size)
6. Review documentation for clarity and completeness
