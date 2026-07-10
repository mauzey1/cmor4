# Validation Pipeline Refactoring Plan

## Status: ✅ COMPLETED AND MERGED

This document describes the validation pipeline refactoring that has been completed and merged into CMOR4.

## Summary

This refactoring transformed CMOR4's validation logic from scattered, monolithic code into a clean, reusable pipeline. The completed refactoring:

1. ✅ **Extracted validation** into `src/cmor4/utils/validation.py`
2. ✅ **Extracted construction** into `src/cmor4/utils/construction.py`
3. ✅ **Renamed `core.py`** to `dataset.py` and refactored it to use new utils
4. ✅ **Removed underscore files** - Deleted `_axis_validation.py` and `_variable_validation.py` (fully replaced by utils)
5. ✅ **Enabled code reuse** - Both `cmorize()` and `DatasetWriter` share the same validation logic

**Implementation:** This was a **clean replacement**, not a wrapper approach. The underscore-prefixed files were deleted and all imports updated to use the new `utils/` modules.

## Current State Analysis

### Validation in `create_dataset()` (lines 50-385 in `dataset.py`, formerly `core.py`)

The current `create_dataset()` function performs validation in several stages:

**Phase 1: Metadata Preparation (lines 119-121)**
```python
dataset, variable = _dataset_for_variable(dataset, variable)
axes = _dataset_axes(dataset, axes, variable)
axes = validate_and_normalize_axes(dataset, variable, axes)
```
- Applies project-specific metadata transformations (CMIP7 remappings)
- Validates axis metadata against coordinate tables
- Normalizes axis values (monotonicity, bounds, units)

**Phase 2: Dataset Construction (lines 123-235)**
- Merges grid axes into axis list
- Builds coords/data_vars dictionaries via `_add_axis()` for each axis
- Adds grid coordinates via `_add_grid_coords()`
- Adds grid mapping variable
- Adds zfactors via `_add_zfactor()` (includes per-zfactor validation)
- Validates data shape against expected dimensions
- Validates data values via `validate_variable_values()` (NaN, range checks)
- Computes external_variables from cell_measures
- Constructs xarray.Dataset

**Phase 3: Encoding and Chunking (lines 241-372)**
- Handles missing_value and _FillValue
- Calculates or validates CMIP7 chunking
- Applies user-provided encoding to all variables
- Enforces CMIP7 single-chunk rule for time coordinates

**Phase 4: Final Validation (lines 374-383)**
```python
_validate_final_components(ds, dataset, variable, axes, zfactors, grid, dims, zfactor_names)
```
- Validates global attributes via `project.validate_global_attributes()`
- Validates dataset structure via `project.validate_dataset()`
- Verifies all expected variables/coordinates were created
- Verifies grid mapping references
- Verifies zfactor bounds

## Validation Overlap with DatasetWriter

### What DatasetWriter MUST Reuse (100% overlap)

**At initialization:**
1. ✅ `_dataset_for_variable()` - Project-specific metadata transformations
2. ✅ `_dataset_axes()` - Add required scalar axes
3. ✅ `validate_and_normalize_axes()` - Axis metadata validation
4. ✅ Grid axis merging logic
5. ✅ Dimension resolution (`_named_dimensions()`, axis_dims mapping)
6. ✅ Zfactor metadata validation
7. ✅ Encoding validation (CMIP7 chunking rules)

**At each write():**
1. ✅ Data shape validation against expected dimensions
2. ✅ `validate_variable_values()` - NaN and range checking
3. ✅ Time monotonicity validation (currently in `_axis_validation.py`)
4. ✅ Zfactor value validation (per-chunk)

**At close():**
1. ✅ `_validate_final_components()` - Final dataset structure checks
2. ✅ `project.validate_global_attributes()`
3. ✅ `project.validate_dataset()`
4. ✅ External variables computation

### What DatasetWriter Does Differently

**Incremental time handling:**
- Time axis can be incomplete at initialization
- Time values/bounds provided per write() call
- Must track cumulative time state across writes
- Must validate time contiguity between chunks

**Zarr-specific operations:**
- Initialize Zarr arrays with resizable time dimension
- Append chunks to Zarr store
- Track write state in metadata document

**Append mode:**
- Open and validate existing NetCDF file
- Compare metadata for compatibility
- Merge time coordinates

## Current Issues with Validation in `dataset.py`

### 1. **Monolithic Function**
`create_dataset()` is 335 lines doing 7 different things:
- Metadata preparation
- Axis processing
- Grid processing  
- Zfactor processing
- Data validation
- Encoding configuration
- Final validation

This makes it hard to:
- Test individual validation steps
- Reuse validation in different contexts (like DatasetWriter)
- Understand what validation happens when

### 2. **Mixed Concerns**
The function interleaves:
- **Validation** (check metadata, check data values)
- **Construction** (build coords/data_vars dictionaries)
- **Transformation** (apply encodings, compute attributes)

### 3. **Hidden Dependencies**
Many helper functions depend on side effects:
- `_add_axis()` mutates 5 different dictionaries
- `_add_grid_coords()` mutates 3 different collections
- `_add_zfactor()` validates but also constructs data_vars entries

### 4. **Validation Scattered Across Multiple Files**
- Axis validation: `_axis_validation.py`
- Variable validation: `_variable_validation.py`
- Final validation: `dataset.py` (`_validate_final_components()`)
- Project validation: `tables.py` (ProjectTables methods)
- Grid validation: implicit in `grid.py`
- Zfactor validation: embedded in `_add_zfactor()`

## Proposed Refactoring

### Goals
1. **Separate validation from construction** - Make validation steps explicit and reusable
2. **Create a validation pipeline** - Clear stages that can be tested independently
3. **Enable DatasetWriter reuse** - Share validation logic without duplicating code
4. **Maintain backward compatibility** - Don't break existing `cmorize()` API
5. **Clean up internal structure** - Replace underscore-prefixed files with organized `utils/` directory

### New Structure

#### 1. **Create `src/cmor4/utils/validation.py`** - Unified validation orchestrator

```python
from dataclasses import dataclass
from typing import Any, Sequence

@dataclass
class ValidationContext:
    """Shared validation state and results."""
    dataset: DatasetInfo
    variable: Variable
    axes: Sequence[Axis]
    zfactors: Sequence[ZFactor] | None
    grid: Grid | None
    
    # Computed during validation
    axis_dims: dict[str, tuple[str, ...]]
    spatial_dims: tuple[str, ...]
    time_axis: Axis | None
    scalar_axes: list[Axis]
    auxiliary_axes: list[Axis]
    
    # Validation results
    warnings: list[str]
    errors: list[str]


def validate_metadata(
    dataset: DatasetInfo,
    variable: Variable,
    axes: Sequence[Axis],
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
) -> ValidationContext:
    """Validate all metadata before any data processing.
    
    This performs all metadata validation that doesn't require data values:
    - Project-specific transformations
    - Axis metadata validation
    - Grid metadata validation  
    - Zfactor metadata validation
    - Dimension compatibility checks
    - Encoding validation
    
    Returns ValidationContext with computed dimension mappings and validation results.
    """
    ...


def validate_data_chunk(
    ctx: ValidationContext,
    data: Any,
    time_values: Any | None = None,
    time_bounds: Any | None = None,
    zfactors: dict[str, Any] | None = None,
) -> None:
    """Validate a data chunk against metadata.
    
    - Data shape validation
    - Data value validation (NaN, range)
    - Time monotonicity (if applicable)
    - Zfactor value validation
    
    Raises ValueError if validation fails.
    """
    ...


def validate_final_dataset(
    ds: xr.Dataset,
    ctx: ValidationContext,
) -> None:
    """Validate the final constructed dataset.
    
    - Global attributes validation
    - Dataset structure validation
    - Variable/coordinate presence checks
    - Grid mapping references
    - Formula terms
    
    Raises ValidationError if final dataset is invalid.
    """
    ...
```

#### 2. **Create `src/cmor4/utils/construction.py`** - Dataset construction helpers

Move construction logic out of validation:

```python
def build_axis_mappings(
    axes: Sequence[Axis],
    grid: Grid | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, tuple[str, ...]]]:
    """Build coords, data_vars, and axis_dims from axes and grid.
    
    Pure function that doesn't perform validation (validation happens first).
    """
    coords: dict[str, Any] = {}
    data_vars: dict[str, Any] = {}
    axis_dims: dict[str, tuple[str, ...]] = {}
    
    for axis in axes:
        _add_axis_to_mappings(axis, coords, data_vars, axis_dims)
    
    if grid is not None:
        _add_grid_to_mappings(grid, coords, data_vars)
    
    return coords, data_vars, axis_dims


def build_dataset_from_mappings(
    variable: Variable,
    data: Any,
    coords: dict[str, Any],
    data_vars: dict[str, Any],
    global_attrs: dict[str, Any],
    encoding: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Construct xarray.Dataset from validated components.
    
    Assumes all validation has already been performed.
    """
    ...
```

#### 3. **Refactor `core.py`**

```python
def create_dataset(
    dataset: DatasetInfo,
    variable: Variable,
    axes: Sequence[Axis],
    data: Any,
    *,
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
    attrs: Mapping[str, Any] | None = None,
    encoding: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Create an xarray dataset from metadata objects.
    
    Simplified to orchestrate validation and construction steps.
    """
    # Phase 1: Validate metadata
    ctx = validate_metadata(dataset, variable, axes, zfactors, grid)
    
    # Phase 2: Validate data
    validate_data_chunk(ctx, data)
    
    # Phase 3: Build dataset
    coords, data_vars, axis_dims = build_axis_mappings(ctx.axes, grid)
    # ... add zfactors, compute attributes, apply encoding ...
    ds = build_dataset_from_mappings(variable, data, coords, data_vars, attrs, encoding)
    
    # Phase 4: Final validation
    validate_final_dataset(ds, ctx)
    
    return ds
```

#### 4. **DatasetWriter implementation** (`writer.py`)

```python
class DatasetWriter:
    def __init__(self, dataset, variable, axes, ...):
        # Phase 1: Validate metadata (shared with create_dataset)
        self._ctx = validate_metadata(dataset, variable, axes, zfactors, grid)
        
        # Phase 2: Initialize Zarr store
        self._init_zarr_store()
        
        self._chunks_written = 0
        self._last_time_value = None
    
    def write(self, data, time_values=None, time_bounds=None, zfactors=None):
        # Validate chunk (shared with create_dataset)
        validate_data_chunk(
            self._ctx,
            data,
            time_values,
            time_bounds,
            zfactors,
        )
        
        # Zarr-specific: append to store
        self._append_to_zarr(data, time_values, time_bounds, zfactors)
        
        self._chunks_written += 1
        self._last_time_value = time_values[-1] if time_values is not None else None
    
    def close(self):
        # Load from Zarr as xarray Dataset
        ds = xr.open_zarr(self._staging_path)
        
        # Apply encoding
        ds = self._apply_encoding(ds)
        
        # Final validation (shared with create_dataset)
        validate_final_dataset(ds, self._ctx)
        
        # Write to NetCDF
        return self._finalize_netcdf(ds)
```

## Code Sharing Breakdown

### Functions to Extract/Refactor

| Current Location | Function | New Location | Used By |
|-----------------|----------|--------------|---------|
| `dataset.py:1010-1024` | `_dataset_for_variable()` | `utils/validation.py` | Both |
| `dataset.py:1027-1035` | `_dataset_axes()` | `utils/validation.py` | Both |
| `_axis_validation.py:53` | `validate_and_normalize_axes()` | `utils/validation.py` | Both |
| `dataset.py:123-134` | Grid axis merging | `utils/validation.py` | Both |
| `_variable_validation.py:14` | `validate_variable_values()` | `utils/validation.py` | Both |
| `dataset.py:374-383` | `_validate_final_components()` | `utils/validation.py` | Both |
| `dataset.py:727-799` | `_add_axis()` | `utils/construction.py` | Both |
| `dataset.py:688-724` | `_add_grid_coords()` | `utils/construction.py` | Both |
| `dataset.py:801-849` | `_add_zfactor()` | `utils/construction.py` | Both |
| `dataset.py:245-295` | Encoding/chunking logic | `utils/encoding.py` | Both |

### Estimated Overlap

**Code that DatasetWriter must duplicate if we don't refactor:** ~60%
- All validation logic
- Dimension mapping logic  
- Encoding validation
- Final dataset validation

**Code that can remain in create_dataset():** ~40%
- Single-pass xarray.Dataset construction
- Immediate NetCDF writing path

## Benefits of Refactoring

### For DatasetWriter Implementation
1. **Less code duplication** - Reuse ~400 lines of validation logic
2. **Consistent validation** - Same rules for both APIs
3. **Easier testing** - Test validation once, use in both contexts
4. **Clearer separation** - DatasetWriter focuses on Zarr mechanics, not validation

### For Existing Code
1. **Better testability** - Each validation step can be tested independently
2. **Clearer control flow** - Explicit validation → construction → finalization pipeline
3. **Easier debugging** - Validation failures have clearer provenance
4. **Better documentation** - Each validation function documents what it checks

### For Future Features
1. **Reusable validation** - New APIs (REST server, CLI tools) can use same validation
2. **Incremental validation** - Can validate metadata before data is available
3. **Custom workflows** - Advanced users can call validation steps directly

## Implementation Status

### Phase 1: Extract Validation and Replace Underscore Files - ✅ COMPLETED
1. ✅ Created `src/cmor4/utils/` directory with `__init__.py`
2. ✅ Created `src/cmor4/utils/validation.py` with `ValidationContext` class
3. ✅ Extracted validation functions from `dataset.py`, `_axis_validation.py`, `_variable_validation.py`
4. ✅ Created `validate_metadata()`, `validate_data_chunk()`, `validate_final_dataset()`
5. ✅ Updated all imports across the codebase to use `from cmor4.utils.validation import ...`
6. ✅ Deleted `_axis_validation.py` and `_variable_validation.py` (fully replaced)
7. ✅ Added tests for new validation functions

### Phase 2: Extract Construction Functions - ✅ COMPLETED
1. ✅ Created `src/cmor4/utils/construction.py`
2. ✅ Extracted `_add_axis()`, `_add_grid_coords()`, `_add_zfactor()` into pure functions
3. ✅ Updated `dataset.py` to use new construction functions directly
4. ✅ Removed old helper functions from `dataset.py` (replaced by utils functions)
5. ✅ Added tests for construction functions

### Phase 3: Rename and Refactor dataset.py (Internal Only) - ✅ COMPLETED
1. ✅ Renamed `src/cmor4/core.py` to `src/cmor4/dataset.py`
2. ✅ Rewrote `create_dataset()` to use new validation and construction pipelines
3. ✅ Maintained exact same API and behavior
4. ✅ Ran full test suite to verify no regressions

### Phase 4: Implement DatasetWriter (New Feature) - ✅ PHASES 1-2 COMPLETED
1. ✅ Implemented `DatasetWriter` using refactored validation/construction in `src/cmor4/writer.py`
2. ✅ Added DatasetWriter tests in `tests/test_incremental_writes.py` and `tests/test_datasetwriter_expanded.py`
3. ✅ Verified validation consistency between `cmorize()` and `DatasetWriter` (both use same validation functions)
4. ✅ Phase 2 (Append mode) completed - 813 lines of tests, full metadata compatibility validation
5. ⏳ Phase 3 (Per-chunk zfactor writes) ready to start
6. ⏳ Phase 4 (Integration tests and documentation) pending

### Phase 5: Cleanup (Optional) - PENDING
1. Remove wrapper functions if they're no longer needed
2. Update documentation to reflect new internal structure
3. Add developer guide explaining validation pipeline

## Testing Strategy

### Validation Tests
- Test each validation function independently
- Test ValidationContext state tracking
- Test error messages and warnings
- Test edge cases (empty arrays, scalar axes, etc.)

### Regression Tests  
- Run full existing test suite after each phase
- Verify `create_dataset()` behavior unchanged
- Verify output files byte-for-byte identical

### Integration Tests
- Test that DatasetWriter and cmorize() produce identical output for same inputs
- Test that validation errors are consistent between APIs

## Risks and Mitigation

### Risk: Breaking Changes from File Removal
**Mitigation:** 
- Carefully audit all imports in the codebase before deleting underscore files
- Update imports systematically (use IDE refactoring tools or grep)
- Run full test suite after import updates to catch any missed imports
- The files being removed (`_axis_validation.py`, `_variable_validation.py`) are internal modules, not part of public API

### Risk: Performance Regression
**Mitigation:** Profile before/after, ensure no extra copies or allocations

### Risk: Code Churn
**Mitigation:** Phase the work, each phase is independently useful

### Risk: Validation Divergence
**Mitigation:** Share code from day 1, add cross-validation tests

## Effort and Results

### Completed Effort
- **Phase 1 (Extract Validation):** ✅ Completed
- **Phase 2 (Extract Construction):** ✅ Completed
- **Phase 3 (Rename and refactor dataset.py):** ✅ Completed
- **Phase 4 (Implement DatasetWriter):** ✅ Phases 1-2 completed (~6-7 days), Phase 3 ready to start
- **Phase 5 (Cleanup):** To be determined

### Achieved Benefits
1. ✅ Eliminated ~400 lines of code duplication
2. ✅ Makes both APIs more maintainable
3. ✅ Created reusable validation infrastructure for future features
4. ✅ Reduced risk of validation bugs/inconsistencies
5. ✅ Makes the codebase easier to understand
6. ✅ No breaking changes - all existing tests pass
7. ✅ Public API unchanged - backward compatible

## Results

**The refactoring was successfully completed with no breaking changes and provides a solid foundation for implementing DatasetWriter.**

The refactored architecture is now in place with:
- Clean separation of validation and construction logic
- Reusable `ValidationContext` for tracking validation state
- Modular validation functions that can be used by any component
- Pure construction functions with no side effects
- Improved testability and maintainability

**Update (Phases 1-2 Complete):** The refactoring has proven highly successful. `DatasetWriter` Phases 1-2 implementation reuses the validation infrastructure seamlessly:
- ✅ Both `DatasetWriter` and `cmorize()` call the same `validate_metadata()` function
- ✅ Both use `validate_data_chunk()` for data validation
- ✅ Both use `validate_final_dataset()` for final checks
- ✅ Zero code duplication in validation logic
- ✅ Consistent behavior and error messages across both APIs
- ✅ 1269 lines of `DatasetWriter` implementation with 1838 lines of tests achieving ≥90% coverage
- ✅ Append mode fully functional with comprehensive compatibility validation
