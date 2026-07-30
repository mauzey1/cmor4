# Implementation Plan: Enhanced Grid Mapping and Forecast Leadtime Support

## Context

CMOR4 needs to add two features from CMOR3:

**Gap 10: Enhanced Grid Mapping/CRS Support** — CMOR3's `set_crs()` validates projection parameters, normalizes paired parameters (e.g., `standard_parallel1` and `standard_parallel2` into CF `standard_parallel`), validates required axes, and supports text CRS attributes. CMOR4's current `Grid` class writes scalar mapping variables but doesn't implement CMOR3's parameter normalization or richer validation.

**Gap 11: Forecast Reference Time and Leadtime** — CMOR3 tests show automatic `leadtime` derivation from `time - reftime`. CMOR4 has generic axis support but no forecast-specific coordinate logic.

We're skipping Gap 7 (companion variables/`store_with`) as current zfactor writes are sufficient.

## Current State

### Grid Implementation (src/cmor4/grid.py)
- `Grid` class stores axes, dimensions, mapping_name, params, attrs, lat/lon arrays
- `mapping_attributes()` writes params to NetCDF attributes, handling `(value, units)` tuples
- `_valid_param()` validates latitude/longitude/nonneg parameters with range checks
- Hard-coded parameter sets: `_LATITUDE_PARAMS`, `_LONGITUDE_PARAMS`, `_NONNEG_PARAMS`
- No parameter normalization (paired standard_parallel1/2 written as-is)
- No grid-table-driven validation of required/optional parameters

### Grid Table Support (src/cmor4/utils/tables.py)
- `GridTable` loads axis_entry, variable_entry (coords), mapping_entry sections
- `GridMappingEntry` wraps raw JSON entry
- `GridTable.build()` merges table defaults into user data
- `ProjectTables.grid()` resolves mapping entry and validates axes against table

### Time Utilities (src/cmor4/utils/time_utils.py)
- `decode_time_value()` converts numeric time to datetime using cftime
- `cftime_interval_days()` computes elapsed days between time values
- `_elapsed_seconds()` handles calendar-aware time differences
- `normalize_time_units()` standardizes "X since YYYY-MM-DD" format
- `add_time_delta()` adds timedelta to cftime/datetime objects

### Axis and Coordinate Construction (src/cmor4/axis.py, utils/construction.py)
- `Axis` class stores values, bounds, dimensions, units, standard_name, etc.
- `build_axis_mappings()` converts Axis objects to xarray coords/data_vars
- `add_axis()` handles scalar, auxiliary, and dimension coordinates
- No automatic coordinate derivation logic (all coordinates explicit)

## Recommended Approach

### Phase 1: Grid Mapping Parameter Normalization (Gap 10)

**Goal:** Normalize paired projection parameters to CF-native form and add validation warnings.

**Files to modify:**
- `src/cmor4/grid.py` — add normalization in `mapping_attributes()`
- `src/cmor4/utils/tables.py` — add `GridMappingEntry` helper methods

**Implementation:**

1. Add parameter normalization to `Grid.mapping_attributes()`:
   - Detect `standard_parallel1` and `standard_parallel2` in `self.params`
   - Convert to single `standard_parallel` array: `[val1, val2]`
   - Handle `(value, units)` tuple format for each
   - Remove `*_units` attributes for numeric projection parameters (CF doesn't use them)
   - Preserve text CRS attributes from `self.attrs` (crs_wkt, GeoTransform, etc.)

2. Add `GridMappingEntry` validation helpers:
   - `required_params()` — extract from entry JSON
   - `optional_params()` — extract from entry JSON
   - `text_params()` — parameters that should be strings
   - `required_axes()` — expected projection axis types

3. Enhance `ProjectTables.grid()` to warn about missing parameters:
   - If grid table entry exists, check user params against required_params
   - Warn (don't fail) for missing recommended parameters
   - Validate axis count/order against required_axes

**Example output change:**
```python
# User provides
params={"standard_parallel1": (30.0, "degrees_north"), 
        "standard_parallel2": (60.0, "degrees_north")}

# CMOR4 writes to NetCDF
grid_mapping_name = "lambert_conformal_conic"
standard_parallel = [30.0, 60.0]  # normalized CF array
# No standard_parallel1_units, standard_parallel2_units
```

### Phase 2: Grid Table Validation Enhancement (Gap 10)

**Goal:** Validate grid mappings against grid table definitions.

**Files to modify:**
- `src/cmor4/utils/validation.py` — add grid validation
- `src/cmor4/tables.py` — enhance `ProjectTables.grid()` validation

**Implementation:**

1. Add `validate_grid_mapping()` to utils/validation.py:
   - Check mapping_name matches entry
   - Validate all user params are allowed by the projection
   - Check required params are present (warn if missing)
   - Validate parameter value types (numeric vs text)
   - Check axes match projection requirements (e.g., X/Y for projected grids)

2. Call from `ProjectTables.grid()` after resolving table entry
   - Only validate when grid table entry exists
   - Fail for unknown projection parameters
   - Warn for missing recommended parameters

### Phase 3: Forecast Leadtime Derivation (Gap 11)

**Goal:** Automatically derive `leadtime` coordinate from `time - reftime`.

**Files to modify:**
- `src/cmor4/utils/construction.py` — add forecast coordinate helper
- `src/cmor4/dataset.py` — integrate into `create_dataset()`

**Implementation:**

1. Add `derive_forecast_coords()` to utils/construction.py:
   ```python
   def derive_forecast_coords(
       axes: Sequence[Axis],
       coords: dict[str, Any],
       coord_table: Any = None,
   ) -> None:
       """Add derived leadtime coordinate if reftime and time exist."""
   ```
   - Search axes for time axis (axis="T")
   - Search axes for reftime axis (standard_name="forecast_reference_time" or name="reftime*")
   - If both found and reftime is scalar or size==1:
     - Extract reftime value
     - Compute leadtime = time_values - reftime
     - Use `_elapsed_seconds()` from time_utils for calendar-aware math
     - Create leadtime coordinate with units matching time axis
     - Look up leadtime metadata from coord_table if available
     - Add to coords dict as auxiliary coordinate

2. Call from `create_dataset()` after `build_axis_mappings()`:
   ```python
   derive_forecast_coords(axes, coords, getattr(dataset, "project", None))
   ```

3. Validation:
   - If user explicitly provides leadtime axis, validate it matches derived values
   - Fail if mismatch exceeds small tolerance (1 second)
   - Skip derivation if no reftime or time axis found

**Example:**
```python
time_axis = Axis(name="time", values=[0, 6, 12, 18], units="hours since 2020-01-01", axis="T")
reftime_axis = Axis(name="reftime", values=[0], units="hours since 2020-01-01", 
                    standard_name="forecast_reference_time", scalar=True)

# Automatically creates:
# leadtime coordinate with values [0, 6, 12, 18] and units "hours"
# standard_name="forecast_period" (from coord table if available)
```

### Phase 4: Testing

**New test files:**
- `tests/test_grid_mapping_normalization.py` — test standard_parallel normalization, text attrs
- `tests/test_grid_validation.py` — test required param warnings, unknown param errors
- `tests/test_forecast_leadtime.py` — test automatic leadtime derivation, validation

**Test cases:**

Grid normalization:
- paired standard_parallel1/2 → single array
- text params (crs_wkt, GeoTransform) preserved
- units handling in tuple format
- CF-compliant output (no *_units attrs for numeric params)

Grid validation:
- missing required params → warning
- unknown params → error
- wrong axis types → error
- valid grid passes silently

Forecast leadtime:
- scalar reftime + time → leadtime derived
- size-1 reftime + time → leadtime derived
- no reftime → no derivation
- explicit leadtime → validated against derived
- calendar-aware time differences (cftime)

## Critical Files

**Read/modify:**
- `src/cmor4/grid.py:319-334` — mapping_attributes()
- `src/cmor4/utils/tables.py:102-126` — GridMappingEntry
- `src/cmor4/tables.py` — ProjectTables.grid() method
- `src/cmor4/utils/construction.py` — build_axis_mappings(), new derive_forecast_coords()
- `src/cmor4/dataset.py:114-121` — create_dataset() integration
- `src/cmor4/utils/validation.py` — new validate_grid_mapping()

**Read for context:**
- `src/cmor4/axis.py` — Axis model
- `src/cmor4/utils/time_utils.py` — time math utilities
- `tests/test_grid_axes.py` — existing grid tests
- `tests/test_cmip7_examples.py` — grid usage examples

## Implementation Order

1. **Grid parameter normalization** — add standard_parallel1/2 → standard_parallel logic to mapping_attributes()
2. **GridMappingEntry helpers** — add required_params(), optional_params() methods
3. **Grid validation** — add validate_grid_mapping() and integrate into ProjectTables.grid()
4. **Forecast coord derivation** — add derive_forecast_coords() helper
5. **Dataset integration** — call derive_forecast_coords() from create_dataset()
6. **Tests** — comprehensive tests for both features

## Verification

**Grid mapping:**
```python
# Create grid with paired standard_parallel
grid = project.grid("lambert_conformal_conic", params={
    "standard_parallel1": (30.0, "degrees_north"),
    "standard_parallel2": (60.0, "degrees_north"),
})

# Write dataset
ds = cmor4.create_dataset(dataset, variable, axes, data, grid=grid)

# Check normalized output
assert ds["crs"].attrs["standard_parallel"] == [30.0, 60.0]
assert "standard_parallel1" not in ds["crs"].attrs
assert "standard_parallel_units" not in ds["crs"].attrs
```

**Forecast leadtime:**
```python
# Create forecast axes
time = project.axis("time", values=[0, 6, 12], units="hours since 2020-01-01")
reftime = project.axis("reftime", values=[0], units="hours since 2020-01-01", 
                       standard_name="forecast_reference_time", scalar=True)

# Create dataset
ds = cmor4.create_dataset(dataset, variable, [time, reftime, ...], data)

# Check derived leadtime
assert "leadtime" in ds.coords
assert list(ds["leadtime"].values) == [0, 6, 12]
assert ds["leadtime"].attrs["units"] == "hours"
```

Run existing tests to ensure no regressions:
```bash
pytest tests/test_grid_axes.py
pytest tests/test_cmip7_examples.py
pytest tests/test_project_tables.py
```
