# DatasetWriter Usage Guide

**Version:** Phase 1 (Incremental Writes)  
**Status:** Production-ready

## Overview

`DatasetWriter` is a memory-bounded incremental writer for creating CMOR-compliant NetCDF files one time slice at a time. It uses Zarr for staging and dask for lazy loading, enabling you to write datasets larger than available RAM without ever loading the complete time series into memory.

## When to Use DatasetWriter

### ✅ Use DatasetWriter when:

- **Writing data incrementally** as a climate model runs (monthly, annual output)
- **Dataset is larger than RAM** and you can't load the complete time series
- **Data becomes available over time** and you want to write it immediately
- **Splitting long time series** across multiple output files
- **Processing streaming data** where complete time range isn't known upfront

### ❌ Use `cmorize()` instead when:

- **Complete dataset fits in memory** and is already in a single array
- **Simple one-shot write** where all data is available at once
- **Prototyping or small datasets** where memory isn't a concern

## Basic Usage

### 1. Incremental Time Slices

Write data one time slice at a time as it becomes available:

```python
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
    "institution_id": "CCCma",
    "source_id": "DUMMY-MODEL",
    "experiment_id": "amip",
    "license_id": "CC-BY-4.0",
    "variant_label": "r1i1p1f1",
    "grid_label": "gn",
    "outpath": "/path/to/output",
})

variable = project.variable("tos_tavg-u-hxy-sea", table_id="ocean")

# Define axes (time axis is empty - values provided per write)
axes = [
    project.axis("time", units="days since 1850-01-01"),
    project.axis("latitude", values=[-45.0, 45.0], bounds=[[-90, 0], [0, 90]]),
    project.axis("longitude", values=[90.0, 270.0], bounds=[[0, 180], [180, 360]]),
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
        data = load_monthly_data(year)  # Your function: returns (12, 2, 2) array
        
        # Write this year
        writer.write(data, time_values=time_values, time_bounds=time_bounds)
        print(f"Wrote {year}: {len(time_values)} time steps")

print(f"Complete dataset written")
# File is automatically finalized and closed by context manager
```

### 2. Pre-Specified Time Values

If you know the complete time range upfront, you can provide it in the time axis:

```python
# Time axis with complete values
time_values = np.arange(0, 3650, 30.0)  # 10 years of monthly data
time_bounds = np.stack([time_values, time_values + 30.0], axis=1)
time_axis = project.axis(
    "time",
    values=time_values.tolist(),
    bounds=time_bounds.tolist(),
    units="days since 1850-01-01",
)

axes = [time_axis, lat, lon]

with cmor4.DatasetWriter(dataset, variable, axes) as writer:
    # Write in chunks - time values are taken from the axis
    for i in range(0, len(time_values), 12):
        chunk = data[i:i+12]
        writer.write(chunk)  # No time_values needed
```

### 3. Explicit Control (No Context Manager)

For more control over when finalization happens:

```python
writer = cmor4.DatasetWriter(dataset, variable, axes, path="/path/to/output.nc")

# Write data
writer.write(data1, time_values=[15.0], time_bounds=[[0.0, 30.0]])
writer.write(data2, time_values=[45.0], time_bounds=[[30.0, 60.0]])

# Finalize and get result
ds, output_path = writer.close()

print(f"Wrote {output_path}")
print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

# Close the returned dataset when done inspecting
ds.close()
```

## Advanced Features

### Time Bounds Formats

`DatasetWriter` accepts time bounds in multiple formats:

```python
# Format 1: Nx2 pairs (most common)
time_bounds = [[0.0, 30.0], [30.0, 60.0]]  # Shape (2, 2)

# Format 2: N+1 edges (automatically converted to pairs)
time_bounds = [0.0, 30.0, 60.0]  # Shape (3,)

# Format 3: Single time slice as [lower, upper]
time_bounds = [[0.0, 30.0]]  # Shape (1, 2)

writer.write(data, time_values=[15.0, 45.0], time_bounds=time_bounds)
```

### Allowing Time Gaps

By default, `DatasetWriter` requires time bounds to be contiguous (edge-to-edge). To allow gaps:

```python
writer = cmor4.DatasetWriter(
    dataset,
    variable,
    axes,
    allow_time_gaps=True,  # Disable contiguity checking
)

# Write non-contiguous time records
writer.write(data1, time_values=[15.0], time_bounds=[[0.0, 30.0]])
# Gap: next time starts at 100 instead of 30
writer.write(data2, time_values=[115.0], time_bounds=[[100.0, 130.0]])

ds, path = writer.close()  # Succeeds despite gap
```

### Custom Output Path

Control where the output file is written:

```python
# Explicit path
writer = cmor4.DatasetWriter(
    dataset,
    variable,
    axes,
    path="/custom/path/to/output.nc",
)

# Path is generated from metadata if not specified
writer = cmor4.DatasetWriter(dataset, variable, axes)  # Uses CMIP7 DRS
```

### Encoding and Compression

Specify NetCDF encoding parameters:

```python
encoding = {
    "tos": {
        "chunksizes": (1, 2, 2),  # Time, lat, lon
        "zlib": True,              # Enable compression
        "complevel": 4,            # Compression level (1-9)
        "shuffle": True,           # Byte shuffle filter
        "_FillValue": 1.0e20,     # Fill value for missing data
    }
}

writer = cmor4.DatasetWriter(
    dataset,
    variable,
    axes,
    encoding=encoding,
)
```

If not specified, CMIP7 auto-chunking rules are applied automatically.

### Additional Global Attributes

Add custom global attributes:

```python
attrs = {
    "comment": "Experimental run with increased CO2",
    "contact": "user@institution.edu",
}

writer = cmor4.DatasetWriter(dataset, variable, axes, attrs=attrs)
```

### Custom Staging Directory

Direct staging to high-performance storage:

```python
writer = cmor4.DatasetWriter(
    dataset,
    variable,
    axes,
    staging_dir="/fast/scratch/staging",  # Fast local SSD, etc.
)
```

Useful when system `/tmp` is small or slow.

### Handling Existing Files

Control behavior when output file already exists:

```python
# Default: raise error if file exists
writer = cmor4.DatasetWriter(dataset, variable, axes, existing="error")

# Overwrite existing file
writer = cmor4.DatasetWriter(dataset, variable, axes, existing="replace")

# Append to existing file (Phase 2 - not yet implemented)
# writer = cmor4.DatasetWriter(dataset, variable, axes, existing="append")
```

## Memory Characteristics

### Memory Usage

`DatasetWriter` is designed for memory-bounded operation:

- **Per-write memory**: O(chunk size) - only current chunk in memory
- **Finalization overhead**: ~20-70 MB for metadata and coordinate arrays
- **Total memory**: **Independent of total dataset size**

Example: Writing a 10 GB dataset with 1 MB chunks:
- Peak memory: ~1 MB (chunk) + 50 MB (overhead) = ~50 MB
- Works on machines with only 1 GB RAM

### Disk Usage

Temporary Zarr staging store uses ~2× final NetCDF file size:

- **10 GB NetCDF** → ~20 GB temporary Zarr store
- Automatically cleaned up on success
- Preserved on error for debugging

### Performance

Write operations are fast - limited by:
1. Data validation (NaN checks, range validation)
2. Zarr write speed (typically very fast on local disk)
3. NetCDF finalization (streaming write, memory-bounded)

Expect similar total write time to `cmorize()` for equivalent dataset.

## Error Handling

### Validation Errors

```python
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
```

### Staging Directory Inspection

On error, the staging directory is preserved for debugging:

```python
writer = cmor4.DatasetWriter(dataset, variable, axes)
print(f"Staging directory: {writer.staging_root}")

try:
    writer.write(bad_data, time_values=[15.0], time_bounds=[[0.0, 30.0]])
    writer.close()
except Exception as e:
    print(f"Error: {e}")
    print(f"Inspect staging at: {writer.staging_path}")
    # Zarr store remains for debugging
```

### Common Errors

**Shape mismatch:**
```python
# Error: data shape (2, 3, 3) doesn't match axes (time, lat=2, lon=2)
writer.write(np.ones((2, 3, 3)), time_values=[15.0, 45.0])
# ValueError: Data chunk shape (2, 3, 3) does not match expected shape (2, 2, 2)
```

**Non-monotonic time:**
```python
writer.write(data1, time_values=[45.0])
writer.write(data2, time_values=[15.0])  # Goes backward!
# ValueError: Time values must be strictly monotonic across writes
```

**Non-contiguous bounds:**
```python
writer.write(data1, time_values=[15.0], time_bounds=[[0.0, 30.0]])
writer.write(data2, time_values=[45.0], time_bounds=[[35.0, 60.0]])  # Gap from 30 to 35
# ValueError: Time bounds must be contiguous across writes unless allow_time_gaps=True
```

**Inconsistent bounds:**
```python
writer.write(data1, time_values=[15.0], time_bounds=[[0.0, 30.0]])
writer.write(data2, time_values=[45.0])  # Oops, forgot bounds
# ValueError: time_bounds must be supplied for every write or omitted for every write
```

## Best Practices

### 1. Use Context Managers

Always use `with` statement for automatic cleanup:

```python
# Good - automatic cleanup
with cmor4.DatasetWriter(dataset, variable, axes) as writer:
    writer.write(data, time_values=times, time_bounds=bounds)
# File automatically finalized

# Avoid - manual cleanup required
writer = cmor4.DatasetWriter(dataset, variable, axes)
writer.write(data, time_values=times, time_bounds=bounds)
writer.close()  # Must remember to call close()
```

### 2. Validate Early

Test with small data first:

```python
# Test with one time slice first
test_data = data[:1]
test_time = time_values[:1]
test_bounds = time_bounds[:1]

with cmor4.DatasetWriter(dataset, variable, axes) as writer:
    writer.write(test_data, time_values=test_time, time_bounds=test_bounds)
print("Validation successful! Proceeding with full write...")
```

### 3. Chunk Size

Choose chunk sizes that balance memory and I/O:

```python
# Too small: many writes, overhead
for i in range(1000):
    writer.write(data[i:i+1], ...)  # 1 time slice per write

# Good: monthly or annual chunks
for year in range(start_year, end_year):
    year_data = load_year(year)  # 12 time slices
    writer.write(year_data, ...)

# Too large: defeats memory-bounded design
writer.write(entire_dataset, ...)  # Just use cmorize() instead
```

### 4. Monitor Progress

```python
with cmor4.DatasetWriter(dataset, variable, axes) as writer:
    for year in range(1850, 2015):
        time_vals, data = load_year(year)
        writer.write(data, time_values=time_vals, time_bounds=bounds)
        
        # Progress indicator
        total_slices = writer._time_offset
        print(f"Year {year}: {total_slices} time slices written")
```

### 5. Handle Errors Gracefully

```python
writer = cmor4.DatasetWriter(dataset, variable, axes)

try:
    for year in years:
        data = load_year(year)
        writer.write(data, time_values=times, time_bounds=bounds)
except Exception as e:
    print(f"Error on year {year}: {e}")
    print(f"Partial data in: {writer.staging_path}")
    raise
else:
    ds, path = writer.close()
    print(f"Success: {path}")
```

## Comparison: DatasetWriter vs cmorize()

| Feature | `cmorize()` | `DatasetWriter` |
|---------|-------------|-----------------|
| **Memory** | O(dataset size) | O(chunk size) |
| **Time specification** | Complete upfront | Incremental or complete |
| **Use case** | One-shot write | Streaming/incremental |
| **Complexity** | Simple | Moderate |
| **Performance** | Fast | Comparable |
| **Max dataset size** | Limited by RAM | Unlimited |
| **Validation** | Full validation | Same validation |
| **Output** | Single NetCDF | Single NetCDF |

**Rule of thumb:** If your complete dataset fits comfortably in memory (say, <50% of RAM), use `cmorize()`. If dataset is larger or arrives incrementally, use `DatasetWriter`.

## Phase 1 Limitations

Current implementation (Phase 1) does not support:

- ❌ **Append mode** - Extending existing NetCDF files with new time records
- ❌ **Preserve mode** - Reusing metadata definitions across multiple files
- ❌ **Per-chunk zfactors** - Writing formula-term variables incrementally
- ❌ **Incremental spatial axes** - Only time dimension can be written incrementally

These features are planned for future phases (Phase 2 and Phase 3).

## API Reference

For complete API documentation, see:
- Module docstring: [`src/cmor4/writer.py`](../src/cmor4/writer.py)
- Class documentation: `help(cmor4.DatasetWriter)`
- Method documentation: `help(cmor4.DatasetWriter.write)`

## Troubleshooting

### Problem: Out of memory errors

**Solution:** Reduce chunk size in `write()` calls:

```python
# Instead of writing 10 years at once
writer.write(data_10years, ...)

# Write 1 year at a time
for year in range(10):
    writer.write(data_1year[year], ...)
```

### Problem: Slow writes

**Possible causes:**
1. Staging directory on slow storage (network filesystem)
2. Very small chunks with high overhead
3. Complex validation (many axes, bounds)

**Solutions:**
- Move staging to fast local storage: `staging_dir="/fast/local/scratch"`
- Increase chunk size: write monthly or annual chunks instead of daily
- Pre-validate metadata before loop

### Problem: Disk full during write

Zarr staging requires ~2× final file size.

**Solutions:**
- Check available disk: `df -h /tmp`
- Use different staging location: `staging_dir="/larger/disk/staging"`
- Split into multiple smaller output files

### Problem: Time validation errors

**Common issue:** Frequency mismatch

```
AxisValidationError: Time interval mismatch detected for frequency: 'mon'. 
Expected interval: 30 days. Actual interval: 28 days (6.7% difference).
```

**Solution:** Use `allow_time_gaps=True` or adjust time values to match expected frequency:

```python
# Monthly data with varying month lengths
time_values = [15.0, 43.5, 74.0, ...]  # Day-of-year for each month

# Use allow_time_gaps for non-standard intervals
writer = cmor4.DatasetWriter(dataset, variable, axes, allow_time_gaps=True)
```

## Examples

See the test files for comprehensive examples:
- [`tests/test_incremental_writes.py`](../tests/test_incremental_writes.py) - Basic usage
- [`tests/test_datasetwriter_expanded.py`](../tests/test_datasetwriter_expanded.py) - Advanced scenarios

## Support

For questions, issues, or feature requests:
- GitHub Issues: https://github.com/mauzey1/cmor4/issues
- Email: PCMDI support

## Acknowledgments

`DatasetWriter` was developed as part of CMOR4 by the Program for Climate Model Diagnosis and Intercomparison (PCMDI) at Lawrence Livermore National Laboratory.
