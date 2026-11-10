"""NetCDF encoding, file writes, and atomic append transactions.

All writes receive an explicit path. Public output naming remains in dataset;
this module does not depend on dataset assembly or the writer state machine.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import xarray as xr

from ._time import bounds_as_pairs

if TYPE_CHECKING:
    from ._metadata import DatasetMetadata
    from .axis import Axis
    from .variable import Variable


CMIP7_MIN_CHUNK_SIZE = 4 * 1024 * 1024  # 4,194,304 bytes


def is_cmip7_dataset(dataset: DatasetMetadata) -> bool:
    """Check if dataset is CMIP7.

    Parameters
    ----------
    dataset
        Dataset metadata.

    Returns
    -------
    bool
        True if mip_era is "CMIP7" (case-insensitive).
    """
    mip_era = dataset.mip_era or ""
    return str(mip_era).upper() == "CMIP7"


def calculate_cmip7_chunks(
    dims: Sequence[str],
    shape: tuple[int, ...],
    dtype: np.dtype,
    dim_names: Sequence[str],
) -> dict[str, int]:
    """Calculate CMIP7-compliant chunk sizes.

    CMIP7 requirements:
    - Time coordinate and bounds variables must have a single chunk along time
    - Data variable chunks must be ≥ 4 MiB

    Strategy:
    - Use full dimensions for the data variable, which always satisfies the
      single-time-chunk requirement and maximizes chunk size for small arrays.

    Parameters
    ----------
    dims
        Dimension names in order.
    shape
        Array shape.
    dtype
        Array data type.
    dim_names
        Logical dimension names.

    Returns
    -------
    dict[str, int]
        Mapping from dimension name to chunk size.
    """
    if len(dims) != len(shape):
        raise ValueError(
            f"Dimension count {len(dims)} does not match shape dimensions {len(shape)}"
        )

    itemsize = np.dtype(dtype).itemsize
    chunks = {}

    # Find time dimension and set it to full length
    time_idx = None
    for i, dim in enumerate(dims):
        dim_lower = str(dim).lower()
        if dim_lower == "time" or dim_lower.startswith("time"):
            time_idx = i
            chunks[dim] = shape[i]
            break

    # Calculate chunk sizes for non-time dimensions
    # Start with full size for all dimensions
    chunk_shape = list(shape)

    # If we have a time dimension, it's already set to full length
    if time_idx is not None:
        chunk_shape[time_idx] = shape[time_idx]

    # Calculate current chunk size in bytes
    current_chunk_size = np.prod(chunk_shape) * itemsize

    # If already meets minimum, use full shape for all dims
    if current_chunk_size >= CMIP7_MIN_CHUNK_SIZE:
        for i, dim in enumerate(dims):
            if dim not in chunks:
                chunks[dim] = shape[i]
        return chunks

    # Otherwise, we need to keep all dimensions at full size to meet minimum
    # This is the simplest compliant strategy for small arrays
    for i, dim in enumerate(dims):
        if dim not in chunks:
            chunks[dim] = shape[i]

    return chunks


def validate_cmip7_chunksizes(
    chunksizes: Sequence[int],
    var_name: str,
    dims: Sequence[str],
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> None:
    """Validate user-provided chunksizes for CMIP7 compliance.

    Checks that chunksizes meet CMIP7 requirements for the data variable:
    - Data variable chunks must be ≥ 4 MiB (4,194,304 bytes)
    - Data variable chunks can have any shape as long as they meet the size requirement
    - Data variables may have multiple chunks along time if each chunk is ≥ 4 MiB

    The single-chunk requirement for time coordinates and time bounds is enforced
    separately in core.py, not by this function.

    Parameters
    ----------
    chunksizes
        User-provided chunk sizes for each dimension.
    var_name
        Variable name (for error messages).
    dims
        Dimension names in order.
    shape
        Array shape.
    dtype
        Array data type.

    Raises
    ------
    ValueError
        If chunksizes do not meet CMIP7 ≥4 MiB requirement.
    """
    if len(chunksizes) != len(dims):
        raise ValueError(
            f"Variable '{var_name}': chunksizes length {len(chunksizes)} "
            f"does not match dimensions {dims}"
        )

    itemsize = np.dtype(dtype).itemsize

    # Check total chunk size meets minimum
    chunk_size_bytes = np.prod(chunksizes) * itemsize
    if chunk_size_bytes < CMIP7_MIN_CHUNK_SIZE:
        raise ValueError(
            f"Variable '{var_name}': CMIP7 requires data chunks ≥ 4 MiB. "
            f"Provided chunksizes {tuple(chunksizes)} result in "
            f"{chunk_size_bytes:,} bytes ({chunk_size_bytes / (1024**2):.2f} MiB). "
            f"Minimum required: {CMIP7_MIN_CHUNK_SIZE:,} bytes (4.00 MiB)."
        )


def apply_dataset_encoding(
    ds: xr.Dataset,
    dataset: DatasetMetadata,
    variable: Variable,
    encoding: Mapping[str, Any] | None,
) -> None:
    """Apply variable encoding, CMIP7 chunks, and coordinate fill-value rules."""
    data_array: Any
    var_name = variable.names()[0]
    missing_value = variable.missing_value or variable.fill_value
    if missing_value is not None:
        ds[var_name].attrs["missing_value"] = missing_value
        ds[var_name].encoding["_FillValue"] = missing_value

    # Extract user-provided chunksizes for the data variable.
    # Priority: variable-specific encoding["var_name"]["chunksizes"] >
    #           top-level encoding["chunksizes"] >
    #           variable metadata (chunksizes/chunks)
    user_provided_chunks = None
    if encoding:
        var_encoding = encoding.get(var_name)
        if isinstance(var_encoding, Mapping) and "chunksizes" in var_encoding:
            user_provided_chunks = var_encoding["chunksizes"]
        elif "chunksizes" in encoding and not isinstance(
            encoding["chunksizes"], Mapping
        ):
            user_provided_chunks = encoding["chunksizes"]

    # Fallback to variable metadata for chunksizes if not in encoding
    chunksizes = user_provided_chunks or variable.chunksizes or variable.chunks

    # For CMIP7 datasets, validate user-provided chunksizes meet the
    # ≥4 MiB requirement. Allow all other encoding parameters to pass through.
    if user_provided_chunks and is_cmip7_dataset(dataset):
        data_array = ds[var_name]
        validate_cmip7_chunksizes(
            user_provided_chunks,
            var_name,
            data_array.dims,
            data_array.shape,
            data_array.dtype,
        )

    # Auto-generate CMIP7-compliant chunks if no chunksizes were provided
    # (neither in encoding nor in variable metadata)
    if chunksizes is None and is_cmip7_dataset(dataset):
        data_array = ds[var_name]
        cmip7_chunks = calculate_cmip7_chunks(
            data_array.dims,
            data_array.shape,
            data_array.dtype,
            data_array.dims,
        )
        ds[var_name].encoding["chunksizes"] = tuple(
            cmip7_chunks[d] for d in data_array.dims
        )
    elif chunksizes:
        ds[var_name].encoding["chunksizes"] = tuple(int(value) for value in chunksizes)

    # For CMIP7 datasets, identify time-related variables that must have
    # a single chunk (time coordinate and its bounds/climatology).
    if is_cmip7_dataset(dataset):
        protected_chunks: set[str] = set()
        for coord_name in ds.coords:
            coord_name_str = str(coord_name)
            lower = coord_name_str.lower()
            # Only protect time coordinates, not other coordinates
            if lower != "time" and not lower.startswith("time"):
                continue
            protected_chunks.add(coord_name_str)
            coord = ds[coord_name_str]
            # Also protect time bounds and climatology bounds
            for attr_name in ("bounds", "climatology"):
                bounds_name = coord.attrs.get(attr_name)
                if bounds_name is not None and str(bounds_name) in ds:
                    protected_chunks.add(str(bounds_name))
    else:
        protected_chunks = set()

    # Apply user-provided encoding to variables
    if encoding:
        for name in ds.variables:
            name_str = str(name)

            # Start with top-level encoding parameters, excluding chunksizes
            # and nested dicts.
            # This mimics xarray's encoding behavior
            for key, value in encoding.items():
                if key == "chunksizes" or isinstance(value, Mapping):
                    continue
                ds[name_str].encoding[key] = value

            # Apply variable-specific encoding overrides
            var_specific = encoding.get(name_str)
            if isinstance(var_specific, Mapping):
                for key, value in var_specific.items():
                    if key == "chunksizes":
                        # Handle chunksizes for non-data variables
                        if name_str == var_name:
                            # Data variable chunksizes already applied above
                            continue

                        array = ds[name_str]
                        if len(value) != array.ndim:
                            raise ValueError(
                                f"Variable {name_str!r}: chunksizes length "
                                f"{len(value)} does not match dimensions {array.dims}"
                            )
                        normalized_chunksizes = tuple(int(v) for v in value)

                        # For CMIP7, enforce single chunk for time and time_bnds
                        if (
                            name_str in protected_chunks
                            and normalized_chunksizes != array.shape
                        ):
                            raise ValueError(
                                f"Variable {name_str!r}: CMIP7 requires time "
                                "coordinates and time bounds to have a single "
                                f"chunk. Got chunk sizes {normalized_chunksizes} "
                                f"but variable shape is {array.shape}."
                            )
                        array.encoding["chunksizes"] = normalized_chunksizes
                    else:
                        # Apply other variable-specific encoding parameters
                        ds[name_str].encoding[key] = value

    # For CMIP7, automatically apply single-chunk encoding to time coordinates
    # and time bounds if not already set by user
    if protected_chunks:
        for name in protected_chunks:
            array = ds[name]
            # Skip scalar variables and empty arrays
            if array.ndim != 0 and not any(size == 0 for size in array.shape):
                # Set chunk size to full array shape (single chunk)
                array.encoding["chunksizes"] = tuple(int(size) for size in array.shape)

    # Suppress the _FillValue attribute on coordinate variables and their bounds
    coordinate_names = {str(name) for name in ds.coords}
    for data_var in ds.variables.values():
        coordinates = data_var.attrs.get("coordinates")
        if coordinates:
            coordinate_names.update(
                name for name in str(coordinates).split() if name in ds.variables
            )

    coordinate_bounds_names: set[str] = set()
    for coord_name in coordinate_names:
        if coord_name not in ds.variables:
            continue
        coord = ds[coord_name]
        for attr_name in ("bounds", "climatology"):
            bounds_name = coord.attrs.get(attr_name)
            if bounds_name is not None and str(bounds_name) in ds.variables:
                coordinate_bounds_names.add(str(bounds_name))

    for name in coordinate_names | coordinate_bounds_names:
        if name not in ds.variables:
            continue
        ds[name].attrs.pop("_FillValue", None)
        ds[name].encoding["_FillValue"] = None


def prepare_append_encoding(merged_ds: xr.Dataset, new_ds: xr.Dataset) -> None:
    for name in merged_ds.variables:
        name_str = str(name)
        encoding = (
            dict(new_ds[name_str].encoding) if name_str in new_ds.variables else {}
        )
        for transient_key in ("preferred_chunks", "source", "original_shape"):
            encoding.pop(transient_key, None)

        array = merged_ds[name_str]
        chunksizes = encoding.get("chunksizes")
        if chunksizes is not None:
            normalized_chunksizes = tuple(
                min(int(chunk), int(size))
                for chunk, size in zip(tuple(chunksizes), array.shape, strict=False)
            )
            if is_time_metadata_variable(name_str, array, merged_ds):
                normalized_chunksizes = tuple(int(size) for size in array.shape)
            encoding["chunksizes"] = normalized_chunksizes

        array.encoding.clear()
        array.encoding.update(encoding)
        if "_FillValue" in array.attrs and "_FillValue" in array.encoding:
            array.attrs.pop("_FillValue", None)


def is_time_metadata_variable(
    name: str,
    array: xr.DataArray,
    ds: xr.Dataset,
) -> bool:
    lower_name = name.lower()
    if lower_name == "time" or lower_name.startswith("time"):
        return True
    for coord_name in ds.coords:
        coord = ds[str(coord_name)]
        if name in {
            str(coord.attrs.get("bounds", "")),
            str(coord.attrs.get("climatology", "")),
        }:
            return True
    return array.ndim > 0 and all(
        str(dim).lower() in {"time", "bnds"} or str(dim).lower().startswith("time")
        for dim in array.dims
    )


def write_dataset(
    ds: xr.Dataset,
    dataset: DatasetMetadata,
    output_path: Path,
    **to_netcdf_kwargs: Any,
) -> Path:
    """Write to an explicit path using the caller's directory-creation policy."""

    # Honour create_subdirectories.  The flag mirrors CMOR3's own behaviour:
    # when False, the output directory must already exist; CMOR3 errors when
    # it cannot create the directory (e.g. a non-existent /CMIP6 root).
    # Default is True for backwards-compatibility with CMOR's original
    # always-create behaviour.
    create_subdirs = (
        True if dataset.create_subdirectories is None else dataset.create_subdirectories
    )
    if create_subdirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif not output_path.parent.exists():
        raise ValueError(
            f"Output directory {str(output_path.parent)!r} does not exist "
            "and create_subdirectories is disabled. "
            "Create the directory first or set create_subdirectories=True "
            "in the dataset metadata."
        )

    ds.to_netcdf(output_path, **to_netcdf_kwargs)

    return output_path


APPEND_IGNORED_GLOBAL_ATTRS = frozenset({
    "creation_date",
    "history",
    "tracking_id",
})
APPEND_IGNORED_VARIABLE_ATTRS = frozenset({"_FillValue"})


def append_to_existing(
    output_path: Path,
    new_ds: xr.Dataset,
    *,
    dataset: DatasetMetadata,
    variable: Variable,
    time_dim: str,
    time_axis: Axis,
    to_netcdf_kwargs: Mapping[str, Any],
) -> Path:
    """Validate and append through temporary files before atomic replacement."""
    if not output_path.exists():
        raise FileNotFoundError(
            f"Cannot append because output file {str(output_path)!r} does not exist."
        )

    temp_path: Path | None = None
    check_path: Path | None = None
    existing_ds = xr.open_dataset(
        output_path,
        decode_times=False,
        mask_and_scale=False,
    )
    existing_check_ds = xr.open_dataset(
        output_path,
        decode_times=False,
        mask_and_scale=False,
        decode_coords=False,
    )
    new_check_ds: xr.Dataset | None = None
    try:
        check_path = temporary_netcdf_path(output_path)
        write_dataset(
            new_ds,
            dataset,
            output_path=check_path,
            **to_netcdf_kwargs,
        )
        if not check_path.exists() or check_path.stat().st_size == 0:
            raise OSError(
                "Append compatibility check did not create a valid "
                f"temporary file {str(check_path)!r}."
            )
        new_check_ds = xr.open_dataset(
            check_path,
            decode_times=False,
            mask_and_scale=False,
            decode_coords=False,
        )
        _validate_append_compatible(existing_check_ds, new_check_ds, variable, time_dim)
        merged_ds = xr.concat(
            [existing_ds, new_ds],
            dim=time_dim,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )
        _validate_merged_time(merged_ds, time_dim, time_axis)
        prepare_append_encoding(merged_ds, new_ds)
        prepare_append_attrs(merged_ds, existing_ds, new_ds)
        temp_path = temporary_netcdf_path(output_path)
        write_dataset(
            merged_ds,
            dataset,
            output_path=temp_path,
            **to_netcdf_kwargs,
        )
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise OSError(
                "Append write did not create a valid temporary file "
                f"{str(temp_path)!r}."
            )
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        if new_check_ds is not None:
            new_check_ds.close()
        if check_path is not None:
            check_path.unlink(missing_ok=True)
        existing_check_ds.close()
        existing_ds.close()

    os.replace(temp_path, output_path)
    return output_path


def _validate_append_compatible(
    existing_ds: xr.Dataset,
    new_ds: xr.Dataset,
    variable: Variable,
    time_dim: str,
) -> None:
    issues: list[str] = []
    var_name = variable.names()[0]

    if var_name not in existing_ds:
        issues.append(f"existing file is missing data variable {var_name!r}")
    if var_name not in new_ds:
        issues.append(f"new dataset is missing data variable {var_name!r}")

    existing_variables = {str(name) for name in existing_ds.variables}
    new_variables = {str(name) for name in new_ds.variables}
    if existing_variables != new_variables:
        missing = sorted(new_variables - existing_variables)
        extra = sorted(existing_variables - new_variables)
        if missing:
            issues.append(f"existing file is missing variables {missing!r}")
        if extra:
            issues.append(f"existing file has unexpected variables {extra!r}")

    for dim, size in new_ds.sizes.items():
        if dim == time_dim:
            continue
        existing_size = existing_ds.sizes.get(dim)
        if existing_size != size:
            issues.append(
                f"dimension {dim!r} has size {existing_size!r} in the "
                f"existing file and {size!r} in the new dataset"
            )

    issues.extend(
        attribute_diff_messages(
            "global attribute",
            existing_ds.attrs,
            new_ds.attrs,
            ignored=APPEND_IGNORED_GLOBAL_ATTRS,
        )
    )

    for name in sorted(existing_variables & new_variables):
        existing_var = existing_ds[name]
        new_var = new_ds[name]
        if existing_var.dims != new_var.dims:
            issues.append(
                f"variable {name!r} has dimensions {existing_var.dims!r} "
                f"in the existing file and {new_var.dims!r} in the new dataset"
            )
            continue
        if np.dtype(existing_var.dtype) != np.dtype(new_var.dtype):
            issues.append(
                f"variable {name!r} has dtype {existing_var.dtype!r} in "
                f"the existing file and {new_var.dtype!r} in the new dataset"
            )
        issues.extend(
            attribute_diff_messages(
                f"variable {name!r} attribute",
                existing_var.attrs,
                new_var.attrs,
                ignored=APPEND_IGNORED_VARIABLE_ATTRS,
            )
        )

        if time_dim in existing_var.dims:
            for dim, existing_size in existing_var.sizes.items():
                if dim == time_dim:
                    continue
                new_size = new_var.sizes[dim]
                if existing_size != new_size:
                    issues.append(
                        f"variable {name!r} dimension {dim!r} has size "
                        f"{existing_size!r} in the existing file and "
                        f"{new_size!r} in the new dataset"
                    )
            continue

        if existing_var.shape != new_var.shape:
            issues.append(
                f"variable {name!r} has shape {existing_var.shape!r} "
                f"in the existing file and {new_var.shape!r} in the new dataset"
            )
            continue
        if not array_values_equal(existing_var.values, new_var.values):
            issues.append(f"variable {name!r} values differ")

    if issues:
        details = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(
            "Existing file is not compatible with append mode:\n" + details
        )


def _validate_merged_time(
    merged_ds: xr.Dataset, time_dim: str, time_axis: Axis
) -> None:
    time_name = time_coord_name(merged_ds, time_dim, time_axis)
    values = np.asarray(merged_ds[time_name].values)
    if values.ndim != 1:
        raise ValueError(
            f"Time coordinate {time_name!r} must be one-dimensional after append."
        )
    if values.size > 1 and np.any(values[1:] <= values[:-1]):
        raise ValueError(
            "Appended time values must be strictly monotonic with the existing file."
        )

    bounds_name = merged_ds[time_name].attrs.get("bounds") or merged_ds[
        time_name
    ].attrs.get("climatology")
    if not bounds_name or str(bounds_name) not in merged_ds:
        return

    bounds = bounds_as_pairs(
        np.asarray(merged_ds[str(bounds_name)].values),
        int(values.size),
    )
    if bounds.shape[0] <= 1:
        return
    previous_ends = bounds[:-1, 1]
    next_starts = bounds[1:, 0]
    numeric_bounds = np.issubdtype(np.asarray(previous_ends).dtype, np.number)
    if numeric_bounds and not np.allclose(
        previous_ends,
        next_starts,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Appended time bounds must be contiguous with the existing file."
        )


def temporary_netcdf_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        return Path(handle.name)


def prepare_append_attrs(
    merged_ds: xr.Dataset,
    existing_ds: xr.Dataset,
    new_ds: xr.Dataset,
) -> None:
    attrs = dict(merged_ds.attrs)
    if "history" in existing_ds.attrs:
        attrs["history"] = existing_ds.attrs["history"]
    for name in ("creation_date", "tracking_id"):
        if name in new_ds.attrs:
            attrs[name] = new_ds.attrs[name]
    merged_ds.attrs = attrs


def normalize_attr_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(normalize_attr_value(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(normalize_attr_value(item) for item in value)
    return value


def array_values_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    try:
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left_array, right_array))


def time_coord_name(ds: xr.Dataset, time_dim: str, axis: Axis) -> str:
    candidates = [
        time_dim,
        str(axis.out_name or ""),
        str(axis.name or ""),
    ]
    for candidate in candidates:
        if candidate and candidate in ds:
            return candidate
    for name in ds.coords:
        coord = ds[str(name)]
        if time_dim in coord.dims:
            return str(name)
    raise ValueError(
        f"No coordinate variable was found for time dimension {time_dim!r}."
    )


def attribute_diff_messages(
    label: str,
    existing_attrs: Mapping[str, Any],
    new_attrs: Mapping[str, Any],
    *,
    ignored: frozenset[str],
) -> list[str]:
    existing_attrs, new_attrs = (
        {
            str(key): normalize_attr_value(value)
            for key, value in attrs.items()
            if str(key) not in ignored
        }
        for attrs in (existing_attrs, new_attrs)
    )
    messages: list[str] = []
    keys = sorted(set(existing_attrs) | set(new_attrs))
    for key in keys:
        if key not in existing_attrs:
            messages.append(f"{label} {key!r} is missing from the existing file")
        elif key not in new_attrs:
            messages.append(f"{label} {key!r} is missing from the new dataset")
        elif existing_attrs[key] != new_attrs[key]:
            messages.append(
                f"{label} {key!r} differs: existing={existing_attrs[key]!r}, "
                f"new={new_attrs[key]!r}"
            )
    return messages
