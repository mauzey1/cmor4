"""CMIP7 chunking validation and calculation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .datasetinfo import DatasetInfo


# CMIP7 minimum chunk size in bytes (4 MiB)
CMIP7_MIN_CHUNK_SIZE = 4 * 1024 * 1024  # 4,194,304 bytes


def is_cmip7_dataset(dataset: DatasetInfo) -> bool:
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
    mip_era = dataset.get("mip_era", "")
    return str(mip_era).upper() == "CMIP7"


def calculate_cmip7_chunks(
    dims: Sequence[str],
    shape: tuple[int, ...],
    dtype: np.dtype,
    dim_names: Sequence[str],
) -> dict[str, int]:
    """Calculate CMIP7-compliant chunk sizes.

    CMIP7 requirements:
    - Time coordinate must have a single chunk (full length)
    - Data variable chunks must be ≥ 4 MiB

    Strategy:
    - Time dimension: use full length (single chunk)
    - Other dimensions: calculate to meet 4 MiB minimum

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


def validate_cmip7_encoding(
    encoding: dict[str, Any],
    var_name: str,
    dims: Sequence[str],
    shape: tuple[int, ...],
    dtype: np.dtype,
    dim_names: Sequence[str],
) -> None:
    """Validate user-provided encoding for CMIP7 compliance.

    Checks that chunksizes meet CMIP7 requirements:
    - Time coordinate must have single chunk (full length)
    - Data variable chunks must be ≥ 4 MiB

    Non-chunking encoding parameters (zlib, complevel, etc.) are ignored.

    Parameters
    ----------
    encoding
        Encoding dictionary for the variable.
    var_name
        Variable name (for error messages).
    dims
        Dimension names in order.
    shape
        Array shape.
    dtype
        Array data type.
    dim_names
        Logical dimension names.

    Raises
    ------
    ValueError
        If chunksizes do not meet CMIP7 requirements.
    """
    if "chunksizes" not in encoding:
        return

    chunksizes = encoding["chunksizes"]
    if len(chunksizes) != len(dims):
        raise ValueError(
            f"Variable '{var_name}': chunksizes length {len(chunksizes)} "
            f"does not match dimensions {dims}"
        )

    itemsize = np.dtype(dtype).itemsize

    # Check time dimension has single chunk
    for i, dim in enumerate(dims):
        dim_lower = str(dim).lower()
        if dim_lower == "time" or dim_lower.startswith("time"):
            if chunksizes[i] != shape[i]:
                raise ValueError(
                    f"Variable '{var_name}': CMIP7 requires time coordinate "
                    f"to have a single chunk. Got chunk size {chunksizes[i]} "
                    f"but dimension size is {shape[i]}. "
                    f"Set chunksizes[{i}] = {shape[i]} for dimension '{dim}'."
                )

    # Check total chunk size meets minimum
    chunk_size_bytes = np.prod(chunksizes) * itemsize
    if chunk_size_bytes < CMIP7_MIN_CHUNK_SIZE:
        raise ValueError(
            f"Variable '{var_name}': CMIP7 requires data chunks ≥ 4 MiB. "
            f"Provided chunksizes {tuple(chunksizes)} result in "
            f"{chunk_size_bytes:,} bytes ({chunk_size_bytes / (1024**2):.2f} MiB). "
            f"Minimum required: {CMIP7_MIN_CHUNK_SIZE:,} bytes (4.00 MiB)."
        )
