"""CMIP7 chunking validation and calculation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .dataset_metadata import DatasetMetadata


# CMIP7 minimum chunk size in bytes (4 MiB)
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
