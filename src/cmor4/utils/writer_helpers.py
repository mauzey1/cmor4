from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from ..axis import Axis


APPEND_IGNORED_GLOBAL_ATTRS = frozenset(
    {
        "creation_date",
        "history",
        "tracking_id",
    }
)
APPEND_IGNORED_VARIABLE_ATTRS = frozenset({"_FillValue"})


def find_time_axis(axes: Sequence[Axis]) -> tuple[int, Axis]:
    """Return the time axis index and axis from a sequence of axes."""

    for index, axis in enumerate(axes):
        names = {
            str(value).lower()
            for value in (
                axis.name,
                axis.out_name,
                axis.table_entry,
                axis.axis_entry,
                axis.coordinate,
                axis.standard_name,
            )
            if value
        }
        if (
            str(axis.axis or "").upper() == "T"
            or "time" in names
            or any(name.startswith("time") for name in names)
        ):
            return index, axis
    raise ValueError(
        "No time axis was found. Pass an axis with axis='T' or time-like "
        "axis metadata."
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


def normalize_attrs(
    attrs: Mapping[Any, Any],
    *,
    ignored: frozenset[str],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in attrs.items():
        key_str = str(key)
        if key_str in ignored:
            continue
        normalized[key_str] = normalize_attr_value(value)
    return normalized


def normalize_attr_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(normalize_attr_value(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(normalize_attr_value(item) for item in value)
    return value


def attribute_diff_messages(
    label: str,
    existing_attrs: Mapping[str, Any],
    new_attrs: Mapping[str, Any],
) -> list[str]:
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


def metadata_time_axis(axis: Axis) -> Axis:
    values = axis.values_array()
    value = values.reshape(-1)[0] if values.size else 0.0
    value_item = value.item() if hasattr(value, "item") else value
    updates: dict[str, Any] = {"values": [value_item]}
    if axis.bounds is not None:
        bounds = bounds_as_pairs(
            axis.bounds_array(),
            len(values) if values.size else 1,
        )
        updates["bounds"] = bounds[:1].tolist()
    else:
        updates["bounds"] = [[float(value) - 0.5, float(value) + 0.5]]
    return axis.updated(**updates)


def bounds_as_pairs(bounds: np.ndarray, time_len: int) -> np.ndarray:
    if bounds.ndim == 1 and bounds.size == time_len + 1:
        return np.stack((bounds[:-1], bounds[1:]), axis=-1)
    if bounds.ndim == 1 and time_len == 1 and bounds.size == 2:
        return bounds.reshape(1, 2)
    if bounds.shape[:1] == (time_len,) and bounds.shape[-1] >= 2:
        return bounds
    raise ValueError(
        f"time_bounds shape {bounds.shape!r} does not match time_values length "
        f"{time_len}."
    )
