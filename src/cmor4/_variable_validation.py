from __future__ import annotations

from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from .axis import Axis
from .exceptions import VariableValidationError


def validate_variable_values(
    variable: Mapping[str, Any],
    axes: Sequence[Axis],
    data: Any,
    dims: Sequence[str],
    axis_dims: Mapping[str, tuple[str, ...]],
    *,
    name: str | None = None,
    table_id: str | None = None,
) -> None:
    """Apply CMOR-style checks to the main data variable values."""

    values = _as_float_masked_array(data)
    if values is None:
        return

    valid_mask = ~np.ma.getmaskarray(values)
    missing_value = variable.get("missing_value", variable.get("fill_value"))
    if missing_value is not None:
        try:
            valid_mask &= ~np.isclose(
                values.filled(np.nan),
                float(missing_value),
                rtol=float(variable.get("tolerance", 1.0e-6)),
                atol=0.0,
                equal_nan=False,
            )
        except (TypeError, ValueError):
            pass

    numeric = values.filled(np.nan)
    nan_mask = np.isnan(numeric) & valid_mask
    if np.any(nan_mask):
        count = int(np.count_nonzero(nan_mask))
        index = _first_index(nan_mask)
        raise VariableValidationError(
            "Invalid value(s) detected for variable "
            f"{_variable_name(variable, name)!r} "
            f"(table: {_table_id(variable, table_id)}): "
            f"{count} values were NaNs. First encountered NaN was at "
            "(axis: index/value):"
            f"{_format_location(index, dims, axes, axis_dims)}"
        )

    active = numeric[valid_mask]
    active = active[np.isfinite(active)]
    if active.size == 0:
        return

    _warn_for_limit(
        variable,
        numeric,
        valid_mask,
        dims,
        axes,
        axis_dims,
        "valid_min",
        np.less,
        "lower than minimum valid value",
        np.nanmin,
        name=name,
        table_id=table_id,
    )
    _warn_for_limit(
        variable,
        numeric,
        valid_mask,
        dims,
        axes,
        axis_dims,
        "valid_max",
        np.greater,
        "greater than maximum valid value",
        np.nanmax,
        name=name,
        table_id=table_id,
    )
    _check_absolute_mean(variable, active, name=name, table_id=table_id)


def _as_float_masked_array(data: Any) -> np.ma.MaskedArray | None:
    try:
        return np.ma.asarray(data, dtype=float)
    except (TypeError, ValueError):
        return None


def _warn_for_limit(
    variable: Variable,
    numeric: np.ndarray,
    valid_mask: np.ndarray,
    dims: Sequence[str],
    axes: Sequence[Axis],
    axis_dims: Mapping[str, tuple[str, ...]],
    key: str,
    compare: Any,
    phrase: str,
    extrema: Any,
    *,
    name: str | None,
    table_id: str | None,
) -> None:
    limit = _numeric_or_none(variable.get(key))
    if limit is None:
        return
    bad_mask = compare(numeric, limit) & valid_mask
    if not np.any(bad_mask):
        return
    count = int(np.count_nonzero(bad_mask))
    bad_values = np.where(bad_mask, numeric, np.nan)
    bad_value = float(extrema(bad_values))
    index = _first_index(bad_mask)
    warnings.warn(
        "Invalid value(s) detected for variable "
        f"{_variable_name(variable, name)!r} "
        f"(table: {_table_id(variable, table_id)}): "
        f"{count} values were {phrase} ({limit:.4g}). "
        f"Encountered bad value ({bad_value:.5g}) was at "
        "(axis: index/value):"
        f"{_format_location(index, dims, axes, axis_dims)}",
        RuntimeWarning,
        stacklevel=3,
    )


def _check_absolute_mean(
    variable: Mapping[str, Any],
    active: np.ndarray,
    *,
    name: str | None,
    table_id: str | None,
) -> None:
    mean_abs = float(np.mean(np.abs(active)))
    ok_min = _numeric_or_none(variable.get("ok_min_mean_abs"))
    if ok_min is not None:
        if mean_abs < 0.1 * ok_min:
            raise VariableValidationError(
                "Invalid Absolute Mean for variable "
                f"{_variable_name(variable, name)!r} "
                f"(table: {_table_id(variable, table_id)}) "
                f"({mean_abs:.5g}) is lower by more than an order of "
                f"magnitude than minimum allowed: {ok_min:.4g}"
            )
        if mean_abs < ok_min:
            warnings.warn(
                "Invalid Absolute Mean for variable "
                f"{_variable_name(variable, name)!r} "
                f"(table: {_table_id(variable, table_id)}) "
                f"({mean_abs:.5g}) is lower "
                f"than minimum allowed: {ok_min:.4g}",
                RuntimeWarning,
                stacklevel=3,
            )

    ok_max = _numeric_or_none(variable.get("ok_max_mean_abs"))
    if ok_max is not None:
        if mean_abs > 10.0 * ok_max:
            raise VariableValidationError(
                "Invalid Absolute Mean for variable "
                f"{_variable_name(variable, name)!r} "
                f"(table: {_table_id(variable, table_id)}) "
                f"({mean_abs:.5g}) is greater by more than an order of "
                f"magnitude than maximum allowed: {ok_max:.4g}"
            )
        if mean_abs > ok_max:
            warnings.warn(
                "Invalid Absolute Mean for variable "
                f"{_variable_name(variable, name)!r} "
                f"(table: {_table_id(variable, table_id)}) "
                f"({mean_abs:.5g}) is greater "
                f"than maximum allowed: {ok_max:.4g}",
                RuntimeWarning,
                stacklevel=3,
            )


def _numeric_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_index(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.argwhere(mask)[0])


def _format_location(
    index: tuple[int, ...],
    dims: Sequence[str],
    axes: Sequence[Axis],
    axis_dims: Mapping[str, tuple[str, ...]],
) -> str:
    axis_by_dim = _axis_by_dim(axes, axis_dims)
    parts: list[str] = []
    for dim, location in zip(dims, index):
        axis = axis_by_dim.get(str(dim))
        value = _axis_value(axis, location) if axis is not None else location
        parts.append(f" {dim}: {location}/{value}")
    return "".join(parts)


def _axis_by_dim(
    axes: Sequence[Axis],
    axis_dims: Mapping[str, tuple[str, ...]],
) -> dict[str, Axis]:
    mapped: dict[str, Axis] = {}
    for axis in axes:
        name = str(axis["name"])
        dims = axis_dims.get(name, ())
        if len(dims) == 1:
            mapped.setdefault(dims[0], axis)
    return mapped


def _axis_value(axis: Axis, location: int) -> Any:
    values = axis.values_array()
    if values.ndim == 1 and location < values.shape[0]:
        value = values[location]
        if hasattr(value, "item"):
            value = value.item()
        return f"{value:.5g}" if isinstance(value, float) else value
    return location


def _variable_name(variable: Mapping[str, Any], name: str | None) -> str:
    if name is not None:
        return name
    names = getattr(variable, "names", None)
    if callable(names):
        return names()[0]
    return str(variable.get("out_name", variable.get("name", "")))


def _table_id(variable: Mapping[str, Any], table_id: str | None) -> str:
    if table_id is not None:
        return str(table_id)
    return str(variable.get("table_id", ""))
