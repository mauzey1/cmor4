from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import re
import warnings

import numpy as np

from ._time_utils import cftime_interval_days
from .axis import Axis
from .exceptions import AxisValidationError


DEFAULT_INTERVAL_WARNING = 0.1
DEFAULT_INTERVAL_ERROR = 0.2

DEFAULT_FREQUENCY_INTERVALS = {
    "1hr": 1.0 / 24.0,
    "1hrcm": 1.0 / 24.0,
    "3hr": 0.125,
    "6hr": 0.25,
    "day": 1.0,
    "dec": 3650.0,
    "fx": 0.0,
    "mon": 30.0,
    "yr": 365.0,
}


@dataclass(frozen=True)
class _IntervalSpec:
    days: float
    warning: float = DEFAULT_INTERVAL_WARNING
    error: float = DEFAULT_INTERVAL_ERROR


def validate_and_normalize_axes(
    dataset: Mapping[str, Any],
    variable: Mapping[str, Any],
    axes: Sequence[Axis],
) -> tuple[Axis, ...]:
    """Return axes after CMOR-style coordinate validation."""

    return tuple(
        _validate_and_normalize_axis(dataset, variable, axis)
        for axis in axes
    )


def validate_axis_values_early(axis: Axis) -> None:
    """Validate axis values without dataset- or frequency-dependent checks."""

    _validate_and_normalize_axis(
        {},
        {},
        axis,
        include_time_checks=False,
        enforce_required_bounds=False,
        normalize=False,
    )


def _validate_and_normalize_axis(
    dataset: Mapping[str, Any],
    variable: Mapping[str, Any],
    axis: Axis,
    *,
    include_time_checks: bool = True,
    enforce_required_bounds: bool = True,
    normalize: bool = True,
) -> Axis:
    values = axis.values_array()
    bounds = axis.bounds_array() if "bounds" in axis else None
    name = str(axis.get("table_entry") or axis.get("name"))
    climatology = _is_truthy(axis.get("climatology"))

    values, bounds = _normalize_bounds_shape(axis, values, bounds)
    if enforce_required_bounds and _requires_bounds(axis) and bounds is None:
        raise AxisValidationError(
            f"axis {name!r} must have bounds, but none were provided."
        )

    if _is_numeric(values):
        values = values.astype("f8", copy=True)
        if bounds is not None and _is_numeric(bounds):
            bounds = bounds.astype("f8", copy=True)
        if _is_longitude(axis):
            values, bounds = _normalize_longitude(axis, values, bounds)
        _validate_requested_values(axis, values, name)
        _validate_valid_range(axis, values, name, is_bounds=False)
        _validate_monotonic(axis, values, name, is_bounds=False)

    if bounds is not None and _is_numeric(bounds):
        _validate_valid_range(axis, bounds, name, is_bounds=True)
        if bounds.shape[-1] == 2:
            _validate_requested_bounds(axis, bounds, name)
            if include_time_checks or (
                not _is_time_axis(axis) and not climatology
            ):
                _validate_monotonic(axis, bounds, name, is_bounds=True)
            if include_time_checks or not _is_time_axis(axis):
                _validate_values_inside_bounds(values, bounds, name)
        if (
            include_time_checks
            and _is_time_axis(axis)
            and not climatology
            and bounds.shape[-1] == 2
        ):
            values = _time_values_from_bounds(values, bounds, name)

    if include_time_checks and _is_time_axis(axis) and not climatology:
        _validate_time_interval(dataset, variable, axis, values)

    if not normalize:
        return axis

    updates: dict[str, Any] = {}
    if not np.array_equal(values, axis.values_array()):
        updates["values"] = _array_to_user_value(values)
    if bounds is not None and "bounds" in axis:
        if not np.array_equal(bounds, axis.bounds_array()):
            updates["bounds"] = _array_to_user_value(bounds)
    return axis.updated(**updates) if updates else axis


def _normalize_bounds_shape(
    axis: Axis, values: np.ndarray, bounds: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    if bounds is None:
        return values, bounds
    values_shape = values.shape
    if axis.get("scalar", False) and values.size == 1:
        if bounds.size == 2:
            return values, bounds.reshape(2)
        raise AxisValidationError(
            "Scalar coordinate bounds must have 2 values."
        )
    if (
        bounds.ndim == 1
        and values.ndim == 1
        and bounds.size == values.size + 1
    ):
        pairs = np.stack((bounds[:-1], bounds[1:]), axis=-1)
        return values, pairs
    if bounds.shape[:-1] == values_shape and bounds.shape[-1] >= 2:
        return values, bounds
    raise AxisValidationError(
        f"axis {axis.get('name')!r} bounds shape {bounds.shape!r} does not "
        f"match coordinate value shape {values_shape!r}."
    )


def _validate_requested_values(
    axis: Axis, values: np.ndarray, name: str
) -> None:
    requested = _numeric_list(axis.get("requested"))
    if not requested:
        return
    flat_values = values.reshape(-1)
    tolerance = _tolerance(axis)
    for index, expected in enumerate(requested):
        eps = abs(1.0e-3 * tolerance * expected)
        if index > 0:
            eps = min(eps, abs(expected - requested[index - 1]) * tolerance)
        if not np.any(np.abs(flat_values - expected) <= eps):
            raise AxisValidationError(
                f"requested value {expected:g} for axis {name!r} was "
                "not found."
            )


def _validate_requested_bounds(
    axis: Axis, bounds: np.ndarray, name: str
) -> None:
    requested = _numeric_list(
        axis.get("requested_bounds", axis.get("bounds_values"))
    )
    if not requested:
        requested = _numeric_list(axis.get("bounds_values"))
    if not requested:
        return
    pairs = bounds.reshape(-1, bounds.shape[-1])
    tolerance = _tolerance(axis)
    first_bounds = pairs[:, 0]
    second_bounds = pairs[:, 1]
    for index, expected in enumerate(requested):
        neighbor = (
            requested[index + 1]
            if index % 2 == 0 and index + 1 < len(requested)
            else requested[index - 1]
            if index > 0
            else expected
        )
        eps = min(
            abs(1.0e-3 * tolerance * expected),
            abs(expected - neighbor) * tolerance,
        )
        candidates = first_bounds if index % 2 == 0 else second_bounds
        if not np.any(np.abs(candidates - expected) <= eps):
            raise AxisValidationError(
                f"requested bounds value {expected:g} for axis {name!r} "
                "was not found."
            )


def _validate_valid_range(
    axis: Axis, values: np.ndarray, name: str, *, is_bounds: bool
) -> None:
    if _is_longitude(axis):
        return
    valid_min = _numeric_or_none(axis.get("valid_min"))
    valid_max = _numeric_or_none(axis.get("valid_max"))
    flat = values.reshape(-1)
    if valid_min is not None:
        eps = abs(1.0e-6 * valid_min)
        bad = flat[flat < valid_min - eps]
        if bad.size:
            target = "bounds" if is_bounds else "value"
            raise AxisValidationError(
                f"axis {name!r} detected {target} {bad[0]:g} when "
                f"valid_min is {valid_min:g}."
            )
    if valid_max is not None:
        eps = abs(1.0e-6 * valid_max)
        bad = flat[flat > valid_max + eps]
        if bad.size:
            target = "bounds" if is_bounds else "value"
            raise AxisValidationError(
                f"axis {name!r} detected {target} {bad[0]:g} when "
                f"valid_max is {valid_max:g}."
            )


def _validate_monotonic(
    axis: Axis, values: np.ndarray, name: str, *, is_bounds: bool
) -> None:
    if values.ndim != 1 and not is_bounds:
        return
    climatology = _is_truthy(axis.get("climatology"))
    if is_bounds:
        if values.shape[-1] < 2:
            return
        pairs = values.reshape(-1, values.shape[-1])
        starts = pairs[:, 0]
        if starts.size >= 3 and not _strictly_monotonic(starts):
            message = f"axis {name!r} has non-monotonic bounds values."
            if climatology:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
            else:
                raise AxisValidationError(message)
        if climatology:
            return
        if pairs.shape[0] >= 2:
            ends = pairs[:, 1]
            deltas = starts[1:] - ends[:-1]
            overlap = deltas * _direction(starts) < -1.0e-12
            if np.any(overlap):
                index = int(np.nonzero(overlap)[0][0])
                raise AxisValidationError(
                    f"axis {name!r} has overlapping bounds values at "
                    f"index {index}."
                )
            gaps = np.abs(deltas) > 1.0e-12
            if np.any(gaps):
                index = int(np.nonzero(gaps)[0][0])
                warnings.warn(
                    f"axis {name!r} has bounds values that leave gaps at "
                    f"index {index}.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        return
    flat = values.reshape(-1)
    if flat.size >= 3 and not _strictly_monotonic(flat):
        raise AxisValidationError(f"axis {name!r} has non-monotonic values.")


def _validate_values_inside_bounds(
    values: np.ndarray, bounds: np.ndarray, name: str
) -> None:
    if not _is_numeric(values):
        return
    pairs = bounds.reshape(-1, bounds.shape[-1])
    flat_values = values.reshape(-1)
    if pairs.shape[0] != flat_values.size:
        return
    lower = np.minimum(pairs[:, 0], pairs[:, 1])
    upper = np.maximum(pairs[:, 0], pairs[:, 1])
    outside = (flat_values < lower) | (flat_values > upper)
    if np.any(outside):
        index = int(np.nonzero(outside)[0][0])
        raise AxisValidationError(
            f"axis {name!r} has value {flat_values[index]:g} not within "
            f"bounds {pairs[index, 0]:g}, {pairs[index, 1]:g} at "
            f"index {index}."
        )


def _time_values_from_bounds(
    values: np.ndarray, bounds: np.ndarray, name: str
) -> np.ndarray:
    pairs = bounds.reshape(-1, bounds.shape[-1])
    if values.size != pairs.shape[0]:
        return values
    midpoints = (pairs[:, 0] + pairs[:, 1]) / 2.0
    reshaped_midpoints = midpoints.reshape(values.shape)
    differences = np.abs(values.reshape(-1) - midpoints)
    if np.any(differences > 1.0e-6):
        index = int(np.nonzero(differences > 1.0e-6)[0][0])
        warnings.warn(
            f"The values provided for axis {name} differ from values "
            "computed from bounds; using bound midpoints instead. "
            f"First mismatch at index {index}: "
            f"{values.reshape(-1)[index]:.6f} will be replaced with "
            f"{midpoints[index]:.6f} between bounds "
            f"{pairs[index, 0]:.6f} and {pairs[index, 1]:.6f}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return reshaped_midpoints


def _validate_time_interval(
    dataset: Mapping[str, Any],
    variable: Mapping[str, Any],
    axis: Axis,
    values: np.ndarray,
) -> None:
    flat = values.reshape(-1)
    if flat.size < 2:
        return
    spec = _interval_spec(dataset, variable)
    if spec is None or spec.days <= 0:
        return
    units = str(axis.get("units", "days since ?"))
    calendar = str(axis.get("calendar", dataset.get("calendar", "standard")))
    interval_days = _time_interval_days(flat, units, calendar)
    if interval_days.size == 0:
        return
    differences = np.abs(interval_days - spec.days) / spec.days
    bad_errors = differences > spec.error
    bad_warnings = differences > spec.warning
    if not np.any(bad_errors | bad_warnings):
        return
    index = int(np.nonzero(bad_errors | bad_warnings)[0][0])
    frequency = str(dataset.get("frequency", variable.get("frequency", "")))
    message = (
        f"Time interval mismatch detected for frequency: {frequency!r}. "
        f"Expected interval between time axis values: {spec.days:g} days. "
        f"Actual interval between time axis values {index} and "
        f"{index + 1}: {interval_days[index]:g} days "
        f"({differences[index] * 100.0:.1f}% difference)."
    )
    if bad_errors[index]:
        raise AxisValidationError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _time_interval_days(
    values: np.ndarray, units: str, calendar: str
) -> np.ndarray:
    cftime_intervals = cftime_interval_days(values, units, calendar)
    if cftime_intervals is not None:
        return cftime_intervals
    interval_values = np.diff(values)
    return np.abs(interval_values) * _time_unit_days(units)


def _interval_spec(
    dataset: Mapping[str, Any], variable: Mapping[str, Any]
) -> _IntervalSpec | None:
    frequency = str(dataset.get("frequency", variable.get("frequency", "")))
    if not frequency:
        return None
    project = getattr(dataset, "project", None)
    cv_frequency = getattr(project, "cv", {}).get("frequency", {})
    if isinstance(cv_frequency, Mapping):
        entry = cv_frequency.get(frequency)
        if isinstance(entry, Mapping):
            value = _numeric_or_none(entry.get("approx_interval"))
            if value is not None:
                return _IntervalSpec(
                    value,
                    _numeric_or_none(entry.get("approx_interval_warning"))
                    or DEFAULT_INTERVAL_WARNING,
                    _numeric_or_none(entry.get("approx_interval_error"))
                    or DEFAULT_INTERVAL_ERROR,
                )
    value = DEFAULT_FREQUENCY_INTERVALS.get(frequency.lower())
    return _IntervalSpec(value) if value is not None else None


def _normalize_longitude(
    axis: Axis, values: np.ndarray, bounds: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    valid_min = _numeric_or_none(axis.get("valid_min"))
    valid_max = _numeric_or_none(axis.get("valid_max"))
    if valid_min is None or valid_max is None:
        return values, bounds
    span = valid_max - valid_min
    if span <= 0:
        return values, bounds
    adjusted = values.copy()
    while np.any(adjusted < valid_min):
        adjusted = np.where(adjusted < valid_min, adjusted + span, adjusted)
    while np.any(adjusted > valid_max):
        adjusted = np.where(adjusted > valid_max, adjusted - span, adjusted)
    if bounds is not None:
        bounds = bounds.copy()
        shift = adjusted.reshape(-1) - values.reshape(-1)
        if bounds.shape[-1] >= 2 and shift.size == bounds.reshape(
            -1, bounds.shape[-1]
        ).shape[0]:
            pairs = bounds.reshape(-1, bounds.shape[-1])
            pairs[:, :2] = pairs[:, :2] + shift[:, None]
            bounds = pairs.reshape(bounds.shape)
    return adjusted, bounds


def _time_unit_days(units: str) -> float:
    match = re.match(r"^\s*([A-Za-z_]+)", units)
    unit = match.group(1).lower() if match else "days"
    return {
        "day": 1.0,
        "days": 1.0,
        "hour": 1.0 / 24.0,
        "hours": 1.0 / 24.0,
        "hr": 1.0 / 24.0,
        "hrs": 1.0 / 24.0,
        "minute": 1.0 / 1440.0,
        "minutes": 1.0 / 1440.0,
        "second": 1.0 / 86400.0,
        "seconds": 1.0 / 86400.0,
        "month": 30.0,
        "months": 30.0,
        "year": 365.0,
        "years": 365.0,
    }.get(unit, 1.0)


def _requires_bounds(axis: Axis) -> bool:
    return _is_truthy(axis.get("must_have_bounds", ""))


def _is_time_axis(axis: Axis) -> bool:
    return str(axis.get("axis", "")).upper() == "T" or (
        str(axis.get("standard_name", "")).lower() == "time"
    )


def _is_longitude(axis: Axis) -> bool:
    units = str(axis.get("units", "")).lower()
    return str(axis.get("axis", "")).upper() == "X" and (
        units.startswith("degree") and units != "degrees"
    )


def _strictly_monotonic(values: np.ndarray) -> bool:
    diffs = np.diff(values)
    return bool(np.all(diffs > 0.0) or np.all(diffs < 0.0))


def _direction(values: np.ndarray) -> float:
    if values.size < 2:
        return 1.0
    return 1.0 if values[-1] >= values[0] else -1.0


def _is_numeric(values: np.ndarray) -> bool:
    return values.dtype.kind in {"i", "u", "f"}


def _is_truthy(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _tolerance(axis: Axis) -> float:
    value = _numeric_or_none(axis.get("tolerance"))
    return value if value is not None else 1.0


def _numeric_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_list(value: Any) -> list[float]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        values: Sequence[Any] = value.split()
    elif isinstance(value, Sequence):
        values = value
    else:
        values = (value,)
    parsed: list[float] = []
    for item in values:
        number = _numeric_or_none(item)
        if number is not None:
            parsed.append(number)
    return parsed


def _array_to_user_value(array: np.ndarray) -> Any:
    if array.shape == ():
        return array.item()
    return array.tolist()
