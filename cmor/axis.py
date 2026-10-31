"""Axis metadata and coordinate values, bounds, ordering, and calendar checks."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, Literal, Sequence

import cftime
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ._metadata import AxisStr, BoolCoerced, CoercedF, MetadataModel, StrSeq
from ._metadata import numeric_or_none as _numeric_or_none
from ._time import cftime_interval_days as _cftime_interval_days
from .exceptions import AxisValidationError

if TYPE_CHECKING:
    from ._metadata import DatasetMetadata
    from .variable import Variable


class Axis(MetadataModel):
    """Metadata and coordinate values for one data axis.

    Construct via :meth:`ProjectTables.axis` to merge authoritative
    coordinate-table metadata::

        axis = project.axis("latitude", values=np.linspace(-90, 90, 180))

    Construct directly when project tables are not available::

        axis = Axis(name="latitude", values=np.linspace(-90, 90, 180),
                    units="degrees_north", axis="Y")

    Parameters
    ----------
    name:
        Logical axis name matching a variable dimension.
    values:
        Coordinate values (list, numpy array, scalar).
    bounds:
        Optional coordinate cell bounds.
    dimensions:
        Underlying dimensions for auxiliary coordinates.
    units:
        CF units string.
    standard_name, long_name:
        CF / NetCDF attributes.
    axis:
        CF axis designator — ``"X"``, ``"Y"``, ``"Z"``, or ``"T"``.
        Lower-case inputs are automatically upper-cased.
    positive:
        CF ``"up"`` or ``"down"`` for vertical axes.
    formula:
        Formula for computing coordinate values.
    valid_min, valid_max:
        Valid coordinate value range.
    out_name:
        Output coordinate variable name.
    table_entry, axis_entry, coordinate:
        Coordinate-table entry name selectors, tried in order.
    grid_table_entry, grid_coordinate:
        Grid coordinate-table entry selectors.
    scalar:
        Write as a scalar coordinate.
    auxiliary:
        Write as an auxiliary coordinate.
    auxiliary_name:
        Output name for the auxiliary coordinate variable.
    auxiliary_attrs:
        Extra attributes for the auxiliary variable.
    climatology:
        Climatology bounds control.
    generic_level_name:
        Generic level selector.
    z_factors, z_bounds_factors:
        Formula-term names for this axis.
    bounds_name, bounds_dim:
        Output bounds variable name / dimension name.
    bounds_attrs:
        Extra attributes for the bounds variable.
    attrs:
        Extra attributes for the coordinate variable.
    stored_direction:
        Expected ordering of coordinate values: ``"increasing"`` or
        ``"decreasing"``.
    """

    name: str
    values: Any = None
    bounds: Any = None
    dimensions: StrSeq = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    axis: AxisStr = None
    positive: Literal["up", "down"] | None = None
    formula: str | None = None
    valid_min: CoercedF = None
    valid_max: CoercedF = None
    out_name: str | None = None
    table_entry: str | None = None
    axis_entry: str | None = None
    coordinate: str | None = None
    grid_table_entry: str | None = None
    grid_coordinate: str | None = None
    scalar: bool | None = None
    auxiliary: bool | None = None
    auxiliary_name: str | None = None
    auxiliary_attrs: dict[str, Any] = Field(default_factory=dict)
    climatology: BoolCoerced = None
    calendar: str | None = None
    generic_level_name: str | None = None
    z_factors: str | None = None
    z_bounds_factors: str | None = None
    requested: Any = None
    requested_bounds: Any = None
    bounds_values: Any = None
    must_have_bounds: BoolCoerced = None
    stored_direction: str | None = None
    tolerance: CoercedF = None
    bounds_name: str | None = None
    bounds_dim: str | None = None
    bounds_attrs: dict[str, Any] = Field(default_factory=dict)
    attrs: dict[str, Any] = Field(default_factory=dict)
    isgridaxis: bool = False

    # ------------------------------------------------------------------
    # NetCDF output helpers
    # ------------------------------------------------------------------

    def attributes(self, *, include_units: bool = True) -> dict[str, Any]:
        """Return NetCDF attributes for this coordinate variable."""
        attrs = self.netcdf_attrs(self.attrs)
        if include_units and self.units is not None:
            attrs["units"] = self.units
        for key, val in (
            ("standard_name", self.standard_name),
            ("long_name", self.long_name),
            ("axis", self.axis),
            ("positive", self.positive),
            ("formula", self.formula),
            ("calendar", self.calendar),
        ):
            if val is not None:
                attrs[key] = val
        return attrs

    def auxiliary_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the auxiliary coordinate variable."""
        return self.netcdf_attrs(self.auxiliary_attrs)

    def bounds_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the bounds variable."""
        return self.netcdf_attrs(self.bounds_attrs)

    def values_array(self) -> np.ndarray:
        """Return coordinate values as a NetCDF-ready numpy array."""
        return self.netcdf_array(self.values if self.values is not None else [])

    def bounds_array(self) -> np.ndarray:
        """Return coordinate bounds as a NetCDF-ready numpy array."""
        return self.netcdf_array(self.bounds)


_DEFAULT_INTERVAL_WARNING = 0.1
_DEFAULT_INTERVAL_ERROR = 0.2

_DEFAULT_FREQUENCY_INTERVALS = {
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


_MIP_INAPPROPRIATE_CALENDARS: frozenset[str] = frozenset({"tai", "all_leap", "366_day"})


class _IntervalSpec(BaseModel):
    """Expected time interval for one frequency code.

    Pydantic enforces the invariants that ``days`` is non-negative and that
    the fractional thresholds are in (0, 1] with ``warning ≤ error``.
    """

    model_config = ConfigDict(frozen=True)

    days: float = Field(ge=0.0)
    warning: float = Field(default=_DEFAULT_INTERVAL_WARNING, gt=0.0, le=1.0)
    error: float = Field(default=_DEFAULT_INTERVAL_ERROR, gt=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        if self.warning > self.error:
            raise ValueError(
                f"_IntervalSpec warning={self.warning} must be ≤ error={self.error}"
            )


def _validate_and_normalize_axes(
    dataset: DatasetMetadata,
    variable: Variable,
    axes: Sequence[Axis],
) -> tuple[Axis, ...]:
    """Return axes after CMOR-style coordinate validation."""
    return tuple(_validate_and_normalize_axis(dataset, variable, axis) for axis in axes)


def _validate_axes(
    dataset: DatasetMetadata | None,
    variable: Variable,
    axes: Sequence[Axis],
) -> None:
    """Validate axis values with dataset and frequency-dependent checks."""
    for axis in axes:
        _validate_and_normalize_axis(
            dataset,
            variable,
            axis,
            enforce_required_bounds=True,
            normalize=False,
        )


def _validate_axis_values_early(axis: Axis) -> None:
    """Validate generic axis values before dataset-level context is available."""
    _validate_and_normalize_axis(
        None,
        None,
        axis,
        defer_time_value_checks=True,
        enforce_required_bounds=False,
        normalize=False,
    )


def _validate_and_normalize_axis(
    dataset: DatasetMetadata | None,
    variable: Variable | None,
    axis: Axis,
    *,
    defer_time_value_checks: bool = False,
    enforce_required_bounds: bool = True,
    normalize: bool = True,
) -> Axis:
    values = axis.values_array()
    bounds = axis.bounds_array() if axis.bounds is not None else None
    name = str(axis.table_entry or axis.name)
    climatology = bool(axis.climatology)
    defer_time_value_checks = defer_time_value_checks and _is_time_axis(axis)

    values, bounds = _normalize_bounds_shape(axis, values, bounds)
    if enforce_required_bounds and bool(axis.must_have_bounds) and bounds is None:
        raise AxisValidationError(
            f"axis {name!r} must have bounds, but none were provided."
        )

    if values.dtype.kind in {"i", "u", "f"}:
        values = values.astype("f8", copy=True)
        if bounds is not None and bounds.dtype.kind in {"i", "u", "f"}:
            bounds = bounds.astype("f8", copy=True)
        if _is_longitude(axis):
            values, bounds = _normalize_longitude(axis, values, bounds)
        _validate_requested_values(axis, values, name)
        _validate_valid_range(axis, values, name, is_bounds=False)
        _validate_stored_direction(axis, values, name)
        _validate_monotonic(axis, values, name, is_bounds=False)

    if bounds is not None and bounds.dtype.kind in {"i", "u", "f"}:
        _validate_valid_range(axis, bounds, name, is_bounds=True)
        if bounds.shape[-1] == 2:
            _validate_requested_bounds(axis, bounds, name)
            if not defer_time_value_checks:
                _validate_monotonic(axis, bounds, name, is_bounds=True)
                _validate_values_inside_bounds(values, bounds, name)
        if (
            not defer_time_value_checks
            and _is_time_axis(axis)
            and not climatology
            and bounds.shape[-1] == 2
        ):
            values = _time_values_from_bounds(values, bounds, name)

    if not defer_time_value_checks and _is_time_axis(axis) and not climatology:
        _validate_time_interval(dataset, variable, axis, values)

    if not normalize:
        return axis

    updates: dict[str, Any] = {}
    if not np.array_equal(values, axis.values_array()):
        updates["values"] = values.item() if values.shape == () else values.tolist()
    if bounds is not None and axis.bounds is not None:
        if not np.array_equal(bounds, axis.bounds_array()):
            updates["bounds"] = bounds.item() if bounds.shape == () else bounds.tolist()
    return axis.updated(**updates) if updates else axis


def _normalize_bounds_shape(
    axis: Axis, values: np.ndarray, bounds: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    if bounds is None:
        return values, bounds
    values_shape = values.shape
    if bool(axis.scalar) and values.size == 1:
        if bounds.size == 2:
            return values, bounds.reshape(2)
        raise AxisValidationError("Scalar coordinate bounds must have 2 values.")
    if bounds.ndim == 1 and values.ndim == 1 and bounds.size == values.size + 1:
        pairs = np.stack((bounds[:-1], bounds[1:]), axis=-1)
        return values, pairs
    if bounds.shape[:-1] == values_shape and bounds.shape[-1] >= 2:
        return values, bounds
    raise AxisValidationError(
        f"axis {axis.name!r} bounds shape {bounds.shape!r} does not "
        f"match coordinate value shape {values_shape!r}."
    )


def _validate_requested_values(axis: Axis, values: np.ndarray, name: str) -> None:
    requested = _numeric_list(axis.requested)
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
                f"requested value {expected:g} for axis {name!r} was not found."
            )


def _validate_requested_bounds(axis: Axis, bounds: np.ndarray, name: str) -> None:
    requested = _numeric_list(axis.requested_bounds or axis.bounds_values)
    if not requested:
        requested = _numeric_list(axis.bounds_values)
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
                f"requested bounds value {expected:g} for axis {name!r} was not found."
            )


def _validate_valid_range(
    axis: Axis, values: np.ndarray, name: str, *, is_bounds: bool
) -> None:
    if _is_longitude(axis):
        return
    valid_min = _numeric_or_none(axis.valid_min)
    valid_max = _numeric_or_none(axis.valid_max)
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
    climatology = bool(axis.climatology)
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
            direction = 1.0 if starts.size < 2 or starts[-1] >= starts[0] else -1.0
            overlap = deltas * direction < -1.0e-12
            if np.any(overlap):
                index = int(np.nonzero(overlap)[0][0])
                raise AxisValidationError(
                    f"axis {name!r} has overlapping bounds values at index {index}."
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
    if values.dtype.kind not in {"i", "u", "f"}:
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
    dataset: DatasetMetadata | None,
    variable: Variable | None,
    axis: Axis,
    values: np.ndarray,
) -> None:
    var_freq = (
        str(getattr(variable, "frequency", "") or "") if variable is not None else ""
    )
    frequency = str(dataset.frequency or var_freq) if dataset is not None else var_freq
    if not frequency:
        if dataset is not None:
            raise AxisValidationError(
                "No frequency attribute provided in the dataset configuration. "
                "A 'frequency' value is required when a time axis is present. "
                "Set frequency in the dataset metadata or ensure the variable "
                "table entry defines a frequency."
            )
        return
    flat = values.reshape(-1)
    if flat.size < 2:
        return
    spec = _interval_spec(dataset, variable)
    if spec is None or spec.days <= 0:
        return
    # calendar may be stored in axis.attrs (user-supplied) or in dataset
    units = str(axis.units or "days since ?")
    calendar = str(
        axis.attrs.get("calendar")
        or (dataset.calendar if dataset is not None else None)
        or "standard"
    )
    interval_days = _time_interval_days(flat, units, calendar)
    if interval_days.size == 0:
        return
    differences = np.abs(interval_days - spec.days) / spec.days
    bad_errors = differences > spec.error
    bad_warnings = differences > spec.warning
    if not np.any(bad_errors | bad_warnings):
        return
    index = int(np.nonzero(bad_errors | bad_warnings)[0][0])
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


def _time_interval_days(values: np.ndarray, units: str, calendar: str) -> np.ndarray:
    intervals = _cftime_interval_days(values, units, calendar)
    return (
        intervals
        if intervals is not None
        else np.abs(np.diff(values)) * _time_unit_days(units)
    )


def _interval_spec(
    dataset: DatasetMetadata | None, variable: Variable | None
) -> _IntervalSpec | None:
    var_freq = (
        str(getattr(variable, "frequency", "") or "") if variable is not None else ""
    )
    frequency = str(dataset.frequency or var_freq) if dataset is not None else var_freq
    if not frequency:
        return None
    project = getattr(dataset, "project", None)
    cv = getattr(project, "cv", None)
    frequency_component = getattr(cv, "frequency", None)
    if frequency_component is not None:
        entry = frequency_component.interval_for(frequency)
        if entry is not None and entry.approx_interval is not None:
            return _IntervalSpec(
                days=entry.approx_interval,
                warning=entry.approx_interval_warning or _DEFAULT_INTERVAL_WARNING,
                error=entry.approx_interval_error or _DEFAULT_INTERVAL_ERROR,
            )
    value = _DEFAULT_FREQUENCY_INTERVALS.get(frequency.lower())
    return _IntervalSpec(days=value) if value is not None else None


def _normalize_longitude(
    axis: Axis, values: np.ndarray, bounds: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    valid_min = _numeric_or_none(axis.valid_min)
    valid_max = _numeric_or_none(axis.valid_max)
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
        if (
            bounds.shape[-1] >= 2
            and shift.size == bounds.reshape(-1, bounds.shape[-1]).shape[0]
        ):
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


def _validate_stored_direction(axis: Axis, values: np.ndarray, name: str) -> None:
    if _is_longitude(axis):
        return
    direction = str(axis.stored_direction or "").lower().strip()
    if direction not in ("increasing", "decreasing"):
        return
    flat = values.reshape(-1)
    if flat.size < 2:
        return
    actual_increasing = flat[-1] > flat[0]
    if direction == "increasing" and not actual_increasing:
        raise AxisValidationError(
            f"axis {name!r} has stored_direction='increasing' but values "
            f"run from {flat[0]:g} to {flat[-1]:g} (decreasing)."
        )
    if direction == "decreasing" and actual_increasing:
        raise AxisValidationError(
            f"axis {name!r} has stored_direction='decreasing' but values "
            f"run from {flat[0]:g} to {flat[-1]:g} (increasing)."
        )


def _validate_calendar(dataset: DatasetMetadata) -> None:
    """Validate the calendar declared in the dataset metadata."""
    calendar = str(dataset.calendar or "").strip()
    if not calendar:
        return
    try:
        cftime.datetime(2000, 1, 1, calendar=calendar)
    except ValueError:
        raise AxisValidationError(
            f"calendar={calendar!r} is not a recognised CF calendar. "
            "Valid calendars include 'standard', 'gregorian', "
            "'proleptic_gregorian', 'noleap', '365_day', '360_day', "
            "'julian', 'all_leap', and '366_day'."
        )
    except Exception:
        return
    if calendar.lower() in _MIP_INAPPROPRIATE_CALENDARS:
        warnings.warn(
            f"calendar={calendar!r} is not appropriate for MIP data. "
            "Please use a more common climate-study calendar such as "
            "'standard', 'gregorian', 'proleptic_gregorian', 'noleap', "
            "'365_day', '360_day', or 'julian'.",
            RuntimeWarning,
            stacklevel=4,
        )


def _is_time_axis(axis: Axis) -> bool:
    return str(axis.axis or "").upper() == "T" or (
        str(axis.standard_name or "").lower() == "time"
    )


def _is_longitude(axis: Axis) -> bool:
    units = str(axis.units or "").lower()
    return str(axis.axis or "").upper() == "X" and (
        units.startswith("degree") and units != "degrees"
    )


def _strictly_monotonic(values: np.ndarray) -> bool:
    diffs = np.diff(values)
    return bool(np.all(diffs > 0.0) or np.all(diffs < 0.0))


def _tolerance(axis: Axis) -> float:
    value = _numeric_or_none(axis.tolerance)
    return value if value is not None else 1.0


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
