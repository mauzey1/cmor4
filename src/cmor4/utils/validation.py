from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union
import re
import warnings

import cftime
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import xarray as xr

from .time_utils import cftime_interval_days
from ..axis import Axis
from ..datasetinfo import DatasetInfo
from ..exceptions import AxisValidationError, VariableValidationError
from ..grid import Grid
from ..variable import Variable
from ..zfactor import ZFactor

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


class _IntervalSpec(BaseModel):
    """Expected time interval for one frequency code.

    Pydantic enforces the invariants that ``days`` is non-negative and that
    the fractional thresholds are in (0, 1] with ``warning ≤ error``.
    """

    model_config = ConfigDict(frozen=True)

    days: float = Field(ge=0.0)
    warning: float = Field(default=DEFAULT_INTERVAL_WARNING, gt=0.0, le=1.0)
    error: float = Field(default=DEFAULT_INTERVAL_ERROR, gt=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        if self.warning > self.error:
            raise ValueError(
                f"_IntervalSpec warning={self.warning} must be ≤ error={self.error}"
            )


@dataclass(frozen=True)
class ValidationContext:
    """Validated metadata and resolved dimension state for dataset creation."""

    dataset: DatasetInfo
    variable: Variable
    axes: tuple[Axis, ...]
    zfactors: tuple[ZFactor, ...]
    grid: Grid | None
    axis_dims: dict[str, tuple[str, ...]]
    dim_names: tuple[str, ...]
    dims: tuple[str, ...]


def validate_metadata(
    dataset: DatasetInfo,
    variable: Variable,
    axes: Sequence[Axis],
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
) -> ValidationContext:
    """Validate project metadata and resolve dimensions without data values."""

    dataset, variable = _dataset_for_variable(dataset, variable)
    axes = _dataset_axes(dataset, axes, variable)
    axes = validate_and_normalize_axes(dataset, variable, axes)
    axes = _merge_grid_axes(axes, grid)
    axis_dims = build_axis_dimension_map(axes)

    if grid is not None:
        dim_names = grid.variable_dimensions(variable)
    else:
        dim_names = None
    if dim_names is None:
        if variable.dimensions is not None:
            dim_names = tuple(str(name) for name in variable.dimensions)
        else:
            dim_names = tuple(axis.name for axis in axes if not bool(axis.auxiliary))
    else:
        dim_names = tuple(str(name) for name in dim_names)

    dims = tuple(dim for name in dim_names for dim in axis_dims.get(name, ()))
    ctx = ValidationContext(
        dataset=dataset,
        variable=variable,
        axes=tuple(axes),
        zfactors=tuple(zfactors or ()),
        grid=grid,
        axis_dims=axis_dims,
        dim_names=dim_names,
        dims=dims,
    )
    validate_zfactor_values(ctx)
    return ctx


def validate_data_chunk(ctx: ValidationContext, data: Any) -> np.ndarray:
    """Validate a complete data array or write chunk against resolved metadata."""

    data_array = np.asarray(data)
    var_name = ctx.variable.names()[0]
    if data_array.ndim != len(ctx.dims):
        expected = " x ".join(ctx.dims) if ctx.dims else "scalar"
        raise ValueError(
            f"Data for {var_name!r} has {data_array.ndim} dimensions, "
            f"but variable dimensions resolve to {expected!r}."
        )
    validate_variable_values(ctx.variable, ctx.axes, data, ctx.dims, ctx.axis_dims)
    return data_array


def validate_zfactor_values(ctx: ValidationContext) -> None:
    """Validate formula-term values against resolved axis dimensions."""

    for zfactor in ctx.zfactors:
        out_name = str(zfactor.out_name or zfactor.name)
        values = zfactor.values_array()
        dims = named_dimensions(zfactor.dimensions or (), ctx.axis_dims)
        if not dims and values.size == 1:
            values = values.reshape(())
        validate_variable_values(
            zfactor,
            ctx.axes,
            values,
            dims,
            ctx.axis_dims,
            name=out_name,
            table_id=str(zfactor.table_entry or "formula_terms"),
        )
        if zfactor.bounds is not None:
            bounds_name = str(zfactor.bounds_name or f"{out_name}_bnds")
            bounds_dims = dims + (str(zfactor.bounds_dim or "bnds"),)
            validate_variable_values(
                zfactor,
                ctx.axes,
                zfactor.bounds_array(),
                bounds_dims,
                ctx.axis_dims,
                name=bounds_name,
                table_id=str(zfactor.table_entry or "formula_terms"),
            )


def validate_final_dataset(
    ds: xr.Dataset,
    ctx: ValidationContext,
    zfactor_names: Sequence[str],
) -> None:
    """Validate final xarray dataset structure and project-level requirements."""

    project = ctx.dataset.project
    if project is not None:
        project.validate_global_attributes(ds.attrs)
        project.validate_dataset(
            ctx.dataset,
            ctx.variable,
            ctx.axes,
            grid=ctx.grid,
            zfactors=ctx.zfactors,
        )

    var_name = ctx.variable.names()[0]
    if var_name not in ds.data_vars:
        raise ValueError(f"Variable {var_name!r} was not created.")
    if tuple(ds[var_name].dims) != tuple(ctx.dims):
        raise ValueError(
            f"Variable {var_name!r} dimensions {tuple(ds[var_name].dims)!r} "
            f"do not match expected dimensions {tuple(ctx.dims)!r}."
        )

    for axis in ctx.axes:
        _validate_final_axis(ds, axis)

    if ctx.grid is not None and ctx.grid.has_mapping:
        if ctx.grid.variable_name not in ds.data_vars:
            raise ValueError(
                f"Grid mapping variable {ctx.grid.variable_name!r} was not created."
            )
        if ds[var_name].attrs.get("grid_mapping") != ctx.grid.variable_name:
            raise ValueError(
                f"Variable {var_name!r} does not reference grid mapping "
                f"{ctx.grid.variable_name!r}."
            )

    if ctx.grid is not None:
        _validate_grid_coords(ds, ctx.grid)

    for zfactor, out_name in zip(ctx.zfactors, zfactor_names):
        _validate_final_zfactor(ds, zfactor, out_name)


def _dataset_for_variable(
    dataset: DatasetInfo,
    variable: Variable,
) -> tuple[DatasetInfo, Variable]:
    """Return dataset and variable prepared for a specific variable."""

    project = dataset.project
    if project is None:
        return dataset, variable
    return project._dataset_for_variable(dataset, variable)


def _dataset_axes(
    dataset: DatasetInfo,
    axes: Sequence[Axis],
    variable: Variable,
) -> tuple[Axis, ...]:
    project = dataset.project
    if project is None:
        return tuple(axes)
    return project._axes(axes, variable)


def _merge_grid_axes(axes: Sequence[Axis], grid: Grid | None) -> tuple[Axis, ...]:
    if grid is None or not grid.axes:
        return tuple(axes)
    existing_names = {str(axis.out_name or axis.name) for axis in axes}
    extra = [
        axis
        for axis in grid.axes
        if str(axis.out_name or axis.name) not in existing_names
    ]
    return tuple(axes) + tuple(extra)


def build_axis_dimension_map(axes: Sequence[Axis]) -> dict[str, tuple[str, ...]]:
    """Resolve logical axis names to physical xarray dimension names."""

    axis_dims: dict[str, tuple[str, ...]] = {}
    for axis in axes:
        name = axis.name
        out_name = str(axis.out_name or axis.name)
        values = axis.values_array()
        if bool(axis.scalar):
            if values.shape == ():
                pass
            elif values.size != 1:
                raise ValueError("Scalar coordinates must contain exactly one value.")
            axis_dims[name] = ()
            add_axis_dim_aliases(axis, axis_dims, ())
        elif axis.auxiliary_name:
            axis_dims[name] = (out_name,)
            add_axis_dim_aliases(axis, axis_dims, (out_name,))
        else:
            dims = (
                named_dimensions(axis.dimensions, axis_dims)
                if axis.dimensions is not None
                else (out_name,)
            )
            if len(dims) == 1:
                axis_dims[name] = dims
                add_axis_dim_aliases(axis, axis_dims, dims)
            if not (bool(axis.auxiliary) or len(dims) > 1):
                axis_dims.setdefault(out_name, dims)
    return axis_dims


def add_axis_dim_aliases(
    axis: Axis,
    axis_dims: dict[str, tuple[str, ...]],
    dims: tuple[str, ...],
) -> None:
    for key, value in (
        ("table_entry", axis.table_entry),
        ("generic_level_name", axis.generic_level_name),
        ("out_name", axis.out_name),
    ):
        if value:
            axis_dims.setdefault(str(value), dims)


def named_dimensions(
    names: Sequence[Any], axis_dims: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    dims: list[str] = []
    for name in names:
        text = str(name)
        resolved = axis_dims.get(text)
        if resolved:
            dims.extend(resolved)
        else:
            dims.append(text)
    return tuple(dims)


def validate_and_normalize_axes(
    dataset: DatasetInfo,
    variable: Variable,
    axes: Sequence[Axis],
) -> tuple[Axis, ...]:
    """Return axes after CMOR-style coordinate validation."""
    return tuple(_validate_and_normalize_axis(dataset, variable, axis) for axis in axes)


def validate_axes(
    dataset: DatasetInfo | None,
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


def validate_axis_values_early(axis: Axis) -> None:
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
    dataset: DatasetInfo | None,
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
            overlap = deltas * _direction(starts) < -1.0e-12
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
    dataset: DatasetInfo | None,
    variable: Variable | None,
    axis: Axis,
    values: np.ndarray,
) -> None:
    var_freq = (
        str(getattr(variable, "frequency", "") or "") if variable is not None else ""
    )
    frequency = (
        str(dataset.get("frequency", var_freq)) if dataset is not None else var_freq
    )
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
        or (dataset.get("calendar", "standard") if dataset is not None else "standard")
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
    cftime_intervals = cftime_interval_days(values, units, calendar)
    if cftime_intervals is not None:
        return cftime_intervals
    interval_values = np.diff(values)
    return np.abs(interval_values) * _time_unit_days(units)


def _interval_spec(
    dataset: DatasetInfo | None, variable: Variable | None
) -> _IntervalSpec | None:
    var_freq = (
        str(getattr(variable, "frequency", "") or "") if variable is not None else ""
    )
    frequency = (
        str(dataset.get("frequency", var_freq)) if dataset is not None else var_freq
    )
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
                    days=value,
                    warning=_numeric_or_none(entry.get("approx_interval_warning"))
                    or DEFAULT_INTERVAL_WARNING,
                    error=_numeric_or_none(entry.get("approx_interval_error"))
                    or DEFAULT_INTERVAL_ERROR,
                )
    value = DEFAULT_FREQUENCY_INTERVALS.get(frequency.lower())
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


_MIP_INAPPROPRIATE_CALENDARS: frozenset[str] = frozenset({"all_leap", "366_day"})


def _validate_calendar(dataset: DatasetInfo) -> None:
    """Validate the calendar declared in the dataset metadata."""
    calendar = str(dataset.get("calendar", "") or "").strip()
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


def _direction(values: np.ndarray) -> float:
    if values.size < 2:
        return 1.0
    return 1.0 if values[-1] >= values[0] else -1.0


def _tolerance(axis: Axis) -> float:
    value = _numeric_or_none(axis.tolerance)
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


def validate_variable_values(
    variable: Union[Variable, ZFactor],
    axes: Sequence[Axis],
    data: Any,
    dims: Sequence[str],
    axis_dims: Mapping[str, tuple[str, ...]],
    *,
    name: str | None = None,
    table_id: str | None = None,
) -> None:
    """Apply CMOR-style checks to data variable and formula-term values."""

    values = _as_float_masked_array(data)
    if values is None:
        return

    valid_mask = ~np.ma.getmaskarray(values)
    missing_value = getattr(variable, "missing_value", None) or getattr(
        variable, "fill_value", None
    )
    if missing_value is not None:
        try:
            valid_mask &= ~np.isclose(
                values.filled(np.nan),
                float(missing_value),
                rtol=float(variable.extra.get("tolerance", 1.0e-6)),
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
    variable: Union[Variable, ZFactor],
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
    limit = _numeric_or_none(getattr(variable, key, None))
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
    variable: Union[Variable, ZFactor],
    active: np.ndarray,
    *,
    name: str | None,
    table_id: str | None,
) -> None:
    mean_abs = float(np.mean(np.abs(active)))
    ok_min = _numeric_or_none(variable.ok_min_mean_abs)
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

    ok_max = _numeric_or_none(variable.ok_max_mean_abs)
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
        name = axis.name
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


def _variable_name(variable: Any, name: str | None) -> str:
    if name is not None:
        return name
    names = getattr(variable, "names", None)
    if callable(names):
        return names()[0]
    return str(getattr(variable, "id", None) or getattr(variable, "name", ""))


def _table_id(variable: Any, table_id: str | None) -> str:
    if table_id is not None:
        return str(table_id)
    return str(getattr(variable, "table_id", "") or "")


def _validate_final_axis(ds: xr.Dataset, axis: Axis) -> None:
    out_name = str(axis.out_name or axis.name)
    value_name = str(axis.auxiliary_name or out_name)
    if out_name not in ds.coords and value_name not in ds.variables:
        raise ValueError(f"Axis {axis.name!r} was not created.")
    if axis.bounds is None:
        return
    climatology_axis = bool(axis.climatology)
    bounds_name = str(
        axis.bounds_name
        or ("climatology_bnds" if climatology_axis else f"{out_name}_bnds")
    )
    if bounds_name not in ds.data_vars:
        raise ValueError(
            f"Bounds variable {bounds_name!r} for axis {axis.name!r} was not created."
        )


def _validate_grid_coords(ds: xr.Dataset, grid: Grid) -> None:
    """Verify that lat/lon/vertex coords declared by the grid were written."""

    for name in ("latitude", "longitude"):
        arr = getattr(grid, name)
        if arr is None:
            continue
        if name not in ds.coords:
            raise ValueError(
                f"Grid coordinate {name!r} was not created in the dataset."
            )
    for field, var_name in (
        ("latitude_vertices", "vertices_latitude"),
        ("longitude_vertices", "vertices_longitude"),
    ):
        if getattr(grid, field) is None:
            continue
        if var_name not in ds.data_vars:
            raise ValueError(
                f"Grid vertex variable {var_name!r} was not created in the dataset."
            )


def _validate_final_zfactor(
    ds: xr.Dataset,
    zfactor: ZFactor,
    out_name: str,
) -> None:
    if out_name not in ds.variables:
        raise ValueError(f"Z-factor {out_name!r} was not created.")
    if zfactor.bounds is None:
        return
    bounds_name = str(zfactor.bounds_name or f"{out_name}_bnds")
    if bounds_name not in ds.data_vars:
        raise ValueError(
            f"Bounds variable {bounds_name!r} for z-factor {out_name!r} "
            "was not created."
        )
