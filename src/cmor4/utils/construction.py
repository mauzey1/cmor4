from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from cf_units import Unit
import numpy as np
import xarray as xr

from ..axis import Axis
from ..grid import Grid
from ..variable import Variable
from ..zfactor import ZFactor
from .time_utils import _elapsed_seconds, decode_time_value
from .validation import add_axis_dim_aliases, named_dimensions


def build_axis_mappings(
    axes: Sequence[Axis],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, tuple[str, ...]],
    list[str],
    list[str],
]:
    """Build xarray coordinate/data-variable mappings from validated axes."""

    coords: dict[str, Any] = {}
    data_vars: dict[str, Any] = {}
    axis_dims: dict[str, tuple[str, ...]] = {}
    scalar_coord_names: list[str] = []
    auxiliary_coord_names: list[str] = []

    for axis in axes:
        add_axis(
            axis,
            coords,
            data_vars,
            axis_dims,
            scalar_coord_names,
            auxiliary_coord_names,
        )
    return coords, data_vars, axis_dims, scalar_coord_names, auxiliary_coord_names


def derive_forecast_coords(
    axes: Sequence[Axis],
    coords: dict[str, Any],
    coord_table: Any = None,
    *,
    calendar: str = "standard",
) -> str | None:
    """Add or validate ``leadtime`` derived from ``time - reftime``.

    A forecast reference-time coordinate may be scalar or a one-element
    coordinate.  An explicitly supplied forecast-period axis is retained, but
    its values must agree with the derived values to within one second.
    """

    reference_axis = next((axis for axis in axes if _is_reftime_axis(axis)), None)
    leadtime_axis = next((axis for axis in axes if _is_leadtime_axis(axis)), None)
    time_axis = next(
        (
            axis
            for axis in axes
            if not _is_reftime_axis(axis)
            and not _is_leadtime_axis(axis)
            and _is_time_axis(axis)
        ),
        None,
    )
    if time_axis is None or reference_axis is None:
        return None

    reference_values = reference_axis.values_array()
    if reference_values.size != 1:
        return None

    time_values = time_axis.values_array()
    time_units = str(time_axis.units or "")
    reference_units = str(reference_axis.units or "")
    duration_units = _duration_units(time_units)
    if not duration_units:
        raise ValueError(
            "Cannot derive leadtime because the time axis has no relative-time units."
        )

    axis_calendar = str(
        time_axis.attrs.get("calendar")
        or reference_axis.attrs.get("calendar")
        or calendar
        or "standard"
    )
    reference_value = reference_values.reshape(-1)[0]
    reference_date = decode_time_value(
        reference_value, reference_units, axis_calendar
    )
    if reference_date is None:
        raise ValueError(
            "Cannot derive leadtime from the forecast reference-time coordinate."
        )

    leadtime_values: list[float] = []
    for value in time_values.reshape(-1):
        valid_date = decode_time_value(value, time_units, axis_calendar)
        seconds = (
            _elapsed_seconds(reference_date, valid_date)
            if valid_date is not None
            else None
        )
        if seconds is None:
            raise ValueError("Cannot derive leadtime from the time coordinate.")
        leadtime_values.append(
            float(Unit("seconds").convert(seconds, Unit(duration_units)))
        )
    derived = np.asarray(leadtime_values, dtype="f8").reshape(time_values.shape)

    if leadtime_axis is not None:
        explicit = np.asarray(leadtime_axis.values_array(), dtype="f8")
        explicit_units = _duration_units(str(leadtime_axis.units or duration_units))
        try:
            comparable = np.asarray(
                Unit(explicit_units).convert(explicit, Unit(duration_units)),
                dtype="f8",
            )
            tolerance = float(
                Unit("seconds").convert(1.0, Unit(duration_units))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot compare leadtime units {leadtime_axis.units!r} "
                f"with time units {duration_units!r}."
            ) from exc
        if explicit.shape != derived.shape or not np.allclose(
            comparable, derived, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                "Explicit leadtime values do not match time - reftime "
                "within the one-second tolerance."
            )
        return str(leadtime_axis.out_name or leadtime_axis.name)

    table = getattr(coord_table, "coordinate_table", coord_table)
    entry = None
    if table is not None and hasattr(table, "resolve_coord"):
        entry = table.resolve_coord({
            "name": "leadtime",
            "standard_name": "forecast_period",
        })

    entry_data = dict(entry.entry) if entry is not None else {}
    out_name = str(entry_data.get("out_name") or getattr(entry, "name", "leadtime"))
    attrs = {
        "units": duration_units,
        "standard_name": str(
            entry_data.get("standard_name") or "forecast_period"
        ),
        "long_name": str(
            entry_data.get("long_name")
            or "Time elapsed since the start of the forecast"
        ),
        "axis": str(entry_data.get("axis") or "T"),
    }
    time_name = str(time_axis.out_name or time_axis.name)
    time_dims = tuple(coords[time_name][0])
    coords[out_name] = (time_dims, derived, attrs)
    return out_name


def _axis_names(axis: Axis) -> tuple[str, ...]:
    return tuple(
        str(value).lower()
        for value in (
            axis.name,
            axis.out_name,
            axis.table_entry,
            axis.axis_entry,
            axis.coordinate,
        )
        if value
    )


def _is_reftime_axis(axis: Axis) -> bool:
    return str(axis.standard_name or "").lower() == "forecast_reference_time" or any(
        name.startswith("reftime") for name in _axis_names(axis)
    )


def _is_leadtime_axis(axis: Axis) -> bool:
    return str(axis.standard_name or "").lower() == "forecast_period" or any(
        name.startswith("leadtime") for name in _axis_names(axis)
    )


def _is_time_axis(axis: Axis) -> bool:
    return (
        str(axis.axis or "").upper() == "T"
        or str(axis.standard_name or "").lower() == "time"
        or "time" in _axis_names(axis)
    )


def _duration_units(units: str) -> str:
    return re.split(r"\s+since\s+", units, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def add_grid_coords(
    grid: Grid,
    variable_dimensions: tuple[str, ...],
    project: Any | None,
    coords: dict[str, Any],
    data_vars: dict[str, Any],
    auxiliary_coord_names: list[str],
) -> None:
    """Write grid lat/lon/vertices directly into xarray mappings."""

    dimensions = grid.dimensions or variable_dimensions
    spatial_dims = [str(dim) for dim in dimensions if str(dim).lower() != "time"]
    coord_table = getattr(project, "coordinate_table", None)
    new_coords, new_data_vars, new_aux_names = grid.to_dataset_coords(
        spatial_dims, coord_table=coord_table
    )
    coords.update(new_coords)
    data_vars.update(new_data_vars)
    auxiliary_coord_names.extend(new_aux_names)


def add_axis(
    axis: Axis,
    coords: dict[str, Any],
    data_vars: dict[str, Any],
    axis_dims: dict[str, tuple[str, ...]],
    scalar_coord_names: list[str],
    auxiliary_coord_names: list[str],
) -> None:
    name = axis.name
    out_name = str(axis.out_name or axis.name)
    values = axis.values_array()
    coord_attrs = axis.attributes()

    if bool(axis.scalar):
        if values.shape == ():
            scalar_value = values.item()
        elif values.size == 1:
            scalar_value = values.reshape(()).item()
        else:
            raise ValueError("Scalar coordinates must contain exactly one value.")
        coords[out_name] = ((), scalar_value, coord_attrs)
        axis_dims[name] = ()
        add_axis_dim_aliases(axis, axis_dims, ())
        scalar_coord_names.append(out_name)
    elif axis.auxiliary_name:
        axis_dims[name] = (out_name,)
        add_axis_dim_aliases(axis, axis_dims, (out_name,))
        coords[out_name] = (
            out_name,
            np.arange(len(values), dtype="i4"),
            axis.attributes(include_units=False),
        )
        aux_name = str(axis.auxiliary_name)
        data_vars[aux_name] = (
            (out_name,),
            values.astype(str),
            axis.auxiliary_attributes(),
        )
        auxiliary_coord_names.append(aux_name)
    else:
        dims = (
            named_dimensions(axis.dimensions, axis_dims)
            if axis.dimensions is not None
            else (out_name,)
        )
        coords[out_name] = (dims, values, coord_attrs)
        if len(dims) == 1:
            axis_dims[name] = dims
            add_axis_dim_aliases(axis, axis_dims, dims)
        auxiliary = bool(axis.auxiliary) or len(dims) > 1
        if auxiliary:
            auxiliary_coord_names.append(out_name)
        else:
            axis_dims.setdefault(out_name, dims)

    if axis.bounds is not None:
        climatology_axis = bool(axis.climatology)
        bounds_name = str(
            axis.bounds_name
            or ("climatology_bnds" if climatology_axis else f"{out_name}_bnds")
        )
        bounds = axis.bounds_array()
        bounds_dims = tuple(coords[out_name][0]) + (str(axis.bounds_dim or "bnds"),)
        data_vars[bounds_name] = (
            bounds_dims,
            bounds,
            axis.bounds_attributes(),
        )
        coord_data = coords[out_name]
        attrs = dict(coord_data[2])
        attrs["climatology" if climatology_axis else "bounds"] = bounds_name
        coords[out_name] = (coord_data[0], coord_data[1], attrs)


def add_zfactor(
    zfactor: ZFactor,
    data_vars: dict[str, Any],
    axis_dims: Mapping[str, tuple[str, ...]],
) -> str:
    name = zfactor.name
    out_name = str(zfactor.out_name or name)
    dims = named_dimensions(zfactor.dimensions or (), axis_dims)
    values = (
        zfactor.values
        if zfactor.values is not None and hasattr(zfactor.values, "__dask_graph__")
        else zfactor.values_array()
    )
    if not dims and values.size == 1:
        values = values.reshape(())
    attrs = zfactor.attributes()
    data_vars[out_name] = (dims, values, attrs)

    if zfactor.bounds is not None:
        bounds_name = str(zfactor.bounds_name or f"{out_name}_bnds")
        bounds_dims = dims + (str(zfactor.bounds_dim or "bnds"),)
        data_vars[bounds_name] = (
            bounds_dims,
            zfactor.bounds_array(),
            zfactor.bounds_attributes(),
        )
        attrs = dict(data_vars[out_name][2])
        attrs["bounds"] = bounds_name
        data_vars[out_name] = (dims, values, attrs)
    return out_name


def set_formula_terms(
    ds: xr.Dataset,
    axes: Sequence[Axis],
    variable: Variable,
    zfactor_names: Sequence[str],
) -> None:
    variable_dims = set(variable.dimensions or ())
    for axis in axes:
        formula_terms = axis.z_factors or variable.formula_terms
        if not formula_terms:
            continue
        axis_name = axis.name
        generic_level_name = axis.generic_level_name
        out_name = str(axis.out_name or axis.name)
        if {
            str(value) for value in (axis_name, generic_level_name, out_name) if value
        } & variable_dims:
            coord_name = str(axis.out_name or axis.name)
            if coord_name in ds.coords:
                ds[coord_name].attrs["formula_terms"] = formula_terms
            bounds_name = (
                ds[coord_name].attrs.get("bounds") if coord_name in ds.coords else None
            )
            if bounds_name and bounds_name in ds:
                bnds_formula_terms = formula_terms
                for factor in ("a", "b"):
                    bnds_name = f"{factor}_bnds"
                    if bnds_name in ds:
                        bnds_formula_terms = bnds_formula_terms.replace(
                            f"{factor}: {factor}", f"{factor}: {bnds_name}"
                        )
                bnds_attrs = {}
                for key in ("formula", "standard_name", "units"):
                    if key in ds[coord_name].attrs:
                        bnds_attrs[key] = ds[coord_name].attrs[key]
                bnds_attrs["formula_terms"] = bnds_formula_terms
                ds[bounds_name].attrs.update(bnds_attrs)
