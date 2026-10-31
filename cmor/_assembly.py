"""Prepare metadata and assemble validated xarray datasets.

Dimension resolution, coordinate construction, formula terms, and final
structure checks live together here. Domain value checks belong to Axis,
Grid, and Variable; NetCDF encoding and file operations belong to _netcdf.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, overload

import numpy as np
import xarray as xr
from cf_units import Unit

from . import _netcdf
from ._metadata import DatasetMetadata
from ._time import _elapsed_seconds, decode_time_value
from .axis import Axis, _validate_and_normalize_axes
from .datasetinfo import DatasetInfo
from .grid import Grid, _validate_grid_coords, _validate_grid_mapping
from .variable import Variable, _validate_variable_values
from .zfactor import ZFactor


@dataclass(frozen=True)
class ValidationContext:
    """Validated metadata and resolved dimension state for dataset creation."""

    dataset: DatasetMetadata
    variable: Variable
    axes: tuple[Axis, ...]
    zfactors: tuple[ZFactor, ...]
    grid: Grid | None
    axis_dims: dict[str, tuple[str, ...]]
    dim_names: tuple[str, ...]
    dims: tuple[str, ...]


def validate_metadata(
    dataset: DatasetMetadata,
    variable: Variable,
    axes: Sequence[Axis],
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
) -> ValidationContext:
    """Validate project metadata and resolve dimensions without data values."""

    dataset, variable = _dataset_for_variable(dataset, variable)
    axes = _dataset_axes(dataset, axes, variable)
    axes = _validate_and_normalize_axes(dataset, variable, axes)
    axes = _merge_grid_axes(axes, grid)
    axis_dims = build_axis_dimension_map(axes)
    if grid is not None and (
        grid.mapping_name
        or grid.grid_mapping_name
        or "grid_mapping_name" in grid.params
    ):
        _validate_grid_mapping(None, grid)

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
    _validate_variable_values(ctx.variable, ctx.axes, data, ctx.dims, ctx.axis_dims)
    return data_array


def validate_zfactor_values(ctx: ValidationContext) -> None:
    """Validate formula-term values against resolved axis dimensions."""

    for zfactor in ctx.zfactors:
        out_name = str(zfactor.out_name or zfactor.name)
        values = zfactor.values_array()
        dims = named_dimensions(zfactor.dimensions or (), ctx.axis_dims)
        if not dims and values.size == 1:
            values = values.reshape(())
        _validate_variable_values(
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
            _validate_variable_values(
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

    project = getattr(ctx.dataset, "project", None)
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


@overload
def _dataset_for_variable(
    dataset: DatasetInfo,
    variable: Variable,
) -> tuple[DatasetInfo, Variable]: ...


@overload
def _dataset_for_variable(
    dataset: DatasetMetadata,
    variable: Variable,
) -> tuple[DatasetMetadata, Variable]: ...


def _dataset_for_variable(
    dataset: DatasetMetadata,
    variable: Variable,
) -> tuple[DatasetMetadata, Variable]:
    """Return dataset and variable prepared for a specific variable."""

    project = getattr(dataset, "project", None)
    if project is None:
        return dataset, variable
    return project._dataset_for_variable(dataset, variable)


def _dataset_axes(
    dataset: DatasetMetadata,
    axes: Sequence[Axis],
    variable: Variable,
) -> tuple[Axis, ...]:
    project = getattr(dataset, "project", None)
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


def build_axis_mappings(
    axes: Sequence[Axis],
    *,
    calendar: str | None = None,
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
            calendar=calendar,
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

    reference_axis = leadtime_axis = time_axis = None
    for axis in axes:
        is_reference = _is_reftime_axis(axis)
        is_leadtime = _is_leadtime_axis(axis)
        if is_reference and reference_axis is None:
            reference_axis = axis
        if is_leadtime and leadtime_axis is None:
            leadtime_axis = axis
        if (
            not is_reference
            and not is_leadtime
            and time_axis is None
            and _is_time_axis(axis)
        ):
            time_axis = axis
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
        time_axis.calendar
        or reference_axis.calendar
        or time_axis.attrs.get("calendar")
        or reference_axis.attrs.get("calendar")
        or calendar
        or "standard"
    )
    reference_value = reference_values.reshape(-1)[0]
    reference_date = decode_time_value(reference_value, reference_units, axis_calendar)
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
            tolerance = float(Unit("seconds").convert(1.0, Unit(duration_units)))
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

    out_name = str(
        getattr(entry, "out_name", None) or getattr(entry, "name", "leadtime")
    )
    attrs = {
        "units": duration_units,
        "standard_name": str(
            getattr(entry, "standard_name", None) or "forecast_period"
        ),
        "long_name": str(
            getattr(entry, "long_name", None)
            or "Time elapsed since the start of the forecast"
        ),
        "axis": str(getattr(entry, "axis", None) or "T"),
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
    *,
    calendar: str | None = None,
) -> None:
    name = axis.name
    out_name = str(axis.out_name or axis.name)
    values = axis.values_array()
    coord_attrs = axis.attributes()
    if (
        calendar
        and "calendar" not in coord_attrs
        and (
            str(axis.axis or "").upper() == "T"
            or str(axis.standard_name or "").lower() == "time"
            or str(axis.name).lower().startswith("time")
        )
    ):
        coord_attrs["calendar"] = calendar

    if bool(axis.scalar):
        scalar_value: Any
        if values.shape == ():
            scalar_value = values.item()
        elif values.size == 1:
            scalar_value = values.reshape(()).item()
        else:
            raise ValueError("Scalar coordinates must contain exactly one value.")
        data_vars[out_name] = ((), scalar_value, coord_attrs)
        axis_dims[name] = ()
        add_axis_dim_aliases(axis, axis_dims, ())
        scalar_coord_names.append(out_name)
    elif axis.auxiliary_name:
        axis_dims[name] = (out_name,)
        add_axis_dim_aliases(axis, axis_dims, (out_name,))
        aux_name = str(axis.auxiliary_name)
        aux_values = values
        if values.dtype.kind in {"O", "S", "U"}:
            text_values = np.asarray(values, dtype=str)
            width = max(1, *(len(value.encode("utf-8")) for value in text_values.flat))
            aux_values = text_values.astype(f"S{width}")
        data_vars[aux_name] = (
            (out_name,),
            aux_values,
            {**axis.attributes(include_units=False), **axis.auxiliary_attributes()},
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
        axis_var = coords[out_name] if out_name in coords else data_vars[out_name]
        bounds_dims = tuple(axis_var[0]) + (str(axis.bounds_dim or "bnds"),)
        data_vars[bounds_name] = (
            bounds_dims,
            bounds,
            axis.bounds_attributes(),
        )
        if str(axis.axis or "").upper() == "Z" and (
            axis.formula is not None or axis.z_factors is not None
        ):
            attrs = dict(data_vars[bounds_name][2])
            for key in ("units", "standard_name"):
                if coord_attrs.get(key) is not None:
                    attrs.setdefault(key, coord_attrs[key])
            data_vars[bounds_name] = (bounds_dims, bounds, attrs)
        attrs = dict(axis_var[2])
        attrs["climatology" if climatology_axis else "bounds"] = bounds_name
        if out_name in coords:
            coords[out_name] = (axis_var[0], axis_var[1], attrs)
        else:
            data_vars[out_name] = (axis_var[0], axis_var[1], attrs)


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


def build_dataset(
    ctx: ValidationContext,
    data_array: Any,
    *,
    attrs: Mapping[str, Any] | None = None,
    encoding: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Create a dataset from metadata and data that was already validated.

    This internal helper keeps final DatasetWriter construction on the same
    path as :func:`create_dataset` without forcing a lazy staged array into
    memory for a second validation pass.
    """

    dataset = ctx.dataset
    variable = ctx.variable
    axes = ctx.axes
    grid = ctx.grid
    var_name, var_labels = variable.names()

    (
        coords,
        data_vars,
        axis_dims,
        scalar_coord_names,
        auxiliary_coord_names,
    ) = build_axis_mappings(axes, calendar=dataset.calendar)

    forecast_coord_name = derive_forecast_coords(
        axes,
        coords,
        getattr(dataset, "project", None),
        calendar=str(dataset.calendar or "standard"),
    )
    if forecast_coord_name is not None:
        auxiliary_coord_names.append(forecast_coord_name)

    # Write lat/lon and vertex arrays directly as dataset coords, bypassing
    # the Axis pipeline — same pattern as CMOR3's cmor_grid associated_variables.
    if grid is not None:
        add_grid_coords(
            grid,
            tuple(variable.dimensions or ()),
            getattr(dataset, "project", None),
            coords,
            data_vars,
            auxiliary_coord_names,
        )

    if grid and grid.has_mapping:
        data_vars[grid.variable_name] = (
            (),
            np.int32(-2147483647),
            grid.mapping_attributes(),
        )
        auxiliary_coord_names.extend(
            str(name) for name in (grid.coordinates or ()) if name
        )

    zfactor_names: list[str] = []
    for zfactor in ctx.zfactors:
        zfactor_names.append(add_zfactor(zfactor, data_vars, axis_dims))

    var_attrs = variable.attributes(var_labels)
    explicit_coordinates = variable.coordinates
    if explicit_coordinates:
        coord_attr = (
            " ".join(str(value) for value in explicit_coordinates)
            if isinstance(explicit_coordinates, (list, tuple))
            else str(explicit_coordinates)
        )
    else:
        coord_names = [*scalar_coord_names, *auxiliary_coord_names]
        coord_attr = " ".join(dict.fromkeys(coord_names))
    if coord_attr:
        var_attrs["coordinates"] = coord_attr
    if grid and grid.variable_name in data_vars:
        var_attrs["grid_mapping"] = grid.variable_name

    # Collect any CF external variables referenced in cell_measures but not
    # written to this file and expose them as a global attribute so that
    # downstream tools and ESGF QC checkers can locate them.
    all_provided_names = set(coords.keys()) | set(data_vars.keys())
    external_vars = _collect_external_variables(variable, all_provided_names)
    if external_vars:
        # Merge computed external_variables with user-supplied attrs;
        # user-supplied attrs take precedence (they come later in the update).
        merged_attrs: dict[str, Any] = {
            "external_variables": " ".join(sorted(external_vars))
        }
        if attrs:
            merged_attrs.update(attrs)
        attrs = merged_attrs

    data_vars[var_name] = (ctx.dims, data_array, var_attrs)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=dataset.global_attributes(variable, attrs),
    )

    if zfactor_names:
        set_formula_terms(ds, axes, variable, zfactor_names)

    _netcdf.apply_dataset_encoding(ds, dataset, variable, encoding)

    validate_final_dataset(ds, ctx, zfactor_names)

    return ds


def _collect_external_variables(
    variable: Variable,
    provided_names: set[str],
) -> set[str]:
    """Return CF external variable names cited in cell_measures but not provided.

    CF Conventions require that any variable named in a ``cell_measures``
    attribute but not included in the file is listed in the global
    ``external_variables`` attribute so that data consumers know where to
    find it.  This helper parses the ``measure: varname`` tokens from the
    variable's ``cell_measures`` field and returns those whose names are not
    among the coordinates and data variables already written.

    Parameters
    ----------
    variable:
        Variable metadata record, which may carry a ``cell_measures`` entry.
    provided_names:
        Names of all coordinates and data variables that will be written to
        the output file.

    Returns
    -------
    set[str]
        External variable names that are referenced but not provided.
    """
    cell_measures = str(variable.cell_measures or "")
    if not cell_measures.strip():
        return set()
    return {
        name
        for name in re.findall(r":\s*(\S+)", cell_measures)
        if name not in provided_names
    }
