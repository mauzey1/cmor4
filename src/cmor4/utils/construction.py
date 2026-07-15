from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from ..axis import Axis
from ..grid import Grid
from ..variable import Variable
from ..zfactor import ZFactor
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


def add_grid_coords(
    grid: Grid,
    variable_dimensions: tuple[str, ...],
    project: Any | None,
    coords: dict[str, Any],
    data_vars: dict[str, Any],
    auxiliary_coord_names: list[str],
) -> None:
    """Write grid lat/lon/vertices directly into xarray mappings."""

    spatial_dims = _grid_spatial_dims(grid, variable_dimensions)
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
    values = _zfactor_values(zfactor)
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


def _zfactor_values(zfactor: ZFactor) -> Any:
    values = zfactor.values
    if values is not None and hasattr(values, "__dask_graph__"):
        return values
    return zfactor.values_array()


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


def _grid_spatial_dims(
    grid: Grid,
    variable_dimensions: tuple[str, ...],
) -> list[str]:
    if grid.dimensions:
        return [str(dim) for dim in grid.dimensions if str(dim).lower() != "time"]
    return [str(dim) for dim in variable_dimensions if str(dim).lower() != "time"]
