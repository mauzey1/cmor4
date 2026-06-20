from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

from ._axis_validation import validate_and_normalize_axes
from ._templates import render_template
from ._time_utils import decode_time_value, add_time_delta, date_part
from ._variable_validation import validate_variable_values
from .axis import Axis
from .dataset import DatasetInfo, INTERNAL_DATASET_KEYS
from .grid import Grid
from .variable import Variable
from .zfactor import ZFactor

DEFAULT_OUTPUT_PATH_TEMPLATE = (
    "<drs_specs><mip_era><activity_id><institution_id><source_id>"
    "<experiment_id><variant_label><region><frequency><variable_id>"
    "<branding_suffix><grid_label><version>"
)

DEFAULT_OUTPUT_FILE_TEMPLATE = (
    "<branded_variable><frequency><region><grid_label><source_id>"
    "<experiment_id><variant_label><time_range>"
)


@dataclass(frozen=True)
class Cmor4Result:
    """Result returned by :func:`cmorize`.

    Parameters
    ----------
    dataset
        In-memory xarray dataset that was written to disk.
    path
        Filesystem path where the NetCDF file was written.
    """

    dataset: xr.Dataset
    path: Path


def create_dataset(
    dataset: DatasetInfo,
    variable: Variable,
    axes: Sequence[Axis],
    data: Any,
    *,
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Create an xarray dataset from metadata objects.

    When ``dataset`` was created by :meth:`ProjectTables.dataset_info`, this
    function uses the associated project tables to fill dataset-level defaults,
    add required scalar axes, validate table metadata for the supplied
    ``Variable``, ``Axis``, ``Grid``, and ``ZFactor`` records, validate the
    final global attributes, and verify that the generated xarray dataset
    contains the expected variables.

    Parameters
    ----------
    dataset:
        Prepared dataset metadata created by ``ProjectTables.dataset_info``.
    variable:
        Main variable metadata created by ``ProjectTables.variable``.
    axes:
        Coordinate axes with ``name``, ``values``, optional ``bounds``,
        optional ``dimensions`` for auxiliary coordinates, and optional
        ``scalar`` for scalar coordinates.
    data:
        Main variable data.
    zfactors:
        Optional hybrid-coordinate formula-term variables.
    grid:
        Optional runtime grid dimensions and grid-mapping metadata.
    attrs:
        Extra global attributes.

    Returns
    -------
    xr.Dataset
        Dataset containing the requested variable, coordinates, bounds,
        formula terms, grid mapping, and global attributes.

    Raises
    ------
    AxisValidationError
        If coordinate values or bounds are inconsistent with axis metadata.
    TableValidationError
        If project-backed metadata does not match the loaded project tables.
    ControlledVocabularyError
        If final global attributes are missing required values or contain
        values that are not allowed by the project controlled vocabulary.
    VariableValidationError
        If data values violate variable validation limits.
    ValueError
        If the data shape or final dataset structure is inconsistent with the
        requested metadata.
    """

    dataset = _dataset_for_variable(dataset, variable)
    axes = _dataset_axes(dataset, axes, variable)
    axes = validate_and_normalize_axes(dataset, variable, axes)

    # Add lat/lon grid coordinates from Grid if provided
    grid_lat_lon_axes = _grid_axes(
        grid, tuple(variable.dimensions or ()), dataset.project
    )
    axes = list(axes) + grid_lat_lon_axes

    coords: dict[str, Any] = {}
    data_vars: dict[str, Any] = {}
    axis_dims: dict[str, tuple[str, ...]] = {}
    scalar_coord_names: list[str] = []
    auxiliary_coord_names: list[str] = []

    for axis in axes:
        _add_axis(
            axis,
            coords,
            data_vars,
            axis_dims,
            scalar_coord_names,
            auxiliary_coord_names,
        )

    if grid and grid.has_mapping:
        data_vars[grid.variable_name] = (
            (),
            np.int32(0),
            grid.mapping_attributes(),
        )
        auxiliary_coord_names.extend(
            str(name) for name in (grid.coordinates or ()) if name
        )

    zfactor_names: list[str] = []
    for zfactor in zfactors or ():
        zfactor_names.append(_add_zfactor(zfactor, axes, data_vars, axis_dims))

    data_array = np.asarray(data)
    var_name, var_labels = variable.names()
    if grid is not None:
        dim_names = grid.variable_dimensions(variable)
    else:
        dim_names = None
    if dim_names is None:
        if variable.dimensions is not None:
            dim_names = tuple(str(name) for name in variable.dimensions)
        else:
            dim_names = tuple(axis.name for axis in axes if not bool(axis.auxiliary))
    dims = tuple(dim for name in dim_names for dim in axis_dims.get(name, ()))

    if data_array.ndim != len(dims):
        expected = " x ".join(dims) if dims else "scalar"
        raise ValueError(
            f"Data for {var_name!r} has {data_array.ndim} dimensions, "
            f"but variable dimensions resolve to {expected!r}."
        )
    validate_variable_values(variable, axes, data, dims, axis_dims)

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

    data_vars[var_name] = (dims, data_array, var_attrs)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=dataset.global_attributes(variable, attrs),
    )

    if zfactor_names:
        _set_formula_terms(ds, axes, variable, zfactor_names)

    missing_value = variable.missing_value or variable.fill_value
    if missing_value is not None:
        ds[var_name].attrs["missing_value"] = missing_value
        ds[var_name].encoding["_FillValue"] = missing_value

    chunksizes = variable.chunksizes or variable.chunks
    if chunksizes:
        ds[var_name].encoding["chunksizes"] = tuple(int(value) for value in chunksizes)

    _validate_final_components(
        ds,
        dataset,
        variable,
        axes,
        zfactors or (),
        grid,
        dims,
        zfactor_names,
    )

    return ds


def write_netcdf(
    ds: xr.Dataset,
    dataset: DatasetInfo,
    variable: Variable,
    path: str | Path | None = None,
    **to_netcdf_kwargs: Any,
) -> Path:
    """Write a dataset to NetCDF and return the resolved path.

    Parameters
    ----------
    ds:
        Dataset to write.
    dataset:
        Dataset metadata used to build the default output path.  When the
        metadata contains ``create_subdirectories=False``, the output
        directory must already exist; CMOR4 will not create it.  The default
        behaviour (``create_subdirectories=True``) mirrors CMOR4's original
        behaviour and creates the full directory tree automatically.
    variable:
        Variable metadata used to build the default output path.
    path:
        Explicit output path. If omitted, the path is rendered from dataset
        and variable metadata.
    **to_netcdf_kwargs:
        Additional keyword arguments forwarded to ``xarray.Dataset.to_netcdf``.

    Returns
    -------
    pathlib.Path
        Path to the written NetCDF file.

    Raises
    ------
    ValueError
        When ``create_subdirectories=False`` and the output directory does
        not already exist.
    """

    output_path = (
        Path(path) if path is not None else build_output_path(dataset, variable, ds)
    )

    # Honour create_subdirectories.  The flag mirrors CMOR3's own behaviour:
    # when False, the output directory must already exist; CMOR3 errors when
    # it cannot create the directory (e.g. a non-existent /CMIP6 root).
    # Default is True for backwards-compatibility with CMOR4's original
    # always-create behaviour.
    create_subdirs = bool(dataset.get("create_subdirectories", True))
    if create_subdirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif not output_path.parent.exists():
        raise ValueError(
            f"Output directory {str(output_path.parent)!r} does not exist "
            "and create_subdirectories is disabled. "
            "Create the directory first or set create_subdirectories=True "
            "in the dataset metadata."
        )

    ds.to_netcdf(output_path, **to_netcdf_kwargs)
    return output_path


def cmorize(
    dataset: DatasetInfo,
    variable: Variable,
    axes: Sequence[Axis],
    data: Any,
    *,
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
    path: str | Path | None = None,
    attrs: Mapping[str, Any] | None = None,
    **to_netcdf_kwargs: Any,
) -> Cmor4Result:
    """Create and write a CMOR-like NetCDF file from metadata objects.

    Parameters
    ----------
    dataset:
        Dataset-level metadata.
    variable:
        Main variable metadata.
    axes:
        Coordinate axes for the variable.
    data:
        Main variable data values.
    zfactors:
        Optional formula-term variables for hybrid coordinates.
    grid:
        Optional runtime grid dimensions and grid-mapping metadata.
    path:
        Explicit output path. If omitted, the CMOR-like path is rendered from
        metadata.
    attrs:
        Extra global attributes to include in the output dataset.
    **to_netcdf_kwargs:
        Additional keyword arguments forwarded to ``xarray.Dataset.to_netcdf``.

    Returns
    -------
    Cmor4Result
        The in-memory dataset and path to the written NetCDF file.
    """

    dataset = _dataset_for_variable(dataset, variable)
    ds = create_dataset(
        dataset,
        variable,
        axes,
        data,
        zfactors=zfactors,
        grid=grid,
        attrs=attrs,
    )
    output_path = write_netcdf(ds, dataset, variable, path=path, **to_netcdf_kwargs)
    return Cmor4Result(dataset=ds, path=output_path)


def open_dataset(path: str | Path, **kwargs: Any) -> xr.Dataset:
    """Open a NetCDF file with xarray.

    Parameters
    ----------
    path:
        Path to the NetCDF file.
    **kwargs:
        Additional keyword arguments forwarded to ``xarray.open_dataset``.

    Returns
    -------
    xr.Dataset
        Opened dataset.
    """

    return xr.open_dataset(path, **kwargs)


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


def build_output_path(
    dataset: DatasetInfo,
    variable: Variable,
    ds: xr.Dataset | None = None,
) -> Path:
    """Build a CMOR-like output path from dataset and variable metadata.

    Template resolution follows a three-level priority chain that mirrors
    CMOR3's behaviour (added in CMOR 3.12):

    1. **User-supplied** — ``output_path_template`` / ``output_file_template``
       keys in the dataset metadata.  These take highest priority.
    2. **CV DRS section** — ``directory_path_template`` /
       ``filename_template`` from the CV's ``DRS`` block, when present.
       All three project CVs shipped with CMOR4 (CMIP7, DRCDP) carry a
       ``DRS`` section; obs4MIPs does not.
    3. **Hard-coded defaults** — ``DEFAULT_OUTPUT_PATH_TEMPLATE`` and
       ``DEFAULT_OUTPUT_FILE_TEMPLATE``.

    Parameters
    ----------
    dataset:
        Dataset-level metadata containing output templates and DRS tokens.
    variable:
        Variable metadata used for filename tokens.
    ds:
        Optional dataset used to derive time-range tokens.

    Returns
    -------
    pathlib.Path
        Rendered output path, including the ``.nc`` filename.
    """

    dataset = _dataset_for_variable(dataset, variable)
    root = Path(str(dataset.get("outpath", "."))).expanduser()
    tokens = _template_tokens(dataset, variable, ds)

    # Read CV DRS templates once; both may be None when the CV lacks a DRS
    # section (e.g. obs4MIPs) or when the CV is not project-backed.
    cv_path_tmpl: str | None = None
    cv_file_tmpl: str | None = None
    cv = getattr(getattr(dataset, "project", None), "cv", None)
    if cv is not None and hasattr(cv, "drs_templates"):
        cv_path_tmpl, cv_file_tmpl = cv.drs_templates()

    path_template = str(
        dataset.get("output_path_template")
        or cv_path_tmpl
        or DEFAULT_OUTPUT_PATH_TEMPLATE
    )
    file_template = str(
        dataset.get("output_file_template")
        or cv_file_tmpl
        or DEFAULT_OUTPUT_FILE_TEMPLATE
    )

    if (
        tokens.get("time_range")
        and "<time_range>" not in file_template
        and "<time-range>" not in file_template
    ):
        file_template += "<time_range>"

    directory = render_template(path_template, tokens, "/")
    filename = render_template(file_template, tokens, "_") + ".nc"

    return root / directory / filename


def string_from_template(
    template: str,
    dataset: DatasetInfo,
    variable: Variable,
    ds: xr.Dataset | None = None,
    separator: str | None = None,
) -> str:
    """Render a template from global attributes and computed path tokens.

    Parameters
    ----------
    template:
        Template string containing ``<token>`` placeholders.
    dataset:
        Dataset-level metadata used as template tokens.
    variable:
        Variable metadata used as template tokens.
    ds:
        Optional dataset used to derive time-range tokens.
    separator:
        Separator inserted between non-empty rendered token values.

    Returns
    -------
    str
        Rendered template string.
    """

    dataset = _dataset_for_variable(dataset, variable)
    return render_template(template, _template_tokens(dataset, variable, ds), separator)


def _grid_axes(
    grid: Grid | None,
    variable_dimensions: tuple[str, ...],
    project: Any | None,
) -> list[Axis]:
    """Return all Axis objects that a Grid contributes to the dataset.

    Produces two groups, in order:

    1. **Dimensional axes** (``grid.axes``) — the indexing dimensions of the
       grid (e.g. ``i_index``, ``j_index``).  These are already fully-formed
       :class:`~cmor4.Axis` instances owned by the grid; they are passed
       through unchanged.

    2. **Auxiliary coordinate axes** — ``latitude`` and ``longitude`` 2-D
       arrays written as auxiliary coordinate variables.  Created here from
       the grid's data arrays; marked ``auxiliary=True``.

    Parameters
    ----------
    grid
        Grid object to inspect.
    variable_dimensions
        Fallback dimension names (from the variable table) used to infer the
        spatial dimension list when ``grid.dimensions`` is not set.  Time is
        filtered out.
    project
        Project tables used to merge grid-coordinate metadata from tables.

    Returns
    -------
    list[Axis]
        Dimensional axes followed by auxiliary coordinate axes.  Empty when
        *grid* is ``None`` or has neither ``axes`` nor lat/lon arrays.
    """
    if grid is None:
        return []

    result: list[Axis] = []

    # --- 1. Dimensional axes (already Axis objects, owned by the grid) -----
    result.extend(grid.axes)

    # --- 2. Auxiliary lat/lon coordinate axes --------------------------------
    # Determine spatial dimension names for the auxiliary arrays.
    if grid.dimensions:
        spatial_dims = [
            str(d) for d in grid.dimensions if str(d).lower() not in ("time",)
        ]
    elif variable_dimensions:
        spatial_dims = [
            str(d) for d in variable_dimensions if str(d).lower() not in ("time",)
        ]
    else:
        spatial_dims = []

    if grid.latitude is not None:
        lat_data: dict[str, Any] = {
            "name": "latitude",
            "grid_coordinate": "latitude",
            "values": grid.latitude,
            "dimensions": spatial_dims,
            "bounds": grid.latitude_vertices,
            "bounds_name": "vertices_latitude",
            "bounds_dim": grid.vertices_dim,
            "auxiliary": True,
        }
        if project is not None:
            lat_data = project.coordinate_table.build(lat_data)
        result.append(Axis.model_validate(lat_data))

    if grid.longitude is not None:
        lon_data: dict[str, Any] = {
            "name": "longitude",
            "grid_coordinate": "longitude",
            "values": grid.longitude,
            "dimensions": spatial_dims,
            "bounds": grid.longitude_vertices,
            "bounds_name": "vertices_longitude",
            "bounds_dim": grid.vertices_dim,
            "auxiliary": True,
        }
        if project is not None:
            lon_data = project.coordinate_table.build(lon_data)
        result.append(Axis.model_validate(lon_data))

    return result


def _add_axis(
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
        _add_axis_dim_aliases(axis, axis_dims, ())
        scalar_coord_names.append(out_name)
    elif axis.auxiliary_name:
        axis_dims[name] = (out_name,)
        _add_axis_dim_aliases(axis, axis_dims, (out_name,))
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
            _named_dimensions(axis.dimensions, axis_dims)
            if axis.dimensions is not None
            else (out_name,)
        )
        coords[out_name] = (dims, values, coord_attrs)
        if len(dims) == 1:
            axis_dims[name] = dims
            _add_axis_dim_aliases(axis, axis_dims, dims)
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


def _add_zfactor(
    zfactor: ZFactor,
    axes: Sequence[Axis],
    data_vars: dict[str, Any],
    axis_dims: Mapping[str, tuple[str, ...]],
) -> str:
    name = zfactor.name
    out_name = str(zfactor.out_name or name)
    values = zfactor.values_array()
    dims = _named_dimensions(zfactor.dimensions or (), axis_dims)
    # If the formula term has no declared dimensions, treat it as a scalar.
    # Accept a size-1 array (CMOR3-compatible input) by squeezing it.
    if not dims:
        if values.size == 1:
            values = values.reshape(())
        # values.ndim > 0 with size > 1 was already caught by validation
    validate_variable_values(
        zfactor,
        axes,
        values,
        dims,
        axis_dims,
        name=out_name,
        table_id=str(zfactor.table_entry or "formula_terms"),
    )
    attrs = zfactor.attributes()
    data_vars[out_name] = (dims, values, attrs)

    if zfactor.bounds is not None:
        bounds_name = str(zfactor.bounds_name or f"{out_name}_bnds")
        bounds_dims = dims + (str(zfactor.bounds_dim or "bnds"),)
        validate_variable_values(
            zfactor,
            axes,
            zfactor.bounds_array(),
            bounds_dims,
            axis_dims,
            name=bounds_name,
            table_id=str(zfactor.table_entry or "formula_terms"),
        )
        data_vars[bounds_name] = (
            bounds_dims,
            zfactor.bounds_array(),
            zfactor.bounds_attributes(),
        )
        attrs = dict(data_vars[out_name][2])
        attrs["bounds"] = bounds_name
        data_vars[out_name] = (dims, values, attrs)
    return out_name


def _set_formula_terms(
    ds: xr.Dataset,
    axes: Sequence[Axis],
    variable: Variable,
    zfactor_names: Sequence[str],
) -> None:
    variable_dims = set(variable.dimensions or ())
    for axis in axes:
        formula_terms = variable.formula_terms or axis.z_factors
        if not formula_terms and set(zfactor_names).issuperset({"a", "b", "p0", "ps"}):
            formula_terms = "a: a b: b p0: p0 ps: ps"
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


def _validate_final_components(
    ds: xr.Dataset,
    dataset: DatasetInfo,
    variable: Variable,
    axes: Sequence[Axis],
    zfactors: Sequence[ZFactor],
    grid: Grid | None,
    dims: Sequence[str],
    zfactor_names: Sequence[str],
) -> None:
    project = dataset.project
    if project is not None:
        project.validate_global_attributes(ds.attrs)
        project.validate_components(
            dataset,
            variable,
            axes,
            grid=grid,
            zfactors=zfactors,
        )

    var_name = variable.names()[0]
    if var_name not in ds.data_vars:
        raise ValueError(f"Variable {var_name!r} was not created.")
    if tuple(ds[var_name].dims) != tuple(dims):
        raise ValueError(
            f"Variable {var_name!r} dimensions {tuple(ds[var_name].dims)!r} "
            f"do not match expected dimensions {tuple(dims)!r}."
        )

    for axis in axes:
        _validate_final_axis(ds, axis)

    if grid is not None and grid.has_mapping:
        if grid.variable_name not in ds.data_vars:
            raise ValueError(
                f"Grid mapping variable {grid.variable_name!r} was not created."
            )
        if ds[var_name].attrs.get("grid_mapping") != grid.variable_name:
            raise ValueError(
                f"Variable {var_name!r} does not reference grid mapping "
                f"{grid.variable_name!r}."
            )

    for zfactor, out_name in zip(zfactors, zfactor_names):
        _validate_final_zfactor(ds, zfactor, out_name)


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


def _dataset_for_variable(
    dataset: DatasetInfo,
    variable: Variable,
) -> DatasetInfo:
    project = dataset.project
    if project is None:
        return dataset
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


def _add_axis_dim_aliases(
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


def _named_dimensions(
    names: Iterable[Any], axis_dims: Mapping[str, tuple[str, ...]]
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


def _template_tokens(
    dataset: DatasetInfo,
    variable: Variable,
    ds: xr.Dataset | None,
) -> dict[str, Any]:
    var_name, labels = variable.names()
    frequency = str(dataset.get("frequency") or variable.frequency or "fx")
    variant_label = dataset.variant_label()
    version = str(dataset.get("version") or f"v{date.today():%Y%m%d}")
    time_range = _time_range(ds, frequency) if frequency != "fx" else None

    tokens = {
        str(key): value
        for key, value in (ds.attrs.items() if ds is not None else ())
        if not str(key).startswith("_")
    }
    tokens.update({
        str(key): value
        for key, value in dataset.items()
        if (key not in INTERNAL_DATASET_KEYS and not str(key).startswith("_"))
    })
    tokens.update({
        "branded_name": labels["branded_name"],
        "branded_variable": labels["branded_name"],
        "branded_variable_name": labels["branded_name"],
        "branding_suffix": labels.get("branding_suffix", ""),
        "frequency": frequency,
        "grid_label": dataset.get("grid_label", tokens.get("grid_label", "gn")),
        "member_id": dataset.get("member_id", variant_label),
        "region": dataset.get("region", tokens.get("region", "glb")),
        "time-range": time_range or "",
        "time_range": time_range or "",
        "variable_id": var_name,
        "variant_label": variant_label,
        "version": version,
    })
    for key in (
        "temporal_label",
        "vertical_label",
        "horizontal_label",
        "area_label",
    ):
        if key in labels:
            tokens[key] = labels[key]
    return tokens


def _time_range(ds: xr.Dataset | None, frequency: str = "mon") -> str | None:
    if ds is None or "time" not in ds.coords:
        return None
    time = ds["time"]
    units = time.attrs.get("units")
    calendar = time.attrs.get("calendar", ds.attrs.get("calendar", "standard"))
    climatology_bounds_name = time.attrs.get("climatology")
    climatology = bool(climatology_bounds_name)
    if climatology:
        if str(climatology_bounds_name) not in ds:
            return None
        bounds = np.asarray(ds[str(climatology_bounds_name)].values)
        if bounds.size == 0:
            return None
        bounds = bounds.reshape(-1, bounds.shape[-1])
        first_value = bounds[0, 0]
        last_value = bounds[-1, -1]
    else:
        values = np.asarray(time.values)
        if values.size == 0:
            return None
        first_value = values.flat[0]
        last_value = values.flat[-1]
    first = decode_time_value(first_value, units, calendar)
    last = decode_time_value(last_value, units, calendar)
    if first is None or last is None:
        return None
    if climatology:
        first = add_time_delta(first, timedelta(hours=1))
        last = add_time_delta(last, timedelta(hours=-1))
    freq = frequency.lower()
    clim_suffix = (
        "-clim"
        if climatology and str(ds.attrs.get("mip_era", "")).upper() != "CMIP7"
        else ""
    )
    if "yr" in freq or "dec" in freq:
        return f"{date_part(first, 'year')}-{date_part(last, 'year')}{clim_suffix}"
    if "monc" in freq or "mon" in freq or climatology:
        return f"{date_part(first, 'month')}-{date_part(last, 'month')}{clim_suffix}"
    if "day" in freq:
        return f"{date_part(first, 'day')}-{date_part(last, 'day')}{clim_suffix}"
    if "subhr" in freq:
        return f"{date_part(first, 'second')}-{date_part(last, 'second')}{clim_suffix}"
    if "hr" in freq or freq in {"hour", "hourly"}:
        return f"{date_part(first, 'minute')}-{date_part(last, 'minute')}{clim_suffix}"
    return f"{date_part(first, 'month')}-{date_part(last, 'month')}{clim_suffix}"
