from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

try:
    import cftime
except ImportError:  # pragma: no cover - cftime is provided by netCDF4 here.
    cftime = None

from .axis import Axis
from .grid import Grid
from .metadata import _MetadataRecord
from .tables import ProjectTables
from ._templates import render_template as _render_template
from .variable import Variable
from .zfactor import ZFactor

INTERNAL_DATASET_KEYS = {
    "_history_template",
    "outpath",
    "output_file_template",
    "output_path_template",
}

RIPF_KEYS = (
    "realization_index",
    "initialization_index",
    "physics_index",
    "forcing_index",
)

DEFAULT_OUTPUT_PATH_TEMPLATE = (
    "<drs_specs>/<mip_era>/<activity_id>/<institution_id>/<source_id>/"
    "<experiment_id>/<variant_label>/<region>/<frequency>/<variable_id>/"
    "<branding_suffix>/<grid_label>/<version>"
)

DEFAULT_OUTPUT_FILE_TEMPLATE = (
    "<branded_variable><frequency><region><grid_label><source_id>"
    "<experiment_id><variant_label><time_range>"
)


@dataclass(frozen=True)
class Cmor4Result:
    """Result returned by :func:`cmorize`."""

    dataset: xr.Dataset
    path: Path


def create_dataset(
    dataset: Mapping[str, Any],
    variable: Variable,
    axes: Sequence[Axis],
    data: Any,
    *,
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
    attrs: Mapping[str, Any] | None = None,
    project: ProjectTables | None = None,
) -> xr.Dataset:
    """Create an xarray dataset from metadata objects.

    Parameters
    ----------
    dataset:
        Global dataset metadata. Common CMOR identity fields such as
        ``activity_id``, ``source_id``, ``experiment_id``, ``grid_label``,
        ``frequency``, and the RIPF index fields are copied to global attrs.
    variable:
        Main variable metadata. ``name`` may be a branded name such as
        ``tas_tavg-h2m-hxy-u``; ``id`` can override the output variable name.
        ``dimensions`` names the axes used by the data array.
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
    """

    if project is not None:
        dataset, variable = project.prepare_inputs(dataset, variable)
        axes = project.prepare_axes(axes, variable)
        zfactors = project.prepare_zfactors(zfactors)
        grid = project.prepare_grid(grid)

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
            str(name) for name in grid.get("coordinates", ()) if name
        )

    zfactor_names: list[str] = []
    for zfactor in zfactors or ():
        zfactor_names.append(_add_zfactor(zfactor, data_vars, axis_dims))

    data_array = np.asarray(data)
    var_name, var_labels = variable.names()
    if grid is not None:
        dim_names = grid.variable_dimensions(variable)
    else:
        dim_names = None
    if dim_names is None:
        if "dimensions" in variable:
            dim_names = tuple(str(name) for name in variable["dimensions"])
        else:
            dim_names = tuple(
                str(axis["name"])
                for axis in axes
                if not axis.get("auxiliary", False)
            )
    dims = tuple(dim for name in dim_names for dim in axis_dims.get(name, ()))

    if data_array.ndim != len(dims):
        expected = " x ".join(dims) if dims else "scalar"
        raise ValueError(
            f"Data for {var_name!r} has {data_array.ndim} dimensions, "
            f"but variable dimensions resolve to {expected!r}."
        )

    var_attrs = variable.attributes(var_labels)
    explicit_coordinates = variable.get("coordinates")
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

    data_vars[var_name] = (dims, data_array, var_attrs)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=_global_attrs(dataset, variable, attrs),
    )

    if zfactor_names:
        _set_formula_terms(ds, axes, variable, zfactor_names)

    missing_value = variable.get("missing_value", variable.get("fill_value"))
    if missing_value is not None:
        ds[var_name].attrs["missing_value"] = missing_value
        ds[var_name].encoding["_FillValue"] = missing_value

    chunksizes = variable.get("chunksizes", variable.get("chunks"))
    if chunksizes:
        ds[var_name].encoding["chunksizes"] = tuple(
            int(value) for value in chunksizes
        )

    return ds


def write_netcdf(
    ds: xr.Dataset,
    dataset: Mapping[str, Any],
    variable: Variable,
    path: str | Path | None = None,
    **to_netcdf_kwargs: Any,
) -> Path:
    """Write a dataset to NetCDF and return the resolved path."""

    output_path = (
        Path(path)
        if path is not None
        else build_output_path(dataset, variable, ds)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path, **to_netcdf_kwargs)
    return output_path


def cmorize(
    dataset: Mapping[str, Any],
    variable: Variable,
    axes: Sequence[Axis],
    data: Any,
    *,
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
    path: str | Path | None = None,
    attrs: Mapping[str, Any] | None = None,
    project: ProjectTables | None = None,
    **to_netcdf_kwargs: Any,
) -> Cmor4Result:
    """Create and write a CMOR-like NetCDF file from metadata objects."""

    if project is not None:
        dataset, variable = project.prepare_inputs(dataset, variable)
        axes = project.prepare_axes(axes, variable)
        zfactors = project.prepare_zfactors(zfactors)
        grid = project.prepare_grid(grid)
    ds = create_dataset(
        dataset,
        variable,
        axes,
        data,
        zfactors=zfactors,
        grid=grid,
        attrs=attrs,
    )
    output_path = write_netcdf(
        ds, dataset, variable, path=path, **to_netcdf_kwargs
    )
    return Cmor4Result(dataset=ds, path=output_path)


def open_dataset(path: str | Path, **kwargs: Any) -> xr.Dataset:
    """Open a NetCDF file with xarray."""

    return xr.open_dataset(path, **kwargs)


def build_output_path(
    dataset: Mapping[str, Any],
    variable: Variable,
    ds: xr.Dataset | None = None,
) -> Path:
    """Build a CMOR-like output path from dataset and variable metadata."""

    root = Path(str(dataset.get("outpath", "."))).expanduser()
    tokens = _template_tokens(dataset, variable, ds)
    path_template = str(
        dataset.get("output_path_template") or DEFAULT_OUTPUT_PATH_TEMPLATE
    )
    file_template = str(
        dataset.get("output_file_template") or DEFAULT_OUTPUT_FILE_TEMPLATE
    )

    directory = root.joinpath(*_render_path_template(path_template, tokens))
    rendered_tokens = _render_path_template(file_template, tokens)

    if (
        dataset.get("output_file_template")
        and tokens.get("time_range")
        and "<time_range>" not in file_template
        and "<time-range>" not in file_template
    ):
        rendered_tokens.append(str(tokens["time_range"]))

    filename = "_".join(rendered_tokens) + ".nc"
    return directory / filename


def render_template(
    template: str,
    dataset: Mapping[str, Any],
    variable: Variable,
    ds: xr.Dataset | None = None,
) -> str:
    """Render a template from global attributes and computed path tokens."""

    return _render_template(
        template,
        _template_tokens(dataset, variable, ds),
    )


def _add_axis(
    axis: Axis,
    coords: dict[str, Any],
    data_vars: dict[str, Any],
    axis_dims: dict[str, tuple[str, ...]],
    scalar_coord_names: list[str],
    auxiliary_coord_names: list[str],
) -> None:
    name = str(axis["name"])
    out_name = str(axis.get("out_name") or axis["name"])
    values = axis.values_array()
    coord_attrs = axis.attributes()

    if axis.get("scalar", False):
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
    elif axis.get("auxiliary_name"):
        axis_dims[name] = (out_name,)
        _add_axis_dim_aliases(axis, axis_dims, (out_name,))
        coords[out_name] = (
            out_name,
            np.arange(len(values), dtype="i4"),
            axis.attributes(include_units=False),
        )
        aux_name = str(axis["auxiliary_name"])
        data_vars[aux_name] = (
            (out_name,),
            values.astype(str),
            axis.auxiliary_attributes(),
        )
        auxiliary_coord_names.append(aux_name)
    else:
        dims = (
            _named_dimensions(axis["dimensions"], axis_dims)
            if "dimensions" in axis
            else (out_name,)
        )
        coords[out_name] = (dims, values, coord_attrs)
        if len(dims) == 1:
            axis_dims[name] = dims
            _add_axis_dim_aliases(axis, axis_dims, dims)
        auxiliary = bool(axis.get("auxiliary", False)) or len(dims) > 1
        if auxiliary:
            auxiliary_coord_names.append(out_name)
        else:
            axis_dims.setdefault(out_name, dims)

    if "bounds" in axis:
        climatology_axis = str(
            axis.get("climatology", "")
        ).lower() in {"1", "true", "yes"}
        bounds_name = str(
            axis.get("bounds_name")
            or ("climatology_bnds" if climatology_axis else f"{out_name}_bnds")
        )
        bounds = axis.bounds_array()
        bounds_dims = tuple(coords[out_name][0]) + (
            str(axis.get("bounds_dim", "bnds")),
        )
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
    data_vars: dict[str, Any],
    axis_dims: Mapping[str, tuple[str, ...]],
) -> str:
    name = str(zfactor["name"])
    out_name = str(zfactor.get("out_name") or name)
    values = zfactor.values_array()
    dims = _named_dimensions(zfactor.get("dimensions", ()), axis_dims)
    if not dims and values.ndim > 0:
        dims = (out_name,)
    attrs = zfactor.attributes()
    data_vars[out_name] = (dims, values, attrs)

    if "bounds" in zfactor:
        bounds_name = str(zfactor.get("bounds_name") or f"{out_name}_bnds")
        data_vars[bounds_name] = (
            dims + (str(zfactor.get("bounds_dim", "bnds")),),
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
    variable_dims = set(variable.get("dimensions", ()))
    for axis in axes:
        formula_terms = variable.get("formula_terms") or axis.get("z_factors")
        if not formula_terms and set(zfactor_names).issuperset(
            {"a", "b", "p0", "ps"}
        ):
            formula_terms = "a: a b: b p0: p0 ps: ps"
        if not formula_terms:
            continue
        axis_name = axis.get("name")
        generic_level_name = axis.get("generic_level_name")
        out_name = str(axis.get("out_name") or axis["name"])
        if {
            str(value)
            for value in (axis_name, generic_level_name, out_name)
            if value
        } & variable_dims:
            coord_name = str(axis.get("out_name") or axis["name"])
            if coord_name in ds.coords:
                ds[coord_name].attrs["formula_terms"] = formula_terms


def _global_attrs(
    dataset: Mapping[str, Any],
    variable: Variable,
    extra_attrs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "Conventions": dataset.get("Conventions", "CF-1.11"),
        "cmor4_version": "0.1.0",
    }
    for key, value in dataset.items():
        if key in INTERNAL_DATASET_KEYS or key.startswith("_"):
            continue
        if _MetadataRecord.is_netcdf_attr_value(value):
            attrs[key] = value

    var_name, labels = variable.names()
    attrs.setdefault("variable_id", var_name)
    attrs.setdefault("branded_variable", labels["branded_name"])
    for key in (
        "branding_suffix",
        "temporal_label",
        "vertical_label",
        "horizontal_label",
        "area_label",
    ):
        if key in labels:
            attrs.setdefault(key, labels[key])
    for key in ("frequency", "realm", "table_id"):
        if key in variable:
            attrs.setdefault(key, variable[key])
    if "table_info" in variable:
        attrs.setdefault("table_info", variable["table_info"])
    attrs.setdefault("variant_label", _variant_label(dataset))
    attrs.setdefault(
        "creation_date", datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    )
    if extra_attrs:
        attrs.update(_MetadataRecord.netcdf_attrs(extra_attrs))
    return attrs


def _add_axis_dim_aliases(
    axis: Axis,
    axis_dims: dict[str, tuple[str, ...]],
    dims: tuple[str, ...],
) -> None:
    for key in ("table_entry", "generic_level_name", "out_name"):
        value = axis.get(key)
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


def _variant_label(dataset: Mapping[str, Any]) -> str:
    if dataset.get("variant_label"):
        return str(dataset["variant_label"])
    values = [dataset.get(key) for key in RIPF_KEYS]
    if all(value not in (None, "") for value in values):
        return "".join(str(value) for value in values)
    return "r1i1p1f1"


def _template_tokens(
    dataset: Mapping[str, Any],
    variable: Variable,
    ds: xr.Dataset | None,
) -> dict[str, Any]:
    var_name, labels = variable.names()
    frequency = str(
        dataset.get("frequency", variable.get("frequency", "fx"))
    )
    variant_label = _variant_label(dataset)
    version = str(dataset.get("version") or f"v{date.today():%Y%m%d}")
    time_range = _time_range(ds, frequency) if frequency != "fx" else None

    tokens = {
        str(key): value
        for key, value in (ds.attrs.items() if ds is not None else ())
        if not str(key).startswith("_")
    }
    tokens.update(
        {
            str(key): value
            for key, value in dataset.items()
            if key not in INTERNAL_DATASET_KEYS and not str(key).startswith("_")
        }
    )
    tokens.update(
        {
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
        }
    )
    for key in (
        "temporal_label",
        "vertical_label",
        "horizontal_label",
        "area_label",
    ):
        if key in labels:
            tokens[key] = labels[key]
    return tokens


def _render_path_template(
    template: str, tokens: Mapping[str, Any]
) -> list[str]:
    parts: list[str] = []
    for section in template.split("/"):
        token_names = re.findall(r"<([^>]+)>", section)
        if (
            token_names
            and "".join(f"<{name}>" for name in token_names) == section
        ):
            parts.extend(
                str(tokens.get(name, ""))
                for name in token_names
                if tokens.get(name, "") not in (None, "")
            )
        else:
            rendered = _render_template(section, tokens)
            if rendered:
                parts.append(rendered)
    return parts


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
    first = _decode_time_value(first_value, units, calendar)
    last = _decode_time_value(last_value, units, calendar)
    if first is None or last is None:
        return None
    if climatology:
        first = _add_time_delta(first, timedelta(hours=1))
        last = _add_time_delta(last, timedelta(hours=-1))
    freq = frequency.lower()
    clim_suffix = (
        "-clim"
        if climatology and str(ds.attrs.get("mip_era", "")).upper() != "CMIP7"
        else ""
    )
    if "yr" in freq or "dec" in freq:
        return f"{_date_part(first, 'year')}-{_date_part(last, 'year')}{clim_suffix}"
    if "monc" in freq or "mon" in freq or climatology:
        return (
            f"{_date_part(first, 'month')}-{_date_part(last, 'month')}"
            f"{clim_suffix}"
        )
    if "day" in freq:
        return f"{_date_part(first, 'day')}-{_date_part(last, 'day')}{clim_suffix}"
    if "subhr" in freq:
        return (
            f"{_date_part(first, 'second')}-{_date_part(last, 'second')}"
            f"{clim_suffix}"
        )
    if "hr" in freq or freq in {"hour", "hourly"}:
        return (
            f"{_date_part(first, 'minute')}-{_date_part(last, 'minute')}"
            f"{clim_suffix}"
        )
    return f"{_date_part(first, 'month')}-{_date_part(last, 'month')}{clim_suffix}"


def _decode_time_value(
    value: Any, units: Any, calendar: Any = "standard"
) -> Any | None:
    if np.issubdtype(np.asarray(value).dtype, np.datetime64):
        text = np.datetime_as_string(value, unit="s")
        return datetime.fromisoformat(text)
    if not units:
        return None
    units_text = _normalize_time_units(str(units))
    if cftime is not None:
        try:
            return cftime.num2date(
                float(value),
                units_text,
                calendar=str(calendar or "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=False,
            )
        except Exception:
            pass
    match = re.match(
        r"^(days|hours|minutes|seconds) since "
        r"(\d{1,4})-(\d{1,2})-(\d{1,2})",
        units_text,
    )
    if not match:
        return None
    unit, year, month, day = match.groups()
    base = datetime(int(year), int(month), int(day))
    numeric = float(value)
    if unit == "days":
        return base + timedelta(days=numeric)
    if unit == "hours":
        return base + timedelta(hours=numeric)
    if unit == "minutes":
        return base + timedelta(minutes=numeric)
    return base + timedelta(seconds=numeric)


def _normalize_time_units(units: str) -> str:
    match = re.match(
        r"^(\w+) since (\d{1,4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?(.*)$",
        units,
    )
    if not match:
        return units
    unit, year, month, day, suffix = match.groups()
    return (
        f"{unit} since {int(year):04d}-{int(month or 1):02d}-"
        f"{int(day or 1):02d}{suffix or ''}"
    )


def _add_time_delta(value: Any, delta: timedelta) -> Any:
    try:
        return value + delta
    except TypeError:
        return (
            datetime(
                value.year,
                value.month,
                value.day,
                int(value.hour),
                int(value.minute),
                int(value.second),
            )
            + delta
        )


def _date_part(value: Any, precision: str) -> str:
    if precision == "minute":
        value = _add_time_delta(value, timedelta(seconds=30))
    else:
        value = _add_time_delta(value, timedelta(seconds=0.5))
    if precision == "year":
        return f"{value.year:04d}"
    if precision == "month":
        return f"{value.year:04d}{value.month:02d}"
    if precision == "day":
        return f"{value.year:04d}{value.month:02d}{value.day:02d}"
    if precision == "minute":
        return (
            f"{value.year:04d}{value.month:02d}{value.day:02d}"
            f"{value.hour:02d}{value.minute:02d}"
        )
    return (
        f"{value.year:04d}{value.month:02d}{value.day:02d}"
        f"{value.hour:02d}{value.minute:02d}{value.second:02d}"
    )
