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

from .metadata import Axis, Variable, ZFactor
from .tables import ProjectTables

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


@dataclass(frozen=True)
class Cmor4Result:
    """Result returned by :func:`cmorize`."""

    dataset: xr.Dataset
    path: Path


def create_dataset(
    dataset: Mapping[str, Any],
    variable: Variable | Mapping[str, Any],
    axes: Sequence[Axis | Mapping[str, Any]],
    data: Any,
    *,
    zfactors: Sequence[ZFactor | Mapping[str, Any]] | None = None,
    grid: Mapping[str, Any] | None = None,
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
        Optional grid-mapping metadata and auxiliary projected-grid controls.
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

    if grid:
        _add_grid_mapping(grid, data_vars)
        auxiliary_coord_names.extend(
            str(name) for name in grid.get("coordinates", ()) if name
        )

    zfactor_names: list[str] = []
    for zfactor in zfactors or ():
        zfactor_names.append(_add_zfactor(zfactor, data_vars, axis_dims))

    data_array = np.asarray(data)
    var_name, var_labels = _variable_names(variable)
    dim_names = _variable_dims(variable, axes)
    dims = tuple(dim for name in dim_names for dim in axis_dims.get(name, ()))

    if data_array.ndim != len(dims):
        expected = " x ".join(dims) if dims else "scalar"
        raise ValueError(
            f"Data for {var_name!r} has {data_array.ndim} dimensions, "
            f"but variable dimensions resolve to {expected!r}."
        )

    var_attrs = _variable_attrs(variable, var_labels)
    coord_attr = _coordinates_attr(
        variable, scalar_coord_names, auxiliary_coord_names
    )
    if coord_attr:
        var_attrs["coordinates"] = coord_attr
    if grid and grid.get("mapping_var", "crs") in data_vars:
        var_attrs["grid_mapping"] = str(grid.get("mapping_var", "crs"))

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
    variable: Variable | Mapping[str, Any],
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
    variable: Variable | Mapping[str, Any],
    axes: Sequence[Axis | Mapping[str, Any]],
    data: Any,
    *,
    zfactors: Sequence[ZFactor | Mapping[str, Any]] | None = None,
    grid: Mapping[str, Any] | None = None,
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
    variable: Variable | Mapping[str, Any],
    ds: xr.Dataset | None = None,
) -> Path:
    """Build a CMOR-like output path from dataset and variable metadata."""

    root = Path(str(dataset.get("outpath", "."))).expanduser()
    var_name, labels = _variable_names(variable)
    branded_name = labels["branded_name"]
    branding_suffix = labels.get("branding_suffix", "")
    version = str(dataset.get("version") or f"v{date.today():%Y%m%d}")
    variant_label = _variant_label(dataset)
    frequency = _as_text(
        dataset.get("frequency", variable.get("frequency", "fx"))
    )
    region = _as_text(dataset.get("region", "glb"))
    grid_label = _as_text(dataset.get("grid_label", "gn"))

    tokens = _path_tokens(
        dataset, variable, ds, labels, version, variant_label, frequency
    )
    path_template = dataset.get("output_path_template")
    file_template = dataset.get("output_file_template")

    if path_template:
        directory = root.joinpath(
            *_render_path_template(str(path_template), tokens)
        )
    else:
        parts = [
            dataset.get("drs_specs"),
            dataset.get("mip_era"),
            dataset.get("activity_id"),
            dataset.get("institution_id"),
            dataset.get("source_id"),
            dataset.get("experiment_id"),
            variant_label,
            region,
            frequency,
            var_name,
            branding_suffix,
            grid_label,
            version,
        ]
        directory = root.joinpath(
            *[_as_text(part) for part in parts if part not in (None, "")]
        )

    time_range = _time_range(ds, frequency) if frequency != "fx" else None
    if file_template:
        rendered_tokens = _render_path_template(str(file_template), tokens)
        if time_range:
            rendered_tokens.append(time_range)
        filename = "_".join(rendered_tokens) + ".nc"
        return directory / filename

    file_tokens = [
        branded_name,
        frequency,
        region,
        grid_label,
        dataset.get("source_id"),
        dataset.get("experiment_id"),
        variant_label,
    ]
    if time_range:
        file_tokens.append(time_range)

    filename = (
        "_".join(
            _as_text(token) for token in file_tokens if token not in (None, "")
        )
        + ".nc"
    )
    return directory / filename


def _add_axis(
    axis: Axis | Mapping[str, Any],
    coords: dict[str, Any],
    data_vars: dict[str, Any],
    axis_dims: dict[str, tuple[str, ...]],
    scalar_coord_names: list[str],
    auxiliary_coord_names: list[str],
) -> None:
    name = str(axis["name"])
    out_name = _axis_out_name(axis)
    values = _array(axis.get("values", []))
    coord_attrs = _axis_attrs(axis)

    if axis.get("scalar", False):
        coords[out_name] = ((), _scalar(values), coord_attrs)
        axis_dims[name] = ()
        _add_axis_dim_aliases(axis, axis_dims, ())
        scalar_coord_names.append(out_name)
    elif axis.get("auxiliary_name"):
        axis_dims[name] = (out_name,)
        _add_axis_dim_aliases(axis, axis_dims, (out_name,))
        coords[out_name] = (
            out_name,
            np.arange(len(values), dtype="i4"),
            _axis_attrs(axis, include_units=False),
        )
        aux_name = str(axis["auxiliary_name"])
        data_vars[aux_name] = (
            (out_name,),
            values.astype(str),
            _attrs(axis.get("auxiliary_attrs", {})),
        )
        auxiliary_coord_names.append(aux_name)
    else:
        dims = _axis_dimensions(axis, axis_dims, default=(out_name,))
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
        climatology_axis = _is_climatology_axis(axis)
        bounds_name = str(
            axis.get("bounds_name")
            or ("climatology_bnds" if climatology_axis else f"{out_name}_bnds")
        )
        bounds = _array(axis["bounds"])
        bounds_dims = tuple(coords[out_name][0]) + (
            str(axis.get("bounds_dim", "bnds")),
        )
        data_vars[bounds_name] = (
            bounds_dims,
            bounds,
            _attrs(axis.get("bounds_attrs", {})),
        )
        coord_data = coords[out_name]
        attrs = dict(coord_data[2])
        attrs["climatology" if climatology_axis else "bounds"] = bounds_name
        coords[out_name] = (coord_data[0], coord_data[1], attrs)


def _add_zfactor(
    zfactor: ZFactor | Mapping[str, Any],
    data_vars: dict[str, Any],
    axis_dims: Mapping[str, tuple[str, ...]],
) -> str:
    name = str(zfactor["name"])
    out_name = str(zfactor.get("out_name") or name)
    values = _array(zfactor.get("values", zfactor.get("data", [])))
    dims = _named_dimensions(zfactor.get("dimensions", ()), axis_dims)
    if not dims and values.ndim > 0:
        dims = (out_name,)
    attrs = _attrs(zfactor.get("attrs", {}))
    for key in ("units", "standard_name", "long_name"):
        if key in zfactor:
            attrs[key] = zfactor[key]
    data_vars[out_name] = (dims, values, attrs)

    if "bounds" in zfactor:
        bounds_name = str(zfactor.get("bounds_name") or f"{out_name}_bnds")
        data_vars[bounds_name] = (
            dims + (str(zfactor.get("bounds_dim", "bnds")),),
            _array(zfactor["bounds"]),
            _attrs(zfactor.get("bounds_attrs", {})),
        )
        attrs = dict(data_vars[out_name][2])
        attrs["bounds"] = bounds_name
        data_vars[out_name] = (dims, values, attrs)
    return out_name


def _add_grid_mapping(
    grid: Mapping[str, Any], data_vars: dict[str, Any]
) -> None:
    mapping_var = str(grid.get("mapping_var", "crs"))
    mapping_name = grid.get("mapping_name", grid.get("grid_mapping_name"))
    attrs = _attrs(grid.get("attrs", {}))
    if mapping_name:
        attrs["grid_mapping_name"] = mapping_name
    for key, value in grid.get("params", {}).items():
        if isinstance(value, (list, tuple)) and value:
            attrs[key] = value[0]
            if len(value) > 1 and value[1]:
                attrs[f"{key}_units"] = value[1]
        else:
            attrs[key] = value
    data_vars[mapping_var] = ((), np.int32(0), attrs)


def _set_formula_terms(
    ds: xr.Dataset,
    axes: Sequence[Axis | Mapping[str, Any]],
    variable: Variable | Mapping[str, Any],
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
        out_name = _axis_out_name(axis)
        if {
            str(value)
            for value in (axis_name, generic_level_name, out_name)
            if value
        } & variable_dims:
            coord_name = _axis_out_name(axis)
            if coord_name in ds.coords:
                ds[coord_name].attrs["formula_terms"] = formula_terms


def _variable_names(
    variable: Variable | Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    branded_name = str(
        variable.get("name")
        or variable.get("id")
        or variable.get("variable_id")
    )
    variable_id = str(
        variable.get("id")
        or variable.get("variable_id")
        or branded_name.split("_", 1)[0]
    )
    labels = {"branded_name": branded_name, "variable_id": variable_id}
    if "_" in branded_name:
        suffix = branded_name.split("_", 1)[1]
        labels["branding_suffix"] = suffix
        parts = suffix.split("-")
        for key, value in zip(
            (
                "temporal_label",
                "vertical_label",
                "horizontal_label",
                "area_label",
            ),
            parts,
        ):
            labels[key] = value
    return variable_id, labels


def _variable_dims(
    variable: Variable | Mapping[str, Any],
    axes: Sequence[Axis | Mapping[str, Any]],
) -> tuple[str, ...]:
    if "dimensions" in variable:
        return tuple(str(name) for name in variable["dimensions"])
    return tuple(
        str(axis["name"]) for axis in axes if not axis.get("auxiliary", False)
    )


def _variable_attrs(
    variable: Variable | Mapping[str, Any], labels: Mapping[str, str]
) -> dict[str, Any]:
    attrs = _attrs(variable.get("attrs", {}))
    for key in (
        "units",
        "standard_name",
        "long_name",
        "cell_methods",
        "cell_measures",
        "comment",
    ):
        if key in variable:
            attrs[key] = variable[key]
    attrs.setdefault("branded_variable_name", labels["branded_name"])
    for key in (
        "branding_suffix",
        "temporal_label",
        "vertical_label",
        "horizontal_label",
        "area_label",
    ):
        if key in labels:
            attrs.setdefault(key, labels[key])
    return attrs


def _coordinates_attr(
    variable: Variable | Mapping[str, Any],
    scalar_coord_names: Sequence[str],
    auxiliary_coord_names: Sequence[str],
) -> str:
    explicit = variable.get("coordinates")
    if explicit:
        return (
            " ".join(str(value) for value in explicit)
            if isinstance(explicit, (list, tuple))
            else str(explicit)
        )
    names = [*scalar_coord_names, *auxiliary_coord_names]
    return " ".join(dict.fromkeys(names))


def _global_attrs(
    dataset: Mapping[str, Any],
    variable: Variable | Mapping[str, Any],
    extra_attrs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "Conventions": dataset.get("Conventions", "CF-1.11"),
        "cmor4_version": "0.1.0",
    }
    for key, value in dataset.items():
        if key in INTERNAL_DATASET_KEYS or key.startswith("_"):
            continue
        if _is_attr_value(value):
            attrs[key] = value

    var_name, labels = _variable_names(variable)
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
        attrs.update(_attrs(extra_attrs))
    return attrs


def _axis_attrs(
    axis: Axis | Mapping[str, Any], *, include_units: bool = True
) -> dict[str, Any]:
    attrs = _attrs(axis.get("attrs", {}))
    if include_units and "units" in axis:
        attrs["units"] = axis["units"]
    for key in (
        "standard_name",
        "long_name",
        "axis",
        "positive",
        "formula",
        "valid_min",
        "valid_max",
    ):
        if key in axis:
            attrs[key] = axis[key]
    return attrs


def _is_climatology_axis(axis: Axis | Mapping[str, Any]) -> bool:
    return str(axis.get("climatology", "")).lower() in {"1", "true", "yes"}


def _axis_out_name(axis: Axis | Mapping[str, Any]) -> str:
    return str(axis.get("out_name") or axis["name"])


def _add_axis_dim_aliases(
    axis: Axis | Mapping[str, Any],
    axis_dims: dict[str, tuple[str, ...]],
    dims: tuple[str, ...],
) -> None:
    for key in ("table_entry", "generic_level_name", "out_name"):
        value = axis.get(key)
        if value:
            axis_dims.setdefault(str(value), dims)


def _axis_dimensions(
    axis: Axis | Mapping[str, Any],
    axis_dims: Mapping[str, tuple[str, ...]],
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if "dimensions" not in axis:
        return default
    return _named_dimensions(axis["dimensions"], axis_dims)


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


def _path_tokens(
    dataset: Mapping[str, Any],
    variable: Variable | Mapping[str, Any],
    ds: xr.Dataset | None,
    labels: Mapping[str, str],
    version: str,
    variant_label: str,
    frequency: str,
) -> dict[str, Any]:
    var_name = labels["variable_id"]
    tokens = {
        key: value for key, value in dataset.items() if not key.startswith("_")
    }
    tokens.update(
        {
            "branded_variable": labels["branded_name"],
            "branding_suffix": labels.get("branding_suffix", ""),
            "frequency": frequency,
            "member_id": dataset.get("member_id", variant_label),
            "time-range": (
                _time_range(ds, frequency) if frequency != "fx" else ""
            ),
            "time_range": (
                _time_range(ds, frequency) if frequency != "fx" else ""
            ),
            "variable_id": var_name,
            "variant_label": variant_label,
            "version": version,
        }
    )
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
                _as_text(tokens.get(name, ""))
                for name in token_names
                if tokens.get(name, "") not in (None, "")
            )
        else:
            rendered = re.sub(
                r"<([^>]+)>",
                lambda match: _as_text(tokens.get(match.group(1), "")),
                section,
            )
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
        return _as_datetime(value) + delta


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


def _as_datetime(value: Any) -> datetime:
    return datetime(
        value.year,
        value.month,
        value.day,
        int(value.hour),
        int(value.minute),
        int(value.second),
    )


def _array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S", "O"}:
        return array.astype(str)
    return array


def _scalar(value: np.ndarray) -> Any:
    if value.shape == ():
        return value.item()
    if value.size != 1:
        raise ValueError("Scalar coordinates must contain exactly one value.")
    return value.reshape(()).item()


def _attrs(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in values.items()
        if _is_attr_value(value)
    }


def _is_attr_value(value: Any) -> bool:
    return isinstance(value, (str, bytes, int, float, np.integer, np.floating))


def _as_text(value: Any) -> str:
    return str(value)
