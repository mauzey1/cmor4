"""Public dataset creation, file access, and output naming entry points."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .axis import Axis
from .datasetinfo import DatasetInfo
from .grid import Grid
from . import _assembly, _netcdf
from ._assembly import ValidationContext
from ._metadata import INTERNAL_DATASET_KEYS, DatasetMetadata
from ._templates import render_template
from ._time import add_time_delta, date_part, decode_time_value
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


def create_dataset(
    dataset: DatasetInfo,
    variable: Variable,
    axes: Sequence[Axis],
    data: Any,
    *,
    zfactors: Sequence[ZFactor] | None = None,
    grid: Grid | None = None,
    path: str | Path | None = None,
    attrs: Mapping[str, Any] | None = None,
    encoding: Mapping[str, Any] | None = None,
    **to_netcdf_kwargs: Any,
) -> Path:
    """Create and write a CMOR-like NetCDF file from metadata objects.

    When ``dataset`` was created by :meth:`ProjectTables.dataset_info`, this
    function uses the associated project tables to fill dataset-level defaults,
    add required scalar axes, validate table metadata for the supplied
    ``Variable``, ``Axis``, ``Grid``, and ``ZFactor`` records, validate the
    final global attributes, and verify that the generated xarray dataset
    contains the expected variables.

    For CMIP7 datasets, CMIP7-compliant chunking is automatically applied unless
    user-provided chunking is specified. User-provided chunking for CMIP7
    datasets is validated for compliance with cloud-optimization requirements.

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
    path:
        Explicit output path. If omitted, the CMOR-like path is rendered from
        metadata.
    attrs:
        Extra global attributes.
    encoding:
        Optional encoding parameters (chunksizes, compression, etc.) to apply
        to variables. For CMIP7 datasets, user-provided chunksizes are validated
        for compliance.
    **to_netcdf_kwargs:
        Additional keyword arguments forwarded to ``xarray.Dataset.to_netcdf``.

    Returns
    -------
    pathlib.Path
        Path to the written NetCDF file.

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
        requested metadata, or if CMIP7 chunking requirements are not met.
    """

    dataset, variable = _assembly._dataset_for_variable(dataset, variable)
    ctx = _assembly.validate_metadata(dataset, variable, axes, zfactors, grid)
    data_array = _assembly.validate_data_chunk(ctx, data)
    ds = create_dataset_from_validated_data(
        ctx,
        data_array,
        attrs=attrs,
        encoding=encoding,
    )
    return write_netcdf(ds, dataset, variable, path=path, **to_netcdf_kwargs)


def create_dataset_from_validated_data(
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

    return _assembly.build_dataset(ctx, data_array, attrs=attrs, encoding=encoding)


def write_netcdf(
    ds: xr.Dataset,
    dataset: DatasetMetadata,
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
        directory must already exist; CMOR will not create it.  The default
        behaviour (``create_subdirectories=True``) mirrors CMOR's original
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
    return _netcdf.write_dataset(ds, dataset, output_path, **to_netcdf_kwargs)


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
    encoding: Mapping[str, Any] | None = None,
    **to_netcdf_kwargs: Any,
) -> Path:
    """Create and write a CMOR-like NetCDF file from metadata objects.

    For CMIP7 datasets, CMIP7-compliant chunking is automatically applied unless
    user-provided chunking is specified via the encoding parameter.

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
    encoding:
        Optional encoding parameters (chunksizes, compression, etc.) to apply
        to variables. For CMIP7 datasets, user-provided chunksizes are validated
        for compliance.
    **to_netcdf_kwargs:
        Additional keyword arguments forwarded to ``xarray.Dataset.to_netcdf``.

    Returns
    -------
    pathlib.Path
        Path to the written NetCDF file.
    """

    return create_dataset(
        dataset,
        variable,
        axes,
        data,
        zfactors=zfactors,
        grid=grid,
        path=path,
        attrs=attrs,
        encoding=encoding,
        **to_netcdf_kwargs,
    )


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


def build_output_path(
    dataset: DatasetMetadata,
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
       All three project CVs shipped with CMOR (CMIP7, DRCDP) carry a
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

    dataset, variable = _assembly._dataset_for_variable(dataset, variable)
    root = Path(str(dataset.outpath or ".")).expanduser()
    tokens = _template_tokens(dataset, variable, ds)

    # Read CV DRS templates once; both may be None when the CV lacks a DRS
    # section (e.g. obs4MIPs) or when the CV is not project-backed.
    cv_path_tmpl: str | None = None
    cv_file_tmpl: str | None = None
    cv = getattr(getattr(dataset, "project", None), "cv", None)
    if cv is not None and cv._drs is not None:
        cv_path_tmpl, cv_file_tmpl = cv._drs.templates()

    path_template = str(
        dataset.output_path_template or cv_path_tmpl or DEFAULT_OUTPUT_PATH_TEMPLATE
    )
    file_template = str(
        dataset.output_file_template or cv_file_tmpl or DEFAULT_OUTPUT_FILE_TEMPLATE
    )

    if (
        tokens.get("time_range")
        and "<time_range>" not in file_template
        and "<time-range>" not in file_template
    ):
        file_template += "<time_range>"

    directory = render_template(path_template, tokens, "/").strip("/")
    filename = render_template(file_template, tokens, "_") + ".nc"

    return root / directory / filename


def string_from_template(
    template: str,
    dataset: DatasetMetadata,
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

    dataset, variable = _assembly._dataset_for_variable(dataset, variable)
    return render_template(template, _template_tokens(dataset, variable, ds), separator)


def _template_tokens(
    dataset: DatasetMetadata,
    variable: Variable,
    ds: xr.Dataset | None,
) -> dict[str, Any]:
    var_name, labels = variable.names()
    dataset_values = dataset.to_dict()
    frequency = str(dataset.frequency or variable.frequency or "fx")
    variant_label = dataset.variant_label()
    version = str(dataset.version or f"v{date.today():%Y%m%d}")
    time_range = _time_range(ds, frequency) if frequency != "fx" else None

    tokens = {
        str(key): value
        for key, value in (ds.attrs.items() if ds is not None else ())
        if not str(key).startswith("_")
    }
    tokens.update({
        str(key): value
        for key, value in dataset_values.items()
        if (key not in INTERNAL_DATASET_KEYS and not str(key).startswith("_"))
    })
    tokens.update({
        "branded_name": labels["branded_name"],
        "branded_variable": labels["branded_name"],
        "branded_variable_name": labels["branded_name"],
        "branding_suffix": labels.get("branding_suffix", ""),
        "frequency": frequency,
        "grid_label": dataset.grid_label or tokens.get("grid_label", "gn"),
        "member_id": dataset.member_id or variant_label,
        "region": dataset.region or tokens.get("region", "glb"),
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
