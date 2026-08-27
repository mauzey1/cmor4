from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .utils.templates import render_template
from .utils.time_utils import decode_time_value, add_time_delta, date_part
from .axis import Axis
from .datasetinfo import DatasetInfo
from .grid import Grid
from .utils.construction import (
    add_grid_coords,
    add_zfactor,
    build_axis_mappings,
    derive_forecast_coords,
    set_formula_terms,
)
from .utils.validation import (
    ValidationContext,
    _dataset_for_variable,
    validate_data_chunk,
    validate_final_dataset,
    validate_metadata,
)
from .utils.dataset_metadata import DatasetMetadata, INTERNAL_DATASET_KEYS
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
    attrs: Mapping[str, Any] | None = None,
    encoding: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Create an xarray dataset from metadata objects.

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
    attrs:
        Extra global attributes.
    encoding:
        Optional encoding parameters (chunksizes, compression, etc.) to apply
        to variables. For CMIP7 datasets, user-provided chunksizes are validated
        for compliance.

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
        requested metadata, or if CMIP7 chunking requirements are not met.
    """

    ctx = validate_metadata(dataset, variable, axes, zfactors, grid)
    data_array = validate_data_chunk(ctx, data)
    return create_dataset_from_validated_data(
        ctx,
        data_array,
        attrs=attrs,
        encoding=encoding,
    )


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
    ) = build_axis_mappings(axes)

    forecast_coord_name = derive_forecast_coords(
        axes,
        coords,
        getattr(dataset, "project", None),
        calendar=str(dataset.get("calendar", "standard") or "standard"),
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
            np.int32(0),
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

    missing_value = variable.missing_value or variable.fill_value
    if missing_value is not None:
        ds[var_name].attrs["missing_value"] = missing_value
        ds[var_name].encoding["_FillValue"] = missing_value

    # Handle chunking with CMIP7 compliance
    from .utils.chunking import (
        calculate_cmip7_chunks,
        is_cmip7_dataset,
        validate_cmip7_chunksizes,
    )

    # Extract user-provided chunksizes for the data variable.
    # Priority: variable-specific encoding["var_name"]["chunksizes"] >
    #           top-level encoding["chunksizes"] >
    #           variable metadata (chunksizes/chunks)
    user_provided_chunks = None
    if encoding:
        var_encoding = encoding.get(var_name)
        if isinstance(var_encoding, Mapping) and "chunksizes" in var_encoding:
            user_provided_chunks = var_encoding["chunksizes"]
        elif "chunksizes" in encoding and not isinstance(
            encoding["chunksizes"], Mapping
        ):
            user_provided_chunks = encoding["chunksizes"]

    # Fallback to variable metadata for chunksizes if not in encoding
    chunksizes = user_provided_chunks or variable.chunksizes or variable.chunks

    # For CMIP7 datasets, validate user-provided chunksizes meet the
    # ≥4 MiB requirement. Allow all other encoding parameters to pass through.
    if user_provided_chunks and is_cmip7_dataset(dataset):
        data_array = ds[var_name]
        validate_cmip7_chunksizes(
            user_provided_chunks,
            var_name,
            data_array.dims,
            data_array.shape,
            data_array.dtype,
        )

    # Auto-generate CMIP7-compliant chunks if no chunksizes were provided
    # (neither in encoding nor in variable metadata)
    if chunksizes is None and is_cmip7_dataset(dataset):
        data_array = ds[var_name]
        cmip7_chunks = calculate_cmip7_chunks(
            data_array.dims,
            data_array.shape,
            data_array.dtype,
            data_array.dims,
        )
        ds[var_name].encoding["chunksizes"] = tuple(
            cmip7_chunks[d] for d in data_array.dims
        )
    elif chunksizes:
        ds[var_name].encoding["chunksizes"] = tuple(int(value) for value in chunksizes)

    # For CMIP7 datasets, identify time-related variables that must have
    # a single chunk (time coordinate and its bounds/climatology).
    if is_cmip7_dataset(dataset):
        protected_chunks: set[str] = set()
        for coord_name in ds.coords:
            coord_name_str = str(coord_name)
            lower = coord_name_str.lower()
            # Only protect time coordinates, not other coordinates
            if lower != "time" and not lower.startswith("time"):
                continue
            protected_chunks.add(coord_name_str)
            coord = ds[coord_name_str]
            # Also protect time bounds and climatology bounds
            for attr_name in ("bounds", "climatology"):
                bounds_name = coord.attrs.get(attr_name)
                if bounds_name is not None and str(bounds_name) in ds:
                    protected_chunks.add(str(bounds_name))
    else:
        protected_chunks = set()

    # Apply user-provided encoding to variables
    if encoding:
        for name in ds.variables:
            name_str = str(name)

            # Start with top-level encoding parameters, excluding chunksizes
            # and nested dicts.
            # This mimics xarray's encoding behavior
            for key, value in encoding.items():
                if key == "chunksizes" or isinstance(value, Mapping):
                    continue
                ds[name_str].encoding[key] = value

            # Apply variable-specific encoding overrides
            var_specific = encoding.get(name_str)
            if isinstance(var_specific, Mapping):
                for key, value in var_specific.items():
                    if key == "chunksizes":
                        # Handle chunksizes for non-data variables
                        if name_str == var_name:
                            # Data variable chunksizes already applied above
                            continue

                        array = ds[name_str]
                        if len(value) != array.ndim:
                            raise ValueError(
                                f"Variable {name_str!r}: chunksizes length "
                                f"{len(value)} does not match dimensions {array.dims}"
                            )
                        normalized_chunksizes = tuple(int(v) for v in value)

                        # For CMIP7, enforce single chunk for time and time_bnds
                        if (
                            name_str in protected_chunks
                            and normalized_chunksizes != array.shape
                        ):
                            raise ValueError(
                                f"Variable {name_str!r}: CMIP7 requires time "
                                "coordinates and time bounds to have a single "
                                f"chunk. Got chunk sizes {normalized_chunksizes} "
                                f"but variable shape is {array.shape}."
                            )
                        array.encoding["chunksizes"] = normalized_chunksizes
                    else:
                        # Apply other variable-specific encoding parameters
                        ds[name_str].encoding[key] = value

    # For CMIP7, automatically apply single-chunk encoding to time coordinates
    # and time bounds if not already set by user
    if protected_chunks:
        for name in protected_chunks:
            array = ds[name]
            # Skip scalar variables and empty arrays
            if array.ndim != 0 and not any(size == 0 for size in array.shape):
                # Set chunk size to full array shape (single chunk)
                array.encoding["chunksizes"] = tuple(int(size) for size in array.shape)

    validate_final_dataset(ds, ctx, zfactor_names)

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
    encoding: Mapping[str, Any] | None = None,
    **to_netcdf_kwargs: Any,
) -> tuple[xr.Dataset, Path]:
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
    tuple[xr.Dataset, pathlib.Path]
        The in-memory dataset and path to the written NetCDF file.
    """

    dataset, variable = _dataset_for_variable(dataset, variable)
    ds = create_dataset(
        dataset,
        variable,
        axes,
        data,
        zfactors=zfactors,
        grid=grid,
        attrs=attrs,
        encoding=encoding,
    )
    output_path = write_netcdf(ds, dataset, variable, path=path, **to_netcdf_kwargs)
    return ds, output_path


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

    dataset, variable = _dataset_for_variable(dataset, variable)
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

    dataset, variable = _dataset_for_variable(dataset, variable)
    return render_template(template, _template_tokens(dataset, variable, ds), separator)


def _template_tokens(
    dataset: DatasetMetadata,
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
