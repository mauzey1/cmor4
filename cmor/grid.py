"""Grid metadata, CF projection registry, and grid shape validation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from .axis import Axis
from ._metadata import MetadataModel, StrSeq, StrTuple
from .exceptions import TableValidationError

if TYPE_CHECKING:
    import xarray as xr

    from ._coordinate_tables import GridMappingTableEntry as GridMappingEntry

_LATITUDE_PARAMS: frozenset[str] = frozenset({
    "grid_north_pole_latitude",
    "latitude_of_projection_origin",
    "standard_parallel",
    "standard_parallel1",
    "standard_parallel2",
})
_LONGITUDE_PARAMS: frozenset[str] = frozenset({
    "grid_north_pole_longitude",
    "longitude_of_prime_meridian",
    "longitude_of_central_meridian",
    "longitude_of_projection_origin",
    "north_pole_grid_longitude",
})
_NONNEG_PARAMS: frozenset[str] = frozenset({
    "scale_factor_at_central_meridian",
    "scale_factor_at_projection_origin",
})
_DEFAULT_HORIZONTAL_DIMS: frozenset[str] = frozenset({
    "lat",
    "latitude",
    "lon",
    "longitude",
})


class Grid(MetadataModel):
    """Runtime grid dimensions and optional CF grid-mapping metadata.

    Grid objects may be constructed in two ways:

    **Axis-based** — supply ``axes`` with already-created :class:`~cmor.Axis`
    instances.  The grid owns those axes as its indexing dimensions; it
    derives ``dimensions`` from them automatically and marks each axis with
    ``isgridaxis=True``.  This is the recommended path for curvilinear and
    unstructured grids::

        i_axis = project.axis("i_index", values=np.arange(192))
        j_axis = project.axis("j_index", values=np.arange(144))

        grid = project.grid(
            axes=[j_axis, i_axis],
            latitude=lat_2d,
            longitude=lon_2d,
            latitude_vertices=blat_3d,
            longitude_vertices=blon_3d,
        )

    **Name-based** — supply ``dimensions`` as a tuple of string names.  The
    grid references axes that the caller keeps separately, matched by name at
    validation time.  Useful for grid-mapping-only grids with no curvilinear
    lat/lon::

        grid = project.grid(
            dimensions=("y", "x"),
            mapping_name="lambert_azimuthal_equal_area",
            params={...},
        )

    Parameters
    ----------
    axes:
        Ordered :class:`~cmor.Axis` objects that form the grid's indexing
        dimensions.  When supplied, ``dimensions`` is derived from them and
        each axis is flagged ``isgridaxis=True``.
    dimensions:
        Spatial (non-time) dimension names.  Derived automatically from
        ``axes`` when ``axes`` is provided; supply explicitly when using the
        name-based path.
    name, table_entry, mapping_entry:
        Grid table entry name selectors.
    mapping_var:
        Name of the scalar grid-mapping variable to write.
    mapping_name, grid_mapping_name:
        CF ``grid_mapping_name``.
    coordinates:
        Auxiliary coordinate names.
    params:
        Grid-mapping parameter dict.  Each value may be a scalar or a
        ``(value, units)`` tuple.
    attrs:
        Extra NetCDF attributes for the grid-mapping variable.
    latitude, longitude:
        Optional 2-D geographic coordinate arrays on the grid.
    latitude_vertices, longitude_vertices:
        Optional cell-vertex arrays (shape ``(*spatial, n_vertices)``).
    vertices_dim:
        Name for the vertices dimension (default ``"vertices"``).
    """

    axes: list[Axis] = Field(default_factory=list)
    dimensions: StrTuple = None
    name: str | None = None
    table_entry: str | None = None
    mapping_entry: str | None = None
    mapping_var: str | None = None
    mapping_name: str | None = None
    grid_mapping_name: str | None = None
    coordinates: StrSeq = None
    params: dict[str, Any] = Field(default_factory=dict)
    attrs: dict[str, Any] = Field(default_factory=dict)
    latitude: Any = None
    longitude: Any = None
    latitude_vertices: Any = None
    longitude_vertices: Any = None
    vertices_dim: str = "vertices"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _sync_axes_and_dimensions(self) -> Grid:
        """Derive dimensions from axes, mark axes as isgridaxis."""
        if self.axes:
            # Mark every dimensional axis as belonging to a grid.
            # Axis is frozen, so build updated copies if not already flagged.
            updated: list[Axis] = [
                a if a.isgridaxis else a.updated(isgridaxis=True) for a in self.axes
            ]
            object.__setattr__(self, "axes", updated)

            # Derive dimensions tuple from axes when the caller did not supply it.
            if not self.dimensions:
                dims = tuple(str(a.out_name or a.name) for a in updated)
                object.__setattr__(self, "dimensions", dims)

        return self

    @model_validator(mode="after")
    def _check_spatial_arrays(self) -> Grid:
        """Validate that latitude / longitude shapes are consistent."""
        lat, lon = self.latitude, self.longitude
        if lat is None or lon is None:
            return self
        la, lo = np.asarray(lat), np.asarray(lon)
        if la.shape != lo.shape:
            raise ValueError(
                f"latitude shape {la.shape} does not match longitude shape {lo.shape}."
            )
        ndims = len(self.dimensions) if self.dimensions is not None else None
        if ndims is not None and la.ndim != ndims:
            raise ValueError(
                f"lat/lon arrays have {la.ndim} dimension(s) but "
                f"{ndims} grid dimension(s) were specified."
            )
        for name, arr, ref in (
            ("latitude_vertices", self.latitude_vertices, la),
            ("longitude_vertices", self.longitude_vertices, lo),
        ):
            if arr is not None:
                a = np.asarray(arr)
                if a.shape[: ref.ndim] != ref.shape:
                    raise ValueError(
                        f"{name} shape {a.shape} is incompatible "
                        f"with coordinate shape {ref.shape}."
                    )
        return self

    # ------------------------------------------------------------------
    # NetCDF output helpers
    # ------------------------------------------------------------------

    @property
    def variable_name(self) -> str:
        """Return the output grid-mapping variable name (default: ``"crs"``)."""
        return str(self.mapping_var or "crs")

    def variable_dimensions(self, variable: Any) -> tuple[str, ...] | None:
        """Return the full dimension tuple for the data variable.

        Combines time from *variable* with the grid's spatial dimensions.
        When ``axes`` is set the spatial dimension names are derived from the
        axis ``out_name`` (or ``name``) values; otherwise ``dimensions`` is
        used directly.
        """
        if self.axes:
            grid_dims = tuple(str(a.out_name or a.name) for a in self.axes)
        elif self.dimensions:
            grid_dims = tuple(str(n) for n in self.dimensions)
        else:
            dims = variable.dimensions
            return tuple(str(n) for n in dims) if dims else None

        var_dims = variable.dimensions
        if var_dims:
            grid_dim_names = {name.lower() for name in grid_dims}
            non_spatial_dims = tuple(
                str(dim)
                for dim in var_dims
                if str(dim).lower() not in grid_dim_names | _DEFAULT_HORIZONTAL_DIMS
            )
            return non_spatial_dims + grid_dims
        return grid_dims

    def to_dataset_coords(
        self,
        spatial_dims: list[str],
        coord_table: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Produce xarray coord/data-var entries for the grid's lat/lon data.

        This mirrors what CMOR3's ``cmor_grid`` does: latitude, longitude,
        vertices_latitude, and vertices_longitude are registered as
        **grid-owned variables** dimensioned by the grid's own spatial
        dimensions (and a ``vertices`` dimension for the cell-corner arrays).
        They are written directly as xarray dataset entries rather than being
        routed through the :class:`~cmor.Axis` machinery.

        Parameters
        ----------
        spatial_dims:
            Ordered list of the grid's spatial dimension names (e.g.
            ``["j", "i"]``).  These become the dimension tuple for the
            lat/lon coordinate arrays.
        coord_table:
            Optional :class:`~cmor._coordinate_tables.CoordinateTable` used to pull
            CF metadata (``standard_name``, ``long_name``, ``units``) for
            latitude and longitude from the project tables.  When ``None``,
            hard-coded CF defaults are used.

        Returns
        -------
        coords : dict
            Mapping of coordinate-variable name → ``(dims, data, attrs)``
            tuples ready for ``xr.Dataset(coords=...)``.
        data_vars : dict
            Mapping of data-variable name → ``(dims, data, attrs)`` tuples
            ready for ``xr.Dataset(data_vars=...)``.  Holds
            ``vertices_latitude`` / ``vertices_longitude`` when vertex arrays
            were supplied.
        auxiliary_coord_names : list[str]
            Names that should be listed in the variable's ``coordinates``
            attribute (lat, lon, and their vertex counterparts when present).
        """
        coords: dict[str, Any] = {}
        data_vars: dict[str, Any] = {}
        auxiliary_coord_names: list[str] = []

        dims = tuple(spatial_dims)

        def _cf_attrs(name: str, default_units: str, default_sn: str) -> dict[str, Any]:
            """Return CF attrs from coord table when available, else defaults."""
            if coord_table is not None:
                entry = coord_table.resolve_grid_coord({
                    "name": name,
                    "grid_coordinate": name,
                }) or coord_table.resolve_coord({"name": name})
                if entry is not None:
                    result: dict[str, Any] = {}
                    for key in ("units", "standard_name", "long_name"):
                        value = getattr(entry, key)
                        if value not in (None, ""):
                            result[key] = str(value)
                    if not result.get("units"):
                        result["units"] = default_units
                    if not result.get("standard_name"):
                        result["standard_name"] = default_sn
                    return result
            # CF hard-coded defaults
            attrs: dict[str, Any] = {
                "units": default_units,
                "standard_name": default_sn,
            }
            return attrs

        for name, units, values, vertices in (
            ("latitude", "degrees_north", self.latitude, self.latitude_vertices),
            ("longitude", "degrees_east", self.longitude, self.longitude_vertices),
        ):
            if values is not None:
                coords[name] = (dims, np.asarray(values), _cf_attrs(name, units, name))
                auxiliary_coord_names.append(name)
            if vertices is not None:
                bounds_name = f"vertices_{name}"
                data_vars[bounds_name] = (
                    dims + (self.vertices_dim,),
                    np.asarray(vertices),
                    _cf_attrs(bounds_name, units, name),
                )
                if name in coords:
                    entry = coords[name]
                    coords[name] = (
                        entry[0],
                        entry[1],
                        {**entry[2], "bounds": bounds_name},
                    )

        return coords, data_vars, auxiliary_coord_names

    @property
    def has_mapping(self) -> bool:
        """True when this grid needs a CF grid-mapping variable."""
        return bool(
            self.mapping_name or self.grid_mapping_name or self.params or self.attrs
        )

    def mapping_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the grid-mapping scalar variable."""
        attrs = self.netcdf_attrs(self.attrs)
        mname = self.mapping_name or self.grid_mapping_name
        if mname:
            attrs["grid_mapping_name"] = mname

        skip_params: set[str] = set()
        if "standard_parallel1" in self.params and "standard_parallel2" in self.params:
            skip_params.update({"standard_parallel1", "standard_parallel2"})
            if "standard_parallel" not in self.params:
                sp1 = self.params["standard_parallel1"]
                sp2 = self.params["standard_parallel2"]
                if _valid_param("standard_parallel1", sp1) and _valid_param(
                    "standard_parallel2", sp2
                ):
                    if isinstance(sp1, dict):
                        sp1 = sp1.get("value", sp1.get("values"))
                    if isinstance(sp2, dict):
                        sp2 = sp2.get("value", sp2.get("values"))
                    if (
                        isinstance(sp1, (list, tuple))
                        and len(sp1) == 2
                        and isinstance(sp1[1], str)
                    ):
                        sp1 = sp1[0]
                    if (
                        isinstance(sp2, (list, tuple))
                        and len(sp2) == 2
                        and isinstance(sp2[1], str)
                    ):
                        sp2 = sp2[0]
                    attrs["standard_parallel"] = [
                        sp1,
                        sp2,
                    ]

        for key, val in self.params.items():
            key = str(key)
            if key in skip_params:
                continue
            if not _valid_param(str(key), val):
                continue
            if isinstance(val, dict):
                val = val.get("value", val.get("values"))
            if (
                isinstance(val, (list, tuple))
                and len(val) == 2
                and isinstance(val[1], str)
            ):
                val = val[0]
            attrs[key] = val

        result: dict[str, Any] = {}
        for key, value in attrs.items():
            if MetadataModel.is_netcdf_attr_value(value):
                result[str(key)] = value
                continue
            if isinstance(value, np.ndarray):
                if value.dtype.kind in {"i", "u", "f"} and value.ndim > 0:
                    result[str(key)] = value.tolist()
                continue
            if isinstance(value, (list, tuple)) and value:
                arr = np.asarray(value)
                if arr.dtype.kind in {"i", "u", "f"} and arr.ndim > 0:
                    result[str(key)] = list(value)
        return result


def _valid_param(name: str, value: Any) -> bool:
    num = _primary_num(value)
    if num is None:
        return True
    if name in _LATITUDE_PARAMS and not -90.0 <= num <= 90.0:
        warnings.warn(
            f"{name} parameter must be between -90 and 90 degrees_north; "
            "it will not be set.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    if name in _LONGITUDE_PARAMS and not -180.0 <= num <= 180.0:
        warnings.warn(
            f"{name} parameter must be between -180 and 180 degrees_east; "
            "it will not be set.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    if name in _NONNEG_PARAMS and num < 0.0:
        warnings.warn(
            f"{name} parameter must be positive; it will not be set.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    return True


def _primary_num(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value", value.get("values"))
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], str)
    ):
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_COMMON_GRID_MAPPING_ATTRIBUTES: frozenset[str] = frozenset({
    "crs_wkt",
    "earth_radius",
    "GeoTransform",
    "geographic_coordinate_system_name",
    "geographic_crs_name",
    "geoid_name",
    "geopotential_datum_name",
    "grid_mapping_name",
    "horizontal_datum_name",
    "inverse_flattening",
    "long_name",
    "longitude_of_prime_meridian",
    "prime_meridian_name",
    "projected_coordinate_system_name",
    "projected_crs_name",
    "reference_ellipsoid_name",
    "semi_major_axis",
    "semi_minor_axis",
    "spatial_ref",
    "towgs84",
})


_TEXT_GRID_MAPPING_ATTRIBUTES: frozenset[str] = frozenset({
    "crs_wkt",
    "fixed_angle_axis",
    "GeoTransform",
    "geographic_coordinate_system_name",
    "geographic_crs_name",
    "geoid_name",
    "geopotential_datum_name",
    "grid_mapping_name",
    "horizontal_datum_name",
    "long_name",
    "prime_meridian_name",
    "projected_coordinate_system_name",
    "projected_crs_name",
    "reference_ellipsoid_name",
    "spatial_ref",
    "sweep_angle_axis",
})


_GRID_MAPPING_ATTRIBUTES: dict[str, frozenset[str]] = {
    "albers_conical_equal_area": frozenset({
        "standard_parallel",
        "longitude_of_central_meridian",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "azimuthal_equidistant": frozenset({
        "longitude_of_projection_origin",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "geostationary": frozenset({
        "latitude_of_projection_origin",
        "longitude_of_projection_origin",
        "perspective_point_height",
        "false_easting",
        "false_northing",
        "sweep_angle_axis",
        "fixed_angle_axis",
    }),
    "lambert_azimuthal_equal_area": frozenset({
        "longitude_of_projection_origin",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "lambert_conformal_conic": frozenset({
        "standard_parallel",
        "longitude_of_central_meridian",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "lambert_cylindrical_equal_area": frozenset({
        "longitude_of_central_meridian",
        "standard_parallel",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "latitude_longitude": frozenset(),
    "mercator": frozenset({
        "longitude_of_projection_origin",
        "standard_parallel",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "oblique_mercator": frozenset({
        "azimuth_of_central_line",
        "latitude_of_projection_origin",
        "longitude_of_projection_origin",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "orthographic": frozenset({
        "longitude_of_projection_origin",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "polar_stereographic": frozenset({
        "longitude_of_projection_origin",
        "straight_vertical_longitude_from_pole",
        "latitude_of_projection_origin",
        "standard_parallel",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "rotated_latitude_longitude": frozenset({
        "grid_north_pole_latitude",
        "grid_north_pole_longitude",
        "north_pole_grid_longitude",
    }),
    "sinusoidal": frozenset({
        "longitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "stereographic": frozenset({
        "longitude_of_projection_origin",
        "latitude_of_projection_origin",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "transverse_mercator": frozenset({
        "scale_factor_at_central_meridian",
        "longitude_of_central_meridian",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "vertical_perspective": frozenset({
        "latitude_of_projection_origin",
        "longitude_of_projection_origin",
        "perspective_point_height",
        "false_easting",
        "false_northing",
    }),
}


def _allowed_grid_mapping_attributes(mapping_name: str) -> frozenset[str] | None:
    """Return CF-1.12 attributes allowed for *mapping_name*, or ``None``."""
    specific = _GRID_MAPPING_ATTRIBUTES.get(mapping_name)
    if specific is None:
        return None
    allowed = set(_COMMON_GRID_MAPPING_ATTRIBUTES)
    allowed.update(specific)
    if "standard_parallel" in allowed:
        allowed.update({"standard_parallel1", "standard_parallel2"})
    return frozenset(allowed)


def _validate_grid_mapping(
    entry: GridMappingEntry | None,
    data: Grid | Mapping[str, Any],
    *,
    axes: Sequence[Axis] = (),
    dimensions: Sequence[Any] | str | None = None,
) -> None:
    """Validate grid-mapping metadata against a grid-table mapping entry."""
    if isinstance(data, Grid):
        mapping_name = data.mapping_name
        grid_mapping_name = data.grid_mapping_name
        params = dict(data.params)
        if dimensions is None:
            dimensions = data.dimensions
        if not axes:
            axes = data.axes
    else:
        mapping_name = data.get("mapping_name")
        grid_mapping_name = data.get("grid_mapping_name")
        params = dict(data.get("params") or {})

    entry_name = entry.name if entry is not None else None
    if entry is not None:
        for key, value in (
            ("mapping_name", mapping_name),
            ("grid_mapping_name", grid_mapping_name),
        ):
            expected = getattr(entry, key)
            if expected not in (None, "", [], ()) and value is not None:
                if str(value) != str(expected):
                    raise TableValidationError(
                        f"grid mapping {entry.name!r} {key}={value!r} does "
                        f"not match table value {expected!r}."
                    )

    supplied_mapping_name = (
        grid_mapping_name or mapping_name or params.get("grid_mapping_name")
    )
    table_mapping_name = None
    if entry is not None:
        table_mapping_name = entry.grid_mapping_name or entry.mapping_name
        if (
            table_mapping_name is None
            and _allowed_grid_mapping_attributes(entry.name) is not None
        ):
            table_mapping_name = entry.name
        if (
            table_mapping_name is not None
            and supplied_mapping_name is not None
            and str(supplied_mapping_name) != str(table_mapping_name)
        ):
            raise TableValidationError(
                f"grid mapping {entry.name!r} grid_mapping_name="
                f"{supplied_mapping_name!r} does not match table value "
                f"{table_mapping_name!r}."
            )

    cf_mapping_name = supplied_mapping_name or table_mapping_name
    cf_mapping_name = str(cf_mapping_name) if cf_mapping_name else None
    cf_allowed_params: frozenset[str] | None = None
    if cf_mapping_name is not None:
        cf_allowed_params = _allowed_grid_mapping_attributes(cf_mapping_name)
        if cf_allowed_params is None:
            raise TableValidationError(
                f"grid_mapping_name {cf_mapping_name!r} is not a CF-1.12 "
                "grid mapping name."
            )

    required_params = (
        set(entry.required_parameter_names()) if entry is not None else set()
    )
    optional_params = (
        set(entry.optional_parameter_names()) if entry is not None else set()
    )
    text_params = set(entry.text_parameter_names()) if entry is not None else set()
    allowed_params = set(required_params | optional_params | text_params)
    if cf_allowed_params is not None:
        allowed_params.update(cf_allowed_params)
        text_params.update(_TEXT_GRID_MAPPING_ATTRIBUTES)
    if "standard_parallel" in allowed_params:
        allowed_params.update({"standard_parallel1", "standard_parallel2"})
    if {"standard_parallel1", "standard_parallel2"} & allowed_params:
        allowed_params.add("standard_parallel")

    if allowed_params:
        unknown = sorted(
            str(param) for param in params if str(param) not in allowed_params
        )
        if unknown:
            label = entry_name or cf_mapping_name or "<unknown>"
            raise TableValidationError(
                f"grid mapping {label!r} does not allow projection "
                f"parameter(s): {', '.join(unknown)}."
            )

    # Legacy grid tables declare projection parameters as ``parameter1``,
    # ``parameter2``, ... and callers may provide those values as top-level
    # grid arguments.  GridTable.build() later moves them into ``Grid.params``;
    # include the same effective values here so construction-time validation
    # cannot be bypassed by using the table-native input form.
    effective_params = dict(params)
    if not isinstance(data, Grid):
        for param in allowed_params:
            if param not in effective_params and data.get(param) is not None:
                effective_params[param] = data[param]

    supplied_params = {
        str(key) for key, value in effective_params.items() if value is not None
    }
    missing_params: list[str] = []
    for param in required_params:
        supplied = param in supplied_params
        if param == "standard_parallel":
            supplied = supplied or (
                {"standard_parallel1", "standard_parallel2"} <= supplied_params
            )
        elif param in {"standard_parallel1", "standard_parallel2"}:
            supplied = supplied or "standard_parallel" in supplied_params
        if not supplied:
            missing_params.append(param)
    if missing_params:
        label = entry_name or cf_mapping_name or "<unknown>"
        warnings.warn(
            f"grid mapping {label!r} is missing table-declared "
            f"parameter(s): {', '.join(missing_params)}. Table defaults will "
            "be used when available.",
            RuntimeWarning,
            stacklevel=3,
        )

    for name, value in effective_params.items():
        name = str(name)
        if allowed_params and name not in allowed_params:
            continue
        if isinstance(value, Mapping):
            value = value.get("value", value.get("values"))
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[1], str)
        ):
            value = value[0]
        if name in text_params:
            if not isinstance(value, str):
                label = entry_name or cf_mapping_name or "<unknown>"
                raise TableValidationError(
                    f"grid mapping {label!r} parameter {name!r} must be a string."
                )
            continue
        arr = np.asarray(value)
        values = arr.reshape(-1).tolist() if arr.ndim else [arr.item()]
        for item in values:
            try:
                float(item)
            except (TypeError, ValueError) as exc:
                label = entry_name or cf_mapping_name or "<unknown>"
                raise TableValidationError(
                    f"grid mapping {label!r} parameter {name!r} must be numeric."
                ) from exc

    required_axes = entry.required_axis_names() if entry is not None else ()
    if not required_axes:
        return
    assert entry is not None
    if axes:
        if len(axes) != len(required_axes):
            raise TableValidationError(
                f"grid mapping {entry.name!r} requires {len(required_axes)} "
                f"grid axis/axes ({', '.join(required_axes)}) but "
                f"{len(axes)} were supplied."
            )
        for index, (axis, expected) in enumerate(zip(axes, required_axes, strict=True)):
            expected_text = str(expected)
            if expected_text.upper() in {"X", "Y", "Z", "T"}:
                matches = str(axis.axis or "").upper() == expected_text.upper()
            else:
                matches = expected_text in {
                    str(value)
                    for value in (
                        axis.name,
                        axis.table_entry,
                        axis.axis_entry,
                        axis.coordinate,
                        axis.out_name,
                        axis.generic_level_name,
                    )
                    if value
                }
            if not matches:
                actual = axis.axis or axis.table_entry or axis.name or axis.out_name
                raise TableValidationError(
                    f"grid mapping {entry.name!r} requires grid axis "
                    f"{expected!r} at position {index}; got {actual!r}."
                )
    elif dimensions is not None:
        dims = (dimensions,) if isinstance(dimensions, str) else tuple(dimensions)
        if len(dims) != len(required_axes):
            raise TableValidationError(
                f"grid mapping {entry.name!r} requires {len(required_axes)} "
                f"grid axis/axes ({', '.join(required_axes)}) but "
                f"dimensions={dims!r} declares {len(dims)}."
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


def _validate_grid_dimensions(
    grid: Grid,
    axes: Sequence[Axis],
) -> None:
    """Validate that grid dimensions correspond to spatial axes in the right order.

    When ``grid.axes`` is non-empty the dimensional axes are already owned by
    the grid; validation checks that each axis carries coordinate values and
    that the lat/lon array shapes match the axis lengths in declared order.

    When ``grid.axes`` is empty (name-based path) every name in
    ``grid.dimensions`` must resolve to one of the supplied *axes* by any
    recognised name variant, and the same shape checks apply.

    Parameters
    ----------
    grid
        Grid whose dimensional axes are to be validated.
    axes
        The axis objects supplied alongside the grid (non-grid axes such as
        time).  Only consulted on the name-based path.

    Raises
    ------
    TableValidationError
        If a grid dimension name cannot be matched to any axis, or if the
        lat/lon array size along a given axis index does not match the axis
        length.
    """
    if getattr(grid, "axes", None):
        # --- Axis-based path: grid owns its dimensional Axis objects -------
        grid_axes = list(grid.axes)

        lat = grid.latitude
        lon = grid.longitude
        for array, label in ((lat, "latitude"), (lon, "longitude")):
            if array is None:
                continue
            arr = np.asarray(array)
            for i, axis in enumerate(grid_axes):
                axis_len = len(axis.values_array())
                if arr.shape[i] != axis_len:
                    dim_name = str(axis.out_name or axis.name)
                    raise TableValidationError(
                        f"Grid {label} array shape {arr.shape} does not match "
                        f"the axis lengths implied by grid.axes. "
                        f"Axis {dim_name!r} (index {i}) has {axis_len} "
                        f"coordinate values but {label} has size "
                        f"{arr.shape[i]} along that axis."
                    )
        return

    # --- Name-based path: match dimension strings to supplied axes ----------
    if not grid.dimensions:
        return

    # Build a lookup from every recognised name variant to the axis object.
    name_to_axis: dict[str, Axis] = {}
    for axis in axes:
        for value in (
            axis.name,
            axis.table_entry,
            axis.axis_entry,
            axis.coordinate,
            axis.out_name,
            axis.generic_level_name,
        ):
            if value:
                name_to_axis.setdefault(str(value), axis)

    grid_dims = [str(d) for d in grid.dimensions]

    for dim_name in grid_dims:
        if dim_name not in name_to_axis:
            raise TableValidationError(
                f"Grid dimension {dim_name!r} does not correspond to any "
                "of the supplied axes.  Each grid spatial dimension must "
                "match an axis by name, out_name, or table_entry."
            )

    lat = grid.latitude
    lon = grid.longitude
    for array, label in ((lat, "latitude"), (lon, "longitude")):
        if array is None:
            continue
        arr = np.asarray(array)
        for i, dim_name in enumerate(grid_dims):
            axis = name_to_axis[dim_name]
            axis_len = len(axis.values_array())
            if arr.shape[i] != axis_len:
                raise TableValidationError(
                    f"Grid {label} array shape {arr.shape} does not match "
                    f"the axis lengths implied by grid dimensions "
                    f"{grid_dims!r}. "
                    f"Dimension {dim_name!r} (index {i}) has {axis_len} "
                    f"coordinate values but {label} has size "
                    f"{arr.shape[i]} along that axis."
                )
