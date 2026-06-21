"""Grid metadata record for runtime grid dimensions and CF projections."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from pydantic import Field, model_validator

from .axis import Axis
from .metadata import MetadataModel, StrSeq, StrTuple

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


class Grid(MetadataModel):
    """Runtime grid dimensions and optional CF grid-mapping metadata.

    Grid objects may be constructed in two ways:

    **Axis-based** — supply ``axes`` with already-created :class:`~cmor4.Axis`
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
            dimensions=("x", "y"),
            mapping_name="lambert_azimuthal_equal_area",
            params={...},
        )

    Parameters
    ----------
    axes:
        Ordered :class:`~cmor4.Axis` objects that form the grid's indexing
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
        from .axis import Axis as _Axis  # local import avoids circular

        if self.axes:
            # Mark every dimensional axis as belonging to a grid.
            # Axis is frozen, so build updated copies if not already flagged.
            updated: list[_Axis] = [
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
            time_dims = tuple(str(d) for d in var_dims if str(d).lower() == "time")
            return time_dims + grid_dims
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
        routed through the :class:`~cmor4.Axis` machinery.

        Parameters
        ----------
        spatial_dims:
            Ordered list of the grid's spatial dimension names (e.g.
            ``["j", "i"]``).  These become the dimension tuple for the
            lat/lon coordinate arrays.
        coord_table:
            Optional :class:`~cmor4._tables.CoordinateTable` used to pull
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
                    raw = dict(entry.entry)
                    result: dict[str, Any] = {}
                    if "units" in raw:
                        result["units"] = str(raw["units"])
                    if "standard_name" in raw:
                        result["standard_name"] = str(raw["standard_name"])
                    if "long_name" in raw:
                        result["long_name"] = str(raw["long_name"])
                    return result
            # CF hard-coded defaults
            attrs: dict[str, Any] = {
                "units": default_units,
                "standard_name": default_sn,
            }
            return attrs

        if self.latitude is not None:
            lat_arr = np.asarray(self.latitude)
            lat_attrs = _cf_attrs("latitude", "degrees_north", "latitude")
            coords["latitude"] = (dims, lat_arr, lat_attrs)
            auxiliary_coord_names.append("latitude")

        if self.longitude is not None:
            lon_arr = np.asarray(self.longitude)
            lon_attrs = _cf_attrs("longitude", "degrees_east", "longitude")
            coords["longitude"] = (dims, lon_arr, lon_attrs)
            auxiliary_coord_names.append("longitude")

        # Vertex arrays — dimensioned (*spatial_dims, vertices_dim)
        if self.latitude_vertices is not None or self.longitude_vertices is not None:
            vdim = self.vertices_dim
            vert_dims = dims + (vdim,)

            if self.latitude_vertices is not None:
                blat_arr = np.asarray(self.latitude_vertices)
                blat_attrs = _cf_attrs("vertices_latitude", "degrees_north", "latitude")
                data_vars["vertices_latitude"] = (vert_dims, blat_arr, blat_attrs)
                # Set bounds pointer on the latitude coord
                if "latitude" in coords:
                    lat_entry = coords["latitude"]
                    updated_attrs = dict(lat_entry[2])
                    updated_attrs["bounds"] = "vertices_latitude"
                    coords["latitude"] = (lat_entry[0], lat_entry[1], updated_attrs)

            if self.longitude_vertices is not None:
                blon_arr = np.asarray(self.longitude_vertices)
                blon_attrs = _cf_attrs(
                    "vertices_longitude", "degrees_east", "longitude"
                )
                data_vars["vertices_longitude"] = (vert_dims, blon_arr, blon_attrs)
                if "longitude" in coords:
                    lon_entry = coords["longitude"]
                    updated_attrs = dict(lon_entry[2])
                    updated_attrs["bounds"] = "vertices_longitude"
                    coords["longitude"] = (lon_entry[0], lon_entry[1], updated_attrs)

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
        for key, val in self.params.items():
            if not _valid_param(str(key), val):
                continue
            if isinstance(val, (list, tuple)) and val:
                attrs[key] = val[0]
                if len(val) > 1 and val[1]:
                    attrs[f"{key}_units"] = val[1]
            else:
                attrs[key] = val
        return self.netcdf_attrs(attrs)


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
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
