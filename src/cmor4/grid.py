from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Mapping
import warnings

from ._table_utils import is_table_value
from .metadata import _MetadataRecord


_LATITUDE_PARAMETERS = {
    "grid_north_pole_latitude",
    "latitude_of_projection_origin",
    "standard_parallel",
    "standard_parallel1",
    "standard_parallel2",
}

_LONGITUDE_PARAMETERS = {
    "grid_north_pole_longitude",
    "longitude_of_prime_meridian",
    "longitude_of_central_meridian",
    "longitude_of_projection_origin",
    "north_pole_grid_longitude",
}

_NON_NEGATIVE_PARAMETERS = {
    "scale_factor_at_central_meridian",
    "scale_factor_at_projection_origin",
}


@dataclass(frozen=True)
class Grid(_MetadataRecord):
    """Runtime grid dimensions and optional grid-mapping metadata.

    Parameters
    ----------
    dimensions
        Output dimensions used for the data variable.
    name
        Requested grid mapping entry name.
    table_entry
        Grid table entry name selector.
    mapping_entry
        Grid table entry name selector.
    mapping_var
        Name of the scalar grid-mapping variable to write.
    mapping_name
        CF grid mapping name.
    grid_mapping_name
        CF grid mapping name.
    coordinates
        Auxiliary coordinate names associated with the grid.
    params
        Grid-mapping parameter values.
    attrs
        Extra NetCDF attributes for the grid-mapping variable.
    latitude
        Optional 2D array of latitude values on the grid. When provided, this
        will be added as an auxiliary coordinate with the grid's spatial
        dimensions.
    longitude
        Optional 2D array of longitude values on the grid. When provided, this
        will be added as an auxiliary coordinate with the grid's spatial
        dimensions.
    latitude_vertices
        Optional 3D array of latitude cell vertices. Shape should be
        ``(*latitude.shape, n_vertices)`` where n_vertices is typically 4.
    longitude_vertices
        Optional 3D array of longitude cell vertices. Shape should be
        ``(*longitude.shape, n_vertices)`` where n_vertices is typically 4.
    vertices_dim
        Name for the vertices dimension. Defaults to ``"vertices"``.
    extra
        Additional mapping keys preserved by the metadata record.
    project
        Optional project tables used to resolve and merge grid metadata during
        construction.

    Examples
    --------
    Create a grid with embedded lat/lon coordinates for a projected grid::

        grid = project.grid(
            dimensions=["time", "x", "y"],
            mapping_name="lambert_azimuthal_equal_area",
            params={
                "latitude_of_projection_origin": [90.0, "degrees_north"],
                "longitude_of_projection_origin": [0.0, "degrees_east"],
            },
            latitude=lat_values,  # shape: (nx, ny)
            longitude=lon_values,
            latitude_vertices=lat_verts,  # shape: (nx, ny, 4)
            longitude_vertices=lon_verts,
        )

        # Now just pass x, y, time axes - grid handles lat/lon
        axes = [
            project.axis("time", ...),
            project.axis("x", ...),
            project.axis("y", ...),
        ]
        ds = cmor4.create_dataset(
            dataset_info, variable, axes, data, grid=grid
        )
    """

    dimensions: tuple[str, ...] | list[str] | None = None
    name: str | None = None
    table_entry: str | None = None
    mapping_entry: str | None = None
    mapping_var: str | None = None
    mapping_name: str | None = None
    grid_mapping_name: str | None = None
    coordinates: tuple[str, ...] | list[str] | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    latitude: Any = None
    longitude: Any = None
    latitude_vertices: Any = None
    longitude_vertices: Any = None
    vertices_dim: str = "vertices"
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)
    project: InitVar[Any | None] = None

    def __post_init__(self, project: Any | None) -> None:
        if project is None:
            return
        merged = self._merge_table_entry(project)
        for key, value in merged.to_dict().items():
            object.__setattr__(self, key, value)

    def _merge_table_entry(self, project: Any) -> "Grid":
        """Merge grid-mapping metadata from the loaded grids table.

        Parameters
        ----------
        project
            Project table loader containing grid mapping entries.

        Returns
        -------
        Grid
            New grid metadata record with table defaults applied.
        """

        merged = self.to_dict()
        entry_name, entry = self.resolve_table_entry(project)
        if entry is None:
            return Grid.from_mapping(merged)
        merged.setdefault("table_entry", entry_name)
        for key in ("mapping_name", "grid_mapping_name"):
            value = entry.get(key)
            if is_table_value(value):
                merged.setdefault(key, value)
        coordinates = entry.get("coordinates")
        if "coordinates" not in merged and is_table_value(coordinates):
            merged["coordinates"] = str(coordinates).split()
        params = dict(merged.get("params", {}))
        for key, value in entry.items():
            if not key.startswith("parameter") or not is_table_value(value):
                continue
            params.setdefault(str(value), merged.get(str(value), 0.0))
        if params:
            merged["params"] = params
        return Grid.from_mapping(merged)

    def resolve_table_entry(
        self, project: Any
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve a grid mapping entry from this grid definition.

        Grid mapping entries define CF-compliant coordinate reference systems
        and projection parameters for non-geographic coordinate systems (e.g.,
        Lambert Conformal Conic, Polar Stereographic).

        Parameters
        ----------
        project
            Project table loader containing grid mapping entries from the
            loaded grids table.

        Returns
        -------
        tuple[str | None, Mapping[str, Any] | None]
            A tuple containing:

            - entry_name (str or None): Matched grid mapping entry name
            - entry (dict or None): Grid mapping entry metadata including
              mapping_name and projection parameters

            Returns ``(None, None)`` if no matching entry is found.

        Examples
        --------
        Resolve a standard grid mapping::

            grid = Grid(name="lambert_conformal_conic")
            entry_name, entry = grid.resolve_table_entry(project)
            # Returns ("lambert_conformal_conic", {...}) with projection params

        Explicit table entry selection::

            grid = Grid(table_entry="rotated_latitude_longitude")
            entry_name, entry = grid.resolve_table_entry(project)
            # Returns ("rotated_latitude_longitude", {...})
        """

        requested = str(
            self.table_entry
            or self.mapping_entry
            or self.name
            or ""
        )
        if requested in project.grid_mapping_entries:
            return requested, project.grid_mapping_entries[requested]
        return None, None

    @property
    def variable_name(self) -> str:
        """Return the output grid-mapping variable name.

        The grid mapping variable is a scalar NetCDF variable that holds
        projection parameters as attributes. This property returns the name
        that will be used for that variable in the output file.

        Returns
        -------
        str
            The grid mapping variable name. Returns the explicitly set
            ``mapping_var`` if provided, otherwise defaults to ``"crs"``
            (Coordinate Reference System).

        Examples
        --------
        Default grid mapping variable name::

            grid = Grid(mapping_name="lambert_conformal_conic")
            grid.variable_name  # Returns "crs"

        Custom grid mapping variable name::

            grid = Grid(
                mapping_name="polar_stereographic",
                mapping_var="projection"
            )
            grid.variable_name  # Returns "projection"
        """

        return str(self.mapping_var or "crs")

    def variable_dimensions(self, variable: Any) -> tuple[str, ...] | None:
        """Return data-variable dimensions implied by this grid.

        When a grid specifies dimensions, those dimensions override the
        variable's default dimensions. This is useful for variables on
        non-rectilinear grids where dimension names differ from coordinate
        table defaults.

        Parameters
        ----------
        variable
            Variable metadata used as a fallback source of dimensions when
            the grid doesn't specify dimensions.

        Returns
        -------
        tuple[str, ...] | None
            Ordered tuple of dimension names for the data variable. Returns
            grid dimensions if defined, otherwise variable dimensions if
            defined, otherwise ``None``.

        Examples
        --------
        Grid with explicit dimensions::

            grid = Grid(dimensions=("j", "i"))
            variable = Variable(name="tos", dimensions=("lat", "lon"))
            dims = grid.variable_dimensions(variable)
            # Returns ("j", "i"), overriding variable dimensions

        Grid without dimensions uses variable defaults::

            grid = Grid(mapping_name="lambert_conformal_conic")
            variable = Variable(name="tas", dimensions=("time", "y", "x"))
            dims = grid.variable_dimensions(variable)
            # Returns ("time", "y", "x") from variable

        Neither grid nor variable specify dimensions::

            grid = Grid()
            variable = Variable(name="orog")
            dims = grid.variable_dimensions(variable)
            # Returns None
        """

        if self.dimensions:
            return tuple(str(name) for name in self.dimensions)
        dimensions = variable.get("dimensions")
        if dimensions:
            return tuple(str(name) for name in dimensions)
        return None

    @property
    def has_mapping(self) -> bool:
        """Return whether this grid should write a grid-mapping variable.

        A grid mapping variable is only created if the grid defines projection
        parameters, a mapping name, or attributes. Grids that only specify
        dimensions without projection information do not produce a mapping
        variable.

        Returns
        -------
        bool
            ``True`` if a grid mapping variable should be written to the NetCDF
            file, ``False`` otherwise. Returns ``True`` when any of
            mapping_name, grid_mapping_name, params, or attrs are defined.

        Examples
        --------
        Grid with projection requires mapping variable::

            grid = Grid(
                mapping_name="lambert_conformal_conic",
                params={
                    "standard_parallel": (30.0, "degrees_north"),
                    "longitude_of_central_meridian": (-100.0, "degrees_east")
                }
            )
            grid.has_mapping  # Returns True

        Grid with only dimensions doesn't need mapping::

            grid = Grid(dimensions=("j", "i"))
            grid.has_mapping  # Returns False

        Empty grid doesn't need mapping::

            grid = Grid()
            grid.has_mapping  # Returns False
        """

        return bool(
            self.mapping_name
            or self.grid_mapping_name
            or self.params
            or self.attrs
        )

    def mapping_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the grid-mapping variable.

        This method constructs the complete set of CF-compliant grid mapping
        attributes, including the grid_mapping_name and all projection
        parameters. Parameters are validated to ensure they fall within
        CF-required ranges (e.g., latitudes between -90 and 90).

        Returns
        -------
        dict[str, Any]
            NetCDF-safe grid mapping attributes suitable for assignment to
            the scalar grid mapping variable. Includes grid_mapping_name and
            all validated projection parameters. Parameters with units are
            split into value and units attributes (e.g., "standard_parallel"
            and "standard_parallel_units").

        Notes
        -----
        Parameters are validated during attribute construction:

        - Latitude parameters must be in [-90, 90] degrees_north
        - Longitude parameters must be in [-180, 180] degrees_east
        - Scale factor parameters must be non-negative

        Invalid parameters trigger warnings and are excluded from output.

        Examples
        --------
        Get attributes for Lambert Conformal Conic projection::

            grid = Grid(
                mapping_name="lambert_conformal_conic",
                params={
                    "standard_parallel": ([30.0, 60.0], "degrees_north"),
                    "longitude_of_central_meridian": (-100.0, "degrees_east"),
                    "latitude_of_projection_origin": (40.0, "degrees_north")
                }
            )
            attrs = grid.mapping_attributes()
            # attrs = {
            #     "grid_mapping_name": "lambert_conformal_conic",
            #     "standard_parallel": [30.0, 60.0],
            #     "standard_parallel_units": "degrees_north",
            #     "longitude_of_central_meridian": -100.0,
            #     "longitude_of_central_meridian_units": "degrees_east",
            #     ...
            # }
        """

        attrs = self.netcdf_attrs(self.attrs)
        mapping_name = self.mapping_name or self.grid_mapping_name
        if mapping_name:
            attrs["grid_mapping_name"] = mapping_name
        for key, value in self.params.items():
            if not _valid_mapping_parameter(str(key), value):
                continue
            if isinstance(value, (list, tuple)) and value:
                attrs[key] = value[0]
                if len(value) > 1 and value[1]:
                    attrs[f"{key}_units"] = value[1]
            else:
                attrs[key] = value
        return self.netcdf_attrs(attrs)


def _valid_mapping_parameter(name: str, value: Any) -> bool:
    numeric = _primary_numeric_value(value)
    if numeric is None:
        return True
    if name in _LATITUDE_PARAMETERS and not -90.0 <= numeric <= 90.0:
        warnings.warn(
            f"{name} parameter must be between -90 and 90 degrees_north; "
            "it will not be set.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    if name in _LONGITUDE_PARAMETERS and not -180.0 <= numeric <= 180.0:
        warnings.warn(
            f"{name} parameter must be between -180 and 180 degrees_east; "
            "it will not be set.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    if name in _NON_NEGATIVE_PARAMETERS and numeric < 0.0:
        warnings.warn(
            f"{name} parameter must be positive; it will not be set.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    return True


def _primary_numeric_value(value: Any) -> float | None:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
