from __future__ import annotations

from dataclasses import dataclass, field
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
    dimensions:
        Output dimensions used for the data variable.
    name:
        Requested grid mapping entry name.
    table_entry, mapping_entry:
        Grid table entry selectors.
    mapping_var:
        Name of the scalar grid-mapping variable to write.
    mapping_name, grid_mapping_name:
        CF grid mapping name.
    coordinates:
        Auxiliary coordinate names associated with the grid.
    params:
        Grid-mapping parameter values.
    attrs:
        Extra NetCDF attributes for the grid-mapping variable.
    extra:
        Additional mapping keys preserved by the metadata record.
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
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def merge_table_entry(self, project: Any) -> "Grid":
        """Merge grid-mapping metadata from the loaded grids table.

        Parameters
        ----------
        project:
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

        Parameters
        ----------
        project:
            Project table loader containing grid mapping entries.

        Returns
        -------
        tuple[str | None, Mapping[str, Any] | None]
            Matched entry name and table metadata, or ``(None, None)``.
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

        Returns
        -------
        str
            Explicit mapping variable name or the default ``crs``.
        """

        return str(self.mapping_var or "crs")

    def variable_dimensions(self, variable: Any) -> tuple[str, ...] | None:
        """Return data-variable dimensions implied by this grid.

        Parameters
        ----------
        variable:
            Variable metadata used as a fallback source of dimensions.

        Returns
        -------
        tuple[str, ...] | None
            Grid dimensions, variable dimensions, or ``None`` when neither is
            defined.
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

        Returns
        -------
        bool
            ``True`` when a mapping name, parameters, or attributes are set.
        """

        return bool(
            self.mapping_name
            or self.grid_mapping_name
            or self.params
            or self.attrs
        )

    def mapping_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the grid-mapping variable.

        Returns
        -------
        dict[str, Any]
            NetCDF-safe grid-mapping attributes after parameter validation.
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
