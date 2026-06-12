from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ._table_utils import (
    entry_bounds,
    entry_values,
    is_table_value,
    metadata_value_matches,
    parse_table_value,
)
from .exceptions import TableValidationError
from .metadata import _MetadataRecord


@dataclass(frozen=True)
class Axis(_MetadataRecord):
    """Metadata and coordinate values for one data axis.

    Parameters
    ----------
    name
        Logical axis name used by variable dimensions.
    values
        Coordinate values.
    bounds
        Optional coordinate bounds.
    dimensions
        Underlying dimensions for auxiliary coordinates.
    units
        Coordinate units attribute.
    standard_name
        CF standard_name attribute.
    long_name
        Coordinate long_name attribute.
    axis
        CF axis attribute (X, Y, Z, or T).
    positive
        CF positive attribute for vertical coordinates (up or down).
    formula
        Formula for computing coordinate values.
    valid_min
        Minimum valid coordinate value.
    valid_max
        Maximum valid coordinate value.
    out_name
        Output coordinate variable name.
    table_entry
        Coordinate table entry name selector.
    axis_entry
        Coordinate table entry name selector.
    coordinate
        Coordinate table entry name selector.
    grid_table_entry
        Grid coordinate table entry name selector.
    grid_coordinate
        Grid coordinate table entry name selector.
    scalar
        Whether this axis is written as a scalar coordinate.
    auxiliary
        Whether this axis is written as an auxiliary coordinate.
    auxiliary_name
        Output name for the auxiliary coordinate variable.
    auxiliary_attrs
        Extra attributes for the auxiliary coordinate variable.
    climatology
        Climatology bounds control.
    generic_level_name
        Generic level selector from coordinate tables.
    z_factors
        Formula-term names associated with this axis.
    z_bounds_factors
        Formula-term names associated with this axis bounds.
    bounds_name
        Output bounds variable name.
    bounds_dim
        Output bounds dimension name.
    bounds_attrs
        Extra attributes for the bounds variable.
    attrs
        Extra attributes for the coordinate variable.
    extra
        Additional mapping keys preserved by the metadata record.
    project
        Optional project tables used to resolve and merge axis metadata during
        construction.
    """

    name: str
    values: Any = None
    bounds: Any = None
    dimensions: tuple[str, ...] | list[str] | None = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    axis: str | None = None
    positive: str | None = None
    formula: str | None = None
    valid_min: Any = None
    valid_max: Any = None
    out_name: str | None = None
    table_entry: str | None = None
    axis_entry: str | None = None
    coordinate: str | None = None
    grid_table_entry: str | None = None
    grid_coordinate: str | None = None
    scalar: bool | None = None
    auxiliary: bool | None = None
    auxiliary_name: str | None = None
    auxiliary_attrs: Mapping[str, Any] = field(default_factory=dict)
    climatology: str | bool | None = None
    generic_level_name: str | None = None
    z_factors: str | None = None
    z_bounds_factors: str | None = None
    requested: Any = None
    requested_bounds: Any = None
    bounds_values: Any = None
    must_have_bounds: Any = None
    stored_direction: str | None = None
    tolerance: Any = None
    bounds_name: str | None = None
    bounds_dim: str | None = None
    bounds_attrs: Mapping[str, Any] = field(default_factory=dict)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)
    project: InitVar[Any | None] = None

    def __post_init__(self, project: Any | None) -> None:
        if project is None:
            return
        merged = self._merge_table_entry(project)
        merged._validate_values_early()
        for key, value in merged.to_dict().items():
            object.__setattr__(self, key, value)

    def _merge_table_entry(self, project: Any) -> "Axis":
        """Merge authoritative coordinate metadata into this axis.

        Parameters
        ----------
        project
            Project table loader containing coordinate and grid entries.

        Returns
        -------
        Axis
            New axis metadata record with table defaults applied.
        """

        merged = self.to_dict()
        entry_name, entry = self.resolve_table_entry(project)
        if entry is None:
            return Axis.from_mapping(merged)
        merged.setdefault("table_entry", entry_name)
        self._validate_metadata(
            "axis",
            entry_name,
            entry,
            (
                "units",
                "standard_name",
                "long_name",
                "axis",
                "positive",
                "formula",
            ),
        )
        for key in (
            "out_name",
            "units",
            "standard_name",
            "long_name",
            "axis",
            "positive",
            "formula",
            "climatology",
            "generic_level_name",
            "z_factors",
            "z_bounds_factors",
            "valid_min",
            "valid_max",
            "requested",
            "requested_bounds",
            "bounds_values",
            "must_have_bounds",
            "stored_direction",
            "tolerance",
        ):
            value = entry.get(key)
            if is_table_value(value):
                merged.setdefault(key, parse_table_value(value))
        merged.setdefault("out_name", entry_name)
        if "values" not in merged:
            values = entry_values(entry)
            if values is not None:
                merged["values"] = values
        if "bounds" not in merged:
            bounds = entry_bounds(entry)
            if bounds is not None:
                merged["bounds"] = bounds
        Axis.from_mapping(merged)._merge_grid_coordinate_metadata(
            project, merged
        )
        return Axis.from_mapping(merged)

    def resolve_table_entry(
        self, project: Any
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve a coordinate table entry from this axis.

        This method searches the project's coordinate tables for an entry
        matching this axis. It tries multiple strategies in order: exact name
        match, generic level name disambiguation, out_name match, and finally
        matching by out_name and standard_name combination.

        Parameters
        ----------
        project
            Project table loader containing coordinate and grid axis entries
            from loaded coordinate and grid tables.

        Returns
        -------
        tuple[str | None, Mapping[str, Any] | None]
            A tuple containing:

            - entry_name (str or None): The matched coordinate table entry name
            - entry (dict or None): The table entry metadata dictionary

            Returns ``(None, None)`` if no matching entry is found.

        Raises
        ------
        TableValidationError
            If a generic level name matches multiple entries without
            disambiguation via table_entry or axis_entry.

        Notes
        -----
        Resolution strategy:

        1. Direct match by table_entry, axis_entry, coordinate, or name
        2. Generic level name match (e.g., "alevel", "olevel"), narrowed by
           standard_name, formula, z_factors, etc. if multiple matches exist
        3. Match by out_name in coordinate entries
        4. Match by out_name and standard_name combination

        Examples
        --------
        Resolve standard coordinate::

            axis = Axis(name="time", values=[...])
            entry_name, entry = axis.resolve_table_entry(project)
            # Returns ("time", {...}) from coordinate table

        Resolve generic level::

            axis = Axis(name="alevel", standard_name="altitude")
            entry_name, entry = axis.resolve_table_entry(project)
            # Returns specific altitude entry matching standard_name

        Disambiguate with table_entry::

            axis = Axis(name="plev", table_entry="plev19")
            entry_name, entry = axis.resolve_table_entry(project)
            # Returns ("plev19", {...}) specifically
        """

        requested = str(
            self.table_entry
            or self.axis_entry
            or self.coordinate
            or self.name
            or ""
        )
        if requested in project.coordinate_entries:
            return requested, project.coordinate_entries[requested]
        generic_matches = self._matching_generic_level_entries(
            project, requested
        )
        if len(generic_matches) == 1:
            return generic_matches[0]
        if len(generic_matches) > 1:
            choices = ", ".join(name for name, _ in generic_matches)
            raise TableValidationError(
                f"Generic level {requested!r} matches multiple coordinate "
                "entries; specify table_entry or axis_entry. "
                f"Choices: {choices}."
            )
        matching_out_names = [
            (name, entry)
            for name, entry in project.coordinate_entries.items()
            if str(entry.get("out_name", "")) == requested
        ]
        if len(matching_out_names) == 1:
            return matching_out_names[0]
        matches = self._matching_coordinate_entries(project)
        if len(matches) == 1:
            return matches[0]
        return None, None

    def resolve_grid_coordinate(
        self, project: Any
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve a grid-coordinate variable entry from this axis.

        Grid coordinates are auxiliary coordinate variables defined in the
        project's grid table, typically representing latitude and longitude
        on non-rectilinear grids (e.g., curvilinear ocean grids). This method
        searches for a matching grid coordinate entry by name or out_name.

        Parameters
        ----------
        project
            Project table loader containing grid coordinate entries from the
            loaded grids table.

        Returns
        -------
        tuple[str | None, Mapping[str, Any] | None]
            A tuple containing:

            - entry_name (str or None): Matched grid coordinate entry name
            - entry (dict or None): Grid coordinate entry metadata

            Returns ``(None, None)`` if no matching entry is found.

        Notes
        -----
        Grid coordinates differ from regular axis coordinates in that they are
        typically 2D auxiliary coordinates (e.g., ``lat(j,i)``, ``lon(j,i)``)
        rather than 1D dimension coordinates.

        Examples
        --------
        Resolve grid coordinate for curvilinear grid::

            axis = Axis(name="latitude", dimensions=("j", "i"), values=lat_2d)
            entry_name, entry = axis.resolve_grid_coordinate(project)
            # Returns ("latitude", {...}) from grid coordinate table

        Explicit grid coordinate selection::

            axis = Axis(name="lat", grid_coordinate="latitude_bnds")
            entry_name, entry = axis.resolve_grid_coordinate(project)
            # Returns ("latitude_bnds", {...}) specifically
        """

        requested = str(
            self.grid_table_entry
            or self.grid_coordinate
            or self.out_name
            or self.name
            or ""
        )
        if requested in project.grid_coordinate_entries:
            return requested, project.grid_coordinate_entries[requested]
        matches = [
            (name, entry)
            for name, entry in project.grid_coordinate_entries.items()
            if str(entry.get("out_name", "")) == requested
        ]
        if len(matches) == 1:
            return matches[0]
        return None, None

    def attributes(self, *, include_units: bool = True) -> dict[str, Any]:
        """Return NetCDF attributes for this coordinate axis.

        This method constructs the complete set of NetCDF attributes for the
        coordinate variable, including CF-required and optional metadata.

        Parameters
        ----------
        include_units
            Whether to include the ``units`` attribute. Set to False when
            creating index-based auxiliary coordinates that should not have
            units (e.g., basin indices).

        Returns
        -------
        dict[str, Any]
            NetCDF-safe coordinate attributes suitable for assignment to an
            xarray coordinate or NetCDF variable. Always includes attributes
            like standard_name, long_name, axis, positive, formula, valid_min,
            and valid_max when they are defined in the axis metadata.

        Notes
        -----
        The units attribute is included by default but can be excluded for
        special coordinate types. Other attributes from the ``attrs`` field
        are merged first, allowing standard attributes to override them.

        Examples
        --------
        Get attributes for a time axis::

            axis = project.axis("time", values=[...])
            attrs = axis.attributes()
            # attrs = {
            #     "units": "days since 1850-01-01",
            #     "standard_name": "time",
            #     "calendar": "noleap",
            #     ...
            # }

        Get attributes without units for index coordinate::

            axis = Axis(
                name="basin",
                values=[1, 2, 3],
                auxiliary_name="basin_label"
            )
            attrs = axis.attributes(include_units=False)
            # attrs does not include "units"
        """

        attrs = self.netcdf_attrs(self.attrs)
        if include_units and "units" in self:
            attrs["units"] = self["units"]
        for key in (
            "standard_name",
            "long_name",
            "axis",
            "positive",
            "formula",
            "valid_min",
            "valid_max",
        ):
            if key in self:
                attrs[key] = self[key]
        return attrs

    def auxiliary_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for this axis' auxiliary variable.

        Auxiliary coordinates are variables that provide alternate or
        supplementary coordinate information, such as string labels for
        integer indices (e.g., basin names for basin indices).

        Returns
        -------
        dict[str, Any]
            NetCDF-safe attributes for the auxiliary coordinate variable,
            filtered to include only values compatible with NetCDF format.

        Examples
        --------
        Create axis with auxiliary coordinate for basin labels::

            axis = Axis(
                name="basin",
                values=[1, 2, 3],
                auxiliary_name="basin_label",
                auxiliary_attrs={
                    "long_name": "Basin Name",
                    "comment": "Integer basin indices"
                }
            )
            attrs = axis.auxiliary_attributes()
            # attrs = {"long_name": "Basin Name", "comment": "..."}
        """

        return self.netcdf_attrs(self.auxiliary_attrs)

    def bounds_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for this axis' bounds variable.

        Bounds variables define the edges or limits of coordinate cells and
        are required for certain types of coordinates (especially time and
        spatial coordinates used in integration or averaging).

        Returns
        -------
        dict[str, Any]
            NetCDF-safe attributes for the bounds variable, filtered to
            include only values compatible with NetCDF format. May include
            units, standard_name, and long_name from the bounds_attrs field
            or grid coordinate bounds entries.

        Examples
        --------
        Get bounds attributes from coordinate table::

            axis = project.axis(
                "time",
                values=[0, 30, 60],
                bounds=[[0, 30], [30, 60], [60, 90]]
            )
            attrs = axis.bounds_attributes()
            # attrs may include units, long_name from table

        Custom bounds attributes::

            axis = Axis(
                name="lat",
                values=[-45, 0, 45],
                bounds=[[-90, -22.5], [-22.5, 22.5], [22.5, 90]],
                bounds_attrs={"comment": "Cell boundaries"}
            )
            attrs = axis.bounds_attributes()
            # attrs = {"comment": "Cell boundaries"}
        """

        return self.netcdf_attrs(self.bounds_attrs)

    def values_array(self) -> np.ndarray:
        """Return this axis' values as a NetCDF-ready array.

        This method converts the axis values to a numpy array with a dtype
        suitable for NetCDF output, handling various input types including
        lists, tuples, numpy arrays, xarray DataArrays, and datetime objects.

        Returns
        -------
        numpy.ndarray
            Coordinate values as a numpy array with appropriate dtype for
            NetCDF serialization. Time coordinates are converted to numeric
            values, and other types are converted to their NetCDF-compatible
            representations.

        Examples
        --------
        Convert time values::

            axis = Axis(
                name="time",
                values=[datetime(2000, 1, 1), datetime(2000, 2, 1)],
                units="days since 1850-01-01"
            )
            arr = axis.values_array()
            # Returns numeric array relative to reference date

        Convert latitude values::

            axis = Axis(name="lat", values=[-90, -45, 0, 45, 90])
            arr = axis.values_array()
            # Returns np.array([-90., -45., 0., 45., 90.])
        """

        return self.netcdf_array(self.get("values", []))

    def bounds_array(self) -> np.ndarray:
        """Return this axis' bounds as a NetCDF-ready array.

        This method converts the axis bounds to a numpy array with shape
        (n_values, 2) or (n_values, n_bounds) for climatology, and with a
        dtype suitable for NetCDF output.

        Returns
        -------
        numpy.ndarray
            Coordinate bounds as a numpy array with shape matching the number
            of coordinate values. Standard bounds have shape (n, 2) where each
            row contains [lower_bound, upper_bound] for the corresponding
            coordinate cell. Climatology bounds may have additional vertices.

        Raises
        ------
        KeyError
            If the axis does not have bounds defined.

        Examples
        --------
        Get bounds for latitude cells::

            axis = Axis(
                name="lat",
                values=[-45, 0, 45],
                bounds=[[-90, -22.5], [-22.5, 22.5], [22.5, 90]]
            )
            bnds = axis.bounds_array()
            # Returns array([[-90., -22.5], [-22.5, 22.5], [22.5, 90.]])

        Get bounds for time coordinate::

            axis = Axis(
                name="time",
                values=[15, 45],
                bounds=[[0, 31], [31, 59]],
                units="days since 2000-01-01"
            )
            bnds = axis.bounds_array()
            # Returns array([[0, 31], [31, 59]])
        """

        return self.netcdf_array(self["bounds"])

    def _matching_coordinate_entries(
        self, project: Any
    ) -> list[tuple[str, Mapping[str, Any]]]:
        if not self.out_name and not self.standard_name:
            return []
        matches = list(project.coordinate_entries.items())
        for key, value in (
            ("out_name", self.out_name),
            ("standard_name", self.standard_name),
        ):
            if value in (None, ""):
                continue
            narrowed = [
                (name, entry)
                for name, entry in matches
                if str(entry.get(key, "")) == str(value)
            ]
            if narrowed:
                matches = narrowed
        return matches if len(matches) == 1 else []

    def _matching_generic_level_entries(
        self,
        project: Any,
        generic_level_name: str,
    ) -> list[tuple[str, Mapping[str, Any]]]:
        generic_entries = getattr(project, "generic_level_entries", {})
        matches = list(generic_entries.get(generic_level_name, {}).items())
        if not matches:
            return []
        for key, value in (
            ("standard_name", self.standard_name),
            ("formula", self.formula),
            ("z_factors", self.z_factors),
            ("z_bounds_factors", self.z_bounds_factors),
            ("positive", self.positive),
            ("units", self.units),
            ("long_name", self.long_name),
        ):
            if value in (None, ""):
                continue
            narrowed = [
                (name, entry)
                for name, entry in matches
                if is_table_value(entry.get(key))
                and metadata_value_matches(value, entry[key])
            ]
            if narrowed:
                matches = narrowed
        return matches

    def _merge_grid_coordinate_metadata(
        self, project: Any, axis: dict[str, Any]
    ) -> None:
        entry_name, entry = self.resolve_grid_coordinate(project)
        if entry is None:
            return
        axis.setdefault("grid_table_entry", entry_name)
        for key in (
            "out_name",
            "units",
            "standard_name",
            "long_name",
            "valid_min",
            "valid_max",
        ):
            value = entry.get(key)
            if is_table_value(value):
                axis[key] = parse_table_value(value)
        bounds_name = axis.get("bounds_name")
        if bounds_name:
            bounds_entry = project.grid_coordinate_entries.get(
                str(bounds_name)
            )
            if bounds_entry:
                bounds_attrs = dict(axis.get("bounds_attrs", {}))
                for key in ("units", "standard_name", "long_name"):
                    value = bounds_entry.get(key)
                    if is_table_value(value):
                        bounds_attrs.setdefault(key, parse_table_value(value))
                if bounds_attrs:
                    axis["bounds_attrs"] = bounds_attrs

    def _validate_metadata(
        self,
        entry_type: str,
        entry_name: str | None,
        table_values: Mapping[str, Any],
        keys: Sequence[str],
    ) -> None:
        user_values = self.to_dict()
        for key in keys:
            expected = table_values.get(key)
            if (
                is_table_value(expected)
                and key in user_values
                and not metadata_value_matches(user_values[key], expected)
            ):
                raise TableValidationError(
                    f"{entry_type} {entry_name!r} {key}={user_values[key]!r} "
                    f"does not match table value {expected!r}."
                )

    def _validate_values_early(self) -> None:
        """Run component-level axis checks that do not need dataset context.

        This is partial validation performed during construction. It checks:
        - Monotonicity
        - Valid ranges
        - Requested values
        - Bounds shape and consistency

        It skips:
        - Time interval validation (needs frequency from dataset/variable)
        - Required bounds enforcement (checked during full validation)
        - Value normalization (done during full validation)
        """

        from ._axis_validation import validate_axis_values_early

        validate_axis_values_early(self)
