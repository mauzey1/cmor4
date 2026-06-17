"""Axis metadata record."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import Field
from typing import Annotated
from pydantic import BeforeValidator

from ._table_utils import (
    entry_bounds,
    entry_values,
    is_table_value,
    metadata_value_matches,
    parse_table_value,
)
from .exceptions import AxisValidationError, TableValidationError
from .metadata import MetadataModel

if TYPE_CHECKING:
    from .tables import ProjectTables


def _str_tuple(v: Any) -> tuple[str, ...] | None:
    if v is None:
        return None
    if isinstance(v, str):
        return (v,)
    return tuple(str(x) for x in v)


def _str_seq(v: Any) -> list[str] | tuple[str, ...] | None:
    """Preserve list-or-tuple of str."""
    if v is None:
        return None
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, tuple):
        return tuple(str(x) for x in v)
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _float_or_none(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def _upper_str(v: Any) -> Any:
    """Upper-case the axis designator so 't'/'x'/'y'/'z' are accepted."""
    if isinstance(v, str):
        return v.upper()
    return v


StrTuple = Annotated[tuple[str, ...] | None, BeforeValidator(_str_tuple)]
StrSeq = Annotated[list[str] | tuple[str, ...] | None, BeforeValidator(_str_seq)]
CoercedF = Annotated[float | None, BeforeValidator(_float_or_none)]
# axis: upper-case but no Literal — invalid values caught by validate_against_entry
AxisStr = Annotated[str | None, BeforeValidator(_upper_str)]


class Axis(MetadataModel):
    """Metadata and coordinate values for one data axis.

    Construct directly when project tables are not available::

        axis = Axis(name="latitude", values=np.linspace(-90, 90, 180),
                    units="degrees_north", axis="Y")

    Construct via :meth:`ProjectTables.axis` to merge authoritative
    coordinate-table metadata::

        axis = project.axis("latitude", values=np.linspace(-90, 90, 180))

    Parameters
    ----------
    name:
        Logical axis name matching a variable dimension.
    values:
        Coordinate values (list, numpy array, scalar).
    bounds:
        Optional coordinate cell bounds.
    dimensions:
        Underlying dimensions for auxiliary coordinates.
    units:
        CF units string.
    standard_name, long_name:
        CF / NetCDF attributes.
    axis:
        CF axis designator — ``"X"``, ``"Y"``, ``"Z"``, or ``"T"``.
        Lower-case inputs are automatically upper-cased.
    positive:
        CF ``"up"`` or ``"down"`` for vertical axes.
    formula:
        Formula for computing coordinate values.
    valid_min, valid_max:
        Valid coordinate value range.
    out_name:
        Output coordinate variable name.
    table_entry, axis_entry, coordinate:
        Coordinate-table entry name selectors, tried in order.
    grid_table_entry, grid_coordinate:
        Grid coordinate-table entry selectors.
    scalar:
        Write as a scalar coordinate.
    auxiliary:
        Write as an auxiliary coordinate.
    auxiliary_name:
        Output name for the auxiliary coordinate variable.
    auxiliary_attrs:
        Extra attributes for the auxiliary variable.
    climatology:
        Climatology bounds control.
    generic_level_name:
        Generic level selector.
    z_factors, z_bounds_factors:
        Formula-term names for this axis.
    bounds_name, bounds_dim:
        Output bounds variable name / dimension name.
    bounds_attrs:
        Extra attributes for the bounds variable.
    attrs:
        Extra attributes for the coordinate variable.
    stored_direction:
        Expected ordering of coordinate values: ``"increasing"`` or
        ``"decreasing"``.
    """

    name: str
    values: Any = None
    bounds: Any = None
    dimensions: StrSeq = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    axis: AxisStr = None
    positive: Literal["up", "down"] | None = None
    formula: str | None = None
    valid_min: CoercedF = None
    valid_max: CoercedF = None
    out_name: str | None = None
    table_entry: str | None = None
    axis_entry: str | None = None
    coordinate: str | None = None
    grid_table_entry: str | None = None
    grid_coordinate: str | None = None
    scalar: bool | None = None
    auxiliary: bool | None = None
    auxiliary_name: str | None = None
    auxiliary_attrs: dict[str, Any] = Field(default_factory=dict)
    climatology: str | bool | None = None
    generic_level_name: str | None = None
    z_factors: str | None = None
    z_bounds_factors: str | None = None
    requested: Any = None
    requested_bounds: Any = None
    bounds_values: Any = None
    must_have_bounds: Any = None
    stored_direction: str | None = None
    tolerance: CoercedF = None
    bounds_name: str | None = None
    bounds_dim: str | None = None
    bounds_attrs: dict[str, Any] = Field(default_factory=dict)
    attrs: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Project-table construction
    # ------------------------------------------------------------------

    @classmethod
    def from_project(cls, project: ProjectTables, name: str, **values: Any) -> Axis:
        """Create an Axis by merging authoritative coordinate-table metadata.

        Also runs :meth:`_validate_values_early` on the result.
        Called by :meth:`ProjectTables.axis`.
        """
        data: dict[str, Any] = {"name": name, **values}
        data = cls._apply_table_defaults(data, project)
        axis = cls.model_validate(data)
        axis._validate_values_early()
        return axis

    @classmethod
    def _apply_table_defaults(
        cls, data: dict[str, Any], project: ProjectTables
    ) -> dict[str, Any]:
        """Resolve coordinate-table entry and merge its defaults into *data*."""
        # Grid coordinates take a separate path (no main coordinate table)
        if data.get("grid_coordinate") or data.get("grid_table_entry"):
            entry_name, entry = cls._resolve_grid_entry(data, project)
            if entry is not None:
                cls._merge_grid_entry(data, entry_name, entry, project)
            return data

        entry_name, entry = cls._resolve_coord_entry(data, project)
        if entry is None:
            return data

        data.setdefault("table_entry", entry_name)
        cls._validate_metadata(
            data,
            entry_name,
            entry,
            ("units", "standard_name", "long_name", "axis", "positive", "formula"),
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
            val = entry.get(key)
            if is_table_value(val):
                data.setdefault(key, parse_table_value(val))
        data.setdefault("out_name", entry_name)
        if "values" not in data:
            v = entry_values(entry)
            if v is not None:
                data["values"] = v
        if "bounds" not in data:
            b = entry_bounds(entry)
            if b is not None:
                data["bounds"] = b
        # Also pull in any matching grid coordinate
        cls._merge_grid_entry_from_data(data, project)
        return data

    # -- coordinate-table resolution ------------------------------------

    @classmethod
    def _resolve_coord_entry(
        cls, data: dict[str, Any], project: ProjectTables
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        requested = str(
            data.get("table_entry")
            or data.get("axis_entry")
            or data.get("coordinate")
            or data.get("name")
            or ""
        )
        coord: dict[str, Mapping[str, Any]] = project.coordinate_entries
        if requested in coord:
            return requested, coord[requested]
        # generic level
        generic = cls._generic_level_matches(data, project, requested)
        if len(generic) == 1:
            return generic[0]
        if len(generic) > 1:
            choices = ", ".join(n for n, _ in generic)
            raise TableValidationError(
                f"Generic level {requested!r} matches multiple coordinate entries; "
                f"specify table_entry or axis_entry.  Choices: {choices}."
            )
        # out_name match
        by_out = [
            (n, e) for n, e in coord.items() if str(e.get("out_name", "")) == requested
        ]
        if len(by_out) == 1:
            return by_out[0]
        # out_name + standard_name match
        m = cls._match_by_attrs(data, project)
        if len(m) == 1:
            return m[0]
        return None, None

    @classmethod
    def _resolve_grid_entry(
        cls, data: dict[str, Any], project: ProjectTables
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        requested = str(
            data.get("grid_table_entry")
            or data.get("grid_coordinate")
            or data.get("out_name")
            or data.get("name")
            or ""
        )
        gc: dict[str, Mapping[str, Any]] = project.grid_coordinate_entries
        if requested in gc:
            return requested, gc[requested]
        m = [(n, e) for n, e in gc.items() if str(e.get("out_name", "")) == requested]
        if len(m) == 1:
            return m[0]
        return None, None

    @classmethod
    def _merge_grid_entry(
        cls,
        data: dict[str, Any],
        entry_name: str | None,
        entry: Mapping[str, Any],
        project: ProjectTables,
    ) -> None:
        if entry_name:
            data.setdefault("grid_table_entry", entry_name)
        for key in (
            "out_name",
            "units",
            "standard_name",
            "long_name",
            "valid_min",
            "valid_max",
        ):
            val = entry.get(key)
            if is_table_value(val):
                data.setdefault(key, parse_table_value(val))
        data.setdefault("out_name", entry_name)
        bname = data.get("bounds_name")
        if bname:
            be = project.grid_coordinate_entries.get(str(bname))
            if be:
                ba = dict(data.get("bounds_attrs") or {})
                for key in ("units", "standard_name", "long_name"):
                    val = be.get(key)
                    if is_table_value(val):
                        ba.setdefault(key, parse_table_value(val))
                if ba:
                    data["bounds_attrs"] = ba

    @classmethod
    def _merge_grid_entry_from_data(
        cls, data: dict[str, Any], project: ProjectTables
    ) -> None:
        en, entry = cls._resolve_grid_entry(data, project)
        if entry is not None:
            cls._merge_grid_entry(data, en, entry, project)

    @classmethod
    def _generic_level_matches(
        cls, data: dict[str, Any], project: ProjectTables, generic_name: str
    ) -> list[tuple[str, Mapping[str, Any]]]:
        generic: dict[str, dict[str, Mapping[str, Any]]] = getattr(
            project, "generic_level_entries", {}
        )
        matches = list(generic.get(generic_name, {}).items())
        if not matches:
            return []
        for key in (
            "standard_name",
            "formula",
            "z_factors",
            "z_bounds_factors",
            "positive",
            "units",
            "long_name",
        ):
            val = data.get(key)
            if val in (None, ""):
                continue
            narrowed = [
                (n, e)
                for n, e in matches
                if is_table_value(e.get(key)) and metadata_value_matches(val, e[key])
            ]
            if narrowed:
                matches = narrowed
        return matches

    @classmethod
    def _match_by_attrs(
        cls, data: dict[str, Any], project: ProjectTables
    ) -> list[tuple[str, Mapping[str, Any]]]:
        out_name = data.get("out_name")
        std_name = data.get("standard_name")
        if not out_name and not std_name:
            return []
        matches = list(project.coordinate_entries.items())
        for key, val in (("out_name", out_name), ("standard_name", std_name)):
            if val in (None, ""):
                continue
            narrowed = [(n, e) for n, e in matches if str(e.get(key, "")) == str(val)]
            if narrowed:
                matches = narrowed
        return matches if len(matches) == 1 else []

    @classmethod
    def _validate_metadata(
        cls,
        data: dict[str, Any],
        entry_name: str | None,
        table_values: Mapping[str, Any],
        keys: Sequence[str],
    ) -> None:
        """Raise TableValidationError if user values conflict with the table."""
        for key in keys:
            expected = table_values.get(key)
            user_val = data.get(key)
            if (
                is_table_value(expected)
                and user_val not in (None, "")
                and not metadata_value_matches(user_val, expected)
            ):
                raise TableValidationError(
                    f"axis {entry_name!r} {key}={user_val!r} "
                    f"does not match table value {expected!r}."
                )

    # ------------------------------------------------------------------
    # Instance methods for post-construction validation (called by tables.py)
    # ------------------------------------------------------------------

    def _validate_metadata_instance(
        self,
        entry_type: str,
        entry_name: str | None,
        table_values: Mapping[str, Any],
        keys: Sequence[str],
    ) -> None:
        """Instance-method shim — validates self against a table entry.

        Called by :meth:`ProjectTables.validate_components` for unprepared axes.
        """
        self._validate_metadata(self.to_dict(), entry_name, table_values, keys)

    def _merge_table_entry(self, project: Any) -> Axis:
        """Return a new Axis with project-table defaults merged in.

        Called by :meth:`ProjectTables._axes` for unprepared axes.
        """
        merged = self._apply_table_defaults(self.to_dict(), project)
        return type(self).model_validate(merged)

    def _validate_values_early(self) -> None:
        """Run lightweight value checks that don't need dataset context."""
        from ._axis_validation import validate_axis_values_early

        validate_axis_values_early(self)

    def _post_project_init(self) -> None:
        """Run lightweight axis-value checks after construction with project=."""
        self._validate_values_early()

        # ------------------------------------------------------------------

    # Public API (existing interface preserved)
    # ------------------------------------------------------------------

    def resolve_table_entry(
        self, project: ProjectTables
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve the coordinate table entry for this axis."""
        return self._resolve_coord_entry(self.to_dict(), project)

    def resolve_grid_coordinate(
        self, project: ProjectTables
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve the grid coordinate table entry for this axis."""
        return self._resolve_grid_entry(self.to_dict(), project)

    def attributes(self, *, include_units: bool = True) -> dict[str, Any]:
        """Return NetCDF attributes for this coordinate variable."""
        attrs = self.netcdf_attrs(self.attrs)
        if include_units and "units" in self:
            attrs["units"] = self["units"]
        for key in ("standard_name", "long_name", "axis", "positive", "formula"):
            if key in self:
                attrs[key] = self[key]
        return attrs

    def auxiliary_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the auxiliary coordinate variable."""
        return self.netcdf_attrs(self.auxiliary_attrs)

    def bounds_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the bounds variable."""
        return self.netcdf_attrs(self.bounds_attrs)

    def values_array(self) -> np.ndarray:
        """Return coordinate values as a NetCDF-ready numpy array."""
        return self.netcdf_array(self.get("values", []))

    def bounds_array(self) -> np.ndarray:
        """Return coordinate bounds as a NetCDF-ready numpy array."""
        return self.netcdf_array(self["bounds"])
