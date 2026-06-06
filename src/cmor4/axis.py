from __future__ import annotations

from dataclasses import dataclass, field
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
from .variable import Variable


@dataclass(frozen=True)
class Axis(_MetadataRecord):
    """Metadata and coordinate values for one data axis."""

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
    bounds_name: str | None = None
    bounds_dim: str | None = None
    bounds_attrs: Mapping[str, Any] = field(default_factory=dict)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def merge_table_entry(self, project: Any) -> "Axis":
        """Merge authoritative coordinate metadata into this axis."""

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
        ):
            value = entry.get(key)
            if is_table_value(value):
                merged.setdefault(key, value)
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
        """Resolve a coordinate table entry from this axis."""

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
        """Resolve a grid-coordinate variable entry from this axis."""

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

    @classmethod
    def missing_scalar_axes(
        cls,
        project: Any,
        axes: Sequence["Axis"],
        variable: Variable,
    ) -> list["Axis"]:
        present = {
            str(value)
            for axis in axes
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
        missing_axes: list[Axis] = []
        for dimension in variable.get("dimensions", ()):
            dimension_name = str(dimension)
            if dimension_name in present:
                continue
            entry = project.coordinate_entries.get(dimension_name)
            if entry is None or not is_table_value(entry.get("value")):
                continue
            axis = cls(
                name=dimension_name,
                table_entry=dimension_name,
                scalar=True,
            ).merge_table_entry(project)
            missing_axes.append(axis)
            present.update(
                str(value)
                for value in (
                    axis.name,
                    axis.table_entry,
                    axis.out_name,
                    axis.generic_level_name,
                )
                if value
            )
        return missing_axes

    def attributes(self, *, include_units: bool = True) -> dict[str, Any]:
        """Return NetCDF attributes for this coordinate axis."""

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
        """Return NetCDF attributes for this axis' auxiliary variable."""

        return self.netcdf_attrs(self.auxiliary_attrs)

    def bounds_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for this axis' bounds variable."""

        return self.netcdf_attrs(self.bounds_attrs)

    def values_array(self) -> np.ndarray:
        """Return this axis' values as a NetCDF-ready array."""

        return self.netcdf_array(self.get("values", []))

    def bounds_array(self) -> np.ndarray:
        """Return this axis' bounds as a NetCDF-ready array."""

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
            bounds_entry = project.grid_coordinate_entries.get(str(bounds_name))
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
