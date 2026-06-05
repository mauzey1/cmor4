from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ._table_utils import (
    metadata_value_matches,
    single_or_original,
    table_dimensions,
)
from .exceptions import TableValidationError
from .metadata import _MetadataRecord


@dataclass(frozen=True)
class VariableEntry:
    """Resolved variable table entry."""

    name: str
    table_id: str
    entry: Mapping[str, Any]
    table_file: Path | None = None
    table_header: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Variable(_MetadataRecord):
    """Metadata for the data variable being written."""

    name: str
    id: str | None = None
    variable_id: str | None = None
    table_id: str | None = None
    dimensions: tuple[str, ...] | list[str] | None = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    cell_methods: str | None = None
    cell_measures: str | None = None
    comment: str | None = None
    missing_value: Any = None
    fill_value: Any = None
    chunksizes: tuple[int, ...] | list[int] | None = None
    chunks: tuple[int, ...] | list[int] | None = None
    coordinates: str | tuple[str, ...] | list[str] | None = None
    formula_terms: str | None = None
    frequency: str | None = None
    realm: str | None = None
    table_info: str | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def names(self) -> tuple[str, dict[str, str]]:
        """Return the output variable id and branded-label metadata."""

        branded_name = str(
            self.get("name") or self.get("id") or self.get("variable_id")
        )
        variable_id = str(
            self.get("id")
            or self.get("variable_id")
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

    def resolve_table_entry(self, project: Any) -> VariableEntry:
        """Find a variable table entry by branded name or variable name."""

        requested = str(self.name or self.variable_id or self.id or "")
        entries_by_name = project._variable_entries_by_name
        if requested in entries_by_name:
            entries = entries_by_name[requested]
            if self.table_id:
                matches = [
                    entry
                    for entry in entries
                    if entry.table_id == str(self.table_id)
                ]
                if len(matches) == 1:
                    return matches[0]
                raise TableValidationError(
                    f"Variable {requested!r} was not found in table "
                    f"{self.table_id!r}."
                )
            if len(entries) == 1:
                return entries[0]
            choices = ", ".join(
                f"{entry.table_id}:{entry.name}" for entry in entries
            )
            raise TableValidationError(
                f"Variable {requested!r} is ambiguous across loaded tables; "
                "specify table_id. "
                f"Choices: {choices}."
            )
        matches = [
            entry
            for entry in project.variable_entries.values()
            if str(entry.entry.get("out_name", entry.name)) == requested
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            names = ", ".join(match.name for match in matches[:10])
            raise TableValidationError(
                f"Variable {requested!r} is ambiguous; use one of: {names}."
            )
        raise TableValidationError(
            f"Variable {requested!r} was not found in loaded variable tables."
        )

    def merge_table_entry(self, variable_entry: VariableEntry) -> "Variable":
        """Merge authoritative table metadata with user variable controls."""

        entry = variable_entry.entry
        merged = self.to_dict()
        table_dims = table_dimensions(entry)
        merged.setdefault("name", variable_entry.name)
        merged.setdefault(
            "id", entry.get("out_name", variable_entry.name.split("_", 1)[0])
        )
        merged.setdefault("variable_id", merged["id"])
        merged.setdefault("dimensions", table_dims)
        merged.setdefault(
            "table_id", entry.get("table_id", variable_entry.table_id)
        )
        if variable_entry.table_file is not None:
            merged.setdefault(
                "table_info", f"Name: {variable_entry.table_file.name};"
            )
        if "frequency" in entry:
            merged.setdefault("frequency", entry["frequency"])
        if "modeling_realm" in entry:
            merged.setdefault(
                "realm", single_or_original(entry["modeling_realm"])
            )
        for key in (
            "units",
            "standard_name",
            "long_name",
            "cell_methods",
            "cell_measures",
            "comment",
        ):
            if entry.get(key) not in (None, ""):
                merged[key] = entry[key]
        return Variable.from_mapping(merged)

    def attributes(self, labels: Mapping[str, str]) -> dict[str, Any]:
        """Return NetCDF attributes for this data variable."""

        attrs = self.netcdf_attrs(self.attrs)
        for key in (
            "units",
            "standard_name",
            "long_name",
            "cell_methods",
            "cell_measures",
            "comment",
        ):
            if key in self:
                attrs[key] = self[key]
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

    def validate_against_entry(self, variable_entry: VariableEntry) -> None:
        """Validate variable metadata against a variable table entry."""

        entry = variable_entry.entry
        values = self.to_dict()
        out_name = str(
            entry.get("out_name", variable_entry.name.split("_", 1)[0])
        )
        for key in ("id", "variable_id"):
            if key in values and str(values[key]) != out_name:
                raise TableValidationError(
                    f"{key}={values[key]!r} does not match table "
                    f"out_name {out_name!r}."
                )
        expected_dims = table_dimensions(entry)
        if self.dimensions is not None and tuple(self.dimensions) != expected_dims:
            raise TableValidationError(
                f"dimensions={tuple(self.dimensions)!r} does not match "
                f"{variable_entry.table_id}:{variable_entry.name} "
                f"dimensions {expected_dims!r}."
            )
        for key in (
            "units",
            "standard_name",
            "long_name",
            "cell_methods",
            "cell_measures",
            "comment",
        ):
            expected = entry.get(key)
            if (
                expected not in (None, "")
                and key in values
                and str(values[key]) != str(expected)
            ):
                raise TableValidationError(
                    f"{key}={values[key]!r} does not match "
                    f"{variable_entry.table_id}:{variable_entry.name} "
                    f"value {expected!r}."
                )
        expected_values = {
            "frequency": entry.get("frequency"),
            "realm": entry.get("modeling_realm"),
            "table_id": variable_entry.table_id,
        }
        for key, expected in expected_values.items():
            if (
                expected not in (None, "")
                and key in values
                and not metadata_value_matches(values[key], expected)
            ):
                raise TableValidationError(
                    f"{key}={values[key]!r} does not match "
                    f"{variable_entry.table_id}:{variable_entry.name} "
                    f"value {expected!r}."
                )
