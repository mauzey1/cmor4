"""Typed models for project-table documents and entries.

The JSON table schema is intentionally kept separate from the runtime metadata
models in :mod:`cmor4.axis`, :mod:`cmor4.variable`, and related modules.  Table
rows describe authoritative defaults and constraints; runtime models describe
fully constructed objects and may contain large array-like values.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..exceptions import TableValidationError


def _numeric_table_value(value: Any) -> Any:
    """Normalize a numeric scalar encoded as a JSON string."""

    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


class TableModel(BaseModel):
    """Immutable, forward-compatible base for data read from project tables."""

    model_config = ConfigDict(
        frozen=True,
        extra="allow",
        coerce_numbers_to_str=True,
        populate_by_name=True,
    )


class TableHeader(TableModel):
    """Known table-header fields, with project extensions retained by Pydantic."""

    table_id: str | None = None
    Conventions: str | None = None
    data_specs_version: str | None = None
    valid_min: Any = None
    valid_max: Any = None
    ok_min_mean_abs: Any = None
    ok_max_mean_abs: Any = None


class NamedTableEntry(TableModel):
    """Common identity and metadata fields shared by table row types."""

    name: str = Field(min_length=1)
    out_name: str | None = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    type: str | None = None

    def validate_metadata(
        self,
        data: dict[str, Any],
        keys: Iterable[str],
        entity_type: str = "entry",
    ) -> None:
        """Reject runtime metadata that conflicts with this typed entry."""

        for key in keys:
            expected = getattr(self, key, None)
            user_value = data.get(key)
            expected_text = str(expected)
            if expected_text.endswith(" since ?"):
                matches = " since " in str(user_value)
            elif "?" in expected_text:
                pattern = re.escape(expected_text).replace(r"\?", ".+")
                matches = re.fullmatch(pattern, str(user_value)) is not None
            else:
                matches = str(user_value) == expected_text
            if (
                expected not in (None, "")
                and user_value not in (None, "")
                and not matches
            ):
                raise TableValidationError(
                    f"{entity_type} {self.name!r} {key}={user_value!r} "
                    f"does not match table value {expected!r}."
                )


class DimensionedTableEntry(NamedTableEntry):
    """Table entry whose dimensions are stored in table (opposite) order."""

    dimensions: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_dimensions(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        value = data.get("dimensions")
        if isinstance(value, str):
            data["dimensions"] = tuple(value.split())
        elif value is None:
            data["dimensions"] = ()
        return data

    @property
    def runtime_dimensions(self) -> tuple[str, ...]:
        """Dimensions in the order used by CMOR4 runtime objects."""

        return tuple(reversed(self.dimensions))


class CoordinateTableEntry(DimensionedTableEntry):
    """One coordinate-table or grid-coordinate row."""

    axis: str | None = None
    positive: str | None = None
    formula: str | None = None
    climatology: Any = None
    generic_level_name: str | None = None
    z_factors: str | None = None
    z_bounds_factors: str | None = None
    valid_min: Any = None
    valid_max: Any = None
    requested: Any = None
    requested_bounds: Any = None
    bounds_values: Any = None
    value: Any = None
    must_have_bounds: Any = None
    stored_direction: str | None = None
    tolerance: Any = None
    is_grid_coord: bool = False

    @property
    def is_scalar(self) -> bool:
        return self.value not in (None, "")

    @property
    def runtime_values(self) -> list[Any] | None:
        value = self.requested if self.requested not in (None, "") else self.value
        if value in (None, ""):
            return None
        values = value if isinstance(value, (list, tuple)) else [value]
        return [_numeric_table_value(item) for item in values]

    @property
    def runtime_bounds(self) -> list[list[Any]] | None:
        value = self.requested_bounds
        if value in (None, ""):
            value = self.bounds_values
        if value in (None, ""):
            return None
        values = value if isinstance(value, (list, tuple)) else str(value).split()
        parsed = [_numeric_table_value(item) for item in values]
        if len(parsed) % 2:
            return None
        return [parsed[index : index + 2] for index in range(0, len(parsed), 2)]


class FormulaTableEntry(DimensionedTableEntry):
    """One formula-term table row."""

    valid_min: Any = None
    valid_max: Any = None
    ok_min_mean_abs: Any = None
    ok_max_mean_abs: Any = None


class GridMappingTableEntry(NamedTableEntry):
    """One grid-mapping row, including extension-defined projection fields."""

    mapping_name: str | None = None
    grid_mapping_name: str | None = None
    mapping_var: str | None = None
    coordinates: tuple[str, ...] = ()
    params: dict[str, Any] = Field(default_factory=dict)
    required_params: Any = None
    required_parameters: Any = None
    required: Any = None
    optional_params: Any = None
    optional_parameters: Any = None
    parameters: Any = None
    text_params: Any = None
    text_parameters: Any = None
    required_axes: Any = None
    required_axis: Any = None
    axes: Any = None
    axis: Any = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_coordinates(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        value = data.get("coordinates")
        if isinstance(value, str):
            data["coordinates"] = tuple(value.split())
        elif value is None:
            data["coordinates"] = ()
        return data

    @staticmethod
    def _tokens(*values: Any) -> tuple[str, ...]:
        result: list[str] = []
        for value in values:
            if value in (None, ""):
                continue
            if isinstance(value, str):
                candidates: Iterable[Any] = re.split(r"[\s,]+", value)
            elif isinstance(value, dict):
                candidates = value.keys()
            elif isinstance(value, Iterable):
                candidates = value
            else:
                candidates = (value,)
            for candidate in candidates:
                token = str(candidate)
                if token and token not in result:
                    result.append(token)
        return tuple(result)

    def numbered_parameters(self) -> tuple[str, ...]:
        numbered: list[tuple[int, str]] = []
        for key, value in (self.model_extra or {}).items():
            match = re.fullmatch(r"parameter(\d+)", key)
            if match and value not in (None, ""):
                numbered.append((int(match.group(1)), str(value)))
        return tuple(value for _, value in sorted(numbered))

    def required_parameter_names(self) -> tuple[str, ...]:
        return self._tokens(
            self.required_params,
            self.required_parameters,
            self.required,
            self.numbered_parameters(),
        )

    def optional_parameter_names(self) -> tuple[str, ...]:
        return self._tokens(
            self.optional_params,
            self.optional_parameters,
            self.parameters,
            self.params,
        )

    def text_parameter_names(self) -> tuple[str, ...]:
        declared = self._tokens(self.text_params, self.text_parameters)
        known = tuple(
            name
            for name in (
                *self.required_parameter_names(),
                *self.optional_parameter_names(),
            )
            if name in {"crs_wkt", "GeoTransform", "spatial_ref"}
        )
        return self._tokens(declared, known)

    def required_axis_names(self) -> tuple[str, ...]:
        return self._tokens(
            self.required_axes, self.required_axis, self.axes, self.axis
        )


class VariableTableRow(DimensionedTableEntry):
    """One variable-table JSON row before source provenance is attached."""

    frequency: Any = None
    modeling_realm: Any = None
    cell_methods: str | None = None
    cell_measures: str | None = None
    comment: str | None = None
    positive: str | None = None
    flag_values: str | None = None
    flag_meanings: str | None = None
    required: str | None = None
    valid_min: Any = None
    valid_max: Any = None
    ok_min_mean_abs: Any = None
    ok_max_mean_abs: Any = None


class VariableTableEntry(VariableTableRow):
    """A validated variable row plus its source-table provenance."""

    table_id: str = Field(min_length=1)
    table_file: Path | None = None
    table_header: TableHeader | None = None
    contextual_attrs: tuple[str, ...] = ()


class _NamedDocument(TableModel):
    """Base that injects mapping keys as immutable entry names."""

    @staticmethod
    def _name_section(data: dict[str, Any], section: str) -> None:
        rows = data.get(section)
        if isinstance(rows, dict):
            data[section] = {
                str(name): (
                    {**row, "name": str(name)} if isinstance(row, dict) else row
                )
                for name, row in rows.items()
            }


class CoordinateTableDocument(_NamedDocument):
    header: TableHeader | None = Field(default=None, alias="Header")
    axis_entry: dict[str, CoordinateTableEntry] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _name_entries(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            cls._name_section(value, "axis_entry")
        return value


class FormulaTableDocument(_NamedDocument):
    header: TableHeader | None = Field(default=None, alias="Header")
    formula_entry: dict[str, FormulaTableEntry] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _name_entries(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            cls._name_section(value, "formula_entry")
        return value


class GridTableDocument(_NamedDocument):
    header: TableHeader | None = Field(default=None, alias="Header")
    axis_entry: dict[str, CoordinateTableEntry] = Field(default_factory=dict)
    variable_entry: dict[str, CoordinateTableEntry] = Field(default_factory=dict)
    mapping_entry: dict[str, GridMappingTableEntry] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _name_entries(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            for section in ("axis_entry", "variable_entry", "mapping_entry"):
                cls._name_section(value, section)
        return value


class VariableTableDocument(_NamedDocument):
    """Typed variable-table file; provenance is attached after validation."""

    header: TableHeader | None = Field(default=None, alias="Header")
    variable_entry: dict[str, VariableTableRow] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _name_entries(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            cls._name_section(value, "variable_entry")
        return value

    def resolved_entries(self, table_file: Path) -> dict[str, VariableTableEntry]:
        header = self.header or TableHeader()
        table_id = str(header.table_id or table_file.stem)
        if table_id.startswith("Table "):
            table_id = table_id.removeprefix("Table ")
        return {
            name: VariableTableEntry.model_validate({
                **row.model_dump(),
                "table_id": table_id,
                "table_file": table_file,
                "table_header": header,
            })
            for name, row in self.variable_entry.items()
        }


def validate_json_file(model: type[TableModel], path: Path) -> TableModel:
    """Parse a table document directly from JSON with Pydantic."""

    return model.model_validate_json(path.read_text())
