"""Shared project-table schema foundations, document loading, and units.

Specialized schemas live with their readers in :mod:`cmor._coordinate_tables`
and :mod:`cmor._variable_tables`; only their shared behavior belongs here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import cf_units
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import TableValidationError


class TableModel(BaseModel):
    """Immutable, forward-compatible base for data read from project tables."""

    model_config = ConfigDict(
        frozen=True,
        extra="allow",
        coerce_numbers_to_str=True,
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_empty_strings(cls, data: Any) -> Any:
        """Represent empty table values consistently as missing values."""

        if not isinstance(data, dict):
            return data
        return {
            key: None if isinstance(value, str) and value == "" else value
            for key, value in data.items()
        }


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
            if expected is not None and user_value not in (None, "") and not matches:
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
        """Dimensions in the order used by CMOR runtime objects."""

        return tuple(reversed(self.dimensions))


class _NamedDocument(TableModel):
    """Base that injects mapping keys as immutable entry names."""

    sections: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _name_entries(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            for section in cls.sections:
                cls._name_section(value, section)
        return value

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


def validate_json_file(model: type[TableModel], path: Path) -> TableModel:
    """Parse a table document directly from JSON with Pydantic."""

    return model.model_validate_json(path.read_text())


def units_are_convertible(user_units: str, table_units: str) -> bool:
    """Return True if *user_units* and *table_units* are dimensionally
    compatible.

    Uses ``cf_units`` (a required dependency) for a proper udunits-based check,
    so that physically equivalent unit strings such as ``"degC"`` and ``"K"``
    or ``"hPa"`` and ``"Pa"`` are accepted even when the strings differ.

    If the unit strings cannot be parsed by ``cf_units`` (e.g. non-standard
    strings) the function falls back to requiring exact string equality.

    Parameters
    ----------
    user_units
        Units string supplied by the user.
    table_units
        Units string from the project table entry.

    Returns
    -------
    bool
        True when the units are the same string or are dimensionally
        convertible; False when they belong to different physical dimensions.

    Examples
    --------
    >>> units_are_convertible("degC", "K")
    True
    >>> units_are_convertible("hPa", "Pa")
    True
    >>> units_are_convertible("m s-1", "K")
    False
    >>> units_are_convertible("kg m-2 s-1", "kg m-2 s-1")
    True
    """
    if user_units == table_units:
        return True
    try:
        a = cf_units.Unit(user_units)
        b = cf_units.Unit(table_units)
        return a.is_convertible(b)
    except Exception:
        # Unit strings not parseable by cf_units: require exact equality.
        return False
