from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


class TableValidationError(ValueError):
    """Raised when user input is not allowed by project tables."""


@dataclass(frozen=True)
class VariableEntry:
    """Resolved variable table entry."""

    name: str
    table_id: str
    entry: Mapping[str, Any]


class ProjectTables:
    """Project CV and variable-table validator.

    Parameters are paths to existing project table files. For example, CMIP7
    can be loaded from ``cmip7-cmor-tables/tables-cvs/cmor-cvs.json`` and
    one or more files under ``cmip7-cmor-tables/tables/``.
    """

    def __init__(
        self, cv_file: str | Path, variable_tables: Sequence[str | Path]
    ):
        self.cv_file = Path(cv_file)
        self.variable_table_files = tuple(
            Path(path) for path in variable_tables
        )
        self.cv = self._read_cv(self.cv_file)
        self.variable_entries: dict[str, VariableEntry] = {}
        self._variable_entries_by_name: dict[str, list[VariableEntry]] = {}
        for table_file in self.variable_table_files:
            self._load_variable_table(table_file)

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        cv_file: str | Path,
        variable_tables: Sequence[str | Path],
    ) -> "ProjectTables":
        """Load tables using paths relative to a project root."""

        root_path = Path(root)
        return cls(
            root_path / cv_file,
            [root_path / table_file for table_file in variable_tables],
        )

    def prepare_inputs(
        self,
        dataset: Mapping[str, Any],
        variable: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate dataset and variable input, returning normalized copies."""

        normalized_dataset = dict(dataset)
        self.validate_dataset(normalized_dataset)
        variable_entry = self.resolve_variable(variable)
        normalized_variable = self.merge_variable(variable, variable_entry)
        self.validate_variable(normalized_variable, variable_entry)
        if (
            "frequency" in normalized_dataset
            and "frequency" in normalized_variable
            and str(normalized_dataset["frequency"])
            != str(normalized_variable["frequency"])
        ):
            raise TableValidationError(
                f"frequency={normalized_dataset['frequency']!r} "
                "does not match "
                f"{variable_entry.table_id}:{variable_entry.name} frequency "
                f"{normalized_variable['frequency']!r}."
            )
        return normalized_dataset, normalized_variable

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None:
        """Validate user-supplied controlled values against the project CV."""

        for key, value in dataset.items():
            if key.startswith("_") or key in {
                "outpath",
                "output_file_template",
                "output_path_template",
            }:
                continue
            if key in self.cv and not self._value_allowed(value, self.cv[key]):
                raise TableValidationError(
                    f"{key}={value!r} is not allowed by {self.cv_file.name}."
                )

    def resolve_variable(self, variable: Mapping[str, Any]) -> VariableEntry:
        """Find a variable table entry by branded name or variable name."""

        requested = str(
            variable.get("name")
            or variable.get("variable_id")
            or variable.get("id")
            or ""
        )
        if requested in self._variable_entries_by_name:
            return self._select_entry(
                requested,
                self._variable_entries_by_name[requested],
                variable.get("table_id"),
            )
        matches = [
            entry
            for entry in self.variable_entries.values()
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

    def _select_entry(
        self,
        requested: str,
        entries: Sequence[VariableEntry],
        table_id: Any,
    ) -> VariableEntry:
        if table_id:
            matches = [
                entry for entry in entries if entry.table_id == str(table_id)
            ]
            if len(matches) == 1:
                return matches[0]
            raise TableValidationError(
                f"Variable {requested!r} was not found in table {table_id!r}."
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

    def merge_variable(
        self,
        variable: Mapping[str, Any],
        variable_entry: VariableEntry,
    ) -> dict[str, Any]:
        """Merge authoritative table metadata with user variable controls."""

        entry = variable_entry.entry
        merged = dict(variable)
        table_dimensions = _table_dimensions(entry)
        merged.setdefault("name", variable_entry.name)
        merged.setdefault(
            "id", entry.get("out_name", variable_entry.name.split("_", 1)[0])
        )
        merged.setdefault("variable_id", merged["id"])
        merged.setdefault("dimensions", table_dimensions)
        merged.setdefault(
            "table_id", entry.get("table_id", variable_entry.table_id)
        )
        if "frequency" in entry:
            merged.setdefault("frequency", entry["frequency"])
        if "modeling_realm" in entry:
            merged.setdefault(
                "realm", _single_or_original(entry["modeling_realm"])
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
                merged.setdefault(key, entry[key])
        return merged

    def validate_variable(
        self,
        variable: Mapping[str, Any],
        variable_entry: VariableEntry,
    ) -> None:
        """Validate user variable metadata against a variable table entry."""

        entry = variable_entry.entry
        out_name = str(
            entry.get("out_name", variable_entry.name.split("_", 1)[0])
        )
        for key in ("id", "variable_id"):
            if key in variable and str(variable[key]) != out_name:
                raise TableValidationError(
                    f"{key}={variable[key]!r} does not match table "
                    f"out_name {out_name!r}."
                )
        expected_dims = _table_dimensions(entry)
        if (
            "dimensions" in variable
            and tuple(variable["dimensions"]) != expected_dims
        ):
            raise TableValidationError(
                f"dimensions={tuple(variable['dimensions'])!r} does not match "
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
                and key in variable
                and str(variable[key]) != str(expected)
            ):
                raise TableValidationError(
                    f"{key}={variable[key]!r} does not match "
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
                and key in variable
                and not _metadata_value_matches(variable[key], expected)
            ):
                raise TableValidationError(
                    f"{key}={variable[key]!r} does not match "
                    f"{variable_entry.table_id}:{variable_entry.name} "
                    f"value {expected!r}."
                )

    @staticmethod
    def _read_cv(cv_file: Path) -> Mapping[str, Any]:
        with cv_file.open() as handle:
            data = json.load(handle)
        return data.get("CV", data)

    def _load_variable_table(self, table_file: Path) -> None:
        with table_file.open() as handle:
            data = json.load(handle)
        entries = data.get("variable_entry", {})
        table_id = _normalize_table_id(
            data.get("Header", {}).get("table_id") or table_file.stem
        )
        for name, entry in entries.items():
            variable_entry = VariableEntry(
                name=name, table_id=str(table_id), entry=entry
            )
            self.variable_entries.setdefault(name, variable_entry)
            self._variable_entries_by_name.setdefault(name, []).append(
                variable_entry
            )

    def _value_allowed(self, value: Any, allowed: Any) -> bool:
        if isinstance(allowed, Mapping):
            return str(value) in allowed
        if isinstance(allowed, str):
            return str(value) == allowed
        if isinstance(allowed, list):
            return any(
                _allowed_list_item(str(value), item) for item in allowed
            )
        return True


def _table_dimensions(entry: Mapping[str, Any]) -> tuple[str, ...]:
    dimensions = entry.get("dimensions", ())
    if isinstance(dimensions, str):
        values = tuple(dimensions.split())
    else:
        values = tuple(str(value) for value in dimensions)
    return tuple(reversed(values))


def _allowed_list_item(value: str, item: Any) -> bool:
    if value == str(item):
        return True
    if not isinstance(item, str):
        return False
    pattern = _posix_regex_to_python(item)
    try:
        return re.fullmatch(pattern, value) is not None
    except re.error:
        return False


def _posix_regex_to_python(pattern: str) -> str:
    return (
        pattern.replace("[[:digit:]]", r"\d")
        .replace("\\{", "{")
        .replace("\\}", "}")
    )


def _normalize_table_id(value: Any) -> str:
    text = str(value)
    if text.startswith("Table "):
        return text.removeprefix("Table ")
    return text


def _single_or_original(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _metadata_value_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return str(value) in {str(item) for item in expected}
    return str(value) == str(expected)
