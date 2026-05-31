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
    can be loaded from ``cmip7-cmor-tables/tables-cvs/cmor-cvs.json``, one or
    more variable tables under ``cmip7-cmor-tables/tables/``, and the project
    coordinate and formula-term tables.
    """

    def __init__(
        self,
        cv_file: str | Path,
        variable_tables: Sequence[str | Path],
        coordinate_table: str | Path | None = None,
        formula_table: str | Path | None = None,
    ):
        self.cv_file = Path(cv_file)
        self.variable_table_files = tuple(
            Path(path) for path in variable_tables
        )
        self.coordinate_table_file = (
            Path(coordinate_table) if coordinate_table is not None else None
        )
        self.formula_table_file = (
            Path(formula_table) if formula_table is not None else None
        )
        self.cv = self._read_cv(self.cv_file)
        self.variable_entries: dict[str, VariableEntry] = {}
        self._variable_entries_by_name: dict[str, list[VariableEntry]] = {}
        for table_file in self.variable_table_files:
            self._load_variable_table(table_file)
        self.coordinate_entries: dict[str, Mapping[str, Any]] = {}
        if self.coordinate_table_file is not None:
            self.coordinate_entries = self._read_entries(
                self.coordinate_table_file, "axis_entry"
            )
        self.coordinate_aliases = _coordinate_aliases(
            self.coordinate_entries
        )
        self.formula_entries: dict[str, Mapping[str, Any]] = {}
        if self.formula_table_file is not None:
            self.formula_entries = self._read_entries(
                self.formula_table_file, "formula_entry"
            )

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        cv_file: str | Path,
        variable_tables: Sequence[str | Path],
        coordinate_table: str | Path | None = None,
        formula_table: str | Path | None = None,
    ) -> "ProjectTables":
        """Load tables using paths relative to a project root."""

        root_path = Path(root)
        resolved_coordinate_table = _resolve_optional_table(
            root_path, coordinate_table, "coordinate"
        )
        resolved_formula_table = _resolve_optional_table(
            root_path, formula_table, "formula_terms"
        )
        return cls(
            root_path / cv_file,
            [root_path / table_file for table_file in variable_tables],
            coordinate_table=resolved_coordinate_table,
            formula_table=resolved_formula_table,
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

    def prepare_axes(
        self, axes: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        """Merge coordinate-axis metadata from the loaded coordinate table."""

        return tuple(self.merge_axis(axis) for axis in axes)

    def prepare_zfactors(
        self, zfactors: Sequence[Mapping[str, Any]] | None
    ) -> tuple[dict[str, Any], ...] | None:
        """Merge z-factor metadata from the loaded formula-term table."""

        if zfactors is None:
            return None
        return tuple(self.merge_zfactor(zfactor) for zfactor in zfactors)

    def merge_axis(self, axis: Mapping[str, Any]) -> dict[str, Any]:
        """Merge authoritative coordinate metadata into an axis definition."""

        merged = dict(axis)
        entry_name, entry = self.resolve_axis(axis)
        if entry is None:
            return merged
        merged.setdefault("table_entry", entry_name)
        self._validate_table_metadata(
            "axis",
            entry_name,
            axis,
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
            "generic_level_name",
            "z_factors",
            "z_bounds_factors",
        ):
            value = entry.get(key)
            if _is_table_value(value):
                merged.setdefault(key, value)
        if "values" not in merged:
            values = _entry_values(entry)
            if values is not None:
                merged["values"] = values
        if "bounds" not in merged:
            bounds = _entry_bounds(entry)
            if bounds is not None:
                merged["bounds"] = bounds
        return merged

    def resolve_axis(
        self, axis: Mapping[str, Any]
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve an axis table entry from a user axis definition."""

        requested = str(
            axis.get("table_entry")
            or axis.get("axis_entry")
            or axis.get("coordinate")
            or axis.get("name")
            or ""
        )
        if requested in self.coordinate_entries:
            return requested, self.coordinate_entries[requested]
        matching_out_names = [
            (name, entry)
            for name, entry in self.coordinate_entries.items()
            if str(entry.get("out_name", "")) == requested
        ]
        if len(matching_out_names) == 1:
            return matching_out_names[0]
        matches = self._matching_coordinate_entries(axis)
        if len(matches) == 1:
            return matches[0]
        return None, None

    def merge_zfactor(self, zfactor: Mapping[str, Any]) -> dict[str, Any]:
        """Merge authoritative formula-term metadata into a z-factor."""

        merged = dict(zfactor)
        entry_name, entry = self.resolve_zfactor(zfactor)
        if entry is None:
            return merged
        merged.setdefault("table_entry", entry_name)
        self._validate_table_metadata(
            "formula term",
            entry_name,
            zfactor,
            entry,
            ("units", "standard_name", "long_name"),
        )
        for key in ("out_name", "units", "standard_name", "long_name"):
            value = entry.get(key)
            if _is_table_value(value):
                merged.setdefault(key, value)
        if "dimensions" not in merged and _is_table_value(
            entry.get("dimensions")
        ):
            merged["dimensions"] = _table_dimensions(entry)
        if "bounds" in merged:
            bounds_name = str(
                merged.get("bounds_name")
                or f"{merged.get('out_name', zfactor['name'])}_bnds"
            )
            bounds_entry = self.formula_entries.get(bounds_name)
            if bounds_entry:
                merged.setdefault("bounds_name", bounds_name)
                bounds_attrs = dict(merged.get("bounds_attrs", {}))
                for key in ("units", "standard_name", "long_name"):
                    value = bounds_entry.get(key)
                    if _is_table_value(value):
                        bounds_attrs.setdefault(key, value)
                if bounds_attrs:
                    merged["bounds_attrs"] = bounds_attrs
        return merged

    def resolve_zfactor(
        self, zfactor: Mapping[str, Any]
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve a formula-term table entry from a user z-factor."""

        requested = str(
            zfactor.get("table_entry")
            or zfactor.get("formula_entry")
            or zfactor.get("name")
            or ""
        )
        if requested in self.formula_entries:
            return requested, self.formula_entries[requested]
        matches = [
            (name, entry)
            for name, entry in self.formula_entries.items()
            if str(entry.get("out_name", "")) == requested
        ]
        if len(matches) == 1:
            return matches[0]
        return None, None

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

    @staticmethod
    def _read_entries(table_file: Path, key: str) -> dict[str, Mapping[str, Any]]:
        with table_file.open() as handle:
            data = json.load(handle)
        return {
            str(name): entry
            for name, entry in data.get(key, {}).items()
            if isinstance(entry, Mapping)
        }

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

    def _matching_coordinate_entries(
        self, axis: Mapping[str, Any]
    ) -> list[tuple[str, Mapping[str, Any]]]:
        matches = list(self.coordinate_entries.items())
        for key in ("out_name", "standard_name", "axis"):
            value = axis.get(key)
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

    def _validate_table_metadata(
        self,
        entry_type: str,
        entry_name: str | None,
        user_values: Mapping[str, Any],
        table_values: Mapping[str, Any],
        keys: Sequence[str],
    ) -> None:
        for key in keys:
            expected = table_values.get(key)
            if (
                _is_table_value(expected)
                and key in user_values
                and not _metadata_value_matches(user_values[key], expected)
            ):
                raise TableValidationError(
                    f"{entry_type} {entry_name!r} {key}={user_values[key]!r} "
                    f"does not match table value {expected!r}."
                )


def _table_dimensions(entry: Mapping[str, Any]) -> tuple[str, ...]:
    dimensions = entry.get("dimensions", ())
    if isinstance(dimensions, str):
        values = tuple(dimensions.split())
    else:
        values = tuple(str(value) for value in dimensions)
    return tuple(reversed(values))


def _coordinate_aliases(
    entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name, entry in entries.items():
        out_name = entry.get("out_name")
        if not _is_table_value(out_name):
            continue
        aliases[str(name)] = str(out_name)
        generic_level_name = entry.get("generic_level_name")
        if _is_table_value(generic_level_name):
            aliases[str(generic_level_name)] = str(out_name)
    return aliases


def _resolve_optional_table(
    root_path: Path, table: str | Path | None, suffix: str
) -> Path | None:
    if table is not None:
        return root_path / table
    for directory in ("tables", "Tables"):
        table_dir = root_path / directory
        if not table_dir.exists():
            continue
        matches = sorted(table_dir.glob(f"*_{suffix}.json"))
        if matches:
            return matches[0]
    return None


def _is_table_value(value: Any) -> bool:
    return value not in (None, "")


def _entry_values(entry: Mapping[str, Any]) -> list[Any] | None:
    requested = entry.get("requested")
    if _is_table_value(requested):
        if isinstance(requested, list):
            return [_parse_table_value(value) for value in requested]
        return [_parse_table_value(requested)]
    value = entry.get("value")
    if _is_table_value(value):
        return [_parse_table_value(value)]
    return None


def _entry_bounds(entry: Mapping[str, Any]) -> list[list[Any]] | None:
    requested_bounds = entry.get("requested_bounds")
    if not _is_table_value(requested_bounds):
        requested_bounds = entry.get("bounds_values")
    if not _is_table_value(requested_bounds):
        return None
    values = (
        requested_bounds
        if isinstance(requested_bounds, list)
        else str(requested_bounds).split()
    )
    parsed = [_parse_table_value(value) for value in values]
    if len(parsed) % 2:
        return None
    return [parsed[index : index + 2] for index in range(0, len(parsed), 2)]


def _parse_table_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


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
    expected_text = str(expected)
    if expected_text.endswith(" since ?"):
        return " since " in str(value)
    if "?" in expected_text:
        pattern = re.escape(expected_text).replace(r"\?", ".+")
        return re.fullmatch(pattern, str(value)) is not None
    return str(value) == str(expected)
