from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._table_utils import (
    is_table_value as _is_table_value,
    single_or_original as _single_or_original,
)
from .axis import Axis
from .cv import ControlledVocabulary
from .dataset import DatasetInfo
from .exceptions import TableValidationError
from .grid import Grid
from .variable import Variable, VariableEntry
from .zfactor import ZFactor


class ProjectTables:
    """Project CV and variable-table validator.

    Parameters are paths to existing project table files. For example, CMIP7
    can be loaded from ``cmip7-cmor-tables/tables-cvs/cmor-cvs.json``, one or
    more variable tables under ``cmip7-cmor-tables/tables/``, and the project
    coordinate and formula-term tables.

    Parameters
    ----------
    cv_file:
        Path to the project controlled-vocabulary JSON file.
    variable_tables:
        Paths to variable table JSON files.
    coordinate_table:
        Optional path to the coordinate table JSON file.
    formula_table:
        Optional path to the formula-terms table JSON file.
    grid_table:
        Optional path to the grids table JSON file.
    """

    def __init__(
        self,
        cv_file: str | Path,
        variable_tables: Sequence[str | Path],
        coordinate_table: str | Path | None = None,
        formula_table: str | Path | None = None,
        grid_table: str | Path | None = None,
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
        self.grid_table_file = (
            Path(grid_table) if grid_table is not None else None
        )
        self.cv = ControlledVocabulary.from_file(self.cv_file)
        self.variable_entries: dict[str, VariableEntry] = {}
        self._variable_entries_by_name: dict[str, list[VariableEntry]] = {}
        for table_file in self.variable_table_files:
            self._load_variable_table(table_file)
        self.grid_axis_entries: dict[str, Mapping[str, Any]] = {}
        self.grid_coordinate_entries: dict[str, Mapping[str, Any]] = {}
        self.grid_mapping_entries: dict[str, Mapping[str, Any]] = {}
        if self.grid_table_file is not None:
            self.grid_axis_entries = self._read_entries(
                self.grid_table_file, "axis_entry"
            )
            self.grid_coordinate_entries = self._read_entries(
                self.grid_table_file, "variable_entry"
            )
            self.grid_mapping_entries = self._read_entries(
                self.grid_table_file, "mapping_entry"
            )
        coordinate_entries: dict[str, Mapping[str, Any]] = {}
        if self.coordinate_table_file is not None:
            coordinate_entries = self._read_entries(
                self.coordinate_table_file, "axis_entry"
            )
        self.coordinate_entries = _overlay_table_entries(
            coordinate_entries, self.grid_axis_entries
        )
        self.generic_level_entries = _generic_level_entries(
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
        grid_table: str | Path | None = None,
    ) -> "ProjectTables":
        """Load tables using paths relative to a project root.

        Parameters
        ----------
        root:
            Project table root directory.
        cv_file:
            Controlled-vocabulary file path relative to ``root``.
        variable_tables:
            Variable table paths relative to ``root``.
        coordinate_table:
            Optional coordinate table path relative to ``root``.
        formula_table:
            Optional formula-terms table path relative to ``root``.
        grid_table:
            Optional grids table path relative to ``root``.

        Returns
        -------
        ProjectTables
            Loaded project table helper.
        """

        root_path = Path(root)
        resolved_coordinate_table = _resolve_optional_table(
            root_path, coordinate_table, "coordinate"
        )
        resolved_formula_table = _resolve_optional_table(
            root_path, formula_table, "formula_terms"
        )
        resolved_grid_table = _resolve_optional_table(
            root_path, grid_table, "grids"
        )
        return cls(
            root_path / cv_file,
            [root_path / table_file for table_file in variable_tables],
            coordinate_table=resolved_coordinate_table,
            formula_table=resolved_formula_table,
            grid_table=resolved_grid_table,
        )

    def dataset_info(
        self,
        dataset: Mapping[str, Any],
    ) -> DatasetInfo:
        """Create prepared dataset metadata from user input and tables.

        Parameters
        ----------
        dataset:
            User-provided dataset-level metadata.

        Returns
        -------
        DatasetInfo
            Validated and defaulted dataset metadata.
        """

        user_info = (
            dataset.user_info
            if isinstance(dataset, DatasetInfo)
            else dataset
        )
        normalized_dataset = self.cv.get_dataset_info(dataset)
        self.cv.validate_dataset_values(normalized_dataset)
        self.validate_source_attributes(normalized_dataset)
        self.validate_experiment(normalized_dataset)
        self.validate_parent_attributes(normalized_dataset)
        return DatasetInfo(
            normalized_dataset,
            project=self,
            user_info=user_info,
        )

    def _dataset_for_variable(
        self,
        dataset: DatasetInfo,
        variable: Variable,
    ) -> tuple[DatasetInfo, Variable]:
        user_info = dataset.user_info
        normalized_dataset = self.cv.get_dataset_info(dataset)
        variable_entry = variable.resolve_table_entry(self)
        self._add_table_header_defaults(normalized_dataset, variable_entry)
        normalized_variable = variable.merge_table_entry(variable_entry)
        self._add_variable_global_defaults(
            normalized_dataset, normalized_variable
        )
        self.validate_dataset(normalized_dataset)
        self.validate_source_attributes(normalized_dataset)
        self.validate_experiment(normalized_dataset)
        self.validate_parent_attributes(normalized_dataset)
        normalized_variable.validate_against_entry(variable_entry)
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
        return DatasetInfo(
            normalized_dataset,
            project=self,
            user_info=user_info,
        ), normalized_variable

    def variable(self, name: str, **values: Any) -> Variable:
        """Create a variable with metadata from the loaded variable tables.

        Parameters
        ----------
        name:
            Variable or branded variable name to resolve.
        **values:
            User-supplied variable metadata overrides.

        Returns
        -------
        Variable
            Variable metadata with table values merged.
        """

        variable = Variable(name=name, **values)
        variable_entry = variable.resolve_table_entry(self)
        normalized = variable.merge_table_entry(variable_entry)
        normalized.validate_against_entry(variable_entry)
        return normalized

    def axis(self, name: str, **values: Any) -> Axis:
        """Create an axis with metadata from the loaded coordinate tables.

        Parameters
        ----------
        name:
            Axis or coordinate table entry name.
        **values:
            User-supplied axis metadata and coordinate values.

        Returns
        -------
        Axis
            Axis metadata with table values merged.
        """

        return self._mark_prepared_axis(
            Axis(name=name, **values).merge_table_entry(self)
        )

    def _axes(
        self,
        axes: Sequence[Axis],
        variable: Variable | None = None,
    ) -> tuple[Axis, ...]:
        """Create a complete axis tuple, including required scalar axes."""

        merged_axes = [
            axis if self._is_prepared_axis(axis)
            else axis.merge_table_entry(self)
            for axis in axes
        ]
        if variable is not None:
            merged_axes.extend(
                Axis.missing_scalar_axes(self, merged_axes, variable)
            )
        return tuple(self._mark_prepared_axis(axis) for axis in merged_axes)

    def grid(self, name: str | None = None, **values: Any) -> Grid:
        """Create a grid with metadata from the loaded grid table.

        Parameters
        ----------
        name:
            Optional grid mapping entry name.
        **values:
            User-supplied grid metadata overrides.

        Returns
        -------
        Grid
            Grid metadata with table values merged.
        """

        return Grid(name=name, **values).merge_table_entry(self)

    def zfactor(self, name: str, **values: Any) -> ZFactor:
        """Create a z-factor with metadata from formula-term tables.

        Parameters
        ----------
        name:
            Formula-term table entry name.
        **values:
            User-supplied formula-term metadata and values.

        Returns
        -------
        ZFactor
            Z-factor metadata with table values merged.
        """

        return ZFactor(name=name, **values).merge_table_entry(self)

    def _mark_prepared_axis(self, axis: Axis) -> Axis:
        object.__setattr__(axis, "_cmor4_project_tables", self)
        return axis

    def _is_prepared_axis(self, axis: Axis) -> bool:
        return getattr(axis, "_cmor4_project_tables", None) is self

    def _add_table_header_defaults(
        self, dataset: dict[str, Any], variable_entry: VariableEntry
    ) -> None:
        """Fill defaults from the loaded variable table header."""

        header = variable_entry.table_header or {}
        for key in ("Conventions", "data_specs_version"):
            value = header.get(key)
            if _is_table_value(value):
                dataset.setdefault(key, value)

    def _add_variable_global_defaults(
        self, dataset: dict[str, Any], variable: Variable
    ) -> None:
        """Fill global attributes that are derived from the variable table."""

        variable_id, labels = variable.names()
        if "variable_id" not in dataset or _is_unresolved_template(
            dataset["variable_id"]
        ):
            dataset["variable_id"] = variable_id
        if "branded_variable" not in dataset or _is_unresolved_template(
            dataset["branded_variable"]
        ):
            dataset["branded_variable"] = labels["branded_name"]
        for key in (
            "branding_suffix",
            "temporal_label",
            "vertical_label",
            "horizontal_label",
            "area_label",
        ):
            if key in labels:
                if key not in dataset or _is_unresolved_template(
                    dataset[key]
                ):
                    dataset[key] = labels[key]
        for key in ("frequency", "realm", "table_id"):
            value = variable.get(key)
            if _is_table_value(value):
                dataset.setdefault(key, _single_or_original(value))

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None:
        """Validate user-supplied controlled values against the project CV.

        Parameters
        ----------
        dataset:
            Dataset metadata to validate.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if validation fails.
        """

        self.cv.validate_dataset(dataset)

    def validate_required_global_attributes(
        self, dataset: Mapping[str, Any]
    ) -> None:
        """Require every CV-listed global attribute that CMOR4 can write.

        Parameters
        ----------
        dataset:
            Dataset metadata to check.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if required attributes are
            missing.
        """

        self.cv.validate_required_global_attributes(dataset)

    def required_global_attributes(self) -> tuple[str, ...]:
        """Return CV-listed required global attributes.

        Returns
        -------
        tuple[str, ...]
            Required global attribute names.
        """

        return self.cv.required_global_attributes()

    def validate_experiment(self, dataset: Mapping[str, Any]) -> None:
        """Validate experiment-specific CV attributes.

        Parameters
        ----------
        dataset:
            Dataset metadata containing an ``experiment_id``.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if experiment metadata is
            inconsistent.
        """

        self.cv.validate_experiment(dataset)

    def validate_source_type(
        self,
        dataset: Mapping[str, Any],
        experiment_entry: Mapping[str, Any],
    ) -> None:
        """Validate experiment-specific required source_type tokens.

        Parameters
        ----------
        dataset:
            Dataset metadata containing ``source_type``.
        experiment_entry:
            Experiment CV entry with required and allowed source types.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if source types are missing
            or disallowed.
        """

        self.cv.validate_source_type(dataset, experiment_entry)

    def validate_source_attributes(self, dataset: Mapping[str, Any]) -> None:
        """Validate source_id-specific CV attributes.

        Parameters
        ----------
        dataset:
            Dataset metadata containing a ``source_id``.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if source-specific metadata
            is inconsistent.
        """

        self.cv.validate_source_attributes(dataset)

    def validate_parent_attributes(self, dataset: Mapping[str, Any]) -> None:
        """Validate CMIP-style parent experiment attributes.

        Parameters
        ----------
        dataset:
            Dataset metadata containing experiment and parent metadata.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if parent metadata is missing
            or inconsistent.
        """

        self.cv.validate_parent_attributes(dataset)

    @staticmethod
    def _read_entries(
        table_file: Path, key: str
    ) -> dict[str, Mapping[str, Any]]:
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
        table_id = str(
            data.get("Header", {}).get("table_id") or table_file.stem
        )
        if table_id.startswith("Table "):
            table_id = table_id.removeprefix("Table ")
        for name, entry in entries.items():
            variable_entry = VariableEntry(
                name=name,
                table_id=str(table_id),
                entry=entry,
                table_file=table_file,
                table_header=data.get("Header", {}),
            )
            self.variable_entries.setdefault(name, variable_entry)
            self._variable_entries_by_name.setdefault(name, []).append(
                variable_entry
            )


def _generic_level_entries(
    entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Mapping[str, Any]]]:
    generic_entries: dict[str, dict[str, Mapping[str, Any]]] = {}
    for name, entry in entries.items():
        generic_level_name = entry.get("generic_level_name")
        if _is_table_value(generic_level_name):
            generic_entries.setdefault(str(generic_level_name), {})[
                str(name)
            ] = entry
    return generic_entries


def _overlay_table_entries(
    base_entries: Mapping[str, Mapping[str, Any]],
    overlay_entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    entries = {name: dict(entry) for name, entry in base_entries.items()}
    for name, overlay in overlay_entries.items():
        entry = dict(entries.get(name, {}))
        for key, value in overlay.items():
            if _is_table_value(value) or key not in entry:
                entry[key] = value
        entries[name] = entry
    return entries


def _is_unresolved_template(value: Any) -> bool:
    return isinstance(value, str) and "<" in value and ">" in value


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
