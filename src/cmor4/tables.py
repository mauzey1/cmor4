from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import uuid

from ._table_utils import (
    is_table_value as _is_table_value,
    metadata_value_matches as _metadata_value_matches,
    single_or_original as _single_or_original,
)
from .axis import Axis
from .exceptions import TableValidationError
from .variable import Variable, VariableEntry
from .zfactor import ZFactor


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
        self.grid_table_file = Path(grid_table) if grid_table is not None else None
        self.cv = self._read_cv(self.cv_file)
        self.variable_entries: dict[str, VariableEntry] = {}
        self._variable_entries_by_name: dict[str, list[VariableEntry]] = {}
        self._variable_axis_entries: dict[str, Mapping[str, Any]] = {}
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
            _overlay_table_entries(coordinate_entries, self.grid_axis_entries),
            self._variable_axis_entries,
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
        grid_table: str | Path | None = None,
    ) -> "ProjectTables":
        """Load tables using paths relative to a project root."""

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

    def prepare_inputs(
        self,
        dataset: Mapping[str, Any],
        variable: Variable,
    ) -> tuple[dict[str, Any], Variable]:
        """Validate dataset and variable input, returning normalized copies."""

        normalized_dataset = dict(dataset)
        self._add_cv_defaults(normalized_dataset)
        variable_entry = variable.resolve_table_entry(self)
        self._add_table_header_defaults(normalized_dataset, variable_entry)
        normalized_variable = variable.merge_table_entry(variable_entry)
        self._add_variable_global_defaults(normalized_dataset, normalized_variable)
        self._add_source_defaults(normalized_dataset)
        self._add_institution_default(normalized_dataset)
        self._add_experiment_defaults(normalized_dataset)
        self._add_license_text(normalized_dataset)
        self._add_runtime_global_defaults(normalized_dataset)
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
        return normalized_dataset, normalized_variable

    def _add_cv_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill scalar CV defaults and derived license text when available."""

        for key, value in self.cv.items():
            if key not in dataset and _is_scalar_cv_default(value):
                dataset[key] = value

    def _add_table_header_defaults(
        self, dataset: dict[str, Any], variable_entry: VariableEntry
    ) -> None:
        """Fill global defaults available from the loaded variable table header."""

        header = variable_entry.table_header or {}
        for key in ("Conventions", "data_specs_version"):
            value = header.get(key)
            if _is_table_value(value):
                dataset.setdefault(key, value)

    def _add_variable_global_defaults(
        self, dataset: dict[str, Any], variable: Variable
    ) -> None:
        """Fill global attributes that are derived from the variable table."""

        variable_id = str(variable.get("id") or variable.get("variable_id"))
        branded_name = str(variable.get("name") or variable_id)
        dataset.setdefault("variable_id", variable_id)
        dataset.setdefault("branded_variable", branded_name)
        if "_" in branded_name:
            suffix = branded_name.split("_", 1)[1]
            dataset.setdefault("branding_suffix", suffix)
            for key, value in zip(
                (
                    "temporal_label",
                    "vertical_label",
                    "horizontal_label",
                    "area_label",
                ),
                suffix.split("-"),
            ):
                dataset.setdefault(key, value)
        for key in ("frequency", "realm", "table_id"):
            value = variable.get(key)
            if _is_table_value(value):
                dataset.setdefault(key, _single_or_original(value))

    def _add_source_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill and validate attributes supplied by a source_id CV entry."""

        source_entries = self.cv.get("source_id")
        source_id = dataset.get("source_id")
        if not isinstance(source_entries, Mapping) or source_id in (None, ""):
            return
        source_entry = source_entries.get(str(source_id))
        if not isinstance(source_entry, Mapping):
            return
        for key, value in source_entry.items():
            if key == "source_id" or not _is_table_value(value):
                continue
            default = _single_cv_default(value)
            if default is not None:
                dataset.setdefault(key, default)

    def _add_institution_default(self, dataset: dict[str, Any]) -> None:
        """Fill institution text from institution_id when the CV provides it."""

        if "institution" in dataset:
            return
        institution_entries = self.cv.get("institution_id")
        institution_id = dataset.get("institution_id")
        if not isinstance(institution_entries, Mapping) or institution_id in (
            None,
            "",
        ):
            return
        institution = institution_entries.get(str(institution_id))
        if isinstance(institution, str) and institution:
            dataset["institution"] = institution

    def _add_experiment_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill scalar attributes supplied by an experiment_id CV entry."""

        experiment_entry = self._experiment_entry(dataset)
        if experiment_entry is None:
            return
        for key, value in experiment_entry.items():
            if key in {
                "additional_allowed_model_components",
                "description",
                "parent_activity_id",
                "parent_experiment_id",
                "required_source_type",
                "source_type",
            }:
                continue
            default = _single_cv_default(value)
            if default is not None:
                dataset.setdefault(key, default)

    def _add_runtime_global_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill required globals that CMOR normally creates while writing."""

        required = self.required_global_attributes()
        if "Conventions" in required:
            dataset.setdefault("Conventions", self._default_conventions())
        if "creation_date" in required:
            dataset.setdefault(
                "creation_date", datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        if "variant_label" in required:
            variant_label = _variant_label(dataset)
            if variant_label:
                dataset.setdefault("variant_label", variant_label)
        if "tracking_id" in required and "tracking_id" not in dataset:
            dataset["tracking_id"] = _new_tracking_id(dataset, self.cv)
        for key in required:
            if key in dataset:
                continue
            default = _single_required_default(self._cv_definition_for(key))
            if default is not None:
                dataset[key] = default

    def _add_license_text(self, dataset: dict[str, Any]) -> None:
        license_cv = self.cv.get("license")
        if "license" in dataset or not isinstance(license_cv, Mapping):
            return
        license_id = dataset.get("license_id")
        if license_id in (None, ""):
            return
        license_entries = license_cv.get("license_id")
        license_template = license_cv.get("license_template")
        if not isinstance(license_entries, Mapping) or not isinstance(
            license_template, str
        ):
            return
        license_info = license_entries.get(str(license_id))
        if not isinstance(license_info, Mapping):
            return
        tokens = {
            **{str(key): value for key, value in dataset.items()},
            **{str(key): value for key, value in license_info.items()},
        }
        dataset["license"] = _render_template(license_template, tokens)

    def prepare_axes(
        self,
        axes: Sequence[Axis],
        variable: Variable | None = None,
    ) -> tuple[Axis, ...]:
        """Merge coordinate-axis metadata from the loaded coordinate table."""

        merged_axes = [axis.merge_table_entry(self) for axis in axes]
        if variable is not None:
            merged_axes.extend(
                Axis.missing_scalar_axes(self, merged_axes, variable)
            )
        return tuple(merged_axes)

    def prepare_zfactors(
        self, zfactors: Sequence[ZFactor] | None
    ) -> tuple[ZFactor, ...] | None:
        """Merge z-factor metadata from the loaded formula-term table."""

        if zfactors is None:
            return None
        return tuple(zfactor.merge_table_entry(self) for zfactor in zfactors)

    def prepare_grid(
        self, grid: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        """Merge grid-mapping metadata from the loaded grids table."""

        if grid is None:
            return None
        merged = dict(grid)
        entry_name, entry = self.resolve_grid_mapping(grid)
        if entry is None:
            return merged
        merged.setdefault("table_entry", entry_name)
        coordinates = entry.get("coordinates")
        if "coordinates" not in merged and _is_table_value(coordinates):
            merged["coordinates"] = str(coordinates).split()
        params = dict(merged.get("params", {}))
        for key, value in entry.items():
            if not key.startswith("parameter") or not _is_table_value(value):
                continue
            params.setdefault(str(value), grid.get(str(value), 0.0))
        if params:
            merged["params"] = params
        return merged

    def resolve_grid_mapping(
        self, grid: Mapping[str, Any]
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve a grid mapping entry from a user grid definition."""

        requested = str(
            grid.get("table_entry")
            or grid.get("mapping_entry")
            or grid.get("name")
            or ""
        )
        if requested in self.grid_mapping_entries:
            return requested, self.grid_mapping_entries[requested]
        return None, None

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None:
        """Validate user-supplied controlled values against the project CV."""

        for key, value in dataset.items():
            if _is_internal_dataset_key(key):
                continue
            allowed = self._cv_definition_for(str(key))
            if allowed is not None and not self._value_allowed(
                str(key), value, allowed, dataset
            ):
                raise TableValidationError(
                    f"{key}={value!r} is not allowed by {self.cv_file.name}."
                )

        self.validate_required_global_attributes(dataset)

    def validate_required_global_attributes(
        self, dataset: Mapping[str, Any]
    ) -> None:
        """Require every CV-listed global attribute that CMOR4 can write."""

        missing = [
            name
            for name in self.required_global_attributes()
            if name not in dataset or dataset.get(name) in (None, "")
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise TableValidationError(
                "Required global attributes are missing: " f"{missing_text}."
            )

    def required_global_attributes(self) -> tuple[str, ...]:
        required = self.cv.get("required_global_attributes", ())
        if not isinstance(required, Sequence) or isinstance(required, str):
            return ()
        return tuple(str(value) for value in required)

    def validate_experiment(self, dataset: Mapping[str, Any]) -> None:
        """Validate experiment-specific CV attributes."""

        experiment_entry = self._experiment_entry(dataset)
        if experiment_entry is None:
            return
        self.validate_source_type(dataset, experiment_entry)
        for key, expected in experiment_entry.items():
            if key in {
                "additional_allowed_model_components",
                "description",
                "parent_activity_id",
                "parent_experiment_id",
                "required_source_type",
                "source_type",
            }:
                continue
            if not _is_table_value(expected) or key not in dataset:
                continue
            if not _metadata_value_matches(dataset[key], expected):
                raise TableValidationError(
                    f"{key}={dataset[key]!r} does not match "
                    f"experiment_id={dataset.get('experiment_id')!r} "
                    f"CV value {expected!r}."
                )
        expected_activity = experiment_entry.get("activity_id")
        if _is_table_value(expected_activity) and "activity_id" in dataset:
            if not _metadata_value_matches(dataset["activity_id"], expected_activity):
                raise TableValidationError(
                    f"activity_id={dataset['activity_id']!r} does not match "
                    f"experiment_id={dataset.get('experiment_id')!r} "
                    f"CV value {expected_activity!r}."
                )

    def validate_source_type(
        self,
        dataset: Mapping[str, Any],
        experiment_entry: Mapping[str, Any],
    ) -> None:
        """Validate experiment-specific required source_type tokens."""

        required = _cv_values(experiment_entry.get("required_source_type"))
        additional = _cv_values(
            experiment_entry.get("additional_allowed_model_components")
        )
        if not required and not additional:
            return
        source_type = dataset.get("source_type")
        if source_type in (None, ""):
            raise TableValidationError("source_type is required.")
        source_type_text = str(source_type)
        tokens = source_type_text.split()
        for expected in required:
            if not _source_type_pattern_matches(source_type_text, expected):
                raise TableValidationError(
                    f"source_type={source_type!r} is missing required "
                    f"source type {expected!r}."
                )
        allowed = (*required, *additional)
        for token in tokens:
            if not any(_source_type_pattern_matches(token, item) for item in allowed):
                raise TableValidationError(
                    f"source_type={source_type!r} contains source type "
                    f"{token!r} that is not allowed by experiment_id="
                    f"{dataset.get('experiment_id')!r}."
                )

    def validate_source_attributes(self, dataset: Mapping[str, Any]) -> None:
        """Validate source_id-specific CV attributes."""

        source_entries = self.cv.get("source_id")
        source_id = dataset.get("source_id")
        if not isinstance(source_entries, Mapping) or source_id in (None, ""):
            return
        source_entry = source_entries.get(str(source_id))
        if not isinstance(source_entry, Mapping):
            return
        for key, expected in source_entry.items():
            if key == "source_id" or key not in dataset:
                continue
            if _is_table_value(expected) and not _metadata_value_matches(
                dataset[key], expected
            ):
                raise TableValidationError(
                    f"{key}={dataset[key]!r} does not match "
                    f"source_id={source_id!r} CV value {expected!r}."
                )

    def validate_parent_attributes(self, dataset: Mapping[str, Any]) -> None:
        """Validate CMIP-style parent experiment attributes."""

        experiment_entry = self._experiment_entry(dataset)
        if experiment_entry is None:
            return
        expected_parent_experiments = _cv_values(
            experiment_entry.get("parent_experiment_id")
        )
        parent_attrs = (
            "parent_activity_id",
            "parent_mip_era",
            "parent_source_id",
            "parent_time_units",
            "parent_variant_label",
            "branch_time_in_child",
            "branch_time_in_parent",
        )
        if not expected_parent_experiments:
            if "parent_experiment_id" in dataset:
                raise TableValidationError(
                    f"experiment_id={dataset.get('experiment_id')!r} does not "
                    "allow parent_experiment_id."
                )
            unexpected = [name for name in parent_attrs if name in dataset]
            if unexpected:
                raise TableValidationError(
                    f"experiment_id={dataset.get('experiment_id')!r} does not "
                    "allow parent attributes: " + ", ".join(unexpected) + "."
                )
            return

        parent_experiment_id = dataset.get("parent_experiment_id")
        if parent_experiment_id in (None, ""):
            raise TableValidationError(
                f"experiment_id={dataset.get('experiment_id')!r} requires "
                "parent_experiment_id."
            )
        if str(parent_experiment_id) not in {
            str(value) for value in expected_parent_experiments
        }:
            raise TableValidationError(
                f"parent_experiment_id={parent_experiment_id!r} does not match "
                f"experiment_id={dataset.get('experiment_id')!r} CV values "
                f"{expected_parent_experiments!r}."
            )
        self._validate_required_parent_value(
            dataset,
            "parent_activity_id",
            experiment_entry.get("parent_activity_id"),
        )
        parent_source_id = dataset.get("parent_source_id")
        if parent_source_id in (None, ""):
            raise TableValidationError("parent_source_id is required.")
        source_entries = self.cv.get("source_id")
        if (
            isinstance(source_entries, Mapping)
            and str(parent_source_id) not in source_entries
        ):
            raise TableValidationError(
                f"parent_source_id={parent_source_id!r} is not in the CV."
            )
        expected_parent_mip_era = str(dataset.get("mip_era") or "")
        if expected_parent_mip_era and dataset.get("parent_mip_era") not in (
            expected_parent_mip_era,
            None,
            "",
        ):
            raise TableValidationError(
                f"parent_mip_era={dataset.get('parent_mip_era')!r} does not "
                f"match {expected_parent_mip_era!r}."
            )
        for key in ("parent_mip_era", "parent_time_units", "parent_variant_label"):
            if dataset.get(key) in (None, ""):
                raise TableValidationError(f"{key} is required.")
        if not re.fullmatch(
            r"days\s+since\s+\d{4}-\d{1,2}-\d{1,2}.*",
            str(dataset["parent_time_units"]),
        ):
            raise TableValidationError(
                f"parent_time_units={dataset['parent_time_units']!r} is invalid."
            )
        if not re.fullmatch(
            r"r\d+i\d+p\d+f\d+", str(dataset["parent_variant_label"])
        ):
            raise TableValidationError(
                f"parent_variant_label={dataset['parent_variant_label']!r} "
                "is invalid."
            )
        for key in ("branch_time_in_child", "branch_time_in_parent"):
            if key not in dataset:
                raise TableValidationError(f"{key} is required.")
            try:
                float(dataset[key])
            except (TypeError, ValueError) as exc:
                raise TableValidationError(
                    f"{key}={dataset[key]!r} must be numeric."
                ) from exc

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
        axis_entries = data.get("axis_entry", {})
        if isinstance(axis_entries, Mapping):
            for name, entry in axis_entries.items():
                if isinstance(entry, Mapping):
                    self._variable_axis_entries[str(name)] = entry

    def _value_allowed(
        self,
        key: str,
        value: Any,
        allowed: Any,
        dataset: Mapping[str, Any],
    ) -> bool:
        if key == "license" and _is_license_template_cv(allowed):
            return True
        if isinstance(allowed, str) and str(value) == allowed:
            return True
        if isinstance(allowed, str) and "<" in allowed and ">" in allowed:
            return _template_value_matches(str(value), allowed, dataset)
        if key in {"license_url", "license_type"}:
            license_info = self._license_info(dataset)
            if license_info is not None and key in license_info:
                return _metadata_value_matches(value, license_info[key])
            return True
        if isinstance(allowed, Mapping):
            if key in {"realm", "source_type"}:
                return all(
                    token in allowed for token in str(value).split() if token
                )
            return str(value) in allowed
        if isinstance(allowed, str):
            return str(value) == allowed
        if isinstance(allowed, list):
            return any(
                _allowed_list_item(str(value), item) for item in allowed
            )
        return True

    def _cv_definition_for(self, key: str) -> Any:
        if key in self.cv:
            return self.cv[key]
        license_cv = self.cv.get("license")
        if isinstance(license_cv, Mapping) and key == "license_id":
            return license_cv.get("license_id")
        return None

    def _license_info(self, dataset: Mapping[str, Any]) -> Mapping[str, Any] | None:
        license_cv = self.cv.get("license")
        if not isinstance(license_cv, Mapping):
            return None
        license_entries = license_cv.get("license_id")
        if not isinstance(license_entries, Mapping):
            return None
        license_id = dataset.get("license_id")
        if license_id in (None, ""):
            return None
        license_info = license_entries.get(str(license_id))
        return license_info if isinstance(license_info, Mapping) else None

    def _experiment_entry(
        self, dataset: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        experiment_entries = self.cv.get("experiment_id")
        experiment_id = dataset.get("experiment_id")
        if not isinstance(experiment_entries, Mapping) or experiment_id in (
            None,
            "",
        ):
            return None
        entry = experiment_entries.get(str(experiment_id))
        return entry if isinstance(entry, Mapping) else None

    def _default_conventions(self) -> str:
        conventions = self.cv.get("Conventions")
        if isinstance(conventions, list) and conventions:
            return str(conventions[0])
        if isinstance(conventions, str) and conventions:
            return conventions
        return "CF-1.11"

    def _validate_required_parent_value(
        self,
        dataset: Mapping[str, Any],
        key: str,
        expected: Any,
    ) -> None:
        value = dataset.get(key)
        if value in (None, ""):
            raise TableValidationError(f"{key} is required.")
        if _is_table_value(expected) and not _metadata_value_matches(value, expected):
            raise TableValidationError(
                f"{key}={value!r} does not match experiment_id="
                f"{dataset.get('experiment_id')!r} CV value {expected!r}."
            )

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


def _is_internal_dataset_key(key: str) -> bool:
    return key.startswith("_") or key in {
        "outpath",
        "output_file_template",
        "output_path_template",
    }


def _is_scalar_cv_default(value: Any) -> bool:
    if isinstance(value, str) and ("<" in value or ">" in value):
        return False
    return isinstance(value, (str, int, float))


def _single_cv_default(value: Any) -> Any:
    if not _is_table_value(value):
        return None
    if isinstance(value, Mapping):
        return None
    if isinstance(value, list):
        if len(value) != 1:
            return None
        return value[0]
    return value


def _single_required_default(value: Any) -> Any:
    if not _is_table_value(value):
        return None
    if isinstance(value, Mapping):
        keys = list(value)
        return keys[0] if len(keys) == 1 else None
    return _single_cv_default(value)


def _cv_values(value: Any) -> tuple[Any, ...]:
    if not _is_table_value(value):
        return ()
    if isinstance(value, list):
        return tuple(item for item in value if _is_table_value(item))
    return (value,)


def _is_license_template_cv(value: Any) -> bool:
    return isinstance(value, Mapping) and isinstance(
        value.get("license_template"), str
    )


def _variant_label(dataset: Mapping[str, Any]) -> str | None:
    if _is_table_value(dataset.get("variant_label")):
        return str(dataset["variant_label"])
    pieces = [
        dataset.get("realization_index"),
        dataset.get("initialization_index"),
        dataset.get("physics_index"),
        dataset.get("forcing_index"),
    ]
    if not all(_is_table_value(piece) for piece in pieces):
        return None
    return "".join(str(piece) for piece in pieces)


def _new_tracking_id(dataset: Mapping[str, Any], cv: Mapping[str, Any]) -> str:
    identifier = str(uuid.uuid4())
    prefix = dataset.get("tracking_prefix")
    if prefix in (None, ""):
        tracking_prefix = cv.get("tracking_id_prefix")
        if isinstance(tracking_prefix, list) and len(tracking_prefix) == 1:
            prefix = tracking_prefix[0]
        elif isinstance(tracking_prefix, str):
            prefix = tracking_prefix
    return f"{prefix}/{identifier}" if prefix not in (None, "") else identifier


def _render_template(template: str, tokens: Mapping[str, Any]) -> str:
    return re.sub(
        r"<([^>]+)>",
        lambda match: str(tokens.get(match.group(1), "")),
        template,
    )


def _template_value_matches(
    value: str, template: str, tokens: Mapping[str, Any]
) -> bool:
    rendered = _render_template(template, tokens)
    if value == rendered:
        return True
    token_names = re.findall(r"<([^>]+)>", template)
    if token_names:
        joined = "-".join(
            str(tokens.get(name, "")) for name in token_names if tokens.get(name, "")
        )
        if value == joined:
            return True
    return False


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
        .replace("[[:space:]]", r"\s")
        .replace("\\{", "{")
        .replace("\\}", "}")
    )


def _normalize_table_id(value: Any) -> str:
    text = str(value)
    if text.startswith("Table "):
        return text.removeprefix("Table ")
    return text


def _source_type_pattern_matches(value: str, pattern: Any) -> bool:
    pattern_text = _posix_regex_to_python(str(pattern))
    try:
        return re.search(pattern_text, value) is not None
    except re.error:
        return value == str(pattern)
