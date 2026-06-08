from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import uuid

from ._table_utils import (
    is_table_value as _is_table_value,
    metadata_value_matches as _metadata_value_matches,
)
from ._templates import render_template as _render_template
from .exceptions import ControlledVocabularyError


class ControlledVocabulary(Mapping[str, Any]):
    """Project controlled vocabulary with defaulting and validation helpers.

    Parameters
    ----------
    data:
        Controlled-vocabulary data, either as a raw CV mapping or a mapping
        containing a top-level ``CV`` key.
    path:
        Path to the source CV file, if loaded from disk.
    """

    def __init__(
        self, data: Mapping[str, Any], path: str | Path | None = None
    ):
        self.path = Path(path) if path is not None else None
        self._data = dict(data.get("CV", data))

    @classmethod
    def from_file(cls, path: str | Path) -> "ControlledVocabulary":
        """Load a controlled vocabulary from a JSON file.

        Parameters
        ----------
        path:
            Path to the CV JSON file.

        Returns
        -------
        ControlledVocabulary
            Loaded controlled-vocabulary helper.
        """

        cv_path = Path(path)
        with cv_path.open() as handle:
            data = json.load(handle)
        return cls(data, path=cv_path)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def filename(self) -> str:
        """Return the display name for this controlled vocabulary.

        Returns
        -------
        str
            Source filename when known, otherwise ``CV``.
        """

        return self.path.name if self.path is not None else "CV"

    def get_dataset_info(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """Get dataset info with CV defaults.

        Parameters
        ----------
        dataset:
            User-provided dataset metadata.

        Returns
        -------
        dict[str, Any]
            Dataset metadata with controlled-vocabulary defaults applied.
        """
        normalized_dataset = dict(dataset)
        self._add_scalar_defaults(normalized_dataset)
        self._add_source_defaults(normalized_dataset)
        self._add_institution_default(normalized_dataset)
        self._add_experiment_defaults(normalized_dataset)
        self._add_license_text(normalized_dataset)
        self._add_runtime_global_defaults(normalized_dataset)

        return normalized_dataset

    def _add_scalar_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill scalar CV defaults that are not templated."""

        for key, value in self.items():
            templated = isinstance(value, str) and (
                "<" in value or ">" in value
            )
            if key not in dataset and not templated and isinstance(
                value, (str, int, float)
            ):
                dataset[key] = value

    def _add_source_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill attributes supplied by a source_id CV entry."""

        source_entries = self.get("source_id")
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
        """Fill institution text from institution_id."""

        if "institution" in dataset:
            return
        institution_entries = self.get("institution_id")
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

        experiment_entry = self.experiment_entry(dataset)
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
            conventions = self.get("Conventions")
            if isinstance(conventions, list) and conventions:
                default_conventions = str(conventions[0])
            elif isinstance(conventions, str) and conventions:
                default_conventions = conventions
            else:
                default_conventions = "CF-1.11"
            dataset.setdefault("Conventions", default_conventions)
        if "creation_date" in required:
            dataset.setdefault(
                "creation_date", datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        if "variant_label" in required:
            variant_label = _variant_label(dataset)
            if variant_label:
                dataset.setdefault("variant_label", variant_label)
        if "tracking_id" in required and "tracking_id" not in dataset:
            dataset["tracking_id"] = _new_tracking_id(dataset, self)
        for key in required:
            if key in dataset:
                continue
            value = self.definition_for(key)
            if not _is_table_value(value):
                default = None
            elif isinstance(value, Mapping):
                keys = list(value)
                default = keys[0] if len(keys) == 1 else None
            else:
                default = _single_cv_default(value)
            if default is not None:
                dataset[key] = default

    def _add_license_text(self, dataset: dict[str, Any]) -> None:
        license_cv = self.get("license")
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

        self.validate_dataset_values(dataset)
        self.validate_required_global_attributes(dataset)

    def validate_dataset_values(self, dataset: Mapping[str, Any]) -> None:
        """Validate controlled values without requiring every global attr.

        Parameters
        ----------
        dataset:
            Dataset metadata to validate.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if a controlled value is not
            allowed.
        """

        for key, value in dataset.items():
            if key.startswith("_") or key in {
                "outpath",
                "output_file_template",
                "output_path_template",
            }:
                continue
            allowed = self.definition_for(str(key))
            if allowed is not None and not self.value_allowed(
                str(key), value, allowed, dataset
            ):
                raise ControlledVocabularyError(
                    f"{key}={value!r} is not allowed by {self.filename}."
                )

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

        missing = [
            name
            for name in self.required_global_attributes()
            if name not in dataset or dataset.get(name) in (None, "")
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise ControlledVocabularyError(
                "Required global attributes are missing: " f"{missing_text}."
            )

    def required_global_attributes(self) -> tuple[str, ...]:
        """Return CV-listed required global attributes.

        Returns
        -------
        tuple[str, ...]
            Required global attribute names.
        """

        required = self.get("required_global_attributes", ())
        if not isinstance(required, Sequence) or isinstance(required, str):
            return ()
        return tuple(str(value) for value in required)

    def validate_experiment(self, dataset: Mapping[str, Any]) -> None:
        """Validate experiment-specific CV attributes.

        Parameters
        ----------
        dataset:
            Dataset metadata containing an ``experiment_id``.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if experiment-specific values
            are inconsistent.
        """

        experiment_entry = self.experiment_entry(dataset)
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
                raise ControlledVocabularyError(
                    f"{key}={dataset[key]!r} does not match "
                    f"experiment_id={dataset.get('experiment_id')!r} "
                    f"CV value {expected!r}."
                )
        expected_activity = experiment_entry.get("activity_id")
        if _is_table_value(expected_activity) and "activity_id" in dataset:
            if not _metadata_value_matches(
                dataset["activity_id"], expected_activity
            ):
                raise ControlledVocabularyError(
                    f"activity_id={dataset['activity_id']!r} does not match "
                    f"experiment_id={dataset.get('experiment_id')!r} "
                    f"CV value {expected_activity!r}."
                )

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

        required = _cv_values(experiment_entry.get("required_source_type"))
        additional = _cv_values(
            experiment_entry.get("additional_allowed_model_components")
        )
        if not required and not additional:
            return
        source_type = dataset.get("source_type")
        if source_type in (None, ""):
            raise ControlledVocabularyError("source_type is required.")
        source_type_text = str(source_type)
        tokens = source_type_text.split()
        for expected in required:
            if not _source_type_pattern_matches(source_type_text, expected):
                raise ControlledVocabularyError(
                    f"source_type={source_type!r} is missing required "
                    f"source type {expected!r}."
                )
        allowed = (*required, *additional)
        for token in tokens:
            if not any(
                _source_type_pattern_matches(token, item)
                for item in allowed
            ):
                raise ControlledVocabularyError(
                    f"source_type={source_type!r} contains source type "
                    f"{token!r} that is not allowed by experiment_id="
                    f"{dataset.get('experiment_id')!r}."
                )

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

        source_entries = self.get("source_id")
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
                raise ControlledVocabularyError(
                    f"{key}={dataset[key]!r} does not match "
                    f"source_id={source_id!r} CV value {expected!r}."
                )

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

        experiment_entry = self.experiment_entry(dataset)
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
                raise ControlledVocabularyError(
                    f"experiment_id={dataset.get('experiment_id')!r} does not "
                    "allow parent_experiment_id."
                )
            unexpected = [name for name in parent_attrs if name in dataset]
            if unexpected:
                raise ControlledVocabularyError(
                    f"experiment_id={dataset.get('experiment_id')!r} does not "
                    "allow parent attributes: " + ", ".join(unexpected) + "."
                )
            return

        parent_experiment_id = dataset.get("parent_experiment_id")
        if parent_experiment_id in (None, ""):
            raise ControlledVocabularyError(
                f"experiment_id={dataset.get('experiment_id')!r} requires "
                "parent_experiment_id."
            )
        if str(parent_experiment_id) not in {
            str(value) for value in expected_parent_experiments
        }:
            raise ControlledVocabularyError(
                f"parent_experiment_id={parent_experiment_id!r} "
                "does not match "
                f"experiment_id={dataset.get('experiment_id')!r} CV values "
                f"{expected_parent_experiments!r}."
            )
        self.validate_required_parent_value(
            dataset,
            "parent_activity_id",
            experiment_entry.get("parent_activity_id"),
        )
        parent_source_id = dataset.get("parent_source_id")
        if parent_source_id in (None, ""):
            raise ControlledVocabularyError("parent_source_id is required.")
        source_entries = self.get("source_id")
        if (
            isinstance(source_entries, Mapping)
            and str(parent_source_id) not in source_entries
        ):
            raise ControlledVocabularyError(
                f"parent_source_id={parent_source_id!r} is not in the CV."
            )
        expected_parent_mip_era = str(dataset.get("mip_era") or "")
        if expected_parent_mip_era and dataset.get("parent_mip_era") not in (
            expected_parent_mip_era,
            None,
            "",
        ):
            raise ControlledVocabularyError(
                f"parent_mip_era={dataset.get('parent_mip_era')!r} does not "
                f"match {expected_parent_mip_era!r}."
            )
        for key in (
            "parent_mip_era",
            "parent_time_units",
            "parent_variant_label",
        ):
            if dataset.get(key) in (None, ""):
                raise ControlledVocabularyError(f"{key} is required.")
        if not re.fullmatch(
            r"days\s+since\s+\d{4}-\d{1,2}-\d{1,2}.*",
            str(dataset["parent_time_units"]),
        ):
            raise ControlledVocabularyError(
                f"parent_time_units={dataset['parent_time_units']!r} "
                "is invalid."
            )
        if not re.fullmatch(
            r"r\d+i\d+p\d+f\d+", str(dataset["parent_variant_label"])
        ):
            raise ControlledVocabularyError(
                f"parent_variant_label={dataset['parent_variant_label']!r} "
                "is invalid."
            )
        for key in ("branch_time_in_child", "branch_time_in_parent"):
            if key not in dataset:
                raise ControlledVocabularyError(f"{key} is required.")
            try:
                float(dataset[key])
            except (TypeError, ValueError) as exc:
                raise ControlledVocabularyError(
                    f"{key}={dataset[key]!r} must be numeric."
                ) from exc

    def value_allowed(
        self,
        key: str,
        value: Any,
        allowed: Any,
        dataset: Mapping[str, Any],
    ) -> bool:
        """Return whether a value is allowed by a CV definition.

        Parameters
        ----------
        key:
            Dataset attribute name being validated.
        value:
            Dataset attribute value to check.
        allowed:
            CV definition for the attribute.
        dataset:
            Full dataset metadata, used to resolve templated CV values.

        Returns
        -------
        bool
            ``True`` when the value is accepted by the CV definition.
        """

        if (
            key == "license"
            and isinstance(allowed, Mapping)
            and isinstance(allowed.get("license_template"), str)
        ):
            return True
        if isinstance(allowed, str) and str(value) == allowed:
            return True
        if isinstance(allowed, str) and "<" in allowed and ">" in allowed:
            separator: str | None
            match key:
                case "branding_suffix":
                    separator = "-"
                case _:
                    separator = None
            return value == _render_template(allowed, dataset, separator)
        if key in {"license_url", "license_type"}:
            license_info = None
            license_cv = self.get("license")
            if isinstance(license_cv, Mapping):
                license_entries = license_cv.get("license_id")
                license_id = dataset.get("license_id")
                if isinstance(license_entries, Mapping) and license_id not in (
                    None,
                    "",
                ):
                    candidate = license_entries.get(str(license_id))
                    if isinstance(candidate, Mapping):
                        license_info = candidate
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

    def definition_for(self, key: str) -> Any:
        """Return the CV definition for a dataset attribute.

        Parameters
        ----------
        key:
            Dataset attribute name.

        Returns
        -------
        Any
            CV definition for ``key``, or ``None`` when the key is unknown.
        """

        if key in self:
            return self[key]
        license_cv = self.get("license")
        if isinstance(license_cv, Mapping) and key == "license_id":
            return license_cv.get("license_id")
        return None

    def experiment_entry(
        self, dataset: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        """Return the CV entry for the dataset experiment.

        Parameters
        ----------
        dataset:
            Dataset metadata containing ``experiment_id``.

        Returns
        -------
        Mapping[str, Any] | None
            Matching experiment CV entry, or ``None`` when unavailable.
        """

        experiment_entries = self.get("experiment_id")
        experiment_id = dataset.get("experiment_id")
        if not isinstance(experiment_entries, Mapping) or experiment_id in (
            None,
            "",
        ):
            return None
        entry = experiment_entries.get(str(experiment_id))
        return entry if isinstance(entry, Mapping) else None

    def validate_required_parent_value(
        self,
        dataset: Mapping[str, Any],
        key: str,
        expected: Any,
    ) -> None:
        """Validate one required parent experiment attribute.

        Parameters
        ----------
        dataset:
            Dataset metadata containing the parent attribute.
        key:
            Parent attribute name to validate.
        expected:
            Expected CV value for the parent attribute.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if the value is missing or
            inconsistent with the CV.
        """

        value = dataset.get(key)
        if value in (None, ""):
            raise ControlledVocabularyError(f"{key} is required.")
        if _is_table_value(expected) and not _metadata_value_matches(
            value, expected
        ):
            raise ControlledVocabularyError(
                f"{key}={value!r} does not match experiment_id="
                f"{dataset.get('experiment_id')!r} CV value {expected!r}."
            )


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


def _cv_values(value: Any) -> tuple[Any, ...]:
    if not _is_table_value(value):
        return ()
    if isinstance(value, list):
        return tuple(item for item in value if _is_table_value(item))
    return (value,)


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


def _source_type_pattern_matches(value: str, pattern: Any) -> bool:
    pattern_text = _posix_regex_to_python(str(pattern))
    try:
        return re.search(pattern_text, value) is not None
    except re.error:
        return value == str(pattern)
