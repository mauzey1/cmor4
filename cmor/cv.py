from __future__ import annotations

import json
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from . import _cv_models
from ._metadata import DatasetMetadata
from .exceptions import ControlledVocabularyError


class ControlledVocabulary:
    """Project controlled vocabulary with defaulting and validation helpers.

    Parameters
    ----------
    data : Mapping[str, Any]
        Controlled-vocabulary data, either as a raw CV mapping or a mapping
        containing a top-level ``CV`` key.
    path : str or pathlib.Path, optional
        Path to the source CV file, if loaded from disk.

    Notes
    -----
    Use ``get_dataset_info`` to prepare metadata and ``validate_dataset_info``
    to validate it. The typed CV components are internal implementation details.
    """

    _rules: dict[str, _cv_models.AttributeRule]
    _defaults: _cv_models.DefaultValuesComponent
    _drs: _cv_models.DRSComponent | None
    _institutions: _cv_models.InstitutionComponent | None
    _license: _cv_models.LicenseComponent | None
    _sources: _cv_models.SourceComponent | None
    _experiments: _cv_models.ExperimentComponent | None
    _source_types: _cv_models.SourceTypeComponent
    _forcing: _cv_models.ForcingComponent | None
    _frequency: _cv_models.FrequencyComponent | None
    _tracking_id: _cv_models.TrackingIdComponent
    _variant_label: _cv_models.VariantLabelComponent
    _grid_label: _cv_models.GridLabelComponent
    required_attributes: tuple[str, ...]

    def __init__(self, data: Mapping[str, Any], path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        try:
            parsed = _cv_models.ControlledVocabularyModel.from_mapping(data)
        except ValueError as error:
            message = str(error)
            if self.path is not None:
                message += f" Check {self.filename}."
            raise ControlledVocabularyError(message) from error
        self._rules = parsed.attributes
        self._defaults = parsed.defaults
        self._drs = parsed.drs
        self._institutions = parsed.institutions
        self._license = parsed.license
        self._sources = parsed.sources
        self._experiments = parsed.experiments
        self._source_types = parsed.source_types
        self._forcing = parsed.forcing
        self._frequency = parsed.frequency
        self._tracking_id = parsed.tracking_id
        self._variant_label = parsed.variant_label
        self._grid_label = parsed.grid_label
        self.required_attributes = parsed.required_global_attributes
        self._warn_double_nested_entries()

    def _warn_double_nested_entries(self) -> None:
        """Emit a RuntimeWarning for double-nested CV entries.

        The most common CV authoring mistake is accidentally wrapping an
        attribute's value inside a dict keyed by the attribute name itself,
        for example::

            "nominal_resolution": {
                "nominal_resolution": ["0.5 km", "1 km", …]
            }

        instead of the correct::

            "nominal_resolution": ["0.5 km", "1 km", …]

        When this happens the parsed lookup rule contains the attribute name
        itself as its only allowed key, rather than the intended values.

        This check mirrors the CMOR3 issue reported at
        github.com/PCMDI/cmor/issues/829.
        """

        issues = self._validate_structure()
        for issue in issues:
            warnings.warn(issue, RuntimeWarning, stacklevel=3)

    def _validate_structure(self) -> list[str]:
        """Return a list of structural issues found in this CV.

        Currently detects double-nested entries — those where the value is a
        mapping whose only key is the entry's own name.  When present, the
        expected constraint is buried one level too deep.

        Returns
        -------
        list[str]
            Human-readable issue descriptions.  Empty when no issues are
            found.

        Examples
        --------
        Detect a double-nested entry::

            cv = ControlledVocabulary({
                "CV": {
                    "nominal_resolution": {
                        "nominal_resolution": ["0.5 km", "1 km"]
                    }
                }
            })
            issues = cv._validate_structure()
            # ["CV entry 'nominal_resolution' appears to be double-nested …"]
        """

        issues: list[str] = []
        for key, rule in self._rules.items():
            if rule.is_double_nested:
                issues.append(
                    f"CV entry {key!r} in {self.filename} appears to be "
                    f"double-nested: its value is a mapping containing only "
                    f"the key {key!r}. This is usually a CV authoring mistake "
                    f"(see github.com/PCMDI/cmor/issues/829). "
                    f"Controlled vocabulary validation for {key!r} may "
                    f"silently pass any value."
                )
        return issues

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

    @property
    def filename(self) -> str:
        """Return the display name for this controlled vocabulary.

        Returns
        -------
        str
            Source filename when known, otherwise ``CV``.
        """

        return self.path.name if self.path is not None else "CV"

    def get_dataset_info(self, dataset: DatasetMetadata) -> DatasetMetadata:
        """Get dataset info with CV defaults.

        Parameters
        ----------
        dataset:
            Dataset metadata to normalize.

        Returns
        -------
        DatasetMetadata
            Dataset metadata with controlled-vocabulary defaults applied.
        """
        normalized_dataset = dataset.to_dict()
        normalized_dataset.update(
            self._defaults.scalar_defaults_for(normalized_dataset)
        )
        self._add_source_defaults(normalized_dataset)
        self._add_institution_default(normalized_dataset)
        self._add_experiment_defaults(normalized_dataset)
        self._add_license_text(normalized_dataset)
        self._add_runtime_global_defaults(normalized_dataset)
        # Run after all dedicated handlers so their setdefault values win over
        # anything this generic handler would inject.
        normalized_dataset.update(
            self._defaults.nested_defaults_for(normalized_dataset)
        )

        return DatasetMetadata.from_mapping(normalized_dataset)

    def _add_source_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill attributes supplied by a source_id CV entry."""

        if self._sources is None:
            return
        for key, value in self._sources.defaults_for(dataset.get("source_id")).items():
            if key not in dataset or dataset[key] in (None, ""):
                dataset[key] = value

    def _add_institution_default(self, dataset: dict[str, Any]) -> None:
        """Fill institution text from institution_id."""

        if self._institutions is None:
            return
        institution = self._institutions.name_for(dataset.get("institution_id"))
        if institution is not None and dataset.get("institution") in (None, ""):
            dataset["institution"] = institution

    def _add_experiment_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill scalar attributes supplied by an experiment_id CV entry."""

        if self._experiments is None:
            return
        for key, value in self._experiments.defaults_for(
            dataset.get("experiment_id")
        ).items():
            if key not in dataset or dataset[key] in (None, ""):
                dataset[key] = value

    def _add_runtime_global_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill required globals that CMOR normally creates while writing."""

        required = self.required_attributes
        if "Conventions" in required:
            dataset.setdefault("Conventions", self._defaults.conventions)
        if "creation_date" in required:
            dataset.setdefault(
                "creation_date", datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        if "variant_label" in required:
            variant_label = self._variant_label.render(dataset)
            if variant_label:
                dataset.setdefault("variant_label", variant_label)
        if "tracking_id" in required and "tracking_id" not in dataset:
            dataset["tracking_id"] = self._tracking_id.format(
                str(uuid.uuid4()), dataset.get("tracking_prefix")
            )
        for key in required:
            if key in dataset:
                continue
            default = self._defaults.required.get(key)
            if default is not None:
                dataset[key] = default

    def _add_license_text(self, dataset: dict[str, Any]) -> None:
        if "license" in dataset or self._license is None:
            return
        rendered = self._license.render(dataset)
        if rendered is not None:
            dataset["license"] = rendered

    def validate_dataset_info(
        self, dataset: DatasetMetadata, *, check_required: bool = True
    ) -> None:
        """Run all dataset metadata checks against the project CV.

        This includes controlled values, grid and variant labels, forcing,
        derived attributes, source and experiment constraints, and parent
        metadata. Experiment validation also checks source-type composition.

        Parameters
        ----------
        dataset:
            Dataset metadata to validate.
        check_required:
            Whether to require every CV-listed global attribute. Set to
            ``False`` during initial preparation, before variable-derived
            attributes are available. All other checks still run, including
            experiment-specific requirements for source and parent metadata.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if validation fails.
        """

        values = dataset.to_dict()
        self._validate_dataset_values(values)
        if check_required:
            self._validate_required_global_attributes(values)
            self._validate_derived_attributes(values)
        self._validate_variant_indices(values)
        self._validate_forcing_terms(values)
        if not check_required:
            # Preserve the initial-preparation order: indices and forcing
            # errors precede derived-attribute mismatches.
            self._validate_derived_attributes(values)
        self._validate_source_attributes(values)
        self._validate_experiment(values)
        self._validate_parent_attributes(values)

    def _validate_forcing_terms(self, dataset: Mapping[str, Any]) -> None:
        """Validate the ``forcing`` global attribute against the CV forcing list.

        The ``forcing`` attribute (used in CMIP6-style projects) is a
        space- or comma-separated string of abbreviations describing the
        applied forcings, optionally followed by a parenthetical annotation,
        for example::

            "GHG Oz SA Sl Vl BC OC (GHG = CO2, N2O, CH4, …)"

        Tokenisation mirrors CMOR3's ``cmor_check_forcing_validity``:

        1. Commas are replaced with spaces.
        2. Everything from the first ``(`` onwards is discarded (annotation
           truncation, not just removal of the parenthetical content).
        3. The remaining text is split on whitespace.
        4. Each non-empty token is looked up in the CV's ``forcing``
           enumeration (list or mapping keys).

        Validation is skipped when the CV does not define a ``forcing`` key
        (e.g. CMIP7, obs4MIPs), when the CV's ``forcing`` value is not a
        list or mapping, or when the dataset has no ``forcing`` attribute.

        Parameters
        ----------
        dataset:
            Dataset metadata that may contain a ``forcing`` attribute.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if any token is not in
            the CV's forcing enumeration.
        """

        if self._forcing is None:
            return
        invalid = self._forcing.invalid_token(dataset.get("forcing"))
        if invalid is not None:
            raise ControlledVocabularyError(
                f"forcing term {invalid!r} is not valid. "
                f"Valid values are: {sorted(self._forcing.values)!r}. "
                f"Check {self.filename}."
            )

    def _validate_variant_indices(self, dataset: Mapping[str, Any]) -> None:
        """Validate variant index integers and the derived variant_label format.

        Each of ``realization_index``, ``initialization_index``,
        ``physics_index``, and ``forcing_index`` must be a positive integer
        no greater than INT32_MAX (2 147 483 647) when present.

        The ``variant_label`` format (``r<N>i<N>p<N>f<N>``) is only validated
        when the CV defines a list of allowed values or patterns for
        ``variant_label``.  When the CV merely *requires* the attribute without
        constraining its values (e.g. obs4MIPs), any user-supplied string is
        accepted — matching CMOR3's behaviour.

        Parameters
        ----------
        dataset:
            Dataset metadata to validate.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if any index value is not a
            positive integer, exceeds INT32_MAX, or if the derived
            variant_label does not match the CV-defined pattern.
        """

        error = self._variant_label.validation_error(dataset)
        if error is not None:
            raise ControlledVocabularyError(error)

    def _validate_dataset_values(self, dataset: Mapping[str, Any]) -> None:
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
                # The 'forcing' attribute is a space/comma-separated list of
                # abbreviations (e.g. "GHG Oz SA").  It cannot be validated
                # as a single value against the CV; _validate_forcing_terms
                # handles it with per-token checks instead.
                "forcing",
            }:
                continue
            rule = self._rules.get(str(key))
            # CMIP6 CVs express several constraints as POSIX BRE regex arrays
            # (single-element lists whose entry contains BRE metacharacters like
            # \{, [[:digit:]], or ^ anchors).  CMOR uses Python re which has
            # different syntax; rather than attempt a partial BRE-to-ERE
            # conversion that may not be reliable, skip validation for any
            # attribute whose CV definition is a list of strings that look
            # like BRE patterns.  The same attributes are validated implicitly
            # by other checks (variant_label format, grid_label enumeration,
            # etc.) that use the CV's enumeration entries.
            if rule is not None and rule.skips_generic_validation:
                continue
            if self.controls(str(key)) and not self._value_allowed(
                str(key), value, dataset
            ):
                raise ControlledVocabularyError(
                    f"{key}={value!r} is not allowed by {self.filename}."
                )

        error = self._grid_label.validation_error(dataset)
        if error is not None:
            raise ControlledVocabularyError(error)

    def _validate_required_global_attributes(self, dataset: Mapping[str, Any]) -> None:
        """Require every CV-listed global attribute that CMOR can write.

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

        dataset_values = dataset
        # Attributes whose CV definition is a POSIX BRE regex array cannot be
        # auto-generated by CMOR.  For CMIP6 this includes 'license' (among
        # others).  Rather than requiring the user to supply these manually in
        # every test context, we waive the enforcement when the value is absent
        # and the CV cannot supply a concrete default — mirroring CMOR3's
        # lenient behaviour for test models like PCMDI-test-1-0.
        bre_required = {
            name
            for name in self.required_attributes
            if (rule := self._rules.get(name)) is not None
            and rule.skips_generic_validation
        }

        missing = [
            name
            for name in self.required_attributes
            if name not in dataset_values or dataset_values.get(name) in (None, "")
            if name not in bre_required
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise ControlledVocabularyError(
                f"Required global attributes are missing: {missing_text}."
            )

    def _validate_experiment(self, dataset: Mapping[str, Any]) -> None:
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

        if self._experiments is None:
            return
        ignored_keys = set()
        if "experiment" not in self.required_attributes:
            ignored_keys.add("experiment")
        error = self._experiments.validation_error(
            dataset.get("experiment_id"),
            dataset,
            self._source_types,
            ignored_keys=ignored_keys,
        )
        if error is not None:
            raise ControlledVocabularyError(error)

    def _validate_derived_attributes(self, dataset: Mapping[str, Any]) -> None:
        """Validate required derived text against its selected ID entry."""

        values = dataset
        expected: dict[str, Any] = {}
        if self._sources is not None:
            source = self._sources.defaults_for(dataset.get("source_id")).get("source")
            if source is not None:
                expected["source"] = source
        if self._experiments is not None:
            experiment = self._experiments.defaults_for(
                dataset.get("experiment_id")
            ).get("experiment")
            if experiment is not None:
                expected["experiment"] = experiment
        if self._institutions is not None:
            institution = self._institutions.name_for(dataset.get("institution_id"))
            if institution is not None:
                expected["institution"] = institution

        for key, expected_value in expected.items():
            if key not in self.required_attributes or key not in values:
                continue
            actual = values[key]
            if actual in (None, "") or str(actual) == str(expected_value):
                continue
            raise ControlledVocabularyError(
                f"{key}={actual!r} does not match the value selected by "
                f"{key}_id CV entry {expected_value!r}."
            )

    def _validate_source_type(
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

        error = self._source_types.validation_error(
            dataset.get("source_type"),
            experiment_entry,
            experiment_id=dataset.get("experiment_id"),
        )
        if error is not None:
            raise ControlledVocabularyError(error)

    def _validate_source_attributes(self, dataset: Mapping[str, Any]) -> None:
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

        if self._sources is None:
            return
        ignored_keys = set()
        if "source" not in self.required_attributes:
            ignored_keys.add("source")
        error = self._sources.validation_error(
            dataset.get("source_id"), dataset, ignored_keys=ignored_keys
        )
        if error is not None:
            raise ControlledVocabularyError(error)

    def _validate_parent_attributes(self, dataset: Mapping[str, Any]) -> None:
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

        if self._experiments is None:
            return
        error = self._experiments.parent_validation_error(
            dataset.get("experiment_id"), dataset, self._sources
        )
        if error is not None:
            raise ControlledVocabularyError(error)

    def value_allowed(
        self,
        key: str,
        value: Any,
        dataset: DatasetMetadata,
    ) -> bool:
        """Return whether a value is allowed by its typed CV component.

        Parameters
        ----------
        key:
            Dataset attribute name being validated.
        value:
            Dataset attribute value to check.
        dataset:
            Full dataset metadata, used to resolve templated CV values.

        Returns
        -------
        bool
            ``True`` when the value is accepted by the component.
        """

        return self._value_allowed(key, value, dataset.to_dict())

    def _value_allowed(self, key: str, value: Any, metadata: Mapping[str, Any]) -> bool:
        if key in {"license", "license_id", "license_url", "license_type"}:
            if self._license is not None:
                return self._license.accepts(key, value, metadata)
        if key == "source_type":
            return self._source_types.accepts(value)
        rule = self._rules.get(key)
        return rule is None or rule.accepts(value, metadata)

    def controls(self, key: str) -> bool:
        """Return whether a typed component constrains *key*."""

        if key in self._rules:
            return True
        if self._license is None:
            return False
        if key == "license_id":
            return bool(self._license.licenses)
        return key in {"license_url", "license_type"}
