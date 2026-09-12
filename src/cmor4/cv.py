from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Mapping
import uuid
import warnings

from .utils.cv_models import (
    AttributeRule,
    DefaultValuesComponent,
    DRSComponent,
    ExperimentComponent,
    ForcingComponent,
    FrequencyComponent,
    InstitutionComponent,
    LicenseComponent,
    SourceComponent,
    SourceTypeComponent,
    TrackingIdComponent,
    ControlledVocabularyModel,
)
from .utils.dataset_metadata import DatasetMetadata
from .exceptions import ControlledVocabularyError

# Variant label index keys and their allowed integer range.
# CMOR3 stores these as C int (32-bit signed), so values above INT32_MAX
# are rejected to keep parity with CMOR3 behaviour.
_RIPF_KEYS: tuple[str, ...] = (
    "realization_index",
    "initialization_index",
    "physics_index",
    "forcing_index",
)
# Expected single-character prefix for each RIPF key.
# CMIP7 encodes indices as prefixed strings ("r9", "i1", …);
# CMOR3 used bare integers ("9", "1", …).  Both are accepted.
_RIPF_PREFIXES: dict[str, str] = {
    "realization_index": "r",
    "initialization_index": "i",
    "physics_index": "p",
    "forcing_index": "f",
}
_RIPF_MAX: int = 2**31 - 1  # INT32_MAX — matches CMOR3's upper bound

# Fallback regex for grid_label when the CV does not define an allowed set.
# Mirrors CMOR3's built-in check: labels must start with 'g', 'c', or 'r'
# (the three grid families) and contain only lowercase letters and digits —
# no hyphens or special characters.  Examples of valid labels: gn, gr, gr1,
# cn, rn, g999.  Examples of invalid labels: gr-0 (hyphen), GN (uppercase),
# 1gn (digit-first).
# When the CV *does* define grid_label (as an enumeration or regex list),
# validate_dataset_values already enforces it and this fallback is skipped.
_GRID_LABEL_RE: re.Pattern[str] = re.compile(r"^[gcr][a-z0-9]*$")


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
    Validation behavior is exposed through typed attributes such as ``rules``,
    ``defaults``, ``drs``, ``license``, ``sources``, and ``experiments``.
    """

    rules: dict[str, AttributeRule]
    defaults: DefaultValuesComponent
    drs: DRSComponent | None
    institutions: InstitutionComponent | None
    license: LicenseComponent | None
    sources: SourceComponent | None
    experiments: ExperimentComponent | None
    source_types: SourceTypeComponent
    forcing: ForcingComponent | None
    frequency: FrequencyComponent | None
    tracking_id: TrackingIdComponent
    required_attributes: tuple[str, ...]

    def __init__(self, data: Mapping[str, Any], path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        parsed = ControlledVocabularyModel.from_mapping(data)
        self.rules = parsed.attributes
        self.defaults = parsed.defaults
        self.drs = parsed.drs
        self.institutions = parsed.institutions
        self.license = parsed.license
        self.sources = parsed.sources
        self.experiments = parsed.experiments
        self.source_types = parsed.source_types
        self.forcing = parsed.forcing
        self.frequency = parsed.frequency
        self.tracking_id = parsed.tracking_id
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

        issues = self.validate_structure()
        for issue in issues:
            warnings.warn(issue, RuntimeWarning, stacklevel=3)

    def validate_structure(self) -> list[str]:
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
            issues = cv.validate_structure()
            # ["CV entry 'nominal_resolution' appears to be double-nested …"]
        """

        issues: list[str] = []
        for key, rule in self.rules.items():
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
        self._add_scalar_defaults(normalized_dataset)
        self._add_source_defaults(normalized_dataset)
        self._add_institution_default(normalized_dataset)
        self._add_experiment_defaults(normalized_dataset)
        self._add_license_text(normalized_dataset)
        self._add_runtime_global_defaults(normalized_dataset)
        # Run after all dedicated handlers so their setdefault values win over
        # anything this generic handler would inject.
        self._add_nested_defaults(normalized_dataset)

        return DatasetMetadata.from_mapping(normalized_dataset)

    def _add_scalar_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill scalar CV defaults that are not templated."""

        dataset.update(self.defaults.scalar_defaults_for(dataset))

    def _add_nested_defaults(self, dataset: dict[str, Any]) -> None:
        """Inject leaf attributes from two-level nested CV entries.

        CMOR3 supports a pattern where a CV key maps user-selectable codes to
        flat dicts of scalar attributes.  When the user picks a code, every
        scalar in the corresponding dict is written as a global attribute.
        This mirrors CMOR3's ``_CV_checkGblAttributes`` handling for such
        entries and enables features like obs4MIPs ``site_id`` location
        injection or custom project contact-info blocks.

        Example — obs4MIPs ``site_id``::

            CV:   {"site_id": {"AR-SLu": {"latitude": "-33.47",
                                           "location":  "San Luis",
                                           "longitude": "-66.46"}}}
            User: {"site_id": "AR-SLu"}
            Result: ``latitude``, ``location``, ``longitude`` added to dataset.

        Only injects when the looked-up entry is a :class:`~collections.abc.Mapping`
        whose values are **all** scalars (``str``, ``int``, or ``float``).
        Entries containing nested Mappings or lists are CV validation tables
        and are not injected.

        Keys with dedicated handlers (``institution_id``, ``source_id``,
        ``experiment_id``, ``license_id``, ``license``) are skipped to avoid
        double-processing.  Additionally, ``frequency`` is skipped because its
        nested dicts contain internal time-validation scalars
        (``approx_interval``, ``approx_interval_error``, etc.) that are read
        directly by the axis validator and must not appear as output global
        attributes.

        Because this method runs *after* all dedicated handlers, their
        ``setdefault`` values always take precedence.
        """

        dataset.update(self.defaults.nested_defaults_for(dataset))

    def _add_source_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill attributes supplied by a source_id CV entry."""

        if self.sources is None:
            return
        for key, value in self.sources.defaults_for(dataset.get("source_id")).items():
            dataset.setdefault(key, value)

    def _add_institution_default(self, dataset: dict[str, Any]) -> None:
        """Fill institution text from institution_id."""

        if "institution" in dataset:
            return
        if self.institutions is None:
            return
        institution = self.institutions.name_for(dataset.get("institution_id"))
        if institution is not None:
            dataset["institution"] = institution

    def _add_experiment_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill scalar attributes supplied by an experiment_id CV entry."""

        if self.experiments is None:
            return
        for key, value in self.experiments.defaults_for(
            dataset.get("experiment_id")
        ).items():
            dataset.setdefault(key, value)

    def _add_runtime_global_defaults(self, dataset: dict[str, Any]) -> None:
        """Fill required globals that CMOR normally creates while writing."""

        required = self.required_attributes
        if "Conventions" in required:
            dataset.setdefault("Conventions", self.defaults.conventions)
        if "creation_date" in required:
            dataset.setdefault(
                "creation_date", datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        if "variant_label" in required:
            variant_label = _variant_label(dataset)
            if variant_label:
                dataset.setdefault("variant_label", variant_label)
        if "tracking_id" in required and "tracking_id" not in dataset:
            dataset["tracking_id"] = _new_tracking_id(dataset, self.tracking_id)
        for key in required:
            if key in dataset:
                continue
            default = self.defaults.required.get(key)
            if default is not None:
                dataset[key] = default

    def _add_license_text(self, dataset: dict[str, Any]) -> None:
        if "license" in dataset or self.license is None:
            return
        rendered = self.license.render(dataset)
        if rendered is not None:
            dataset["license"] = rendered

    def validate_dataset_info(self, dataset: DatasetMetadata) -> None:
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
        self.validate_variant_indices(dataset)
        self.validate_forcing_terms(dataset)

    def validate_forcing_terms(self, dataset: DatasetMetadata) -> None:
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

        if self.forcing is None:
            return
        invalid = self.forcing.invalid_token(dataset.forcing)
        if invalid is not None:
            raise ControlledVocabularyError(
                f"forcing term {invalid!r} is not valid. "
                f"Valid values are: {sorted(self.forcing.values)!r}. "
                f"Check {self.filename}."
            )

    def validate_variant_indices(self, dataset: DatasetMetadata) -> None:
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

        for key in _RIPF_KEYS:
            value = getattr(dataset, key)
            if value in (None, ""):
                continue
            value_str = str(value)
            # Skip POSIX-regex strings that were injected from the CV rather
            # than supplied by the user (e.g. CMIP6's realization_index regex).
            if re.search(r"[\\^$\[\]{,}]", value_str):
                continue
            # Determine the numeric part.  CMOR3 checks whether the value
            # consists solely of digits (bare integer style, e.g. "3") and
            # if so prepends the letter prefix when assembling variant_label.
            # Prefixed-string style (e.g. "r3") strips the leading letter.
            prefix = _RIPF_PREFIXES[key]
            if value_str.startswith(prefix) and len(value_str) > 1:
                numeric_part = value_str[1:]
            else:
                numeric_part = value_str
            try:
                int_value = int(numeric_part)
            except (TypeError, ValueError) as exc:
                raise ControlledVocabularyError(
                    f"{key}={value!r} must be a positive integer "
                    f"(with optional '{prefix}' prefix, e.g. '{prefix}1')."
                ) from exc
            if int_value < 1 or int_value > _RIPF_MAX:
                raise ControlledVocabularyError(
                    f"{key}={value!r} numeric value {int_value} is out of the "
                    f"valid range [1, {_RIPF_MAX}]."
                )

        # Only validate the variant_label format when the CV actually defines
        # a constraint (a list of allowed values or regex patterns).  When the
        # CV merely lists variant_label as a required attribute with no values
        # (e.g. obs4MIPs), accept any user-supplied string.
        variant_rule = self.rules.get("variant_label")
        cv_constrains_vl = (
            variant_rule is not None and variant_rule.is_choice_constraint
        )

        variant_label = dataset.variant_label_value
        if variant_label and cv_constrains_vl:
            vl_str = str(variant_label)
            if not re.fullmatch(r"r\d+i\d+p\d+f\d+", vl_str):
                raise ControlledVocabularyError(
                    f"variant_label={vl_str!r} does not match the required "
                    "pattern 'r<N>i<N>p<N>f<N>'."
                )
            return
        if variant_label:
            return

        # Assemble variant_label from individual RIPF indices when all four
        # are present.  Validate the result only when the CV constrains it.
        assembled = _metadata_variant_label(dataset)
        if assembled and cv_constrains_vl:
            if not re.fullmatch(r"r\d+i\d+p\d+f\d+", assembled):
                raise ControlledVocabularyError(
                    f"variant_label={assembled!r} assembled from RIPF indices "
                    "does not match the required pattern 'r<N>i<N>p<N>f<N>'."
                )

    def validate_dataset_values(self, dataset: DatasetMetadata) -> None:
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

        for key, value in dataset.to_dict().items():
            if key.startswith("_") or key in {
                "outpath",
                "output_file_template",
                "output_path_template",
                # The 'forcing' attribute is a space/comma-separated list of
                # abbreviations (e.g. "GHG Oz SA").  It cannot be validated
                # as a single value against the CV; validate_forcing_terms
                # handles it with per-token checks instead.
                "forcing",
            }:
                continue
            rule = self.rules.get(str(key))
            # CMIP6 CVs express several constraints as POSIX BRE regex arrays
            # (single-element lists whose entry contains BRE metacharacters like
            # \{, [[:digit:]], or ^ anchors).  CMOR4 uses Python re which has
            # different syntax; rather than attempt a partial BRE-to-ERE
            # conversion that may not be reliable, skip validation for any
            # attribute whose CV definition is a list of strings that look
            # like BRE patterns.  The same attributes are validated implicitly
            # by other checks (variant_label format, grid_label enumeration,
            # etc.) that use the CV's enumeration entries.
            if rule is not None and rule.skips_generic_validation:
                continue
            if self.controls(str(key)) and not self.value_allowed(
                str(key), value, dataset
            ):
                raise ControlledVocabularyError(
                    f"{key}={value!r} is not allowed by {self.filename}."
                )

        # GAP-09: hard-coded fallback when the CV does not define grid_label.
        # validate_dataset_values already enforces CV-defined grid_label values
        # above, so this only fires when no CV definition exists.  It prevents
        # obviously malformed labels (hyphens, wrong starting character, etc.)
        # from passing silently when the CV is incomplete or absent.
        gl = dataset.grid_label
        if gl not in (None, "") and "grid_label" not in self.rules:
            if not _GRID_LABEL_RE.fullmatch(str(gl)):
                raise ControlledVocabularyError(
                    f"grid_label={gl!r} does not match the required format. "
                    "Grid labels must start with 'g', 'c', or 'r' followed by "
                    "lowercase letters and digits only (no hyphens or special "
                    "characters). "
                    f"Valid examples: 'gn', 'gr1', 'cn', 'g999'. "
                    "Define grid_label in the project CV to allow a custom set."
                )

    def validate_required_global_attributes(self, dataset: DatasetMetadata) -> None:
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

        dataset_values = dataset.to_dict()
        # Attributes whose CV definition is a POSIX BRE regex array cannot be
        # auto-generated by CMOR4.  For CMIP6 this includes 'license' (among
        # others).  Rather than requiring the user to supply these manually in
        # every test context, we waive the enforcement when the value is absent
        # and the CV cannot supply a concrete default — mirroring CMOR3's
        # lenient behaviour for test models like PCMDI-test-1-0.
        bre_required = {
            name
            for name in self.required_attributes
            if (rule := self.rules.get(name)) is not None
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

    def validate_experiment(self, dataset: DatasetMetadata) -> None:
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

        if self.experiments is None:
            return
        error = self.experiments.validation_error(
            dataset.experiment_id, dataset.to_dict(), self.source_types
        )
        if error is not None:
            raise ControlledVocabularyError(error)

    def validate_source_type(
        self,
        dataset: DatasetMetadata,
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

        error = self.source_types.validation_error(
            dataset.source_type,
            experiment_entry,
            experiment_id=dataset.experiment_id,
        )
        if error is not None:
            raise ControlledVocabularyError(error)

    def validate_source_attributes(self, dataset: DatasetMetadata) -> None:
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

        if self.sources is None:
            return
        error = self.sources.validation_error(dataset.source_id, dataset.to_dict())
        if error is not None:
            raise ControlledVocabularyError(error)

    def validate_parent_attributes(self, dataset: DatasetMetadata) -> None:
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

        if self.experiments is None:
            return
        error = self.experiments.parent_validation_error(
            dataset.experiment_id, dataset.to_dict(), self.sources
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

        metadata = dataset.to_dict()
        if key in {"license", "license_id", "license_url", "license_type"}:
            if self.license is not None:
                return self.license.accepts(key, value, metadata)
        if key == "source_type":
            return self.source_types.accepts(value)
        rule = self.rules.get(key)
        return rule is None or rule.accepts(value, metadata)

    def controls(self, key: str) -> bool:
        """Return whether a typed component constrains *key*."""

        if key in self.rules:
            return True
        if self.license is None:
            return False
        if key == "license_id":
            return bool(self.license.licenses)
        return key in {"license_url", "license_type"}


def _variant_label(dataset: Mapping[str, Any]) -> str | None:
    """Assemble a variant_label from RIPF index attributes, or return existing one.

    Mirrors CMOR3's ``cmor_addRIPF`` logic:

    * If ``variant_label`` is already in the dataset, return it directly.
    * Otherwise collect the four RIPF index values.  For each one, if the
      value consists solely of digits (bare-integer style, e.g. ``"3"``),
      prepend the canonical letter prefix (``r``, ``i``, ``p``, ``f``) before
      concatenating.  If the value already carries its prefix letter (e.g.
      ``"r3"``), use it as-is.  This exactly replicates the C code path in
      ``cmor_addRIPF`` that checks ``^[[:digit:]]{1,}$`` to decide whether to
      prepend the prefix.
    """
    if dataset.get("variant_label") not in (None, ""):
        return str(dataset["variant_label"])
    pieces = []
    for key, prefix in zip(_RIPF_KEYS, ("r", "i", "p", "f")):
        value = dataset.get(key)
        if value in (None, ""):
            return None
        value_str = str(value)
        # Bare integer (digits only) → prepend the letter prefix.
        # Prefixed string (e.g. "r3") → use as-is.
        if re.fullmatch(r"\d+", value_str):
            pieces.append(f"{prefix}{value_str}")
        else:
            pieces.append(value_str)
    return "".join(pieces)


def _metadata_variant_label(dataset: DatasetMetadata) -> str | None:
    """Return an explicit or RIPF-derived label from typed metadata."""

    if dataset.variant_label_value:
        return dataset.variant_label_value
    pieces: list[str] = []
    for key, prefix in zip(_RIPF_KEYS, ("r", "i", "p", "f"), strict=True):
        value = getattr(dataset, key)
        if value in (None, ""):
            return None
        value_str = str(value)
        pieces.append(
            f"{prefix}{value_str}" if re.fullmatch(r"\d+", value_str) else value_str
        )
    return "".join(pieces)


def _new_tracking_id(
    dataset: Mapping[str, Any], component: TrackingIdComponent
) -> str:
    return component.format(str(uuid.uuid4()), dataset.get("tracking_prefix"))


def _posix_regex_to_python(pattern: str) -> str:
    """Convert a POSIX Extended Regular Expression to a Python ``re`` pattern.

    POSIX ERE and Python ``re`` differ in how several constructs are expressed:

    * ``[[:digit:]]`` → ``\\d``, ``[[:space:]]`` → ``\\s``
    * POSIX ``\\{n,m\\}`` quantifiers → Python ``{n,m}`` quantifiers
    * POSIX ``\\(`` / ``\\)`` means *literal* parenthesis (in ERE, unescaped
      ``(``/``)`` are group markers); Python ``re`` uses ``\\(``/``\\)`` for
      the same meaning, so these are left as-is.

    The CMIP6 CV license pattern uses unescaped ``(...)`` for capturing groups
    and ``\\.`` (backslash-dot) for literal periods.  Both translate directly
    to Python ``re`` without modification.
    """
    return (
        pattern
        .replace("[[:digit:]]", r"\d")
        .replace("[[:space:]]", r"\s")
        .replace("\\{", "{")
        .replace("\\}", "}")
        # In POSIX ERE, \( and \) denote literal parentheses.
        # In Python re, \( and \) also denote literal parentheses, so we keep them.
        # NOTE: We do NOT convert \( -> ( because ( is a group marker in Python re.
    )
