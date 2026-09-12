"""Typed components parsed from a project controlled vocabulary.

The JSON files used by CMOR are intentionally extensible and contain several
different kinds of data under one ``CV`` object.  This module turns the parts
that affect dataset validation into small Pydantic models.  The coordinating
:class:`cmor4.cv.ControlledVocabulary` consumes these typed components rather
than behaving like the source JSON mapping.

The models are deliberately project-neutral.  A new project can add ordinary
attributes without changing a monolithic schema: scalar, list, template, and
lookup-table definitions become :class:`AttributeRule` instances.  Sections
whose meaning spans multiple attributes use dedicated components instead.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .templates import is_unresolved_template, render_template


ScalarValue = str | int | float | bool

# Sections with dedicated semantics, or internal lookup data that must not be
# copied to NetCDF attributes by generic nested-default processing.
_NESTED_DEFAULT_EXCLUSIONS = frozenset({
    "DRS",
    "experiment_id",
    "frequency",
    "institution_id",
    "license",
    "license_id",
    "source_id",
    "source_type",
})

_EXPERIMENT_DEFAULT_EXCLUSIONS = frozenset({
    "additional_allowed_model_components",
    "end_year",
    "min_number_yrs_per_sim",
    "parent_activity_id",
    "parent_experiment_id",
    "required_source_type",
    "source_type",
    "start_year",
    "tier",
})


class _CVModel(BaseModel):
    """Immutable base for parsed CV components."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ExactMatcher(_CVModel):
    """Match one scalar CV value."""

    kind: Literal["exact"] = "exact"
    expected: str

    def accepts(self, value: Any, metadata: Mapping[str, Any]) -> bool:
        del metadata
        return str(value) == self.expected


class TemplateMatcher(_CVModel):
    """Match a value rendered from other dataset attributes."""

    kind: Literal["template"] = "template"
    template: str
    separator: str | None = None

    def accepts(self, value: Any, metadata: Mapping[str, Any]) -> bool:
        candidate = str(value)
        return candidate == self.template or candidate == render_template(
            self.template, metadata, self.separator
        )


class ChoiceMatcher(_CVModel):
    """Match one of a list of literal values or regular expressions."""

    kind: Literal["choices"] = "choices"
    choices: tuple[str, ...]

    def accepts(self, value: Any, metadata: Mapping[str, Any]) -> bool:
        del metadata
        candidate = str(value)
        return any(_choice_accepts(candidate, choice) for choice in self.choices)


class LookupMatcher(_CVModel):
    """Match a key in a CV lookup table."""

    kind: Literal["lookup"] = "lookup"
    keys: frozenset[str]
    tokens: bool = False

    def accepts(self, value: Any, metadata: Mapping[str, Any]) -> bool:
        del metadata
        values = str(value).split() if self.tokens else (str(value),)
        return all(candidate in self.keys for candidate in values if candidate)


class OpenMatcher(_CVModel):
    """Represent a CV value that does not define a generic constraint."""

    kind: Literal["open"] = "open"

    def accepts(self, value: Any, metadata: Mapping[str, Any]) -> bool:
        del value, metadata
        return True


ValueMatcher = (
    ExactMatcher
    | TemplateMatcher
    | ChoiceMatcher
    | LookupMatcher
    | OpenMatcher
)


class AttributeRule(_CVModel):
    """Generic validation component for one dataset attribute."""

    name: str
    matcher: ValueMatcher = Field(discriminator="kind")

    @classmethod
    def from_definition(cls, name: str, definition: Any) -> "AttributeRule":
        """Build the appropriate matcher from a raw JSON CV definition."""

        if isinstance(definition, Mapping):
            matcher: ValueMatcher = LookupMatcher(
                keys=frozenset(str(key) for key in definition),
                tokens=name in {"realm", "source_type"},
            )
        elif isinstance(definition, list):
            matcher = ChoiceMatcher(choices=tuple(str(item) for item in definition))
        elif is_unresolved_template(definition):
            matcher = TemplateMatcher(
                template=str(definition),
                separator="-" if name == "branding_suffix" else None,
            )
        elif isinstance(definition, (str, int, float, bool)):
            matcher = ExactMatcher(expected=str(definition))
        else:
            matcher = OpenMatcher()
        return cls(name=name, matcher=matcher)

    def accepts(self, value: Any, metadata: Mapping[str, Any]) -> bool:
        """Return whether *value* satisfies this component."""

        return self.matcher.accepts(value, metadata)

    @property
    def allowed_values(self) -> tuple[str, ...]:
        """Return explicit choices or lookup keys for introspection."""

        if isinstance(self.matcher, ChoiceMatcher):
            return self.matcher.choices
        if isinstance(self.matcher, LookupMatcher):
            return tuple(sorted(self.matcher.keys))
        if isinstance(self.matcher, ExactMatcher):
            return (self.matcher.expected,)
        return ()

    @property
    def skips_generic_validation(self) -> bool:
        """Whether this rule contains a legacy POSIX regex best handled elsewhere."""

        return isinstance(self.matcher, ChoiceMatcher) and _looks_like_posix_regex_list(
            list(self.matcher.choices)
        )

    @property
    def is_choice_constraint(self) -> bool:
        """Whether the definition is a non-empty list of choices or patterns."""

        return isinstance(self.matcher, ChoiceMatcher) and bool(self.matcher.choices)

    @property
    def is_double_nested(self) -> bool:
        """Whether a lookup is suspiciously wrapped in its own attribute name."""

        return (
            isinstance(self.matcher, LookupMatcher)
            and len(self.matcher.keys) == 1
            and self.name in self.matcher.keys
        )


class DRSComponent(_CVModel):
    """Directory and filename templates from the CV's ``DRS`` section."""

    model_config = ConfigDict(frozen=True, extra="allow")

    directory_path_template: str | None = None
    filename_template: str | None = None

    @field_validator("directory_path_template", "filename_template", mode="before")
    @classmethod
    def _ignore_non_string_templates(cls, value: Any) -> str | None:
        return value if isinstance(value, str) and value else None

    @classmethod
    def from_definition(cls, definition: Any) -> "DRSComponent | None":
        if not isinstance(definition, Mapping):
            return None
        return cls.model_validate(definition)

    def templates(self) -> tuple[str | None, str | None]:
        """Return ``(directory_path_template, filename_template)``."""

        return self.directory_path_template, self.filename_template


class NestedDefaultComponent(_CVModel):
    """Scalar attributes contributed by selected two-level CV entries."""

    entries: dict[str, dict[str, dict[str, ScalarValue]]] = Field(
        default_factory=dict
    )

    @classmethod
    def from_mapping(cls, cv: Mapping[str, Any]) -> "NestedDefaultComponent":
        catalogs: dict[str, dict[str, dict[str, ScalarValue]]] = {}
        for attribute, definition in cv.items():
            if attribute in _NESTED_DEFAULT_EXCLUSIONS or not isinstance(
                definition, Mapping
            ):
                continue
            entries: dict[str, dict[str, ScalarValue]] = {}
            for selector, raw_values in definition.items():
                if not isinstance(raw_values, Mapping) or not all(
                    isinstance(value, (str, int, float))
                    for value in raw_values.values()
                ):
                    continue
                entries[str(selector)] = {
                    str(key): value for key, value in raw_values.items()
                }
            if entries:
                catalogs[str(attribute)] = entries
        return cls(entries=catalogs)

    def defaults_for(self, metadata: Mapping[str, Any]) -> dict[str, ScalarValue]:
        """Return defaults contributed by every selected nested entry."""

        defaults: dict[str, ScalarValue] = {}
        for attribute, entries in self.entries.items():
            selected = metadata.get(attribute)
            if selected in (None, ""):
                continue
            for key, value in entries.get(str(selected), {}).items():
                defaults.setdefault(key, value)
        return defaults


class DefaultValuesComponent(_CVModel):
    """Static scalar, nested, and required-attribute defaults."""

    scalars: dict[str, ScalarValue] = Field(default_factory=dict)
    nested: NestedDefaultComponent = Field(default_factory=NestedDefaultComponent)
    required: dict[str, ScalarValue] = Field(default_factory=dict)
    conventions: str = "CF-1.12"

    @classmethod
    def from_mapping(cls, cv: Mapping[str, Any]) -> "DefaultValuesComponent":
        scalars = {
            str(key): value
            for key, value in cv.items()
            if isinstance(value, (str, int, float))
            and not (isinstance(value, str) and ("<" in value or ">" in value))
        }
        required_names = cv.get("required_global_attributes", ())
        required: dict[str, ScalarValue] = {}
        if isinstance(required_names, (list, tuple)):
            for raw_name in required_names:
                name = str(raw_name)
                definition = _definition_for(name, cv)
                default = _definition_default(definition)
                if default is not None:
                    required[name] = default
        return cls(
            scalars=scalars,
            nested=NestedDefaultComponent.from_mapping(cv),
            required=required,
            conventions=_conventions_default(cv.get("Conventions")),
        )

    def scalar_defaults_for(
        self, metadata: Mapping[str, Any]
    ) -> dict[str, ScalarValue]:
        """Return scalar defaults missing from *metadata*."""

        return {
            key: value for key, value in self.scalars.items() if key not in metadata
        }

    def nested_defaults_for(
        self, metadata: Mapping[str, Any]
    ) -> dict[str, ScalarValue]:
        """Return selected nested defaults missing from *metadata*."""

        return {
            key: value
            for key, value in self.nested.defaults_for(metadata).items()
            if key not in metadata
        }


class LicenseRecord(_CVModel):
    """Metadata contributed by one ``license_id`` entry."""

    model_config = ConfigDict(frozen=True, extra="allow")

    license_type: str | None = None
    license_url: str | None = None


class LicenseComponent(_CVModel):
    """License selection and template rendering rules."""

    rule: AttributeRule
    licenses: dict[str, LicenseRecord] = Field(default_factory=dict)
    template: str | None = None

    @classmethod
    def from_definition(cls, definition: Any) -> "LicenseComponent":
        """Parse scalar, regex-list, or catalog/template license formats."""

        licenses: dict[str, LicenseRecord] = {}
        template: str | None = None
        if isinstance(definition, Mapping):
            raw_licenses = definition.get("license_id")
            raw_template = definition.get("license_template")
            if isinstance(raw_licenses, Mapping):
                for identifier, record in raw_licenses.items():
                    if isinstance(record, Mapping):
                        licenses[str(identifier)] = LicenseRecord.model_validate(record)
            if isinstance(raw_template, str):
                template = raw_template
        return cls(
            rule=AttributeRule.from_definition("license", definition),
            licenses=licenses,
            template=template,
        )

    def render(self, metadata: Mapping[str, Any]) -> str | None:
        """Render the selected license, or return ``None`` when not templated."""

        if self.template is None:
            return None
        license_id = metadata.get("license_id")
        if license_id in (None, ""):
            return None
        record = self.licenses.get(str(license_id))
        if record is None:
            return None
        tokens = {**metadata, **record.model_dump(exclude_none=True)}
        return render_template(self.template, tokens)

    @property
    def default_text(self) -> str | None:
        """Return a literal project license, when the CV defines one."""

        if isinstance(self.rule.matcher, ExactMatcher):
            return self.rule.matcher.expected
        return None

    def accepts(
        self, name: str, value: Any, metadata: Mapping[str, Any]
    ) -> bool:
        """Validate a license-related attribute against the selected license."""

        if name == "license":
            # Templated licenses are generated by this component. Preserve the
            # historical allowance for explicitly supplied project text.
            return self.template is not None or self.rule.accepts(value, metadata)
        if name == "license_id":
            return str(value) in self.licenses
        if name in {"license_url", "license_type"}:
            expected = self.expected_selected_value(name, metadata)
            return expected is None or str(value) == expected
        return True

    def expected_selected_value(
        self, name: str, metadata: Mapping[str, Any]
    ) -> str | None:
        """Return license metadata selected by ``license_id``."""

        license_id = metadata.get("license_id")
        if license_id in (None, ""):
            return None
        record = self.licenses.get(str(license_id))
        if record is None:
            return None
        value = getattr(record, name, None)
        return str(value) if value is not None else None


class SourceComponent(_CVModel):
    """Typed catalog of source IDs and their contributed attributes."""

    entries: dict[str, dict[str, Any]]

    @classmethod
    def from_definition(cls, definition: Any) -> "SourceComponent | None":
        if not isinstance(definition, Mapping):
            return None
        entries = {
            str(identifier): dict(entry)
            for identifier, entry in definition.items()
            if isinstance(entry, Mapping)
        }
        return cls(entries=entries)

    def entry_for(self, source_id: Any) -> Mapping[str, Any] | None:
        if source_id in (None, ""):
            return None
        return self.entries.get(str(source_id))

    def defaults_for(self, source_id: Any) -> dict[str, Any]:
        """Return scalar/singleton defaults contributed by a source entry."""

        entry = self.entry_for(source_id)
        if entry is None:
            return {}
        return {
            key: default
            for key, value in entry.items()
            if key != "source_id"
            if (default := _single_default(value)) is not None
        }

    def validation_error(
        self, source_id: Any, metadata: Mapping[str, Any]
    ) -> str | None:
        """Return the first source-specific metadata mismatch."""

        entry = self.entry_for(source_id)
        if entry is None:
            return None
        for key, expected in entry.items():
            if key == "source_id" or key not in metadata or expected in (None, ""):
                continue
            actual = metadata[key]
            matches = (
                str(actual) in {str(item) for item in expected}
                if isinstance(expected, list)
                else str(actual) == str(expected)
            )
            if not matches:
                return (
                    f"{key}={actual!r} does not match source_id={source_id!r} "
                    f"CV value {expected!r}."
                )
        return None


class ExperimentComponent(_CVModel):
    """Typed catalog of experiment IDs and their cross-field constraints."""

    entries: dict[str, dict[str, Any]]

    @classmethod
    def from_definition(cls, definition: Any) -> "ExperimentComponent | None":
        if not isinstance(definition, Mapping):
            return None
        entries = {
            str(identifier): dict(entry)
            for identifier, entry in definition.items()
            if isinstance(entry, Mapping)
        }
        return cls(entries=entries)

    def entry_for(self, experiment_id: Any) -> Mapping[str, Any] | None:
        if experiment_id in (None, ""):
            return None
        return self.entries.get(str(experiment_id))

    def defaults_for(self, experiment_id: Any) -> dict[str, Any]:
        """Return output defaults contributed by an experiment entry."""

        entry = self.entry_for(experiment_id)
        if entry is None:
            return {}
        return {
            key: default
            for key, value in entry.items()
            if key not in _EXPERIMENT_DEFAULT_EXCLUSIONS
            if (default := _single_default(value)) is not None
        }

    def validation_error(
        self,
        experiment_id: Any,
        metadata: Mapping[str, Any],
        source_types: "SourceTypeComponent",
    ) -> str | None:
        """Return the first experiment-specific metadata mismatch."""

        entry = self.entry_for(experiment_id)
        if entry is None:
            return None
        source_type_error = source_types.validation_error(
            metadata.get("source_type"), entry, experiment_id=experiment_id
        )
        if source_type_error is not None:
            return source_type_error
        excluded = {
            "additional_allowed_model_components",
            "description",
            "parent_activity_id",
            "parent_experiment_id",
            "required_source_type",
            "source_type",
        }
        for key, expected in entry.items():
            if key in excluded or key not in metadata or expected in (None, ""):
                continue
            actual = metadata[key]
            matches = (
                str(actual) in {str(item) for item in expected}
                if isinstance(expected, list)
                else str(actual) == str(expected)
            )
            if not matches:
                return (
                    f"{key}={actual!r} does not match "
                    f"experiment_id={experiment_id!r} CV value {expected!r}."
                )
        return None

    def parent_validation_error(
        self,
        experiment_id: Any,
        metadata: Mapping[str, Any],
        sources: SourceComponent | None,
    ) -> str | None:
        """Validate parent metadata selected by an experiment entry."""

        entry = self.entry_for(experiment_id)
        if entry is None:
            return None
        expected_experiments = _values(entry.get("parent_experiment_id"))
        if all(str(value) == "no parent" for value in expected_experiments):
            expected_experiments = ()
        parent_attributes = (
            "parent_activity_id",
            "parent_mip_era",
            "parent_source_id",
            "parent_time_units",
            "parent_variant_label",
            "branch_time_in_child",
            "branch_time_in_parent",
        )
        parent_experiment_id = metadata.get("parent_experiment_id")
        if not expected_experiments:
            if parent_experiment_id is not None:
                return (
                    f"experiment_id={experiment_id!r} does not allow "
                    "parent_experiment_id."
                )
            unexpected = [
                name for name in parent_attributes if metadata.get(name) is not None
            ]
            if unexpected:
                return (
                    f"experiment_id={experiment_id!r} does not allow parent "
                    f"attributes: {', '.join(unexpected)}."
                )
            return None

        if parent_experiment_id in (None, ""):
            return f"experiment_id={experiment_id!r} requires parent_experiment_id."
        if str(parent_experiment_id) not in {
            str(value) for value in expected_experiments
        }:
            return (
                f"parent_experiment_id={parent_experiment_id!r} does not match "
                f"experiment_id={experiment_id!r} CV values "
                f"{expected_experiments!r}."
            )

        parent_activity_error = _required_value_error(
            metadata,
            "parent_activity_id",
            entry.get("parent_activity_id"),
            experiment_id,
        )
        if parent_activity_error is not None:
            return parent_activity_error
        parent_source_id = metadata.get("parent_source_id")
        if parent_source_id in (None, ""):
            return "parent_source_id is required."
        if sources is not None and sources.entry_for(parent_source_id) is None:
            return f"parent_source_id={parent_source_id!r} is not in the CV."

        mip_era = str(metadata.get("mip_era") or "")
        parent_mip_era = metadata.get("parent_mip_era")
        if mip_era and parent_mip_era not in (mip_era, None, ""):
            return f"parent_mip_era={parent_mip_era!r} does not match {mip_era!r}."
        for key in ("parent_mip_era", "parent_time_units", "parent_variant_label"):
            if metadata.get(key) in (None, ""):
                return f"{key} is required."
        if not re.fullmatch(
            r"days\s+since\s+\d{4}-\d{1,2}-\d{1,2}.*",
            str(metadata.get("parent_time_units")),
        ):
            parent_time_units = metadata.get("parent_time_units")
            return f"parent_time_units={parent_time_units!r} is invalid."
        if not re.fullmatch(
            r"r\d+i\d+p\d+f\d+", str(metadata.get("parent_variant_label"))
        ):
            return (
                f"parent_variant_label={metadata.get('parent_variant_label')!r} "
                "is invalid."
            )
        for key in ("branch_time_in_child", "branch_time_in_parent"):
            value = metadata.get(key)
            if value is None:
                return f"{key} is required."
            try:
                float(value)
            except (TypeError, ValueError):
                return f"{key}={value!r} must be numeric."
        return None


class InstitutionComponent(_CVModel):
    """Institution names selected by ``institution_id``."""

    entries: dict[str, str]

    @classmethod
    def from_definition(cls, definition: Any) -> "InstitutionComponent | None":
        if not isinstance(definition, Mapping):
            return None
        entries = {
            str(identifier): name
            for identifier, name in definition.items()
            if isinstance(name, str) and name
        }
        return cls(entries=entries)

    def name_for(self, institution_id: Any) -> str | None:
        if institution_id in (None, ""):
            return None
        return self.entries.get(str(institution_id))


class SourceTypeComponent(_CVModel):
    """Source-type vocabulary and experiment-specific composition rules."""

    values: frozenset[str] = frozenset()

    @classmethod
    def from_definition(cls, definition: Any) -> "SourceTypeComponent":
        if isinstance(definition, Mapping):
            values = frozenset(str(value) for value in definition)
        elif isinstance(definition, list):
            values = frozenset(str(value) for value in definition)
        else:
            values = frozenset()
        return cls(values=values)

    def accepts(self, source_type: Any) -> bool:
        """Return whether every source-type token is in the project CV."""

        if not self.values:
            return True
        return all(
            token in self.values for token in str(source_type).split() if token
        )

    def validation_error(
        self,
        source_type: Any,
        experiment_entry: Mapping[str, Any],
        *,
        experiment_id: Any = None,
    ) -> str | None:
        """Return an experiment composition error, or ``None`` when valid."""

        required_source_types = _values(
            experiment_entry.get("required_source_type")
        )
        required = required_source_types or _values(
            experiment_entry.get("required_model_components")
        )
        additional = _values(
            experiment_entry.get("additional_allowed_model_components")
        )
        if not required and not additional:
            return None
        if source_type in (None, ""):
            return "source_type is required." if required_source_types else None

        source_type_text = str(source_type)
        for expected in required:
            if not _pattern_matches(source_type_text, expected):
                return (
                    f"source_type={source_type!r} is missing required "
                    f"source type {expected!r}."
                )
        allowed = (*required, *additional)
        for token in source_type_text.split():
            if not any(_pattern_matches(token, item) for item in allowed):
                context = (
                    f" by experiment_id={experiment_id!r}"
                    if experiment_id not in (None, "")
                    else ""
                )
                return (
                    f"source_type={source_type!r} contains source type "
                    f"{token!r} that is not allowed{context}."
                )
        return None


class ForcingComponent(_CVModel):
    """Token vocabulary for the annotated CMIP ``forcing`` attribute."""

    values: frozenset[str]

    @classmethod
    def from_definition(cls, definition: Any) -> "ForcingComponent | None":
        if isinstance(definition, Mapping):
            return cls(values=frozenset(str(value) for value in definition))
        if isinstance(definition, list):
            return cls(values=frozenset(str(value) for value in definition))
        return None

    def invalid_token(self, forcing: Any) -> str | None:
        """Return the first invalid token, ignoring any parenthetical annotation."""

        text = str(forcing or "").replace(",", " ")
        annotation = text.find("(")
        if annotation != -1:
            text = text[:annotation]
        for token in text.split():
            if token not in self.values:
                return token
        return None


class FrequencyRecord(_CVModel):
    """Optional interval metadata attached to a frequency code."""

    model_config = ConfigDict(frozen=True, extra="allow")

    approx_interval: float | None = None
    approx_interval_warning: float | None = None
    approx_interval_error: float | None = None


class FrequencyComponent(_CVModel):
    """Frequency identifiers and their optional time-interval metadata."""

    entries: dict[str, FrequencyRecord | None]

    @classmethod
    def from_definition(cls, definition: Any) -> "FrequencyComponent | None":
        if not isinstance(definition, Mapping):
            return None
        entries = {
            str(name): (
                FrequencyRecord.model_validate(value)
                if isinstance(value, Mapping)
                else None
            )
            for name, value in definition.items()
        }
        return cls(entries=entries)

    def interval_for(self, frequency: str) -> FrequencyRecord | None:
        """Return interval metadata for *frequency*, when defined."""

        return self.entries.get(frequency)


class TrackingIdComponent(_CVModel):
    """Rules used to prefix generated tracking identifiers."""

    prefix: str | None = None

    @classmethod
    def from_definitions(
        cls, prefix_definition: Any, tracking_definition: Any
    ) -> "TrackingIdComponent":
        prefix: str | None = None
        if isinstance(prefix_definition, list) and len(prefix_definition) == 1:
            prefix = str(prefix_definition[0])
        elif isinstance(prefix_definition, str) and prefix_definition:
            prefix = prefix_definition
        if prefix is None and isinstance(tracking_definition, list):
            if tracking_definition:
                candidate = str(tracking_definition[0]).lstrip("^").rstrip("$")
                if ".*" in candidate:
                    prefix = candidate.split(".*")[0]
                else:
                    match = re.match(r"^([^[({*+?]+)", candidate)
                    if match:
                        prefix = match.group(1).replace("\\.", ".")
        return cls(prefix=prefix)

    def format(self, identifier: str, override: Any = None) -> str:
        """Apply the CV or dataset tracking prefix to an identifier."""

        prefix = override if override not in (None, "") else self.prefix
        if prefix in (None, ""):
            return identifier
        prefix_text = str(prefix)
        if not prefix_text.endswith("/"):
            prefix_text += "/"
        return f"{prefix_text}{identifier}"


class ControlledVocabularyModel(_CVModel):
    """Parsed validation components for one controlled vocabulary document."""

    attributes: dict[str, AttributeRule]
    required_global_attributes: tuple[str, ...] = ()
    defaults: DefaultValuesComponent
    drs: DRSComponent | None = None
    institutions: InstitutionComponent | None = None
    license: LicenseComponent | None = None
    sources: SourceComponent | None = None
    experiments: ExperimentComponent | None = None
    source_types: SourceTypeComponent
    forcing: ForcingComponent | None = None
    frequency: FrequencyComponent | None = None
    tracking_id: TrackingIdComponent

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ControlledVocabularyModel":
        """Parse either a complete ``{"CV": ...}`` document or its CV body."""

        nested = data.get("CV", data)
        if not isinstance(nested, Mapping):
            raise ValueError("controlled vocabulary 'CV' value must be an object")
        cv = {str(key): value for key, value in nested.items()}
        required = cv.get("required_global_attributes", ())
        required_values = (
            tuple(str(value) for value in required)
            if isinstance(required, (list, tuple))
            else ()
        )
        return cls(
            attributes={
                name: AttributeRule.from_definition(name, definition)
                for name, definition in cv.items()
            },
            required_global_attributes=required_values,
            defaults=DefaultValuesComponent.from_mapping(cv),
            drs=DRSComponent.from_definition(cv.get("DRS")),
            institutions=InstitutionComponent.from_definition(
                cv.get("institution_id")
            ),
            license=(
                LicenseComponent.from_definition(cv["license"])
                if "license" in cv
                else None
            ),
            sources=SourceComponent.from_definition(cv.get("source_id")),
            experiments=ExperimentComponent.from_definition(cv.get("experiment_id")),
            source_types=SourceTypeComponent.from_definition(cv.get("source_type")),
            forcing=ForcingComponent.from_definition(cv.get("forcing")),
            frequency=FrequencyComponent.from_definition(cv.get("frequency")),
            tracking_id=TrackingIdComponent.from_definitions(
                cv.get("tracking_id_prefix"), cv.get("tracking_id")
            ),
        )


def _choice_accepts(value: str, choice: str) -> bool:
    """Match a list item literally first and as a translated POSIX regex second."""

    if value == choice:
        return True
    pattern = (
        choice.replace("[[:digit:]]", r"\d")
        .replace("[[:space:]]", r"\s")
        .replace("[[:alpha:]]", r"[A-Za-z]")
        .replace("\\{", "{")
        .replace("\\}", "}")
    )
    try:
        return re.fullmatch(pattern, value) is not None
    except re.error:
        return False


def _single_default(value: Any) -> Any:
    if value in (None, "") or isinstance(value, Mapping):
        return None
    if isinstance(value, list):
        return value[0] if len(value) == 1 else None
    return value


def _definition_for(name: str, cv: Mapping[str, Any]) -> Any:
    if name in cv:
        return cv[name]
    license_definition = cv.get("license")
    if name == "license_id" and isinstance(license_definition, Mapping):
        return license_definition.get("license_id")
    return None


def _definition_default(definition: Any) -> ScalarValue | None:
    if isinstance(definition, Mapping):
        keys = list(definition)
        default: Any = keys[0] if len(keys) == 1 else None
    else:
        default = _single_default(definition)
    if _looks_like_posix_regex_list(definition):
        return None
    if isinstance(default, str) and re.search(r"[\\^$\[\]{,}()|]", default):
        return None
    return default if isinstance(default, (str, int, float, bool)) else None


def _conventions_default(definition: Any) -> str:
    if isinstance(definition, list) and "CF-1.12" in definition:
        return "CF-1.12"
    if isinstance(definition, list) and definition:
        candidate = str(definition[-1])
        return (
            "CF-1.7 CMIP-6.2"
            if re.search(r"[\\^$\[\]{,}()|]", candidate)
            else candidate
        )
    if isinstance(definition, str) and definition:
        return definition
    return "CF-1.12"


def _looks_like_posix_regex_list(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str):
        return False
    item = value[0]
    return any(
        indicator in item
        for indicator in ("\\{", "[[:digit:]]", "[[:space:]]", "[[:alpha:]]")
    ) or bool(re.match(r"^\^.*\.\*", item))


def _values(value: Any) -> tuple[Any, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, list):
        return tuple(item for item in value if item not in (None, ""))
    return (value,)


def _required_value_error(
    metadata: Mapping[str, Any], key: str, expected: Any, experiment_id: Any
) -> str | None:
    value = metadata.get(key)
    if value in (None, ""):
        return f"{key} is required."
    matches = (
        str(value) in {str(item) for item in expected}
        if isinstance(expected, list)
        else str(value) == str(expected)
    )
    if expected not in (None, "") and not matches:
        return (
            f"{key}={value!r} does not match experiment_id={experiment_id!r} "
            f"CV value {expected!r}."
        )
    return None


def _pattern_matches(value: str, pattern: Any) -> bool:
    pattern_text = (
        str(pattern)
        .replace("[[:digit:]]", r"\d")
        .replace("[[:space:]]", r"\s")
        .replace("[[:alpha:]]", r"[A-Za-z]")
        .replace("\\{", "{")
        .replace("\\}", "}")
    )
    try:
        return re.search(pattern_text, value) is not None
    except re.error:
        return value == str(pattern)
