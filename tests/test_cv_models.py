from __future__ import annotations

from collections.abc import Mapping

import pytest

from cmor.cv import ControlledVocabulary
from cmor.exceptions import ControlledVocabularyError
from cmor.utils.cv_models import (
    AttributeRule,
    DRSComponent,
    ExperimentComponent,
    ForcingComponent,
    FrequencyComponent,
    GridLabelComponent,
    LicenseComponent,
    SourceComponent,
    SourceTypeComponent,
    TrackingIdComponent,
    VariantLabelComponent,
)
from cmor.utils.dataset_metadata import DatasetMetadata


def _sample_cv(
    required_global_attributes: list[str] | None = None,
) -> ControlledVocabulary:
    required = required_global_attributes or [
        "Conventions",
        "activity_id",
        "source_id",
    ]
    return ControlledVocabulary({
        "CV": {
            "Conventions": ["CF-1.11", "CF-1.12"],
            "DRS": {
                "directory_path_template": "<mip_era><source_id>",
                "filename_template": "<variable_id><source_id>",
            },
            "activity_id": ["CMIP", "^Scenario.*$"],
            "branding_suffix": "<frequency><realm>",
            "experiment_id": {
                "historical": {
                    "activity_id": ["CMIP"],
                    "experiment": "historical simulation",
                    "required_source_type": ["AOGCM"],
                    "additional_allowed_model_components": ["AER"],
                }
            },
            "forcing": ["GHG", "NAT"],
            "frequency": {
                "mon": {
                    "approx_interval": 30.0,
                    "approx_interval_error": 0.2,
                    "approx_interval_warning": 0.1,
                }
            },
            "institution_id": {"PCMDI": "Program for Climate Model Diagnosis"},
            "license": {
                "license_id": {
                    "CC-BY-4.0": {
                        "license_type": "Creative Commons Attribution 4.0",
                        "license_url": "https://example.test/license",
                    }
                },
                "license_template": (
                    "<license_id>; data from <institution_id> uses "
                    "<license_type> (<license_url>)"
                ),
            },
            "mip_era": "CMIP7",
            "profile": {"standard": {"contact": "support@example.test", "priority": 1}},
            "required_global_attributes": [
                *required,
            ],
            "source_id": {
                "MODEL-A": {
                    "institution_id": "PCMDI",
                    "source": "Model A",
                    "source_type": "AOGCM",
                }
            },
            "source_type": {"AER": "aerosol", "AOGCM": "coupled model"},
            "tracking_id": ["hdl:21.14100/.*"],
        }
    })


def test_generic_rules_match_literals_regexes_lookups_and_templates():
    cv = _sample_cv()
    metadata = {
        "activity_id": "ScenarioMIP",
        "frequency": "mon",
        "realm": "atmos",
    }

    activity = cv._rules["activity_id"]
    branding = cv._rules["branding_suffix"]

    assert isinstance(activity, AttributeRule)
    assert activity.accepts("CMIP", metadata)
    assert activity.accepts("ScenarioMIP", metadata)
    assert not activity.accepts("invalid", metadata)
    assert isinstance(branding, AttributeRule)
    assert branding.accepts("mon-atmos", metadata)


def test_specialized_components_are_selected_for_catalogs():
    cv = _sample_cv()

    assert not isinstance(cv, Mapping)
    assert isinstance(cv._license, LicenseComponent)
    assert isinstance(cv._sources, SourceComponent)
    assert isinstance(cv._experiments, ExperimentComponent)
    assert isinstance(cv._drs, DRSComponent)
    assert isinstance(cv._source_types, SourceTypeComponent)
    assert isinstance(cv._forcing, ForcingComponent)
    assert isinstance(cv._frequency, FrequencyComponent)
    assert isinstance(cv._tracking_id, TrackingIdComponent)
    assert isinstance(cv._variant_label, VariantLabelComponent)
    assert isinstance(cv._grid_label, GridLabelComponent)
    assert not hasattr(cv, "get")
    assert cv.required_attributes == (
        "Conventions",
        "activity_id",
        "source_id",
    )


def test_components_drive_defaults_and_templated_license():
    cv = _sample_cv()
    metadata = DatasetMetadata.from_mapping({
        "activity_id": "CMIP",
        "experiment_id": "historical",
        "license_id": "CC-BY-4.0",
        "profile": "standard",
        "source_id": "MODEL-A",
    })

    prepared = cv.get_dataset_info(metadata).to_dict()

    assert prepared["institution_id"] == "PCMDI"
    assert prepared["source"] == "Model A"
    assert prepared["source_type"] == "AOGCM"
    assert prepared["institution"] == "Program for Climate Model Diagnosis"
    assert prepared["experiment"] == "historical simulation"
    assert prepared["mip_era"] == "CMIP7"
    assert prepared["Conventions"] == "CF-1.12"
    assert prepared["contact"] == "support@example.test"
    assert prepared["priority"] == 1
    assert prepared["license"] == (
        "CC-BY-4.0; data from PCMDI uses Creative Commons Attribution 4.0 "
        "(https://example.test/license)"
    )


def test_optional_derived_attributes_preserve_user_values():
    cv = _sample_cv()
    metadata = DatasetMetadata.from_mapping({
        "activity_id": "CMIP",
        "experiment": "user experiment description",
        "experiment_id": "historical",
        "institution": "user institution description",
        "institution_id": "PCMDI",
        "source": "user source description",
        "source_id": "MODEL-A",
        "source_type": "AOGCM",
    })

    prepared = cv.get_dataset_info(metadata)
    values = prepared.to_dict()

    assert values["source"] == "user source description"
    assert values["experiment"] == "user experiment description"
    assert values["institution"] == "user institution description"
    cv._validate_derived_attributes(prepared)
    cv._validate_source_attributes(prepared)
    cv._validate_experiment(prepared)


def test_empty_optional_derived_attributes_are_filled_from_cv():
    cv = _sample_cv()
    metadata = DatasetMetadata.from_mapping({
        "activity_id": "CMIP",
        "experiment": "",
        "experiment_id": "historical",
        "institution": "",
        "institution_id": "PCMDI",
        "source": "",
        "source_id": "MODEL-A",
    })

    prepared = cv.get_dataset_info(metadata).to_dict()

    assert prepared["source"] == "Model A"
    assert prepared["experiment"] == "historical simulation"
    assert prepared["institution"] == "Program for Climate Model Diagnosis"


def test_required_derived_attributes_must_match_id_defaults():
    cv = _sample_cv(
        required_global_attributes=[
            "activity_id",
            "source",
            "experiment",
            "institution",
        ]
    )
    valid = {
        "activity_id": "CMIP",
        "experiment": "historical simulation",
        "experiment_id": "historical",
        "institution": "Program for Climate Model Diagnosis",
        "institution_id": "PCMDI",
        "source": "Model A",
        "source_id": "MODEL-A",
        "source_type": "AOGCM",
    }

    cv.validate_dataset_info(cv.get_dataset_info(DatasetMetadata.from_mapping(valid)))
    for key in ("source", "experiment", "institution"):
        metadata = DatasetMetadata.from_mapping({
            **valid,
            key: f"user {key} description",
        })
        prepared = cv.get_dataset_info(metadata)

        try:
            cv.validate_dataset_info(prepared)
        except ControlledVocabularyError as error:
            assert f"{key}='user {key} description'" in str(error)
        else:
            raise AssertionError(f"required {key} should be validated")


def test_specialized_catalogs_are_used_by_existing_validation_api():
    cv = _sample_cv()
    valid = DatasetMetadata.from_mapping({
        "activity_id": "CMIP",
        "experiment_id": "historical",
        "institution_id": "PCMDI",
        "source_id": "MODEL-A",
        "source_type": "AOGCM",
    })
    invalid = valid.updated(source_type="AER")

    assert cv._sources is not None
    assert (
        cv._sources.validation_error(
            "MODEL-A", invalid.to_dict(), ignored_keys={"source"}
        )
        is not None
    )
    assert cv._experiments is not None
    assert (
        cv._experiments.validation_error(
            "historical",
            valid.updated(activity_id="ScenarioMIP").to_dict(),
            cv._source_types,
        )
        is not None
    )
    assert (
        cv._experiments.parent_validation_error(
            "historical",
            valid.updated(parent_experiment_id="piControl").to_dict(),
            cv._sources,
        )
        is not None
    )

    cv._validate_experiment(valid)
    cv._validate_source_attributes(valid)

    try:
        cv._validate_source_attributes(invalid)
    except ControlledVocabularyError as error:
        assert "source_id='MODEL-A'" in str(error)
    else:
        raise AssertionError("source-specific metadata should be validated")


def test_drs_and_default_components_can_be_used_directly():
    cv = _sample_cv()

    assert cv._drs is not None
    assert cv._drs.templates() == (
        "<mip_era><source_id>",
        "<variable_id><source_id>",
    )
    assert cv._defaults.scalar_defaults_for({})["mip_era"] == "CMIP7"
    assert cv._defaults.nested_defaults_for({"profile": "standard"}) == {
        "contact": "support@example.test",
        "priority": 1,
    }


def test_source_type_component_validates_experiment_composition():
    cv = _sample_cv()
    assert cv._experiments is not None
    experiment = cv._experiments.entry_for("historical")
    assert experiment is not None

    assert cv._source_types.accepts("AOGCM AER")
    assert not cv._source_types.accepts("AOGCM CHEM")
    assert cv._source_types.validation_error("AOGCM AER", experiment) is None
    assert "missing required" in str(
        cv._source_types.validation_error("AER", experiment)
    )
    assert "not allowed" in str(
        cv._source_types.validation_error("AOGCM CHEM", experiment)
    )


def test_frequency_forcing_and_tracking_components_are_typed():
    cv = _sample_cv()

    assert cv._frequency is not None
    interval = cv._frequency.interval_for("mon")
    assert interval is not None
    assert interval.approx_interval == 30.0

    assert cv._forcing is not None
    assert cv._forcing.invalid_token("GHG, NAT (natural and anthropogenic)") is None
    assert cv._forcing.invalid_token("GHG INVALID") == "INVALID"

    assert cv._tracking_id.format("abc") == "hdl:21.14100/abc"


@pytest.mark.parametrize(
    "indices",
    [
        (9, 2, 3, 4),
        ("9", "2", "3", "4"),
        ("r9", "i2", "p3", "f4"),
        (9, "i2", "3", "f4"),
    ],
)
def test_variant_component_assembles_indices_and_drives_defaults(indices):
    cv = ControlledVocabulary({
        "CV": {
            "required_global_attributes": ["variant_label"],
            "variant_label": [r"r\d+i\d+p\d+f\d+"],
        }
    })
    metadata = dict(
        zip(
            (
                "realization_index",
                "initialization_index",
                "physics_index",
                "forcing_index",
            ),
            indices,
            strict=True,
        )
    )

    assert cv._variant_label.render(metadata) == "r9i2p3f4"
    assert cv._variant_label.validation_error(metadata) is None
    prepared = cv.get_dataset_info(DatasetMetadata.from_mapping(metadata))
    assert prepared.variant_label_value == "r9i2p3f4"
    cv._validate_variant_indices(prepared)


@pytest.mark.parametrize(
    "key,prefix",
    [
        ("realization_index", "r"),
        ("initialization_index", "i"),
        ("physics_index", "p"),
        ("forcing_index", "f"),
    ],
)
def test_variant_component_validates_each_index_without_a_label_constraint(key, prefix):
    component = VariantLabelComponent.from_definition(None)

    for value in (1, str(2**31 - 1), f"{prefix}1", f"{prefix}{2**31 - 1}"):
        assert component.validation_error({key: value}) is None
    for value in (0, -1, 2**31, f"{prefix}0", f"{prefix}{2**31}", "x1", "1.5"):
        error = component.validation_error({key: value, "variant_label": "r1i1p1f1"})
        assert error is not None
        assert key in error
    for value in (None, "", r"^[[:digit:]]{1,}$"):
        assert component.validation_error({key: value}) is None


@pytest.mark.parametrize("definition", [None, [], "custom", "<member_id>"])
def test_variant_component_preserves_unconstrained_explicit_labels(definition):
    component = VariantLabelComponent.from_definition(definition)
    metadata = {"variant_label": "custom", "realization_index": "r2"}

    assert component.render(metadata) == "custom"
    assert component.validation_error(metadata) is None
    assert component.render({}) is None
    assert component.render({"realization_index": "r2"}) is None


def test_variant_component_reports_explicit_and_assembled_format_errors():
    cv = ControlledVocabulary({"variant_label": [r"r\d+i\d+p\d+f\d+"]})
    assert cv._variant_label.validation_error({"variant_label": "r1i1p1f1"}) is None
    invalid = {"variant_label": "r1i1p1"}
    error = cv._variant_label.validation_error(invalid)
    assert error is not None
    assert "required pattern" in error
    with pytest.raises(ControlledVocabularyError, match="required pattern"):
        cv._validate_variant_indices(DatasetMetadata.from_mapping(invalid))

    # Legacy regex placeholders skip integer validation but cannot form a label.
    error = cv._variant_label.validation_error({
        "realization_index": r"^[[:digit:]]{1,}$",
        "initialization_index": 1,
        "physics_index": 1,
        "forcing_index": 1,
    })
    assert error is not None
    assert "assembled from RIPF indices" in error


@pytest.mark.parametrize("label", [None, "", "gn", "gr1", "cn", "rn", "g999", "g"])
def test_grid_component_accepts_standard_or_missing_labels(label):
    component = GridLabelComponent()
    assert component.validation_error({"grid_label": label}) is None
    assert component.validation_error({}) is None


@pytest.mark.parametrize("label", ["gr-0", "GN", "1gn", "g n", "gn\n", "other"])
def test_grid_component_drives_fallback_validation(label):
    cv = ControlledVocabulary({})
    metadata = {"grid_label": label}
    error = cv._grid_label.validation_error(metadata)

    assert error is not None
    assert "does not match the required format" in error
    with pytest.raises(ControlledVocabularyError) as caught:
        cv._validate_dataset_values(DatasetMetadata.from_mapping(metadata))
    assert str(caught.value) == error


@pytest.mark.parametrize(
    "definition",
    [
        {"custom-label": "custom grid"},
        ["custom-label"],
        ["custom-.*"],
        "custom-label",
        "<grid>",
        None,
    ],
)
def test_grid_component_defers_to_explicit_project_definitions(definition):
    cv = ControlledVocabulary({"grid_label": definition})
    metadata = {"grid_label": "custom-label", "grid": "custom-label"}

    assert cv._grid_label.rule is not None
    assert cv._grid_label.validation_error(metadata) is None
    cv._validate_dataset_values(DatasetMetadata.from_mapping(metadata))


def test_grid_component_preserves_explicit_empty_choice_constraint():
    cv = ControlledVocabulary({"grid_label": []})
    assert cv._grid_label.validation_error({"grid_label": "gn"}) is None
    with pytest.raises(ControlledVocabularyError, match="is not allowed"):
        cv._validate_dataset_values(DatasetMetadata.from_mapping({"grid_label": "gn"}))


def test_dataset_info_validation_can_defer_required_globals():
    cv = ControlledVocabulary({
        "required_global_attributes": ["variable_id"],
        "activity_id": ["CMIP"],
    })
    metadata = DatasetMetadata.from_mapping({"activity_id": "CMIP"})

    assert cv.validate_dataset_info(metadata, check_required=False) is None
    with pytest.raises(ControlledVocabularyError, match="variable_id"):
        cv.validate_dataset_info(metadata)
    assert cv.validate_dataset_info(metadata.updated(variable_id="tas")) is None
    with pytest.raises(ControlledVocabularyError, match="activity_id"):
        cv.validate_dataset_info(
            metadata.updated(activity_id="invalid"), check_required=False
        )


@pytest.mark.parametrize("check_required", [True, False])
@pytest.mark.parametrize(
    "definition,metadata,message",
    [
        (
            {"source_id": {"model": {"institution_id": "expected"}}},
            {"source_id": "model", "institution_id": "different"},
            "does not match source_id",
        ),
        (
            {"experiment_id": {"historical": {"activity_id": ["CMIP"]}}},
            {"experiment_id": "historical", "activity_id": "other"},
            "does not match experiment_id",
        ),
        (
            {"experiment_id": {"historical": {"required_source_type": ["AOGCM"]}}},
            {"experiment_id": "historical"},
            "source_type is required",
        ),
        (
            {"experiment_id": {"historical": {"parent_experiment_id": ["piControl"]}}},
            {"experiment_id": "historical"},
            "requires parent_experiment_id",
        ),
        (
            {"experiment_id": {"amip": {}}},
            {"experiment_id": "amip", "parent_experiment_id": "piControl"},
            "does not allow parent_experiment_id",
        ),
        (
            {"forcing": ["GHG"]},
            {"forcing": "UNKNOWN"},
            "forcing term",
        ),
        (
            {
                "variant_label": [
                    r"^r[[:digit:]]+i[[:digit:]]+p[[:digit:]]+f[[:digit:]]+$"
                ]
            },
            {"variant_label": "r1i1p1"},
            "required pattern",
        ),
        ({}, {"grid_label": "gr-0"}, "required format"),
        (
            {
                "required_global_attributes": ["institution"],
                "institution_id": {"PCMDI": "expected"},
            },
            {"institution_id": "PCMDI", "institution": "different"},
            "does not match the value selected",
        ),
    ],
)
def test_dataset_info_validation_runs_all_constraints(
    definition, metadata, message, check_required
):
    cv = ControlledVocabulary(definition)
    with pytest.raises(ControlledVocabularyError, match=message):
        cv.validate_dataset_info(
            DatasetMetadata.from_mapping(metadata), check_required=check_required
        )
