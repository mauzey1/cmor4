from __future__ import annotations

from collections.abc import Mapping

from cmor4.cv import ControlledVocabulary
from cmor4.exceptions import ControlledVocabularyError
from cmor4.utils.cv_models import (
    AttributeRule,
    DRSComponent,
    ExperimentComponent,
    ForcingComponent,
    FrequencyComponent,
    LicenseComponent,
    SourceComponent,
    SourceTypeComponent,
    TrackingIdComponent,
)
from cmor4.utils.dataset_metadata import DatasetMetadata


def _sample_cv() -> ControlledVocabulary:
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
            "profile": {
                "standard": {"contact": "support@example.test", "priority": 1}
            },
            "required_global_attributes": [
                "Conventions",
                "activity_id",
                "source_id",
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

    activity = cv.rules["activity_id"]
    branding = cv.rules["branding_suffix"]

    assert isinstance(activity, AttributeRule)
    assert activity.accepts("CMIP", metadata)
    assert activity.accepts("ScenarioMIP", metadata)
    assert not activity.accepts("invalid", metadata)
    assert isinstance(branding, AttributeRule)
    assert branding.accepts("mon-atmos", metadata)


def test_specialized_components_are_selected_for_catalogs():
    cv = _sample_cv()

    assert not isinstance(cv, Mapping)
    assert isinstance(cv.license, LicenseComponent)
    assert isinstance(cv.sources, SourceComponent)
    assert isinstance(cv.experiments, ExperimentComponent)
    assert isinstance(cv.drs, DRSComponent)
    assert isinstance(cv.source_types, SourceTypeComponent)
    assert isinstance(cv.forcing, ForcingComponent)
    assert isinstance(cv.frequency, FrequencyComponent)
    assert isinstance(cv.tracking_id, TrackingIdComponent)
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


def test_specialized_catalogs_are_used_by_existing_validation_api():
    cv = _sample_cv()
    valid = DatasetMetadata.from_mapping({
        "activity_id": "CMIP",
        "experiment_id": "historical",
        "institution_id": "PCMDI",
        "source_id": "MODEL-A",
        "source_type": "AOGCM",
    })
    invalid = valid.updated(source="A different model")

    assert cv.sources is not None
    assert cv.sources.validation_error("MODEL-A", invalid.to_dict()) is not None
    assert cv.experiments is not None
    assert (
        cv.experiments.validation_error(
            "historical",
            valid.updated(activity_id="ScenarioMIP").to_dict(),
            cv.source_types,
        )
        is not None
    )
    assert (
        cv.experiments.parent_validation_error(
            "historical",
            valid.updated(parent_experiment_id="piControl").to_dict(),
            cv.sources,
        )
        is not None
    )

    cv.validate_experiment(valid)
    cv.validate_source_attributes(valid)

    try:
        cv.validate_source_attributes(invalid)
    except ControlledVocabularyError as error:
        assert "source_id='MODEL-A'" in str(error)
    else:
        raise AssertionError("source-specific metadata should be validated")


def test_drs_and_default_components_can_be_used_directly():
    cv = _sample_cv()

    assert cv.drs is not None
    assert cv.drs.templates() == (
        "<mip_era><source_id>",
        "<variable_id><source_id>",
    )
    assert cv.defaults.scalar_defaults_for({})["mip_era"] == "CMIP7"
    assert cv.defaults.nested_defaults_for({"profile": "standard"}) == {
        "contact": "support@example.test",
        "priority": 1,
    }


def test_source_type_component_validates_experiment_composition():
    cv = _sample_cv()
    assert cv.experiments is not None
    experiment = cv.experiments.entry_for("historical")
    assert experiment is not None

    assert cv.source_types.accepts("AOGCM AER")
    assert not cv.source_types.accepts("AOGCM CHEM")
    assert cv.source_types.validation_error("AOGCM AER", experiment) is None
    assert "missing required" in str(
        cv.source_types.validation_error("AER", experiment)
    )
    assert "not allowed" in str(
        cv.source_types.validation_error("AOGCM CHEM", experiment)
    )


def test_frequency_forcing_and_tracking_components_are_typed():
    cv = _sample_cv()

    assert cv.frequency is not None
    interval = cv.frequency.interval_for("mon")
    assert interval is not None
    assert interval.approx_interval == 30.0

    assert cv.forcing is not None
    assert cv.forcing.invalid_token("GHG, NAT (natural and anthropogenic)") is None
    assert cv.forcing.invalid_token("GHG INVALID") == "INVALID"

    assert cv.tracking_id.format("abc") == "hdl:21.14100/abc"
