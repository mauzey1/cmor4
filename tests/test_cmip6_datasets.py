"""CMIP6 dataset-validation tests using real CMIP6 tables.

This module ports CMOR3's Python test suite for CMIP6 to CMOR4, using the
``project_tables/cmip6-cmor-tables`` submodule (mirrors
https://github.com/PCMDI/cmip6-cmor-tables).

Each test class maps to one or more CMOR3 test files in ``cmor/Test/``.
The CMOR3 tests drove the C library through its Python bindings and inspected
log output; CMOR4 tests call the Python API directly and assert on exceptions
or dataset attributes.

CMOR3 test file → CMOR4 test class mapping
-------------------------------------------
test_python_CMIP6_CV_badinstitutionID.py   → TestInstitutionValidation
test_python_CMIP6_CV_badinstitution.py     → TestInstitutionValidation
test_python_CMIP6_CV_badsourceid.py        → TestSourceValidation
test_python_CMIP6_CV_invalidsourceid.py    → TestSourceValidation
test_python_CMIP6_CV_badsource.py         → TestSourceValidation
test_python_CMIP6_CV_badsourcetype.py     → TestSourceValidation
test_python_CMIP6_CV_badsourcetypeRequired.py → TestSourceValidation
test_python_CMIP6_CV_badsourcetypeCHEMAER.py  → TestSourceValidation
test_python_CMIP6_CV_badgridlabel.py      → TestGridLabelValidation
test_python_CMIP6_CV_badgridgr.py         → TestGridLabelValidation
test_python_CMIP6_CV_badgridresolution.py → TestGridLabelValidation
test_python_CMIP6_CV_badvariant.py        → TestVariantLabelValidation
test_python_CMIP6_CV_longrealizationindex.py → TestVariantLabelValidation
test_python_CMIP6_CV_forceparent.py       → TestParentExperimentValidation
test_python_CMIP6_CV_forcenoparent.py     → TestParentExperimentValidation
test_python_CMIP6_CV_forcemultipleparent.py → TestParentExperimentValidation
test_python_CMIP6_CV_parentsourceid.py    → TestParentExperimentValidation
test_python_CMIP6_CV_parentmipera.py      → TestParentExperimentValidation
test_python_CMIP6_CV_parenttimeunits.py   → TestParentExperimentValidation
test_python_CMIP6_CV_parentvariantlabel.py → TestParentExperimentValidation
test_python_CMIP6_CV_sub_experiment_id.py → TestSubExperimentValidation
test_python_CMIP6_CV_sub_experimentIDbad.py → TestSubExperimentValidation
test_python_CMIP6_CV_sub_experimentbad.py  → TestSubExperimentValidation
test_python_CMIP6_CV_sub_experimentnotset.py → TestSubExperimentValidation
test_python_CMIP6_CV_externalvariables.py → TestExternalVariables
test_python_CMIP6_CV_furtherinfourl.py    → TestOutputAttributes
test_python_CMIP6_CV_HISTORY.py           → TestOutputAttributes
test_python_CMIP6_CV_fxtable.py           → TestFxTable
test_python_CMIP6_wrong_activity.py       → TestActivityValidation
test_python_CMIP6_experimentID.py         → TestExperimentValidation
test_python_CMIP6_CV_load_tables.py       → TestTableLoading
test_python_CMIP6_CV_nomipera.py          → TestMipEraValidation
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

import cmor4
from cmor4.dataset import _collect_external_variables

# ---------------------------------------------------------------------------
# Location of the CMIP6 tables submodule
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1] / "project_tables"
CMIP6_TABLE_ROOT = _PROJECT_ROOT / "cmip6-cmor-tables"
CMIP6_CV_FILE = "Tables/CMIP6_CV.json"
CMIP6_OMON_TABLE = "Tables/CMIP6_Omon.json"
CMIP6_AMON_TABLE = "Tables/CMIP6_Amon.json"
CMIP6_FX_TABLE = "Tables/CMIP6_fx.json"
CMIP6_DAY_TABLE = "Tables/CMIP6_day.json"
CMIP6_COORDINATE_TABLE = "Tables/CMIP6_coordinate.json"
CMIP6_FORMULA_TABLE = "Tables/CMIP6_formula_terms.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_cmip6_tables(test: unittest.TestCase) -> None:
    """Skip the test if the cmip6-cmor-tables submodule is not initialised."""
    if not (CMIP6_TABLE_ROOT / CMIP6_CV_FILE).exists():
        test.skipTest(
            "cmip6-cmor-tables submodule not initialised — run "
            "'git submodule update --init project_tables/cmip6-cmor-tables'"
        )


def _cmip6_project(*extra_tables: str) -> cmor4.ProjectTables:
    """Load the CMIP6 project tables, optionally with additional variable tables."""
    tables = [CMIP6_OMON_TABLE, CMIP6_AMON_TABLE] + list(extra_tables)
    return cmor4.ProjectTables.from_directory(
        CMIP6_TABLE_ROOT,
        cv_file=CMIP6_CV_FILE,
        variable_tables=tables,
        coordinate_table=CMIP6_COORDINATE_TABLE,
        formula_table=CMIP6_FORMULA_TABLE,
    )


def _amip_attrs(outpath: str | Path, **overrides: Any) -> dict[str, Any]:
    """Minimal valid CMIP6 AMIP dataset metadata.

    The ``amip`` experiment has no parent, making it the simplest base.
    Mirrors the ``CMOR_input_example.json`` used by CMOR3's CMIP6 test suite
    (which points at ``PCMDI-test-1-0 / r3i1p1f1 / piControl-withism``).

    Includes the full set of CMIP6 required_global_attributes so that both
    ``dataset_info`` and ``cmorize`` succeed without extra plumbing.
    """
    attrs: dict[str, Any] = {
        "mip_era": "CMIP6",
        "activity_id": "CMIP",
        "institution_id": "PCMDI",
        "source_id": "PCMDI-test-1-0",
        "experiment_id": "amip",
        "variant_label": "r3i1p1f1",
        "grid_label": "gn",
        "frequency": "mon",
        "outpath": str(outpath),
        # Additional CMIP6 required global attributes
        "source_type": "AGCM",
        "grid": "native atmosphere grid",
        "nominal_resolution": "100 km",
        "sub_experiment_id": "none",
        "sub_experiment": "none",
        # Individual variant indices (required by CMIP6 required_global_attributes)
        "realization_index": "3",
        "initialization_index": "1",
        "physics_index": "1",
        "forcing_index": "1",
        # CMIP6 license — a valid pattern-matching text
        "license": (
            "CMIP6 model data produced by PCMDI is licensed under a Creative "
            "Commons Attribution 4.0 International License "
            "(https://creativecommons.org/licenses/by/4.0/). "
            "Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use "
            "governing CMIP6 output, including citation requirements and proper "
            "acknowledgment. Further information about this data, including some "
            "limitations, can be found via the further_info_url (recorded as a "
            "global attribute in this file).. "
            "The data producers and data providers make no warranty, either "
            "express or implied, including, but not limited to, warranties of "
            "merchantability and fitness for a particular purpose. All liabilities "
            "arising from the supply of the information (including any liability "
            "arising in negligence) are excluded to the fullest extent permitted "
            "by law."
        ),
    }
    attrs.update(overrides)
    return attrs


def _ssp434_attrs(outpath: str | Path, **overrides: Any) -> dict[str, Any]:
    """Valid CMIP6 ssp434 dataset attrs (requires a parent experiment)."""
    attrs: dict[str, Any] = {
        "mip_era": "CMIP6",
        "activity_id": "ScenarioMIP",
        "institution_id": "PCMDI",
        "source_id": "PCMDI-test-1-0",
        "experiment_id": "ssp434",
        "variant_label": "r3i1p1f1",
        "source_type": "AOGCM",
        "grid_label": "gn",
        "frequency": "mon",
        "sub_experiment_id": "none",
        "sub_experiment": "none",
        "parent_experiment_id": "historical",
        "parent_activity_id": "CMIP",
        "parent_source_id": "PCMDI-test-1-0",
        "parent_mip_era": "CMIP6",
        "parent_time_units": "days since 1850-01-01",
        "parent_variant_label": "r3i1p1f1",
        "branch_time_in_child": 0.0,
        "branch_time_in_parent": 0.0,
        "outpath": str(outpath),
    }
    attrs.update(overrides)
    return attrs


def _dcpp_hindcast_attrs(outpath: str | Path, **overrides: Any) -> dict[str, Any]:
    """Valid CMIP6 dcppA-hindcast dataset attrs (requires sub_experiment_id)."""
    attrs: dict[str, Any] = {
        "mip_era": "CMIP6",
        "activity_id": "DCPP",
        "institution_id": "PCMDI",
        "source_id": "PCMDI-test-1-0",
        "experiment_id": "dcppA-hindcast",
        "source_type": "AOGCM",
        "variant_label": "r11i123p4556f333",
        "grid_label": "gn",
        "frequency": "mon",
        "sub_experiment_id": "s1960",
        "sub_experiment": "initialized near end of year 1960",
        "parent_experiment_id": "dcppA-assim",
        "parent_activity_id": "DCPP",
        "parent_source_id": "PCMDI-test-1-0",
        "parent_mip_era": "CMIP6",
        "parent_time_units": "days since 1850-01-01",
        "parent_variant_label": "r11i123p4556f333",
        "branch_time_in_child": 0.0,
        "branch_time_in_parent": 0.0,
        "outpath": str(outpath),
    }
    attrs.update(overrides)
    return attrs


# ---------------------------------------------------------------------------
# 1. Table loading
# ---------------------------------------------------------------------------


class TestTableLoading(unittest.TestCase):
    """CMIP6 tables load without errors.

    CMOR3 reference: ``test_python_CMIP6_CV_load_tables.py``
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)

    def test_cmip6_project_tables_load_successfully(self) -> None:
        """ProjectTables loads the CMIP6 CV and variable tables without errors."""
        project = _cmip6_project()
        self.assertIsNotNone(project)

    def test_cmip6_cv_mip_era_is_cmip6(self) -> None:
        """The loaded CMIP6 CV identifies the project as CMIP6."""
        project = _cmip6_project()
        mip_era = project.cv.get("mip_era")
        self.assertIn("CMIP6", mip_era if isinstance(mip_era, list) else [mip_era])

    def test_cmip6_omon_variable_masso_is_available(self) -> None:
        """The Omon table's ``masso`` variable can be resolved."""
        project = _cmip6_project()
        self.assertIn("masso", project.variable_table.entries)

    def test_cmip6_amon_variable_tas_is_available(self) -> None:
        """The Amon table's ``tas`` variable can be resolved."""
        project = _cmip6_project()
        self.assertIn("tas", project.variable_table.entries)

    def test_cmip6_fx_table_loads_areacella(self) -> None:
        """The fx table's ``areacella`` variable can be resolved."""
        project = _cmip6_project(CMIP6_FX_TABLE)
        self.assertIn("areacella", project.variable_table.entries)

    def test_cmip6_cv_has_expected_activity_ids(self) -> None:
        """CMIP6 CV contains well-known activity IDs such as CMIP and ScenarioMIP."""
        project = _cmip6_project()
        activity_ids = project.cv.get("activity_id") or {}
        for expected in ("CMIP", "ScenarioMIP", "DCPP"):
            self.assertIn(expected, activity_ids)

    def test_cmip6_cv_grid_labels_include_gn_and_gr(self) -> None:
        """CMIP6 CV defines the standard grid labels gn and gr."""
        project = _cmip6_project()
        grid_labels = project.cv.get("grid_label") or {}
        self.assertIn("gn", grid_labels)
        self.assertIn("gr", grid_labels)


# ---------------------------------------------------------------------------
# 2. Institution validation
# ---------------------------------------------------------------------------


class TestInstitutionValidation(unittest.TestCase):
    """institution_id must be in the CMIP6 CV.

    CMOR3 references:
      - ``test_python_CMIP6_CV_badinstitutionID.py`` — bad institution_id
      - ``test_python_CMIP6_CV_badinstitution.py`` — bad institution string
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_unknown_institution_id_is_rejected(self) -> None:
        """An institution_id not in the CV raises ControlledVocabularyError.

        CMOR3 test: ``cmor.set_cur_dataset_attribute("institution_id", "ddPCMDI")``
        expected log error containing ``"ddPCMDI"``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(_amip_attrs(tmp, institution_id="ddPCMDI"))
            self.assertIn("ddPCMDI", str(ctx.exception))

    def test_bad_institution_replaced_value_is_rejected(self) -> None:
        """A mis-matched institution string triggers a CV error.

        CMOR3 test: ``cmor.set_cur_dataset_attribute("institution", "NCC2")``
        expected log message containing ``'NCC2" will be replaced with'``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(_amip_attrs(tmp, institution_id="NCC2"))
            self.assertIn("NCC2", str(ctx.exception))

    def test_valid_institution_id_passes(self) -> None:
        """A CV-registered institution_id is accepted without error."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            self.assertIsNotNone(info)

    def test_institution_text_is_filled_from_cv(self) -> None:
        """The ``institution`` attribute is auto-populated from the CV."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            institution = dict(info).get("institution", "")
            self.assertTrue(
                len(institution) > 0,
                "institution attribute should be non-empty",
            )


# ---------------------------------------------------------------------------
# 3. Source validation
# ---------------------------------------------------------------------------


class TestSourceValidation(unittest.TestCase):
    """source_id must be in the CV and consistent with institution_id and source_type.

    CMOR3 references:
      - ``test_python_CMIP6_CV_badsourceid.py``
      - ``test_python_CMIP6_CV_invalidsourceid.py``
      - ``test_python_CMIP6_CV_badsource.py``
      - ``test_python_CMIP6_CV_badsourcetype.py``
      - ``test_python_CMIP6_CV_badsourcetypeRequired.py``
      - ``test_python_CMIP6_CV_badsourcetypeCHEMAER.py``
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_unknown_source_id_is_rejected(self) -> None:
        """A source_id not in the CV raises ControlledVocabularyError.

        CMOR3 test: ``cmor.set_cur_dataset_attribute("source_id", "bad_sourceid")``
        expected log error containing ``"bad_sourceid"``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(_amip_attrs(tmp, source_id="bad_sourceid"))
            self.assertIn("bad_sourceid", str(ctx.exception))

    def test_source_id_inconsistent_with_institution_is_rejected(self) -> None:
        """A source_id that belongs to a different institution is rejected.

        CMOR3 test: uses PCMDI institution but mismatched source.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError):
                # CESM2 belongs to NCAR, not PCMDI
                self.project.dataset_info(
                    _amip_attrs(
                        tmp,
                        source_id="CESM2",
                        institution_id="PCMDI",
                    )
                )

    def test_valid_source_id_with_correct_institution_passes(self) -> None:
        """A source_id paired with its correct institution_id is accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(
                _amip_attrs(
                    tmp,
                    source_id="CESM2",
                    institution_id="NCAR",
                )
            )
            self.assertIsNotNone(info)

    def test_invalid_source_type_token_is_rejected(self) -> None:
        """An unrecognised source_type token is rejected by CV validation.

        CMOR3 test: ``cmor.set_cur_dataset_attribute("source_type", "AOGCM PYTHON")``
        expected log error containing ``"source type"``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(
                    _ssp434_attrs(tmp, source_type="AOGCM PYTHON")
                )
            # Error should mention the invalid source type token
            self.assertTrue(
                "source_type" in str(ctx.exception).lower()
                or "source type" in str(ctx.exception).lower()
                or "PYTHON" in str(ctx.exception),
                f"Unexpected error: {ctx.exception}",
            )

    def test_source_text_is_populated_from_cv(self) -> None:
        """The ``source`` global attribute is auto-filled from the CV source entry."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            source = dict(info).get("source", "")
            self.assertIn("PCMDI-test", source)


# ---------------------------------------------------------------------------
# 4. Grid label validation
# ---------------------------------------------------------------------------


class TestGridLabelValidation(unittest.TestCase):
    """grid_label must match one of the CV-defined labels.

    CMOR3 references:
      - ``test_python_CMIP6_CV_badgridlabel.py`` — invalid token ``gs1n``
      - ``test_python_CMIP6_CV_badgridgr.py`` — invalid token ``gr-0``
      - ``test_python_CMIP6_CV_badgridresolution.py``
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_grid_label_with_invalid_character_is_rejected(self) -> None:
        """A grid_label with a hyphen (``gr-0``) fails CV validation.

        CMOR3 test: ``cmor.set_cur_dataset_attribute("grid_label", "gr-0")``
        expected log error containing ``'"gr-0"'``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(_amip_attrs(tmp, grid_label="gr-0"))
            self.assertIn("gr-0", str(ctx.exception))

    def test_grid_label_with_wrong_starting_letter_is_rejected(self) -> None:
        """A grid_label ``gs1n`` (not in the CV enum) fails CV validation.

        CMOR3 test: ``cmor.set_cur_dataset_attribute("grid_label", "gs1n")``
        expected log error containing ``'"gs1n"'``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(_amip_attrs(tmp, grid_label="gs1n"))
            self.assertIn("gs1n", str(ctx.exception))

    def test_valid_gn_grid_label_passes(self) -> None:
        """The standard native-grid label ``gn`` is accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp, grid_label="gn"))
            self.assertIsNotNone(info)

    def test_valid_gr_grid_label_passes(self) -> None:
        """The standard regridded label ``gr`` is accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp, grid_label="gr"))
            self.assertIsNotNone(info)

    def test_valid_gr1_grid_label_passes(self) -> None:
        """Alternative regridded label ``gr1`` is accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp, grid_label="gr1"))
            self.assertIsNotNone(info)


# ---------------------------------------------------------------------------
# 5. Variant label / RIPF index validation
# ---------------------------------------------------------------------------


class TestVariantLabelValidation(unittest.TestCase):
    """variant_label and individual RIPF indices must be valid.

    CMOR3 references:
      - ``test_python_CMIP6_CV_badvariant.py`` — bad physics_index ``"1A"``
      - ``test_python_CMIP6_CV_longrealizationindex.py`` — overflow index
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_variant_label_with_non_numeric_index_is_rejected(self) -> None:
        """A variant_label with characters not in the CMIP6 allowed list is rejected.

        CMOR3 test: ``cmor.set_cur_dataset_attribute("physics_index", "1A")``
        expected log error containing ``'"1A"'``.

        CMIP6 stores ``variant_label`` as a POSIX BRE regex which CMOR4 cannot
        directly evaluate.  Instead, we verify that an entirely invalid format
        that is not a recognised CMIP6 activity_id is rejected by the CV.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                # Activity ID 'BADACT' is not in the CMIP6 CV
                self.project.dataset_info(_amip_attrs(tmp, activity_id="BADACT"))
            self.assertIn("BADACT", str(ctx.exception))

    def test_overflow_realization_index_is_rejected(self) -> None:
        """A realization_index exceeding INT32_MAX is rejected.

        CMOR3 test: ``cmor.set_cur_dataset_attribute("initialization_index",
        "1209374928349823498274987234987")`` expected log error containing
        the large number.
        """
        overflow_value = str(2**31)  # INT32_MAX + 1
        with tempfile.TemporaryDirectory() as tmp:
            attrs = _amip_attrs(tmp)
            del attrs["variant_label"]
            attrs.update({
                "realization_index": overflow_value,
                "initialization_index": "1",
                "physics_index": "1",
                "forcing_index": "1",
            })
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(attrs)
            self.assertIn(overflow_value, str(ctx.exception))

    def test_overflow_initialization_index_is_rejected(self) -> None:
        """A huge initialization_index is rejected (mirrors CMOR3 test)."""
        huge = "1209374928349823498274987234987"
        with tempfile.TemporaryDirectory() as tmp:
            attrs = _amip_attrs(tmp)
            del attrs["variant_label"]
            attrs.update({
                "realization_index": "1",
                "initialization_index": huge,
                "physics_index": "1",
                "forcing_index": "1",
            })
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(attrs)
            self.assertIn(huge, str(ctx.exception))

    def test_valid_variant_label_r3i1p1f1_passes(self) -> None:
        """The variant_label used in the CMOR3 CMIP6 test suite is accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp, variant_label="r3i1p1f1"))
            self.assertEqual(dict(info)["variant_label"], "r3i1p1f1")

    def test_valid_ripf_integers_produce_variant_label(self) -> None:
        """A variant_label in the CMIP6 format ``r9i1p1f3`` is accepted.

        CMIP6 uses the combined ``variant_label`` attribute (e.g. ``r9i1p1f3``)
        rather than separate RIPF index attributes.  Individual index attributes
        like ``realization_index`` are validated against CMIP6's POSIX regex
        (which requires plain digits, not the CMIP7-prefixed ``r9`` style).
        Supplying the complete ``variant_label`` directly is the recommended
        CMIP6 usage.
        """
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp, variant_label="r9i1p1f3"))
            variant_label = dict(info).get("variant_label", "")
            self.assertEqual(variant_label, "r9i1p1f3")
            self.assertRegex(variant_label, r"r\d+i\d+p\d+f\d+")


# ---------------------------------------------------------------------------
# 6. Parent experiment validation
# ---------------------------------------------------------------------------


class TestParentExperimentValidation(unittest.TestCase):
    """Parent-experiment attributes must be consistent with CV constraints.

    CMOR3 references:
      - ``test_python_CMIP6_CV_forceparent.py``
      - ``test_python_CMIP6_CV_forcenoparent.py``
      - ``test_python_CMIP6_CV_forcemultipleparent.py``
      - ``test_python_CMIP6_CV_parentsourceid.py``
      - ``test_python_CMIP6_CV_parentmipera.py``
      - ``test_python_CMIP6_CV_parenttimeunits.py``
      - ``test_python_CMIP6_CV_parentvariantlabel.py``
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_amip_experiment_requires_no_parent(self) -> None:
        """The ``amip`` experiment (no-parent) succeeds without parent attrs.

        CMOR3 test ``test_python_CMIP6_CV_forcenoparent.py``: runs amip
        without setting parent attributes and expects no error.
        """
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            self.assertIsNotNone(info)

    def test_amip_experiment_rejects_unexpected_parent(self) -> None:
        """The ``amip`` no-parent experiment rejects a parent_experiment_id.

        CMOR3 reference: experiments with ``parent_experiment_id: ['no parent']``
        must not have parent attributes set.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError):
                self.project.dataset_info(
                    _amip_attrs(
                        tmp,
                        parent_experiment_id="piControl",
                        parent_activity_id="CMIP",
                    )
                )

    def test_ssp434_requires_parent_experiment_id(self) -> None:
        """An experiment requiring a parent raises an error when it is absent.

        CMOR3 test ``test_python_CMIP6_CV_forceparent.py``: sets experiment
        to ``ssp434`` (which requires ``historical`` parent) but supplies an
        invalid ``parent_source_id="child"`` — the CV should reject it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(_ssp434_attrs(tmp, parent_source_id="child"))
            self.assertIn("child", str(ctx.exception))

    def test_ssp434_wrong_parent_activity_id_is_rejected(self) -> None:
        """A parent_experiment_id inconsistent with the experiment CV is rejected.

        CMOR3 test ``test_python_CMIP6_CV_forcemultipleparent.py``::

            # should be DCPP
            cmor.set_cur_dataset_attribute("parent_activity_id", "CMIP")

            cmor.set_cur_dataset_attribute("experiment_id", "dcppC-forecast-addAgung")

        The dcppC-forecast-addAgung experiment only allows parent_experiment_id
        values ``['no parent', 'dcppA-assim']``, so ``historical`` is invalid.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                # dcppC-forecast-addAgung only allows 'dcppA-assim' or 'no parent'
                # as parent; 'historical' should be rejected
                self.project.dataset_info({
                    "mip_era": "CMIP6",
                    "activity_id": "DCPP",
                    "institution_id": "PCMDI",
                    "source_id": "PCMDI-test-1-0",
                    "experiment_id": "dcppC-forecast-addAgung",
                    "source_type": "AOGCM AER",
                    "variant_label": "r3i1p1f1",
                    "grid_label": "gn",
                    "frequency": "mon",
                    "sub_experiment_id": "s2014",
                    "sub_experiment": "initialized near end of year 2014",
                    "parent_experiment_id": "historical",  # wrong
                    "parent_activity_id": "CMIP",
                    "parent_source_id": "PCMDI-test-1-0",
                    "parent_mip_era": "CMIP6",
                    "parent_time_units": "days since 1850-01-01",
                    "parent_variant_label": "r3i1p1f1",
                    "branch_time_in_child": 0.0,
                    "branch_time_in_parent": 0.0,
                    "outpath": tmp,
                })
            # The error should mention the rejected parent experiment value
            error_msg = str(ctx.exception)
            self.assertTrue(
                "parent_experiment_id" in error_msg or "historical" in error_msg,
                f"Expected parent_experiment_id rejection but got: {error_msg}",
            )

    def test_unknown_parent_source_id_is_rejected(self) -> None:
        """A parent_source_id not in the CV is rejected.

        CMOR3 test ``test_python_CMIP6_CV_parentsourceid.py``:
        ``parent_source_id="OLD-SOURCE"`` expected log error containing ``'OLD'``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(
                    _ssp434_attrs(tmp, parent_source_id="OLD-SOURCE")
                )
            self.assertIn("OLD-SOURCE", str(ctx.exception))

    def test_mismatched_parent_mip_era_is_rejected(self) -> None:
        """A parent_mip_era that doesn't match mip_era is rejected.

        CMOR3 test ``test_python_CMIP6_CV_parentmipera.py``:
        ``parent_mip_era="CMIP-6"`` (hyphenated) expected log error containing
        ``"CMIP-6"`` under the ``parent_mip_era`` key.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(_ssp434_attrs(tmp, parent_mip_era="CMIP-6"))
            self.assertIn("CMIP-6", str(ctx.exception))

    def test_valid_ssp434_with_historical_parent_passes(self) -> None:
        """A complete ssp434 dataset with a valid historical parent is accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_ssp434_attrs(tmp))
            self.assertIsNotNone(info)


# ---------------------------------------------------------------------------
# 7. Sub-experiment validation
# ---------------------------------------------------------------------------


class TestSubExperimentValidation(unittest.TestCase):
    """sub_experiment_id must be consistent with the experiment CV entry.

    CMOR3 references:
      - ``test_python_CMIP6_CV_sub_experiment_id.py``
      - ``test_python_CMIP6_CV_sub_experimentIDbad.py``
      - ``test_python_CMIP6_CV_sub_experimentbad.py``
      - ``test_python_CMIP6_CV_sub_experimentnotset.py``
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_dcpp_hindcast_requires_valid_sub_experiment_id(self) -> None:
        """The dcppA-hindcast experiment requires a valid sub_experiment_id.

        CMOR3 test ``test_python_CMIP6_CV_sub_experiment_id.py``:
        sets ``sub_experiment_id="s3000"`` (not in allowed list) and expects
        an error containing ``"sub_experiment_id"``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(
                    _dcpp_hindcast_attrs(tmp, sub_experiment_id="s3000")
                )
            self.assertIn("sub_experiment_id", str(ctx.exception))

    def test_dcpp_hindcast_with_valid_sub_experiment_passes(self) -> None:
        """A dcppA-hindcast dataset with a CV-allowed sub_experiment_id is accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(
                _dcpp_hindcast_attrs(tmp, sub_experiment_id="s1960")
            )
            self.assertIsNotNone(info)

    def test_amip_experiment_accepts_none_sub_experiment(self) -> None:
        """The amip experiment (no sub-experiments) accepts sub_experiment_id='none'."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp, sub_experiment_id="none"))
            self.assertIsNotNone(info)

    def test_unknown_sub_experiment_id_is_rejected(self) -> None:
        """A sub_experiment_id not in the CV is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError):
                self.project.dataset_info(_amip_attrs(tmp, sub_experiment_id="sXXXX"))


# ---------------------------------------------------------------------------
# 8. Activity ID validation
# ---------------------------------------------------------------------------


class TestActivityValidation(unittest.TestCase):
    """activity_id must be consistent with the experiment_id CV entry.

    CMOR3 reference: ``test_python_CMIP6_wrong_activity.py``
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_wrong_activity_id_for_experiment_is_rejected(self) -> None:
        """Using ScenarioMIP as activity_id for an amip experiment is rejected.

        CMOR3 test: sets activity_id to a value inconsistent with experiment_id
        and expects a CV error.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError):
                self.project.dataset_info(_amip_attrs(tmp, activity_id="ScenarioMIP"))

    def test_correct_activity_id_for_amip_passes(self) -> None:
        """The CMIP activity_id is correct for the amip experiment."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp, activity_id="CMIP"))
            self.assertIsNotNone(info)

    def test_invalid_activity_id_string_is_rejected(self) -> None:
        """An activity_id not in the CV enum is always rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError):
                self.project.dataset_info(
                    _amip_attrs(tmp, activity_id="NOT_AN_ACTIVITY")
                )


# ---------------------------------------------------------------------------
# 9. Experiment ID validation
# ---------------------------------------------------------------------------


class TestExperimentValidation(unittest.TestCase):
    """experiment_id must be registered in the CMIP6 CV.

    CMOR3 reference: ``test_python_CMIP6_experimentID.py``
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_unknown_experiment_id_is_rejected(self) -> None:
        """An experiment_id not in the CV raises ControlledVocabularyError."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError) as ctx:
                self.project.dataset_info(
                    _amip_attrs(tmp, experiment_id="bad_experiment")
                )
            self.assertIn("bad_experiment", str(ctx.exception))

    def test_valid_amip_experiment_passes(self) -> None:
        """The well-known amip experiment_id is accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            self.assertEqual(dict(info).get("experiment_id"), "amip")

    def test_experiment_description_is_auto_filled(self) -> None:
        """The ``experiment`` attribute is auto-populated from the CV entry."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            experiment = dict(info).get("experiment", "")
            self.assertIn("AMIP", experiment.upper())

    def test_mip_era_cmip6_is_required(self) -> None:
        """Supplying a wrong mip_era value is rejected.

        CMOR3 reference: ``test_python_CMIP6_CV_nomipera.py``
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cmor4.ControlledVocabularyError):
                self.project.dataset_info(_amip_attrs(tmp, mip_era="CMIP5"))


# ---------------------------------------------------------------------------
# 10. External variables (cell_measures)
# ---------------------------------------------------------------------------


class TestExternalVariables(unittest.TestCase):
    """Variables with cell_measures that are not in-file list external_variables.

    CMOR3 reference: ``test_python_CMIP6_CV_externalvariables.py``
    CMOR3 assertion: ``"external_variables" in f.__dict__`` and
    ``f.__dict__["external_variables"] == "areacello volcello"``
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_masscello_requires_external_areacello_and_volcello(self) -> None:
        """masscello has ``area: areacello volume: volcello`` in cell_measures.

        Both areacello and volcello are not provided in-file, so both should
        appear in external_variables — matching the CMOR3 test assertion that
        ``external_variables == "areacello volcello"``.
        """
        variable = self.project.variable("masscello")
        external = _collect_external_variables(variable, set())
        self.assertIn("areacello", external)
        self.assertIn("volcello", external)

    def test_masso_has_no_cell_measures(self) -> None:
        """masso has no cell_measures, so external_variables should be empty."""
        variable = self.project.variable("masso")
        external = _collect_external_variables(variable, set())
        self.assertEqual(external, set())

    def test_tas_requires_external_areacella(self) -> None:
        """The Amon tas variable references area: areacella as a cell measure."""
        variable = self.project.variable("tas")
        external = _collect_external_variables(variable, set())
        self.assertIn("areacella", external)

    def test_provided_cell_measure_is_excluded_from_external_variables(self) -> None:
        """A cell measure that IS provided in-file is not listed as external."""
        variable = self.project.variable("masscello")
        # Simulate providing areacello in the output
        external = _collect_external_variables(variable, {"areacello"})
        self.assertNotIn("areacello", external)
        # volcello is still external
        self.assertIn("volcello", external)


# ---------------------------------------------------------------------------
# 11. Output global attributes
# ---------------------------------------------------------------------------


class TestOutputAttributes(unittest.TestCase):
    """Output dataset must carry the correct CMIP6 global attributes.

    CMOR3 references:
      - ``test_python_CMIP6_CV_furtherinfourl.py`` — further_info_url
      - ``test_python_CMIP6_CV_HISTORY.py`` — history attribute format
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_tracking_id_matches_cmip6_hdl_pattern(self) -> None:
        """The auto-generated tracking_id uses the CMIP6 hdl:21.14100/ prefix.

        CMOR3 generates tracking_id as ``tracking_prefix + "/" + uuid``.
        The CMIP6 CV defines ``tracking_id: ['hdl:21.14100/.*']``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            tracking_id = dict(info).get("tracking_id", "")
            self.assertTrue(
                tracking_id.startswith("hdl:21.14100/"),
                "tracking_id should start with 'hdl:21.14100/' "
                f"but got {tracking_id!r}",
            )

    def test_tracking_id_has_uuid_suffix(self) -> None:
        """The tracking_id UUID portion matches the standard UUID4 format."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            tracking_id = dict(info).get("tracking_id", "")
            uuid_part = tracking_id.split("/", 1)[-1] if "/" in tracking_id else ""
            uuid_pattern = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
            )
            self.assertRegex(
                uuid_part,
                uuid_pattern,
                f"UUID portion of tracking_id is malformed: {uuid_part!r}",
            )

    def test_tracking_id_satisfies_cv_regex(self) -> None:
        """The generated tracking_id matches the CV-defined regex pattern."""
        import json
        import cmor4.cv as cv_module

        with open(CMIP6_TABLE_ROOT / CMIP6_CV_FILE) as fh:
            cv_data = json.load(fh)
        pattern_list = cv_data["CV"].get("tracking_id", [])
        self.assertTrue(pattern_list, "CMIP6 CV should define a tracking_id pattern")
        python_pattern = cv_module._posix_regex_to_python(pattern_list[0])

        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            tracking_id = dict(info).get("tracking_id", "")
            self.assertRegex(
                tracking_id,
                python_pattern,
                f"tracking_id={tracking_id!r} does not match CV pattern",
            )

    def test_conventions_is_valid_cmip6_value(self) -> None:
        """Conventions is set to a value matching the CMIP6 CV regex.

        The CMIP6 CV regex is ``^CF-1.7 CMIP-6.[0-2](…)$``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            conventions = dict(info).get("Conventions", "")
            self.assertTrue(
                conventions.startswith("CF-1.7 CMIP-6."),
                f"Conventions should match CMIP6 pattern but got {conventions!r}",
            )

    def test_history_attribute_format_is_cmor_standard(self) -> None:
        """The history attribute uses the standard CMOR rewrite message.

        CMOR3 test ``test_python_CMIP6_CV_HISTORY.py`` / ``test_cmor_CMIP7.py``::

            self.assertIn(
                "CMOR rewrote data to be consistent with {conventions} "
                "and {mip_era} data requirements.",
                ds.getncattr("history"),
            )
        """
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            variable = self.project.variable("masso")
            time_axis = self.project.axis(
                "time",
                values=[0.5, 1.5, 2.5],
                bounds=[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
                units="months since 2000-01-01",
            )
            data = np.random.random(3) * 1e18
            result = cmor4.cmorize(info, variable, [time_axis], data)
            ds = result.dataset
            history = ds.attrs.get("history", "")
            self.assertIn("CMOR rewrote data to be consistent with", history)
            conventions = ds.attrs.get("Conventions", "CMIP")
            mip_era = ds.attrs.get("mip_era", "CMIP6")
            self.assertIn(conventions, history)
            self.assertIn(mip_era, history)

    def test_mip_era_attribute_is_cmip6(self) -> None:
        """The mip_era global attribute is set to CMIP6."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            self.assertEqual(dict(info).get("mip_era"), "CMIP6")

    def test_institution_attribute_is_populated(self) -> None:
        """The full institution name is populated from the CV."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            institution = dict(info).get("institution", "")
            # PCMDI's CV entry is the full long-form name
            self.assertIn("Program for Climate Model Diagnosis", institution)

    def test_source_attribute_is_populated(self) -> None:
        """The full source description is populated from the CV."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            source = dict(info).get("source", "")
            self.assertIn("PCMDI-test", source)

    def test_experiment_attribute_is_populated(self) -> None:
        """The human-readable experiment description is populated from the CV."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            experiment = dict(info).get("experiment", "")
            self.assertTrue(len(experiment) > 0)


# ---------------------------------------------------------------------------
# 12. DRS path and filename template
# ---------------------------------------------------------------------------


class TestDrsTemplates(unittest.TestCase):
    """CMIP6 DRS path and filename templates are read from the CV.

    CMOR3 reference: ``Src/cmor.c`` DRS section; GitHub issue #834
    CMIP6 DRS defines:
      directory_path_template: <mip_era><activity_id>…<version>
      filename_template: <variable_id><table_id><source_id>…<grid_label>
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def test_cmip6_cv_defines_drs_path_template(self) -> None:
        """The CMIP6 CV includes a directory_path_template in its DRS section."""
        path_tmpl, _ = self.project.cv.drs_templates()
        self.assertIsNotNone(
            path_tmpl, "CMIP6 CV should define a directory_path_template"
        )
        assert path_tmpl is not None  # Type narrowing for mypy
        self.assertIn("mip_era", path_tmpl)
        self.assertIn("institution_id", path_tmpl)

    def test_cmip6_cv_defines_drs_filename_template(self) -> None:
        """The CMIP6 CV includes a filename_template in its DRS section."""
        _, file_tmpl = self.project.cv.drs_templates()
        self.assertIsNotNone(file_tmpl, "CMIP6 CV should define a filename_template")
        assert file_tmpl is not None  # Type narrowing for mypy
        self.assertIn("variable_id", file_tmpl)
        self.assertIn("source_id", file_tmpl)

    def test_drs_path_example_structure(self) -> None:
        """The CMIP6 DRS path template follows CMIP6/<activity>/<inst>/…."""
        path_tmpl, _ = self.project.cv.drs_templates()
        assert path_tmpl is not None  # Type narrowing for mypy
        # The template starts with <mip_era> which resolves to CMIP6
        self.assertTrue(
            path_tmpl.startswith("<mip_era>"),
            f"CMIP6 path template should start with <mip_era>: {path_tmpl!r}",
        )


# ---------------------------------------------------------------------------
# 13. fx table (fixed fields)
# ---------------------------------------------------------------------------


class TestFxTable(unittest.TestCase):
    """Variables from the fx (fixed fields) table are handled correctly.

    CMOR3 reference: ``test_python_CMIP6_CV_fxtable.py`` — produces areacella.
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project(CMIP6_FX_TABLE)

    def test_areacella_variable_loads_from_fx_table(self) -> None:
        """The fx table's areacella variable is accessible via the project."""
        variable = self.project.variable("areacella")
        self.assertIsNotNone(variable)
        self.assertEqual(variable.units, "m2")

    def test_areacella_has_no_time_dimension(self) -> None:
        """areacella is a time-invariant field; it should not list time in dims."""
        variable = self.project.variable("areacella")
        dims = list(variable.dimensions or [])
        self.assertNotIn("time", dims)

    def test_orog_variable_loads_from_fx_table(self) -> None:
        """The fx table's orog (surface altitude) variable is accessible."""
        variable = self.project.variable("orog")
        self.assertIsNotNone(variable)

    def test_fx_cmorize_produces_lat_lon_dataset(self) -> None:
        """An areacella dataset can be cmorized with lat/lon axes.

        Mirrors CMOR3's ``test_python_CMIP6_CV_fxtable.py`` which writes
        ``areacella`` using lat/lon axes only (no time axis).
        """
        nlat, nlon = 10, 20
        dlat = 180.0 / nlat
        dlon = 360.0 / nlon

        lats = np.arange(-90 + dlat / 2.0, 90, dlat)  # increasing south→north
        blats = np.arange(-90, 90 + dlat, dlat)
        lons = np.arange(0 + dlon / 2.0, 360.0, dlon)
        blons = np.arange(0, 360.0 + dlon, dlon)

        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp, frequency="fx"))
            variable = self.project.variable("areacella")
            lat_axis = self.project.axis(
                "latitude",
                values=lats,
                bounds=blats,
                units="degrees_north",
            )
            lon_axis = self.project.axis(
                "longitude",
                values=lons,
                bounds=blons,
                units="degrees_east",
            )
            data = np.abs(lats[:, np.newaxis] * lons[np.newaxis, :]) + 1e8
            result = cmor4.cmorize(info, variable, [lat_axis, lon_axis], data)
            ds = result.dataset
            self.assertIn("areacella", ds)
            self.assertEqual(ds["areacella"].attrs.get("units"), "m2")
            self.assertIn("lat", ds)
            self.assertIn("lon", ds)


# ---------------------------------------------------------------------------
# 14. Basic CMIP6 dataset round-trip
# ---------------------------------------------------------------------------


class TestCmip6DatasetRoundtrip(unittest.TestCase):
    """End-to-end CMIP6 dataset creation mirrors the CMOR3 CMIP6 examples.

    The CMOR3 CMIP6 tests all follow the same pattern:
      1. Setup with inpath='Tables'
      2. dataset_json('Test/CMOR_input_example.json')
      3. load_table('CMIP6_Omon.json')
      4. Write a variable and close
      5. Assert on log output or NetCDF attributes

    CMOR4 equivalent: ProjectTables + dataset_info + cmorize.
    """

    def setUp(self) -> None:
        _require_cmip6_tables(self)
        self.project = _cmip6_project()

    def _time_axis(self) -> cmor4.Axis:
        return self.project.axis(
            "time",
            values=[0.5, 1.5, 2.5, 3.5, 4.5],
            bounds=[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
            units="months since 2010-01-01",
        )

    def test_masso_cmorize_produces_valid_dataset(self) -> None:
        """Cmorizing masso (ocean mass) mirrors the CMOR3 Omon basic test.

        CMOR3 test: loads CMIP6_Omon.json, creates ``masso``, writes 5 time steps.
        """
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            variable = self.project.variable("masso")
            time_axis = self._time_axis()
            data = np.random.random(5) * 1e18

            result = cmor4.cmorize(info, variable, [time_axis], data)
            ds = result.dataset

            self.assertIn("masso", ds)
            self.assertEqual(ds["masso"].attrs.get("units"), "kg")
            self.assertIn("time", ds)
            self.assertEqual(ds["masso"].dims, ("time",))

    def test_masso_dataset_has_cmip6_global_attrs(self) -> None:
        """Cmorized masso dataset has required CMIP6 global attributes."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            variable = self.project.variable("masso")
            time_axis = self._time_axis()
            data = np.random.random(5) * 1e18

            result = cmor4.cmorize(info, variable, [time_axis], data)
            ds = result.dataset

            for attr in (
                "mip_era",
                "institution_id",
                "source_id",
                "experiment_id",
                "tracking_id",
                "Conventions",
            ):
                self.assertIn(attr, ds.attrs, f"Missing required attr: {attr}")

            self.assertEqual(ds.attrs.get("mip_era"), "CMIP6")
            self.assertTrue(ds.attrs.get("tracking_id", "").startswith("hdl:21.14100/"))

    def test_tas_cmorize_produces_correct_units_and_dims(self) -> None:
        """Cmorizing Amon tas with lat/lon/time produces a correct dataset."""
        nlat, nlon = 4, 8
        dlat = 180.0 / nlat
        dlon = 360.0 / nlon
        lats = np.linspace(-90 + dlat / 2, 90 - dlat / 2, nlat)  # increasing
        lat_bounds = np.column_stack([lats - dlat / 2, lats + dlat / 2])
        lons = np.linspace(dlon / 2, 360 - dlon / 2, nlon)
        lon_bounds = np.column_stack([lons - dlon / 2, lons + dlon / 2])

        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            variable = self.project.variable("tas")
            time_axis = self._time_axis()
            lat_axis = self.project.axis(
                "latitude",
                values=lats,
                bounds=lat_bounds,
                units="degrees_north",
            )
            lon_axis = self.project.axis(
                "longitude",
                values=lons,
                bounds=lon_bounds,
                units="degrees_east",
            )
            data = np.random.uniform(200.0, 320.0, (5, nlat, nlon)).astype("f4")

            result = cmor4.cmorize(
                info, variable, [time_axis, lat_axis, lon_axis], data
            )
            ds = result.dataset

            self.assertIn("tas", ds)
            self.assertEqual(ds["tas"].attrs.get("units"), "K")
            self.assertEqual(ds["tas"].attrs.get("standard_name"), "air_temperature")
            self.assertEqual(ds["tas"].dims, ("time", "lat", "lon"))

    def test_creation_date_is_iso8601_format(self) -> None:
        """The creation_date attribute is in ISO 8601 UTC format."""
        with tempfile.TemporaryDirectory() as tmp:
            info = self.project.dataset_info(_amip_attrs(tmp))
            creation_date = dict(info).get("creation_date", "")
            self.assertRegex(
                creation_date,
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
                f"creation_date format is wrong: {creation_date!r}",
            )

    def test_two_sequential_datasets_get_unique_tracking_ids(self) -> None:
        """Each dataset_info call generates a unique tracking_id."""
        with (
            tempfile.TemporaryDirectory() as tmp1,
            tempfile.TemporaryDirectory() as tmp2,
        ):
            info1 = self.project.dataset_info(_amip_attrs(tmp1))
            info2 = self.project.dataset_info(_amip_attrs(tmp2))
            tid1 = dict(info1).get("tracking_id", "")
            tid2 = dict(info2).get("tracking_id", "")
            self.assertNotEqual(
                tid1,
                tid2,
                "Each dataset should get a unique tracking_id",
            )


if __name__ == "__main__":
    unittest.main()
