"""Tests for Phase 1 validation gaps.

Covers:
  GAP-01 — frequency required when a time axis is present
  GAP-02 — external_variables global attribute from cell_measures
  GAP-03 — variant index (RIPF) format and overflow validation
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import cmor4
from cmor4 import Axis, ControlledVocabulary, DatasetInfo, ProjectTables, Variable
from cmor4._axis_validation import _validate_time_interval, _is_time_axis
from cmor4.core import _collect_external_variables, create_dataset
from cmor4.exceptions import AxisValidationError, ControlledVocabularyError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj) + "\n")


def _make_project(
    tmp: Path, cv: dict, variables: dict, coordinates: dict
) -> ProjectTables:
    cv_path = tmp / "cv.json"
    var_path = tmp / "vars.json"
    coord_path = tmp / "coords.json"
    _write(cv_path, cv)
    _write(var_path, {"variable_entry": variables})
    _write(coord_path, {"axis_entry": coordinates})
    return ProjectTables(
        cv_file=cv_path,
        variable_tables=[var_path],
        coordinate_table=coord_path,
    )


_MINIMAL_CV: dict = {
    "CV": {
        "activity_id": ["CMIP"],
        "institution_id": {"NCAR": "National Center for Atmospheric Research"},
        "source_id": {"CESM2": {"institution_id": ["NCAR"]}},
        "experiment_id": {
            "amip": {"experiment_id": "amip", "activity_id": ["CMIP"]},
        },
        "required_global_attributes": ["activity_id", "institution_id"],
        "mip_era": "CMIP7",
    }
}

# CMIP7 RIPF indices are prefixed strings: "r1", "i1", "p1", "f1".
# Plain integer strings ("1", "2", …) are the CMOR3 style; both are
# accepted by validate_variant_indices.
_MINIMAL_DATASET: dict = {
    "activity_id": "CMIP",
    "institution_id": "NCAR",
    "source_id": "CESM2",
    "experiment_id": "amip",
    "mip_era": "CMIP7",
    "realization_index": "r1",
    "initialization_index": "i1",
    "physics_index": "p1",
    "forcing_index": "f1",
    "grid_label": "gn",
    "frequency": "mon",
    "outpath": ".",
}

_MINIMAL_COORDINATES: dict = {
    "time": {
        "axis": "T",
        "standard_name": "time",
        "out_name": "time",
        "units": "days since 2000-01-01",
    },
    "lat": {
        "axis": "Y",
        "units": "degrees_north",
        "standard_name": "latitude",
        "out_name": "lat",
    },
    "lon": {
        "axis": "X",
        "units": "degrees_east",
        "standard_name": "longitude",
        "out_name": "lon",
    },
}

_MINIMAL_VARIABLES: dict = {
    "tas": {
        "dimensions": ["time", "lat", "lon"],
        "out_name": "tas",
        "units": "K",
        "standard_name": "air_temperature",
        "frequency": "mon",
        "realm": "atmos",
    }
}


# ---------------------------------------------------------------------------
# GAP-01 — frequency required when a time axis is present
# ---------------------------------------------------------------------------


class TestFrequencyRequired(unittest.TestCase):
    """GAP-01: A time axis without a declared frequency must raise AxisValidationError."""

    def _time_axis(self, n_steps: int = 3) -> Axis:
        return Axis(
            name="time",
            axis="T",
            standard_name="time",
            units="days since 2000-01-01",
            values=list(range(0, 30 * n_steps, 30)),
        )

    # -----------------------------------------------------------------------
    # Error cases
    # -----------------------------------------------------------------------

    def test_missing_frequency_raises_for_multi_step_time_axis(self):
        """No frequency + ≥2 time steps → AxisValidationError.

        The dataset must be non-empty to trigger the frequency requirement;
        an empty dict is the 'no dataset' sentinel used by validate_components
        for structure-only validation.
        """
        axis = self._time_axis(n_steps=3)
        dataset_with_content = {"institution_id": "NCAR"}  # non-empty, no frequency
        with self.assertRaises(AxisValidationError) as ctx:
            _validate_time_interval(dataset_with_content, {}, axis, axis.values_array())
        self.assertIn("frequency", str(ctx.exception).lower())

    def test_missing_frequency_raises_for_single_step_time_axis(self):
        """No frequency + 1 time step → AxisValidationError.

        CMOR3 errors on missing frequency regardless of the number of steps.
        A non-empty dataset is used to trigger the check (empty dict is the
        structure-only sentinel).
        """
        axis = Axis(
            name="time",
            axis="T",
            standard_name="time",
            units="days since 2000-01-01",
            values=[15],
        )
        dataset_with_content = {"institution_id": "NCAR"}
        with self.assertRaises(AxisValidationError) as ctx:
            _validate_time_interval(dataset_with_content, {}, axis, axis.values_array())
        self.assertIn("frequency", str(ctx.exception).lower())

    def test_empty_string_frequency_raises(self):
        """Explicit empty string treated same as absent."""
        axis = self._time_axis()
        with self.assertRaises(AxisValidationError):
            _validate_time_interval({"frequency": ""}, {}, axis, axis.values_array())

    def test_error_message_contains_guidance(self):
        """Error message should guide the user toward setting frequency.

        A non-empty dataset is required to trigger the check; an empty dict
        is the structure-only sentinel used by validate_components.
        """
        axis = self._time_axis()
        dataset_with_content = {"institution_id": "NCAR"}
        with self.assertRaises(AxisValidationError) as ctx:
            _validate_time_interval(dataset_with_content, {}, axis, axis.values_array())
        msg = str(ctx.exception)
        self.assertIn("frequency", msg.lower())
        self.assertIn("required", msg.lower())

    # -----------------------------------------------------------------------
    # Happy-path cases
    # -----------------------------------------------------------------------

    def test_frequency_in_dataset_passes(self):
        """frequency in dataset metadata → no error."""
        axis = self._time_axis()
        _validate_time_interval(
            {"frequency": "mon", "calendar": "360_day"},
            {},
            axis,
            axis.values_array(),
        )

    def test_frequency_in_variable_passes(self):
        """frequency carried on the variable → no error."""
        axis = self._time_axis()
        _validate_time_interval(
            {},
            {"frequency": "mon"},
            axis,
            axis.values_array(),
        )

    def test_non_time_axis_is_never_passed_to_validator(self):
        """_is_time_axis correctly returns False for latitude, preventing the call."""
        lat_axis = Axis(
            name="lat",
            axis="Y",
            units="degrees_north",
            values=[-45.0, 0.0, 45.0],
        )
        self.assertFalse(_is_time_axis(lat_axis))

    def test_validate_axes_raises_for_time_without_frequency(self):
        """Full validate_axes path raises when frequency is absent from a real dataset.

        A non-empty dataset is needed to trigger the check; an empty dict is
        the structure-only placeholder passed by validate_components(None, …).
        """
        from cmor4._axis_validation import validate_axes

        time_axis = Axis(
            name="time",
            axis="T",
            standard_name="time",
            units="days since 2000-01-01",
            values=[15, 45, 75],
        )
        dataset_with_content = {"institution_id": "NCAR"}  # no frequency key
        with self.assertRaises(AxisValidationError):
            validate_axes(dataset_with_content, {}, [time_axis])

    def test_validate_axes_passes_for_lat_without_frequency(self):
        """Non-time axes are not affected by the frequency requirement."""
        from cmor4._axis_validation import validate_axes

        lat_axis = Axis(
            name="lat",
            axis="Y",
            units="degrees_north",
            values=[-45.0, 45.0],
        )
        validate_axes({}, {}, [lat_axis])  # no error expected

    def test_project_tables_create_dataset_raises_without_frequency(self):
        """create_dataset raises AxisValidationError when frequency is absent from
        both the dataset metadata and the variable table entry."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            # Variable table entry deliberately has no frequency key.
            variables_no_freq = {
                "tas": {
                    "dimensions": ["time", "lat", "lon"],
                    "out_name": "tas",
                    "units": "K",
                    "standard_name": "air_temperature",
                    "realm": "atmos",
                    # NOTE: no "frequency" key here
                }
            }
            project = _make_project(
                tmp, _MINIMAL_CV, variables_no_freq, _MINIMAL_COORDINATES
            )
            dataset_no_freq = {
                k: v for k, v in _MINIMAL_DATASET.items() if k != "frequency"
            }
            dataset_info = project.dataset_info(dataset_no_freq)
            variable = project.variable("tas")

            time_axis = project.axis(
                "time", values=[15.0, 45.0], units="days since 2000-01-01"
            )
            lat_axis = project.axis("lat", values=[-45.0, 45.0])
            lon_axis = project.axis("lon", values=[0.0, 90.0])
            data = np.ones((2, 2, 2))

            with self.assertRaises(AxisValidationError):
                create_dataset(
                    dataset_info, variable, [time_axis, lat_axis, lon_axis], data
                )


# ---------------------------------------------------------------------------
# GAP-02 — external_variables global attribute
# ---------------------------------------------------------------------------


class TestExternalVariables(unittest.TestCase):
    """GAP-02: cell_measures references to absent variables → external_variables attr."""

    def _variable_with_cell_measures(self, cell_measures: str) -> Variable:
        return Variable(name="tos", units="degC", cell_measures=cell_measures)

    # -----------------------------------------------------------------------
    # _collect_external_variables unit tests
    # -----------------------------------------------------------------------

    def test_single_external_variable(self):
        var = self._variable_with_cell_measures("area: areacello")
        result = _collect_external_variables(var, set())
        self.assertEqual(result, {"areacello"})

    def test_multiple_external_variables(self):
        var = self._variable_with_cell_measures("area: areacello volume: volcello")
        result = _collect_external_variables(var, set())
        self.assertEqual(result, {"areacello", "volcello"})

    def test_provided_variable_not_in_result(self):
        """If a cell_measures variable is provided as a coordinate, skip it."""
        var = self._variable_with_cell_measures("area: areacello volume: volcello")
        result = _collect_external_variables(var, {"areacello"})
        self.assertEqual(result, {"volcello"})

    def test_all_provided_returns_empty_set(self):
        var = self._variable_with_cell_measures("area: areacello volume: volcello")
        result = _collect_external_variables(var, {"areacello", "volcello"})
        self.assertEqual(result, set())

    def test_empty_cell_measures_returns_empty_set(self):
        var = self._variable_with_cell_measures("")
        self.assertEqual(_collect_external_variables(var, set()), set())

    def test_none_cell_measures_returns_empty_set(self):
        var = Variable(name="tos", units="degC")
        self.assertEqual(_collect_external_variables(var, set()), set())

    def test_space_only_cell_measures_returns_empty_set(self):
        var = self._variable_with_cell_measures("   ")
        self.assertEqual(_collect_external_variables(var, set()), set())

    # -----------------------------------------------------------------------
    # create_dataset integration tests
    # -----------------------------------------------------------------------

    def _run_create_dataset(self, cell_measures: str, tmp: Path) -> dict:
        """Build a minimal dataset with the given cell_measures and return global attrs."""
        variables = {
            "tos": {
                "dimensions": ["time", "lat", "lon"],
                "out_name": "tos",
                "units": "K",
                "standard_name": "sea_surface_temperature",
                "frequency": "mon",
                "realm": "ocean",
                "cell_measures": cell_measures,
            }
        }
        project = _make_project(tmp, _MINIMAL_CV, variables, _MINIMAL_COORDINATES)
        dataset_info = project.dataset_info(_MINIMAL_DATASET)
        variable = project.variable("tos")
        time_axis = project.axis(
            "time", values=[15.0, 45.0], units="days since 2000-01-01"
        )
        lat_axis = project.axis("lat", values=[-45.0, 45.0])
        lon_axis = project.axis("lon", values=[0.0, 90.0])
        data = np.ones((2, 2, 2)) * 290.0
        ds = create_dataset(
            dataset_info, variable, [time_axis, lat_axis, lon_axis], data
        )
        return ds.attrs

    def test_external_variable_written_to_global_attrs(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            attrs = self._run_create_dataset("area: areacello", Path(tmp_str))
        self.assertIn("external_variables", attrs)
        self.assertEqual(attrs["external_variables"], "areacello")

    def test_multiple_external_variables_space_separated_sorted(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            attrs = self._run_create_dataset(
                "area: areacello volume: volcello", Path(tmp_str)
            )
        self.assertIn("external_variables", attrs)
        self.assertEqual(attrs["external_variables"], "areacello volcello")

    def test_no_external_variables_when_cell_measures_empty(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            attrs = self._run_create_dataset("", Path(tmp_str))
        self.assertNotIn("external_variables", attrs)

    def test_user_supplied_attrs_override_computed_external_variables(self):
        """Explicit attrs= override the computed external_variables."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            variables = {
                "tos": {
                    "dimensions": ["time", "lat", "lon"],
                    "out_name": "tos",
                    "units": "K",
                    "standard_name": "sea_surface_temperature",
                    "frequency": "mon",
                    "realm": "ocean",
                    "cell_measures": "area: areacello",
                }
            }
            project = _make_project(tmp, _MINIMAL_CV, variables, _MINIMAL_COORDINATES)
            dataset_info = project.dataset_info(_MINIMAL_DATASET)
            variable = project.variable("tos")
            time_axis = project.axis(
                "time", values=[15.0, 45.0], units="days since 2000-01-01"
            )
            lat_axis = project.axis("lat", values=[-45.0, 45.0])
            lon_axis = project.axis("lon", values=[0.0, 90.0])
            data = np.ones((2, 2, 2)) * 290.0
            ds = create_dataset(
                dataset_info,
                variable,
                [time_axis, lat_axis, lon_axis],
                data,
                attrs={"external_variables": "custom_var"},
            )
        self.assertEqual(ds.attrs["external_variables"], "custom_var")

    def test_no_external_variables_when_all_provided_as_coords(self):
        """If areacello is provided as a coordinate it does not appear in external_variables."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            variables = {
                "tos": {
                    "dimensions": ["time", "lat", "lon"],
                    "out_name": "tos",
                    "units": "K",
                    "standard_name": "sea_surface_temperature",
                    "frequency": "mon",
                    "realm": "ocean",
                    "cell_measures": "area: areacello",
                }
            }
            coordinates = dict(_MINIMAL_COORDINATES)
            coordinates["areacello"] = {
                "units": "m2",
                "standard_name": "cell_area",
                "out_name": "areacello",
            }
            project = _make_project(tmp, _MINIMAL_CV, variables, coordinates)
            dataset_info = project.dataset_info(_MINIMAL_DATASET)
            variable = project.variable("tos")
            time_axis = project.axis(
                "time", values=[15.0, 45.0], units="days since 2000-01-01"
            )
            lat_axis = project.axis("lat", values=[-45.0, 45.0])
            lon_axis = project.axis("lon", values=[0.0, 90.0])
            areacello_axis = project.axis(
                "areacello",
                values=[[1e10, 1e10], [1e10, 1e10]],
                dimensions=["lat", "lon"],
                auxiliary=True,
            )
            data = np.ones((2, 2, 2)) * 290.0
            ds = create_dataset(
                dataset_info,
                variable,
                [time_axis, lat_axis, lon_axis, areacello_axis],
                data,
            )
        self.assertNotIn("external_variables", ds.attrs)


# ---------------------------------------------------------------------------
# GAP-03 — variant index (RIPF) format and overflow validation
# ---------------------------------------------------------------------------


class TestVariantIndexValidation(unittest.TestCase):
    """GAP-03: RIPF indices must be valid positive integers within INT32 range.

    CMIP7 uses prefixed strings ("r1", "i1", "p1", "f1"); CMOR3 used bare
    integers ("1", "9", …).  Both formats are accepted; the numeric portion
    must satisfy 1 ≤ N ≤ 2^31 - 1.
    """

    def _cv(self) -> ControlledVocabulary:
        return ControlledVocabulary(_MINIMAL_CV)

    # -----------------------------------------------------------------------
    # Overflow error cases — bare-integer style (CMOR3 parity)
    # -----------------------------------------------------------------------

    def test_realization_index_overflow_raises(self):
        """A 31-digit realization_index mirrors the CMOR3 longrealizationindex test."""
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv.validate_variant_indices(
                {"realization_index": "1209374928349823498274987234987"}
            )
        self.assertIn("realization_index", str(ctx.exception))

    def test_initialization_index_overflow_raises(self):
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError):
            cv.validate_variant_indices({"initialization_index": str(2**31)})

    def test_physics_index_overflow_raises(self):
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError):
            cv.validate_variant_indices({"physics_index": str(2**63)})

    def test_forcing_index_overflow_raises(self):
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError):
            cv.validate_variant_indices({"forcing_index": str(2**31)})

    # -----------------------------------------------------------------------
    # Overflow error cases — prefixed-string style (CMIP7)
    # -----------------------------------------------------------------------

    def test_prefixed_realization_index_overflow_raises(self):
        """Overflow is caught even when the 'r' prefix is included."""
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError):
            cv.validate_variant_indices(
                {"realization_index": "r1209374928349823498274987234987"}
            )

    def test_prefixed_forcing_index_overflow_raises(self):
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError):
            cv.validate_variant_indices({"forcing_index": f"f{2**31}"})

    # -----------------------------------------------------------------------
    # Non-integer / zero / negative error cases
    # -----------------------------------------------------------------------

    def test_zero_realization_index_raises(self):
        """Zero is not a valid realization index (must be ≥1)."""
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv.validate_variant_indices({"realization_index": "0"})
        self.assertIn("realization_index", str(ctx.exception))

    def test_negative_index_raises(self):
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError):
            cv.validate_variant_indices({"realization_index": "-1"})

    def test_non_integer_realization_index_raises(self):
        """A completely non-numeric value must raise."""
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv.validate_variant_indices({"realization_index": "ensemble_1"})
        self.assertIn("realization_index", str(ctx.exception))

    def test_float_string_raises(self):
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError):
            cv.validate_variant_indices({"realization_index": "1.5"})

    # -----------------------------------------------------------------------
    # variant_label format — explicit value validated via CV constraint
    # -----------------------------------------------------------------------

    def test_explicit_malformed_variant_label_caught_by_cv_validate_dataset_values(self):
        """Explicit variant_label format is validated by validate_dataset_values
        when the CV defines a regex constraint for it.  validate_variant_indices
        intentionally defers this to the CV layer so that non-CMIP projects
        (e.g., obs4MIPs) can use custom labels like 'CMORGuide' without error.
        """
        cv_with_constraint = ControlledVocabulary(
            {
                "CV": {
                    **_MINIMAL_CV["CV"],
                    # CMIP7-style variant_label constraint as a regex list.
                    "variant_label": [r"r[[:digit:]]+i[[:digit:]]+p[[:digit:]]+f[[:digit:]]+"],
                }
            }
        )
        # Malformed label (missing 'f' component) is caught by validate_dataset_values.
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv_with_constraint.validate_dataset_values({"variant_label": "r1i1p1"})
        self.assertIn("variant_label", str(ctx.exception))

    def test_wrong_prefix_on_index_is_caught_before_assembly(self):
        """When an index carries the wrong prefix letter (e.g. 'r1' instead of
        'f1' for forcing_index), the integer-parsing step raises rather than
        silently assembling a malformed variant_label.
        """
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv.validate_variant_indices(
                {
                    "realization_index": "r1",
                    "initialization_index": "i1",
                    "physics_index": "p1",
                    "forcing_index": "r1",   # 'r' prefix is wrong for forcing_index
                }
            )
        # Should be caught at the integer-parsing level for forcing_index,
        # since stripping 'f' from 'r1' leaves 'r1' which is not an integer.
        self.assertIn("forcing_index", str(ctx.exception))

    # -----------------------------------------------------------------------
    # Happy-path cases — bare integers (CMOR3 style)
    # -----------------------------------------------------------------------

    def test_valid_bare_integer_ripf_indices_pass(self):
        cv = self._cv()
        cv.validate_variant_indices(
            {
                "realization_index": "1",
                "initialization_index": "1",
                "physics_index": "1",
                "forcing_index": "1",
            }
        )

    def test_max_int32_value_passes(self):
        cv = self._cv()
        cv.validate_variant_indices({"realization_index": str(2**31 - 1)})

    def test_large_but_valid_index_passes(self):
        cv = self._cv()
        cv.validate_variant_indices({"realization_index": "9999"})

    # -----------------------------------------------------------------------
    # Happy-path cases — prefixed strings (CMIP7 style)
    # -----------------------------------------------------------------------

    def test_valid_prefixed_ripf_indices_pass(self):
        cv = self._cv()
        cv.validate_variant_indices(
            {
                "realization_index": "r1",
                "initialization_index": "i1",
                "physics_index": "p1",
                "forcing_index": "f1",
            }
        )

    def test_valid_prefixed_large_indices_pass(self):
        cv = self._cv()
        cv.validate_variant_indices(
            {
                "realization_index": "r9",
                "initialization_index": "i3",
                "physics_index": "p2",
                "forcing_index": "f4",
            }
        )

    def test_max_int32_prefixed_passes(self):
        cv = self._cv()
        cv.validate_variant_indices({"realization_index": f"r{2**31 - 1}"})

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    def test_indices_absent_passes(self):
        cv = self._cv()
        cv.validate_variant_indices({})

    def test_explicit_variant_label_valid_passes(self):
        cv = self._cv()
        cv.validate_variant_indices({"variant_label": "r3i2p1f4"})

    def test_explicit_variant_label_large_indices_passes(self):
        cv = self._cv()
        cv.validate_variant_indices({"variant_label": "r100i200p50f3"})

    def test_none_value_skipped(self):
        cv = self._cv()
        cv.validate_variant_indices({"realization_index": None})

    def test_empty_string_value_skipped(self):
        cv = self._cv()
        cv.validate_variant_indices({"realization_index": ""})

    # -----------------------------------------------------------------------
    # Integration via ProjectTables
    # -----------------------------------------------------------------------

    def test_project_tables_dataset_info_raises_on_overflow(self):
        """ProjectTables.dataset_info catches overflow at dataset creation time."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, _MINIMAL_CV, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            bad_dataset = dict(_MINIMAL_DATASET)
            bad_dataset["realization_index"] = "1209374928349823498274987234987"
            with self.assertRaises(ControlledVocabularyError) as ctx:
                project.dataset_info(bad_dataset)
            self.assertIn("realization_index", str(ctx.exception))

    def test_project_tables_dataset_info_prefixed_overflow_raises(self):
        """ProjectTables.dataset_info catches prefixed-string overflow."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, _MINIMAL_CV, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            bad_dataset = dict(_MINIMAL_DATASET)
            bad_dataset["initialization_index"] = f"i{2**31}"
            with self.assertRaises(ControlledVocabularyError):
                project.dataset_info(bad_dataset)

    def test_project_tables_dataset_info_valid_indices_pass(self):
        """ProjectTables.dataset_info accepts all valid RIPF indices."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, _MINIMAL_CV, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            good_dataset = dict(_MINIMAL_DATASET)
            good_dataset.update(
                {
                    "realization_index": "r9",
                    "initialization_index": "i1",
                    "physics_index": "p1",
                    "forcing_index": "f3",
                }
            )
            info = project.dataset_info(good_dataset)
            self.assertIsNotNone(info)

    def test_project_tables_dataset_info_bare_integer_indices_pass(self):
        """CMOR3-style bare integer RIPF indices are also accepted."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, _MINIMAL_CV, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            good_dataset = dict(_MINIMAL_DATASET)
            good_dataset.update(
                {
                    "realization_index": "9",
                    "initialization_index": "1",
                    "physics_index": "1",
                    "forcing_index": "3",
                }
            )
            info = project.dataset_info(good_dataset)
            self.assertIsNotNone(info)


# ---------------------------------------------------------------------------
# GAP-04 — forcing terms validation against CV forcing list
# ---------------------------------------------------------------------------

# A minimal CV that defines a forcing enumeration, mirroring how CMIP6 CVs
# define valid forcing abbreviations.
_CV_WITH_FORCING: dict = {
    "CV": {
        **_MINIMAL_CV["CV"],
        "forcing": ["GHG", "Oz", "SA", "Sl", "Vl", "BC", "OC", "Nat", "Ant"],
    }
}


class TestForcingTermsValidation(unittest.TestCase):
    """GAP-04: forcing attribute tokens must be in the CV forcing enumeration.

    CMOR3 reference: ``cmor_check_forcing_validity`` in ``Src/cmor.c``.
    """

    def _cv(self) -> ControlledVocabulary:
        return ControlledVocabulary(_CV_WITH_FORCING)

    def _cv_no_forcing(self) -> ControlledVocabulary:
        """A CV without any forcing enumeration (e.g. CMIP7, obs4MIPs)."""
        return ControlledVocabulary(_MINIMAL_CV)

    # -----------------------------------------------------------------------
    # Error cases
    # -----------------------------------------------------------------------

    def test_unknown_single_token_raises(self):
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv.validate_forcing_terms({"forcing": "UNKNOWN"})
        self.assertIn("UNKNOWN", str(ctx.exception))

    def test_unknown_token_in_multi_token_string_raises(self):
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv.validate_forcing_terms({"forcing": "GHG Oz BADTERM"})
        self.assertIn("BADTERM", str(ctx.exception))

    def test_comma_separated_with_unknown_token_raises(self):
        """Comma-separated format is also supported (CMOR3 parity)."""
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv.validate_forcing_terms({"forcing": "GHG, Oz, BADTERM"})
        self.assertIn("BADTERM", str(ctx.exception))

    def test_error_message_includes_valid_values(self):
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv.validate_forcing_terms({"forcing": "JUNK"})
        msg = str(ctx.exception)
        self.assertIn("GHG", msg)  # valid values listed

    # -----------------------------------------------------------------------
    # Happy-path cases
    # -----------------------------------------------------------------------

    def test_single_valid_token_passes(self):
        cv = self._cv()
        cv.validate_forcing_terms({"forcing": "GHG"})

    def test_space_separated_valid_tokens_pass(self):
        cv = self._cv()
        cv.validate_forcing_terms({"forcing": "GHG Oz SA Sl Vl BC OC"})

    def test_comma_separated_valid_tokens_pass(self):
        cv = self._cv()
        cv.validate_forcing_terms({"forcing": "GHG, Oz, SA"})

    def test_mixed_comma_and_space_separated_tokens_pass(self):
        cv = self._cv()
        cv.validate_forcing_terms({"forcing": "GHG, Oz SA, Vl"})

    def test_annotation_in_parentheses_is_stripped(self):
        """Parenthetical annotation is truncated before tokenising — CMOR3 parity.

        'GHG Oz (GHG = CO2, N2O, …)' → tokens are ['GHG', 'Oz'].
        The annotation content (including any unrecognised terms inside it)
        is ignored.
        """
        cv = self._cv()
        cv.validate_forcing_terms(
            {"forcing": "GHG Oz (GHG = CO2, N2O, CH4, UNKNOWNGAS)"}
        )

    def test_annotation_truncation_not_just_removal(self):
        """Everything from the first '(' is dropped, not just the parenthetical.

        CMOR3 does astr[i] = '\\0' at the first '(', so:
        'GHG (note BADTERM after paren) Oz' → tokens ['GHG'] only.
        'Oz' after the closing ')' is NOT tokenised.
        """
        cv = self._cv()
        # 'Oz' appears after the closing paren; CMOR3 truncates at '(' so
        # only 'GHG' is validated — 'Oz' and 'BADTERM' are both dropped.
        cv.validate_forcing_terms(
            {"forcing": "GHG (note BADTERM here) Oz"}
        )

    def test_comma_replaced_with_space_before_truncation(self):
        """Commas become spaces before the '(' truncation step."""
        cv = self._cv()
        cv.validate_forcing_terms({"forcing": "GHG,Oz,SA (annotation)"})

    # -----------------------------------------------------------------------
    # Edge cases — no-ops
    # -----------------------------------------------------------------------

    def test_cv_without_forcing_key_skips_validation(self):
        """CMIP7 / obs4MIPs CVs have no forcing key — validation is a no-op."""
        cv = self._cv_no_forcing()
        # Even a completely unknown token should not raise.
        cv.validate_forcing_terms({"forcing": "ANYTHING_AT_ALL"})

    def test_dataset_without_forcing_attribute_passes(self):
        cv = self._cv()
        cv.validate_forcing_terms({})

    def test_empty_forcing_string_passes(self):
        cv = self._cv()
        cv.validate_forcing_terms({"forcing": ""})

    def test_none_forcing_passes(self):
        cv = self._cv()
        cv.validate_forcing_terms({"forcing": None})

    def test_only_whitespace_passes(self):
        cv = self._cv()
        cv.validate_forcing_terms({"forcing": "   "})

    def test_only_annotation_passes(self):
        """A forcing string that is purely an annotation (starts with '(')."""
        cv = self._cv()
        cv.validate_forcing_terms({"forcing": "(just an annotation)"})

    def test_forcing_dict_cv_also_works(self):
        """CV forcing defined as a mapping (CMIP6 style)."""
        cv_dict = ControlledVocabulary(
            {
                "CV": {
                    **_MINIMAL_CV["CV"],
                    "forcing": {
                        "GHG": "Greenhouse gases",
                        "Oz": "Ozone",
                        "Nat": "Natural forcings",
                    },
                }
            }
        )
        cv_dict.validate_forcing_terms({"forcing": "GHG Oz"})
        with self.assertRaises(ControlledVocabularyError):
            cv_dict.validate_forcing_terms({"forcing": "GHG UNKNOWN"})

    # -----------------------------------------------------------------------
    # Integration via validate_dataset and ProjectTables
    # -----------------------------------------------------------------------

    def test_validate_dataset_includes_forcing_check(self):
        cv = self._cv()
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv.validate_dataset(
                {
                    "activity_id": "CMIP",
                    "institution_id": "NCAR",
                    "forcing": "GHG BADTOKEN",
                }
            )
        self.assertIn("BADTOKEN", str(ctx.exception))

    def test_project_tables_dataset_info_raises_on_bad_forcing(self):
        """ProjectTables.dataset_info catches bad forcing tokens at creation time."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, _CV_WITH_FORCING, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            bad_dataset = dict(_MINIMAL_DATASET)
            bad_dataset["forcing"] = "GHG INVALIDTERM"
            with self.assertRaises(ControlledVocabularyError) as ctx:
                project.dataset_info(bad_dataset)
            self.assertIn("INVALIDTERM", str(ctx.exception))

    def test_project_tables_dataset_info_valid_forcing_passes(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, _CV_WITH_FORCING, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            good_dataset = dict(_MINIMAL_DATASET)
            good_dataset["forcing"] = "GHG Oz SA Sl Vl BC OC (GHG = CO2, N2O)"
            info = project.dataset_info(good_dataset)
            self.assertIsNotNone(info)

    def test_project_tables_no_forcing_cv_ignores_forcing_field(self):
        """Projects without a CV forcing key accept any forcing text."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, _MINIMAL_CV, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            dataset = dict(_MINIMAL_DATASET)
            dataset["forcing"] = "N/A"
            info = project.dataset_info(dataset)
            self.assertIsNotNone(info)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# GAP-05 — DRS path/filename templates read from the CV
# ---------------------------------------------------------------------------

from cmor4.core import build_output_path, DEFAULT_OUTPUT_PATH_TEMPLATE, DEFAULT_OUTPUT_FILE_TEMPLATE  # noqa: E402


_CV_WITH_DRS: dict = {
    "CV": {
        **_MINIMAL_CV["CV"],
        "DRS": {
            "directory_path_example": "PROJ/CMIP/NCAR/CESM2/amip/r1i1p1f1/tas/gn/v20240101",
            "directory_path_template": "<mip_era><activity_id><institution_id><source_id><experiment_id><variant_label>",
            "filename_example": "tas_mon_CESM2_amip_r1i1p1f1_gn.nc",
            "filename_template": "<variable_id><frequency><source_id><experiment_id><variant_label><grid_label>",
        },
    }
}

_CV_WITHOUT_DRS: dict = _MINIMAL_CV  # no DRS section


class TestDrsTemplates(unittest.TestCase):
    """GAP-05: CV DRS section templates override hard-coded defaults.

    CMOR3 reference: ``cmor.c`` DRS section handling; GitHub issue #834.
    Priority chain (highest → lowest):
      1. User-supplied ``output_path_template`` / ``output_file_template``
      2. CV ``DRS.directory_path_template`` / ``DRS.filename_template``
      3. Hard-coded ``DEFAULT_OUTPUT_PATH_TEMPLATE`` / ``DEFAULT_OUTPUT_FILE_TEMPLATE``
    """

    # -----------------------------------------------------------------------
    # ControlledVocabulary.drs_templates() unit tests
    # -----------------------------------------------------------------------

    def test_drs_templates_returns_both_when_defined(self):
        cv = ControlledVocabulary(_CV_WITH_DRS)
        path_tmpl, file_tmpl = cv.drs_templates()
        self.assertEqual(
            path_tmpl,
            "<mip_era><activity_id><institution_id><source_id><experiment_id><variant_label>",
        )
        self.assertEqual(
            file_tmpl,
            "<variable_id><frequency><source_id><experiment_id><variant_label><grid_label>",
        )

    def test_drs_templates_returns_none_when_no_drs_section(self):
        cv = ControlledVocabulary(_CV_WITHOUT_DRS)
        path_tmpl, file_tmpl = cv.drs_templates()
        self.assertIsNone(path_tmpl)
        self.assertIsNone(file_tmpl)

    def test_drs_templates_returns_none_for_missing_individual_keys(self):
        cv_partial = ControlledVocabulary(
            {
                "CV": {
                    **_MINIMAL_CV["CV"],
                    "DRS": {"directory_path_example": "example/only/no/templates"},
                }
            }
        )
        path_tmpl, file_tmpl = cv_partial.drs_templates()
        self.assertIsNone(path_tmpl)
        self.assertIsNone(file_tmpl)

    def test_drs_templates_ignores_non_mapping_drs_value(self):
        cv_bad = ControlledVocabulary(
            {"CV": {**_MINIMAL_CV["CV"], "DRS": "not-a-dict"}}
        )
        path_tmpl, file_tmpl = cv_bad.drs_templates()
        self.assertIsNone(path_tmpl)
        self.assertIsNone(file_tmpl)

    # -----------------------------------------------------------------------
    # build_output_path priority-chain integration tests
    # -----------------------------------------------------------------------

    def _make_dataset_and_variable(self, cv_dict: dict, tmp: Path, extra: dict | None = None):
        """Helper: build a DatasetInfo + Variable from a given CV dict."""
        project = _make_project(tmp, cv_dict, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES)
        dataset_attrs = {**_MINIMAL_DATASET, **(extra or {})}
        dataset_info = project.dataset_info(dataset_attrs)
        variable = project.variable("tas")
        return dataset_info, variable

    def test_cv_drs_path_template_used_when_no_user_override(self):
        """When no user template is set, build_output_path uses the CV DRS template."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_info, variable = self._make_dataset_and_variable(
                _CV_WITH_DRS, tmp
            )
            path = build_output_path(dataset_info, variable)
        # CV template: <mip_era><activity_id><institution_id><source_id>
        #              <experiment_id><variant_label>
        # With _MINIMAL_DATASET values:
        #   mip_era=CMIP7, activity_id=CMIP, institution_id=NCAR,
        #   source_id=CESM2, experiment_id=amip, variant_label=r1i1p1f1
        directory = str(path.parent)
        self.assertIn("CMIP7", directory)
        self.assertIn("CMIP", directory)
        self.assertIn("NCAR", directory)
        self.assertIn("CESM2", directory)
        self.assertIn("amip", directory)

    def test_cv_drs_file_template_used_when_no_user_override(self):
        """build_output_path uses the CV DRS filename template."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_info, variable = self._make_dataset_and_variable(
                _CV_WITH_DRS, tmp
            )
            path = build_output_path(dataset_info, variable)
        # CV filename template: <variable_id><frequency><source_id>
        #                       <experiment_id><variant_label><grid_label>
        filename = path.name
        self.assertIn("tas", filename)
        self.assertIn("mon", filename)
        self.assertIn("CESM2", filename)

    def test_user_path_template_overrides_cv_drs(self):
        """A user-supplied output_path_template takes priority over the CV DRS."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_info, variable = self._make_dataset_and_variable(
                _CV_WITH_DRS,
                tmp,
                extra={"output_path_template": "<source_id><experiment_id>"},
            )
            path = build_output_path(dataset_info, variable)
        directory = str(path.parent)
        # The user template only has source_id and experiment_id.
        # The CV template would also have mip_era, activity_id etc.
        self.assertIn("CESM2", directory)
        self.assertIn("amip", directory)
        # mip_era should NOT appear in a path produced from the narrow user template.
        self.assertNotIn("CMIP7", directory)

    def test_user_file_template_overrides_cv_drs(self):
        """A user-supplied output_file_template takes priority over the CV DRS."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_info, variable = self._make_dataset_and_variable(
                _CV_WITH_DRS,
                tmp,
                extra={"output_file_template": "<source_id><variable_id>"},
            )
            path = build_output_path(dataset_info, variable)
        filename = path.name
        self.assertIn("CESM2", filename)
        self.assertIn("tas", filename)
        # The CV template would include frequency ('mon'); the user template does not.
        self.assertNotIn("mon", filename)

    def test_hard_coded_default_used_when_no_cv_drs_and_no_user_override(self):
        """When neither the CV nor the user supplies templates, fall back to defaults."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_info, variable = self._make_dataset_and_variable(
                _CV_WITHOUT_DRS, tmp
            )
            path = build_output_path(dataset_info, variable)
        # The default path template starts with <drs_specs><mip_era>…
        # With _MINIMAL_DATASET there is no drs_specs/version, but mip_era=CMIP7
        # and the other tokens are present.
        directory = str(path.parent)
        self.assertIn("CMIP7", directory)

    def test_cmip7_project_uses_cv_drs_templates(self):
        """The real CMIP7 project tables carry a DRS section that is used."""
        from table_helpers import cmip7_project

        project = cmip7_project()
        cv_path_tmpl, cv_file_tmpl = project.cv.drs_templates()

        # CMIP7 DRS section defines both templates.
        self.assertIsNotNone(cv_path_tmpl)
        self.assertIsNotNone(cv_file_tmpl)
        # Templates contain the expected tokens.
        self.assertIn("<mip_era>", cv_path_tmpl)
        self.assertIn("<variable_id>", cv_file_tmpl)

    def test_drcdp_project_uses_cv_drs_templates(self):
        """The DRCDP project tables also carry a DRS section."""
        from table_helpers import drcdp_project

        project = drcdp_project()
        cv_path_tmpl, cv_file_tmpl = project.cv.drs_templates()

        self.assertIsNotNone(cv_path_tmpl)
        self.assertIsNotNone(cv_file_tmpl)

    def test_obs4mips_project_has_no_drs_templates(self):
        """obs4MIPs has no DRS section; drs_templates() returns (None, None)."""
        from table_helpers import obs4mips_project

        project = obs4mips_project()
        cv_path_tmpl, cv_file_tmpl = project.cv.drs_templates()

        self.assertIsNone(cv_path_tmpl)
        self.assertIsNone(cv_file_tmpl)


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------------------------------------------
# GAP-06 — CV JSON structure validation (double-nesting detection)
# ---------------------------------------------------------------------------


class TestCVStructureValidation(unittest.TestCase):
    """GAP-06: Double-nested CV entries emit RuntimeWarning and are reported.

    CMOR3 reference: ``cmor_CV.c``; GitHub issue #829
    (obs4MIPs double-nested ``nominal_resolution`` silently skipped validation).

    The double-nesting pattern is::

        "nominal_resolution": {
            "nominal_resolution": ["0.5 km", "1 km", …]
        }

    instead of the correct::

        "nominal_resolution": ["0.5 km", "1 km", …]
    """

    # -----------------------------------------------------------------------
    # validate_structure() unit tests
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_broken_cv(data: dict) -> ControlledVocabulary:
        """Construct a deliberately malformed CV, suppressing the expected warning.

        Tests that exercise ``validate_structure()`` or the validation bypass
        behaviour call this helper so they don't leak RuntimeWarnings into the
        test output.  Tests that specifically verify *warning emission* use
        ``warnings.catch_warnings(record=True)`` directly.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return ControlledVocabulary(data)

    def test_well_formed_cv_has_no_issues(self):
        cv = ControlledVocabulary(
            {
                "CV": {
                    "nominal_resolution": ["100 km", "250 km"],
                    "institution_id": {"NCAR": "National Center …"},
                    "mip_era": "CMIP7",
                }
            }
        )
        self.assertEqual(cv.validate_structure(), [])

    def test_double_nested_entry_is_detected(self):
        cv = self._make_broken_cv(
            {
                "CV": {
                    "nominal_resolution": {
                        "nominal_resolution": ["0.5 km", "1 km", "10 km"]
                    }
                }
            }
        )
        issues = cv.validate_structure()
        self.assertEqual(len(issues), 1)
        self.assertIn("nominal_resolution", issues[0])
        self.assertIn("double-nested", issues[0])

    def test_multiple_double_nested_entries_all_detected(self):
        cv = self._make_broken_cv(
            {
                "CV": {
                    "nominal_resolution": {"nominal_resolution": ["10 km"]},
                    "activity_id": {"activity_id": ["CMIP", "ScenarioMIP"]},
                    "institution_id": {"NCAR": "National Center …"},  # not double-nested
                }
            }
        )
        issues = cv.validate_structure()
        self.assertEqual(len(issues), 2)
        keys_mentioned = {line.split("'")[1] for line in issues}
        self.assertEqual(keys_mentioned, {"nominal_resolution", "activity_id"})

    def test_mapping_with_multiple_keys_is_not_double_nested(self):
        """A mapping value with more than one key is a lookup table, not a bug."""
        cv = ControlledVocabulary(
            {
                "CV": {
                    "experiment_id": {
                        "historical": {"description": "historical"},
                        "amip": {"description": "amip"},
                    }
                }
            }
        )
        self.assertEqual(cv.validate_structure(), [])

    def test_mapping_with_different_key_is_not_double_nested(self):
        """A mapping value whose single key differs from the parent is a lookup table."""
        cv = ControlledVocabulary(
            {"CV": {"institution_id": {"NCAR": "National Center …"}}}
        )
        self.assertEqual(cv.validate_structure(), [])

    def test_list_value_is_not_double_nested(self):
        cv = ControlledVocabulary({"CV": {"activity_id": ["CMIP", "ScenarioMIP"]}})
        self.assertEqual(cv.validate_structure(), [])

    def test_string_value_is_not_double_nested(self):
        cv = ControlledVocabulary({"CV": {"mip_era": "CMIP7"}})
        self.assertEqual(cv.validate_structure(), [])

    def test_issue_message_mentions_github_reference(self):
        """Issue description links to the bug report for user guidance."""
        cv = self._make_broken_cv(
            {"CV": {"nominal_resolution": {"nominal_resolution": ["10 km"]}}}
        )
        issues = cv.validate_structure()
        self.assertTrue(any("829" in issue for issue in issues))

    def test_issue_message_explains_validation_consequence(self):
        """Users are told that validation may silently pass any value."""
        cv = self._make_broken_cv(
            {"CV": {"nominal_resolution": {"nominal_resolution": ["10 km"]}}}
        )
        issues = cv.validate_structure()
        self.assertTrue(
            any("silently" in issue.lower() for issue in issues)
        )

    # -----------------------------------------------------------------------
    # RuntimeWarning emission tests
    # -----------------------------------------------------------------------

    def test_double_nested_entry_emits_runtime_warning(self):
        """ControlledVocabulary.__init__ emits RuntimeWarning for double-nesting."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ControlledVocabulary(
                {
                    "CV": {
                        "nominal_resolution": {
                            "nominal_resolution": ["0.5 km", "10 km"]
                        }
                    }
                }
            )
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime_warnings), 1)
        self.assertIn("nominal_resolution", str(runtime_warnings[0].message))

    def test_multiple_double_nested_entries_emit_one_warning_each(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ControlledVocabulary(
                {
                    "CV": {
                        "nominal_resolution": {"nominal_resolution": ["10 km"]},
                        "activity_id": {"activity_id": ["CMIP"]},
                    }
                }
            )
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime_warnings), 2)

    def test_well_formed_cv_emits_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ControlledVocabulary(
                {"CV": {"nominal_resolution": ["100 km", "250 km"]}}
            )
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime_warnings), 0)

    def test_from_file_emits_warning_for_double_nested_cv(self):
        """Warning is emitted when loading a double-nested CV from disk."""
        with tempfile.TemporaryDirectory() as tmp_str:
            cv_path = Path(tmp_str) / "bad_cv.json"
            cv_path.write_text(
                json.dumps(
                    {
                        "CV": {
                            "nominal_resolution": {
                                "nominal_resolution": ["10 km", "50 km"]
                            },
                            "institution_id": {"NCAR": "National Center …"},
                        }
                    }
                )
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ControlledVocabulary.from_file(cv_path)

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime_warnings), 1)
        self.assertIn("nominal_resolution", str(runtime_warnings[0].message))

    # -----------------------------------------------------------------------
    # Interaction with validate_dataset_values
    # -----------------------------------------------------------------------

    def test_double_nested_nominal_resolution_silently_accepts_bad_value(self):
        """Demonstrate the bug: double-nesting causes an incorrect value to be accepted.

        With correct nesting, only values in the allowed list pass.
        With double-nesting, the check becomes "is the submitted value a key
        of the inner dict?" — so the key name itself (``"nominal_resolution"``)
        is mistakenly accepted, and the true allowed values (like ``"100 km"``)
        are rejected, which is the opposite of the intended behaviour.
        """
        cv_correct = ControlledVocabulary(
            {"CV": {"nominal_resolution": ["100 km", "250 km"]}}
        )
        # Correct CV: "100 km" passes, the key-name string does not.
        cv_correct.validate_dataset_values({"nominal_resolution": "100 km"})
        with self.assertRaises(ControlledVocabularyError):
            cv_correct.validate_dataset_values({"nominal_resolution": "nominal_resolution"})

        cv_broken = self._make_broken_cv(
            {"CV": {"nominal_resolution": {"nominal_resolution": ["100 km"]}}}
        )
        # Broken CV: "100 km" is wrongly REJECTED (it is not a key of the
        # inner dict), while "nominal_resolution" is wrongly ACCEPTED (it IS
        # a key of the inner dict).
        with self.assertRaises(ControlledVocabularyError):
            cv_broken.validate_dataset_values({"nominal_resolution": "100 km"})
        # The key-name string passes silently — wrong.
        cv_broken.validate_dataset_values({"nominal_resolution": "nominal_resolution"})

    # -----------------------------------------------------------------------
    # Real project CV checks
    # -----------------------------------------------------------------------

    def test_cmip7_cv_has_no_structural_issues(self):
        """The shipped CMIP7 CV should be well-formed."""
        from table_helpers import cmip7_project

        project = cmip7_project()
        issues = project.cv.validate_structure()
        self.assertEqual(issues, [], msg=f"CMIP7 CV structural issues: {issues}")

    def test_obs4mips_cv_has_no_structural_issues(self):
        from table_helpers import obs4mips_project

        project = obs4mips_project()
        issues = project.cv.validate_structure()
        self.assertEqual(issues, [], msg=f"obs4MIPs CV structural issues: {issues}")

    def test_drcdp_cv_has_no_structural_issues(self):
        from table_helpers import drcdp_project

        project = drcdp_project()
        issues = project.cv.validate_structure()
        self.assertEqual(issues, [], msg=f"DRCDP CV structural issues: {issues}")


if __name__ == "__main__":
    unittest.main()
