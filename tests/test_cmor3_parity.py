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

