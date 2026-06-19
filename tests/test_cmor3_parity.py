"""CMOR3 validation parity tests.

Verifies that CMOR4 replicates the validation behaviour present in CMOR3
but absent from the initial CMOR4 implementation.  Each test class covers
one functional area; the list below maps them to the corresponding CMOR3
source references.

Coverage
--------
1.  Frequency required when a time axis is present
    CMOR3: ``Src/cmor_variables.c`` frequency check;
           ``Test/test_cmor_frequency_required.py``

2.  ``external_variables`` global attribute written from ``cell_measures``
    CMOR3: ``Src/cmor.c``; ``Test/test_python_CMIP6_CV_externalvariables.py``

3.  Variant index (RIPF) format and overflow validation
    CMOR3: ``Src/cmor_CV.c::_CV_ValidateVariantLabel``;
           ``Test/test_python_CMIP6_CV_longrealizationindex.py``

4.  ``forcing`` attribute terms validated against the CV forcing list
    CMOR3: ``Src/cmor.c::cmor_check_forcing_validity``

5.  DRS path/filename templates read from the CV ``DRS`` section
    CMOR3: ``Src/cmor.c`` DRS section handling; GitHub issue #834

6.  CV JSON double-nesting detected and reported
    CMOR3: ``Src/cmor_CV.c``; GitHub issue #829

7.  ``history`` attribute uses actual ``Conventions`` and ``mip_era`` tokens
    CMOR3: ``Test/test_cmor_CMIP7.py``

8.  Hierarchical nested CV entries inject leaf attributes into the dataset
    CMOR3: ``Src/cmor_CV.c::_CV_checkGblAttributes``;
           ``Test/test_python_CMIP6_CV_hierarchicalattr.py``

9.  ``grid_label`` regex fallback when the CV does not define an allowed set
    CMOR3: ``cmor.c`` hard-coded grid_label check;
           ``Test/test_python_CMIP6_CV_badgridgr.py``,
           ``Test/test_python_CMIP6_CV_badgridlabel.py``

10. ``create_subdirectories=False`` requires the output directory to exist
    CMOR3: ``Test/test_python_CMIP6_CV_baddirectory.py``
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from cmor4 import Axis, ControlledVocabulary, ProjectTables, Variable
from cmor4 import DatasetInfo, Variable
from cmor4._axis_validation import _validate_time_interval, _is_time_axis
from cmor4.core import _collect_external_variables, create_dataset, build_output_path
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
# ---------------------------------------------------------------------------
# 1. Frequency required when a time axis is present
# ---------------------------------------------------------------------------


class TestFrequencyRequired(unittest.TestCase):
    """A time axis without a declared frequency must raise AxisValidationError."""

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
        with self.assertRaises(AxisValidationError) as ctx:
            _validate_time_interval(DatasetInfo(institution_id="NCAR"), None, axis, axis.values_array())
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
        with self.assertRaises(AxisValidationError) as ctx:
            _validate_time_interval(DatasetInfo(institution_id="NCAR"), None, axis, axis.values_array())
        self.assertIn("frequency", str(ctx.exception).lower())

    def test_empty_string_frequency_raises(self):
        """Explicit empty string treated same as absent."""
        axis = self._time_axis()
        with self.assertRaises(AxisValidationError):
            _validate_time_interval(DatasetInfo(frequency=""), None, axis, axis.values_array())

    def test_error_message_contains_guidance(self):
        """Error message should guide the user toward setting frequency.

        A non-empty dataset is required to trigger the check; an empty dict
        is the structure-only sentinel used by validate_components.
        """
        axis = self._time_axis()
        with self.assertRaises(AxisValidationError) as ctx:
            _validate_time_interval(DatasetInfo(institution_id="NCAR"), None, axis, axis.values_array())
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
            DatasetInfo(frequency="mon", calendar="360_day"),
            None,
            axis,
            axis.values_array(),
        )

    def test_frequency_in_variable_passes(self):
        """frequency carried on the variable → no error."""
        axis = self._time_axis()
        _validate_time_interval(
            None,
            Variable(name="time", frequency="mon"),
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
        with self.assertRaises(AxisValidationError):
            validate_axes(DatasetInfo(institution_id="NCAR"), None, [time_axis])

    def test_validate_axes_passes_for_lat_without_frequency(self):
        """Non-time axes are not affected by the frequency requirement."""
        from cmor4._axis_validation import validate_axes

        lat_axis = Axis(
            name="lat",
            axis="Y",
            units="degrees_north",
            values=[-45.0, 45.0],
        )
        validate_axes(None, None, [lat_axis])  # no error expected

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
# ---------------------------------------------------------------------------
# 2. external_variables global attribute from cell_measures
# ---------------------------------------------------------------------------


class TestExternalVariables(unittest.TestCase):
    """cell_measures references to absent variables → external_variables attr."""

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
        """
        Build a minimal dataset with the given cell_measures and return global attrs.
        """
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
        """
        If areacello is provided as a coordinate it does not appear in
        external_variables.
        """
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
# ---------------------------------------------------------------------------
# 3. Variant index (RIPF) format and overflow validation
# ---------------------------------------------------------------------------


class TestVariantIndexValidation(unittest.TestCase):
    """RIPF indices must be valid positive integers within INT32 range.

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

    def test_explicit_malformed_variant_label_caught_by_cv_validate_dataset_values(
        self,
    ):
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
                    "variant_label": [
                        r"r[[:digit:]]+i[[:digit:]]+p[[:digit:]]+f[[:digit:]]+"
                    ],
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
                    "forcing_index": "r1",  # 'r' prefix is wrong for forcing_index
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
# ---------------------------------------------------------------------------
# 4. forcing attribute terms validated against the CV forcing list
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
    """forcing attribute tokens must be in the CV forcing enumeration.

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
        cv.validate_forcing_terms({"forcing": "GHG (note BADTERM here) Oz"})

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
# ---------------------------------------------------------------------------
# 5. DRS path/filename templates read from the CV DRS section
# ---------------------------------------------------------------------------

_CV_WITH_DRS: dict = {
    "CV": {
        **_MINIMAL_CV["CV"],
        "DRS": {
            "directory_path_example": "PROJ/CMIP/NCAR/CESM2/amip/r1i1p1f1/tas/gn/"
            "v20240101",
            "directory_path_template": "<mip_era><activity_id><institution_id>"
            "<source_id><experiment_id>"
            "<variant_label>",
            "filename_example": "tas_mon_CESM2_amip_r1i1p1f1_gn.nc",
            "filename_template": "<variable_id><frequency><source_id><experiment_id>"
            "<variant_label><grid_label>",
        },
    }
}

_CV_WITHOUT_DRS: dict = _MINIMAL_CV  # no DRS section


class TestDrsTemplates(unittest.TestCase):
    """CV DRS section templates override hard-coded defaults.

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
            "<mip_era><activity_id><institution_id><source_id><experiment_id>"
            "<variant_label>",
        )
        self.assertEqual(
            file_tmpl,
            "<variable_id><frequency><source_id><experiment_id><variant_label>"
            "<grid_label>",
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

    def _make_dataset_and_variable(
        self, cv_dict: dict, tmp: Path, extra: dict | None = None
    ):
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
            dataset_info, variable = self._make_dataset_and_variable(_CV_WITH_DRS, tmp)
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
            dataset_info, variable = self._make_dataset_and_variable(_CV_WITH_DRS, tmp)
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
        """
        When neither the CV nor the user supplies templates, fall back to defaults.
        """
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
# ---------------------------------------------------------------------------
# 6. CV JSON double-nesting detected and reported
# ---------------------------------------------------------------------------


class TestCVStructureValidation(unittest.TestCase):
    """Double-nested CV entries emit RuntimeWarning and are reported by
    validate_structure().

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
                    "institution_id": {
                        "NCAR": "National Center …"
                    },  # not double-nested
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
        """
        A mapping value whose single key differs from the parent is a lookup table.
        """
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
        self.assertTrue(any("silently" in issue.lower() for issue in issues))

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
            ControlledVocabulary({"CV": {"nominal_resolution": ["100 km", "250 km"]}})
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
            cv_correct.validate_dataset_values(
                {"nominal_resolution": "nominal_resolution"}
            )

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

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 7. history attribute uses actual Conventions and mip_era tokens
# ---------------------------------------------------------------------------

import re as _re  # noqa: E402 (appended)


class TestHistoryAttribute(unittest.TestCase):
    """history attribute uses actual Conventions and mip_era tokens, not hard-coded
    strings.

    CMOR3 reference: ``Test/test_cmor_CMIP7.py``::

        self.assertIn(
            f"CMOR rewrote data to be consistent with {conventions} "
            "and CMIP7 data requirements.",
            ds.getncattr("history"),
        )

    The history string must therefore reflect the *actual* Conventions and
    mip_era values from the dataset rather than hard-coded project constants.
    """

    # Regex matching the full history format:
    #   "<ISO-8601Z> ; CMOR rewrote data to be consistent with "
    #   "<X> and <Y> data requirements."
    _HISTORY_RE = _re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
        r" ; CMOR rewrote data to be consistent with "
        r".+ and .+ data requirements\.$"
    )

    def _make_dataset(self, tmp: Path, extra: dict | None = None) -> xr.Dataset:
        """Build a minimal dataset and return the resulting xarray Dataset."""
        import numpy as np
        from cmor4.core import create_dataset

        project = _make_project(
            tmp, _MINIMAL_CV, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
        )
        dataset_info = project.dataset_info({**_MINIMAL_DATASET, **(extra or {})})
        variable = project.variable("tas")
        time_axis = project.axis(
            "time", values=[15.0, 45.0], units="days since 2000-01-01"
        )
        lat_axis = project.axis("lat", values=[-45.0, 45.0])
        lon_axis = project.axis("lon", values=[0.0, 90.0])
        data = np.ones((2, 2, 2))
        return create_dataset(
            dataset_info, variable, [time_axis, lat_axis, lon_axis], data
        )

    # -----------------------------------------------------------------------
    # Format correctness
    # -----------------------------------------------------------------------

    def test_history_matches_cmor3_format(self):
        """The full history string must match the CMOR3 pattern exactly."""
        with tempfile.TemporaryDirectory() as tmp_str:
            ds = self._make_dataset(Path(tmp_str))
        self.assertRegex(ds.attrs["history"], self._HISTORY_RE)

    def test_history_contains_actual_conventions(self):
        """Conventions token in history must match the actual Conventions attribute."""
        with tempfile.TemporaryDirectory() as tmp_str:
            ds = self._make_dataset(Path(tmp_str))
        conventions = ds.attrs["Conventions"]
        self.assertIn(
            f"be consistent with {conventions} and",
            ds.attrs["history"],
        )

    def test_history_contains_actual_mip_era(self):
        """mip_era token in history must reflect the dataset's mip_era."""
        with tempfile.TemporaryDirectory() as tmp_str:
            ds = self._make_dataset(Path(tmp_str))
        mip_era = ds.attrs.get("mip_era", "CMIP")
        self.assertIn(
            f"and {mip_era} data requirements.",
            ds.attrs["history"],
        )

    def test_history_conventions_not_hardcoded(self):
        """A non-default Conventions value must appear in history, not 'CF-1.12'."""
        cv_custom_conventions = {
            "CV": {
                **_MINIMAL_CV["CV"],
                "Conventions": "CF-1.9",
            }
        }
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, cv_custom_conventions, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            dataset_info = project.dataset_info(_MINIMAL_DATASET)
            variable = project.variable("tas")
            import numpy as np
            from cmor4.core import create_dataset

            time_axis = project.axis(
                "time", values=[15.0, 45.0], units="days since 2000-01-01"
            )
            lat_axis = project.axis("lat", values=[-45.0, 45.0])
            lon_axis = project.axis("lon", values=[0.0, 90.0])
            ds = create_dataset(
                dataset_info,
                variable,
                [time_axis, lat_axis, lon_axis],
                np.ones((2, 2, 2)),
            )
        self.assertIn("CF-1.9", ds.attrs["history"])
        self.assertNotIn("CF-1.12", ds.attrs["history"])

    def test_history_mip_era_not_hardcoded(self):
        """A non-CMIP7 mip_era must appear in history, not the literal 'CMIP7'."""
        cv_custom_mip = {
            "CV": {
                **_MINIMAL_CV["CV"],
                "mip_era": "obs4MIPs",
            }
        }
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, cv_custom_mip, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            dataset_info = project.dataset_info(
                {**_MINIMAL_DATASET, "mip_era": "obs4MIPs"}
            )
            variable = project.variable("tas")
            import numpy as np
            from cmor4.core import create_dataset

            time_axis = project.axis(
                "time", values=[15.0, 45.0], units="days since 2000-01-01"
            )
            lat_axis = project.axis("lat", values=[-45.0, 45.0])
            lon_axis = project.axis("lon", values=[0.0, 90.0])
            ds = create_dataset(
                dataset_info,
                variable,
                [time_axis, lat_axis, lon_axis],
                np.ones((2, 2, 2)),
            )
        self.assertIn("obs4MIPs", ds.attrs["history"])
        self.assertNotIn("CMIP7", ds.attrs["history"])

    def test_history_starts_with_iso8601_date(self):
        """history must begin with a UTC ISO-8601 timestamp."""
        with tempfile.TemporaryDirectory() as tmp_str:
            ds = self._make_dataset(Path(tmp_str))
        self.assertRegex(
            ds.attrs["history"],
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        )

    # -----------------------------------------------------------------------
    # Preservation of user-supplied history
    # -----------------------------------------------------------------------

    def test_user_supplied_history_is_not_overwritten(self):
        """A history value in the dataset must survive into the output unchanged."""
        custom = "Pre-existing processing step applied 2024-01-01."
        with tempfile.TemporaryDirectory() as tmp_str:
            ds = self._make_dataset(Path(tmp_str), extra={"history": custom})
        self.assertEqual(ds.attrs["history"], custom)

    def test_extra_attrs_history_takes_priority(self):
        """history supplied via extra_attrs must override the generated default."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            import numpy as np
            from cmor4.core import create_dataset

            project = _make_project(
                tmp, _MINIMAL_CV, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            dataset_info = project.dataset_info(_MINIMAL_DATASET)
            variable = project.variable("tas")
            time_axis = project.axis(
                "time", values=[15.0, 45.0], units="days since 2000-01-01"
            )
            lat_axis = project.axis("lat", values=[-45.0, 45.0])
            lon_axis = project.axis("lon", values=[0.0, 90.0])
            override = "Custom history override."
            ds = create_dataset(
                dataset_info,
                variable,
                [time_axis, lat_axis, lon_axis],
                np.ones((2, 2, 2)),
                attrs={"history": override},
            )
        self.assertEqual(ds.attrs["history"], override)

    # -----------------------------------------------------------------------
    # CMOR3 parity assertion (mirrors test_cmor_CMIP7.py)
    # -----------------------------------------------------------------------

    def test_cmor3_parity_assertin_passes(self):
        """Replicate the exact assertIn check from CMOR3's test_cmor_CMIP7.py."""
        with tempfile.TemporaryDirectory() as tmp_str:
            ds = self._make_dataset(Path(tmp_str))
        conventions = ds.attrs["Conventions"]
        mip_era = ds.attrs.get("mip_era", "CMIP")
        # This mirrors: self.assertIn(f"CMOR rewrote data to be consistent with
        # {conventions} and CMIP7 data requirements.", ds.getncattr("history"))
        self.assertIn(
            f"CMOR rewrote data to be consistent with {conventions} "
            f"and {mip_era} data requirements.",
            ds.attrs["history"],
        )


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 8. Hierarchical nested CV entries inject leaf attributes into the dataset
# ---------------------------------------------------------------------------

# CV structures that mirror the CMOR3 test_python_CMIP6_CV_hierarchicalattr.py
# scenario.  The CMOR3 test uses a two-level lookup:
#
#   CV:   {"hierarchical_attr_setting": {"information": {leaf_attrs}}}
#   User: {"hierarchical_attr_setting": "information"}
#   Out:  leaf_attrs written as global attributes
#
# This pattern is also used by obs4MIPs site_id to inject per-site location
# metadata.

_CV_WITH_NESTED_ATTRS: dict = {
    "CV": {
        **_MINIMAL_CV["CV"],
        # Two-level lookup: user selects "information" → leaf attrs injected
        "hierarchical_attr_setting": {
            "information": {
                "coder": "Denis Nadeau",
                "creator": "PCMDI",
                "model": "Ocean Model",
                "country": "USA",
            }
        },
        # obs4MIPs-style site location metadata
        "site_id": {
            "AR-SLu": {
                "latitude": "-33.4648",
                "location": "San Luis",
                "longitude": "-66.4598",
            }
        },
    }
}


class TestNestedCVAttributes(unittest.TestCase):
    """Two-level nested CV entries inject leaf attributes into the dataset.

    CMOR3 reference: ``Test/test_python_CMIP6_CV_hierarchicalattr.py``.
    """

    def _cv(self) -> ControlledVocabulary:
        return ControlledVocabulary(_CV_WITH_NESTED_ATTRS)

    # -----------------------------------------------------------------------
    # _add_nested_defaults unit tests (via get_dataset_info)
    # -----------------------------------------------------------------------

    def test_leaf_attributes_injected_when_user_selects_code(self):
        """Selecting a code injects all scalar leaf attributes from the CV entry."""
        cv = self._cv()
        dataset = cv.get_dataset_info({"hierarchical_attr_setting": "information"})
        self.assertEqual(dataset["coder"], "Denis Nadeau")
        self.assertEqual(dataset["creator"], "PCMDI")
        self.assertEqual(dataset["model"], "Ocean Model")
        self.assertEqual(dataset["country"], "USA")

    def test_selector_attribute_itself_is_preserved(self):
        """The user-set selector key is also present in the dataset."""
        cv = self._cv()
        dataset = cv.get_dataset_info({"hierarchical_attr_setting": "information"})
        self.assertEqual(dataset["hierarchical_attr_setting"], "information")

    def test_site_id_location_attrs_injected(self):
        """obs4MIPs-style site_id lookup injects latitude/longitude/location."""
        cv = self._cv()
        dataset = cv.get_dataset_info({"site_id": "AR-SLu"})
        self.assertEqual(dataset["latitude"], "-33.4648")
        self.assertEqual(dataset["location"], "San Luis")
        self.assertEqual(dataset["longitude"], "-66.4598")

    def test_no_injection_when_user_does_not_set_key(self):
        """Leaf attributes are NOT injected when the user omits the selector."""
        cv = self._cv()
        dataset = cv.get_dataset_info({})
        self.assertNotIn("coder", dataset)
        self.assertNotIn("creator", dataset)

    def test_no_injection_for_unknown_code(self):
        """An unrecognised code injects nothing (no KeyError)."""
        cv = self._cv()
        dataset = cv.get_dataset_info({"hierarchical_attr_setting": "nonexistent"})
        self.assertNotIn("coder", dataset)

    def test_user_values_not_overwritten_by_injection(self):
        """Leaf attributes already set by the user are not overwritten (setdefault)."""
        cv = self._cv()
        dataset = cv.get_dataset_info(
            {
                "hierarchical_attr_setting": "information",
                "coder": "override",
            }
        )
        self.assertEqual(dataset["coder"], "override")

    def test_dedicated_handler_values_not_overwritten(self):
        """Attributes set by dedicated handlers (e.g. experiment defaults) win."""
        cv = ControlledVocabulary(
            {
                "CV": {
                    **_MINIMAL_CV["CV"],
                    "experiment_id": {
                        "amip": {
                            "experiment_id": "amip",
                            "activity_id": ["CMIP"],
                            # experiment entry also has 'description'
                            "description": "Experiment description from dedicated "
                            "handler.",
                        }
                    },
                    # A nested entry that would also inject 'description'
                    "profile": {
                        "standard": {
                            "description": "Should NOT win over experiment "
                            "description.",
                            "org": "PCMDI",
                        }
                    },
                }
            }
        )
        dataset = cv.get_dataset_info(
            {
                "experiment_id": "amip",
                "profile": "standard",
            }
        )
        # _add_experiment_defaults runs before _add_nested_defaults, so its
        # setdefault("description", ...) wins.
        self.assertEqual(
            dataset["description"],
            "Experiment description from dedicated handler.",
        )
        # 'org' has no conflict, so it is injected normally.
        self.assertEqual(dataset["org"], "PCMDI")

    def test_entry_with_non_scalar_values_not_injected(self):
        """Entries whose looked-up value contains a list or dict are not injected."""
        cv = ControlledVocabulary(
            {
                "CV": {
                    **_MINIMAL_CV["CV"],
                    "complex_key": {
                        "code_a": {
                            "allowed_values": ["v1", "v2"],  # list → skip injection
                            "scalar_attr": "ok",
                        }
                    },
                }
            }
        )
        dataset = cv.get_dataset_info({"complex_key": "code_a"})
        self.assertNotIn("allowed_values", dataset)
        self.assertNotIn("scalar_attr", dataset)

    # -----------------------------------------------------------------------
    # Exclusion of internal-validation-only CV keys
    # -----------------------------------------------------------------------

    def test_frequency_approx_interval_not_injected(self):
        """CMIP7 frequency's approx_interval must not appear as a global attribute.

        approx_interval is read directly by the axis validator and must not
        leak into output files.
        """
        cv = ControlledVocabulary(
            {
                "CV": {
                    **_MINIMAL_CV["CV"],
                    "frequency": {
                        "mon": {
                            "approx_interval": 30.0,
                            "approx_interval_error": 0.2,
                            "approx_interval_warning": 0.1,
                            "description": "Monthly samples.",
                        },
                    },
                }
            }
        )
        dataset = cv.get_dataset_info({"frequency": "mon"})
        self.assertNotIn("approx_interval", dataset)
        self.assertNotIn("approx_interval_error", dataset)
        self.assertNotIn("approx_interval_warning", dataset)
        # 'frequency' itself is still set (user-supplied value)
        self.assertEqual(dataset.get("frequency"), "mon")

    # -----------------------------------------------------------------------
    # Integration: global attrs written to output dataset
    # -----------------------------------------------------------------------

    def test_hierarchical_leaf_attrs_written_to_netcdf_global_attrs(self):
        """Leaf attributes from nested CV entries appear in the output NetCDF attrs."""
        import numpy as np
        from cmor4.core import create_dataset

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, _CV_WITH_NESTED_ATTRS, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            dataset_info = project.dataset_info(
                {
                    **_MINIMAL_DATASET,
                    "hierarchical_attr_setting": "information",
                }
            )
            variable = project.variable("tas")
            time_axis = project.axis(
                "time", values=[15.0, 45.0], units="days since 2000-01-01"
            )
            lat_axis = project.axis("lat", values=[-45.0, 45.0])
            lon_axis = project.axis("lon", values=[0.0, 90.0])
            ds = create_dataset(
                dataset_info,
                variable,
                [time_axis, lat_axis, lon_axis],
                np.ones((2, 2, 2)),
            )

        self.assertEqual(ds.attrs["coder"], "Denis Nadeau")
        self.assertEqual(ds.attrs["creator"], "PCMDI")
        self.assertEqual(ds.attrs["model"], "Ocean Model")
        self.assertEqual(ds.attrs["country"], "USA")
        self.assertEqual(ds.attrs["hierarchical_attr_setting"], "information")

    def test_cmip7_real_project_frequency_does_not_inject_approx_interval(self):
        """With the real CMIP7 tables, approx_interval must not appear in output."""
        import sys
        import numpy as np
        from cmor4.core import create_dataset

        # table_helpers.py lives in the tests directory; add it to path if needed
        tests_dir = str(Path(__file__).parent)
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from table_helpers import cmip7_project

        # Import the CMIP7-specific helpers from test_cmor4 module
        import test_cmor4 as _tc4

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_path = Path(tmp_str)
            project = cmip7_project()
            variable = project.variable("tas_tavg-h2m-hxy-u", table_id="atmos")
            raw = _tc4.dataset_info(tmp_path)
            info = project.dataset_info(raw)
            ds = create_dataset(
                info,
                variable,
                [_tc4.time_axis(project), *_tc4.horizontal_axes(project)],
                np.ones((2, 2, 2), dtype="f4"),
            )

        self.assertNotIn("approx_interval", ds.attrs)
        self.assertNotIn("approx_interval_error", ds.attrs)
        self.assertNotIn("approx_interval_warning", ds.attrs)


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 9. grid_label regex fallback when the CV does not define an allowed set
# ---------------------------------------------------------------------------


class TestGridLabelFallback(unittest.TestCase):
    """When the CV does not define grid_label, a built-in regex guards the format.

    CMOR3 reference: ``cmor.c`` hard-coded grid_label check;
    ``Test/test_python_CMIP6_CV_badgridgr.py`` (gr-0 rejected),
    ``Test/test_python_CMIP6_CV_badgridlabel.py`` (gs1n is outside enumeration).

    The fallback regex ``^[gcr][a-z0-9]*$`` ensures:
    - labels start with one of the three grid families (g, c, r)
    - only lowercase letters and digits follow (no hyphens, no uppercase,
      no special characters)

    When the CV *does* define ``grid_label`` (enumeration or regex list),
    ``validate_dataset_values`` enforces the CV definition instead and the
    fallback is not applied.
    """

    def _cv_without_grid_label(self) -> ControlledVocabulary:
        """A CV that deliberately omits the grid_label key."""
        return ControlledVocabulary(
            {
                "CV": {
                    "activity_id": ["CMIP"],
                    "institution_id": {"NCAR": "National Center …"},
                    # no grid_label entry
                }
            }
        )

    def _cv_with_grid_label(self) -> ControlledVocabulary:
        """A CV that provides an explicit grid_label enumeration."""
        return ControlledVocabulary(
            {
                "CV": {
                    "activity_id": ["CMIP"],
                    "grid_label": {"gn": "native grid", "gr1": "regridded"},
                }
            }
        )

    # -----------------------------------------------------------------------
    # Fallback fires when CV lacks grid_label
    # -----------------------------------------------------------------------

    def test_hyphen_in_label_rejected_by_fallback(self):
        """gr-0 is rejected — mirrors CMOR3 test_python_CMIP6_CV_badgridgr."""
        cv = self._cv_without_grid_label()
        with self.assertRaises(ControlledVocabularyError) as ctx:
            cv.validate_dataset_values({"grid_label": "gr-0"})
        self.assertIn("gr-0", str(ctx.exception))

    def test_uppercase_label_rejected_by_fallback(self):
        cv = self._cv_without_grid_label()
        with self.assertRaises(ControlledVocabularyError):
            cv.validate_dataset_values({"grid_label": "GN"})

    def test_digit_first_label_rejected_by_fallback(self):
        cv = self._cv_without_grid_label()
        with self.assertRaises(ControlledVocabularyError):
            cv.validate_dataset_values({"grid_label": "1gn"})

    def test_wrong_starting_character_rejected(self):
        """Labels starting with something other than g, c, or r are invalid."""
        cv = self._cv_without_grid_label()
        for bad in ("abc", "xgn", "zonal", "native"):
            with self.subTest(label=bad):
                with self.assertRaises(ControlledVocabularyError):
                    cv.validate_dataset_values({"grid_label": bad})

    def test_special_characters_rejected(self):
        cv = self._cv_without_grid_label()
        for bad in ("gn!", "gr_1", "g.999", "c@n"):
            with self.subTest(label=bad):
                with self.assertRaises(ControlledVocabularyError):
                    cv.validate_dataset_values({"grid_label": bad})

    def test_empty_string_not_checked(self):
        """An empty or None grid_label skips the fallback check."""
        cv = self._cv_without_grid_label()
        cv.validate_dataset_values({"grid_label": ""})
        cv.validate_dataset_values({})

    # -----------------------------------------------------------------------
    # Fallback happy-path: labels that should pass
    # -----------------------------------------------------------------------

    def test_cmip6_style_native_passes(self):
        cv = self._cv_without_grid_label()
        for valid in ("gn", "gr", "gr1", "gr2", "cn", "rn", "gna", "grz"):
            with self.subTest(label=valid):
                cv.validate_dataset_values({"grid_label": valid})

    def test_cmip7_style_numeric_passes(self):
        cv = self._cv_without_grid_label()
        for valid in ("g100", "g101", "g999", "g1"):
            with self.subTest(label=valid):
                cv.validate_dataset_values({"grid_label": valid})

    def test_single_letter_label_passes(self):
        """A label consisting of just the starting character is valid."""
        cv = self._cv_without_grid_label()
        cv.validate_dataset_values({"grid_label": "g"})

    # -----------------------------------------------------------------------
    # CV definition takes precedence over the fallback
    # -----------------------------------------------------------------------

    def test_cv_enumeration_wins_over_fallback(self):
        """When the CV defines grid_label, only CV-listed values are accepted."""
        cv = self._cv_with_grid_label()
        # 'gn' is in the CV → accepted
        cv.validate_dataset_values({"grid_label": "gn"})
        # 'gr' is NOT in the CV (only 'gr1' is) → rejected by CV, not fallback
        with self.assertRaises(ControlledVocabularyError):
            cv.validate_dataset_values({"grid_label": "gr"})

    def test_cv_defined_value_outside_fallback_regex_is_accepted(self):
        """A CV-defined label that wouldn't pass the fallback regex is still accepted.

        This confirms the fallback is skipped when the CV provides a definition —
        even a label that the fallback would reject (e.g. one with a hyphen)
        is fine when it's an explicit CV entry.
        """
        cv = ControlledVocabulary(
            {
                "CV": {
                    "grid_label": {
                        "gn": "native",
                        "custom-label": "custom grid with hyphen in CV",
                    }
                }
            }
        )
        # 'custom-label' is explicitly in the CV → accepted
        cv.validate_dataset_values({"grid_label": "custom-label"})

    def test_cmip7_real_project_grid_label_validated_by_cv(self):
        """With real CMIP7 tables, grid_label is validated against the CV dict."""
        from table_helpers import cmip7_project

        project = cmip7_project()

        # A CMIP7 CV-defined label passes
        project.cv.validate_dataset_values({"grid_label": "g999"})

        # A label not in the CMIP7 CV dict is rejected (by CV, not fallback)
        with self.assertRaises(ControlledVocabularyError):
            project.cv.validate_dataset_values({"grid_label": "gn"})

    def test_fallback_not_applied_when_cmip7_cv_defines_grid_label(self):
        """The fallback must not run when the CMIP7 CV enumerates grid_label.

        This verifies that a CMIP7-format label like 'g100' is NOT
        double-checked by the fallback after passing the CV enumeration.
        """
        from table_helpers import cmip7_project

        project = cmip7_project()
        # 'g100' is in the CMIP7 CV — should pass cleanly, no fallback noise
        project.cv.validate_dataset_values({"grid_label": "g100"})


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 10. create_subdirectories=False requires the output directory to exist
# ---------------------------------------------------------------------------

import numpy as _np  # noqa: E402


class TestCreateSubdirectories(unittest.TestCase):
    """create_subdirectories=False requires the output dir to already exist.

    CMOR3 reference: ``Test/test_python_CMIP6_CV_baddirectory.py``.
    When ``create_subdirectories=0`` and the outpath cannot be created, CMOR3
    errors with "unable to create this directory".  CMOR4 mirrors this by
    raising ``ValueError`` when ``create_subdirectories=False`` and the
    output parent directory is absent.

    The tests pass an explicit ``path=`` argument to ``write_netcdf`` so the
    directory-existence check is exercised independently of ``build_output_path``
    (whose DRS template rendering is project-specific).
    """

    def _build_minimal_ds(self, tmp: Path) -> tuple:
        """Return (xr.Dataset, DatasetInfo, Variable) for the minimal project."""
        from cmor4.core import create_dataset

        project = _make_project(
            tmp, _MINIMAL_CV, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
        )
        dataset_info = project.dataset_info(_MINIMAL_DATASET)
        variable = project.variable("tas")
        time_axis = project.axis(
            "time", values=[15.0, 45.0], units="days since 2000-01-01"
        )
        lat_axis = project.axis("lat", values=[-45.0, 45.0])
        lon_axis = project.axis("lon", values=[0.0, 90.0])
        ds = create_dataset(
            dataset_info,
            variable,
            [time_axis, lat_axis, lon_axis],
            _np.ones((2, 2, 2)),
        )
        return ds, dataset_info, variable

    def _patch_dataset(self, dataset_info, extra: dict):
        """Return a new DatasetInfo with extra keys merged in."""
        from cmor4.dataset import DatasetInfo

        return DatasetInfo(
            {**dict(dataset_info), **extra},
            project=dataset_info.project,
        )

    # -----------------------------------------------------------------------
    # create_subdirectories=False error cases
    # -----------------------------------------------------------------------

    def test_missing_output_dir_raises_when_create_subdirectories_false(self):
        """Non-existent output dir with create_subdirectories=False raises ValueError.

        Mirrors CMOR3 test_python_CMIP6_CV_baddirectory: CMOR3 errors when it
        cannot write to the requested outpath.
        """
        from cmor4.core import write_netcdf

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ds, info, variable = self._build_minimal_ds(tmp)
            patched = self._patch_dataset(info, {"create_subdirectories": False})

            nonexistent = tmp / "missing_dir" / "output.nc"
            with self.assertRaises(ValueError) as ctx:
                write_netcdf(ds, patched, variable, path=nonexistent)

        self.assertIn("does not exist", str(ctx.exception))
        self.assertIn("create_subdirectories", str(ctx.exception))

    def test_error_message_guides_user(self):
        """Error message tells the user how to fix the problem."""
        from cmor4.core import write_netcdf

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ds, info, variable = self._build_minimal_ds(tmp)
            patched = self._patch_dataset(info, {"create_subdirectories": False})

            with self.assertRaises(ValueError) as ctx:
                write_netcdf(ds, patched, variable, path=tmp / "no_dir" / "f.nc")

        self.assertIn("create_subdirectories", str(ctx.exception))

    def test_deeply_nested_missing_dir_raises(self):
        """The check fires even for deeply nested missing paths."""
        from cmor4.core import write_netcdf

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ds, info, variable = self._build_minimal_ds(tmp)
            patched = self._patch_dataset(info, {"create_subdirectories": False})

            nested = tmp / "a" / "b" / "c" / "d" / "output.nc"
            with self.assertRaises(ValueError):
                write_netcdf(ds, patched, variable, path=nested)

    # -----------------------------------------------------------------------
    # create_subdirectories=False happy path
    # -----------------------------------------------------------------------

    def test_existing_output_dir_passes_when_create_subdirectories_false(self):
        """Pre-existing output dir with create_subdirectories=False succeeds."""
        from cmor4.core import write_netcdf

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ds, info, variable = self._build_minimal_ds(tmp)
            patched = self._patch_dataset(info, {"create_subdirectories": False})

            existing_dir = tmp / "existing"
            existing_dir.mkdir()
            output_file = existing_dir / "output.nc"

            path = write_netcdf(ds, patched, variable, path=output_file)
            self.assertTrue(path.exists())

    # -----------------------------------------------------------------------
    # create_subdirectories=True (default) — existing behaviour unchanged
    # -----------------------------------------------------------------------

    def test_missing_dir_created_automatically_by_default(self):
        """Default behaviour creates the output directory tree automatically."""
        from cmor4.core import write_netcdf

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ds, info, variable = self._build_minimal_ds(tmp)

            # No create_subdirectories set — defaults to True
            nonexistent = tmp / "auto" / "created" / "output.nc"
            path = write_netcdf(ds, info, variable, path=nonexistent)
            self.assertTrue(path.exists())

    def test_create_subdirectories_true_creates_dirs(self):
        """Explicit create_subdirectories=True also creates directories."""
        from cmor4.core import write_netcdf

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ds, info, variable = self._build_minimal_ds(tmp)
            patched = self._patch_dataset(info, {"create_subdirectories": True})

            nonexistent = tmp / "auto_true" / "output.nc"
            path = write_netcdf(ds, patched, variable, path=nonexistent)
            self.assertTrue(path.exists())

    # -----------------------------------------------------------------------
    # create_subdirectories is not written as a global attribute
    # -----------------------------------------------------------------------

    def test_create_subdirectories_not_in_output_global_attrs(self):
        """create_subdirectories must never appear as a NetCDF global attribute."""
        from cmor4.core import create_dataset

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, _MINIMAL_CV, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            attrs = {**_MINIMAL_DATASET, "create_subdirectories": False}
            dataset_info = project.dataset_info(attrs)
            variable = project.variable("tas")
            time_axis = project.axis(
                "time", values=[15.0, 45.0], units="days since 2000-01-01"
            )
            lat_axis = project.axis("lat", values=[-45.0, 45.0])
            lon_axis = project.axis("lon", values=[0.0, 90.0])
            ds = create_dataset(
                dataset_info,
                variable,
                [time_axis, lat_axis, lon_axis],
                _np.ones((2, 2, 2)),
            )

        self.assertNotIn("create_subdirectories", ds.attrs)

    def test_outpath_not_in_output_global_attrs(self):
        """outpath must never appear as a NetCDF global attribute (pre-existing)."""
        from cmor4.core import create_dataset

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            project = _make_project(
                tmp, _MINIMAL_CV, _MINIMAL_VARIABLES, _MINIMAL_COORDINATES
            )
            dataset_info = project.dataset_info(_MINIMAL_DATASET)
            variable = project.variable("tas")
            time_axis = project.axis(
                "time", values=[15.0, 45.0], units="days since 2000-01-01"
            )
            lat_axis = project.axis("lat", values=[-45.0, 45.0])
            lon_axis = project.axis("lon", values=[0.0, 90.0])
            ds = create_dataset(
                dataset_info,
                variable,
                [time_axis, lat_axis, lon_axis],
                _np.ones((2, 2, 2)),
            )

        self.assertNotIn("outpath", ds.attrs)


if __name__ == "__main__":
    unittest.main()
