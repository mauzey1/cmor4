"""Axis time metadata must survive assembly, validation, and writing."""

import numpy as np
import pytest
import xarray as xr

import cmor
from cmor import _assembly
from cmor.axis import validate_time_interval
from test_project_tables import _build_project


@pytest.fixture
def project(tmp_path):
    return _build_project(
        tmp_path,
        cv={
            "CV": {
                "required_global_attributes": ["frequency"],
                "frequency": {
                    "mon": {"approx_interval": 30},
                    "day": {"approx_interval": 1},
                    "fx": {},
                    "custom": {"approx_interval": 2},
                },
            }
        },
        variable_entries={
            "sample": {"dimensions": ["time"], "units": "K"},
            "daily": {"dimensions": ["time"], "units": "K", "frequency": "day"},
            "fixed": {"dimensions": [], "units": "K"},
        },
        coordinate_entries={"time": {"axis": "T", "standard_name": "time"}},
    )


def time_axis(project, **updates):
    values = dict(
        values=[15.0, 45.0],
        units="days since 2001-01-01",
        calendar="360_day",
        frequency="mon",
    )
    values.update(updates)
    return project.axis("time", **values)


def test_axis_metadata_reaches_globals_paths_and_final_validation(project, tmp_path):
    info = project.dataset_info({
        "outpath": str(tmp_path),
        "output_path_template": "<frequency>",
        "output_file_template": "<variable_id><frequency><time_range>",
    })
    variable = project.variable("sample")
    axis = time_axis(project, values=[59.0, 89.0])
    ctx = _assembly.validate_metadata(info, variable, [axis])
    assert ctx.dataset.project is project
    assert info.frequency is None
    assert ctx.dataset.frequency == "mon"
    path = cmor.create_dataset(info, variable, [axis], np.ones(2))
    assert path == tmp_path / "mon" / "sample_mon_200102-200103.nc"
    with xr.open_dataset(path, decode_times=False) as ds:
        assert ds.attrs["frequency"] == "mon"
        assert "calendar" not in ds.attrs
        assert ds.time.attrs["calendar"] == "360_day"
        assert "frequency" not in ds.time.attrs
        assert cmor.build_output_path(info, variable, ds) == path


@pytest.mark.parametrize("frequency", [None, "day"])
def test_variable_table_frequency_default(project, tmp_path, frequency):
    axis = time_axis(project, values=[0.5, 1.5], frequency=frequency)
    path = cmor.create_dataset(
        project.dataset_info({}),
        project.variable("daily"),
        [axis],
        np.ones(2),
        path=tmp_path / "daily.nc",
    )
    with xr.open_dataset(path, decode_times=False) as ds:
        assert ds.attrs["frequency"] == "day"


def test_frequency_conflict_rejected_with_or_without_dataset(project):
    axis = time_axis(project)
    variable = project.variable("daily")
    for info in (None, project.dataset_info({})):
        with pytest.raises(cmor.TableValidationError, match="frequency"):
            project._validate_dataset(info, variable, [axis])


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"frequency": "invalid"}, cmor.ControlledVocabularyError),
        ({"frequency": "day"}, cmor.TableValidationError),
    ],
)
def test_final_global_validation_still_rejects_overrides(
    project, tmp_path, overrides, error
):
    path = tmp_path / "invalid.nc"
    with pytest.raises(error, match="frequency"):
        cmor.create_dataset(
            project.dataset_info({}),
            project.variable("sample"),
            [time_axis(project)],
            np.ones(2),
            attrs=overrides,
            path=path,
        )
    assert not path.exists()


def test_calendar_used_for_interval_arithmetic(project):
    # cftime permits months-since units only with 360_day. Reading the wrong
    # calendar would fall back to numeric differences and report one day.
    axis = time_axis(
        project,
        values=[0.0, 1.0],
        units="months since 2001-01-01",
    )
    validate_time_interval(
        project.dataset_info({}), project.variable("sample"), axis, axis.values_array()
    )
    assert axis.attributes()["calendar"] == "360_day"


def test_cv_interval_tolerances_are_used(project):
    axis = time_axis(project, values=[0.0, 2.0], frequency="custom")
    info = project.dataset_info({})
    variable = project.variable("sample")
    validate_time_interval(info, variable, axis, axis.values_array())
    with pytest.raises(cmor.AxisValidationError, match="Time interval mismatch"):
        validate_time_interval(info, variable, axis, np.array([0.0, 3.0]))


@pytest.mark.parametrize("values", [[], [0.0], [0.0, 1.0]])
def test_missing_frequency_rejected(project, values):
    axis = time_axis(project, values=values, frequency=None)
    with pytest.raises(cmor.AxisValidationError, match="frequency"):
        _assembly.validate_metadata(
            cmor.DatasetInfo(), project.variable("sample"), [axis]
        )


def test_fixed_field_needs_no_time_metadata(project, tmp_path):
    path = cmor.create_dataset(
        project.dataset_info({}),
        project.variable("fixed"),
        [],
        np.array(1.0),
        path=tmp_path / "fixed.nc",
    )
    with xr.open_dataset(path) as ds:
        assert ds.attrs["frequency"] == "fx"
        assert "time" not in ds
        assert "calendar" not in ds.attrs


def test_default_calendar_written_on_time_coordinate(project, tmp_path):
    axis = time_axis(project, calendar=None)
    path = cmor.create_dataset(
        project.dataset_info({}),
        project.variable("sample"),
        [axis],
        np.ones(2),
        path=tmp_path / "standard.nc",
    )
    with xr.open_dataset(path, decode_times=False) as ds:
        assert ds.time.attrs["calendar"] == "standard"


def test_factory_rejects_invalid_calendar(project):
    with pytest.raises(cmor.AxisValidationError, match="martian"):
        time_axis(project, calendar="martian")


def test_final_validation_rechecks_calendar(project):
    axis = time_axis(project).updated(calendar="martian")
    with pytest.raises(cmor.AxisValidationError, match="martian"):
        project._validate_dataset(
            project.dataset_info({}), project.variable("sample"), [axis]
        )


def test_different_axis_frequencies_rejected():
    axes = [
        cmor.Axis(name="time", frequency="mon"),
        cmor.Axis(name="time1", frequency="day"),
    ]
    with pytest.raises(ValueError, match="agree"):
        _assembly.validate_metadata(
            cmor.DatasetInfo(), cmor.Variable(name="sample"), axes
        )


def test_legacy_dataset_calendar_cannot_silently_change_dates(project):
    with pytest.raises(
        cmor.AxisValidationError, match="time axis instead of DatasetInfo"
    ):
        _assembly.validate_metadata(
            cmor.DatasetInfo(calendar="360_day"),
            project.variable("sample"),
            [time_axis(project)],
        )


def test_time_table_selector_resolves_axis_frequency_before_cv_validation(project):
    axis = cmor.Axis(
        name="sample_time",
        table_entry="time",
        values=[0.5, 1.5],
        units="days since 2000-01-01",
        frequency="day",
        calendar="noleap",
    )
    ctx = _assembly.validate_metadata(
        project.dataset_info({}), project.variable("sample"), [axis]
    )
    assert ctx.dataset.frequency == "day"
    assert ctx.axes[0].calendar == "noleap"


@pytest.mark.parametrize("calendar", [None, "360_day"])
def test_calendar_in_attrs_rejected(calendar):
    with pytest.raises(ValueError, match="Pass calendar="):
        cmor.Axis(name="time", calendar=calendar, attrs={"calendar": "noleap"})


def test_reference_calendar_inherited_only_during_preparation(project):
    time = time_axis(project)
    reference = cmor.Axis(
        name="reftime",
        standard_name="forecast_reference_time",
        axis="T",
        values=[0.0],
        units=time.units,
        scalar=True,
    )
    assert reference.calendar is None
    ctx = _assembly.validate_metadata(
        cmor.DatasetInfo(), project.variable("sample"), [time, reference]
    )
    assert ctx.axes[1].calendar == "360_day"
    assert reference.calendar is None
    explicit = reference.updated(calendar="noleap")
    ctx = _assembly.validate_metadata(
        cmor.DatasetInfo(), project.variable("sample"), [time, explicit]
    )
    assert ctx.axes[1].calendar == "noleap"
