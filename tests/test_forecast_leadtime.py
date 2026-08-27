from __future__ import annotations

import numpy as np
import pytest
from netCDF4 import Dataset

from cmor4 import Axis, DatasetInfo, DatasetWriter, Variable, cmorize
from cmor4.utils.construction import build_axis_mappings, derive_forecast_coords
from cmor4.utils.tables import CoordinateTable


def _time(values=(0.0, 6.0, 12.0), units="hours since 2020-01-01") -> Axis:
    return Axis(
        name="time",
        values=values,
        units=units,
        standard_name="time",
        axis="T",
    )


def _reftime(
    values=(0.0,),
    units="hours since 2020-01-01",
    *,
    scalar=True,
) -> Axis:
    return Axis(
        name="reftime1",
        out_name="reftime",
        values=values,
        units=units,
        standard_name="forecast_reference_time",
        scalar=scalar,
    )


def _coords(axes: list[Axis]) -> dict:
    return build_axis_mappings(axes)[0]


def test_generated_file_contains_derived_leadtime(tmp_path):
    axes = [_time(), _reftime()]

    _, path = cmorize(
        DatasetInfo(calendar="standard", frequency="6hr"),
        Variable(name="tas", dimensions=("time",), units="K"),
        axes,
        np.ones(3),
        path=tmp_path / "forecast.nc",
    )

    assert path.is_file()
    with Dataset(path) as output:
        leadtime = output.variables["leadtime"]
        np.testing.assert_allclose(leadtime[:], [0.0, 6.0, 12.0])
        assert leadtime.dimensions == ("time",)
        assert leadtime.units == "hours"
        assert leadtime.standard_name == "forecast_period"
        assert leadtime.long_name == "Time elapsed since the start of the forecast"
        assert leadtime.axis == "T"
        assert output.variables["tas"].coordinates == "reftime leadtime"


def test_generated_file_supports_size_one_nonscalar_reftime(tmp_path):
    axes = [
        _time(values=(30.0, 60.0), units="days since 2000-01-01"),
        _reftime(
            values=(10.0,),
            units="days since 2000-01-01",
            scalar=False,
        ),
    ]

    _, path = cmorize(
        DatasetInfo(calendar="standard", frequency="mon"),
        Variable(name="tas", dimensions=("time", "reftime1"), units="K"),
        axes,
        np.ones((2, 1)),
        path=tmp_path / "nonscalar-reftime.nc",
    )

    assert path.is_file()
    with Dataset(path) as output:
        assert output.variables["reftime"].dimensions == ("reftime",)
        np.testing.assert_allclose(output.variables["reftime"][:], [10.0])
        np.testing.assert_allclose(output.variables["leadtime"][:], [20.0, 50.0])


def test_dataset_writer_derives_leadtime_across_chunked_writes(tmp_path):
    writer = DatasetWriter(
        DatasetInfo(calendar="360_day", frequency="mon"),
        Variable(name="tas", dimensions=("time",), units="K"),
        [
            _time(values=(), units="days since 2000-01-01"),
            _reftime(values=(10.0,), units="days since 2000-01-01"),
        ],
        path=tmp_path / "chunked-forecast.nc",
    )
    writer.write(
        np.asarray([1.0, 2.0]),
        time_values=[30.0, 60.0],
        time_bounds=[[15.0, 45.0], [45.0, 75.0]],
    )
    writer.write(
        np.asarray([3.0, 4.0]),
        time_values=[90.0, 120.0],
        time_bounds=[[75.0, 105.0], [105.0, 135.0]],
    )

    result, path = writer.close()
    result.close()

    assert path.is_file()
    with Dataset(path) as output:
        np.testing.assert_allclose(
            output.variables["time"][:],
            [30.0, 60.0, 90.0, 120.0],
        )
        np.testing.assert_allclose(output.variables["reftime"][:], 10.0)
        np.testing.assert_allclose(
            output.variables["leadtime"][:],
            [20.0, 50.0, 80.0, 110.0],
        )
        assert output.variables["leadtime"].dimensions == ("time",)
        assert output.variables["leadtime"].standard_name == "forecast_period"
        assert output.variables["tas"].coordinates == "reftime leadtime"


def test_no_reftime_does_not_add_leadtime():
    axes = [_time()]
    coords = _coords(axes)

    assert derive_forecast_coords(axes, coords) is None
    assert "leadtime" not in coords


def test_generated_file_retains_valid_explicit_leadtime(tmp_path):
    leadtime = Axis(
        name="leadtime",
        values=[0.0, 0.25, 0.5],
        dimensions=("time",),
        units="days",
        standard_name="forecast_period",
        auxiliary=True,
    )
    axes = [_time(), _reftime(), leadtime]

    _, path = cmorize(
        DatasetInfo(calendar="standard", frequency="6hr"),
        Variable(name="tas", dimensions=("time",), units="K"),
        axes,
        np.ones(3),
        path=tmp_path / "explicit-leadtime.nc",
    )

    assert path.is_file()
    with Dataset(path) as output:
        leadtime = output.variables["leadtime"]
        np.testing.assert_allclose(leadtime[:], [0.0, 0.25, 0.5])
        assert leadtime.units == "days"
        assert output.variables["tas"].coordinates == "reftime leadtime"


def test_explicit_leadtime_mismatch_raises():
    leadtime = Axis(
        name="leadtime",
        values=[0.0, 7.0, 12.0],
        dimensions=("time",),
        units="hours",
        standard_name="forecast_period",
        auxiliary=True,
    )
    axes = [_time(), _reftime(), leadtime]

    with pytest.raises(ValueError, match="do not match time - reftime"):
        derive_forecast_coords(axes, _coords(axes))


def test_generated_file_uses_the_requested_calendar(tmp_path):
    axes = [
        _time(values=(0.0, 30.0), units="days since 2001-01-01"),
        _reftime(values=(0.0,), units="days since 2000-12-01"),
    ]
    _, path = cmorize(
        DatasetInfo(calendar="360_day", frequency="mon"),
        Variable(name="tas", dimensions=("time",), units="K"),
        axes,
        np.ones(2),
        path=tmp_path / "360-day-forecast.nc",
    )

    assert path.is_file()
    with Dataset(path) as output:
        np.testing.assert_allclose(output.variables["leadtime"][:], [30.0, 60.0])


def test_leadtime_metadata_comes_from_coordinate_table():
    table = CoordinateTable(
        {
            "leadtime": {
                "out_name": "forecast_lead",
                "standard_name": "forecast_period",
                "long_name": "Forecast elapsed time",
                "axis": "T",
            }
        },
        {},
        {},
    )
    axes = [_time(), _reftime()]
    coords = _coords(axes)

    assert derive_forecast_coords(axes, coords, table) == "forecast_lead"
    assert coords["forecast_lead"][2]["long_name"] == "Forecast elapsed time"
