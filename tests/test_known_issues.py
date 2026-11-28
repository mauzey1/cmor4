"""Known bugs, recorded as strict xfails.

Each test states the *correct* behavior and is marked
``xfail(strict=True, raises=...)`` with the exception it raises today. When the
bug is fixed, the test starts to pass, and strict mode turns that XPASS into a
failure. The fixing PR must then remove the marker. ``raises=`` ensures that an
unrelated breakage is reported as a failure, not as the expected one.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
from netCDF4 import Dataset
import pytest
import xarray as xr

import cmor
from cmor import _assembly
from dataset_helpers import dataset_info, horizontal_axes, time_axis
from table_helpers import cmip7_project


TAS = "tas_tavg-h2m-hxy-u"


@pytest.fixture(scope="module")
def project() -> cmor.ProjectTables:
    return cmip7_project()


def _tas(project: cmor.ProjectTables, **kwargs) -> cmor.Variable:
    return project.variable(TAS, table_id="atmos", **kwargs)


def _one_month(project: cmor.ProjectTables) -> list[cmor.Axis]:
    return [
        time_axis(project, values=[15.0], bounds=[[0.0, 30.0]]),
        *horizontal_axes(project),
    ]


def _data() -> np.ndarray:
    return np.ones((1, 2, 2), dtype="f4")


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="append fails for variables with a scalar coordinate",
)
def test_append_for_variable_with_scalar_coordinate(project, tmp_path):
    info = project.dataset_info(dataset_info(tmp_path))
    variable = _tas(project)
    output_path = tmp_path / "append.nc"

    with cmor.DatasetWriter(
        info,
        variable,
        [time_axis(project), *horizontal_axes(project)],
        path=output_path,
    ) as writer:
        writer.write(_data(), time_values=[15.0], time_bounds=[[0.0, 30.0]])

    with cmor.DatasetWriter(
        info,
        variable,
        [time_axis(project), *horizontal_axes(project)],
        path=output_path,
        existing="append",
    ) as writer:
        writer.write(_data(), time_values=[45.0], time_bounds=[[30.0, 60.0]])

    with xr.open_dataset(output_path) as output:
        assert output.sizes["time"] == 2
        assert "height" in output.coords


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="missing_value is written as float64 on a float32 variable",
)
def test_missing_value_has_variable_type(project, tmp_path):
    info = project.dataset_info(dataset_info(tmp_path))
    variable = _tas(project, missing_value=np.float32(1.0e20))

    path = cmor.cmorize(
        info, variable, _one_month(project), _data(), path=tmp_path / "tas.nc"
    )

    with Dataset(path) as output:
        tas = output.variables["tas"]
        assert tas.dtype == np.float32
        assert np.asarray(tas.getncattr("_FillValue")).dtype == np.float32
        assert np.asarray(tas.getncattr("missing_value")).dtype == np.float32


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="Variable.chunksizes bypasses the CMIP7 chunk-size check",
)
def test_variable_chunksizes_are_validated(project, tmp_path):
    info = project.dataset_info(dataset_info(tmp_path))
    variable = _tas(project, chunksizes=(1, 1, 1))

    with pytest.raises(ValueError, match="4 MiB"):
        cmor.cmorize(
            info, variable, _one_month(project), _data(), path=tmp_path / "tas.nc"
        )


def _hybrid_axis(project: cmor.ProjectTables, name: str) -> cmor.Axis:
    if name.startswith("ocean"):
        return project.axis(name, values=[-0.5], bounds=[[0.0, -1.0]])
    return project.axis(name, values=[0.5], bounds=[[1.0, 0.0]])


def _bounds_formula_terms(axis: cmor.Axis) -> str | None:
    """Run ``set_formula_terms`` on a minimal dataset for *axis*.

    The dataset holds the coordinate, its bounds, and every ``*_bnds``
    variable named in the table's ``z_bounds_factors``, so the expected terms
    are all available.
    """

    out_name = str(axis.out_name or axis.name)
    bounds_name = f"{out_name}_bnds"
    ds = xr.Dataset(coords={out_name: ([out_name], [0.5], {"bounds": bounds_name})})
    ds[bounds_name] = ((out_name, "bnds"), [[1.0, 0.0]])
    for term in re.findall(r"\w+:\s*(\w+)", str(axis.z_bounds_factors)):
        if term.endswith("_bnds") and term not in ds:
            ds[term] = ((out_name, "bnds"), [[1.0, 0.0]])
    variable = cmor.Variable(name="x", dimensions=(axis.name,))
    _assembly.set_formula_terms(ds, [axis], variable, [])
    return ds[bounds_name].attrs.get("formula_terms")


def test_bounds_formula_terms_standard_hybrid_sigma(project):
    # Already correct: the a/b text replacement happens to give the table value.
    axis = _hybrid_axis(project, "standard_hybrid_sigma")
    assert axis.z_bounds_factors
    assert _bounds_formula_terms(axis) == axis.z_bounds_factors


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bounds formula_terms ignore the table's z_bounds_factors",
)
@pytest.mark.parametrize(
    "axis_name",
    [
        "alternate_hybrid_sigma",
        "hybrid_height",
        "hybrid_sigma_ln_pressure",
        "ocean_sigma",
        "ocean_sigma_z",
        "standard_sigma",
    ],
)
def test_bounds_formula_terms_follow_table(project, axis_name):
    axis = _hybrid_axis(project, axis_name)
    assert axis.z_bounds_factors
    assert _bounds_formula_terms(axis) == axis.z_bounds_factors


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="a DatasetWriter block with no writes raises and leaks staging",
)
def test_writer_block_without_writes_leaves_no_staging(project, tmp_path):
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    info = project.dataset_info(dataset_info(tmp_path))

    try:
        with cmor.DatasetWriter(
            info,
            _tas(project),
            [time_axis(project), *horizontal_axes(project)],
            path=tmp_path / "unused.nc",
            staging_dir=staging_dir,
        ):
            pass
    finally:
        leftovers = list(staging_dir.iterdir())

    assert leftovers == []
    assert not (tmp_path / "unused.nc").exists()


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="assigning a str to DatasetWriter.path breaks close()",
)
def test_writer_path_accepts_str(project, tmp_path):
    info = project.dataset_info(dataset_info(tmp_path))
    writer = cmor.DatasetWriter(
        info,
        _tas(project),
        [time_axis(project), *horizontal_axes(project)],
        path=tmp_path / "s1.nc",
    )
    writer.write(_data(), time_values=[15.0], time_bounds=[[0.0, 30.0]])
    writer.close(preserve_definition=True)

    writer.path = str(tmp_path / "s2.nc")
    writer.write(_data(), time_values=[45.0], time_bounds=[[30.0, 60.0]])
    path = writer.close()

    assert Path(path) == tmp_path / "s2.nc"
    assert Path(path).is_file()


@pytest.mark.xfail(
    strict=True,
    raises=UserWarning,
    reason="re-passing a prepared DatasetInfo warns about tracking_id",
)
def test_dataset_info_round_trip_does_not_warn(project, tmp_path):
    prepared = project.dataset_info(dataset_info(tmp_path))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        project.dataset_info(prepared)
        project.dataset_info(prepared.to_dict())
