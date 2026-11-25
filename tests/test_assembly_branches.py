"""Characterization tests for branches of ``cmor._assembly`` (W0.2).

They pin today's behavior, including error text, so the ``_assembly`` split
(W6) can show that it changed nothing. Public entry points are used where they
reach the branch. The internal-invariant guards in ``validate_final_dataset``
can't be reached through them, because ``build_dataset`` always creates what
they check. For those, the tests build a valid dataset, remove one piece, and
call the guard directly. All uses of private ``_assembly`` names are in this
file, so the W6 renames have one place to update.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cmor
from cmor import Axis, DatasetInfo, Variable, ZFactor, _assembly
from cmor.exceptions import TableValidationError
from table_helpers import cmip7_project
from test_append_mode import curvilinear_grid, dataset_info, horizontal_axes, time_axis


@pytest.fixture(scope="module")
def project() -> cmor.ProjectTables:
    return cmip7_project()


# ---------------------------------------------------------------------------
# Forecast leadtime derivation


def _time(values=(0.0, 6.0), units="hours since 2020-01-01", **kwargs) -> Axis:
    return Axis(
        name="time",
        values=values,
        units=units,
        standard_name="time",
        axis="T",
        calendar="standard",
        frequency="6hr",
        **kwargs,
    )


def _reftime(values=(0.0,), units="hours since 2020-01-01", scalar=True) -> Axis:
    return Axis(
        name="reftime1",
        out_name="reftime",
        values=values,
        units=units,
        standard_name="forecast_reference_time",
        scalar=scalar,
    )


def _leadtime(values=(0.0, 6.0), units="hours") -> Axis:
    return Axis(
        name="leadtime",
        values=values,
        dimensions=("time",),
        units=units,
        standard_name="forecast_period",
        auxiliary=True,
    )


def _derive(axes: list[Axis]) -> str | None:
    coords = _assembly.build_axis_mappings(axes)[0]
    return _assembly.derive_forecast_coords(axes, coords)


def test_multi_value_reference_time_does_not_derive_leadtime():
    assert _derive([_time(), _reftime(values=(0.0, 6.0), scalar=False)]) is None


def test_leadtime_requires_relative_time_units():
    with pytest.raises(ValueError) as error:
        _derive([_time(units=""), _reftime()])
    assert str(error.value) == (
        "Cannot derive leadtime because the time axis has no relative-time units."
    )


def test_leadtime_requires_decodable_reference_time():
    with pytest.raises(ValueError) as error:
        _derive([_time(), _reftime(units="hours")])
    assert str(error.value) == (
        "Cannot derive leadtime from the forecast reference-time coordinate."
    )


def test_leadtime_requires_decodable_time_values():
    # "months" is not a duration that the time decoder accepts.
    with pytest.raises(ValueError) as error:
        _derive([_time(units="months since 2020-01-01"), _reftime()])
    assert str(error.value) == "Cannot derive leadtime from the time coordinate."


def test_explicit_leadtime_with_incomparable_units_is_rejected():
    with pytest.raises(ValueError) as error:
        _derive([_time(), _reftime(), _leadtime(units="kg")])
    assert str(error.value) == (
        "Cannot compare leadtime units 'kg' with time units 'hours'."
    )


# ---------------------------------------------------------------------------
# Data and scalar-axis validation


def _simple_ctx() -> _assembly.ValidationContext:
    return _assembly.validate_metadata(
        DatasetInfo(),
        Variable(name="tas", dimensions=("time",), units="K"),
        [_time()],
    )


def test_data_chunk_with_wrong_number_of_dimensions_is_rejected():
    with pytest.raises(ValueError) as error:
        _assembly.validate_data_chunk(_simple_ctx(), np.ones((2, 1)))
    assert str(error.value) == (
        "Data for 'tas' has 2 dimensions, but variable dimensions resolve to 'time'."
    )


def test_scalar_axis_with_several_values_is_rejected():
    height = Axis(name="height", values=[2.0, 10.0], units="m", scalar=True)

    with pytest.raises(ValueError, match="exactly one value"):
        _assembly.build_axis_dimension_map([_time(), height])
    with pytest.raises(ValueError, match="exactly one value"):
        _assembly.build_axis_mappings([_time(), height])


@pytest.mark.parametrize("value", [2.0, [2.0]])
def test_scalar_axis_value_becomes_zero_dimensional(value):
    height = Axis(name="height", values=value, units="m", scalar=True)

    assert _assembly.build_axis_dimension_map([height])["height"] == ()
    data_vars = _assembly.build_axis_mappings([height])[1]
    assert data_vars["height"][0] == ()
    assert data_vars["height"][1] == 2.0


# ---------------------------------------------------------------------------
# Final-dataset guards


def _built(project, tmp_path, *, grid: bool = False, hybrid: bool = False):
    info = project.dataset_info(dataset_info(tmp_path))
    months = time_axis(project, values=[15.0], bounds=[[0.0, 30.0]])
    zfactors: list[ZFactor] = []
    if grid:
        variable = project.variable("hfls_tavg-u-hxy-u", table_id="atmos")
        axes = [months]
        data = np.ones((1, 3, 4), dtype="f4")
    elif hybrid:
        variable = project.variable("tnhusscpbl_tavg-al-hxy-u", table_id="atmos")
        axes = [
            months,
            project.axis("standard_hybrid_sigma", values=[0.5], bounds=[[1.0, 0.0]]),
            *horizontal_axes(project),
        ]
        zfactors = [
            project.zfactor("a", values=[0.5], bounds=[[1.0, 0.0]]),
            project.zfactor("b", values=[0.5], bounds=[[1.0, 0.0]]),
            project.zfactor("p0", values=100000.0),
            project.zfactor("ps", values=np.full((1, 2, 2), 99000.0, dtype="f4")),
        ]
        data = np.ones((1, 1, 2, 2), dtype="f4")
    else:
        variable = project.variable("tas_tavg-h2m-hxy-u", table_id="atmos")
        axes = [months, *horizontal_axes(project)]
        data = np.ones((1, 2, 2), dtype="f4")
    ctx = _assembly.validate_metadata(
        info,
        variable,
        axes,
        zfactors,
        curvilinear_grid(project) if grid else None,
    )
    ds = _assembly.build_dataset(ctx, data)
    names = [str(z.out_name or z.name) for z in ctx.zfactors]
    return ds, ctx, names


def _check(ds, ctx, names) -> str:
    with pytest.raises((ValueError, TableValidationError)) as error:
        _assembly.validate_final_dataset(ds, ctx, names)
    return str(error.value)


def test_final_dataset_guards_accept_built_dataset(project, tmp_path):
    for kwargs in ({}, {"grid": True}, {"hybrid": True}):
        ds, ctx, names = _built(project, tmp_path, **kwargs)
        _assembly.validate_final_dataset(ds, ctx, names)


def test_final_dataset_guard_frequency(project, tmp_path):
    ds, ctx, names = _built(project, tmp_path)
    ds.attrs["frequency"] = "day"
    # The project validation runs first and reports the mismatch in its own way.
    assert "frequency" in _check(ds, ctx, names)


def test_final_dataset_guard_missing_variable(project, tmp_path):
    ds, ctx, names = _built(project, tmp_path)
    assert _check(ds.drop_vars("tas"), ctx, names) == "Variable 'tas' was not created."


def test_final_dataset_guard_dimension_order(project, tmp_path):
    ds, ctx, names = _built(project, tmp_path)
    ds["tas"] = ds["tas"].transpose("time", "lon", "lat")
    assert _check(ds, ctx, names) == (
        "Variable 'tas' dimensions ('time', 'lon', 'lat') do not match "
        "expected dimensions ('time', 'lat', 'lon')."
    )


def test_final_dataset_guard_missing_axis_and_bounds(project, tmp_path):
    ds, ctx, names = _built(project, tmp_path)
    assert _check(ds.drop_vars("lat_bnds"), ctx, names) == (
        "Bounds variable 'lat_bnds' for axis 'latitude' was not created."
    )
    assert _check(ds.drop_vars("height"), ctx, names) == (
        "Axis 'height2m' was not created."
    )


def test_final_dataset_guard_grid_mapping(project, tmp_path):
    ds, ctx, names = _built(project, tmp_path, grid=True)
    mapping = ctx.grid.variable_name
    assert _check(ds.drop_vars(mapping), ctx, names) == (
        f"Grid mapping variable {mapping!r} was not created."
    )
    unlinked = ds.copy()
    unlinked["hfls"].attrs["grid_mapping"] = "other"
    assert _check(unlinked, ctx, names) == (
        f"Variable 'hfls' does not reference grid mapping {mapping!r}."
    )


def test_final_dataset_guard_zfactors(project, tmp_path):
    ds, ctx, names = _built(project, tmp_path, hybrid=True)
    assert _check(ds.drop_vars("ps"), ctx, names) == "Z-factor 'ps' was not created."
    assert _check(ds.drop_vars("a_bnds"), ctx, names) == (
        "Bounds variable 'a_bnds' for z-factor 'a' was not created."
    )


# ---------------------------------------------------------------------------
# set_formula_terms


def _lev_dataset(bounds: bool = True) -> xr.Dataset:
    attrs = {"bounds": "lev_bnds"} if bounds else {}
    ds = xr.Dataset(coords={"lev": ("lev", [0.5], attrs)})
    if bounds:
        ds["lev_bnds"] = (("lev", "bnds"), [[1.0, 0.0]])
    return ds


def _lev_axis(**kwargs) -> Axis:
    return Axis(name="lev", values=[0.5], units="1", axis="Z", **kwargs)


def test_formula_terms_skip_axes_without_terms_or_not_on_variable():
    ds = _lev_dataset()
    variable = Variable(name="x", dimensions=("lev",))
    _assembly.set_formula_terms(ds, [_lev_axis()], variable, [])
    assert "formula_terms" not in ds["lev"].attrs

    other = Variable(name="x", dimensions=("plev",))
    _assembly.set_formula_terms(ds, [_lev_axis(z_factors="a: a b: b")], other, [])
    assert "formula_terms" not in ds["lev"].attrs


def test_formula_terms_without_bounds_only_set_coordinate():
    ds = _lev_dataset(bounds=False)
    variable = Variable(name="x", dimensions=("lev",))
    _assembly.set_formula_terms(ds, [_lev_axis(z_factors="a: a b: b")], variable, [])
    assert ds["lev"].attrs["formula_terms"] == "a: a b: b"


def test_formula_terms_fall_back_to_variable_terms():
    ds = _lev_dataset()
    variable = Variable(name="x", dimensions=("lev",), formula_terms="a: a b: b")
    _assembly.set_formula_terms(ds, [_lev_axis()], variable, [])
    assert ds["lev"].attrs["formula_terms"] == "a: a b: b"
    # Without a_bnds and b_bnds in the dataset, the terms are copied unchanged,
    # and the bounds get no formula/standard_name/units the coordinate lacks.
    assert ds["lev_bnds"].attrs == {"formula_terms": "a: a b: b"}
