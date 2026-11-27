"""Golden snapshots of complete ``cmorize()`` output.

These are characterization tests: they record what CMOR writes today for a
representative set of cases, so a refactor that is meant to keep behavior the
same can show that it did. A PR that changes output on purpose must regenerate
the snapshots and explain the diff:

    CMOR_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_datasets.py

Each snapshot records, after normalization, the dimensions, which names are
coordinates and which are data variables, and for each variable its dims, dtype,
attributes, key encoding entries and values. Global attributes that change on
every run (``tracking_id``, ``creation_date``, ``history``) are recorded only as
present. The temporary output directory is replaced with ``<tmp>``.
"""

from __future__ import annotations

from collections.abc import Callable
import difflib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

import cmor
from cmor import Axis, DatasetInfo, Variable
from dataset_helpers import curvilinear_grid, dataset_info, horizontal_axes, time_axis
from table_helpers import cmip7_project


GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("CMOR_UPDATE_GOLDEN") == "1"
VOLATILE_GLOBALS = frozenset({"tracking_id", "creation_date", "history"})
ENCODING_KEYS = ("dtype", "_FillValue", "chunksizes", "zlib", "complevel", "shuffle")


# ---------------------------------------------------------------------------
# Normalization


def _plain(value: Any, tmp: str) -> Any:
    """Convert *value* to JSON-friendly data with a stable representation."""

    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value.replace(tmp, "<tmp>")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "O", "U"}:
            return [_plain(item, tmp) for item in value.tolist()]
        return _plain(value.tolist(), tmp)
    if isinstance(value, np.generic):
        return _plain(value.item(), tmp)
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        return float(f"{value:.10g}")
    if isinstance(value, (list, tuple)):
        return [_plain(item, tmp) for item in value]
    return value


def _variable_snapshot(var: xr.Variable, tmp: str) -> dict[str, Any]:
    values = var.values
    encoding = {
        key: _plain(var.encoding[key], tmp)
        for key in ENCODING_KEYS
        if key in var.encoding and var.encoding[key] is not None
    }
    if "dtype" in encoding:
        encoding["dtype"] = str(np.dtype(var.encoding["dtype"]))
    attrs = {key: _plain(val, tmp) for key, val in sorted(var.attrs.items())}
    for key in ("_FillValue", "missing_value"):
        if key in var.attrs:
            attrs[f"{key}:dtype"] = str(np.asarray(var.attrs[key]).dtype)
    return {
        "dims": list(var.dims),
        "dtype": str(var.dtype),
        "attrs": attrs,
        "encoding": encoding,
        "values": _plain(values, tmp),
    }


def snapshot(ds: xr.Dataset, tmp: str) -> dict[str, Any]:
    globals_: dict[str, Any] = {}
    for key, val in sorted(ds.attrs.items()):
        globals_[key] = "<volatile>" if key in VOLATILE_GLOBALS else _plain(val, tmp)
    return {
        "dims": {str(name): int(size) for name, size in sorted(ds.sizes.items())},
        "coords": sorted(str(name) for name in ds.coords),
        "data_vars": sorted(str(name) for name in ds.data_vars),
        "variables": {
            str(name): _variable_snapshot(ds.variables[name], tmp)
            for name in sorted(ds.variables, key=str)
        },
        "global_attributes": globals_,
    }


def _open_raw(path: Path) -> xr.Dataset:
    """Open without CF decoding, so the snapshot shows what is in the file."""

    with xr.open_dataset(
        path, decode_cf=False, mask_and_scale=False, decode_times=False
    ) as opened:
        return opened.load()


# ---------------------------------------------------------------------------
# Cases. Each returns the path written by cmor.cmorize().


@pytest.fixture(scope="module")
def project() -> cmor.ProjectTables:
    return cmip7_project()


def _months(n: int, project: cmor.ProjectTables) -> Axis:
    values = [15.0 + 30.0 * i for i in range(n)]
    bounds = [[30.0 * i, 30.0 * (i + 1)] for i in range(n)]
    return time_axis(project, values=values, bounds=bounds)


def case_cmip7_atmos_tas(project, tmp_path: Path) -> Path:
    info = project.dataset_info(dataset_info(tmp_path))
    variable = project.variable("tas_tavg-h2m-hxy-u", table_id="atmos")
    data = np.arange(8, dtype="f4").reshape(2, 2, 2) + 270.0
    return cmor.cmorize(
        info, variable, [_months(2, project), *horizontal_axes(project)], data
    )


def case_cmip7_ocean_tos(project, tmp_path: Path) -> Path:
    info = project.dataset_info(dataset_info(tmp_path))
    variable = project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
    data = np.arange(8, dtype="f4").reshape(2, 2, 2) + 10.0
    return cmor.cmorize(
        info, variable, [_months(2, project), *horizontal_axes(project)], data
    )


def case_cmip7_hybrid_sigma(project, tmp_path: Path) -> Path:
    info = project.dataset_info(dataset_info(tmp_path))
    variable = project.variable("tnhusscpbl_tavg-al-hxy-u", table_id="atmos")
    axes = [
        _months(2, project),
        project.axis(
            "standard_hybrid_sigma",
            values=[0.9, 0.1],
            bounds=[[1.0, 0.5], [0.5, 0.0]],
        ),
        *horizontal_axes(project),
    ]
    zfactors = [
        project.zfactor("a", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
        project.zfactor("b", values=[0.9, 0.1], bounds=[[1.0, 0.5], [0.5, 0.0]]),
        project.zfactor("p0", values=100000.0),
        project.zfactor("ps", values=np.full((2, 2, 2), 99000.0, dtype="f4")),
    ]
    data = np.arange(16, dtype="f4").reshape(2, 2, 2, 2) * 1.0e-6
    return cmor.cmorize(info, variable, axes, data, zfactors=zfactors)


def case_cmip7_grid_mapping(project, tmp_path: Path) -> Path:
    info = project.dataset_info(dataset_info(tmp_path))
    variable = project.variable("hfls_tavg-u-hxy-u", table_id="atmos")
    data = np.arange(24, dtype="f4").reshape(2, 3, 4)
    return cmor.cmorize(
        info,
        variable,
        [_months(2, project)],
        data,
        grid=curvilinear_grid(project),
    )


def _forecast_axes() -> list[Axis]:
    return [
        Axis(
            name="time",
            values=(0.0, 6.0, 12.0),
            units="hours since 2020-01-01",
            standard_name="time",
            axis="T",
            calendar="standard",
            frequency="6hr",
        ),
        Axis(
            name="reftime1",
            out_name="reftime",
            values=(0.0,),
            units="hours since 2020-01-01",
            standard_name="forecast_reference_time",
            scalar=True,
        ),
    ]


def case_forecast_derived_leadtime(project, tmp_path: Path) -> Path:
    return cmor.cmorize(
        DatasetInfo(),
        Variable(name="tas", dimensions=("time",), units="K"),
        _forecast_axes(),
        np.array([280.0, 281.0, 282.0]),
        path=tmp_path / "forecast.nc",
    )


def case_forecast_explicit_leadtime(project, tmp_path: Path) -> Path:
    leadtime = Axis(
        name="leadtime",
        values=[0.0, 0.25, 0.5],
        dimensions=("time",),
        units="days",
        standard_name="forecast_period",
        auxiliary=True,
    )
    return cmor.cmorize(
        DatasetInfo(),
        Variable(name="tas", dimensions=("time",), units="K"),
        [*_forecast_axes(), leadtime],
        np.array([280.0, 281.0, 282.0]),
        path=tmp_path / "forecast-explicit.nc",
    )


def case_climatology(project, tmp_path: Path) -> Path:
    raw_info = dataset_info(tmp_path)
    raw_info.update({
        "output_file_template": "<variable_id>_<frequency>",
        "output_path_template": "<activity_id>",
    })
    axes = [
        Axis(
            name="time2",
            calendar="360_day",
            frequency="mon",
            values=[15.0, 45.0],
            bounds=[[0.0, 31.0], [31.0, 60.0]],
            units="days since 2018",
            standard_name="time",
            axis="T",
            climatology="yes",
            out_name="time",
        )
    ]
    variable = Variable(name="co2_tclm-u-hm-u", dimensions=["time2"])
    info = DatasetInfo.from_prepared(raw_info)
    return cmor.cmorize(info, variable, axes, np.array([400.0, 401.0], dtype="f4"))


CASES: dict[str, Callable[[cmor.ProjectTables, Path], Path]] = {
    "cmip7_atmos_tas": case_cmip7_atmos_tas,
    "cmip7_ocean_tos": case_cmip7_ocean_tos,
    "cmip7_hybrid_sigma": case_cmip7_hybrid_sigma,
    "cmip7_grid_mapping": case_cmip7_grid_mapping,
    "forecast_derived_leadtime": case_forecast_derived_leadtime,
    "forecast_explicit_leadtime": case_forecast_explicit_leadtime,
    "climatology": case_climatology,
}


# ---------------------------------------------------------------------------
# Test


@pytest.mark.parametrize("case", sorted(CASES))
def test_golden_dataset(case, project, tmp_path):
    path = Path(CASES[case](project, tmp_path))
    actual = snapshot(_open_raw(path), str(tmp_path))
    actual["file_name"] = path.name
    actual_text = json.dumps(actual, indent=1, sort_keys=True) + "\n"

    golden_path = GOLDEN_DIR / f"{case}.json"
    if UPDATE:
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_text(actual_text)
        return

    if not golden_path.exists():
        pytest.fail(
            f"Missing snapshot {golden_path.name}. Run with CMOR_UPDATE_GOLDEN=1."
        )
    expected_text = golden_path.read_text()
    if actual_text != expected_text:
        diff = "".join(
            difflib.unified_diff(
                expected_text.splitlines(keepends=True),
                actual_text.splitlines(keepends=True),
                fromfile=f"golden/{golden_path.name}",
                tofile="actual",
            )
        )
        pytest.fail(
            f"Output for {case!r} differs from its snapshot. If the change is "
            "intended, regenerate with CMOR_UPDATE_GOLDEN=1 and explain the "
            f"diff in the PR.\n{diff}"
        )
