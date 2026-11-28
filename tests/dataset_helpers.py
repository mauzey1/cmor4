"""Shared helpers for building and reading CMOR datasets in tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
import unittest

import numpy as np
import xarray as xr

import cmor


def load_dataset(path: str | Path) -> xr.Dataset:
    with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as opened:
        loaded = opened.load()
    for variable in loaded.variables.values():
        coordinates = variable.encoding.get("coordinates")
        if coordinates is not None and "coordinates" not in variable.attrs:
            variable.attrs["coordinates"] = coordinates
    return loaded


def open_created_dataset(*args: Any, **kwargs: Any) -> xr.Dataset:
    if "path" in kwargs and kwargs["path"] is not None:
        return load_dataset(cmor.cmorize(*args, **kwargs))
    with tempfile.TemporaryDirectory() as tmp_dir:
        kwargs["path"] = Path(tmp_dir) / "created.nc"
        return load_dataset(cmor.cmorize(*args, **kwargs))


def cmorize_and_open(*args: Any, **kwargs: Any) -> tuple[xr.Dataset, Path]:
    path = cmor.cmorize(*args, **kwargs)
    return load_dataset(path), path


# ---------------------------------------------------------------------------
# Standard CMIP7 metadata, axes, grid and z-factors


def dataset_info(tmp_path: Path) -> dict[str, str]:
    """Standard CMIP7 dataset metadata for tests."""
    return {
        "activity_id": "CMIP",
        "experiment_id": "amip",
        "forcing_index": "f3",
        "grid_label": "g101",
        "initialization_index": "i1",
        "institution_id": "CNRM-CERFACS",
        "license_id": "CC-BY-4.0",
        "mip_era": "CMIP7",
        "nominal_resolution": "100 km",
        "outpath": str(tmp_path),
        "physics_index": "p1",
        "realization_index": "r9",
        "region": "glb",
        "source_id": "CNRM-ESM2-1e",
        "version": "v20200101",
    }


def horizontal_axes(project: cmor.ProjectTables) -> list[cmor.Axis]:
    """Standard horizontal axes for tests."""
    return [
        project.axis(
            "latitude",
            values=[-45.0, 45.0],
            bounds=[[-90.0, 0.0], [0.0, 90.0]],
        ),
        project.axis(
            "longitude",
            values=[90.0, 270.0],
            bounds=[[0.0, 180.0], [180.0, 360.0]],
        ),
    ]


def time_axis(
    project: cmor.ProjectTables,
    values: list[float] | None = None,
    bounds: list[list[float]] | None = None,
    **kwargs,
) -> cmor.Axis:
    """Create a time axis with optional values and bounds."""
    axis_kwargs = {
        "units": "days since 2000-01-01",
        "calendar": "standard",
        "frequency": "mon",
    }
    if values is not None:
        axis_kwargs["values"] = values
    if bounds is not None:
        axis_kwargs["bounds"] = bounds
    axis_kwargs.update(kwargs)
    return project.axis("time", **axis_kwargs)


def curvilinear_grid(
    project: cmor.ProjectTables,
    *,
    mapping_latitude: float = 90.0,
) -> cmor.Grid:
    y_axis = project.axis(
        "y",
        values=[0.0, 10000.0, 20000.0],
        bounds=[[-5000.0, 5000.0], [5000.0, 15000.0], [15000.0, 25000.0]],
        units="m",
    )
    x_axis = project.axis(
        "x",
        values=[0.0, 10000.0, 20000.0, 30000.0],
        bounds=[
            [-5000.0, 5000.0],
            [5000.0, 15000.0],
            [15000.0, 25000.0],
            [25000.0, 35000.0],
        ],
        units="m",
    )
    latitude = np.array(
        [[10.0, 12.0, 14.0, 16.0], [20.0, 22.0, 24.0, 26.0], [30.0, 32.0, 34.0, 36.0]],
        dtype="f8",
    )
    longitude = np.array(
        [
            [100.0, 110.0, 120.0, 130.0],
            [102.0, 112.0, 122.0, 132.0],
            [104.0, 114.0, 124.0, 134.0],
        ],
        dtype="f8",
    )
    latitude_vertices = np.stack(
        (latitude - 0.5, latitude - 0.25, latitude + 0.5, latitude + 0.25),
        axis=-1,
    )
    longitude_vertices = np.stack(
        (longitude - 0.5, longitude + 0.5, longitude + 0.5, longitude - 0.5),
        axis=-1,
    )
    return project.grid(
        axes=[y_axis, x_axis],
        grid_mapping_name="lambert_azimuthal_equal_area",
        params={
            "latitude_of_projection_origin": [mapping_latitude, "degrees_north"],
            "longitude_of_projection_origin": [0.0, "degrees_east"],
        },
        latitude=latitude,
        longitude=longitude,
        latitude_vertices=latitude_vertices,
        longitude_vertices=longitude_vertices,
    )


def hybrid_axis(project: cmor.ProjectTables) -> cmor.Axis:
    """Two-level ``standard_hybrid_sigma`` axis with bounds."""
    return project.axis(
        "standard_hybrid_sigma",
        values=[0.9, 0.1],
        bounds=[[1.0, 0.5], [0.5, 0.0]],
    )


def hybrid_zfactors(
    project: cmor.ProjectTables,
    ps_values: np.ndarray | None = None,
    *,
    a_values: list[float] | None = None,
) -> list[cmor.ZFactor]:
    """Return the a, b, p0 and ps z-factors for ``hybrid_axis``.

    Without *ps_values*, ``ps`` has no values; supply them per write.
    """
    return [
        project.zfactor(
            "a",
            values=a_values or [0.9, 0.1],
            bounds=[[1.0, 0.5], [0.5, 0.0]],
        ),
        project.zfactor(
            "b",
            values=[0.9, 0.1],
            bounds=[[1.0, 0.5], [0.5, 0.0]],
        ),
        project.zfactor("p0", values=100000.0),
        project.zfactor(
            "ps",
            **({} if ps_values is None else {"values": ps_values}),
        ),
    ]


def equivalent(ds: xr.Dataset) -> xr.Dataset:
    """Normalize dataset for comparison by removing time-dependent attributes."""
    normalized = ds.load().copy(deep=True)
    normalized.attrs.pop("creation_date", None)
    normalized.attrs.pop("history", None)
    normalized.attrs.pop("tracking_id", None)
    for variable in normalized.variables.values():
        variable.attrs.pop("_FillValue", None)
    return normalized


def assert_files_equivalent(
    test: unittest.TestCase,
    actual_path: Path,
    expected_path: Path,
) -> None:
    """Assert that two NetCDF files are equivalent (ignoring timestamps)."""
    test.assertTrue(actual_path.exists(), f"Actual file missing: {actual_path}")
    test.assertTrue(expected_path.exists(), f"Expected file missing: {expected_path}")
    with xr.open_dataset(actual_path, decode_times=False) as actual_opened:
        actual = actual_opened.load()
    with xr.open_dataset(expected_path, decode_times=False) as expected_opened:
        expected = expected_opened.load()
    xr.testing.assert_identical(equivalent(actual), equivalent(expected))
