"""Shared CMIP7 setup helpers for the test suite.

These build the dataset metadata, axes and grid that many tests use, so that
test modules don't import helpers from one another.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import cmor


def dataset_info(tmp_path: Path) -> dict[str, str]:
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
) -> cmor.Axis:
    kwargs = {
        "units": "days since 2000-01-01",
        "calendar": "standard",
        "frequency": "mon",
    }
    if values is not None:
        kwargs["values"] = values
    if bounds is not None:
        kwargs["bounds"] = bounds
    return project.axis("time", **kwargs)


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
