#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cmor4
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = REPO_ROOT / "project_tables" / "cmip7-cmor-tables"


def write_example(output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    project = cmor4.ProjectTables.from_directory(
        TABLE_ROOT,
        cv_file="tables-cvs/cmor-cvs.json",
        variable_tables=["tables/CMIP7_atmos.json"],
        coordinate_table="tables/CMIP7_coordinate.json",
        formula_table="tables/CMIP7_formula_terms.json",
        grid_table="tables/CMIP7_grids.json",
        long_name_override_table="tables/CMIP7_long_name_overrides.json",
        cell_measures_table="tables/CMIP7_cell_measures.json",
    )
    dataset = project.dataset_info({
        "activity_id": "CMIP",
        "calendar": "360_day",
        "experiment_id": "amip",
        "forcing_index": "f1",
        "frequency": "mon",
        "grid_label": "g010",
        "initialization_index": "i1",
        "institution_id": "MOHC",
        "license_id": "CC-BY-4.0",
        "nominal_resolution": "100 km",
        "outpath": str(output_dir),
        "physics_index": "p1",
        "realization_index": "r1",
        "region": "glb",
        "source_id": "ACCESS-ESM1-6",
    })
    y_axis = project.axis(
        "y",
        values=np.array([0.0, 10000.0, 20000.0], dtype="d"),
        bounds=np.array(
            [[-5000.0, 5000.0], [5000.0, 15000.0], [15000.0, 25000.0]],
            dtype="d",
        ),
        units="m",
    )
    x_axis = project.axis(
        "x",
        values=np.array([0.0, 10000.0, 20000.0, 30000.0], dtype="d"),
        bounds=np.array(
            [
                [-5000.0, 5000.0],
                [5000.0, 15000.0],
                [15000.0, 25000.0],
                [25000.0, 35000.0],
            ],
            dtype="d",
        ),
        units="m",
    )
    latitude = np.array(
        [[10.0, 8.0, 6.0, 4.0], [20.0, 18.0, 16.0, 14.0], [30.0, 28.0, 26.0, 24.0]],
        dtype="d",
    )
    longitude = np.array(
        [
            [280.0, 290.0, 300.0, 310.0],
            [282.0, 292.0, 302.0, 312.0],
            [284.0, 294.0, 304.0, 314.0],
        ],
        dtype="d",
    )
    latitude_vertices = np.empty((3, 4, 4), dtype="d")
    longitude_vertices = np.empty((3, 4, 4), dtype="d")
    for j in range(3):
        for i in range(4):
            latitude_vertices[j, i] = [
                latitude[j, i] - 5.0,
                latitude[j, i] - 4.0,
                latitude[j, i] + 5.0,
                latitude[j, i] + 4.0,
            ]
            longitude_vertices[j, i] = [
                longitude[j, i] - 5.0,
                longitude[j, i] + 5.0,
                longitude[j, i] + 5.0,
                longitude[j, i] - 5.0,
            ]
    grid = project.grid(
        axes=[y_axis, x_axis],
        latitude=latitude,
        longitude=longitude,
        latitude_vertices=latitude_vertices,
        longitude_vertices=longitude_vertices,
        mapping_var="lambert_conformal_conic",
        mapping_name="lambert_conformal_conic",
        params={
            "standard_parallel1": [-20.0, ""],
            "longitude_of_central_meridian": [175.0, ""],
            "latitude_of_projection_origin": [13.0, ""],
            "false_easting": [8.0, ""],
            "false_northing": [0.0, ""],
            "standard_parallel2": [20.0, ""],
        },
    )
    variable = project.variable(
        "hfls_tavg-u-hxy-u", table_id="atmos", missing_value=np.float64(1.0e20)
    )
    time_axis = project.axis(
        "time",
        values=np.array([15.0, 45.0], dtype="d"),
        bounds=np.array([[0.0, 30.0], [30.0, 60.0]], dtype="d"),
        units="days since 1979-01-01",
    )
    data = np.array(
        [
            80.0,
            82.0,
            84.0,
            86.0,
            88.0,
            90.0,
            92.0,
            94.0,
            96.0,
            98.0,
            100.0,
            102.0,
            81.0,
            83.0,
            85.0,
            87.0,
            89.0,
            91.0,
            93.0,
            95.0,
            97.0,
            99.0,
            101.0,
            103.0,
        ],
        dtype="f4",
    ).reshape(2, 3, 4)
    return str(cmor4.cmorize(dataset, variable, [time_axis], data, grid=grid))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CMIP7 example 6 with CMOR4.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()
    print(write_example(args.output_dir))


if __name__ == "__main__":
    main()
