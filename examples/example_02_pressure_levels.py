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
    plev = np.array(
        [
            100000.0,
            92500.0,
            85000.0,
            70000.0,
            60000.0,
            50000.0,
            40000.0,
            30000.0,
            25000.0,
            20000.0,
            15000.0,
            10000.0,
            7000.0,
            5000.0,
            3000.0,
            2000.0,
            1000.0,
            500.0,
            100.0,
        ],
        dtype="d",
    )
    axes = [
        project.axis(
            "time",
            values=np.array([15.0, 45.0], dtype="d"),
            bounds=np.array([[0.0, 30.0], [30.0, 60.0]], dtype="d"),
            units="days since 1979-01-01",
        ),
        project.axis("plev19", values=plev),
        project.axis(
            "latitude",
            values=np.array([10.0, 20.0, 30.0], dtype="d"),
            bounds=np.array([[5.0, 15.0], [15.0, 25.0], [25.0, 35.0]], dtype="d"),
        ),
        project.axis(
            "longitude",
            values=np.array([0.0, 90.0, 180.0, 270.0], dtype="d"),
            bounds=np.array(
                [[-45.0, 45.0], [45.0, 135.0], [135.0, 225.0], [225.0, 315.0]],
                dtype="d",
            ),
        ),
    ]
    variable = project.variable(
        "ta_tavg-p19-hxy-air", table_id="atmos", missing_value=np.float64(1.0e20)
    )
    data = np.linspace(250.0, 275.0, 2 * 19 * 3 * 4, dtype="f4").reshape(
        2, 19, 3, 4
    )
    data[0, 0, 0, 0] = np.float32(1.0e20)
    return str(cmor4.cmorize(dataset, variable, axes, data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CMIP7 example 2 with CMOR4.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()
    print(write_example(args.output_dir))


if __name__ == "__main__":
    main()
