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
    time = project.axis(
        "time",
        values=np.array([15.0, 45.0], dtype="d"),
        bounds=np.array([[0.0, 30.0], [30.0, 60.0]], dtype="d"),
        units="days since 1979-01-01",
    )
    lev = project.axis(
        "standard_hybrid_sigma",
        values=np.array([0.92, 0.72, 0.50, 0.30, 0.10], dtype="d"),
        bounds=np.array(
            [[1.00, 0.83], [0.83, 0.61], [0.61, 0.40], [0.40, 0.20], [0.20, 0.00]],
            dtype="d",
        ),
    )
    axes = [
        time,
        lev,
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
    zfactors = [
        project.zfactor(
            "a",
            values=np.array([0.12, 0.22, 0.30, 0.20, 0.10], dtype="d"),
            bounds=np.array(
                [[0.06, 0.18], [0.18, 0.26], [0.26, 0.25], [0.25, 0.15], [0.15, 0.00]],
                dtype="d",
            ),
        ),
        project.zfactor(
            "b",
            values=np.array([0.80, 0.50, 0.20, 0.10, 0.00], dtype="d"),
            bounds=np.array(
                [[0.94, 0.65], [0.65, 0.35], [0.35, 0.15], [0.15, 0.05], [0.05, 0.00]],
                dtype="d",
            ),
        ),
        project.zfactor("p0", values=100000.0),
        project.zfactor(
            "ps",
            values=np.array(
                [
                    97000.0,
                    97400.0,
                    97800.0,
                    98200.0,
                    98600.0,
                    99000.0,
                    99400.0,
                    99800.0,
                    100200.0,
                    100600.0,
                    101000.0,
                    101400.0,
                    97100.0,
                    97500.0,
                    97900.0,
                    98300.0,
                    98700.0,
                    99100.0,
                    99500.0,
                    99900.0,
                    100300.0,
                    100700.0,
                    101100.0,
                    101500.0,
                ],
                dtype="f4",
            ).reshape(2, 3, 4),
        ),
    ]
    variable = project.variable(
        "cl_tavg-al-hxy-u", table_id="atmos", missing_value=np.float64(1.0e20)
    )
    data = np.array(
        [
            72.8, 73.2, 73.6, 74.0, 71.6, 72.0, 72.4, 72.4, 70.4, 70.8, 70.8, 71.2,
            67.6, 69.2, 69.6, 70.0, 66.0, 66.4, 66.8, 67.2, 64.8, 65.2, 65.6, 66.0,
            63.6, 64.0, 64.4, 64.4, 60.8, 61.2, 62.8, 63.2, 59.6, 59.6, 60.0, 60.4,
            58.0, 58.4, 58.8, 59.2, 56.8, 57.2, 57.6, 58.0, 54.0, 54.4, 54.8, 56.4,
            52.8, 53.2, 53.2, 53.6, 51.6, 51.6, 52.0, 52.4, 50.0, 50.4, 50.8, 51.2,
            72.9, 73.3, 73.7, 74.1, 71.7, 72.1, 72.5, 72.5, 70.5, 70.9, 70.9, 71.3,
            67.7, 69.3, 69.7, 70.1, 66.1, 66.5, 66.9, 67.3, 64.9, 65.3, 65.7, 66.1,
            63.7, 64.1, 64.5, 64.5, 60.9, 61.3, 62.9, 63.3, 59.7, 59.7, 60.1, 60.5,
            58.1, 58.5, 58.9, 59.3, 56.9, 57.3, 57.7, 58.1, 54.1, 54.5, 54.9, 56.5,
            52.9, 53.3, 53.3, 53.7, 51.7, 51.7, 52.1, 52.5, 50.1, 50.5, 50.9, 51.3,
        ],
        dtype="f4",
    ).reshape(2, 5, 3, 4)
    return str(cmor4.cmorize(dataset, variable, axes, data, zfactors=zfactors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CMIP7 example 5 with CMOR4.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()
    print(write_example(args.output_dir))


if __name__ == "__main__":
    main()
