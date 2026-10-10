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
        "forcing_index": "f2",
        "frequency": "mon",
        "grid_label": "g010",
        "initialization_index": "i1",
        "institution_id": "MOHC",
        "license_id": "CC-BY-4.0",
        "nominal_resolution": "100 km",
        "outpath": str(output_dir),
        "physics_index": "p1",
        "realization_index": "r9",
        "region": "glb",
        "source_id": "ACCESS-ESM1-6",
    })
    axes = [
        project.axis(
            "time",
            values=np.array([15.0, 45.0], dtype="d"),
            bounds=np.array([[0.0, 30.0], [30.0, 60.0]], dtype="d"),
            units="days since 1979-01-01",
        ),
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
        "tas_tavg-h2m-hxy-u", table_id="atmos", missing_value=np.float64(1.0e20)
    )
    data = np.array(
        [
            254.0895,
            258.4085,
            250.5549,
            258.7101,
            258.6680,
            258.2990,
            252.1237,
            255.0432,
            253.7254,
            251.2460,
            254.3168,
            255.4808,
            259.7908,
            252.2754,
            257.1892,
            253.3132,
            253.8823,
            253.4698,
            253.5381,
            254.9730,
            256.1002,
            251.8168,
            259.3698,
            250.2994,
        ],
        dtype="f4",
    ).reshape(2, 3, 4)
    return str(cmor4.cmorize(dataset, variable, axes, data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CMIP7 example 3 with CMOR4.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()
    print(write_example(args.output_dir))


if __name__ == "__main__":
    main()
