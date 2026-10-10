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
        variable_tables=["tables/CMIP7_land.json"],
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
        "frequency": "fx",
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
    axes = [
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
        "rootd_ti-u-hxy-lnd", table_id="land", missing_value=np.float64(1.0e20)
    )
    data = np.array(
        [[0.50, 0.45, 1.0e20, 0.55], [0.60, 0.60, 1.0e20, 0.55], [1.0e20, 0.45, 0.50, 0.50]],
        dtype="f4",
    )
    return str(cmor4.cmorize(dataset, variable, axes, data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CMIP7 example 7 with CMOR4.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()
    print(write_example(args.output_dir))


if __name__ == "__main__":
    main()
