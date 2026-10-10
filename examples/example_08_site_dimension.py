#!/usr/bin/env python3
"""Write near-surface air temperature for CMIP7 point-observation sites."""

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
    site_values = np.arange(1, 127, dtype="i")
    site = project.axis("site", units="1", values=site_values)
    time = project.axis(
        "time1",
        values=np.array([0.0, 30.0], dtype="d"),
        units="days since 1979-01-01",
    )
    site_latitudes = np.linspace(-60.0, 60.0, site_values.size, dtype="f")
    site_longitudes = np.linspace(0.5, 359.5, site_values.size, dtype="f")
    grid = project.grid(
        dimensions=["site"],
        latitude=site_latitudes,
        longitude=site_longitudes,
        latitude_vertices=np.column_stack((site_latitudes - 0.25, site_latitudes + 0.25)),
        longitude_vertices=np.column_stack((site_longitudes - 0.25, site_longitudes + 0.25)),
    )
    variable = project.variable(
        "tas_tpt-h2m-hs-u", table_id="atmos", missing_value=np.float64(1.0e20)
    )
    data = np.linspace(275.0, 290.0, 2 * site_values.size, dtype="f4").reshape(
        2, site_values.size
    )
    return str(cmor4.cmorize(dataset, variable, [time, site], data, grid=grid))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write CMIP7 near-surface air temperature at sites with CMOR4."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()
    print(write_example(args.output_dir))


if __name__ == "__main__":
    main()
