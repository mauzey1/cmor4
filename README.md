# CMOR4  (WORK IN PROGRESS)

`cmor4` is a small, dictionary-driven CMOR-like package for creating CF-style
NetCDF datasets with `xarray`. It is designed for new drivers that can provide
Python dictionaries directly instead of writing CMOR JSON input files.

The package does not try to preserve older CMOR or CMIP6-specific setup
parameters. A driver provides dataset metadata, variable metadata, coordinate
axes, optional grids or z-factors, and data arrays; `cmor4` creates an
`xarray.Dataset` and can write the result to a CMOR-like output path.

## Project Tables

For table-backed validation, pass the project CV file and the variable table
files that define the variables you want to write. Coordinate metadata is read
from the project coordinate table, with any `axis_entry` metadata in the loaded
variable table files taking precedence for matching axes. Formula-term metadata
is read from the project formula terms table. These files come from the project
table repositories, not from `cmor4`.

With `project=`, the caller normally supplies data values, bounds, source-time
units, missing values, and non-table custom metadata. Standardized variable
attributes (`units`, `standard_name`, `long_name`, `cell_methods`,
`cell_measures`, and `comment`), axis attributes, and z-factor attributes come
from the loaded project tables. Scalar axes such as `height2m` are added from
the table when a variable requires them and the table provides a fixed value.

Examples:

- CMIP7: `cmip7-cmor-tables/tables-cvs/cmor-cvs.json` and variable tables such
  as `cmip7-cmor-tables/tables/CMIP7_ocean.json`
- obs4MIPs: `obs4MIPs-cmor-tables/Tables/obs4MIPs_CV.json` and variable tables
  such as `obs4MIPs-cmor-tables/Tables/obs4MIPs_Amon.json`
- DRCDP: `PCMDI/DRCDP` table files such as `Tables/DRCDP_CV.json`,
  `Tables/DRCDP_AP1hr.json`, and `Tables/DRCDP_APday.json`

When `project=` is provided, `cmor4` validates controlled values against the CV
and validates variable names, dimensions, frequency, realm, and table identity
against the loaded variable table entries. Table-backed variable attributes
such as units, standard names, long names, cell methods, cell measures, and
comments are applied from the variable table entries.

The test suite uses project table repositories checked out as git submodules
under `project_tables/`:

```bash
git submodule update --init --recursive
```

## Installation

From this repository:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

For development in this checkout, use the included environment:

```bash
./venv/bin/python -m pip install -e .
git submodule update --init --recursive
./venv/bin/python -m unittest discover -s tests
```

## Short Example

```python
import numpy as np
import cmor4

project = cmor4.ProjectTables.from_directory(
    "../cmip7-cmor-tables",
    cv_file="tables-cvs/cmor-cvs.json",
    variable_tables=["tables/CMIP7_atmos.json"],
    coordinate_table="tables/CMIP7_coordinate.json",
    formula_table="tables/CMIP7_formula_terms.json",
)

dataset = {
    "activity_id": "CMIP",
    "mip_era": "CMIP7",
    "institution_id": "CCCma",
    "source_id": "DUMMY-MODEL",
    "experiment_id": "amip",
    "realization_index": "r9",
    "initialization_index": "i1",
    "physics_index": "p1",
    "forcing_index": "f3",
    "frequency": "mon",
    "region": "glb",
    "grid_label": "g999",
    "outpath": "cmor_output",
}

variable = {"name": "tas_tavg-h2m-hxy-u", "missing_value": np.float32(1.0e20)}

axes = [
    {
        "name": "time",
        "values": [15.0, 45.0],
        "bounds": [[0.0, 30.0], [30.0, 60.0]],
        "units": "days since 2000-01-01",
    },
    {
        "name": "latitude",
        "values": [-45.0, 45.0],
        "bounds": [[-90.0, 0.0], [0.0, 90.0]],
    },
    {
        "name": "longitude",
        "values": [90.0, 270.0],
        "bounds": [[0.0, 180.0], [180.0, 360.0]],
    },
]

data = np.ones((2, 2, 2), dtype="f4") * 288.0
result = cmor4.cmorize(dataset, variable, axes, data, project=project)

print(result.path)
print(result.dataset)
```

Use `cmor4.create_dataset(...)` when you want the `xarray.Dataset` without
writing a file, and `cmor4.open_dataset(path)` to read NetCDF output back with
`xarray`.

## Notebooks

CMIP7 notebook examples with bundled synthetic NetCDF inputs live in
`notebooks/`. They write cmor4 output files and include matplotlib
visualizations.
