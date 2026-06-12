# CMOR4

**Climate Model Output Rewriter (CMOR) - Version 4**

CMOR4 is a Python package for creating CF-compliant climate model output in NetCDF format. It validates metadata against project tables (CMIP7, obs4MIPs, DRCDP) and constructs xarray datasets from Python metadata objects, streamlining the production of standards-compliant climate datasets.

## Key Features

- **Pythonic API**: Create datasets using Python objects instead of JSON configuration files
- **Project table validation**: Ensures compliance with projects using CMOR table formats, currently supporting [CMIP7](https://github.com/WCRP-CMIP/cmip7-cmor-tables), [obs4MIPs](https://github.com/PCMDI/obs4MIPs-cmor-tables), and [DRCDP](https://github.com/PCMDI/DRCDP)
- **xarray integration**: Built on xarray for modern, Pythonic data handling
- **Minimal metadata entry**: Only specify variable names and data - CF attributes are applied automatically from tables
- **CF compliance**: Produces Climate and Forecast conventions-compliant NetCDF files

## Installation

Install from source:

```bash
git clone https://github.com/mauzey1/cmor4.git
cd cmor4
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
git submodule update --init --recursive
```

## Testing

Run the test suite:

```bash
python -m unittest discover -s tests
```

Run code style checks:

```bash
python -m pycodestyle src tests
```

## Workflow

1. **Load project tables**: Initialize `ProjectTables` with CV and variable tables
2. **Define metadata**: Use factory methods to create dataset, variable, and axis objects
3. **Provide data**: Supply numpy arrays or xarray DataArrays
4. **Create output**: Call `cmorize()` to validate and write CF-compliant NetCDF

## Example CMOR4 program

```python
import numpy as np
import cmor4

# Load project tables
project = cmor4.ProjectTables.from_directory(
    "project_tables/cmip7-cmor-tables",
    cv_file="tables-cvs/cmor-cvs.json",
    variable_tables=["tables/CMIP7_atmos.json"],
    coordinate_table="tables/CMIP7_coordinate.json",
    formula_table="tables/CMIP7_formula_terms.json",
)

# Define dataset metadata
dataset = project.dataset_info({
    "mip_era": "CMIP7",
    "activity_id": "CMIP",
    "institution_id": "CCCma",
    "source_id": "DUMMY-MODEL",
    "experiment_id": "amip",
    "license_id": "CC-BY-4.0",
    "variant_label": "r1i1p1f1",
    "grid_label": "gn",
})

# Create variable and axes
variable = project.variable("tas", missing_value=1.0e20)

axes = [
    project.axis("time", values=[15.0, 45.0], 
                 bounds=[[0.0, 30.0], [30.0, 60.0]],
                 units="days since 2000-01-01"),
    project.axis("lat", values=np.linspace(-90, 90, 180)),
    project.axis("lon", values=np.linspace(0, 360, 360)),
]

# Prepare data
data = np.random.randn(2, 180, 360) + 288.0

# Write CMOR-compliant NetCDF file
result = cmor4.cmorize(dataset, variable, axes, data)
print(f"Created: {result.path}")
```

## Example notebooks

Jupyter notebook examples with visualizations are available in [notebooks](./notebooks/)

## Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/mauzey1/cmor4/issues)

## Acknowledgments

CMOR4 is developed and maintained by the Program for Climate Model Diagnosis and Intercomparison ([PCMDI](https://pcmdi.llnl.gov/)) at [Lawrence Livermore National Laboratory](https://www.llnl.gov/).
