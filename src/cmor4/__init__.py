"""Dictionary-driven CMOR-like NetCDF creation with xarray."""

from .core import (
    Cmor4Result,
    build_output_path,
    cmorize,
    create_dataset,
    open_dataset,
    write_netcdf,
)
from .tables import ProjectTables, TableValidationError

__all__ = [
    "Cmor4Result",
    "build_output_path",
    "cmorize",
    "create_dataset",
    "open_dataset",
    "ProjectTables",
    "TableValidationError",
    "write_netcdf",
]

__version__ = "0.1.0"
