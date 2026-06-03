"""CMOR-like NetCDF creation with xarray."""

from .core import (
    Cmor4Result,
    build_output_path,
    cmorize,
    create_dataset,
    open_dataset,
    write_netcdf,
)
from .axis import Axis
from .exceptions import TableValidationError
from .variable import Variable
from .zfactor import ZFactor
from .tables import ProjectTables

__all__ = [
    "Axis",
    "Cmor4Result",
    "build_output_path",
    "cmorize",
    "create_dataset",
    "open_dataset",
    "ProjectTables",
    "TableValidationError",
    "Variable",
    "write_netcdf",
    "ZFactor",
]

__version__ = "0.1.0"
