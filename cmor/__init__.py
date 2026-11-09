"""CF-compliant NetCDF creation with xarray."""

__version__ = "4.0.0a1"

from .dataset import (
    build_output_path,
    cmorize,
    create_dataset,
    open_dataset,
    write_netcdf,
)
from .axis import Axis
from .cv import ControlledVocabulary
from .datasetinfo import DatasetInfo
from .exceptions import (
    TableValidationError,
    AxisValidationError,
    ControlledVocabularyError,
    VariableValidationError,
)
from .grid import Grid
from .variable import Variable
from .writer import DatasetWriter
from .zfactor import ZFactor
from .tables import ProjectTables

__all__ = [
    "Axis",
    "AxisValidationError",
    "ControlledVocabularyError",
    "build_output_path",
    "cmorize",
    "create_dataset",
    "ControlledVocabulary",
    "DatasetWriter",
    "Grid",
    "open_dataset",
    "DatasetInfo",
    "ProjectTables",
    "TableValidationError",
    "Variable",
    "VariableValidationError",
    "write_netcdf",
    "ZFactor",
]
