"""ZFactor metadata record for hybrid-coordinate formula terms."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from .metadata import CoercedF, MetadataModel, StrSeq


class ZFactor(MetadataModel):
    """Metadata and values for one hybrid-coordinate formula term.

    Construct via :meth:`ProjectTables.zfactor` to merge authoritative
    formula-table metadata::

        zf = project.zfactor("ps", values=surface_pressure)

    Parameters
    ----------
    name:
        Formula-term name.
    values:
        Formula-term data values.
    dimensions:
        Logical dimensions for the formula-term variable.
    units, standard_name, long_name:
        CF / NetCDF attributes.
    out_name:
        Output variable name.
    table_entry, formula_entry:
        Formula table entry name selectors.
    bounds:
        Optional formula-term bounds values.
    bounds_name, bounds_dim:
        Output bounds variable name / dimension.
    bounds_attrs:
        Extra attributes for the bounds variable.
    valid_min, valid_max, ok_min_mean_abs, ok_max_mean_abs:
        Data-quality limits.
    attrs:
        Extra NetCDF attributes.
    """

    name: str
    values: Any = None
    dimensions: StrSeq = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    out_name: str | None = None
    table_entry: str | None = None
    formula_entry: str | None = None
    bounds: Any = None
    bounds_name: str | None = None
    bounds_dim: str | None = None
    bounds_attrs: dict[str, Any] = Field(default_factory=dict)
    valid_min: CoercedF = None
    valid_max: CoercedF = None
    ok_min_mean_abs: CoercedF = None
    ok_max_mean_abs: CoercedF = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # NetCDF output helpers
    # ------------------------------------------------------------------

    def attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for this formula-term variable."""
        attrs = self.netcdf_attrs(self.attrs)
        for key, val in (
            ("units", self.units),
            ("standard_name", self.standard_name),
            ("long_name", self.long_name),
        ):
            if val is not None:
                attrs[key] = val
        return attrs

    def bounds_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the bounds variable."""
        return self.netcdf_attrs(self.bounds_attrs)

    def values_array(self) -> np.ndarray:
        """Return formula-term values as a NetCDF-ready numpy array."""
        return self.netcdf_array(self.values if self.values is not None else [])

    def bounds_array(self) -> np.ndarray:
        """Return formula-term bounds as a NetCDF-ready numpy array."""
        return self.netcdf_array(self.bounds)
