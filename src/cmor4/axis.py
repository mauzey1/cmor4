"""Axis metadata record."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import Field

from .metadata import AxisStr, BoolCoerced, CoercedF, MetadataModel, StrSeq


class Axis(MetadataModel):
    """Metadata and coordinate values for one data axis.

    Construct via :meth:`ProjectTables.axis` to merge authoritative
    coordinate-table metadata::

        axis = project.axis("latitude", values=np.linspace(-90, 90, 180))

    Construct directly when project tables are not available::

        axis = Axis(name="latitude", values=np.linspace(-90, 90, 180),
                    units="degrees_north", axis="Y")

    Parameters
    ----------
    name:
        Logical axis name matching a variable dimension.
    values:
        Coordinate values (list, numpy array, scalar).
    bounds:
        Optional coordinate cell bounds.
    dimensions:
        Underlying dimensions for auxiliary coordinates.
    units:
        CF units string.
    standard_name, long_name:
        CF / NetCDF attributes.
    axis:
        CF axis designator — ``"X"``, ``"Y"``, ``"Z"``, or ``"T"``.
        Lower-case inputs are automatically upper-cased.
    positive:
        CF ``"up"`` or ``"down"`` for vertical axes.
    formula:
        Formula for computing coordinate values.
    valid_min, valid_max:
        Valid coordinate value range.
    out_name:
        Output coordinate variable name.
    table_entry, axis_entry, coordinate:
        Coordinate-table entry name selectors, tried in order.
    grid_table_entry, grid_coordinate:
        Grid coordinate-table entry selectors.
    scalar:
        Write as a scalar coordinate.
    auxiliary:
        Write as an auxiliary coordinate.
    auxiliary_name:
        Output name for the auxiliary coordinate variable.
    auxiliary_attrs:
        Extra attributes for the auxiliary variable.
    climatology:
        Climatology bounds control.
    generic_level_name:
        Generic level selector.
    z_factors, z_bounds_factors:
        Formula-term names for this axis.
    bounds_name, bounds_dim:
        Output bounds variable name / dimension name.
    bounds_attrs:
        Extra attributes for the bounds variable.
    attrs:
        Extra attributes for the coordinate variable.
    stored_direction:
        Expected ordering of coordinate values: ``"increasing"`` or
        ``"decreasing"``.
    """

    name: str
    values: Any = None
    bounds: Any = None
    dimensions: StrSeq = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    axis: AxisStr = None
    positive: Literal["up", "down"] | None = None
    formula: str | None = None
    valid_min: CoercedF = None
    valid_max: CoercedF = None
    out_name: str | None = None
    table_entry: str | None = None
    axis_entry: str | None = None
    coordinate: str | None = None
    grid_table_entry: str | None = None
    grid_coordinate: str | None = None
    scalar: bool | None = None
    auxiliary: bool | None = None
    auxiliary_name: str | None = None
    auxiliary_attrs: dict[str, Any] = Field(default_factory=dict)
    climatology: BoolCoerced = None
    generic_level_name: str | None = None
    z_factors: str | None = None
    z_bounds_factors: str | None = None
    requested: Any = None
    requested_bounds: Any = None
    bounds_values: Any = None
    must_have_bounds: BoolCoerced = None
    stored_direction: str | None = None
    tolerance: CoercedF = None
    bounds_name: str | None = None
    bounds_dim: str | None = None
    bounds_attrs: dict[str, Any] = Field(default_factory=dict)
    attrs: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # NetCDF output helpers
    # ------------------------------------------------------------------

    def attributes(self, *, include_units: bool = True) -> dict[str, Any]:
        """Return NetCDF attributes for this coordinate variable."""
        attrs = self.netcdf_attrs(self.attrs)
        if include_units and self.units is not None:
            attrs["units"] = self.units
        for key, val in (
            ("standard_name", self.standard_name),
            ("long_name", self.long_name),
            ("axis", self.axis),
            ("positive", self.positive),
            ("formula", self.formula),
        ):
            if val is not None:
                attrs[key] = val
        return attrs

    def auxiliary_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the auxiliary coordinate variable."""
        return self.netcdf_attrs(self.auxiliary_attrs)

    def bounds_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the bounds variable."""
        return self.netcdf_attrs(self.bounds_attrs)

    def values_array(self) -> np.ndarray:
        """Return coordinate values as a NetCDF-ready numpy array."""
        return self.netcdf_array(self.values if self.values is not None else [])

    def bounds_array(self) -> np.ndarray:
        """Return coordinate bounds as a NetCDF-ready numpy array."""
        return self.netcdf_array(self.bounds)
