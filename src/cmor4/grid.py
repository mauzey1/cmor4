"""Grid metadata record for runtime grid dimensions and CF projections."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from pydantic import Field, model_validator

from .metadata import MetadataModel, StrSeq, StrTuple

_LATITUDE_PARAMS: frozenset[str] = frozenset(
    {
        "grid_north_pole_latitude",
        "latitude_of_projection_origin",
        "standard_parallel",
        "standard_parallel1",
        "standard_parallel2",
    }
)
_LONGITUDE_PARAMS: frozenset[str] = frozenset(
    {
        "grid_north_pole_longitude",
        "longitude_of_prime_meridian",
        "longitude_of_central_meridian",
        "longitude_of_projection_origin",
        "north_pole_grid_longitude",
    }
)
_NONNEG_PARAMS: frozenset[str] = frozenset(
    {
        "scale_factor_at_central_meridian",
        "scale_factor_at_projection_origin",
    }
)


class Grid(MetadataModel):
    """Runtime grid dimensions and optional CF grid-mapping metadata.

    Construct via :meth:`ProjectTables.grid` to merge grid-table metadata::

        grid = project.grid(
            dimensions=["x", "y"],
            mapping_name="lambert_azimuthal_equal_area",
            params={...},
            latitude=lat_2d,
            longitude=lon_2d,
        )

    Parameters
    ----------
    dimensions:
        Spatial (non-time) dimensions for the data variable.
    name, table_entry, mapping_entry:
        Grid table entry name selectors.
    mapping_var:
        Name of the scalar grid-mapping variable to write.
    mapping_name, grid_mapping_name:
        CF ``grid_mapping_name``.
    coordinates:
        Auxiliary coordinate names.
    params:
        Grid-mapping parameter dict.  Each value may be a scalar or a
        ``(value, units)`` tuple.
    attrs:
        Extra NetCDF attributes for the grid-mapping variable.
    latitude, longitude:
        Optional 2-D geographic coordinate arrays on the grid.
    latitude_vertices, longitude_vertices:
        Optional cell-vertex arrays (shape ``(*spatial, n_vertices)``).
    vertices_dim:
        Name for the vertices dimension (default ``"vertices"``).
    """

    dimensions: StrTuple = None
    name: str | None = None
    table_entry: str | None = None
    mapping_entry: str | None = None
    mapping_var: str | None = None
    mapping_name: str | None = None
    grid_mapping_name: str | None = None
    coordinates: StrSeq = None
    params: dict[str, Any] = Field(default_factory=dict)
    attrs: dict[str, Any] = Field(default_factory=dict)
    latitude: Any = None
    longitude: Any = None
    latitude_vertices: Any = None
    longitude_vertices: Any = None
    vertices_dim: str = "vertices"

    @model_validator(mode="after")
    def _check_spatial_arrays(self) -> Grid:
        """Validate that latitude / longitude shapes are consistent."""
        lat, lon = self.latitude, self.longitude
        if lat is None or lon is None:
            return self
        la, lo = np.asarray(lat), np.asarray(lon)
        if la.shape != lo.shape:
            raise ValueError(
                f"latitude shape {la.shape} does not match longitude shape {lo.shape}."
            )
        if self.dimensions is not None and la.ndim != len(self.dimensions):
            raise ValueError(
                f"lat/lon arrays have {la.ndim} dimension(s) but "
                f"{len(self.dimensions)} grid dimension(s) were specified."
            )
        for name, arr, ref in (
            ("latitude_vertices", self.latitude_vertices, la),
            ("longitude_vertices", self.longitude_vertices, lo),
        ):
            if arr is not None:
                a = np.asarray(arr)
                if a.shape[: ref.ndim] != ref.shape:
                    raise ValueError(
                        f"{name} shape {a.shape} is incompatible "
                        f"with coordinate shape {ref.shape}."
                    )
        return self

    # ------------------------------------------------------------------
    # NetCDF output helpers
    # ------------------------------------------------------------------

    @property
    def variable_name(self) -> str:
        """Return the output grid-mapping variable name (default: ``"crs"``)."""
        return str(self.mapping_var or "crs")

    def variable_dimensions(self, variable: Any) -> tuple[str, ...] | None:
        """Return the full dimension tuple for the data variable.

        Combines time from *variable* with the grid's spatial dimensions.
        """
        if self.dimensions:
            grid_dims = tuple(str(n) for n in self.dimensions)
            var_dims = variable.dimensions
            if var_dims:
                time_dims = tuple(str(d) for d in var_dims if str(d).lower() == "time")
                return time_dims + grid_dims
            return grid_dims
        dims = variable.dimensions
        if dims:
            return tuple(str(n) for n in dims)
        return None

    @property
    def has_mapping(self) -> bool:
        """True when this grid needs a CF grid-mapping variable."""
        return bool(
            self.mapping_name or self.grid_mapping_name or self.params or self.attrs
        )

    def mapping_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the grid-mapping scalar variable."""
        attrs = self.netcdf_attrs(self.attrs)
        mname = self.mapping_name or self.grid_mapping_name
        if mname:
            attrs["grid_mapping_name"] = mname
        for key, val in self.params.items():
            if not _valid_param(str(key), val):
                continue
            if isinstance(val, (list, tuple)) and val:
                attrs[key] = val[0]
                if len(val) > 1 and val[1]:
                    attrs[f"{key}_units"] = val[1]
            else:
                attrs[key] = val
        return self.netcdf_attrs(attrs)


def _valid_param(name: str, value: Any) -> bool:
    num = _primary_num(value)
    if num is None:
        return True
    if name in _LATITUDE_PARAMS and not -90.0 <= num <= 90.0:
        warnings.warn(
            f"{name} parameter must be between -90 and 90 degrees_north; "
            "it will not be set.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    if name in _LONGITUDE_PARAMS and not -180.0 <= num <= 180.0:
        warnings.warn(
            f"{name} parameter must be between -180 and 180 degrees_east; "
            "it will not be set.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    if name in _NONNEG_PARAMS and num < 0.0:
        warnings.warn(
            f"{name} parameter must be positive; it will not be set.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    return True


def _primary_num(value: Any) -> float | None:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
