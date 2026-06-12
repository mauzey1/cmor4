from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ._table_utils import (
    is_table_value,
    metadata_value_matches,
    parse_table_value,
    table_dimensions,
)
from .exceptions import TableValidationError
from .metadata import _MetadataRecord


@dataclass(frozen=True)
class ZFactor(_MetadataRecord):
    """Metadata and values for one hybrid-coordinate formula term.

    Parameters
    ----------
    name
        Requested formula-term name.
    values
        Formula-term data values.
    data
        Formula-term data values (alternative to values).
    dimensions
        Logical dimensions for the formula-term variable.
    units
        Formula-term units attribute.
    standard_name
        CF standard_name attribute.
    long_name
        Formula-term long_name attribute.
    out_name
        Output formula-term variable name.
    table_entry
        Formula-term table entry name selector.
    formula_entry
        Formula-term table entry name selector.
    bounds
        Optional formula-term bounds values.
    bounds_name
        Output bounds variable name.
    bounds_dim
        Output bounds dimension name.
    bounds_attrs
        Extra attributes for the bounds variable.
    valid_min
        Minimum valid value for formula term.
    valid_max
        Maximum valid value for formula term.
    ok_min_mean_abs
        Minimum acceptable absolute mean value.
    ok_max_mean_abs
        Maximum acceptable absolute mean value.
    attrs
        Extra NetCDF attributes for the formula-term variable.
    extra
        Additional mapping keys preserved by the metadata record.
    project
        Optional project tables used to resolve and merge formula-term
        metadata during construction.
    """

    name: str
    values: Any = None
    data: Any = None
    dimensions: tuple[str, ...] | list[str] | None = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    out_name: str | None = None
    table_entry: str | None = None
    formula_entry: str | None = None
    bounds: Any = None
    bounds_name: str | None = None
    bounds_dim: str | None = None
    bounds_attrs: Mapping[str, Any] = field(default_factory=dict)
    valid_min: Any = None
    valid_max: Any = None
    ok_min_mean_abs: Any = None
    ok_max_mean_abs: Any = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)
    project: InitVar[Any | None] = None

    def __post_init__(self, project: Any | None) -> None:
        if project is None:
            return
        merged = self._merge_table_entry(project)
        for key, value in merged.to_dict().items():
            object.__setattr__(self, key, value)

    def _merge_table_entry(self, project: Any) -> "ZFactor":
        """Merge authoritative formula-term metadata into this z-factor.

        Parameters
        ----------
        project
            Project table loader containing formula-term entries.

        Returns
        -------
        ZFactor
            New z-factor metadata record with table defaults applied.
        """

        merged = self.to_dict()
        entry_name, entry = self.resolve_table_entry(project)
        if entry is None:
            return ZFactor.from_mapping(merged)
        merged.setdefault("table_entry", entry_name)
        self._validate_metadata(
            "formula term",
            entry_name,
            entry,
            ("units", "standard_name", "long_name"),
        )
        for key in ("out_name", "units", "standard_name", "long_name"):
            value = entry.get(key)
            if is_table_value(value):
                merged.setdefault(key, value)
        for key in (
            "valid_min",
            "valid_max",
            "ok_min_mean_abs",
            "ok_max_mean_abs",
        ):
            value = entry.get(key)
            if is_table_value(value):
                merged.setdefault(key, parse_table_value(value))
        if "dimensions" not in merged and is_table_value(
            entry.get("dimensions")
        ):
            merged["dimensions"] = table_dimensions(entry)
        if "bounds" in merged:
            bounds_name = str(
                merged.get("bounds_name")
                or f"{merged.get('out_name', self.name)}_bnds"
            )
            bounds_entry = project.formula_entries.get(bounds_name)
            if bounds_entry:
                merged.setdefault("bounds_name", bounds_name)
                bounds_attrs = dict(merged.get("bounds_attrs", {}))
                for key in ("units", "standard_name", "long_name"):
                    value = bounds_entry.get(key)
                    if is_table_value(value):
                        bounds_attrs.setdefault(key, value)
                if bounds_attrs:
                    merged["bounds_attrs"] = bounds_attrs
        return ZFactor.from_mapping(merged)

    def resolve_table_entry(
        self, project: Any
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve a formula-term table entry from this z-factor.

        Formula terms (z-factors) are variables used in coordinate conversion
        formulas for hybrid vertical coordinates, such as hybrid sigma-pressure
        coordinates. This method searches the project's formula table for a
        matching entry by name or out_name.

        Parameters
        ----------
        project
            Project table loader containing formula-term entries from the
            loaded formula terms table.

        Returns
        -------
        tuple[str | None, Mapping[str, Any] | None]
            A tuple containing:

            - entry_name (str or None): Matched formula-term entry name
            - entry (dict or None): Formula-term entry metadata including
              units, standard_name, and dimensions

            Returns ``(None, None)`` if no matching entry is found.

        Examples
        --------
        Resolve standard formula term::

            zfactor = ZFactor(name="ap", values=[...])
            entry_name, entry = zfactor.resolve_table_entry(project)
            # Returns ("ap", {...}) from formula terms table

        Resolve by out_name::

            zfactor = ZFactor(name="a", formula_entry="ap")
            entry_name, entry = zfactor.resolve_table_entry(project)
            # Returns ("ap", {...}) matching the formula_entry
        """

        requested = str(
            self.table_entry or self.formula_entry or self.name or ""
        )
        if requested in project.formula_entries:
            return requested, project.formula_entries[requested]
        matches = [
            (name, entry)
            for name, entry in project.formula_entries.items()
            if str(entry.get("out_name", "")) == requested
        ]
        if len(matches) == 1:
            return matches[0]
        return None, None

    def attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for this formula-term variable.

        This method constructs the complete set of NetCDF attributes for the
        formula-term variable, including CF-required metadata (units,
        standard_name, long_name) merged with any user-provided attributes.

        Returns
        -------
        dict[str, Any]
            NetCDF-safe formula-term attributes suitable for assignment to
            the formula-term variable in the output dataset. Includes units,
            standard_name, and long_name when defined, plus any additional
            attributes from the ``attrs`` field.

        Examples
        --------
        Get attributes for hybrid sigma coefficient::

            zfactor = project.zfactor("ap", values=[...])
            attrs = zfactor.attributes()
            # attrs = {
            #     "units": "Pa",
            #     "long_name": "vertical coordinate formula term: ap(k)",
            #     ...
            # }

        Custom attributes for formula term::

            zfactor = ZFactor(
                name="ps",
                values=surface_pressure,
                attrs={"comment": "Surface pressure at model timestep"}
            )
            attrs = zfactor.attributes()
            # Includes custom comment plus standard attributes
        """

        attrs = self.netcdf_attrs(self.attrs)
        for key in ("units", "standard_name", "long_name"):
            if key in self:
                attrs[key] = self[key]
        return attrs

    def bounds_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for this formula-term bounds variable.

        Some formula terms require bounds variables when they represent
        quantities that vary across coordinate cell interfaces (e.g., hybrid
        coordinate coefficients at layer boundaries).

        Returns
        -------
        dict[str, Any]
            NetCDF-safe attributes for the formula-term bounds variable,
            filtered to include only values compatible with NetCDF format.
            May include units, standard_name, and long_name from the
            bounds_attrs field or formula table bounds entries.

        Examples
        --------
        Get bounds attributes for hybrid coefficient::

            zfactor = ZFactor(
                name="ap",
                values=[...],
                bounds=[...],
                bounds_attrs={"long_name": "vertical coordinate formula ..."}
            )
            attrs = zfactor.bounds_attributes()
            # attrs = {"long_name": "vertical coordinate formula ..."}
        """

        return self.netcdf_attrs(self.bounds_attrs)

    def values_array(self) -> np.ndarray:
        """Return this formula term's values as a NetCDF-ready array.

        This method converts the formula term values to a numpy array with
        appropriate dtype for NetCDF output, handling various input types
        including lists, tuples, numpy arrays, and xarray DataArrays.

        Returns
        -------
        numpy.ndarray
            Formula-term values as a numpy array with appropriate dtype for
            NetCDF serialization. Returns an empty array if neither ``values``
            nor ``data`` fields are defined.

        Notes
        -----
        This method checks both ``values`` and ``data`` fields, preferring
        ``values`` if both are present. This allows flexibility in how
        formula term data is provided.

        Examples
        --------
        Get values for hybrid sigma coefficient::

            zfactor = ZFactor(
                name="ap",
                values=[100000, 95000, 90000, ...]
            )
            arr = zfactor.values_array()
            # Returns np.array([100000., 95000., 90000., ...])

        Using data field instead of values::

            zfactor = ZFactor(name="ps", data=surface_pressure_array)
            arr = zfactor.values_array()
            # Returns surface_pressure_array as numpy array
        """

        return self.netcdf_array(self.get("values", self.get("data", [])))

    def bounds_array(self) -> np.ndarray:
        """Return this formula term's bounds as a NetCDF-ready array.

        This method converts the formula term bounds to a numpy array with
        shape (n_values, 2) and dtype suitable for NetCDF output. Bounds are
        required for formula terms that represent quantities at coordinate
        cell interfaces.

        Returns
        -------
        numpy.ndarray
            Formula-term bounds as a numpy array with shape (n, 2) where each
            row contains [lower_bound, upper_bound] for the corresponding
            formula term level or interface.

        Raises
        ------
        KeyError
            If the formula term does not have bounds defined.

        Examples
        --------
        Get bounds for hybrid coefficient at interfaces::

            zfactor = ZFactor(
                name="ap",
                values=[100000, 95000, 90000],
                bounds=[[102000, 98000], [98000, 92000], [92000, 88000]]
            )
            bnds = zfactor.bounds_array()
            # Returns array([[102000, 98000], [98000, 92000], [92000, 88000]])
        """

        return self.netcdf_array(self["bounds"])

    def _validate_metadata(
        self,
        entry_type: str,
        entry_name: str | None,
        table_values: Mapping[str, Any],
        keys: Sequence[str],
    ) -> None:
        user_values = self.to_dict()
        for key in keys:
            expected = table_values.get(key)
            if (
                is_table_value(expected)
                and key in user_values
                and not metadata_value_matches(user_values[key], expected)
            ):
                raise TableValidationError(
                    f"{entry_type} {entry_name!r} {key}={user_values[key]!r} "
                    f"does not match table value {expected!r}."
                )
