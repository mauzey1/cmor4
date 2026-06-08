from __future__ import annotations

from dataclasses import dataclass, field
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
    name:
        Requested formula-term name.
    values, data:
        Formula-term data values.
    dimensions:
        Logical dimensions for the formula-term variable.
    units, standard_name, long_name:
        NetCDF formula-term metadata attributes.
    out_name:
        Output formula-term variable name.
    table_entry, formula_entry:
        Formula-term table entry selectors.
    bounds:
        Optional formula-term bounds values.
    bounds_name, bounds_dim:
        Output bounds variable and bounds dimension names.
    bounds_attrs:
        Extra attributes for the bounds variable.
    valid_min, valid_max, ok_min_mean_abs, ok_max_mean_abs:
        Value validation limits.
    attrs:
        Extra NetCDF attributes for the formula-term variable.
    extra:
        Additional mapping keys preserved by the metadata record.
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

    def merge_table_entry(self, project: Any) -> "ZFactor":
        """Merge authoritative formula-term metadata into this z-factor.

        Parameters
        ----------
        project:
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

        Parameters
        ----------
        project:
            Project table loader containing formula-term entries.

        Returns
        -------
        tuple[str | None, Mapping[str, Any] | None]
            Matched entry name and table metadata, or ``(None, None)``.
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

        Returns
        -------
        dict[str, Any]
            NetCDF-safe formula-term attributes.
        """

        attrs = self.netcdf_attrs(self.attrs)
        for key in ("units", "standard_name", "long_name"):
            if key in self:
                attrs[key] = self[key]
        return attrs

    def bounds_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for this formula-term bounds variable.

        Returns
        -------
        dict[str, Any]
            NetCDF-safe bounds variable attributes.
        """

        return self.netcdf_attrs(self.bounds_attrs)

    def values_array(self) -> np.ndarray:
        """Return this formula term's values as a NetCDF-ready array.

        Returns
        -------
        numpy.ndarray
            Formula-term values converted to a NetCDF-compatible array.
        """

        return self.netcdf_array(self.get("values", self.get("data", [])))

    def bounds_array(self) -> np.ndarray:
        """Return this formula term's bounds as a NetCDF-ready array.

        Returns
        -------
        numpy.ndarray
            Formula-term bounds converted to a NetCDF-compatible array.
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
