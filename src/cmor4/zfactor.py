"""ZFactor metadata record for hybrid-coordinate formula terms."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from pydantic import Field
from typing import Annotated
from pydantic import BeforeValidator

from ._table_utils import (
    is_table_value,
    metadata_value_matches,
    parse_table_value,
    table_dimensions,
)
from .exceptions import TableValidationError
from .metadata import MetadataModel

if TYPE_CHECKING:
    from .tables import ProjectTables


def _str_tuple(v: Any) -> tuple[str, ...] | None:
    if v is None:
        return None
    if isinstance(v, str):
        return (v,)
    return tuple(str(x) for x in v)


def _str_seq(v: Any) -> list[str] | tuple[str, ...] | None:
    if v is None:
        return None
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, tuple):
        return tuple(str(x) for x in v)
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _float_or_none(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


StrTuple = Annotated[tuple[str, ...] | None, BeforeValidator(_str_tuple)]
StrSeq   = Annotated[list[str] | tuple[str, ...] | None, BeforeValidator(_str_seq)]
CoercedF = Annotated[float | None,           BeforeValidator(_float_or_none)]


class ZFactor(MetadataModel):
    """Metadata and values for one hybrid-coordinate formula term.

    Parameters
    ----------
    name:
        Formula-term name.
    values, data:
        Formula-term data values (``values`` takes precedence).
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
    data: Any = None
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
    # Project-table construction
    # ------------------------------------------------------------------

    @classmethod
    def from_project(cls, project: ProjectTables, name: str, **values: Any) -> ZFactor:
        """Create a ZFactor by merging authoritative formula-table metadata."""
        data: dict[str, Any] = {"name": name, **values}
        data = cls._apply_table_defaults(data, project)
        return cls.model_validate(data)

    @classmethod
    def _apply_table_defaults(
        cls, data: dict[str, Any], project: ProjectTables
    ) -> dict[str, Any]:
        entry_name, entry = cls._resolve_entry(data, project)
        if entry is None:
            return data
        data.setdefault("table_entry", entry_name)
        cls._validate_metadata(data, entry_name, entry, ("units", "standard_name", "long_name"))
        for key in ("out_name", "units", "standard_name", "long_name"):
            val = entry.get(key)
            if is_table_value(val):
                data.setdefault(key, val)
        for key in ("valid_min", "valid_max", "ok_min_mean_abs", "ok_max_mean_abs"):
            val = entry.get(key)
            if is_table_value(val):
                data.setdefault(key, parse_table_value(val))
        if "dimensions" not in data and is_table_value(entry.get("dimensions")):
            data["dimensions"] = table_dimensions(entry)
        if "bounds" in data:
            bname = str(data.get("bounds_name") or
                        f"{data.get('out_name', data.get('name', ''))}_bnds")
            be = project.formula_entries.get(bname)
            if be:
                data.setdefault("bounds_name", bname)
                ba = dict(data.get("bounds_attrs") or {})
                for key in ("units", "standard_name", "long_name"):
                    val = be.get(key)
                    if is_table_value(val):
                        ba.setdefault(key, val)
                if ba:
                    data["bounds_attrs"] = ba
        return data

    @classmethod
    def _resolve_entry(
        cls, data: dict[str, Any], project: ProjectTables
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        requested = str(
            data.get("table_entry") or data.get("formula_entry") or data.get("name") or ""
        )
        fe: dict[str, Mapping[str, Any]] = project.formula_entries
        if requested in fe:
            return requested, fe[requested]
        m = [(n, e) for n, e in fe.items() if str(e.get("out_name", "")) == requested]
        if len(m) == 1:
            return m[0]
        return None, None

    @classmethod
    def _validate_metadata(
        cls,
        data: dict[str, Any],
        entry_name: str | None,
        table_values: Mapping[str, Any],
        keys: Sequence[str],
    ) -> None:
        for key in keys:
            expected = table_values.get(key)
            user_val = data.get(key)
            if (
                is_table_value(expected)
                and user_val not in (None, "")
                and not metadata_value_matches(user_val, expected)
            ):
                raise TableValidationError(
                    f"formula term {entry_name!r} {key}={user_val!r} "
                    f"does not match table value {expected!r}."
                )

    # ------------------------------------------------------------------
    # Public instance API (existing interface preserved)
    # ------------------------------------------------------------------

    def resolve_table_entry(
        self, project: ProjectTables
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve a formula-term table entry for this ZFactor."""
        return self._resolve_entry(self.to_dict(), project)

    # Instance-method shim called by ProjectTables.validate_components
    def _validate_metadata_instance(
        self,
        entry_type: str,
        entry_name: str | None,
        table_values: Mapping[str, Any],
        keys: Sequence[str],
    ) -> None:
        self._validate_metadata(self.to_dict(), entry_name, table_values, keys)

    def attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for this formula-term variable."""
        attrs = self.netcdf_attrs(self.attrs)
        for key in ("units", "standard_name", "long_name"):
            if key in self:
                attrs[key] = self[key]
        return attrs

    def bounds_attributes(self) -> dict[str, Any]:
        """Return NetCDF attributes for the bounds variable."""
        return self.netcdf_attrs(self.bounds_attrs)

    def values_array(self) -> np.ndarray:
        """Return formula-term values as a NetCDF-ready numpy array."""
        return self.netcdf_array(self.get("values", self.get("data", [])))

    def bounds_array(self) -> np.ndarray:
        """Return formula-term bounds as a NetCDF-ready numpy array."""
        return self.netcdf_array(self["bounds"])
