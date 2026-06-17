"""Variable metadata record."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import Field
from typing import Annotated
from pydantic import BeforeValidator

from ._table_utils import (
    is_table_value,
    metadata_value_matches,
    parse_table_value,
    single_or_original,
    table_dimensions,
)
from .exceptions import TableValidationError
from .metadata import MetadataModel
from ._unit_conversion import units_are_convertible as _units_convertible

if TYPE_CHECKING:
    from .tables import ProjectTables


# ---------------------------------------------------------------------------
# Field-level coercions
# ---------------------------------------------------------------------------


def _str_tuple(v: Any) -> tuple[str, ...] | None:
    if v is None:
        return None
    if isinstance(v, str):
        return (v,)
    return tuple(str(x) for x in v)


def _str_seq(v: Any) -> list[str] | tuple[str, ...] | None:
    """Coerce to list-or-tuple of str, preserving the container type."""
    if v is None:
        return None
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, tuple):
        return tuple(str(x) for x in v)
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _int_tuple(v: Any) -> tuple[int, ...] | None:
    if v is None:
        return None
    return tuple(int(x) for x in v)


def _str_or_tuple(v: Any) -> str | tuple[str, ...] | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    items = tuple(str(x) for x in v)
    return items[0] if len(items) == 1 else items


def _float_or_none(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


StrTuple = Annotated[tuple[str, ...] | None, BeforeValidator(_str_tuple)]
StrSeq = Annotated[list[str] | tuple[str, ...] | None, BeforeValidator(_str_seq)]
IntTuple = Annotated[tuple[int, ...] | None, BeforeValidator(_int_tuple)]
StrOrTuple = Annotated[str | tuple[str, ...] | None, BeforeValidator(_str_or_tuple)]
CoercedF = Annotated[float | None, BeforeValidator(_float_or_none)]


# ---------------------------------------------------------------------------
# VariableEntry — lightweight result of a table lookup
# ---------------------------------------------------------------------------


class VariableEntry:
    """Resolved variable table entry.

    Parameters
    ----------
    name:
        Variable entry name in the table.
    table_id:
        Identifier of the table supplying the entry.
    entry:
        Raw variable-entry metadata dict from the JSON table.
    table_file:
        Path to the source table file, if available.
    table_header:
        Header metadata from the table file, if available.
    """

    __slots__ = ("name", "table_id", "entry", "table_file", "table_header")

    def __init__(
        self,
        name: str,
        table_id: str,
        entry: Mapping[str, Any],
        table_file: Path | None = None,
        table_header: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.table_id = table_id
        self.entry = entry
        self.table_file = table_file
        self.table_header = table_header


# ---------------------------------------------------------------------------
# Variable
# ---------------------------------------------------------------------------


class Variable(MetadataModel):
    """Metadata for the data variable being written.

    Construct directly when no project tables are needed::

        variable = Variable(name="tas", units="K")

    Construct via :meth:`ProjectTables.variable` to merge authoritative
    table metadata (units, standard_name, dimensions, etc.)::

        variable = project.variable("tas")

    Parameters
    ----------
    name:
        Variable or branded-variable name.
    id, variable_id:
        Output variable identifier (override).
    table_id:
        Table identifier when the same variable appears in multiple tables.
    dimensions:
        Ordered logical dimensions, coerced to a ``tuple[str, ...]``.
    units:
        CF units string.
    standard_name, long_name, cell_methods, cell_measures, comment:
        Standard CF / NetCDF attributes.
    flag_values, flag_meanings:
        CF flag-encoding attributes.
    missing_value, fill_value:
        Missing-value sentinel.
    chunksizes, chunks:
        NetCDF chunk sizes, coerced to ``tuple[int, ...]``.
    coordinates:
        Explicit ``coordinates`` attribute.
    formula_terms:
        Explicit ``formula_terms`` attribute.
    positive:
        CF vertical-positive direction: ``"up"`` or ``"down"``.
    frequency, realm, table_info:
        Project-table metadata fields.
    valid_min, valid_max, ok_min_mean_abs, ok_max_mean_abs:
        Data-quality limits, coerced from strings when necessary.
    attrs:
        Additional NetCDF variable attributes.
    """

    name: str
    id: str | None = None
    variable_id: str | None = None
    table_id: str | None = None
    dimensions: StrSeq = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    cell_methods: str | None = None
    cell_measures: str | None = None
    comment: str | None = None
    flag_values: str | None = None
    flag_meanings: str | None = None
    missing_value: float | int | None = None
    fill_value: float | int | None = None
    chunksizes: IntTuple = None
    chunks: IntTuple = None
    coordinates: StrOrTuple = None
    formula_terms: str | None = None
    positive: str | None = None
    frequency: str | None = None
    realm: str | None = None
    table_info: str | None = None
    valid_min: CoercedF = None
    valid_max: CoercedF = None
    ok_min_mean_abs: CoercedF = None
    ok_max_mean_abs: CoercedF = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Project-table construction (class-level; no project stored on model)
    # ------------------------------------------------------------------

    @classmethod
    def from_project(cls, project: ProjectTables, name: str, **values: Any) -> Variable:
        """Create a Variable by merging authoritative table metadata.

        Called by :meth:`ProjectTables.variable`; exceptions from table
        resolution (e.g. :exc:`~cmor4.exceptions.TableValidationError`)
        propagate naturally since merging happens before Pydantic validation.
        """
        data: dict[str, Any] = {"name": name, **values}
        data = cls._apply_table_defaults(data, project)
        return cls.model_validate(data)

    @classmethod
    def _apply_table_defaults(
        cls, data: dict[str, Any], project: ProjectTables
    ) -> dict[str, Any]:
        """Resolve the table entry for *data* and merge its defaults in-place."""
        entry = cls._resolve_entry(data, project)
        return cls._merge_entry(data, entry)

    @classmethod
    def _resolve_entry(
        cls, data: dict[str, Any], project: ProjectTables
    ) -> VariableEntry:
        """Look up the VariableEntry matching *data* in *project*.

        Raises
        ------
        TableValidationError
            If the variable is not found or is ambiguous.
        """
        requested = str(
            data.get("name") or data.get("variable_id") or data.get("id") or ""
        )
        entries_by_name: dict[str, list[VariableEntry]] = (
            project._variable_entries_by_name
        )
        table_id = data.get("table_id")

        if requested in entries_by_name:
            entries = entries_by_name[requested]
            if table_id:
                matches = [e for e in entries if e.table_id == str(table_id)]
                if len(matches) == 1:
                    return matches[0]
                raise TableValidationError(
                    f"Variable {requested!r} was not found in table {table_id!r}."
                )
            if len(entries) == 1:
                return entries[0]
            choices = ", ".join(f"{e.table_id}:{e.name}" for e in entries)
            raise TableValidationError(
                f"Variable {requested!r} is ambiguous across loaded tables; "
                f"specify table_id.  Choices: {choices}."
            )

        # Fall back to out_name matching
        matches = [
            e
            for e in project.variable_entries.values()
            if str(e.entry.get("out_name", e.name)) == requested
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            names = ", ".join(m.name for m in matches[:10])
            raise TableValidationError(
                f"Variable {requested!r} is ambiguous; use one of: {names}."
            )
        raise TableValidationError(
            f"Variable {requested!r} was not found in loaded variable tables."
        )

    @classmethod
    def _merge_entry(cls, data: dict[str, Any], entry: VariableEntry) -> dict[str, Any]:
        """Merge *entry* defaults into *data* (modifies and returns *data*)."""
        e = entry.entry
        data.setdefault("name", entry.name)
        data.setdefault("id", e.get("out_name", entry.name.split("_", 1)[0]))
        data.setdefault("variable_id", data["id"])
        data.setdefault("dimensions", table_dimensions(e))
        data.setdefault("table_id", e.get("table_id", entry.table_id))
        if entry.table_file is not None:
            data.setdefault("table_info", f"Name: {entry.table_file.name};")
        if "frequency" in e:
            data.setdefault("frequency", e["frequency"])
        if "modeling_realm" in e:
            data.setdefault("realm", single_or_original(e["modeling_realm"]))
        for key in ("valid_min", "valid_max", "ok_min_mean_abs", "ok_max_mean_abs"):
            value = e.get(key)
            if not is_table_value(value) and entry.table_header:
                value = entry.table_header.get(key)
            if is_table_value(value):
                data.setdefault(key, parse_table_value(value))
        for key in (
            "units",
            "standard_name",
            "long_name",
            "cell_methods",
            "cell_measures",
            "comment",
            "positive",
            "flag_values",
            "flag_meanings",
        ):
            if e.get(key) not in (None, ""):
                data[key] = e[key]
        return data

    # ------------------------------------------------------------------
    # Public instance API (existing interface preserved)
    # ------------------------------------------------------------------

    def resolve_table_entry(self, project: ProjectTables) -> VariableEntry:
        """Find the variable table entry for this Variable.

        Raises
        ------
        TableValidationError
            If the variable is not found or ambiguous.
        """
        return self._resolve_entry(self.to_dict(), project)

    def names(self) -> tuple[str, dict[str, str]]:
        """Return ``(variable_id, labels_dict)``.

        ``labels_dict`` always contains ``"branded_name"`` and
        ``"variable_id"``; optional keys are ``"branding_suffix"``,
        ``"temporal_label"``, ``"vertical_label"``, ``"horizontal_label"``,
        and ``"area_label"``.
        """
        branded = str(self.get("name") or self.get("id") or self.get("variable_id"))
        var_id = str(
            self.get("id") or self.get("variable_id") or branded.split("_", 1)[0]
        )
        labels: dict[str, str] = {"branded_name": branded, "variable_id": var_id}
        if "_" in branded:
            suffix = branded.split("_", 1)[1]
            labels["branding_suffix"] = suffix
            for key, val in zip(
                ("temporal_label", "vertical_label", "horizontal_label", "area_label"),
                suffix.split("-"),
            ):
                labels[key] = val
        return var_id, labels

    def attributes(self, labels: Mapping[str, str]) -> dict[str, Any]:
        """Return NetCDF attributes for this data variable."""
        attrs = self.netcdf_attrs(self.attrs)
        for key in (
            "units",
            "standard_name",
            "long_name",
            "cell_methods",
            "cell_measures",
            "comment",
            "positive",
            "flag_values",
            "flag_meanings",
        ):
            if key in self:
                attrs[key] = self[key]
        attrs.setdefault("branded_variable_name", labels["branded_name"])
        for key in (
            "branding_suffix",
            "temporal_label",
            "vertical_label",
            "horizontal_label",
            "area_label",
        ):
            if key in labels:
                attrs.setdefault(key, labels[key])
        return attrs

    def validate_against_entry(self, entry: VariableEntry) -> None:
        """Validate this variable's metadata against a table entry.

        Raises
        ------
        TableValidationError
            On any metadata mismatch.
        """
        e = entry.entry
        values = self.to_dict()
        out_name = str(e.get("out_name", entry.name.split("_", 1)[0]))
        for key in ("id", "variable_id"):
            if key in values and str(values[key]) != out_name:
                raise TableValidationError(
                    f"{key}={values[key]!r} does not match table out_name {out_name!r}."
                )
        expected_dims = table_dimensions(e)
        if self.dimensions is not None and tuple(self.dimensions) != expected_dims:
            raise TableValidationError(
                f"dimensions={tuple(self.dimensions)!r} does not match "
                f"{entry.table_id}:{entry.name} dimensions {expected_dims!r}."
            )
        # Units
        table_units = e.get("units")
        user_units = values.get("units")
        if (
            is_table_value(table_units)
            and str(table_units) != "?"
            and user_units not in (None, "")
            and str(user_units) != str(table_units)
            and not _units_convertible(str(user_units), str(table_units))
        ):
            raise TableValidationError(
                f"units={user_units!r} does not match {entry.table_id}:{entry.name} "
                f"value {table_units!r} and the two are not dimensionally convertible."
            )
        for key in (
            "standard_name",
            "long_name",
            "cell_methods",
            "cell_measures",
            "comment",
        ):
            expected = e.get(key)
            if (
                expected not in (None, "")
                and key in values
                and str(values[key]) != str(expected)
            ):
                raise TableValidationError(
                    f"{key}={values[key]!r} does not match "
                    f"{entry.table_id}:{entry.name} value {expected!r}."
                )
        required = set(str(e.get("required", "")).split())
        table_pos = e.get("positive")
        user_pos = values.get("positive")
        if user_pos not in (None, ""):
            if str(user_pos).lower() not in {"up", "down"}:
                raise TableValidationError(
                    f"positive={user_pos!r} is not valid; allowed values are 'up' and 'down'."
                )
            if (
                is_table_value(table_pos)
                and str(user_pos).lower() != str(table_pos).lower()
            ):
                raise TableValidationError(
                    f"positive={user_pos!r} does not match "
                    f"{entry.table_id}:{entry.name} value {table_pos!r}."
                )
        if (
            "positive" in required
            and is_table_value(table_pos)
            and user_pos in (None, "")
        ):
            raise TableValidationError(
                f"variable {entry.table_id}:{entry.name} requires 'positive' "
                f"(expected {table_pos!r})."
            )
        for attr in required - {"positive"}:
            tval = e.get(attr)
            if is_table_value(tval) and values.get(attr) in (None, ""):
                raise TableValidationError(
                    f"variable {entry.table_id}:{entry.name} requires attribute "
                    f"{attr!r} (expected {tval!r})."
                )
        tfv = e.get("flag_values")
        tfm = e.get("flag_meanings")
        hfv, hfm = is_table_value(tfv), is_table_value(tfm)
        if hfv != hfm:
            missing = "flag_meanings" if hfv else "flag_values"
            present = "flag_values" if hfv else "flag_meanings"
            raise TableValidationError(
                f"{entry.table_id}:{entry.name} has {present!r} but missing {missing!r}."
            )
        if hfv and hfm:
            nv = len(str(tfv).split())
            nm = len(str(tfm).split())
            if nv != nm:
                raise TableValidationError(
                    f"{entry.table_id}:{entry.name} flag_values has {nv} token(s) "
                    f"but flag_meanings has {nm} token(s)."
                )
        for key, expected in {
            "frequency": e.get("frequency"),
            "realm": e.get("modeling_realm"),
            "table_id": entry.table_id,
        }.items():
            if (
                expected not in (None, "")
                and key in values
                and not metadata_value_matches(values[key], expected)
            ):
                raise TableValidationError(
                    f"{key}={values[key]!r} does not match "
                    f"{entry.table_id}:{entry.name} value {expected!r}."
                )
