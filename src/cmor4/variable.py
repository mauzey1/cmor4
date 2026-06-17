"""Variable metadata record."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import Field

from .metadata import CoercedF, IntTuple, MetadataModel, StrOrTuple, StrSeq


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

    Construct via :meth:`ProjectTables.variable` to merge authoritative
    table metadata (units, standard_name, dimensions, etc.)::

        variable = project.variable("tas")

    Construct directly when no project tables are needed::

        variable = Variable(name="tas", units="K")

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
    # NetCDF output helpers
    # ------------------------------------------------------------------

    def names(self) -> tuple[str, dict[str, str]]:
        """Return ``(variable_id, labels_dict)``.

        ``labels_dict`` always contains ``"branded_name"`` and
        ``"variable_id"``; optional keys are ``"branding_suffix"``,
        ``"temporal_label"``, ``"vertical_label"``, ``"horizontal_label"``,
        and ``"area_label"``.
        """
        branded = self.name
        var_id = str(self.id or self.variable_id or branded.split("_", 1)[0])
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
        for key, val in (
            ("units", self.units),
            ("standard_name", self.standard_name),
            ("long_name", self.long_name),
            ("cell_methods", self.cell_methods),
            ("cell_measures", self.cell_measures),
            ("comment", self.comment),
            ("positive", self.positive),
            ("flag_values", self.flag_values),
            ("flag_meanings", self.flag_meanings),
        ):
            if val is not None:
                attrs[key] = val
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
