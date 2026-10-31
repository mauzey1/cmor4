"""Variable-table schemas, readers, and contextual metadata remapping.

Keep source-row validation, table provenance, resolution, and remapping in one
module so changes to a variable table's interpretation have a single owner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

from pydantic import Field

from ._metadata import DatasetMetadata
from ._table_common import DimensionedTableEntry, TableHeader, _NamedDocument
from ._table_common import units_are_convertible as _units_convertible
from .exceptions import TableValidationError
from .variable import Variable


class VariableTableRow(DimensionedTableEntry):
    """One variable-table JSON row before source provenance is attached."""

    frequency: Any = None
    modeling_realm: Any = None
    cell_methods: str | None = None
    cell_measures: str | None = None
    comment: str | None = None
    positive: str | None = None
    flag_values: str | None = None
    flag_meanings: str | None = None
    required: str | None = None
    valid_min: Any = None
    valid_max: Any = None
    ok_min_mean_abs: Any = None
    ok_max_mean_abs: Any = None


class VariableTableEntry(VariableTableRow):
    """A validated variable row plus its source-table provenance."""

    table_id: str = Field(min_length=1)
    table_file: Path | None = None
    table_header: TableHeader | None = None
    contextual_attrs: tuple[str, ...] = ()


class VariableTableDocument(_NamedDocument):
    """Typed variable-table file; provenance is attached after validation."""

    sections: ClassVar[tuple[str, ...]] = ("variable_entry",)

    header: TableHeader | None = Field(default=None, alias="Header")
    variable_entry: dict[str, VariableTableRow] = Field(default_factory=dict)

    def resolved_entries(self, table_file: Path) -> dict[str, VariableTableEntry]:
        header = self.header or TableHeader()
        table_id = str(header.table_id or table_file.stem)
        if table_id.startswith("Table "):
            table_id = table_id.removeprefix("Table ")
        return {
            name: VariableTableEntry.model_validate({
                **row.model_dump(),
                "table_id": table_id,
                "table_file": table_file,
                "table_header": header,
            })
            for name, row in self.variable_entry.items()
        }


VariableEntry = VariableTableEntry


class VariableRemappingTable:
    """Indexed contextual variable metadata entries.

    Keys follow the CMIP7 remapping format:
    ``{realm}.{variable_id}.{branding_suffix}.{frequency}.{region}``.
    Values are strings and may be empty; an empty value is still an explicit
    table value and must be distinguished from a missing key.
    """

    def __init__(self, section: str, values: dict[str, str]) -> None:
        self.section = section
        self._values = values

    @classmethod
    def from_file(
        cls, remapping_table: Path | None, section: str
    ) -> "VariableRemappingTable":
        values: dict[str, str] = {}
        if remapping_table is not None:
            with remapping_table.open() as handle:
                data = json.load(handle)
            raw_values = data.get(section)
            if not isinstance(raw_values, Mapping):
                raise TableValidationError(
                    f"Remapping table {str(remapping_table)!r} must contain a "
                    f"{section!r} mapping."
                )
            bad_values = [
                name for name, value in raw_values.items() if not isinstance(value, str)
            ]
            if bad_values:
                names = ", ".join(str(n) for n in bad_values[:5])
                raise TableValidationError(
                    f"Remapping table {str(remapping_table)!r} has non-string "
                    f"values for {section!r}: {names}."
                )
            values = {str(name): value for name, value in raw_values.items()}
        return cls(section, values)

    def lookup(self, keys: Sequence[str]) -> tuple[bool, str | None]:
        """Return ``(found, value)`` for the first matching composite key."""

        for key in keys:
            if key in self._values:
                return True, self._values[key]
        return False, None


class VariableTable:
    """Indexed variable entries across one or more variable table files.

    Each project typically supplies one file per modeling realm
    (e.g. ``CMIP7_atmos.json``, ``CMIP7_ocean.json``).  All files are
    loaded at construction time and merged into a single index.

    Parameters
    ----------
    table_files:
        Paths to variable table JSON files to load.
    """

    def __init__(
        self,
        table_files: Sequence[Path],
        *,
        long_name_override_table: Path | None = None,
        cell_measures_table: Path | None = None,
    ) -> None:
        """Load variable tables and optional contextual remapping tables.

        Parameters
        ----------
        table_files:
            Variable table JSON files to index.
        long_name_override_table:
            Optional table containing ``long_name_overrides`` entries keyed by
            realm, variable id, branding suffix, frequency, and region.
        cell_measures_table:
            Optional table containing ``cell_measures`` entries keyed by the
            same composite context.
        """

        self.entries: dict[str, VariableEntry] = {}
        self._by_name: dict[str, list[VariableEntry]] = {}
        for path in table_files:
            self._load(path)
        self._contextual_remaps = {
            "long_name": VariableRemappingTable.from_file(
                long_name_override_table, "long_name_overrides"
            ),
            "cell_measures": VariableRemappingTable.from_file(
                cell_measures_table, "cell_measures"
            ),
        }

    @classmethod
    def from_file(
        cls,
        table_files: Sequence[Path],
        *,
        long_name_override_table: Path | None = None,
        cell_measures_table: Path | None = None,
    ) -> "VariableTable":
        """Construct a VariableTable from table file paths.

        Parameters
        ----------
        table_files:
            Paths to variable table JSON files to load.
        long_name_override_table:
            Optional contextual long-name override table.
        cell_measures_table:
            Optional contextual cell-measures table.

        Returns
        -------
        VariableTable
            Loaded variable table instance with remapping indexes attached.
        """
        return cls(
            table_files,
            long_name_override_table=long_name_override_table,
            cell_measures_table=cell_measures_table,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, table_file: Path) -> None:
        """Load one variable table JSON file and index its entries."""
        document = VariableTableDocument.model_validate_json(table_file.read_text())
        for name, variable_entry in document.resolved_entries(table_file).items():
            # Full key is unique; short name may appear in multiple tables.
            self.entries.setdefault(name, variable_entry)
            self._by_name.setdefault(name, []).append(variable_entry)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, request: Variable | Mapping[str, Any]) -> VariableEntry:
        """Return the :class:`VariableEntry` matching *data*.

        Raises :exc:`~cmor.exceptions.TableValidationError` if the
        variable is not found or is ambiguous across tables.
        """
        data = request.to_dict() if isinstance(request, Variable) else dict(request)
        requested = str(
            data.get("name") or data.get("variable_id") or data.get("id") or ""
        )
        table_id = data.get("table_id")

        if requested in self._by_name:
            entries = self._by_name[requested]
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
            e for e in self.entries.values() if str(e.out_name or e.name) == requested
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

    # ------------------------------------------------------------------
    # Build (resolve + merge)
    # ------------------------------------------------------------------

    def build(self, request: Variable) -> Variable:
        """Return a variable constructed from a typed request and table metadata."""

        data = self._merge(request.model_dump(exclude_none=True), self.resolve(request))
        return Variable.model_validate(data)

    def contextual_entry(
        self,
        entry: VariableEntry,
        variable: Variable,
        dataset: DatasetMetadata | Mapping[str, Any] | None,
    ) -> VariableEntry:
        """Return *entry* with context-specific metadata overlaid.

        The returned entry is a copy of the resolved variable-table row when a
        remapping applies, with ``long_name`` and/or ``cell_measures`` replaced
        by the contextual table value. If no remapping key matches, the original
        entry is returned unchanged.

        Context keys are derived from the resolved table entry, the prepared
        variable, and dataset metadata. The table id is considered before the
        variable's possibly multi-token ``modeling_realm`` so CMIP7 realm tables
        such as ``aerosol`` can match entries whose modeling realm is
        ``"aerosol atmosChem"``.
        """

        keys = self._remapping_keys(entry, variable, dataset)
        if not keys:
            return entry

        updates: dict[str, str | None] = {}
        for attr, remap_table in self._contextual_remaps.items():
            found, value = remap_table.lookup(keys)
            if found:
                updates[attr] = None if value == "" else str(value)

        if not updates:
            return entry

        return entry.model_copy(
            update={**updates, "contextual_attrs": tuple(sorted(updates))}
        )

    def apply_contextual_metadata(
        self, variable: Variable, entry: VariableEntry
    ) -> Variable:
        """Overlay contextual entry metadata onto *variable*.

        Only attributes listed in ``entry.contextual_attrs`` are copied. Empty
        Empty remap values are already normalized to ``None``, so the resulting
        NetCDF variable omits the attribute while ``contextual_attrs`` still
        records that the table explicitly cleared it.
        """

        if not entry.contextual_attrs:
            return variable

        updates: dict[str, str | None] = {}
        for attr in entry.contextual_attrs:
            value = getattr(entry, attr)
            updates[attr] = None if value is None else str(value)
        return variable.model_copy(update=updates)

    def _remapping_keys(
        self,
        entry: VariableEntry,
        variable: Variable,
        dataset: DatasetMetadata | Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        """Return candidate remapping keys ordered from most to least specific.

        Candidate realms include the variable table id, dataset realm, table
        ``modeling_realm`` tokens, and variable realm. Candidate frequencies
        prefer dataset metadata, then variable metadata, then table metadata.
        Region is dataset-specific; without it no remapping key is produced.
        """

        data = (
            dataset.to_dict()
            if isinstance(dataset, DatasetMetadata)
            else dict(dataset or {})
        )
        variable_id = str(
            entry.out_name
            or variable.id
            or variable.variable_id
            or entry.name.split("_", 1)[0]
        )
        branding_suffix = entry.name.split("_", 1)[1] if "_" in entry.name else ""

        realms = _unique_strings([
            entry.table_id,
            *_split_context_values(data.get("realm")),
            *_split_context_values(entry.modeling_realm),
            *_split_context_values(variable.realm),
        ])
        frequencies = _unique_strings([
            *_split_context_values(data.get("frequency")),
            *_split_context_values(variable.frequency),
            *_split_context_values(entry.frequency),
        ])
        regions = _unique_strings(_split_context_values(data.get("region")))

        if not all((realms, variable_id, frequencies, regions)):
            return ()

        return tuple(
            f"{realm}.{variable_id}.{branding_suffix}.{frequency}.{region}"
            for realm in realms
            for frequency in frequencies
            for region in regions
        )

    def _merge(self, data: dict[str, Any], entry: VariableEntry) -> dict[str, Any]:
        """Copy *entry* defaults into *data*."""
        data.setdefault("name", entry.name)
        data.setdefault("id", entry.out_name or entry.name.split("_", 1)[0])
        data.setdefault("variable_id", data["id"])
        data.setdefault("dimensions", entry.runtime_dimensions)
        data.setdefault("table_id", entry.table_id)
        if entry.table_file is not None:
            data.setdefault("table_info", f"Name: {entry.table_file.name};")
        if entry.frequency is not None:
            data.setdefault("frequency", entry.frequency)
        if entry.modeling_realm is not None:
            realm = entry.modeling_realm
            data.setdefault(
                "realm",
                realm[0] if isinstance(realm, list) and len(realm) == 1 else realm,
            )
        for key in ("valid_min", "valid_max", "ok_min_mean_abs", "ok_max_mean_abs"):
            value = getattr(entry, key)
            if value is None and entry.table_header:
                value = getattr(entry.table_header, key)
            if value is not None:
                data.setdefault(key, value)
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
            value = getattr(entry, key)
            if value is not None:
                data[key] = value
        return data

    def validate_against(self, variable: Variable, entry: VariableEntry) -> None:
        """Validate *variable* metadata against a resolved/effective table entry.

        Parameters
        ----------
        variable
            Variable to validate.
        entry
            Resolved variable table entry. This may be a contextual entry with
            remapped ``long_name`` or ``cell_measures`` values already overlaid.
        Raises
        ------
        TableValidationError
            On any mismatch between variable and table entry.

        Notes
        -----
        Called by :meth:`~cmor.tables.ProjectTables.validate_dataset`. For
        contextual empty-string remaps, the accepted variable value is ``None``
        or ``""``; non-empty values are rejected.
        """
        out_name = str(entry.out_name or entry.name.split("_", 1)[0])
        for attr, user_val in (
            ("id", variable.id),
            ("variable_id", variable.variable_id),
        ):
            if user_val is not None and str(user_val) != out_name:
                raise TableValidationError(
                    f"{attr}={user_val!r} does not match table out_name {out_name!r}."
                )
        expected_dims = entry.runtime_dimensions
        if (
            variable.dimensions is not None
            and tuple(variable.dimensions) != expected_dims
        ):
            raise TableValidationError(
                f"dimensions={tuple(variable.dimensions)!r} does not match "
                f"{entry.table_id}:{entry.name} dimensions {expected_dims!r}."
            )
        table_units = entry.units
        user_units = variable.units
        if (
            table_units is not None
            and str(table_units) != "?"
            and user_units not in (None, "")
            and str(user_units) != str(table_units)
            and not _units_convertible(str(user_units), str(table_units))
        ):
            raise TableValidationError(
                f"units={user_units!r} does not match {entry.table_id}:{entry.name} "
                f"value {table_units!r} and the two are not dimensionally convertible."
            )
        contextual_attrs = set(entry.contextual_attrs)
        for key in (
            "standard_name",
            "long_name",
            "cell_methods",
            "cell_measures",
            "comment",
        ):
            expected = getattr(entry, key)
            user_val = getattr(variable, key, None)
            if expected is None:
                if key in contextual_attrs and user_val not in (None, ""):
                    raise TableValidationError(
                        f"{key}={user_val!r} does not match "
                        f"{entry.table_id}:{entry.name} contextual value {expected!r}."
                    )
                continue
            if user_val is not None and str(user_val) != str(expected):
                raise TableValidationError(
                    f"{key}={user_val!r} does not match "
                    f"{entry.table_id}:{entry.name} value {expected!r}."
                )
        required = set(str(entry.required or "").split())
        table_pos = entry.positive
        user_pos = variable.positive
        if (
            user_pos not in (None, "")
            and table_pos is not None
            and str(user_pos).lower() != str(table_pos).lower()
        ):
            raise TableValidationError(
                f"positive={user_pos!r} does not match "
                f"{entry.table_id}:{entry.name} value {table_pos!r}."
            )
        if "positive" in required and table_pos is not None and user_pos in (None, ""):
            raise TableValidationError(
                f"variable {entry.table_id}:{entry.name} requires 'positive' "
                f"(expected {table_pos!r})."
            )
        vdict = variable.to_dict()
        for attr in required - {"positive"}:
            tval = getattr(entry, attr, None)
            if tval is not None and vdict.get(attr) in (None, ""):
                raise TableValidationError(
                    f"variable {entry.table_id}:{entry.name} requires attribute "
                    f"{attr!r} (expected {tval!r})."
                )
        tfv = entry.flag_values
        tfm = entry.flag_meanings
        hfv = tfv is not None
        hfm = tfm is not None
        if hfv != hfm:
            missing = "flag_meanings" if hfv else "flag_values"
            present = "flag_values" if hfv else "flag_meanings"
            raise TableValidationError(
                f"{entry.table_id}:{entry.name} has "
                f"{present!r} but missing {missing!r}."
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
            "frequency": entry.frequency,
            "realm": entry.modeling_realm,
            "table_id": entry.table_id,
        }.items():
            user_val = vdict.get(key)
            if isinstance(expected, list):
                matches = str(user_val) in {str(item) for item in expected}
            else:
                matches = str(user_val) == str(expected)
            if expected is not None and user_val is not None and not matches:
                raise TableValidationError(
                    f"{key}={user_val!r} does not match "
                    f"{entry.table_id}:{entry.name} value {expected!r}."
                )


def _split_context_values(value: Any) -> list[str]:
    """Return remapping context tokens from strings or simple sequences.

    CMIP table realms are often encoded as space-separated strings, while some
    dataset metadata can arrive as one-item lists. This helper normalizes those
    forms for composite-key generation.
    """

    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part for part in value.split() if part]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result: list[str] = []
        for item in value:
            result.extend(_split_context_values(item))
        return result
    return [str(value)]


def _unique_strings(values: Sequence[str]) -> list[str]:
    """Return non-empty strings with first-seen ordering preserved.

    Ordering controls remapping precedence, so this intentionally keeps the
    first occurrence of each token rather than sorting.
    """

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
