"""Indexed table objects and entry types for CMOR4 metadata.

Entry classes — lightweight resolved-entry containers, one per table type:

* :class:`AxisEntry` — one coordinate or grid-coordinate entry
* :class:`ZFactorEntry` — one formula-term entry
* :class:`GridMappingEntry` — one grid-mapping entry
* :class:`VariableEntry` — one variable entry (may span multiple table files)

Table classes — own raw entries, resolution logic, and merge logic:

* :class:`CoordinateTable` — coordinate and grid-coordinate axes
* :class:`FormulaTable` — hybrid-coordinate formula terms
* :class:`GridTable` — grid-mapping projections
* :class:`VariableTable` — data variables (one or more realm files)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from ._table_utils import (
    entry_bounds,
    entry_values,
    is_table_value,
    metadata_value_matches,
    parse_table_value,
    single_or_original,
    table_dimensions,
    validate_table_metadata,
)
from ._unit_conversion import units_are_convertible as _units_convertible
from .exceptions import TableValidationError
from .variable import Variable

# ---------------------------------------------------------------------------
# Entry classes
# ---------------------------------------------------------------------------


class AxisEntry(BaseModel):
    """One resolved entry from a coordinate or grid-coordinate table.

    Parameters
    ----------
    name:
        Entry name in the table.  Must be a non-empty string.
    entry:
        Raw entry metadata dict from the JSON table.
    is_grid_coord:
        ``True`` when the entry came from the grid-coordinate table rather
        than the main coordinate table.  The two differ in which fields are
        authoritative and how bounds attributes are looked up.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    entry: dict[str, Any]
    is_grid_coord: bool = False

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("AxisEntry name must not be empty")
        return v

    @field_validator("entry", mode="before")
    @classmethod
    def _coerce_entry(cls, v: Any) -> dict[str, Any]:
        return dict(v) if not isinstance(v, dict) else v


class ZFactorEntry(BaseModel):
    """One resolved entry from a formula-terms table.

    Parameters
    ----------
    name:
        Entry name in the table.  Must be a non-empty string.
    entry:
        Raw entry metadata dict from the JSON table.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    entry: dict[str, Any]

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("ZFactorEntry name must not be empty")
        return v

    @field_validator("entry", mode="before")
    @classmethod
    def _coerce_entry(cls, v: Any) -> dict[str, Any]:
        return dict(v) if not isinstance(v, dict) else v


class GridMappingEntry(BaseModel):
    """One resolved entry from a grid-mapping table.

    Parameters
    ----------
    name:
        Entry name in the table (e.g. ``"lambert_azimuthal_equal_area"``).
        Must be a non-empty string.
    entry:
        Raw entry metadata dict from the JSON table.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    entry: dict[str, Any]

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("GridMappingEntry name must not be empty")
        return v

    @field_validator("entry", mode="before")
    @classmethod
    def _coerce_entry(cls, v: Any) -> dict[str, Any]:
        return dict(v) if not isinstance(v, dict) else v


class VariableEntry(BaseModel):
    """One resolved entry from a variable table.

    Parameters
    ----------
    name:
        Variable entry name in the table.  Must be a non-empty string.
    table_id:
        Identifier of the table supplying the entry.  Must be non-empty.
    entry:
        Raw variable-entry metadata dict from the JSON table.
    table_file:
        Path to the source table file, if available.
    table_header:
        Header metadata from the table file, if available.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    table_id: str
    entry: dict[str, Any]
    table_file: Path | None = None
    table_header: dict[str, Any] | None = None

    @field_validator("name", "table_id")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("VariableEntry name and table_id must not be empty")
        return v

    @field_validator("entry", mode="before")
    @classmethod
    def _coerce_entry(cls, v: Any) -> dict[str, Any]:
        return dict(v) if not isinstance(v, dict) else v

    @field_validator("table_header", mode="before")
    @classmethod
    def _coerce_header(cls, v: Any) -> dict[str, Any] | None:
        if v is None:
            return None
        return dict(v) if not isinstance(v, dict) else v


# ---------------------------------------------------------------------------
# CoordinateTable
# ---------------------------------------------------------------------------


class CoordinateTable:
    """Indexed coordinate and grid-coordinate entries.

    Parameters
    ----------
    coord_entries:
        Raw entries from the coordinate table JSON (``axis_entry`` section).
    grid_axis_entries:
        Raw axis entries from the grids table JSON (``axis_entry`` section).
        These overlay *coord_entries*: grid-specific names take precedence.
    grid_coord_entries:
        Raw coordinate entries from the grids table JSON
        (``variable_entry`` section) — auxiliary lat/lon variables.
    """

    def __init__(
        self,
        coord_entries: dict[str, Mapping[str, Any]],
        grid_axis_entries: dict[str, Mapping[str, Any]],
        grid_coord_entries: dict[str, Mapping[str, Any]],
    ) -> None:
        self._coord = coord_entries
        self._grid_coord = grid_coord_entries
        self._all_coord: dict[str, Mapping[str, Any]] = {
            **coord_entries,
            **grid_axis_entries,
        }
        self.scalar_entries: dict[str, Mapping[str, Any]] = {
            name: entry
            for name, entry in self._all_coord.items()
            if is_table_value(entry.get("value"))
        }
        self.generic_level_entries: dict[str, dict[str, Mapping[str, Any]]] = (
            _build_generic_level_index(self._all_coord)
        )

    @classmethod
    def from_file(
        cls,
        coordinate_table: Path | None = None,
        grid_table: Path | None = None,
    ) -> "CoordinateTable":
        """Construct a CoordinateTable from table file paths.

        Parameters
        ----------
        coordinate_table:
            Optional path to the coordinate table JSON file.
        grid_table:
            Optional path to the grids table JSON file.

        Returns
        -------
        CoordinateTable
            Loaded coordinate table instance.
        """
        coord_entries: dict[str, Mapping[str, Any]] = {}
        grid_axis_entries: dict[str, Mapping[str, Any]] = {}
        grid_coord_entries: dict[str, Mapping[str, Any]] = {}

        if coordinate_table is not None:
            coord_entries = _read_table_entries(coordinate_table, "axis_entry")
        if grid_table is not None:
            grid_axis_entries = _read_table_entries(grid_table, "axis_entry")
            grid_coord_entries = _read_table_entries(grid_table, "variable_entry")

        return cls(coord_entries, grid_axis_entries, grid_coord_entries)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_coord(self, data: dict[str, Any]) -> AxisEntry | None:
        """Return the best-matching coordinate :class:`AxisEntry`, or ``None``.

        Tries, in order: direct name, generic-level (raises if ambiguous),
        ``out_name``, and ``out_name`` + ``standard_name`` attribute match.
        """
        requested = str(
            data.get("table_entry")
            or data.get("axis_entry")
            or data.get("coordinate")
            or data.get("name")
            or ""
        )
        entry = self._all_coord.get(requested)
        if entry is not None:
            return AxisEntry(name=requested, entry=entry)

        generic = self._generic_level_matches(data, requested)
        if len(generic) == 1:
            name, entry = generic[0]
            return AxisEntry(name=name, entry=entry)
        if len(generic) > 1:
            choices = ", ".join(n for n, _ in generic)
            raise TableValidationError(
                f"Generic level {requested!r} matches multiple coordinate entries; "
                f"specify table_entry or axis_entry.  Choices: {choices}."
            )

        by_out = [
            (n, e)
            for n, e in self._all_coord.items()
            if str(e.get("out_name", "")) == requested
        ]
        if len(by_out) == 1:
            name, entry = by_out[0]
            return AxisEntry(name=name, entry=entry)

        matches = self._match_by_attrs(data)
        if len(matches) == 1:
            name, entry = matches[0]
            return AxisEntry(name=name, entry=entry)

        return None

    def resolve_grid_coord(self, data: dict[str, Any]) -> AxisEntry | None:
        """Return the best-matching grid-coordinate :class:`AxisEntry`, or ``None``."""
        requested = str(
            data.get("grid_table_entry")
            or data.get("grid_coordinate")
            or data.get("out_name")
            or data.get("name")
            or ""
        )
        entry = self._grid_coord.get(requested)
        if entry is not None:
            return AxisEntry(name=requested, entry=entry, is_grid_coord=True)

        m = [
            (n, e)
            for n, e in self._grid_coord.items()
            if str(e.get("out_name", "")) == requested
        ]
        if len(m) == 1:
            name, entry = m[0]
            return AxisEntry(name=name, entry=entry, is_grid_coord=True)

        return None

    def get_grid_coord_entry(self, name: str) -> Mapping[str, Any] | None:
        """Return the raw grid-coordinate entry dict for *name*, or ``None``."""
        return self._grid_coord.get(name)

    # ------------------------------------------------------------------
    # Build (resolve + merge)
    # ------------------------------------------------------------------

    def build(self, data: dict[str, Any]) -> dict[str, Any]:
        """Merge coordinate-table defaults into *data* and return it.

        Handles the grid-coordinate path (``grid_coordinate=`` /
        ``grid_table_entry=``) and the regular coordinate path, including
        a second pass to overlay any matching grid-coordinate entry after
        the main coordinate merge.
        """
        if data.get("grid_coordinate") or data.get("grid_table_entry"):
            axis_entry = self.resolve_grid_coord(data)
            if axis_entry is not None:
                self._merge_grid_coord_fields(data, axis_entry)
            return data

        axis_entry = self.resolve_coord(data)
        if axis_entry is None:
            return data

        entry_name = axis_entry.name
        entry = axis_entry.entry
        data.setdefault("table_entry", entry_name)
        validate_table_metadata(
            data,
            entry_name,
            entry,
            ("units", "standard_name", "long_name", "axis", "positive", "formula"),
            "axis",
        )
        for key in (
            "out_name",
            "units",
            "standard_name",
            "long_name",
            "axis",
            "positive",
            "formula",
            "climatology",
            "generic_level_name",
            "z_factors",
            "z_bounds_factors",
            "valid_min",
            "valid_max",
            "requested",
            "requested_bounds",
            "bounds_values",
            "must_have_bounds",
            "stored_direction",
            "tolerance",
        ):
            val = entry.get(key)
            if is_table_value(val):
                data.setdefault(key, parse_table_value(val))
        data.setdefault("out_name", entry_name)
        if "values" not in data:
            v = entry_values(entry)
            if v is not None:
                data["values"] = v
        if "bounds" not in data:
            b = entry_bounds(entry)
            if b is not None:
                data["bounds"] = b
        # Overlay any matching grid-coordinate entry
        grid_axis_entry = self.resolve_grid_coord(data)
        if grid_axis_entry is not None:
            self._merge_grid_coord_fields(data, grid_axis_entry)
        return data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _merge_grid_coord_fields(
        self, data: dict[str, Any], axis_entry: AxisEntry
    ) -> None:
        """Merge grid-coordinate AxisEntry fields into *data*."""
        entry_name = axis_entry.name
        entry = axis_entry.entry
        data.setdefault("grid_table_entry", entry_name)
        for key in (
            "out_name",
            "units",
            "standard_name",
            "long_name",
            "valid_min",
            "valid_max",
        ):
            val = entry.get(key)
            if is_table_value(val):
                data.setdefault(key, parse_table_value(val))
        data.setdefault("out_name", entry_name)
        bname = data.get("bounds_name")
        if bname:
            be = self.get_grid_coord_entry(str(bname))
            if be:
                ba = dict(data.get("bounds_attrs") or {})
                for key in ("units", "standard_name", "long_name"):
                    val = be.get(key)
                    if is_table_value(val):
                        ba.setdefault(key, parse_table_value(val))
                if ba:
                    data["bounds_attrs"] = ba

    def _generic_level_matches(
        self, data: dict[str, Any], generic_name: str
    ) -> list[tuple[str, Mapping[str, Any]]]:
        matches = list(self.generic_level_entries.get(generic_name, {}).items())
        if not matches:
            return []
        for key in (
            "standard_name",
            "formula",
            "z_factors",
            "z_bounds_factors",
            "positive",
            "units",
            "long_name",
        ):
            val = data.get(key)
            if val in (None, ""):
                continue
            narrowed = [
                (n, e)
                for n, e in matches
                if is_table_value(e.get(key)) and metadata_value_matches(val, e[key])
            ]
            if narrowed:
                matches = narrowed
        return matches

    def _match_by_attrs(
        self, data: dict[str, Any]
    ) -> list[tuple[str, Mapping[str, Any]]]:
        out_name = data.get("out_name")
        std_name = data.get("standard_name")
        if not out_name and not std_name:
            return []
        matches = list(self._all_coord.items())
        for key, val in (("out_name", out_name), ("standard_name", std_name)):
            if val in (None, ""):
                continue
            narrowed = [(n, e) for n, e in matches if str(e.get(key, "")) == str(val)]
            if narrowed:
                matches = narrowed
        return matches if len(matches) == 1 else []


# ---------------------------------------------------------------------------
# FormulaTable
# ---------------------------------------------------------------------------


class FormulaTable:
    """Indexed formula-term entries.

    Parameters
    ----------
    entries:
        Raw entries from the formula-terms table JSON
        (``formula_entry`` section).
    """

    def __init__(self, entries: dict[str, Mapping[str, Any]]) -> None:
        self._entries = entries

    @classmethod
    def from_file(cls, formula_table: Path | None = None) -> "FormulaTable":
        """Construct a FormulaTable from a table file path.

        Parameters
        ----------
        formula_table:
            Optional path to the formula-terms table JSON file.

        Returns
        -------
        FormulaTable
            Loaded formula table instance.
        """
        entries: dict[str, Mapping[str, Any]] = {}
        if formula_table is not None:
            entries = _read_table_entries(formula_table, "formula_entry")
        return cls(entries)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, data: dict[str, Any]) -> ZFactorEntry | None:
        """Return the matching :class:`ZFactorEntry`, or ``None``.

        Tries direct name match first, then ``out_name`` match.
        """
        requested = str(
            data.get("table_entry")
            or data.get("formula_entry")
            or data.get("name")
            or ""
        )
        entry = self._entries.get(requested)
        if entry is not None:
            return ZFactorEntry(name=requested, entry=entry)

        m = [
            (n, e)
            for n, e in self._entries.items()
            if str(e.get("out_name", "")) == requested
        ]
        if len(m) == 1:
            name, entry = m[0]
            return ZFactorEntry(name=name, entry=entry)

        return None

    def get_entry(self, name: str) -> Mapping[str, Any] | None:
        """Return the raw entry dict for *name*, or ``None``."""
        return self._entries.get(name)

    # ------------------------------------------------------------------
    # Build (resolve + merge)
    # ------------------------------------------------------------------

    def build(self, data: dict[str, Any]) -> dict[str, Any]:
        """Merge formula-table defaults into *data* and return it."""
        zf_entry = self.resolve(data)
        if zf_entry is None:
            return data
        entry_name = zf_entry.name
        entry = zf_entry.entry
        data.setdefault("table_entry", entry_name)
        validate_table_metadata(
            data,
            entry_name,
            entry,
            ("units", "standard_name", "long_name"),
            "formula term",
        )
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
            bname = str(
                data.get("bounds_name")
                or f"{data.get('out_name', data.get('name', ''))}_bnds"
            )
            be = self.get_entry(bname)
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


# ---------------------------------------------------------------------------
# GridTable
# ---------------------------------------------------------------------------


class GridTable:
    """Indexed grid-axis, grid-coordinate, and grid-mapping entries.

    Parameters
    ----------
    axis_entries:
        Raw axis entries from the grids table JSON (``axis_entry`` section).
        Passed to :class:`CoordinateTable` for overlay.
    coord_entries:
        Raw coordinate entries from the grids table JSON
        (``variable_entry`` section) — auxiliary lat/lon variables.
    mapping_entries:
        Raw mapping entries from the grids table JSON
        (``mapping_entry`` section) — CF grid-mapping projections.
    """

    def __init__(
        self,
        axis_entries: dict[str, Mapping[str, Any]],
        coord_entries: dict[str, Mapping[str, Any]],
        mapping_entries: dict[str, Mapping[str, Any]],
    ) -> None:
        self._axis = axis_entries
        self._coord = coord_entries
        self._raw_mapping = mapping_entries

    @classmethod
    def from_file(cls, grid_table: Path | None = None) -> "GridTable":
        """Construct a GridTable from a table file path.

        Parameters
        ----------
        grid_table:
            Optional path to the grids table JSON file.

        Returns
        -------
        GridTable
            Loaded grid table instance.
        """
        axis_entries: dict[str, Mapping[str, Any]] = {}
        coord_entries: dict[str, Mapping[str, Any]] = {}
        mapping_entries: dict[str, Mapping[str, Any]] = {}

        if grid_table is not None:
            axis_entries = _read_table_entries(grid_table, "axis_entry")
            coord_entries = _read_table_entries(grid_table, "variable_entry")
            mapping_entries = _read_table_entries(grid_table, "mapping_entry")

        return cls(axis_entries, coord_entries, mapping_entries)

    @property
    def axis_entries(self) -> dict[str, Mapping[str, Any]]:
        """Raw axis entries (passed to :class:`CoordinateTable` for overlay)."""
        return self._axis

    @property
    def coord_entries(self) -> dict[str, Mapping[str, Any]]:
        """Raw grid-coordinate entries."""
        return self._coord

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_mapping(self, name: str) -> GridMappingEntry | None:
        """Return the :class:`GridMappingEntry` for *name*, or ``None``."""
        entry = self._raw_mapping.get(name)
        if entry is not None:
            return GridMappingEntry(name=name, entry=entry)
        return None

    # ------------------------------------------------------------------
    # Build (resolve + merge)
    # ------------------------------------------------------------------

    def build(self, data: dict[str, Any]) -> dict[str, Any]:
        """Merge grid-mapping-table defaults into *data* and return it."""
        requested = str(
            data.get("table_entry")
            or data.get("mapping_entry")
            or data.get("name")
            or ""
        )
        if not requested:
            return data
        gm_entry = self.resolve_mapping(requested)
        if gm_entry is None:
            return data
        entry = gm_entry.entry
        for key in ("mapping_name", "grid_mapping_name", "mapping_var"):
            val = entry.get(key)
            if is_table_value(val):
                data.setdefault(key, val)
        coords = entry.get("coordinates")
        if is_table_value(coords):
            if isinstance(coords, str):
                coords = coords.split()
            data.setdefault("coordinates", coords)
        table_params = entry.get("params") or {}
        if isinstance(table_params, dict):
            mp = dict(data.get("params") or {})
            for k, v in table_params.items():
                mp.setdefault(k, v)
            data["params"] = mp
        params = dict(data.get("params") or {})
        for key, param_name in entry.items():
            if not key.startswith("parameter") or not is_table_value(param_name):
                continue
            params.setdefault(str(param_name), data.get(str(param_name), 0.0))
        if params:
            data["params"] = params
        return data


# ---------------------------------------------------------------------------
# VariableTable
# ---------------------------------------------------------------------------


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

    def __init__(self, table_files: Sequence[Path]) -> None:
        self.entries: dict[str, VariableEntry] = {}
        self._by_name: dict[str, list[VariableEntry]] = {}
        for path in table_files:
            self._load(path)

    @classmethod
    def from_file(cls, table_files: Sequence[Path]) -> "VariableTable":
        """Construct a VariableTable from table file paths.

        Parameters
        ----------
        table_files:
            Paths to variable table JSON files to load.

        Returns
        -------
        VariableTable
            Loaded variable table instance.
        """
        return cls(table_files)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, table_file: Path) -> None:
        """Load one variable table JSON file and index its entries."""
        with table_file.open() as handle:
            data = json.load(handle)
        header = data.get("Header", {})
        table_id = str(header.get("table_id") or table_file.stem)
        if table_id.startswith("Table "):
            table_id = table_id.removeprefix("Table ")
        for name, entry in data.get("variable_entry", {}).items():
            if not isinstance(entry, Mapping):
                continue
            variable_entry = VariableEntry(
                name=name,
                table_id=str(table_id),
                entry=entry,
                table_file=table_file,
                table_header=header,
            )
            # Full key is unique; short name may appear in multiple tables.
            self.entries.setdefault(name, variable_entry)
            self._by_name.setdefault(name, []).append(variable_entry)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, data: dict[str, Any]) -> VariableEntry:
        """Return the :class:`VariableEntry` matching *data*.

        Raises :exc:`~cmor4.exceptions.TableValidationError` if the
        variable is not found or is ambiguous across tables.
        """
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
            e
            for e in self.entries.values()
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

    # ------------------------------------------------------------------
    # Build (resolve + merge)
    # ------------------------------------------------------------------

    def build(self, data: dict[str, Any]) -> dict[str, Any]:
        """Resolve and merge variable-table defaults into *data* and return it."""
        entry = self.resolve(data)
        return self._merge(data, entry)

    def _merge(self, data: dict[str, Any], entry: VariableEntry) -> dict[str, Any]:
        """Copy *entry* defaults into *data*."""
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

    def validate_against(self, variable: Variable, entry: VariableEntry) -> None:
        """Validate *variable* metadata against a resolved table entry.

        Raises :exc:`~cmor4.exceptions.TableValidationError` on any mismatch.
        Called by :meth:`~cmor4.tables.ProjectTables.validate_dataset`.
        """
        e = entry.entry
        out_name = str(e.get("out_name", entry.name.split("_", 1)[0]))
        for attr, user_val in (
            ("id", variable.id),
            ("variable_id", variable.variable_id),
        ):
            if user_val is not None and str(user_val) != out_name:
                raise TableValidationError(
                    f"{attr}={user_val!r} does not match table out_name {out_name!r}."
                )
        expected_dims = table_dimensions(e)
        if (
            variable.dimensions is not None
            and tuple(variable.dimensions) != expected_dims
        ):
            raise TableValidationError(
                f"dimensions={tuple(variable.dimensions)!r} does not match "
                f"{entry.table_id}:{entry.name} dimensions {expected_dims!r}."
            )
        table_units = e.get("units")
        user_units = variable.units
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
            user_val = getattr(variable, key, None)
            if (
                expected not in (None, "")
                and user_val is not None
                and str(user_val) != str(expected)
            ):
                raise TableValidationError(
                    f"{key}={user_val!r} does not match "
                    f"{entry.table_id}:{entry.name} value {expected!r}."
                )
        required = set(str(e.get("required", "")).split())
        table_pos = e.get("positive")
        user_pos = variable.positive
        if (
            user_pos not in (None, "")
            and is_table_value(table_pos)
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
        vdict = variable.to_dict()
        for attr in required - {"positive"}:
            tval = e.get(attr)
            if is_table_value(tval) and vdict.get(attr) in (None, ""):
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
            "frequency": e.get("frequency"),
            "realm": e.get("modeling_realm"),
            "table_id": entry.table_id,
        }.items():
            user_val = vdict.get(key)
            if (
                expected not in (None, "")
                and user_val is not None
                and not metadata_value_matches(user_val, expected)
            ):
                raise TableValidationError(
                    f"{key}={user_val!r} does not match "
                    f"{entry.table_id}:{entry.name} value {expected!r}."
                )


# ---------------------------------------------------------------------------
# Module-level private helpers
# ---------------------------------------------------------------------------


def _build_generic_level_index(
    coordinate_entries: dict[str, Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Build a two-level index: ``generic_level_name → {entry_name → entry}``."""
    index: dict[str, dict[str, Mapping[str, Any]]] = {}
    for name, entry in coordinate_entries.items():
        generic = entry.get("generic_level_name")
        if is_table_value(generic):
            index.setdefault(str(generic), {})[name] = entry
    return index


def _read_table_entries(table_file: Path, key: str) -> dict[str, Mapping[str, Any]]:
    """Read entries from a table file for a given key."""
    with table_file.open() as handle:
        data = json.load(handle)
    return {
        str(name): entry
        for name, entry in data.get(key, {}).items()
        if isinstance(entry, Mapping)
    }
