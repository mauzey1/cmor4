"""Indexed table objects and entry types for CMOR4 metadata.

Entry classes — lightweight resolved-entry containers, one per table type:

* :class:`AxisEntry` — one coordinate or grid-coordinate entry
* :class:`ZFactorEntry` — one formula-term entry
* :class:`GridMappingEntry` — one grid-mapping entry
* :class:`VariableEntry` — one variable entry (may span multiple table files)

Table classes — own typed entries, resolution logic, and construction logic:

* :class:`CoordinateTable` — coordinate and grid-coordinate axes
* :class:`FormulaTable` — hybrid-coordinate formula terms
* :class:`GridTable` — grid-mapping projections
* :class:`VariableTable` — data variables (one or more realm files)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constraints import has_value as is_table_value
from .constraints import value_matches_constraint
from .dataset_metadata import DatasetMetadata
from .table_models import (
    CoordinateTableDocument,
    CoordinateTableEntry as AxisEntry,
    FormulaTableDocument,
    FormulaTableEntry as ZFactorEntry,
    GridMappingTableEntry as GridMappingEntry,
    GridTableDocument,
    VariableTableDocument,
    VariableTableEntry as VariableEntry,
)
from .unit_conversion import units_are_convertible as _units_convertible
from ..exceptions import TableValidationError
from ..axis import Axis
from ..grid import Grid
from ..variable import Variable
from ..zfactor import ZFactor

# ---------------------------------------------------------------------------
# CoordinateTable
# ---------------------------------------------------------------------------


class CoordinateTable:
    """Indexed coordinate and grid-coordinate entries.

    Parameters
    ----------
    coord_entries:
        Typed entries from the coordinate table JSON (``axis_entry`` section).
    grid_axis_entries:
        Typed axis entries from the grids table JSON (``axis_entry`` section).
        These overlay *coord_entries*: grid-specific names take precedence.
    grid_coord_entries:
        Typed coordinate entries from the grids table JSON
        (``variable_entry`` section) — auxiliary lat/lon variables.
    """

    def __init__(
        self,
        coord_entries: Mapping[str, Any],
        grid_axis_entries: Mapping[str, Any],
        grid_coord_entries: Mapping[str, Any],
    ) -> None:
        coord_entries = CoordinateTableDocument.model_validate(
            {"axis_entry": dict(coord_entries)}
        ).axis_entry
        grid_document = GridTableDocument.model_validate({
            "axis_entry": dict(grid_axis_entries),
            "variable_entry": dict(grid_coord_entries),
        })
        grid_axis_entries = grid_document.axis_entry
        grid_coord_entries = {
            name: entry.model_copy(update={"is_grid_coord": True})
            for name, entry in grid_document.variable_entry.items()
        }
        self._coord = coord_entries
        self._grid_coord = grid_coord_entries
        self._all_coord: dict[str, AxisEntry] = {
            **coord_entries,
            **grid_axis_entries,
        }
        self.scalar_entries: dict[str, AxisEntry] = {
            name: entry
            for name, entry in self._all_coord.items()
            if entry.is_scalar
        }
        self.generic_level_entries: dict[str, dict[str, AxisEntry]] = (
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
        coord_entries: dict[str, AxisEntry] = {}
        grid_axis_entries: dict[str, AxisEntry] = {}
        grid_coord_entries: dict[str, AxisEntry] = {}

        if coordinate_table is not None:
            document = CoordinateTableDocument.model_validate_json(
                coordinate_table.read_text()
            )
            coord_entries = document.axis_entry
        if grid_table is not None:
            document = GridTableDocument.model_validate_json(grid_table.read_text())
            grid_axis_entries = document.axis_entry
            grid_coord_entries = {
                name: entry.model_copy(update={"is_grid_coord": True})
                for name, entry in document.variable_entry.items()
            }

        return cls(coord_entries, grid_axis_entries, grid_coord_entries)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_coord(
        self, request: Axis | Mapping[str, Any]
    ) -> AxisEntry | None:
        """Return the best-matching coordinate :class:`AxisEntry`, or ``None``.

        Tries, in order: direct name, generic-level (raises if ambiguous),
        ``out_name``, and ``out_name`` + ``standard_name`` attribute match.
        """
        data = request.to_dict() if isinstance(request, Axis) else dict(request)
        requested = str(
            data.get("table_entry")
            or data.get("axis_entry")
            or data.get("coordinate")
            or data.get("name")
            or ""
        )
        entry = self._all_coord.get(requested)
        if entry is not None:
            return entry

        generic = self._generic_level_matches(data, requested)
        if len(generic) == 1:
            _, entry = generic[0]
            return entry
        if len(generic) > 1:
            choices = ", ".join(n for n, _ in generic)
            raise TableValidationError(
                f"Generic level {requested!r} matches multiple coordinate entries; "
                f"specify table_entry or axis_entry.  Choices: {choices}."
            )

        by_out = [
            (n, e)
            for n, e in self._all_coord.items()
            if str(e.out_name or "") == requested
        ]
        if len(by_out) == 1:
            _, entry = by_out[0]
            return entry

        matches = self._match_by_attrs(data)
        if len(matches) == 1:
            _, entry = matches[0]
            return entry

        return None

    def resolve_grid_coord(
        self, request: Axis | Mapping[str, Any]
    ) -> AxisEntry | None:
        """Return the best-matching grid-coordinate :class:`AxisEntry`, or ``None``."""
        data = request.to_dict() if isinstance(request, Axis) else dict(request)
        requested = str(
            data.get("grid_table_entry")
            or data.get("grid_coordinate")
            or data.get("out_name")
            or data.get("name")
            or ""
        )
        entry = self._grid_coord.get(requested)
        if entry is not None:
            return entry

        m = [
            (n, e)
            for n, e in self._grid_coord.items()
            if str(e.out_name or "") == requested
        ]
        if len(m) == 1:
            _, entry = m[0]
            return entry

        return None

    def get_grid_coord_entry(self, name: str) -> AxisEntry | None:
        """Return the typed grid-coordinate entry for *name*, or ``None``."""
        return self._grid_coord.get(name)

    # ------------------------------------------------------------------
    # Build (resolve + merge)
    # ------------------------------------------------------------------

    def build(self, request: Axis) -> Axis:
        """Return an axis constructed from a typed request and table metadata."""

        data = self._build_data(_model_data(request))
        return Axis.model_validate(data)

    def _build_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Merge coordinate-table defaults into serialized request data.

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
        entry = axis_entry
        data.setdefault("table_entry", entry_name)
        entry.validate_metadata(
            data,
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
            val = getattr(entry, key)
            if is_table_value(val):
                data.setdefault(key, val)
        data.setdefault("out_name", entry_name)
        if "values" not in data:
            v = entry.runtime_values
            if v is not None:
                data["values"] = v
        if "bounds" not in data:
            b = entry.runtime_bounds
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
        entry = axis_entry
        data.setdefault("grid_table_entry", entry_name)
        for key in (
            "out_name",
            "units",
            "standard_name",
            "long_name",
            "valid_min",
            "valid_max",
        ):
            val = getattr(entry, key)
            if is_table_value(val):
                data.setdefault(key, val)
        data.setdefault("out_name", entry_name)
        bname = data.get("bounds_name")
        if bname:
            be = self.get_grid_coord_entry(str(bname))
            if be:
                ba = dict(data.get("bounds_attrs") or {})
                for key in ("units", "standard_name", "long_name"):
                    val = getattr(be, key)
                    if is_table_value(val):
                        ba.setdefault(key, val)
                if ba:
                    data["bounds_attrs"] = ba

    def _generic_level_matches(
        self, data: dict[str, Any], generic_name: str
    ) -> list[tuple[str, AxisEntry]]:
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
                if is_table_value(getattr(e, key))
                and value_matches_constraint(val, getattr(e, key))
            ]
            if narrowed:
                matches = narrowed
        return matches

    def _match_by_attrs(
        self, data: dict[str, Any]
    ) -> list[tuple[str, AxisEntry]]:
        out_name = data.get("out_name")
        std_name = data.get("standard_name")
        if not out_name and not std_name:
            return []
        matches = list(self._all_coord.items())
        for key, val in (("out_name", out_name), ("standard_name", std_name)):
            if val in (None, ""):
                continue
            narrowed = [
                (n, e) for n, e in matches if str(getattr(e, key, "")) == str(val)
            ]
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
        Typed entries from the formula-terms table JSON
        (``formula_entry`` section).
    """

    def __init__(self, entries: dict[str, ZFactorEntry]) -> None:
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
        entries: dict[str, ZFactorEntry] = {}
        if formula_table is not None:
            document = FormulaTableDocument.model_validate_json(
                formula_table.read_text()
            )
            entries = document.formula_entry
        return cls(entries)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self, request: ZFactor | Mapping[str, Any]
    ) -> ZFactorEntry | None:
        """Return the matching :class:`ZFactorEntry`, or ``None``.

        Tries direct name match first, then ``out_name`` match.
        """
        data = request.to_dict() if isinstance(request, ZFactor) else dict(request)
        requested = str(
            data.get("table_entry")
            or data.get("formula_entry")
            or data.get("name")
            or ""
        )
        entry = self._entries.get(requested)
        if entry is not None:
            return entry

        m = [
            (n, e)
            for n, e in self._entries.items()
            if str(e.out_name or "") == requested
        ]
        if len(m) == 1:
            _, entry = m[0]
            return entry

        return None

    def get_entry(self, name: str) -> ZFactorEntry | None:
        """Return the typed formula entry for *name*, or ``None``."""
        return self._entries.get(name)

    # ------------------------------------------------------------------
    # Build (resolve + merge)
    # ------------------------------------------------------------------

    def build(self, request: ZFactor) -> ZFactor:
        """Return a z-factor constructed from a typed request and table metadata."""

        data = self._build_data(_model_data(request))
        return ZFactor.model_validate(data)

    def _build_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Merge formula-table defaults into serialized request data."""
        zf_entry = self.resolve(data)
        if zf_entry is None:
            return data
        entry_name = zf_entry.name
        entry = zf_entry
        data.setdefault("table_entry", entry_name)
        entry.validate_metadata(
            data,
            ("units", "standard_name", "long_name"),
            "formula term",
        )
        for key in ("out_name", "units", "standard_name", "long_name"):
            val = getattr(entry, key)
            if is_table_value(val):
                data.setdefault(key, val)
        for key in ("valid_min", "valid_max", "ok_min_mean_abs", "ok_max_mean_abs"):
            val = getattr(entry, key)
            if is_table_value(val):
                data.setdefault(key, val)
        if "dimensions" not in data and entry.dimensions:
            data["dimensions"] = entry.runtime_dimensions
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
                    val = getattr(be, key)
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
        Typed axis entries from the grids table JSON (``axis_entry`` section).
        Passed to :class:`CoordinateTable` for overlay.
    coord_entries:
        Typed coordinate entries from the grids table JSON
        (``variable_entry`` section) — auxiliary lat/lon variables.
    mapping_entries:
        Typed mapping entries from the grids table JSON
        (``mapping_entry`` section) — CF grid-mapping projections.
    """

    def __init__(
        self,
        axis_entries: dict[str, AxisEntry],
        coord_entries: dict[str, AxisEntry],
        mapping_entries: dict[str, GridMappingEntry],
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
        axis_entries: dict[str, AxisEntry] = {}
        coord_entries: dict[str, AxisEntry] = {}
        mapping_entries: dict[str, GridMappingEntry] = {}

        if grid_table is not None:
            document = GridTableDocument.model_validate_json(grid_table.read_text())
            axis_entries = document.axis_entry
            coord_entries = {
                name: entry.model_copy(update={"is_grid_coord": True})
                for name, entry in document.variable_entry.items()
            }
            mapping_entries = document.mapping_entry

        return cls(axis_entries, coord_entries, mapping_entries)

    @property
    def axis_entries(self) -> dict[str, AxisEntry]:
        """Typed grid-axis entries."""
        return self._axis

    @property
    def coord_entries(self) -> dict[str, AxisEntry]:
        """Typed grid-coordinate entries."""
        return self._coord

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_mapping(self, name: str) -> GridMappingEntry | None:
        """Return the :class:`GridMappingEntry` for *name*, or ``None``."""
        entry = self._raw_mapping.get(name)
        if entry is not None:
            return entry
        return None

    # ------------------------------------------------------------------
    # Build (resolve + merge)
    # ------------------------------------------------------------------

    def build(self, request: Grid) -> Grid:
        """Return a grid constructed from a typed request and table metadata."""

        data = self._build_data(_model_data(request))
        return Grid.model_validate(data)

    def _build_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Merge grid-table defaults into serialized request data."""
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
        entry = gm_entry
        for key in ("mapping_name", "grid_mapping_name", "mapping_var"):
            val = getattr(entry, key)
            if is_table_value(val):
                data.setdefault(key, val)
        if entry.coordinates:
            data.setdefault("coordinates", list(entry.coordinates))
        table_params = entry.params
        if isinstance(table_params, dict):
            mp = dict(data.get("params") or {})
            for k, v in table_params.items():
                mp.setdefault(k, v)
            data["params"] = mp
        params = dict(data.get("params") or {})
        for key, param_name in (entry.model_extra or {}).items():
            if not key.startswith("parameter") or not is_table_value(param_name):
                continue
            params.setdefault(str(param_name), data.get(str(param_name), 0.0))
        if params:
            data["params"] = params
        return data


# ---------------------------------------------------------------------------
# VariableRemappingTable
# ---------------------------------------------------------------------------


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

        Raises :exc:`~cmor4.exceptions.TableValidationError` if the
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
            e
            for e in self.entries.values()
            if str(e.out_name or e.name) == requested
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

        data = self._merge(_model_data(request), self.resolve(request))
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

        updates: dict[str, str] = {}
        for attr, remap_table in self._contextual_remaps.items():
            found, value = remap_table.lookup(keys)
            if found:
                updates[attr] = str(value)

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
        string remap values are converted to ``None`` so the resulting NetCDF
        variable omits the attribute while validation still knows the effective
        table expected an explicit empty value.
        """

        if not entry.contextual_attrs:
            return variable

        updates: dict[str, str | None] = {}
        for attr in entry.contextual_attrs:
            value = getattr(entry, attr)
            updates[attr] = None if value == "" else str(value)
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
            if not is_table_value(value) and entry.table_header:
                value = getattr(entry.table_header, key)
            if is_table_value(value):
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
            if value not in (None, ""):
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
        Called by :meth:`~cmor4.tables.ProjectTables.validate_dataset`. For
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
            if expected in (None, ""):
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
            tval = getattr(entry, attr, None)
            if is_table_value(tval) and vdict.get(attr) in (None, ""):
                raise TableValidationError(
                    f"variable {entry.table_id}:{entry.name} requires attribute "
                    f"{attr!r} (expected {tval!r})."
                )
        tfv = entry.flag_values
        tfm = entry.flag_meanings
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
            "frequency": entry.frequency,
            "realm": entry.modeling_realm,
            "table_id": entry.table_id,
        }.items():
            user_val = vdict.get(key)
            if (
                expected not in (None, "")
                and user_val is not None
                and not value_matches_constraint(user_val, expected)
            ):
                raise TableValidationError(
                    f"{key}={user_val!r} does not match "
                    f"{entry.table_id}:{entry.name} value {expected!r}."
                )


# ---------------------------------------------------------------------------
# Module-level private helpers
# ---------------------------------------------------------------------------


def _model_data(model: Axis | Grid | Variable | ZFactor) -> dict[str, Any]:
    """Serialize a construction request without flattening explicit extras."""

    return model.model_dump(exclude_none=True)


def _build_generic_level_index(
    coordinate_entries: dict[str, AxisEntry],
) -> dict[str, dict[str, AxisEntry]]:
    """Build a two-level index: ``generic_level_name → {entry_name → entry}``."""
    index: dict[str, dict[str, AxisEntry]] = {}
    for name, entry in coordinate_entries.items():
        generic = entry.generic_level_name
        if is_table_value(generic):
            index.setdefault(str(generic), {})[name] = entry
    return index


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
