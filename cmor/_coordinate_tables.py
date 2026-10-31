"""Coordinate, formula-term, and grid schemas together with their readers.

Table rows describe authoritative defaults and constraints; the domain models
in :mod:`cmor.axis`, :mod:`cmor.grid`, and :mod:`cmor.zfactor` hold runtime data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from ._table_common import (
    DimensionedTableEntry,
    NamedTableEntry,
    TableHeader,
    _NamedDocument,
)
from ._table_common import units_are_convertible as _units_convertible
from .axis import Axis
from .exceptions import TableValidationError
from .grid import Grid, _validate_grid_dimensions
from .zfactor import ZFactor


def _numeric_table_value(value: Any) -> Any:
    """Normalize a numeric scalar encoded as a JSON string."""

    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


class CoordinateTableEntry(DimensionedTableEntry):
    """One coordinate-table or grid-coordinate row."""

    axis: str | None = None
    positive: str | None = None
    formula: str | None = None
    climatology: Any = None
    generic_level_name: str | None = None
    z_factors: str | None = None
    z_bounds_factors: str | None = None
    valid_min: Any = None
    valid_max: Any = None
    requested: Any = None
    requested_bounds: Any = None
    bounds_values: Any = None
    value: Any = None
    must_have_bounds: Any = None
    stored_direction: str | None = None
    tolerance: Any = None
    is_grid_coord: bool = False

    @property
    def is_scalar(self) -> bool:
        return self.value is not None

    @property
    def runtime_values(self) -> list[Any] | None:
        value = self.requested if self.requested is not None else self.value
        if value is None:
            return None
        values = value if isinstance(value, (list, tuple)) else [value]
        return [_numeric_table_value(item) for item in values]

    @property
    def runtime_bounds(self) -> list[list[Any]] | None:
        value = self.requested_bounds
        if value is None:
            value = self.bounds_values
        if value is None:
            return None
        values = value if isinstance(value, (list, tuple)) else str(value).split()
        parsed = [_numeric_table_value(item) for item in values]
        if len(parsed) % 2:
            return None
        return [parsed[index : index + 2] for index in range(0, len(parsed), 2)]


class FormulaTableEntry(DimensionedTableEntry):
    """One formula-term table row."""

    valid_min: Any = None
    valid_max: Any = None
    ok_min_mean_abs: Any = None
    ok_max_mean_abs: Any = None


class GridMappingTableEntry(NamedTableEntry):
    """One grid-mapping row, including extension-defined projection fields."""

    mapping_name: str | None = None
    grid_mapping_name: str | None = None
    mapping_var: str | None = None
    coordinates: tuple[str, ...] = ()
    params: dict[str, Any] = Field(default_factory=dict)
    required_params: Any = None
    required_parameters: Any = None
    required: Any = None
    optional_params: Any = None
    optional_parameters: Any = None
    parameters: Any = None
    text_params: Any = None
    text_parameters: Any = None
    required_axes: Any = None
    required_axis: Any = None
    axes: Any = None
    axis: Any = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_coordinates(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        value = data.get("coordinates")
        if isinstance(value, str):
            data["coordinates"] = tuple(value.split())
        elif value is None:
            data["coordinates"] = ()
        return data

    @staticmethod
    def _tokens(*values: Any) -> tuple[str, ...]:
        result: list[str] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                candidates: Iterable[Any] = re.split(r"[\s,]+", value)
            elif isinstance(value, dict):
                candidates = value.keys()
            elif isinstance(value, Iterable):
                candidates = value
            else:
                candidates = (value,)
            for candidate in candidates:
                token = str(candidate)
                if token and token not in result:
                    result.append(token)
        return tuple(result)

    def numbered_parameters(self) -> tuple[str, ...]:
        numbered: list[tuple[int, str]] = []
        for key, value in (self.model_extra or {}).items():
            match = re.fullmatch(r"parameter(\d+)", key)
            if match and value is not None:
                numbered.append((int(match.group(1)), str(value)))
        return tuple(value for _, value in sorted(numbered))

    def required_parameter_names(self) -> tuple[str, ...]:
        return self._tokens(
            self.required_params,
            self.required_parameters,
            self.required,
            self.numbered_parameters(),
        )

    def optional_parameter_names(self) -> tuple[str, ...]:
        return self._tokens(
            self.optional_params,
            self.optional_parameters,
            self.parameters,
            self.params,
        )

    def text_parameter_names(self) -> tuple[str, ...]:
        declared = self._tokens(self.text_params, self.text_parameters)
        known = tuple(
            name
            for name in (
                *self.required_parameter_names(),
                *self.optional_parameter_names(),
            )
            if name in {"crs_wkt", "GeoTransform", "spatial_ref"}
        )
        return self._tokens(declared, known)

    def required_axis_names(self) -> tuple[str, ...]:
        return self._tokens(
            self.required_axes, self.required_axis, self.axes, self.axis
        )


class CoordinateTableDocument(_NamedDocument):
    sections: ClassVar[tuple[str, ...]] = ("axis_entry",)
    header: TableHeader | None = Field(default=None, alias="Header")
    axis_entry: dict[str, CoordinateTableEntry] = Field(default_factory=dict)


class FormulaTableDocument(_NamedDocument):
    sections: ClassVar[tuple[str, ...]] = ("formula_entry",)
    header: TableHeader | None = Field(default=None, alias="Header")
    formula_entry: dict[str, FormulaTableEntry] = Field(default_factory=dict)


class GridTableDocument(_NamedDocument):
    sections: ClassVar[tuple[str, ...]] = (
        "axis_entry",
        "variable_entry",
        "mapping_entry",
    )
    header: TableHeader | None = Field(default=None, alias="Header")
    axis_entry: dict[str, CoordinateTableEntry] = Field(default_factory=dict)
    variable_entry: dict[str, CoordinateTableEntry] = Field(default_factory=dict)
    mapping_entry: dict[str, GridMappingTableEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _mark_grid_coordinates(self) -> GridTableDocument:
        # Copy rows and their mapping; caller-owned entries remain untouched.
        object.__setattr__(
            self,
            "variable_entry",
            {
                name: entry.model_copy(update={"is_grid_coord": True})
                for name, entry in self.variable_entry.items()
            },
        )
        return self


# Reader terminology for the specialized immutable row models.
AxisEntry = CoordinateTableEntry
ZFactorEntry = FormulaTableEntry
GridMappingEntry = GridMappingTableEntry


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
        coord_entries = CoordinateTableDocument.model_validate({
            "axis_entry": dict(coord_entries)
        }).axis_entry
        grid_document = GridTableDocument.model_validate({
            "axis_entry": dict(grid_axis_entries),
            "variable_entry": dict(grid_coord_entries),
        })
        grid_axis_entries = grid_document.axis_entry
        grid_coord_entries = grid_document.variable_entry
        self._coord = coord_entries
        self._grid_coord = grid_coord_entries
        self._all_coord: dict[str, AxisEntry] = {
            **coord_entries,
            **grid_axis_entries,
        }
        self.scalar_entries: dict[str, AxisEntry] = {
            name: entry for name, entry in self._all_coord.items() if entry.is_scalar
        }
        self.generic_level_entries: dict[str, dict[str, AxisEntry]] = (
            _build_generic_level_index(self._all_coord)
        )

    @classmethod
    def from_file(
        cls,
        coordinate_table: Path | None = None,
        grid_table: Path | GridTable | None = None,
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
            coordinate_document = CoordinateTableDocument.model_validate_json(
                coordinate_table.read_text()
            )
            coord_entries = coordinate_document.axis_entry
        if grid_table is not None:
            grid = (
                grid_table
                if isinstance(grid_table, GridTable)
                else GridTable.from_file(grid_table)
            )
            grid_axis_entries = grid.axis_entries
            grid_coord_entries = grid.coord_entries

        return cls(coord_entries, grid_axis_entries, grid_coord_entries)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_coord(self, request: Axis | Mapping[str, Any]) -> AxisEntry | None:
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

    def resolve_grid_coord(self, request: Axis | Mapping[str, Any]) -> AxisEntry | None:
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

        data = self._build_data(request.model_dump(exclude_none=True))
        return Axis.model_validate(data)

    def validate_metadata(self, axis: Axis) -> None:
        # Check if this is a grid coordinate (auxiliary lat/lon)
        adict = axis.to_dict()
        if ae := self.resolve_grid_coord(axis):
            # Grid coordinates validated against grid coordinate table
            ae.validate_metadata(
                adict,
                ("units", "standard_name", "long_name"),
                "grid coordinate",
            )
        elif ae := self.resolve_coord(axis):
            # Regular coordinates validated against coordinate table
            ae.validate_metadata(
                adict,
                (
                    "units",
                    "standard_name",
                    "long_name",
                    "axis",
                    "positive",
                    "formula",
                ),
                "axis",
            )

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
            if val is not None:
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
            if val is not None:
                data.setdefault(key, val)
        data.setdefault("out_name", entry_name)
        bname = data.get("bounds_name")
        if bname:
            be = self.get_grid_coord_entry(str(bname))
            if be:
                ba = dict(data.get("bounds_attrs") or {})
                for key in ("units", "standard_name", "long_name"):
                    val = getattr(be, key)
                    if val is not None:
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
                if getattr(e, key) is not None
                and (
                    " since " in str(val)
                    if str(getattr(e, key)).endswith(" since ?")
                    else re.fullmatch(
                        re.escape(str(getattr(e, key))).replace(r"\?", ".+"),
                        str(val),
                    )
                    is not None
                )
            ]
            if narrowed:
                matches = narrowed
        return matches

    def _match_by_attrs(self, data: dict[str, Any]) -> list[tuple[str, AxisEntry]]:
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

    def resolve(self, request: ZFactor | Mapping[str, Any]) -> ZFactorEntry | None:
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

        data = self._build_data(request.model_dump(exclude_none=True))
        return ZFactor.model_validate(data)

    def validate_metadata(self, zfactor: ZFactor) -> None:
        formula_entry = self.resolve(zfactor)
        formula_entry_name = formula_entry.name if formula_entry else None
        if formula_entry is None:
            return

        # Units must be dimensionally convertible, not just equal.
        # Use the same cf_units-based check as Variable units validation.
        table_units = formula_entry.units
        user_units = zfactor.units
        if (
            table_units is not None
            and str(table_units) != "?"
            and user_units not in (None, "")
            and str(user_units) != str(table_units)
            and not _units_convertible(str(user_units), str(table_units))
        ):
            raise TableValidationError(
                f"formula term {formula_entry_name!r} units={user_units!r} does "
                f"not match table value {table_units!r} and the two are "
                f"not dimensionally convertible."
            )

        # Validate remaining metadata (standard_name, long_name)
        # by exact match.
        formula_entry.validate_metadata(
            zfactor.to_dict(),
            ("standard_name", "long_name"),
            "formula term",
        )

        # When the formula-term table entry has no declared
        # dimensions the term is expected to be a scalar.  Accept a
        # size-1 array (CMOR3-compatible) but reject larger arrays.
        if not formula_entry.dimensions and zfactor.values is not None:
            arr = np.asarray(zfactor.values)
            if arr.ndim > 0 and arr.size != 1:
                raise TableValidationError(
                    f"formula term {formula_entry_name!r} has no declared "
                    "dimensions (expected a scalar value) but values "
                    f"with shape {arr.shape} were provided."
                )

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
            if val is not None:
                data.setdefault(key, val)
        for key in ("valid_min", "valid_max", "ok_min_mean_abs", "ok_max_mean_abs"):
            val = getattr(entry, key)
            if val is not None:
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
                    if val is not None:
                        ba.setdefault(key, val)
                if ba:
                    data["bounds_attrs"] = ba
        return data


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
            coord_entries = document.variable_entry
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

        data = self._build_data(request.model_dump(exclude_none=True))
        return Grid.model_validate(data)

    def validate_metadata(self, grid: Grid, axes: Sequence[Axis]) -> None:
        requested = str(grid.table_entry or grid.mapping_entry or grid.name or "")
        _gm = self.resolve_mapping(requested) if requested else None
        entry_name, entry = (_gm.name, _gm) if _gm else (None, None)
        if entry is not None:
            for key, user_val in (
                ("mapping_name", grid.mapping_name),
                ("grid_mapping_name", grid.grid_mapping_name),
            ):
                expected = getattr(entry, key)
                if (
                    expected is not None
                    and user_val is not None
                    and str(user_val) != str(expected)
                ):
                    raise TableValidationError(
                        f"grid mapping {entry_name!r} {key}="
                        f"{user_val!r} does not match table value "
                        f"{expected!r}."
                    )

        # Validate dimensional axes and lat/lon shape consistency.
        grid_axes = getattr(grid, "axes", [])
        if grid_axes:
            _validate_grid_dimensions(grid, axes)
        elif grid.dimensions:
            _validate_grid_dimensions(grid, axes)

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
            if val is not None:
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
            if not key.startswith("parameter") or param_name is None:
                continue
            params.setdefault(str(param_name), data.get(str(param_name), 0.0))
        if params:
            data["params"] = params
        return data


def _build_generic_level_index(
    coordinate_entries: dict[str, AxisEntry],
) -> dict[str, dict[str, AxisEntry]]:
    """Build a two-level index: ``generic_level_name → {entry_name → entry}``."""
    index: dict[str, dict[str, AxisEntry]] = {}
    for name, entry in coordinate_entries.items():
        generic = entry.generic_level_name
        if generic is not None:
            index.setdefault(str(generic), {})[name] = entry
    return index
