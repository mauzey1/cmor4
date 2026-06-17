"""Table-resolution logic for Axis, Variable, ZFactor, and Grid.

All functions here work with :class:`~cmor4.tables.ProjectTables` to look
up table entries and merge authoritative metadata into the data-class
fields.  The data classes themselves (Axis, Variable, ZFactor, Grid) are
pure data holders with no knowledge of project tables.

Public builders — called by :class:`~cmor4.tables.ProjectTables` factory
methods::

    build_axis(project, "latitude", values=lat)
    build_variable(project, "tas")
    build_zfactor(project, "ps", values=ps)
    build_grid(project, "lambert_azimuthal_equal_area", params={...})

Public resolvers — used by :meth:`~cmor4.tables.ProjectTables.validate_components`
to look up table entries for already-constructed objects::

    axis_table_entry(project, axis)
    variable_table_entry(project, variable)
    ...

Public validators — run metadata consistency checks against table entries::

    validate_axis_metadata(axis, entity_type, entry_name, table_values, keys)
    validate_variable_against_entry(variable, entry)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

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
from .axis import Axis
from .exceptions import TableValidationError
from .grid import Grid
from .variable import Variable, VariableEntry
from .zfactor import ZFactor

if TYPE_CHECKING:
    from .tables import ProjectTables


# ---------------------------------------------------------------------------
# Axis — public API
# ---------------------------------------------------------------------------


def build_axis(project: ProjectTables, name: str, **kwargs: Any) -> Axis:
    """Create an Axis with project-table defaults merged and values validated."""
    from ._axis_validation import validate_axis_values_early

    data: dict[str, Any] = {"name": name, **kwargs}
    data = _apply_axis_defaults(data, project)
    axis = Axis.model_validate(data)
    validate_axis_values_early(axis)
    return axis


def merge_unprepared_axis(project: ProjectTables, axis: Axis) -> Axis:
    """Re-apply project-table defaults to an Axis not created via project.axis().

    Called by :meth:`~cmor4.tables.ProjectTables._axes` when a user
    constructs an ``Axis`` directly and passes it to ``cmorize()``.
    """
    data = _apply_axis_defaults(axis.to_dict(), project)
    return Axis.model_validate(data)


def axis_table_entry(
    project: ProjectTables, axis: Axis
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Resolve the coordinate table entry for *axis*."""
    return _resolve_coord_entry(axis.to_dict(), project)


def axis_grid_coordinate(
    project: ProjectTables, axis: Axis
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Resolve the grid coordinate table entry for *axis*."""
    return _resolve_grid_entry(axis.to_dict(), project)


def validate_axis_metadata(
    axis: Axis,
    entity_type: str,
    entry_name: str | None,
    table_values: Mapping[str, Any],
    keys: Sequence[str],
) -> None:
    """Raise TableValidationError if axis fields conflict with a table entry."""
    validate_table_metadata(axis.to_dict(), entry_name, table_values, keys, entity_type)


# ---------------------------------------------------------------------------
# Variable — public API
# ---------------------------------------------------------------------------


def build_variable(project: ProjectTables, name: str, **kwargs: Any) -> Variable:
    """Create a Variable with project-table defaults merged."""
    data: dict[str, Any] = {"name": name, **kwargs}
    data = _apply_variable_defaults(data, project)
    return Variable.model_validate(data)


def variable_table_entry(project: ProjectTables, variable: Variable) -> VariableEntry:
    """Look up the VariableEntry for *variable* in *project*.

    Raises :exc:`~cmor4.exceptions.TableValidationError` if not found or ambiguous.
    """
    return _resolve_variable_entry(variable.to_dict(), project)


def validate_variable_against_entry(variable: Variable, entry: VariableEntry) -> None:
    """Validate *variable* metadata against a table entry.

    Raises :exc:`~cmor4.exceptions.TableValidationError` on any mismatch.
    """
    e = entry.entry
    out_name = str(e.get("out_name", entry.name.split("_", 1)[0]))
    for attr, user_val in (("id", variable.id), ("variable_id", variable.variable_id)):
        if user_val is not None and str(user_val) != out_name:
            raise TableValidationError(
                f"{attr}={user_val!r} does not match table out_name {out_name!r}."
            )
    expected_dims = table_dimensions(e)
    if variable.dimensions is not None and tuple(variable.dimensions) != expected_dims:
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
    for key in ("standard_name", "long_name", "cell_methods", "cell_measures", "comment"):
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
    if user_pos not in (None, ""):
        if str(user_pos).lower() not in {"up", "down"}:
            raise TableValidationError(
                f"positive={user_pos!r} is not valid; "
                "allowed values are 'up' and 'down'."
            )
        if is_table_value(table_pos) and str(user_pos).lower() != str(table_pos).lower():
            raise TableValidationError(
                f"positive={user_pos!r} does not match "
                f"{entry.table_id}:{entry.name} value {table_pos!r}."
            )
    if "positive" in required and is_table_value(table_pos) and user_pos in (None, ""):
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
# ZFactor — public API
# ---------------------------------------------------------------------------


def build_zfactor(project: ProjectTables, name: str, **kwargs: Any) -> ZFactor:
    """Create a ZFactor with project-table defaults merged."""
    data: dict[str, Any] = {"name": name, **kwargs}
    data = _apply_zfactor_defaults(data, project)
    return ZFactor.model_validate(data)


def zfactor_table_entry(
    project: ProjectTables, zfactor: ZFactor
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Resolve the formula-term table entry for *zfactor*."""
    return _resolve_zfactor_entry(zfactor.to_dict(), project)


def validate_zfactor_metadata(
    zfactor: ZFactor,
    entity_type: str,
    entry_name: str | None,
    table_values: Mapping[str, Any],
    keys: Sequence[str],
) -> None:
    """Raise TableValidationError if zfactor fields conflict with a table entry."""
    validate_table_metadata(zfactor.to_dict(), entry_name, table_values, keys, entity_type)


# ---------------------------------------------------------------------------
# Grid — public API
# ---------------------------------------------------------------------------


def build_grid(
    project: ProjectTables, name: str | None = None, **kwargs: Any
) -> Grid:
    """Create a Grid with grid-table defaults merged."""
    data: dict[str, Any] = {
        k: v for k, v in {"name": name, **kwargs}.items() if v is not None
    }
    data = _apply_grid_defaults(data, project)
    return Grid.model_validate(data)


def grid_table_entry(
    project: ProjectTables, grid: Grid
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Resolve the grid mapping table entry for *grid*."""
    requested = str(grid.table_entry or grid.mapping_entry or grid.name or "")
    if not requested:
        return None, None
    ge: dict[str, Mapping[str, Any]] = getattr(project, "grid_mapping_entries", {})
    entry = ge.get(requested)
    if entry:
        return requested, entry
    return None, None


# ---------------------------------------------------------------------------
# Axis — private helpers
# ---------------------------------------------------------------------------


def _apply_axis_defaults(
    data: dict[str, Any], project: ProjectTables
) -> dict[str, Any]:
    """Resolve coordinate-table entry and merge its defaults into *data*."""
    if data.get("grid_coordinate") or data.get("grid_table_entry"):
        entry_name, entry = _resolve_grid_entry(data, project)
        if entry is not None:
            _merge_grid_entry(data, entry_name, entry, project)
        return data

    entry_name, entry = _resolve_coord_entry(data, project)
    if entry is None:
        return data

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
    _merge_grid_entry_from_data(data, project)
    return data


def _resolve_coord_entry(
    data: dict[str, Any], project: ProjectTables
) -> tuple[str | None, Mapping[str, Any] | None]:
    requested = str(
        data.get("table_entry")
        or data.get("axis_entry")
        or data.get("coordinate")
        or data.get("name")
        or ""
    )
    coord: dict[str, Mapping[str, Any]] = project.coordinate_entries
    if requested in coord:
        return requested, coord[requested]
    generic = _generic_level_matches(data, project, requested)
    if len(generic) == 1:
        return generic[0]
    if len(generic) > 1:
        choices = ", ".join(n for n, _ in generic)
        raise TableValidationError(
            f"Generic level {requested!r} matches multiple coordinate entries; "
            f"specify table_entry or axis_entry.  Choices: {choices}."
        )
    by_out = [
        (n, e) for n, e in coord.items() if str(e.get("out_name", "")) == requested
    ]
    if len(by_out) == 1:
        return by_out[0]
    m = _match_by_attrs(data, project)
    if len(m) == 1:
        return m[0]
    return None, None


def _resolve_grid_entry(
    data: dict[str, Any], project: ProjectTables
) -> tuple[str | None, Mapping[str, Any] | None]:
    requested = str(
        data.get("grid_table_entry")
        or data.get("grid_coordinate")
        or data.get("out_name")
        or data.get("name")
        or ""
    )
    gc: dict[str, Mapping[str, Any]] = project.grid_coordinate_entries
    if requested in gc:
        return requested, gc[requested]
    m = [(n, e) for n, e in gc.items() if str(e.get("out_name", "")) == requested]
    if len(m) == 1:
        return m[0]
    return None, None


def _merge_grid_entry(
    data: dict[str, Any],
    entry_name: str | None,
    entry: Mapping[str, Any],
    project: ProjectTables,
) -> None:
    if entry_name:
        data.setdefault("grid_table_entry", entry_name)
    for key in ("out_name", "units", "standard_name", "long_name", "valid_min", "valid_max"):
        val = entry.get(key)
        if is_table_value(val):
            data.setdefault(key, parse_table_value(val))
    data.setdefault("out_name", entry_name)
    bname = data.get("bounds_name")
    if bname:
        be = project.grid_coordinate_entries.get(str(bname))
        if be:
            ba = dict(data.get("bounds_attrs") or {})
            for key in ("units", "standard_name", "long_name"):
                val = be.get(key)
                if is_table_value(val):
                    ba.setdefault(key, parse_table_value(val))
            if ba:
                data["bounds_attrs"] = ba


def _merge_grid_entry_from_data(
    data: dict[str, Any], project: ProjectTables
) -> None:
    en, entry = _resolve_grid_entry(data, project)
    if entry is not None:
        _merge_grid_entry(data, en, entry, project)


def _generic_level_matches(
    data: dict[str, Any], project: ProjectTables, generic_name: str
) -> list[tuple[str, Mapping[str, Any]]]:
    generic: dict[str, dict[str, Mapping[str, Any]]] = getattr(
        project, "generic_level_entries", {}
    )
    matches = list(generic.get(generic_name, {}).items())
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
    data: dict[str, Any], project: ProjectTables
) -> list[tuple[str, Mapping[str, Any]]]:
    out_name = data.get("out_name")
    std_name = data.get("standard_name")
    if not out_name and not std_name:
        return []
    matches = list(project.coordinate_entries.items())
    for key, val in (("out_name", out_name), ("standard_name", std_name)):
        if val in (None, ""):
            continue
        narrowed = [(n, e) for n, e in matches if str(e.get(key, "")) == str(val)]
        if narrowed:
            matches = narrowed
    return matches if len(matches) == 1 else []


# ---------------------------------------------------------------------------
# Variable — private helpers
# ---------------------------------------------------------------------------


def _apply_variable_defaults(
    data: dict[str, Any], project: ProjectTables
) -> dict[str, Any]:
    entry = _resolve_variable_entry(data, project)
    return _merge_variable_entry(data, entry)


def _resolve_variable_entry(
    data: dict[str, Any], project: ProjectTables
) -> VariableEntry:
    requested = str(data.get("name") or data.get("variable_id") or data.get("id") or "")
    entries_by_name: dict[str, list[VariableEntry]] = project._variable_entries_by_name
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


def _merge_variable_entry(
    data: dict[str, Any], entry: VariableEntry
) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# ZFactor — private helpers
# ---------------------------------------------------------------------------


def _apply_zfactor_defaults(
    data: dict[str, Any], project: ProjectTables
) -> dict[str, Any]:
    entry_name, entry = _resolve_zfactor_entry(data, project)
    if entry is None:
        return data
    data.setdefault("table_entry", entry_name)
    validate_table_metadata(
        data, entry_name, entry, ("units", "standard_name", "long_name"), "formula term"
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


def _resolve_zfactor_entry(
    data: dict[str, Any], project: ProjectTables
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


# ---------------------------------------------------------------------------
# Grid — private helper
# ---------------------------------------------------------------------------


def _apply_grid_defaults(
    data: dict[str, Any], project: ProjectTables
) -> dict[str, Any]:
    requested = str(
        data.get("table_entry") or data.get("mapping_entry") or data.get("name") or ""
    )
    if not requested:
        return data
    ge: dict[str, Mapping[str, Any]] = getattr(project, "grid_mapping_entries", {})
    entry = ge.get(requested)
    if entry is None:
        return data
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
