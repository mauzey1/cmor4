from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..exceptions import TableValidationError


def table_dimensions(entry: Mapping[str, Any]) -> tuple[str, ...]:
    dimensions = entry.get("dimensions", ())
    if isinstance(dimensions, str):
        values = tuple(dimensions.split())
    else:
        values = tuple(str(value) for value in dimensions)
    return tuple(reversed(values))


def is_table_value(value: Any) -> bool:
    return value not in (None, "")


def entry_values(entry: Mapping[str, Any]) -> list[Any] | None:
    requested = entry.get("requested")
    if is_table_value(requested):
        if isinstance(requested, list):
            return [parse_table_value(value) for value in requested]
        return [parse_table_value(requested)]
    value = entry.get("value")
    if is_table_value(value):
        return [parse_table_value(value)]
    return None


def entry_bounds(entry: Mapping[str, Any]) -> list[list[Any]] | None:
    requested_bounds = entry.get("requested_bounds")
    if not is_table_value(requested_bounds):
        requested_bounds = entry.get("bounds_values")
    if not is_table_value(requested_bounds):
        return None
    values = (
        requested_bounds
        if isinstance(requested_bounds, list)
        else str(requested_bounds).split()
    )
    parsed = [parse_table_value(value) for value in values]
    if len(parsed) % 2:
        return None
    return [parsed[index : index + 2] for index in range(0, len(parsed), 2)]


def parse_table_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def metadata_value_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return str(value) in {str(item) for item in expected}
    expected_text = str(expected)
    if expected_text.endswith(" since ?"):
        return " since " in str(value)
    if "?" in expected_text:
        pattern = re.escape(expected_text).replace(r"\?", ".+")
        return re.fullmatch(pattern, str(value)) is not None
    return str(value) == str(expected)


def validate_table_metadata(
    data: dict[str, Any],
    entry_name: str | None,
    table_values: Mapping[str, Any],
    keys: Sequence[str],
    entity_type: str = "entry",
) -> None:
    """Raise TableValidationError if user values conflict with a table entry.

    Parameters
    ----------
    data:
        User-supplied key/value pairs (e.g. from an Axis or ZFactor).
    entry_name:
        Name of the table entry being checked (used in error messages).
    table_values:
        The raw table entry dict whose values are the authoritative source.
    keys:
        Field names to check.
    entity_type:
        Human-readable label for error messages, e.g. ``"axis"`` or
        ``"formula term"``.
    """
    for key in keys:
        expected = table_values.get(key)
        user_val = data.get(key)
        if (
            is_table_value(expected)
            and user_val not in (None, "")
            and not metadata_value_matches(user_val, expected)
        ):
            raise TableValidationError(
                f"{entity_type} {entry_name!r} {key}={user_val!r} "
                f"does not match table value {expected!r}."
            )
