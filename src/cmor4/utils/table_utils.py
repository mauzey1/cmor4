from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from ..exceptions import TableValidationError


def is_table_value(value: Any) -> bool:
    return value not in (None, "")


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
    table_values: Mapping[str, Any] | BaseModel,
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
        Typed table entry or mapping whose values are the authoritative source.
    keys:
        Field names to check.
    entity_type:
        Human-readable label for error messages, e.g. ``"axis"`` or
        ``"formula term"``.
    """
    for key in keys:
        expected = (
            getattr(table_values, key, None)
            if isinstance(table_values, BaseModel)
            else table_values.get(key)
        )
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
