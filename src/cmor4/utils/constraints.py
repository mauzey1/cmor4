"""Small predicates for matching values declared by controlled vocabularies."""

from __future__ import annotations

import re
from typing import Any


def has_value(value: Any) -> bool:
    """Return whether a metadata constraint is present."""

    return value not in (None, "")


def value_matches_constraint(value: Any, expected: Any) -> bool:
    """Return whether *value* satisfies a CV/table constraint.

    Lists express alternatives and ``?`` in strings expresses a wildcard.
    These semantics belong to CV constraints rather than table parsing.
    """

    if isinstance(expected, list):
        return str(value) in {str(item) for item in expected}
    expected_text = str(expected)
    if expected_text.endswith(" since ?"):
        return " since " in str(value)
    if "?" in expected_text:
        pattern = re.escape(expected_text).replace(r"\?", ".+")
        return re.fullmatch(pattern, str(value)) is not None
    return str(value) == str(expected)
