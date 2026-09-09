"""Small predicates for values declared by controlled vocabularies."""

from __future__ import annotations

from typing import Any


def has_value(value: Any) -> bool:
    """Return whether a metadata constraint is present."""

    return value not in (None, "")
