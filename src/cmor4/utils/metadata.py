"""Pydantic base class shared by all CMOR4 metadata records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Self

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

# ---------------------------------------------------------------------------
# Shared coercion helpers and annotated type aliases
# ---------------------------------------------------------------------------
# Axis, Variable, ZFactor, and Grid import these instead of duplicating them.
# ---------------------------------------------------------------------------


def _str_seq(v: Any) -> list[str] | tuple[str, ...] | None:
    """Coerce to list-or-tuple of str, preserving the container type."""
    if v is None:
        return None
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, tuple):
        return tuple(str(x) for x in v)
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _to_bool(v: Any) -> bool | None:
    """Coerce truthy table strings (``"1"``, ``"true"``, ``"yes"``) to ``bool``.

    Table JSON files store boolean flags as strings.  Accepting ``str | bool``
    at the field level and delegating the conversion to this validator means
    every consumer receives a genuine ``bool | None`` and no longer needs the
    ``_is_truthy()`` guard function.
    """
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes"}


StrTuple = Annotated[
    tuple[str, ...] | None,
    BeforeValidator(lambda value: (value,) if isinstance(value, str) else value),
]
StrSeq = Annotated[list[str] | tuple[str, ...] | None, BeforeValidator(_str_seq)]
IntTuple = tuple[int, ...] | None
StrOrTuple = Annotated[
    str | tuple[str, ...] | None,
    AfterValidator(
        lambda value: value[0]
        if isinstance(value, tuple) and len(value) == 1
        else value
    ),
]
CoercedF = Annotated[
    float | None, BeforeValidator(lambda value: None if value == "" else value)
]
AxisStr = Annotated[str, StringConstraints(to_upper=True)] | None
BoolCoerced = Annotated[bool | None, BeforeValidator(_to_bool)]


# ---------------------------------------------------------------------------
# MetadataModel
# ---------------------------------------------------------------------------


class MetadataModel(BaseModel):
    """Frozen Pydantic base for CMOR4 metadata records.

    Subclasses (Axis, Variable, ZFactor, Grid) are pure data holders: they
    declare typed fields and provide NetCDF output helpers.  All project-table
    resolution and merge logic lives on the table classes in
    :mod:`cmor4.utils.tables`.

    Design notes
    ------------
    * ``frozen=True`` enforces immutability after construction.
      Use :meth:`updated` to create a modified copy.
    * ``extra`` is a **declared field** rather than Pydantic's ``model_extra``.
      Unknown kwargs are routed there by ``_collect_extras`` so they end up
      in :meth:`to_dict` and are written as NetCDF attributes when valid.
    * ``extra="ignore"`` in model_config: truly unknown keys are routed into
      ``self.extra`` by the ``_collect_extras`` before-validator.
    * ``coerce_numbers_to_str=True`` converts numeric table-entry values
      (e.g. ``"units": 1``) silently for ``str`` fields.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        coerce_numbers_to_str=True,
    )

    extra: dict[str, Any] = Field(default_factory=dict, repr=False)

    @model_validator(mode="before")
    @classmethod
    def _collect_extras(cls, data: Any) -> Any:
        """Spread ``extra=`` dict and collect unknown kwargs into ``extra``."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        explicit_extra = data.pop("extra", None)
        if isinstance(explicit_extra, dict):
            for k, v in explicit_extra.items():
                data.setdefault(k, v)
        known = set(cls.model_fields.keys())
        unknown = {k: v for k, v in list(data.items()) if k not in known}
        if unknown:
            for k in unknown:
                del data[k]
            data["extra"] = {**unknown, **(data.get("extra") or {})}
        return data

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return all meaningful (non-None, non-empty-dict) field values.

        ``extra`` field contents are inlined at the top level.
        """
        result = {
            name: value
            for name, field in type(self).model_fields.items()
            if name != "extra"
            and not getattr(field, "exclude", False)
            and (value := getattr(self, name)) is not None
            and (not isinstance(value, dict) or value)
        }
        result.update(
            (key, value)
            for key, value in self.extra.items()
            if value is not None and key not in result
        )
        return result

    def updated(self, **updates: Any) -> Self:
        """Return a new instance with *updates* applied (no table re-merge)."""
        return type(self).model_validate({**self.to_dict(), **updates})

    # ------------------------------------------------------------------
    # NetCDF helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_netcdf_attr_value(value: Any) -> bool:
        return isinstance(value, (str, bytes, int, float, np.integer, np.floating))

    @staticmethod
    def netcdf_attrs(values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(k): v
            for k, v in values.items()
            if MetadataModel.is_netcdf_attr_value(v)
        }

    @staticmethod
    def netcdf_array(value: Any) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype.kind in {"U", "S", "O"}:
            return arr.astype(str)
        return arr
