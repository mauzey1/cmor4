"""Pydantic base class shared by all CMOR4 metadata records."""
from __future__ import annotations

from collections.abc import Iterator, KeysView, ItemsView, Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetadataModel(BaseModel):
    """Frozen Pydantic base with Mapping protocol and NetCDF helpers.

    Key design decisions
    --------------------
    * **No direct ``Mapping`` inheritance.** ``MetadataModel`` implements the
      full :class:`~collections.abc.Mapping` interface manually and is
      registered as a virtual ``Mapping`` subclass via
      ``Mapping.register(MetadataModel)``.  This avoids Pydantic's
      "Field name 'values' shadows an attribute in parent" warning that
      would occur because ``Mapping`` supplies a ``values()`` method and
      ``Axis`` / ``ZFactor`` declare a field named ``values``.
    * ``extra`` is a **declared field** (``dict[str, Any]``), not Pydantic's
      ``model_extra``.  This preserves backward compatibility: callers may
      still pass ``Variable(..., extra={"k": "v"})`` *or* pass unknown kwargs
      directly (``Variable(..., k="v")``); both end up in ``self.extra``.
      It also lets test code do ``object.__setattr__(model, "extra", {})`` to
      bypass the frozen guard in setup helpers.
    * ``extra="ignore"`` is set in model_config; truly unknown keys are
      routed into ``self.extra`` by the ``_collect_extras`` before-validator.
    * ``coerce_numbers_to_str=True`` means numeric values from JSON table
      entries (e.g. ``"units": 1``) are quietly converted for ``str`` fields.
    * ``frozen=True`` enforces immutability.  Use :meth:`updated` for copies.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",         # unknown keys handled by _collect_extras below
        arbitrary_types_allowed=True,
        populate_by_name=True,
        coerce_numbers_to_str=True,
    )

    # Explicit extra-metadata bucket (mirrors original dataclass field)
    extra: dict[str, Any] = Field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Before-validator: normalise extra= kwarg and unknown kwargs
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _collect_extras(cls, data: Any) -> Any:
        """Spread ``extra=`` dict and collect truly unknown kwargs into it.

        This runs for both ``Model(**kwargs)`` and ``Model.model_validate(d)``
        construction paths, so the ``extra`` field is always the single
        authoritative location for non-declared keys.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.pop("project", None)   # consumed by __init__ before Pydantic sees it
        # Spread an explicit extra= dict first
        explicit_extra = data.pop("extra", None)
        if isinstance(explicit_extra, dict):
            for k, v in explicit_extra.items():
                data.setdefault(k, v)
        # Collect all remaining unknown keys into extra
        known = set(cls.model_fields.keys())
        unknown = {k: v for k, v in list(data.items()) if k not in known}
        if unknown:
            for k in unknown:
                del data[k]
            data["extra"] = {**unknown, **(data.get("extra") or {})}
        return data

    # ------------------------------------------------------------------
    # Backward-compatible project= kwarg support
    # ------------------------------------------------------------------

    def __init__(self, **data: Any) -> None:
        """Accept an optional ``project=`` kwarg for table-backed construction.

        ``project`` is popped here — *before* Pydantic validation — so any
        :exc:`~cmor4.exceptions.TableValidationError` raised by the table
        merging propagates directly to the caller, not wrapped in a
        ``pydantic.ValidationError``.
        """
        project = data.pop("project", None)
        if project is not None:
            data = type(self)._apply_table_defaults(data, project)
        super().__init__(**data)
        if project is not None:
            self._post_project_init()

    @classmethod
    def _apply_table_defaults(
        cls, data: dict[str, Any], project: Any
    ) -> dict[str, Any]:
        """Merge project-table defaults into *data*.  No-op on the base class."""
        return data

    def _post_project_init(self) -> None:
        """Called after frozen construction when ``project=`` was supplied."""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _data(self) -> dict[str, Any]:
        """Return all meaningful (non-None, non-empty-dict) field values.

        The ``extra`` field itself is *not* included; its contents are inlined
        at the top level (mirroring the original ``_MetadataRecord.to_dict``).
        Fields with ``exclude=True`` are also omitted.
        """
        result: dict[str, Any] = {}
        for name, info in type(self).model_fields.items():
            if name == "extra" or getattr(info, "exclude", False):
                continue
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, dict) and not value:
                continue
            result[name] = value
        # Inline extra contents
        for key, value in self.extra.items():
            if value is not None:
                result.setdefault(key, value)
        return result

    # ------------------------------------------------------------------
    # Mapping protocol (explicit, no ABC inheritance)
    #
    # Inheriting from Mapping[str, Any] would put Mapping.values() in the
    # MRO, causing Pydantic to warn that the 'values' field in Axis and
    # ZFactor shadows that method.  Instead we implement the protocol by
    # hand and register MetadataModel as a virtual Mapping subclass at the
    # bottom of this module so that isinstance(obj, Mapping) still holds.
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())

    def __contains__(self, key: object) -> bool:
        return key in self._data()

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> KeysView[str]:
        """Return a view of all Mapping keys."""
        return self._data().keys()

    def items(self) -> ItemsView[str, Any]:
        """Return a view of all ``(key, value)`` pairs."""
        return self._data().items()

    # Note: no .values() method — 'values' is a data field on Axis/ZFactor.

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> MetadataModel:
        """Construct from a plain mapping."""
        return cls.model_validate(dict(values))

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable copy of the Mapping view."""
        return self._data()

    def updated(self, **updates: Any) -> MetadataModel:
        """Return a new instance with *updates* applied (no table re-merge)."""
        return type(self).model_validate({**self.to_dict(), **updates})

    # NetCDF helpers -------------------------------------------------

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


# Register as a virtual Mapping subclass so that
# isinstance(variable, Mapping) and isinstance(axis, Mapping) remain True
# without putting Mapping.values() in the MRO.
Mapping.register(MetadataModel)
