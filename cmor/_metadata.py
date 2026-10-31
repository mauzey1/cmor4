"""Shared metadata models, typed coercion, and dataset attributes.

Domain records use these foundations; project-table resolution belongs to the
coordinate and variable table modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Annotated, Any, Self

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from . import __version__ as _VERSION

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


def numeric_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


RIPF_KEYS: tuple[str, ...] = (
    "realization_index",
    "initialization_index",
    "physics_index",
    "forcing_index",
)

# Expected single-character prefix for each RIPF key (e.g. "r9", "i1", ...).
_RIPF_PREFIXES: dict[str, str] = {
    "realization_index": "r",
    "initialization_index": "i",
    "physics_index": "p",
    "forcing_index": "f",
}
# CMOR3 upper bound: 32-bit signed integer.
_RIPF_MAX: int = 2**31 - 1


def ripf_index_error(key: str, value: Any) -> str | None:
    """Return an index format/range error, preserving optional prefix semantics."""
    prefix = _RIPF_PREFIXES[key]
    text = str(value)
    numeric = text[1:] if text.startswith(prefix) and len(text) > 1 else text
    try:
        number = int(numeric)
    except (TypeError, ValueError):
        return (
            f"{key}={value!r} must be a positive integer "
            f"(with optional '{prefix}' prefix, e.g. '{prefix}1')."
        )
    if not 1 <= number <= _RIPF_MAX:
        return (
            f"{key}={value!r} numeric value {number} is out of the "
            f"valid range [1, {_RIPF_MAX}]."
        )
    return None


StrTuple = Annotated[
    tuple[str, ...] | None,
    BeforeValidator(lambda value: (value,) if isinstance(value, str) else value),
]
StrSeq = Annotated[list[str] | tuple[str, ...] | None, BeforeValidator(_str_seq)]
IntTuple = tuple[int, ...] | None
StrOrTuple = Annotated[
    str | tuple[str, ...] | None,
    AfterValidator(
        lambda value: (
            value[0] if isinstance(value, tuple) and len(value) == 1 else value
        )
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
    """Frozen Pydantic base for CMOR metadata records.

    Subclasses (Axis, Variable, ZFactor, Grid) are pure data holders: they
    declare typed fields and provide NetCDF output helpers.  All project-table
    resolution and merge logic lives on the coordinate and variable table
    classes.

    Design notes
    ------------
    * ``frozen=True`` enforces immutability after construction.
      Use :meth:`updated` to create a modified copy.
    * ``extra`` is an explicit compatibility field. Unknown constructor keys
      are rejected; callers should use the component's ``attrs`` field for
      output attributes.
    * ``coerce_numbers_to_str=True`` converts numeric table-entry values
      (e.g. ``"units": 1``) silently for ``str`` fields.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        coerce_numbers_to_str=True,
    )

    extra: dict[str, Any] = Field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return all meaningful (non-None, non-empty-dict) field values.

        ``extra`` field contents are inlined at the top level.
        """
        result = {
            str(field.serialization_alias or field.alias or name): value
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
        return type(self).model_validate({**self.model_dump(), **updates})

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


# Keys that must not appear in NetCDF global attributes.
INTERNAL_DATASET_KEYS: frozenset[str] = frozenset({
    "_history_template",
    "create_subdirectories",
    "outpath",
    "output_file_template",
    "output_path_template",
    "tracking_prefix",
})


class DatasetMetadata(MetadataModel):
    """Shared dataset-level metadata fields and helpers."""

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        coerce_numbers_to_str=True,
    )

    # Typed CMIP DRS fields
    mip_era: str | None = None
    activity_id: str | None = None
    institution_id: str | None = None
    source_id: str | None = None
    experiment_id: str | None = None
    grid_label: str | None = None
    license_id: str | None = None
    realization_index: str | int | None = None
    initialization_index: str | int | None = None
    physics_index: str | int | None = None
    forcing_index: str | int | None = None
    calendar: str | None = None
    frequency: str | None = None
    nominal_resolution: str | None = None
    outpath: str | None = None
    version: str | None = None
    variant_label_value: str | None = Field(
        default=None,
        alias="variant_label",
        serialization_alias="variant_label",
    )
    realm: str | None = None
    source_type: str | None = None
    forcing: str | None = None
    member_id: str | None = None
    region: str | None = None

    # Output and template controls
    create_subdirectories: BoolCoerced = None
    output_file_template: str | None = None
    output_path_template: str | None = None
    tracking_prefix: str | None = None

    # Variable-derived dataset metadata
    variable_id: str | None = None
    branded_variable: str | None = None
    branding_suffix: str | None = None
    temporal_label: str | None = None
    vertical_label: str | None = None
    horizontal_label: str | None = None
    area_label: str | None = None
    table_id: str | None = None
    table_info: str | None = None

    # Parent-experiment metadata
    parent_experiment_id: str | None = None
    parent_activity_id: str | None = None
    parent_mip_era: str | None = None
    parent_source_id: str | None = None
    parent_time_units: str | None = None
    parent_variant_label: str | None = None
    branch_time_in_child: float | None = None
    branch_time_in_parent: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _collect_project_attributes(cls, data: Any) -> Any:
        """Retain project-specific global attributes in the explicit extra field."""

        if not isinstance(data, dict):
            return data
        data = dict(data)
        explicit_extra = data.pop("extra", None)
        extra = dict(explicit_extra) if isinstance(explicit_extra, dict) else {}
        known = set(cls.model_fields)
        known.update(
            str(field.alias)
            for field in cls.model_fields.values()
            if field.alias is not None
        )
        for key in tuple(data):
            if key not in known:
                extra.setdefault(key, data.pop(key))
        data["extra"] = extra
        return data

    def __init__(
        self,
        data: Mapping[str, Any] | DatasetMetadata | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        """Accept an optional positional metadata mapping."""
        source = data.to_dict() if isinstance(data, DatasetMetadata) else data
        merged = {**dict(source), **kwargs} if source is not None else kwargs
        super().__init__(**merged)

    @field_validator(
        "realization_index",
        "initialization_index",
        "physics_index",
        "forcing_index",
        mode="before",
    )
    @classmethod
    def _validate_ripf_index(cls, v: Any, info: Any) -> str | int | None:
        """Validate RIPF index format and range."""
        if v in (None, ""):
            return None
        error = ripf_index_error(info.field_name, v)
        if error is not None:
            raise ValueError(error)
        return v

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> Self:
        """Create dataset metadata from a mapping-like object."""
        return cls.model_validate(dict(values))

    def variant_label(self) -> str:
        """Return the explicit or RIPF-derived variant label."""
        if self.variant_label_value:
            return self.variant_label_value
        vals = [getattr(self, key) for key in RIPF_KEYS]
        if all(v not in (None, "") for v in vals):
            return "".join(str(v) for v in vals)
        return "r1i1p1f1"

    def global_attributes(
        self,
        variable: Any,
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return complete NetCDF global attributes for this dataset and variable."""
        creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        metadata = self.to_dict()
        attrs: dict[str, Any] = {
            "Conventions": metadata.get("Conventions", "CF-1.12"),
            "cmor_version": _VERSION,
        }
        for key, val in metadata.items():
            if key in INTERNAL_DATASET_KEYS or key.startswith("_"):
                continue
            if MetadataModel.is_netcdf_attr_value(val):
                attrs[key] = val

        var_id, labels = variable.names()
        attrs.setdefault("variable_id", var_id)
        attrs.setdefault("branded_variable", labels["branded_name"])
        for key in (
            "branding_suffix",
            "temporal_label",
            "vertical_label",
            "horizontal_label",
            "area_label",
        ):
            if key in labels:
                attrs.setdefault(key, labels[key])
        for key, val in (
            ("frequency", variable.frequency),
            ("realm", variable.realm),
            ("table_id", variable.table_id),
        ):
            if val is not None:
                attrs.setdefault(key, val)
        if variable.table_info is not None:
            attrs.setdefault("table_info", variable.table_info)
        attrs.setdefault("variant_label", self.variant_label())
        attrs.setdefault("creation_date", creation_date)

        conv = attrs.get("Conventions", "CF-1.12")
        mip_era = attrs.get("mip_era") or metadata.get("mip_era", "CMIP")
        attrs.setdefault(
            "history",
            f"{creation_date} ; CMOR rewrote data to be consistent with "
            f"{conv} and {mip_era} data requirements.",
        )
        src = attrs.get("source_id") or metadata.get("source_id", "")
        if src:
            attrs.setdefault("title", f"{src} output prepared for {mip_era}")

        if extra_attrs:
            attrs.update(MetadataModel.netcdf_attrs(extra_attrs))
        return attrs
