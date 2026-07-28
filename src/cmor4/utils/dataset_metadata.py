"""Shared dataset-level metadata model."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from datetime import datetime, timezone
from typing import Any, Self

from pydantic import field_validator

from ..metadata import MetadataModel

# Keys that must not appear in NetCDF global attributes
INTERNAL_DATASET_KEYS: frozenset[str] = frozenset({
    "_history_template",
    "create_subdirectories",
    "outpath",
    "output_file_template",
    "output_path_template",
    "tracking_prefix",
})

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


class DatasetMetadata(MetadataModel):
    """Shared dataset-level metadata fields and helpers."""

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

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        """Accept an optional positional metadata mapping."""
        merged = {**dict(data), **kwargs} if data is not None else kwargs
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
        field_name: str = info.field_name
        prefix = _RIPF_PREFIXES[field_name]
        s = str(v)
        numeric_part = s[1:] if s.startswith(prefix) and len(s) > 1 else s
        try:
            n = int(numeric_part)
        except (TypeError, ValueError):
            raise ValueError(
                f"{field_name}={v!r} must be a positive integer "
                f"(with optional '{prefix}' prefix, e.g. '{prefix}1')."
            )
        if not (1 <= n <= _RIPF_MAX):
            raise ValueError(
                f"{field_name}={v!r} numeric value {n} is out of the "
                f"valid range [1, {_RIPF_MAX}]."
            )
        return v

    def to_dict(self) -> dict[str, Any]:
        """Mutable copy of the public metadata view."""
        return super().to_dict()

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def items(self) -> ItemsView[str, Any]:
        return self.to_dict().items()

    def keys(self) -> KeysView[str]:
        return self.to_dict().keys()

    def values(self) -> ValuesView[Any]:
        return self.to_dict().values()

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> Self:
        """Create dataset metadata from a mapping-like object."""
        return cls.model_validate(dict(values))

    def variant_label(self) -> str:
        """Return the explicit or RIPF-derived variant label."""
        metadata = self.to_dict()
        vl = metadata.get("variant_label")
        if vl:
            return str(vl)
        vals = [metadata.get(k) for k in RIPF_KEYS]
        if all(v not in (None, "") for v in vals):
            return "".join(str(v) for v in vals)
        return "r1i1p1f1"

    def global_attributes(
        self,
        variable: Any,
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return complete NetCDF global attributes for this dataset and variable."""
        from .. import __version__ as _ver

        creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        metadata = self.to_dict()
        attrs: dict[str, Any] = {
            "Conventions": metadata.get("Conventions", "CF-1.12"),
            "cmor_version": _ver,
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
