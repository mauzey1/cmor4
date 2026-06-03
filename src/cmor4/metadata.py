from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar


MetadataRecordT = TypeVar("MetadataRecordT", bound="_MetadataRecord")


class _MetadataRecord(Mapping[str, Any]):
    """Mapping-compatible base for public metadata records."""

    extra: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls: type[MetadataRecordT], values: Mapping[str, Any]
    ) -> MetadataRecordT:
        field_names = {field.name for field in fields(cls)}
        kwargs = {
            key: value
            for key, value in values.items()
            if key in field_names and key != "extra"
        }
        extra = {
            str(key): value
            for key, value in values.items()
            if key not in field_names and value is not None
        }
        if extra:
            kwargs["extra"] = extra
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field_info in fields(self):
            if field_info.name == "extra":
                continue
            value = getattr(self, field_info.name)
            if value is None:
                continue
            if isinstance(value, Mapping) and not value:
                continue
            data[field_info.name] = value
        for key, value in self.extra.items():
            if value is not None:
                data.setdefault(str(key), value)
        return data

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class Variable(_MetadataRecord):
    """Metadata for the data variable being written."""

    name: str
    id: str | None = None
    variable_id: str | None = None
    table_id: str | None = None
    dimensions: tuple[str, ...] | list[str] | None = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    cell_methods: str | None = None
    cell_measures: str | None = None
    comment: str | None = None
    missing_value: Any = None
    fill_value: Any = None
    chunksizes: tuple[int, ...] | list[int] | None = None
    chunks: tuple[int, ...] | list[int] | None = None
    coordinates: str | tuple[str, ...] | list[str] | None = None
    formula_terms: str | None = None
    frequency: str | None = None
    realm: str | None = None
    table_info: str | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Axis(_MetadataRecord):
    """Metadata and coordinate values for one data axis."""

    name: str
    values: Any = None
    bounds: Any = None
    dimensions: tuple[str, ...] | list[str] | None = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    axis: str | None = None
    positive: str | None = None
    formula: str | None = None
    valid_min: Any = None
    valid_max: Any = None
    out_name: str | None = None
    table_entry: str | None = None
    axis_entry: str | None = None
    coordinate: str | None = None
    grid_table_entry: str | None = None
    grid_coordinate: str | None = None
    scalar: bool | None = None
    auxiliary: bool | None = None
    auxiliary_name: str | None = None
    auxiliary_attrs: Mapping[str, Any] = field(default_factory=dict)
    climatology: str | bool | None = None
    generic_level_name: str | None = None
    z_factors: str | None = None
    z_bounds_factors: str | None = None
    bounds_name: str | None = None
    bounds_dim: str | None = None
    bounds_attrs: Mapping[str, Any] = field(default_factory=dict)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ZFactor(_MetadataRecord):
    """Metadata and values for one hybrid-coordinate formula term."""

    name: str
    values: Any = None
    data: Any = None
    dimensions: tuple[str, ...] | list[str] | None = None
    units: str | None = None
    standard_name: str | None = None
    long_name: str | None = None
    out_name: str | None = None
    table_entry: str | None = None
    formula_entry: str | None = None
    bounds: Any = None
    bounds_name: str | None = None
    bounds_dim: str | None = None
    bounds_attrs: Mapping[str, Any] = field(default_factory=dict)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)
