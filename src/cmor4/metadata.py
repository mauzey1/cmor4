from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields
from typing import Any, TypeVar


MetadataRecordT = TypeVar("MetadataRecordT", bound="_MetadataRecord")


class _MetadataRecord(Mapping[str, Any]):
    """Base for metadata records with controlled serialization helpers."""

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

    def updated(self: MetadataRecordT, **updates: Any) -> MetadataRecordT:
        return type(self).from_mapping({**self.to_dict(), **updates})

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())
