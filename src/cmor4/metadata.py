from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields
from typing import Any, TypeVar

import numpy as np


MetadataRecordT = TypeVar("MetadataRecordT", bound="_MetadataRecord")


class _MetadataRecord(Mapping[str, Any]):
    """Base for metadata records with controlled serialization helpers.

    Parameters
    ----------
    extra:
        Additional mapping keys preserved by concrete metadata records.
    """

    extra: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls: type[MetadataRecordT], values: Mapping[str, Any]
    ) -> MetadataRecordT:
        """Create a metadata record from a mapping.

        Parameters
        ----------
        values:
            Metadata values to map onto dataclass fields and ``extra`` keys.

        Returns
        -------
        _MetadataRecord
            Instance of the concrete metadata record class.
        """

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
        """Return metadata as a dictionary without empty optional values.

        Returns
        -------
        dict[str, Any]
            Serializable metadata values, including preserved ``extra`` keys.
        """

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
        """Return a copy of this record with updates applied.

        Parameters
        ----------
        **updates:
            Metadata values to override or add.

        Returns
        -------
        _MetadataRecord
            New metadata record of the same concrete type.
        """

        return type(self).from_mapping({**self.to_dict(), **updates})

    @staticmethod
    def is_netcdf_attr_value(value: Any) -> bool:
        """Return whether a value can be written as a simple NetCDF attribute.

        Parameters
        ----------
        value:
            Value to test.

        Returns
        -------
        bool
            ``True`` for scalar string, bytes, integer, or floating values.
        """

        return isinstance(
            value, (str, bytes, int, float, np.integer, np.floating)
        )

    @staticmethod
    def netcdf_attrs(values: Mapping[str, Any]) -> dict[str, Any]:
        """Filter a mapping to NetCDF-safe attribute values.

        Parameters
        ----------
        values:
            Candidate attribute values.

        Returns
        -------
        dict[str, Any]
            Attributes whose values can be written directly to NetCDF.
        """

        return {
            str(key): value
            for key, value in values.items()
            if _MetadataRecord.is_netcdf_attr_value(value)
        }

    @staticmethod
    def netcdf_array(value: Any) -> np.ndarray:
        """Convert a value to a NetCDF-ready array.

        Parameters
        ----------
        value:
            Scalar or array-like value.

        Returns
        -------
        numpy.ndarray
            Array with object and string-like dtypes normalized to strings.
        """

        array = np.asarray(value)
        if array.dtype.kind in {"U", "S", "O"}:
            return array.astype(str)
        return array

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())
