"""Prepared dataset metadata record."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import Field

from .utils.dataset_metadata import DatasetMetadata


class DatasetInfo(DatasetMetadata):
    """Validated/defaulted dataset metadata ready for writing.

    Parameters
    ----------
    project : ProjectTables, optional
        :class:`~cmor4.tables.ProjectTables` that prepared this record.
        Excluded from the dict-like view and serialisation.
    user_info : dict, optional
        Original user-supplied metadata before CV defaults were applied.
        Excluded from the dict-like view and serialisation.
    """

    project: Any = Field(default=None, exclude=True, repr=False)
    user_info: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    @classmethod
    def from_prepared(
        cls,
        values: DatasetMetadata,
        *,
        project: Any = None,
        user_info: Mapping[str, Any] | None = None,
    ) -> DatasetInfo:
        """Create prepared dataset metadata with project context attached."""

        if not isinstance(values, DatasetMetadata):
            values = DatasetMetadata.from_mapping(cast(Mapping[str, Any], values))

        data = values.to_dict()
        data["project"] = project
        data["user_info"] = dict(user_info) if user_info is not None else dict(values)
        return cls.model_validate(data)
