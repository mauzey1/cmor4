from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .metadata import _MetadataRecord


INTERNAL_DATASET_KEYS = {
    "_history_template",
    "outpath",
    "output_file_template",
    "output_path_template",
}

RIPF_KEYS = (
    "realization_index",
    "initialization_index",
    "physics_index",
    "forcing_index",
)


@dataclass(frozen=True)
class DatasetInfo(Mapping[str, Any]):
    """Prepared dataset-level metadata.

    ``DatasetInfo`` is mapping-compatible so existing APIs that accept a
    dataset metadata dictionary can use it directly. When created from project
    tables it contains user-provided dataset metadata plus project CV defaults
    and runtime global attributes. Variable-derived global attributes are added
    when a variable is passed to ``create_dataset`` or related helpers.
    """

    data: Mapping[str, Any]
    project: Any = field(default=None, repr=False, compare=False)
    user_info: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data",
            {str(key): value for key, value in self.data.items()},
        )
        object.__setattr__(
            self,
            "user_info",
            {str(key): value for key, value in self.user_info.items()},
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        project: Any = None,
    ) -> "DatasetInfo":
        """Create dataset info directly from user metadata."""

        return cls(
            values,
            project=project,
            user_info=values,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable copy of the prepared dataset metadata."""

        return dict(self.data)

    def global_attributes(
        self,
        variable: Any,
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return NetCDF global attributes for this dataset and variable."""

        attrs: dict[str, Any] = {
            "Conventions": self.get("Conventions", "CF-1.11"),
            "cmor4_version": "0.1.0",
        }
        for key, value in self.items():
            if key in INTERNAL_DATASET_KEYS or key.startswith("_"):
                continue
            if _MetadataRecord.is_netcdf_attr_value(value):
                attrs[key] = value

        var_name, labels = variable.names()
        attrs.setdefault("variable_id", var_name)
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
        for key in ("frequency", "realm", "table_id"):
            if key in variable:
                attrs.setdefault(key, variable[key])
        if "table_info" in variable:
            attrs.setdefault("table_info", variable["table_info"])
        attrs.setdefault("variant_label", self.variant_label())
        attrs.setdefault(
            "creation_date", datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        )
        if extra_attrs:
            attrs.update(_MetadataRecord.netcdf_attrs(extra_attrs))
        return attrs

    def variant_label(self) -> str:
        """Return the explicit or RIPF-derived variant label."""

        if self.get("variant_label"):
            return str(self["variant_label"])
        values = [self.get(key) for key in RIPF_KEYS]
        if all(value not in (None, "") for value in values):
            return "".join(str(value) for value in values)
        return "r1i1p1f1"

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
