from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._table_utils import is_table_value
from .metadata import _MetadataRecord


@dataclass(frozen=True)
class Grid(_MetadataRecord):
    """Runtime grid dimensions and optional grid-mapping metadata."""

    dimensions: tuple[str, ...] | list[str] | None = None
    name: str | None = None
    table_entry: str | None = None
    mapping_entry: str | None = None
    mapping_var: str | None = None
    mapping_name: str | None = None
    grid_mapping_name: str | None = None
    coordinates: tuple[str, ...] | list[str] | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def merge_table_entry(self, project: Any) -> "Grid":
        """Merge grid-mapping metadata from the loaded grids table."""

        merged = self.to_dict()
        entry_name, entry = self.resolve_table_entry(project)
        if entry is None:
            return Grid.from_mapping(merged)
        merged.setdefault("table_entry", entry_name)
        coordinates = entry.get("coordinates")
        if "coordinates" not in merged and is_table_value(coordinates):
            merged["coordinates"] = str(coordinates).split()
        params = dict(merged.get("params", {}))
        for key, value in entry.items():
            if not key.startswith("parameter") or not is_table_value(value):
                continue
            params.setdefault(str(value), merged.get(str(value), 0.0))
        if params:
            merged["params"] = params
        return Grid.from_mapping(merged)

    def resolve_table_entry(
        self, project: Any
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        """Resolve a grid mapping entry from this grid definition."""

        requested = str(
            self.table_entry
            or self.mapping_entry
            or self.name
            or ""
        )
        if requested in project.grid_mapping_entries:
            return requested, project.grid_mapping_entries[requested]
        return None, None

    @property
    def variable_name(self) -> str:
        return str(self.mapping_var or "crs")

    def variable_dimensions(self, variable: Any) -> tuple[str, ...] | None:
        if self.dimensions:
            return tuple(str(name) for name in self.dimensions)
        dimensions = variable.get("dimensions")
        if dimensions:
            return tuple(str(name) for name in dimensions)
        return None

    @property
    def has_mapping(self) -> bool:
        return bool(
            self.mapping_name
            or self.grid_mapping_name
            or self.params
            or self.attrs
        )

    def mapping_attributes(self) -> dict[str, Any]:
        attrs = dict(self.attrs)
        mapping_name = self.mapping_name or self.grid_mapping_name
        if mapping_name:
            attrs["grid_mapping_name"] = mapping_name
        for key, value in self.params.items():
            if isinstance(value, (list, tuple)) and value:
                attrs[key] = value[0]
                if len(value) > 1 and value[1]:
                    attrs[f"{key}_units"] = value[1]
            else:
                attrs[key] = value
        return attrs
