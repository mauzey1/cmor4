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

    Parameters
    ----------
    data:
        Prepared dataset metadata values.
    project:
        Project table loader that prepared the metadata, if any.
    user_info:
        Original user-provided dataset metadata.
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
        """Create dataset info directly from user metadata.

        This is a convenience constructor that creates a ``DatasetInfo``
        record from a dictionary-like mapping of dataset attributes. When
        created without project tables, the dataset info contains only the
        user-provided values. When created with project tables, use
        ``ProjectTables.dataset_info()`` instead for CV validation and
        defaults.

        Parameters
        ----------
        values
            Dataset metadata values containing global attributes like
            mip_era, institution_id, source_id, experiment_id, etc.
        project
            Optional project table loader. If provided, associates this
            dataset with project-specific validation but does not apply
            CV defaults.

        Returns
        -------
        DatasetInfo
            Mapping-compatible dataset metadata record.

        See Also
        --------
        ProjectTables.dataset_info : Recommended method for creating dataset
            info with project table validation and defaults.

        Examples
        --------
        Create dataset info without project tables::

            dataset = DatasetInfo.from_mapping({
                "mip_era": "CMIP7",
                "institution_id": "NCAR",
                "source_id": "CESM2"
            })
        """

        return cls(
            values,
            project=project,
            user_info=values,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable copy of the prepared dataset metadata.

        This method creates a new dictionary containing all dataset metadata,
        including user-provided values, project CV defaults, and any runtime
        attributes that have been added. The returned dictionary is independent
        of the ``DatasetInfo`` record and can be freely modified.

        Returns
        -------
        dict[str, Any]
            Dataset metadata as a new mutable dictionary containing all
            key-value pairs from the prepared dataset data.

        Examples
        --------
        Create a modified copy of dataset metadata::

            dataset = project.dataset_info({"mip_era": "CMIP7"})
            attrs = dataset.to_dict()
            attrs["custom_field"] = "custom_value"
        """

        return dict(self.data)

    def global_attributes(
        self,
        variable: Any,
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return NetCDF global attributes for this dataset and variable.

        This method constructs the complete set of global attributes for a
        NetCDF output file by merging dataset metadata, variable-derived
        attributes (like variable_id, branded_variable, frequency, realm),
        runtime defaults (like creation_date), and user-provided overrides.
        Internal dataset keys (those in ``INTERNAL_DATASET_KEYS`` or starting
        with underscore) are excluded from the output.

        Parameters
        ----------
        variable
            Variable metadata record whose name, labels, frequency, realm, and
            table_id are extracted and merged into global attributes.
        extra_attrs
            Optional mapping of additional global attributes that override any
            generated or default values. Only NetCDF-compatible values are
            included.

        Returns
        -------
        dict[str, Any]
            Complete set of NetCDF-safe global attributes ready to be assigned
            to an xarray Dataset or written to a NetCDF file.

        Notes
        -----
        The following attributes are always included:

        - ``Conventions``: Defaults to "CF-1.11"
        - ``cmor4_version``: Package version
        - ``creation_date``: Timestamp when attributes were generated
        - ``variant_label``: From dataset or derived from RIPF indices

        Variable-derived attributes (variable_id, frequency, realm, etc.) are
        added with ``setdefault`` so dataset values take precedence.

        Examples
        --------
        Get global attributes for a dataset and variable::

            dataset = project.dataset_info({...})
            variable = project.variable("tas")
            attrs = dataset.global_attributes(variable)
            # attrs includes mip_era, institution_id, variable_id, frequency,
            # creation_date, etc.

        Override generated attributes::

            attrs = dataset.global_attributes(
                variable,
                extra_attrs={"comment": "Custom processing applied"}
            )
        """

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
        """Return the explicit or RIPF-derived variant label.

        The variant label identifies the specific ensemble member and is
        constructed from realization, initialization, physics, and forcing
        indices. This method first checks for an explicit ``variant_label``
        attribute, then attempts to construct one from RIPF indices, and
        finally falls back to the default.

        Returns
        -------
        str
            The variant label string. Returns the explicit ``variant_label`` if
            present, a label constructed from ``realization_index``,
            ``initialization_index``, ``physics_index``, and ``forcing_index``
            if all four are defined, or the default ``"r1i1p1f1"`` otherwise.

        Examples
        --------
        Explicit variant label::

            dataset = DatasetInfo({"variant_label": "r3i1p2f1"})
            dataset.variant_label()  # Returns "r3i1p2f1"

        Constructed from RIPF indices::

            dataset = DatasetInfo({
                "realization_index": "r5",
                "initialization_index": "i2",
                "physics_index": "p1",
                "forcing_index": "f3"
            })
            dataset.variant_label()  # Returns "r5i2p1f3"

        Default when not specified::

            dataset = DatasetInfo({})
            dataset.variant_label()  # Returns "r1i1p1f1"
        """

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
