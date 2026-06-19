"""DatasetInfo — dataset-level metadata record."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .metadata import MetadataModel

# Keys that must not appear in NetCDF global attributes
INTERNAL_DATASET_KEYS: frozenset[str] = frozenset(
    {
        "_history_template",
        "create_subdirectories",
        "outpath",
        "output_file_template",
        "output_path_template",
        "tracking_prefix",
    }
)

RIPF_KEYS: tuple[str, ...] = (
    "realization_index",
    "initialization_index",
    "physics_index",
    "forcing_index",
)

# Expected single-character prefix for each RIPF key (e.g. "r9", "i1", …).
_RIPF_PREFIXES: dict[str, str] = {
    "realization_index": "r",
    "initialization_index": "i",
    "physics_index": "p",
    "forcing_index": "f",
}
# CMOR3 upper bound: 32-bit signed integer.
_RIPF_MAX: int = 2**31 - 1


class DatasetInfo(BaseModel, Mapping[str, Any]):
    """Prepared dataset-level metadata.

    Implements the :class:`~collections.abc.Mapping` protocol so it can be
    used wherever a plain metadata dict is expected.  The Mapping view
    exposes all CV attributes but excludes the internal ``project`` and
    ``user_info`` references.

    Well-known CMIP DRS attributes are declared as typed fields.  Any
    additional project-specific or user-provided attributes are stored via
    ``extra="allow"`` and appear in the Mapping view.

    Parameters
    ----------
    mip_era, activity_id, institution_id, source_id, experiment_id,
    variant_label, grid_label, license_id:
        Standard CMIP DRS global attributes.
    realization_index, initialization_index, physics_index, forcing_index:
        RIPF ensemble indices.  Accepts CMIP7-style prefixed strings
        (``"r9"``, ``"i1"``), bare integers (``9``, ``1``), or string
        integers (``"9"``).  Values are validated to be positive and ≤
        INT32_MAX at construction time so downstream code can rely on their
        format.
    calendar:
        CF calendar name for the time axis.
    frequency:
        Data frequency string (e.g. ``"mon"``).
    nominal_resolution:
        Approximate horizontal resolution description.
    outpath:
        Output directory for NetCDF files (internal, not written to file).
    version:
        Dataset version string.
    project:
        :class:`~cmor4.tables.ProjectTables` that prepared this record.
        Excluded from the Mapping view and serialisation.
    user_info:
        Original user-supplied metadata.
        Excluded from the Mapping view and serialisation.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
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

    # Internal references — excluded from Mapping / dump
    project: Any = Field(default=None, exclude=True, repr=False)
    user_info: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    # ------------------------------------------------------------------
    # Positional-dict __init__ shim for backward compatibility.
    # Pydantic's BaseModel.__init__ does not accept positional non-self
    # args, so we keep a one-line override that merges the optional dict
    # into kwargs, then calls super().__init__ so all field_validators
    # (including the RIPF checks below) run in the normal way.
    # ------------------------------------------------------------------

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        """Accept an optional positional *data* mapping for backward compatibility.

        Merges *data* into *kwargs* and calls ``super().__init__`` so all
        Pydantic field validators (including the RIPF checks below) run
        in the normal way.

        The original ``DatasetInfo`` was a dataclass whose first field was
        ``data: Mapping[str, Any]``, allowing::

            DatasetInfo({...}, project=project)

        Use :meth:`from_mapping` for new code; this shim exists only to
        preserve the historic calling convention.
        """
        merged = {**dict(data), **kwargs} if data is not None else kwargs
        super().__init__(**merged)

    # ------------------------------------------------------------------
    # RIPF field validators — validate format and range at construction
    # time rather than deferring to CV validation.
    # ------------------------------------------------------------------

    @field_validator(
        "realization_index",
        "initialization_index",
        "physics_index",
        "forcing_index",
        mode="before",
    )
    @classmethod
    def _validate_ripf_index(cls, v: Any, info: Any) -> str | int | None:
        """Validate RIPF index format and range.

        Accepts CMIP7-style prefixed strings (``"r9"``), bare integers
        (``9``), and string integers (``"9"``).  Rejects non-positive
        values, values exceeding INT32_MAX, and non-numeric strings.
        """
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

    # ------------------------------------------------------------------
    # Internal data view
    # ------------------------------------------------------------------

    def _data(self) -> dict[str, Any]:
        """All non-None, non-excluded field values plus extra keys."""
        result: dict[str, Any] = {}
        for name, info in type(self).model_fields.items():
            if getattr(info, "exclude", False):
                continue
            val = getattr(self, name)
            if val is None:
                continue
            result[name] = val
        for key, val in (self.model_extra or {}).items():
            if val is not None:
                result.setdefault(key, val)
        return result

    # Mapping protocol -----------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())

    # Factory --------------------------------------------------------

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        project: Any = None,
    ) -> DatasetInfo:
        """Create DatasetInfo from a user metadata mapping.

        Parameters
        ----------
        values:
            Dataset metadata (global attributes, internal controls …).
        project:
            Optional :class:`~cmor4.tables.ProjectTables` to attach.
        """
        data = dict(values)
        data["project"] = project
        data["user_info"] = dict(values)
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Mutable copy of the Mapping view."""
        return self._data()

    # Domain helpers -------------------------------------------------

    def variant_label(self) -> str:
        """Return the explicit or RIPF-derived variant label.

        Priority: explicit ``variant_label`` → constructed from RIPF
        indices → default ``"r1i1p1f1"``.
        """
        vl = self._data().get("variant_label")
        if vl:
            return str(vl)
        vals = [self._data().get(k) for k in RIPF_KEYS]
        if all(v not in (None, "") for v in vals):
            return "".join(str(v) for v in vals)
        return "r1i1p1f1"

    def global_attributes(
        self,
        variable: Any,
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return complete NetCDF global attributes for this dataset and variable.

        Parameters
        ----------
        variable:
            A :class:`~cmor4.variable.Variable` whose names, frequency,
            realm, and table_id are merged in.
        extra_attrs:
            Additional overrides applied last.
        """
        from . import __version__ as _ver

        creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        attrs: dict[str, Any] = {
            "Conventions": self._data().get("Conventions", "CF-1.12"),
            "cmor_version": _ver,
        }
        for key, val in self._data().items():
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
        mip_era = attrs.get("mip_era") or self._data().get("mip_era", "CMIP")
        attrs.setdefault(
            "history",
            f"{creation_date} ; CMOR rewrote data to be consistent with "
            f"{conv} and {mip_era} data requirements.",
        )
        src = attrs.get("source_id") or self._data().get("source_id", "")
        if src:
            attrs.setdefault("title", f"{src} output prepared for {mip_era}")

        if extra_attrs:
            attrs.update(MetadataModel.netcdf_attrs(extra_attrs))
        return attrs
