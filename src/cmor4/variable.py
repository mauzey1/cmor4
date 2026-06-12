from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ._table_utils import (
    is_table_value,
    metadata_value_matches,
    parse_table_value,
    single_or_original,
    table_dimensions,
)
from .exceptions import TableValidationError
from .metadata import _MetadataRecord


@dataclass(frozen=True)
class VariableEntry:
    """Resolved variable table entry.

    Parameters
    ----------
    name:
        Name of the variable entry in the table.
    table_id:
        Identifier of the table that supplied the entry.
    entry:
        Raw variable-entry metadata from the table.
    table_file:
        Path to the table file, if available.
    table_header:
        Header metadata from the table file, if available.
    """

    name: str
    table_id: str
    entry: Mapping[str, Any]
    table_file: Path | None = None
    table_header: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Variable(_MetadataRecord):
    """Metadata for the data variable being written.

    Parameters
    ----------
    name:
        Requested variable name or branded variable name.
    id, variable_id:
        Output variable identifier overrides.
    table_id:
        Table identifier used to disambiguate variables.
    dimensions:
        Ordered logical variable dimensions.
    units, standard_name, long_name, cell_methods, cell_measures, comment:
        NetCDF variable metadata attributes.
    missing_value, fill_value:
        Missing-value marker written to attributes and encoding.
    chunksizes, chunks:
        Optional NetCDF chunk sizes.
    coordinates:
        Explicit ``coordinates`` attribute value.
    formula_terms:
        Explicit formula-term attribute value.
    frequency, realm, table_info:
        Table-derived metadata that may also be copied to global attributes.
    valid_min, valid_max, ok_min_mean_abs, ok_max_mean_abs:
        Value validation limits.
    attrs:
        Additional NetCDF attributes for the data variable.
    extra:
        Additional mapping keys preserved by the metadata record.
    project:
        Optional project tables used to resolve and merge variable metadata
        during construction.
    """

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
    valid_min: Any = None
    valid_max: Any = None
    ok_min_mean_abs: Any = None
    ok_max_mean_abs: Any = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)
    project: InitVar[Any | None] = None

    def __post_init__(self, project: Any | None) -> None:
        if project is None:
            return
        variable_entry = self.resolve_table_entry(project)
        merged = self._merge_table_entry(variable_entry)
        # Note: Full validation happens in ProjectTables._dataset_for_variable
        # and validate_components, not during construction
        for key, value in merged.to_dict().items():
            object.__setattr__(self, key, value)

    def names(self) -> tuple[str, dict[str, str]]:
        """Return the output variable id and branded-label metadata.

        This method parses the variable name to extract the base variable ID
        and any branding suffix components. Branded variable names follow the
        pattern ``<variable_id>_<temporal>-<vertical>-<horizontal>-<area>``,
        where the suffix is optional and components can be omitted.

        Returns
        -------
        tuple[str, dict[str, str]]
            A tuple containing:

            - variable_id (str): The base variable identifier (e.g., "tas")
            - labels (dict): Dictionary with the following keys:

              - "branded_name": Full branded variable name
              - "variable_id": Base variable identifier
              - "branding_suffix": Optional suffix after first underscore
              - "temporal_label": Optional first component of suffix
              - "vertical_label": Optional second component of suffix
              - "horizontal_label": Optional third component of suffix
              - "area_label": Optional fourth component of suffix

        Examples
        --------
        Simple variable without branding::

            variable = Variable(name="tas")
            var_id, labels = variable.names()
            # var_id = "tas"
            # labels = {"branded_name": "tas", "variable_id": "tas"}

        Fully branded variable::

            variable = Variable(name="tas_ann-lev-reg-mean")
            var_id, labels = variable.names()
            # var_id = "tas"
            # labels = {
            #     "branded_name": "tas_ann-lev-reg-mean",
            #     "variable_id": "tas",
            #     "branding_suffix": "ann-lev-reg-mean",
            #     "temporal_label": "ann",
            #     "vertical_label": "lev",
            #     "horizontal_label": "reg",
            #     "area_label": "mean"
            # }

        Partial branding::

            variable = Variable(name="pr_day")
            var_id, labels = variable.names()
            # var_id = "pr"
            # labels = {
            #     "branded_name": "pr_day",
            #     "variable_id": "pr",
            #     "branding_suffix": "day",
            #     "temporal_label": "day"
            # }
        """

        branded_name = str(
            self.get("name") or self.get("id") or self.get("variable_id")
        )
        variable_id = str(
            self.get("id")
            or self.get("variable_id")
            or branded_name.split("_", 1)[0]
        )
        labels = {"branded_name": branded_name, "variable_id": variable_id}
        if "_" in branded_name:
            suffix = branded_name.split("_", 1)[1]
            labels["branding_suffix"] = suffix
            parts = suffix.split("-")
            for key, value in zip(
                (
                    "temporal_label",
                    "vertical_label",
                    "horizontal_label",
                    "area_label",
                ),
                parts,
            ):
                labels[key] = value
        return variable_id, labels

    def resolve_table_entry(self, project: Any) -> VariableEntry:
        """Find a variable table entry by branded name or variable name.

        This method searches the project's loaded variable tables for an entry
        matching this variable. It first looks for an exact match by name, then
        falls back to matching by out_name. If multiple tables contain the same
        variable, ``table_id`` must be specified to disambiguate.

        Parameters
        ----------
        project
            Project table loader containing loaded variable entries from one or
            more variable tables.

        Returns
        -------
        VariableEntry
            The resolved variable table entry containing the variable's
            metadata, dimensions, table_id, and table header information.

        Raises
        ------
        TableValidationError
            If the variable name is not found in any loaded table, if the name
            is ambiguous across multiple tables without a table_id, or if the
            specified table_id doesn't contain the variable.

        Examples
        --------
        Resolve a unique variable::

            variable = Variable(name="tas")
            entry = variable.resolve_table_entry(project)
            # Returns the single matching VariableEntry

        Disambiguate with table_id::

            variable = Variable(name="tas", table_id="Amon")
            entry = variable.resolve_table_entry(project)
            # Returns the VariableEntry from the Amon table specifically

        Resolve by out_name::

            variable = Variable(name="tos")  # out_name in table
            entry = variable.resolve_table_entry(project)
            # Matches variable entry whose out_name is "tos"
        """

        requested = str(self.name or self.variable_id or self.id or "")
        entries_by_name = project._variable_entries_by_name
        if requested in entries_by_name:
            entries = entries_by_name[requested]
            if self.table_id:
                matches = [
                    entry
                    for entry in entries
                    if entry.table_id == str(self.table_id)
                ]
                if len(matches) == 1:
                    return matches[0]
                raise TableValidationError(
                    f"Variable {requested!r} was not found in table "
                    f"{self.table_id!r}."
                )
            if len(entries) == 1:
                return entries[0]
            choices = ", ".join(
                f"{entry.table_id}:{entry.name}" for entry in entries
            )
            raise TableValidationError(
                f"Variable {requested!r} is ambiguous across loaded tables; "
                "specify table_id. "
                f"Choices: {choices}."
            )
        matches = [
            entry
            for entry in project.variable_entries.values()
            if str(entry.entry.get("out_name", entry.name)) == requested
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            names = ", ".join(match.name for match in matches[:10])
            raise TableValidationError(
                f"Variable {requested!r} is ambiguous; use one of: {names}."
            )
        raise TableValidationError(
            f"Variable {requested!r} was not found in loaded variable tables."
        )

    def _merge_table_entry(self, variable_entry: VariableEntry) -> "Variable":
        """Merge authoritative table metadata with user variable controls.

        Parameters
        ----------
        variable_entry:
            Resolved table entry to merge into this variable.

        Returns
        -------
        Variable
            New variable metadata record with table defaults applied.
        """

        entry = variable_entry.entry
        merged = self.to_dict()
        table_dims = table_dimensions(entry)
        merged.setdefault("name", variable_entry.name)
        merged.setdefault(
            "id", entry.get("out_name", variable_entry.name.split("_", 1)[0])
        )
        merged.setdefault("variable_id", merged["id"])
        merged.setdefault("dimensions", table_dims)
        merged.setdefault(
            "table_id", entry.get("table_id", variable_entry.table_id)
        )
        if variable_entry.table_file is not None:
            merged.setdefault(
                "table_info", f"Name: {variable_entry.table_file.name};"
            )
        if "frequency" in entry:
            merged.setdefault("frequency", entry["frequency"])
        if "modeling_realm" in entry:
            merged.setdefault(
                "realm", single_or_original(entry["modeling_realm"])
            )
        for key in (
            "valid_min",
            "valid_max",
            "ok_min_mean_abs",
            "ok_max_mean_abs",
        ):
            value = entry.get(key)
            if not is_table_value(value) and variable_entry.table_header:
                value = variable_entry.table_header.get(key)
            if is_table_value(value):
                merged.setdefault(key, parse_table_value(value))
        for key in (
            "units",
            "standard_name",
            "long_name",
            "cell_methods",
            "cell_measures",
            "comment",
        ):
            if entry.get(key) not in (None, ""):
                merged[key] = entry[key]
        return Variable.from_mapping(merged)

    def attributes(self, labels: Mapping[str, str]) -> dict[str, Any]:
        """Return NetCDF attributes for this data variable.

        This method constructs the complete set of NetCDF attributes for the
        main data variable by combining CF-compliant metadata (units,
        standard_name, long_name, cell_methods, cell_measures) with branding
        labels and any extra user-provided attributes.

        Parameters
        ----------
        labels
            Branded variable labels dictionary returned by :meth:`names`,
            containing "branded_name" and optional branding suffix components.

        Returns
        -------
        dict[str, Any]
            NetCDF-safe variable attributes suitable for assignment to an
            xarray DataArray or NetCDF variable. Includes standard CF
            attributes plus branded_variable_name and optional label
            components.

        Notes
        -----
        The following attributes are included if defined in the variable:

        - units, standard_name, long_name (CF-required metadata)
        - cell_methods, cell_measures, comment (CF-optional metadata)
        - branded_variable_name (always included from labels)
        - branding_suffix, temporal_label, vertical_label, horizontal_label,
          area_label (included if present in labels)

        Additional attributes from the ``attrs`` field are merged first,
        allowing standard attributes to override them.

        Examples
        --------
        Get attributes for a variable::

            variable = project.variable("tas")
            var_id, labels = variable.names()
            attrs = variable.attributes(labels)
            # attrs = {
            #     "units": "K",
            #     "standard_name": "air_temperature",
            #     "long_name": "Near-Surface Air Temperature",
            #     "branded_variable_name": "tas",
            #     ...
            # }
        """

        attrs = self.netcdf_attrs(self.attrs)
        for key in (
            "units",
            "standard_name",
            "long_name",
            "cell_methods",
            "cell_measures",
            "comment",
        ):
            if key in self:
                attrs[key] = self[key]
        attrs.setdefault("branded_variable_name", labels["branded_name"])
        for key in (
            "branding_suffix",
            "temporal_label",
            "vertical_label",
            "horizontal_label",
            "area_label",
        ):
            if key in labels:
                attrs.setdefault(key, labels[key])
        return attrs

    def validate_against_entry(self, variable_entry: VariableEntry) -> None:
        """Validate variable metadata against a variable table entry.

        This method checks that user-provided variable metadata is
        consistent with the resolved table entry. It validates the variable
        ID, dimensions, CF attributes (units, standard_name, etc.), and
        table-derived metadata (frequency, realm, table_id).

        Parameters
        ----------
        variable_entry
            The resolved variable table entry containing authoritative metadata
            from the project variable tables.

        Raises
        ------
        TableValidationError
            If any of the following validation checks fail:

            - variable_id or id doesn't match table out_name
            - dimensions don't match table dimensions
            - units, standard_name, long_name, cell_methods, cell_measures, or
              comment don't match table values (when both are specified)
            - frequency, realm, or table_id don't match expected values

        Notes
        -----
        This validation is performed automatically by ``create_dataset`` and
        ``validate_components``. It can also be called directly to verify
        metadata before dataset creation.

        Examples
        --------
        Validate variable metadata::

            project = ProjectTables(...)
            variable = Variable(name="tas", dimensions=("time", "lat", "lon"))
            entry = variable.resolve_table_entry(project)
            variable.validate_against_entry(entry)
            # Raises TableValidationError if dimensions don't match

        Validate with overridden metadata::

            variable = Variable(
                name="tas",
                units="degC"  # Table requires "K"
            )
            entry = variable.resolve_table_entry(project)
            variable.validate_against_entry(entry)
            # Raises: TableValidationError: units='degC' does not match
            # Amon:tas value 'K'
        """

        entry = variable_entry.entry
        values = self.to_dict()
        out_name = str(
            entry.get("out_name", variable_entry.name.split("_", 1)[0])
        )
        for key in ("id", "variable_id"):
            if key in values and str(values[key]) != out_name:
                raise TableValidationError(
                    f"{key}={values[key]!r} does not match table "
                    f"out_name {out_name!r}."
                )
        expected_dims = table_dimensions(entry)
        if (
            self.dimensions is not None
            and tuple(self.dimensions) != expected_dims
        ):
            raise TableValidationError(
                f"dimensions={tuple(self.dimensions)!r} does not match "
                f"{variable_entry.table_id}:{variable_entry.name} "
                f"dimensions {expected_dims!r}."
            )
        for key in (
            "units",
            "standard_name",
            "long_name",
            "cell_methods",
            "cell_measures",
            "comment",
        ):
            expected = entry.get(key)
            if (
                expected not in (None, "")
                and key in values
                and str(values[key]) != str(expected)
            ):
                raise TableValidationError(
                    f"{key}={values[key]!r} does not match "
                    f"{variable_entry.table_id}:{variable_entry.name} "
                    f"value {expected!r}."
                )
        expected_values = {
            "frequency": entry.get("frequency"),
            "realm": entry.get("modeling_realm"),
            "table_id": variable_entry.table_id,
        }
        for key, expected in expected_values.items():
            if (
                expected not in (None, "")
                and key in values
                and not metadata_value_matches(values[key], expected)
            ):
                raise TableValidationError(
                    f"{key}={values[key]!r} does not match "
                    f"{variable_entry.table_id}:{variable_entry.name} "
                    f"value {expected!r}."
                )
