"""Public project-table orchestration over private table and domain modules."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from ._coordinate_tables import CoordinateTable, FormulaTable, GridTable
from ._metadata import DatasetMetadata, warn_tracking_id_ignored
from ._templates import is_unresolved_template as _is_unresolved_template
from ._variable_tables import VariableEntry, VariableTable
from .axis import (
    Axis,
    validate_axes,
    validate_axis_values_early,
)
from .cv import ControlledVocabulary
from .datasetinfo import DatasetInfo
from .exceptions import ControlledVocabularyError, TableValidationError
from .grid import Grid, validate_grid_mapping
from .variable import Variable
from .zfactor import ZFactor


class ProjectTables:
    """Load a project's CV and tables and build metadata objects from them.

    The methods :meth:`dataset_info`, :meth:`variable`, :meth:`axis`,
    :meth:`grid`, and :meth:`zfactor` fill in metadata from the tables and
    check what you supply. Pass the results to :func:`cmorize` or
    :class:`DatasetWriter`.

    Most code creates this object with :meth:`from_directory`. See
    :doc:`/usage/basic_usage` for an example.

    Parameters
    ----------
    cv_file
        Path to the project controlled vocabulary (CV) JSON file.
    variable_tables
        Paths to one or more variable table JSON files.
    coordinate_table
        Path to the coordinate table JSON file.
    formula_table
        Path to the formula-terms table JSON file. Needed for hybrid
        vertical coordinates.
    grid_table
        Path to the grids table JSON file. Needed for curvilinear and
        projected grids.
    long_name_override_table
        Path to a table that sets ``long_name`` per context (for example,
        by realm, frequency, or region).
    cell_measures_table
        Path to a table that sets ``cell_measures`` per context. An empty
        string in this table removes the attribute.
    """

    def __init__(
        self,
        cv_file: str | Path,
        variable_tables: Sequence[str | Path],
        coordinate_table: str | Path | None = None,
        formula_table: str | Path | None = None,
        grid_table: str | Path | None = None,
        long_name_override_table: str | Path | None = None,
        cell_measures_table: str | Path | None = None,
    ):
        self.cv_file = Path(cv_file)
        self.variable_table_files = tuple(Path(path) for path in variable_tables)
        self.coordinate_table_file = (
            Path(coordinate_table) if coordinate_table is not None else None
        )
        self.formula_table_file = (
            Path(formula_table) if formula_table is not None else None
        )
        self.grid_table_file = Path(grid_table) if grid_table is not None else None
        self.long_name_override_table_file = (
            Path(long_name_override_table)
            if long_name_override_table is not None
            else None
        )
        self.cell_measures_table_file = (
            Path(cell_measures_table) if cell_measures_table is not None else None
        )
        self.cv = ControlledVocabulary.from_file(self.cv_file)
        self.variable_table = VariableTable.from_file(
            self.variable_table_files,
            long_name_override_table=self.long_name_override_table_file,
            cell_measures_table=self.cell_measures_table_file,
        )
        self.grid_table = GridTable.from_file(self.grid_table_file)
        self.coordinate_table = CoordinateTable.from_file(
            self.coordinate_table_file, self.grid_table
        )
        self.formula_table = FormulaTable.from_file(self.formula_table_file)

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        cv_file: str | Path,
        variable_tables: Sequence[str | Path],
        coordinate_table: str | Path | None = None,
        formula_table: str | Path | None = None,
        grid_table: str | Path | None = None,
        long_name_override_table: str | Path | None = None,
        cell_measures_table: str | Path | None = None,
    ) -> "ProjectTables":
        """Load a project's tables from paths relative to a root directory.

        Paths you give are relative to ``root``. If you omit an optional
        table, CMOR looks for it in ``root/tables`` or ``root/Tables``
        by its file-name suffix (for example, ``*_coordinate.json``).

        Parameters
        ----------
        root
            Root directory of the project tables.
        cv_file
            CV file path, relative to ``root``.
        variable_tables
            Variable table paths, relative to ``root``.
        coordinate_table
            Coordinate table path, relative to ``root``.
        formula_table
            Formula-terms table path, relative to ``root``.
        grid_table
            Grids table path, relative to ``root``.
        long_name_override_table
            ``long_name`` overrides table path, relative to ``root``.
        cell_measures_table
            ``cell_measures`` table path, relative to ``root``.

        Returns
        -------
        ProjectTables
            The loaded tables.
        """

        root_path = Path(root)
        resolved_coordinate_table = _resolve_optional_table(
            root_path, coordinate_table, "coordinate"
        )
        resolved_formula_table = _resolve_optional_table(
            root_path, formula_table, "formula_terms"
        )
        resolved_grid_table = _resolve_optional_table(root_path, grid_table, "grids")
        resolved_long_name_override_table = _resolve_optional_table(
            root_path, long_name_override_table, "long_name_overrides"
        )
        resolved_cell_measures_table = _resolve_optional_table(
            root_path, cell_measures_table, "cell_measures"
        )
        return cls(
            root_path / cv_file,
            [root_path / table_file for table_file in variable_tables],
            coordinate_table=resolved_coordinate_table,
            formula_table=resolved_formula_table,
            grid_table=resolved_grid_table,
            long_name_override_table=resolved_long_name_override_table,
            cell_measures_table=resolved_cell_measures_table,
        )

    def dataset_info(
        self,
        dataset: Mapping[str, Any] | DatasetMetadata,
    ) -> DatasetInfo:
        """Build dataset metadata, filling defaults from the CV.

        CMOR fills attributes the CV can derive (for example, from
        ``source_id`` or ``experiment_id``) and checks the values you give.
        Attributes that depend on the variable are checked later, when the
        file is written. See :doc:`/usage/controlled_vocabularies`.

        Parameters
        ----------
        dataset
            Global attributes and output options, as a mapping.

        Returns
        -------
        DatasetInfo
            Dataset metadata to pass to :func:`cmorize` or
            :class:`DatasetWriter`.

        Raises
        ------
        ControlledVocabularyError
            If a value is not allowed by the CV.
        """

        user_info = (
            dataset.to_dict() if isinstance(dataset, DatasetMetadata) else dict(dataset)
        )
        if user_info.pop("tracking_id", None) is not None:
            warn_tracking_id_ignored("dataset_info()")
        try:
            metadata = DatasetMetadata.from_mapping(user_info)
        except ValidationError as exc:
            raise ControlledVocabularyError(str(exc)) from exc
        normalized_dataset = self.cv.get_dataset_info(metadata)
        self.cv.validate_dataset_info(normalized_dataset, check_required=False)
        return DatasetInfo.from_prepared(
            normalized_dataset,
            project=self,
            user_info=user_info,
        )

    def _dataset_for_variable(
        self,
        dataset_info: DatasetMetadata,
        variable: Variable,
    ) -> tuple[DatasetInfo, Variable]:
        """Prepare dataset info with variable-specific metadata and validation.

        This is called by cmorize to merge variable metadata into
        dataset and perform initial validation. Full component validation
        happens later during dataset validation. If contextual remapping
        tables are loaded, matching ``long_name`` and ``cell_measures`` values
        are applied to the returned ``Variable`` and used as the expected table
        values for validation.

        Returns
        -------
        tuple[DatasetInfo, Variable]
            Prepared dataset metadata and variable, with contextual remappings
            applied when a key matches the dataset and variable context.
        """
        normalized_dataset = self.cv.get_dataset_info(dataset_info)
        variable_entry = self.variable_table.resolve(variable)
        variable_entry = self.variable_table.contextual_entry(
            variable_entry, variable, normalized_dataset
        )
        variable = self.variable_table.apply_contextual_metadata(
            variable, variable_entry
        )

        normalized_data = normalized_dataset.to_dict()
        self._add_table_header_defaults(normalized_data, variable_entry)
        self._add_variable_global_defaults(normalized_data, variable)
        normalized_dataset = DatasetMetadata.from_mapping(normalized_data)
        self.cv.validate_dataset_info(normalized_dataset)

        # Note: Full variable and dataset-variable consistency validation
        # happens in validate_components, not here
        user_info = getattr(dataset_info, "user_info", None) or dataset_info.to_dict()
        prepared_dataset = DatasetInfo.from_prepared(
            normalized_dataset,
            project=self,
            user_info=dict(user_info),
        )

        self.variable_table.validate_against(variable, variable_entry)

        return prepared_dataset, variable

    def variable(self, name: str, **values: Any) -> Variable:
        """Build variable metadata from the variable tables.

        Table values (units, dimensions, ``standard_name``, and so on) fill in
        anything you do not supply.

        Parameters
        ----------
        name
            Variable name (``"tas"``) or branded variable name
            (``"tas_tavg-h2m-hxy-u"``).
        **values
            Any :class:`Variable` field, such as ``table_id`` (to choose a
            table when the name is in more than one), ``missing_value``, or
            ``attrs``.

        Returns
        -------
        Variable
            Variable metadata.

        Raises
        ------
        TableValidationError
            If the name is not in the tables, is in more than one table and
            ``table_id`` is not given, or a value conflicts with the table.

        See Also
        --------
        :doc:`/usage/basic_usage` : Examples, including missing values.
        """

        request = Variable.model_validate({"name": name, **values})
        return self.variable_table.build(request)

    def axis(self, name: str, **values: Any) -> Axis:
        """Build an axis from the coordinate tables.

        Table values (units, ``standard_name``, valid range, and so on) fill
        in anything you do not supply, and CMOR checks the coordinate values
        and bounds.

        Parameters
        ----------
        name
            Coordinate table entry name, such as ``"time"``, ``"latitude"``,
            ``"plev19"``, or a generic level such as ``"alevel"``.
        **values
            Any :class:`Axis` field. Usually ``values`` and ``bounds``. Time
            axes also need ``units`` with a reference date, and may set
            ``calendar``.

        Returns
        -------
        Axis
            Axis metadata and values.

        Raises
        ------
        TableValidationError
            If ``name`` matches more than one generic level entry, or a value
            conflicts with the table.
        AxisValidationError
            If the values or bounds are invalid (for example, not monotonic or
            out of range).

        See Also
        --------
        :doc:`/usage/basic_usage` : Basic axes.
        :doc:`/usage/time_coordinates` : Time axes and calendars.
        :doc:`/usage/vertical_coordinates` : Pressure and hybrid levels.
        """

        request = Axis.model_validate({"name": name, **values})
        axis = self.coordinate_table.build(request)
        validate_axis_values_early(axis)
        return self._mark_prepared_axis(axis)

    def _axes(
        self,
        axes: Sequence[Axis],
        variable: Variable | None = None,
    ) -> tuple[Axis, ...]:
        """Create a complete axis tuple, including required scalar axes.

        Axes created via this ProjectTables instance (using axis() factory
        method or with project=self) are already prepared. Axes from other
        sources are merged with this ProjectTables instance to ensure
        consistent table data.
        """

        merged_axes = [
            (
                axis
                if self._is_prepared_axis(axis)
                else self.coordinate_table.build(axis)
            )
            for axis in axes
        ]
        if variable is not None:
            merged_axes.extend(self._scalar_axes_for(variable, merged_axes))
        return tuple(self._mark_prepared_axis(axis) for axis in merged_axes)

    def _scalar_axes_for(
        self,
        variable: Variable,
        axes: Sequence[Axis] = (),
    ) -> tuple[Axis, ...]:
        """Return fixed scalar axes required by a variable and not supplied.

        Scalar axes are coordinates with fixed values defined in the coordinate
        table (e.g., height2m = 2.0 meters). Variables that list these
        coordinates in their dimensions require them in the output, but they
        don't need explicit values from the user. This method identifies which
        required scalar axes are missing from the provided axes list.

        Parameters
        ----------
        variable
            Variable whose dimensions are checked for required scalar axes.
        axes
            Already-provided axes to check against. Scalar axes present in
            this list are not returned.

        Returns
        -------
        tuple[Axis, ...]
            Tuple of ``Axis`` records for scalar coordinates that are required
            by the variable's dimensions but not present in the provided axes.
            Each axis is marked as scalar and includes the table-defined value.

        This is used internally while preparing axes for dataset creation.
        """

        present = {
            str(value)
            for axis in axes
            for value in (
                axis.name,
                axis.table_entry,
                axis.out_name,
                axis.generic_level_name,
            )
            if value
        }
        missing_axes: list[Axis] = []
        for dimension in variable.dimensions or ():
            dimension_name = str(dimension)
            if dimension_name in present:
                continue
            if dimension_name not in self.coordinate_table.scalar_entries:
                continue
            request = Axis(
                name=dimension_name,
                table_entry=dimension_name,
                scalar=True,
            )
            axis = self.coordinate_table.build(request)
            validate_axis_values_early(axis)
            missing_axes.append(axis)
            present.update(
                str(value)
                for value in (
                    axis.name,
                    axis.table_entry,
                    axis.out_name,
                    axis.generic_level_name,
                )
                if value
            )
        return tuple(self._mark_prepared_axis(axis) for axis in missing_axes)

    def grid(
        self,
        name: str | None = None,
        *,
        axes: Sequence[Axis] = (),
        **values: Any,
    ) -> Grid:
        """Build a grid from the grid table.

        Use a grid for data on a curvilinear grid or a map projection. Pass
        the grid's index axes in ``axes`` (curvilinear grids), a CF
        ``grid_mapping_name`` with ``params`` (projections), or both.

        Parameters
        ----------
        name
            Name of a ``mapping_entry`` in the grid table. Omit it unless the
            table defines the mapping you want. To use a CF projection
            directly, pass ``grid_mapping_name="lambert_conformal_conic"``
            (for example) instead.
        axes
            Index axes for the grid (for example, ``j_index`` and
            ``i_index``), built with :meth:`axis`. The grid's dimensions are
            taken from these axes.
        **values
            Any :class:`Grid` field, such as ``latitude``, ``longitude``,
            ``latitude_vertices``, ``longitude_vertices``, or ``params``.

        Returns
        -------
        Grid
            Grid metadata.

        Raises
        ------
        TableValidationError
            If an axis is not an index axis in the grid table.

        See Also
        --------
        :doc:`/usage/grids` : Curvilinear and projected grid examples.
        """
        data = {"name": name, **values}
        requested = str(data.get("name") or "")
        gm_entry = self.grid_table.resolve_mapping(requested) if requested else None

        # Validate that each axis corresponds to an entry in the grids table.
        grid_axis_names = set(self.grid_table.axis_entries)
        for axis in axes:
            candidate = str(axis.table_entry or axis.name or axis.out_name)
            if grid_axis_names and candidate not in grid_axis_names:
                raise TableValidationError(
                    f"Axis {candidate!r} is not in the grid table axis_entry. "
                    f"Valid grid axes are: {sorted(grid_axis_names)}."
                )
        if gm_entry is not None or data.get("grid_mapping_name") is not None:
            validate_grid_mapping(
                gm_entry,
                data,
                axes=axes,
                dimensions=data.get("dimensions"),
            )

        data = {k: v for k, v in data.items() if v is not None}
        if gm_entry is not None:
            params = dict(data.get("params") or {})
            for parameter in gm_entry.numbered_parameters():
                if parameter in data:
                    params.setdefault(parameter, data.pop(parameter))
            if params:
                data["params"] = params
        if axes:
            data["axes"] = list(axes)
        return self.grid_table.build(Grid.model_validate(data))

    def zfactor(self, name: str, **values: Any) -> ZFactor:
        """Build a formula term for a hybrid vertical coordinate.

        Table values (units, ``long_name``, and so on) fill in anything you
        do not supply. For example, hybrid sigma-pressure levels need the
        terms ``a``, ``b``, ``p0``, and ``ps``.

        Parameters
        ----------
        name
            Formula-term table entry name, such as ``"a"``, ``"b"``, or
            ``"ps"``.
        **values
            Any :class:`ZFactor` field. Usually ``values``, ``bounds``, and
            ``dimensions``.

        Returns
        -------
        ZFactor
            Formula-term metadata and values.

        Raises
        ------
        TableValidationError
            If a value conflicts with the table.
        VariableValidationError
            If the values fail the table's range checks.

        See Also
        --------
        :doc:`/usage/vertical_coordinates` : Hybrid coordinate examples.
        """

        request = ZFactor.model_validate({"name": name, **values})
        return self.formula_table.build(request)

    def _validate_dataset(  # noqa: C901 (see ruff.toml)
        self,
        dataset_info: DatasetMetadata | None,
        variable: Variable,
        axes: Sequence[Axis],
        *,
        grid: Grid | None = None,
        zfactors: Sequence[ZFactor] = (),
    ) -> None:
        """Validate metadata records and dataset configuration comprehensively.

        This is the complete validation check that ensures all components are
        consistent with each other and with the loaded project tables. Dataset
        creation calls this before writing data.

        Components created via this ProjectTables instance (using factory
        methods or with project=self) already have validated attributes
        stored and are trusted. Components from other sources are validated
        here to ensure they match table constraints.

        This validation works with the stored attributes in each component
        rather than re-fetching table data.

        Parameters
        ----------
        dataset
            Dataset metadata to validate. If provided, enables additional
            checks:
            - Frequency consistency between dataset and variable
            - Time axis validation with frequency context
            - Dataset global attribute completeness
        variable
            Main variable metadata to validate against the loaded variable
            tables.
        axes
            Coordinate axis metadata to validate against coordinate and grid
            coordinate tables.
        grid
            Optional grid mapping metadata to validate against the loaded grid
            table.
        zfactors
            Optional hybrid-coordinate formula-term metadata to validate
            against formula-term tables.

        Returns
        -------
        None
            Raises ``TableValidationError`` if metadata is inconsistent with
            the loaded project tables or if components are inconsistent with
            each other.

        """

        # must_call_cmor_grid fast-fail: if any axis in the user-provided axes
        # list carries isgridaxis=True but no Grid was supplied, raise
        # immediately before any further validation.  Grid dimensional axes
        # belong in Grid.axes, not in the regular axes sequence; this guard
        # catches the mistake early with a clear error.
        for axis in axes:
            if axis.isgridaxis and grid is None:
                dim = str(axis.out_name or axis.name)
                raise TableValidationError(
                    f"Axis {dim!r} is a grid axis (isgridaxis=True) but no "
                    "Grid was provided.  Pass the Grid via the grid= "
                    "parameter of cmorize()."
                )

        # Variable validation: ensure stored attributes match table entry
        # Note: This may be redundant with validation in _dataset_for_variable,
        # but we validate again here to ensure consistency if the variable was
        # modified after preparation.
        variable_entry = self.variable_table.resolve(variable)

        if dataset_info is not None:
            normalized_dataset = self.cv.get_dataset_info(dataset_info)
            variable_entry = self.variable_table.contextual_entry(
                variable_entry, variable, normalized_dataset
            )

        self.variable_table.validate_against(variable, variable_entry)

        # Axis validation: only validate axes not prepared by this instance
        for axis in axes:
            if not self._is_prepared_axis(axis):
                self.coordinate_table.validate_metadata(axis)

        # Dataset-axis consistency checks
        # (must_have_bounds, time interval,etc.)
        # Run unconditionally so that must_have_bounds and the variable's
        # table-declared frequency are enforced even when no dataset is
        # provided.
        validate_axes(dataset_info, variable, axes)

        # Check that required scalar axes are present (if not auto-added)
        present = {
            str(value)
            for axis in axes
            for value in (
                axis.name,
                axis.table_entry,
                axis.out_name,
                axis.generic_level_name,
            )
            if value
        }
        # Also register dimension names covered by the grid's own axes so the
        # scalar-axis check below does not fire for grid dimensions.
        if grid is not None:
            for axis in getattr(grid, "axes", []):
                for value in (axis.name, axis.out_name, axis.table_entry):
                    if value:
                        present.add(str(value))
            if grid.dimensions:
                present.update(str(d) for d in grid.dimensions)

        for dimension in variable.dimensions or ():
            dimension_name = str(dimension)
            if dimension_name not in present:
                if dimension_name in self.coordinate_table.scalar_entries:
                    raise TableValidationError(
                        f"Variable requires scalar axis {dimension_name!r} "
                        "but it was not provided. Supply the required scalar "
                        "axis when creating the dataset."
                    )
                # Non-scalar dimension not found - will be caught elsewhere
                # (e.g., when building dataset if dimension truly missing)

        # Grid validation: ensure stored attributes match tables
        if grid is not None:
            self.grid_table.validate_metadata(grid, axes)

        # ZFactor validation: ensure stored attributes match tables
        for zfactor in zfactors:
            self.formula_table.validate_metadata(zfactor)

    def _validate_global_attributes(self, attrs: Mapping[str, Any]) -> None:
        """Validate final NetCDF global attributes against project tables.

        Parameters
        ----------
        attrs
            Global attributes from the generated dataset. These include
            dataset metadata, variable-derived global attributes, runtime
            defaults, and any user-supplied attribute overrides.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if required attributes are
            missing or controlled global attribute values are invalid.
        """

        metadata = DatasetMetadata.from_mapping(attrs)
        self.cv.validate_dataset_info(metadata)

    def _mark_prepared_axis(self, axis: Axis) -> Axis:
        object.__setattr__(axis, "_cmor_project_tables", self)
        return axis

    def _is_prepared_axis(self, axis: Axis) -> bool:
        return getattr(axis, "_cmor_project_tables", None) is self

    def _add_table_header_defaults(
        self, dataset: dict[str, Any], variable_entry: VariableEntry
    ) -> None:
        """Fill defaults from the loaded variable table header."""

        header = variable_entry.table_header
        for key in ("Conventions", "data_specs_version"):
            value = getattr(header, key, None)
            if value is not None:
                dataset.setdefault(key, value)

        if "table_info" not in dataset and variable_entry.table_file is not None:
            dataset["table_info"] = _build_table_info(
                variable_entry.table_file,
                getattr(header, "table_date", None),
            )

    def _add_variable_global_defaults(
        self, dataset: dict[str, Any], variable: Variable
    ) -> None:
        """Fill global attributes that are derived from the variable table."""

        variable_id, labels = variable.names()
        if "variable_id" not in dataset or _is_unresolved_template(
            dataset["variable_id"]
        ):
            dataset["variable_id"] = variable_id
        if "branded_variable" not in dataset or _is_unresolved_template(
            dataset["branded_variable"]
        ):
            dataset["branded_variable"] = labels["branded_name"]
        for key in (
            "branding_suffix",
            "temporal_label",
            "vertical_label",
            "horizontal_label",
            "area_label",
        ):
            if key in labels:
                if key not in dataset or _is_unresolved_template(dataset[key]):
                    dataset[key] = labels[key]
        for key in ("frequency", "realm", "table_id"):
            value = getattr(variable, key, None)
            if value not in (None, ""):
                dataset.setdefault(
                    key,
                    value[0] if isinstance(value, list) and len(value) == 1 else value,
                )


def _resolve_optional_table(
    root_path: Path, table: str | Path | None, suffix: str
) -> Path | None:
    if table is not None:
        return root_path / table
    for directory in ("tables", "Tables"):
        table_dir = root_path / directory
        if not table_dir.exists():
            continue
        matches = sorted(table_dir.glob(f"*_{suffix}.json"))
        if matches:
            return matches[0]
    return None


def _build_table_info(table_file: Path, table_date: str | None = None) -> str:
    """Build a table_info string with the filename, table date, and MD5 hash.

    Matches the format written by CMOR3:
    ``Name: <file>; Creation Date:(<date>) MD5:<hash>``

    The date is the ``table_date`` from the table header, copied as written.
    When the header has no ``table_date``, the date part is left out.
    """
    try:
        md5 = hashlib.md5(table_file.read_bytes()).hexdigest()
    except OSError:
        return f"Name: {table_file.name};"
    if table_date:
        return f"Name: {table_file.name}; Creation Date:({table_date}) MD5:{md5}"
    return f"Name: {table_file.name}; MD5:{md5}"
