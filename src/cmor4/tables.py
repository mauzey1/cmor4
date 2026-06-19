from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._table_utils import (
    is_table_value as _is_table_value,
    single_or_original as _single_or_original,
    validate_table_metadata as _validate_table_metadata,
)
from ._templates import is_unresolved_template as _is_unresolved_template
from ._axis_validation import validate_axes as _validate_axes
from ._axis_validation import validate_axis_values_early as _validate_axis_values_early
from ._axis_validation import _validate_calendar
from ._tables import (
    CoordinateTable,
    FormulaTable,
    GridTable,
    VariableEntry,
    VariableTable,
)
from .axis import Axis
from .cv import ControlledVocabulary
from .dataset import DatasetInfo
from .exceptions import TableValidationError
from .grid import Grid
from .variable import Variable
from ._unit_conversion import units_are_convertible as _units_are_convertible
from .zfactor import ZFactor


class ProjectTables:
    """Project CV and variable-table validator.

    Parameters are paths to existing project table files. For example, CMIP7
    can be loaded from ``cmip7-cmor-tables/tables-cvs/cmor-cvs.json``, one or
    more variable tables under ``cmip7-cmor-tables/tables/``, and the project
    coordinate and formula-term tables.

    Parameters
    ----------
    cv_file
        Path to the project controlled-vocabulary JSON file.
    variable_tables
        Paths to variable table JSON files.
    coordinate_table
        Optional path to the coordinate table JSON file.
    formula_table
        Optional path to the formula-terms table JSON file.
    grid_table
        Optional path to the grids table JSON file.
    """

    def __init__(
        self,
        cv_file: str | Path,
        variable_tables: Sequence[str | Path],
        coordinate_table: str | Path | None = None,
        formula_table: str | Path | None = None,
        grid_table: str | Path | None = None,
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
        self.cv = ControlledVocabulary.from_file(self.cv_file)
        self.variable_table = VariableTable(self.variable_table_files)

        # Load raw grid table data first — coordinate table needs the axis
        # entries from it for the overlay that gives grid-specific names priority.
        raw_grid_axis: dict[str, Mapping[str, Any]] = {}
        raw_grid_coord: dict[str, Mapping[str, Any]] = {}
        raw_grid_mapping: dict[str, Mapping[str, Any]] = {}
        if self.grid_table_file is not None:
            raw_grid_axis = self._read_entries(self.grid_table_file, "axis_entry")
            raw_grid_coord = self._read_entries(self.grid_table_file, "variable_entry")
            raw_grid_mapping = self._read_entries(self.grid_table_file, "mapping_entry")
        self.grid_table = GridTable(raw_grid_axis, raw_grid_coord, raw_grid_mapping)

        raw_coord: dict[str, Mapping[str, Any]] = {}
        if self.coordinate_table_file is not None:
            raw_coord = self._read_entries(self.coordinate_table_file, "axis_entry")
        self.coordinate_table = CoordinateTable(
            raw_coord, raw_grid_axis, raw_grid_coord
        )

        raw_formula: dict[str, Mapping[str, Any]] = {}
        if self.formula_table_file is not None:
            raw_formula = self._read_entries(self.formula_table_file, "formula_entry")
        self.formula_table = FormulaTable(raw_formula)

    # ------------------------------------------------------------------
    # Backward-compatible properties (delegate to table objects)
    # ------------------------------------------------------------------

    @property
    def coordinate_entries(self) -> dict[str, Mapping[str, Any]]:
        """Overlaid coordinate entries (grid axis entries take precedence)."""
        return self.coordinate_table._all_coord

    @property
    def grid_axis_entries(self) -> dict[str, Mapping[str, Any]]:
        """Raw axis entries from the grids table."""
        return self.grid_table.axis_entries

    @property
    def grid_coordinate_entries(self) -> dict[str, Mapping[str, Any]]:
        """Raw grid-coordinate entries from the grids table."""
        return self.grid_table.coord_entries

    @property
    def grid_mapping_entries(self) -> dict[str, Mapping[str, Any]]:
        """Raw grid-mapping entries from the grids table."""
        return self.grid_table._raw_mapping

    @property
    def scalar_axis_entries(self) -> dict[str, Mapping[str, Any]]:
        """Coordinate entries that carry a fixed scalar value."""
        return self.coordinate_table.scalar_entries

    @property
    def generic_level_entries(self) -> dict[str, dict[str, Mapping[str, Any]]]:
        """Two-level index: generic_level_name → {entry_name → entry}."""
        return self.coordinate_table.generic_level_entries

    @property
    def formula_entries(self) -> dict[str, Mapping[str, Any]]:
        """Raw formula-term entries."""
        return self.formula_table._entries

    @property
    def variable_entries(self) -> dict[str, VariableEntry]:
        """Variable entries indexed by name (first table wins on duplicates)."""
        return self.variable_table.entries

    @property
    def _variable_entries_by_name(self) -> dict[str, list[VariableEntry]]:
        """Variable entries grouped by short name (may span multiple tables)."""
        return self.variable_table._by_name

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
    ) -> "ProjectTables":
        """Load tables using paths relative to a project root.

        Parameters
        ----------
        root
            Project table root directory.
        cv_file
            Controlled-vocabulary file path relative to ``root``.
        variable_tables
            Variable table paths relative to ``root``.
        coordinate_table
            Optional coordinate table path relative to ``root``.
        formula_table
            Optional formula-terms table path relative to ``root``.
        grid_table
            Optional grids table path relative to ``root``.

        Returns
        -------
        ProjectTables
            Loaded project table helper.
        """

        root_path = Path(root)
        resolved_coordinate_table = _resolve_optional_table(
            root_path, coordinate_table, "coordinate"
        )
        resolved_formula_table = _resolve_optional_table(
            root_path, formula_table, "formula_terms"
        )
        resolved_grid_table = _resolve_optional_table(root_path, grid_table, "grids")
        return cls(
            root_path / cv_file,
            [root_path / table_file for table_file in variable_tables],
            coordinate_table=resolved_coordinate_table,
            formula_table=resolved_formula_table,
            grid_table=resolved_grid_table,
        )

    def dataset_info(
        self,
        dataset: Mapping[str, Any],
    ) -> DatasetInfo:
        """Create prepared dataset metadata from user input and tables.

        Parameters
        ----------
        dataset
            User-provided dataset-level metadata.

        Returns
        -------
        DatasetInfo
            Validated and defaulted dataset metadata.
        """

        normalized_dataset = self.cv.get_dataset_info(dataset)
        self.cv.validate_dataset_values(normalized_dataset)
        self.cv.validate_variant_indices(normalized_dataset)
        self.cv.validate_forcing_terms(normalized_dataset)
        self.validate_source_attributes(normalized_dataset)
        self.validate_experiment(normalized_dataset)
        self.validate_parent_attributes(normalized_dataset)
        return DatasetInfo.from_mapping(normalized_dataset, project=self)

    def _dataset_for_variable(
        self,
        dataset: DatasetInfo,
        variable: Variable,
    ) -> DatasetInfo:
        """Prepare dataset info with variable-specific metadata and validation.

        This is called by create_dataset to merge variable metadata into
        dataset and perform initial validation. Full component validation
        happens later via validate_components.
        """
        normalized_dataset = self.cv.get_dataset_info(dataset)
        variable_entry = self.variable_table.resolve(variable.to_dict())
        self._add_table_header_defaults(normalized_dataset, variable_entry)
        self._add_variable_global_defaults(normalized_dataset, variable)
        self.validate_dataset(normalized_dataset)
        self.validate_source_attributes(normalized_dataset)
        self.validate_experiment(normalized_dataset)
        self.validate_parent_attributes(normalized_dataset)

        # Note: Full variable and dataset-variable consistency validation
        # happens in validate_components, not here
        prepared_dataset = DatasetInfo.from_mapping(normalized_dataset, project=self)

        # Quick validation check for dataset-variable consistency
        # This is duplicated in validate_components but done early for fast
        # failure
        self.variable_table.validate_against(variable, variable_entry)
        self._validate_dataset_variable_consistency(
            prepared_dataset, variable, variable_entry
        )

        return prepared_dataset

    def variable(self, name: str, **values: Any) -> Variable:
        """Create a variable with metadata from the loaded variable tables.

        This factory method creates a ``Variable`` metadata record by resolving
        the variable name against loaded tables and merging table metadata with
        user-provided values. Table metadata (units, standard_name, dimensions,
        etc.) are authoritative and will override conflicting user values. User
        values are used for data-specific attributes like missing_value.

        Parameters
        ----------
        name
            Variable name or branded variable name to resolve in the loaded
            tables. Can be a simple variable name (e.g., "tas") or a branded
            name with suffix (e.g., "tas_ann-lev-reg-mean").
        **values
            Optional user-supplied variable metadata. Common keywords include
            missing_value, fill_value, chunksizes, valid_min, valid_max,
            ok_min_mean_abs, ok_max_mean_abs, coordinates, and attrs for
            additional NetCDF attributes.

        Returns
        -------
        Variable
            Variable metadata record with table values merged and validated.

        Raises
        ------
        TableValidationError
            If the variable name is not found in loaded tables, if the name is
            ambiguous across multiple tables without specifying table_id, or if
            user-supplied metadata conflicts with table requirements.

        Examples
        --------
        Create a simple variable from table::

            project = ProjectTables.from_directory(...)
            variable = project.variable("tas")
            # Returns Variable with units="K",
            # dimensions=("time", "lat", "lon")

        Create variable with data-specific attributes::

            variable = project.variable(
                "tas",
                missing_value=-999.0,
                valid_min=200.0,
                valid_max=330.0
            )

        Disambiguate variable across tables::

            variable = project.variable("tas", table_id="Amon")
            # Uses monthly atmospheric table specifically
        """

        data = self.variable_table.build({"name": name, **values})
        return Variable.model_validate(data)

    def axis(self, name: str, **values: Any) -> Axis:
        """Create an axis with metadata from the loaded coordinate tables.

        This factory method creates an ``Axis`` metadata record by resolving
        the axis name against loaded coordinate and grid tables, merging table
        metadata with user-provided values. The created axis is marked as
        prepared by this ProjectTables instance for efficient validation later.

        Parameters
        ----------
        name
            Axis or coordinate table entry name. Can be a standard coordinate
            name (e.g., "time", "lat", "lon"), a generic level name (e.g.,
            "alevel", "plev"), or a grid coordinate name.
        **values
            User-supplied axis metadata and coordinate values. Required keyword
            is typically ``values`` for the coordinate array. Optional keywords
            include bounds, units, standard_name, out_name, dimensions (for
            auxiliary coordinates), scalar, valid_min, valid_max, and various
            table entry selectors (table_entry, axis_entry, coordinate).

        Returns
        -------
        Axis
            Axis metadata record with table values merged and marked as
            prepared by this ProjectTables instance.

        Raises
        ------
        TableValidationError
            If the axis name matches multiple generic level entries without
            disambiguation, or if user-supplied metadata conflicts with table
            requirements.
        AxisValidationError
            If coordinate values or bounds are invalid (non-monotonic,
            out-of-range, inconsistent shapes).

        Examples
        --------
        Create a time axis::

            project = ProjectTables.from_directory(...)
            time_axis = project.axis(
                "time",
                values=[0, 31, 59, 90],
                bounds=[[0, 31], [31, 59], [59, 90], [90, 120]]
            )

        Create a latitude axis from table::

            lat_axis = project.axis("lat", values=np.linspace(-90, 90, 180))

        Create a pressure level axis::

            plev_axis = project.axis(
                "plev",
                values=[100000, 92500, 85000, 70000, 50000, 25000, 10000]
            )

        Disambiguate generic level with standard_name::

            alevel_axis = project.axis(
                "alevel",
                standard_name="altitude",
                values=[10, 50, 100, 500, 1000]
            )
        """

        data = self.coordinate_table.build({"name": name, **values})
        axis = Axis.model_validate(data)
        _validate_axis_values_early(axis)
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
                else Axis.model_validate(self.coordinate_table.build(axis.to_dict()))
            )
            for axis in axes
        ]
        if variable is not None:
            merged_axes.extend(self.scalar_axes_for(variable, merged_axes))
        return tuple(self._mark_prepared_axis(axis) for axis in merged_axes)

    def scalar_axes_for(
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

        Notes
        -----
        This method is called automatically by ``create_dataset`` when using
        project-backed metadata. It can also be called directly to preview
        which scalar axes will be auto-added.

        Examples
        --------
        Check which scalar axes a variable needs::

            project = ProjectTables.from_directory(...)
            variable = project.variable("tas")  # dimensions include "height2m"
            lat = project.axis("lat", values=[...])
            lon = project.axis("lon", values=[...])
            time = project.axis("time", values=[...])

            missing = project.scalar_axes_for(variable, [lat, lon, time])
            # Returns tuple with height2m axis

        Preview scalar axes before dataset creation::

            variable = project.variable("tas")
            scalar_axes = project.scalar_axes_for(variable)
            # Returns all required scalar axes if no axes provided
        """

        present = {
            str(value)
            for axis in axes
            for value in (
                axis.name,
                axis.table_entry,
                axis.axis_entry,
                axis.coordinate,
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
            data = self.coordinate_table.build(
                {"name": dimension_name, "table_entry": dimension_name, "scalar": True}
            )
            axis = Axis.model_validate(data)
            _validate_axis_values_early(axis)
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

    def complete_axes(
        self,
        variable: Variable,
        axes: Sequence[Axis],
    ) -> tuple[Axis, ...]:
        """Return supplied axes plus fixed scalar axes required by variable.

        This convenience method combines user-provided axes with any required
        scalar axes, returning a complete set ready for dataset creation. It
        ensures all axes are merged with project table metadata.

        Parameters
        ----------
        variable
            Variable whose dimensions determine which scalar axes are required.
        axes
            User-provided axes for the variable.

        Returns
        -------
        tuple[Axis, ...]
            Complete tuple of axes including both the provided axes (merged
            with table metadata if needed) and any required scalar axes.

        See Also
        --------
        scalar_axes_for : Get only the missing scalar axes without merging.

        Examples
        --------
        Get complete axis set for a variable::

            project = ProjectTables.from_directory(...)
            variable = project.variable("tas")
            time = project.axis("time", values=[...])
            lat = project.axis("lat", values=[...])
            lon = project.axis("lon", values=[...])

            complete = project.complete_axes(variable, [time, lat, lon])
            # Returns (time, lat, lon, height2m) with height2m auto-added
        """

        return self._axes(axes, variable)

    def grid(
        self,
        name: str | None = None,
        *,
        axes: Sequence[Axis] = (),
        **values: Any,
    ) -> Grid:
        """Create a grid with metadata from the loaded grid table.

        This factory method creates a ``Grid`` metadata record for variables
        on non-rectilinear grids or with coordinate reference systems.  It
        resolves grid mapping entries from the project's grid table and merges
        projection parameters with user-provided values.

        When *axes* are supplied the grid owns those :class:`Axis` objects as
        its indexing dimensions.  Each axis is validated against the grid
        table's ``axis_entry`` section and flagged ``isgridaxis=True``.  The
        ``dimensions`` tuple is derived automatically from the axis
        ``out_name`` values so it need not be supplied separately.

        Parameters
        ----------
        name
            Optional grid mapping entry name (e.g.,
            ``"lambert_conformal_conic"``, ``"rotated_latitude_longitude"``).
            If ``None``, the grid must specify mapping metadata via other
            parameters.
        axes
            :class:`Axis` objects that form the grid's indexing dimensions
            (e.g. ``i_index``, ``j_index``).  Each must correspond to an
            entry in the grid table's ``axis_entry`` section.  Pass an empty
            sequence (the default) for grids that use only ``dimensions``
            names or have no spatial index axes.
        **values
            User-supplied grid metadata.  Common keywords include
            ``dimensions`` (string names, when not derived from *axes*),
            ``mapping_name`` / ``grid_mapping_name``, ``params`` (projection
            parameters dict), ``coordinates`` (auxiliary coordinate names),
            ``mapping_var`` (grid mapping variable name), ``latitude``,
            ``longitude``, ``latitude_vertices``, ``longitude_vertices``, and
            ``attrs`` for additional attributes.

        Returns
        -------
        Grid
            Grid metadata record with table values and projection parameters
            merged.  When *axes* is non-empty the returned ``Grid.axes``
            contains updated copies of the supplied axes with
            ``isgridaxis=True``.

        Raises
        ------
        TableValidationError
            If any supplied axis name is not present in the grid table's
            ``axis_entry`` section.

        Examples
        --------
        Axis-based curvilinear ocean grid::

            project = ProjectTables.from_directory(...)
            i_axis = project.axis("i_index", values=np.arange(192))
            j_axis = project.axis("j_index", values=np.arange(144))

            grid = project.grid(
                axes=[j_axis, i_axis],
                latitude=lat_2d,
                longitude=lon_2d,
                latitude_vertices=blat_3d,
                longitude_vertices=blon_3d,
            )

        Name-based rotated pole grid::

            grid = project.grid(
                "rotated_latitude_longitude",
                params={
                    "grid_north_pole_latitude": (37.5, "degrees_north"),
                    "grid_north_pole_longitude": (-177.5, "degrees_east"),
                },
            )
        """
        # Validate that each axis corresponds to an entry in the grids table.
        grid_axis_names = set(self.grid_table.axis_entries)
        for axis in axes:
            candidate = str(
                axis.table_entry
                or axis.axis_entry
                or axis.coordinate
                or axis.name
                or axis.out_name
            )
            if grid_axis_names and candidate not in grid_axis_names:
                raise TableValidationError(
                    f"Axis {candidate!r} is not in the grid table axis_entry. "
                    f"Valid grid axes are: {sorted(grid_axis_names)}."
                )

        data = {k: v for k, v in {"name": name, **values}.items() if v is not None}
        if axes:
            data["axes"] = list(axes)
        return Grid.model_validate(self.grid_table.build(data))

    def zfactor(self, name: str, **values: Any) -> ZFactor:
        """Create a z-factor with metadata from formula-term tables.

        This factory method creates a ``ZFactor`` metadata record for
        hybrid-coordinate formula terms (e.g., coefficients for hybrid
        sigma-pressure coordinates). It resolves the formula term name against
        the loaded formula table and merges table metadata with user values.

        Parameters
        ----------
        name
            Formula-term table entry name (e.g., "ap", "b", "ps", "p0",
            "orog").
        **values
            User-supplied formula-term metadata and values. Required keyword is
            typically ``values`` or ``data`` for the formula term array.
            Optional keywords include dimensions, bounds, out_name, valid_min,
            valid_max, ok_min_mean_abs, ok_max_mean_abs, and attrs for
            additional NetCDF attributes.

        Returns
        -------
        ZFactor
            Formula-term metadata record with table values merged and
            validated.

        Raises
        ------
        TableValidationError
            If user-supplied metadata conflicts with table requirements.
        VariableValidationError
            If formula term values fail validation checks.

        Notes
        -----
        Formula terms are required for variables on hybrid vertical
        coordinates. The most common case is hybrid sigma-pressure coordinates
        which require ``ap``, ``b``, ``ps``, and optionally ``p0`` terms.

        Examples
        --------
        Create hybrid sigma-pressure formula terms::

            project = ProjectTables.from_directory(...)

            # Hybrid coefficient a (Pa)
            ap = project.zfactor("ap", values=[0, 2000, 5000, 10000])

            # Hybrid coefficient b (dimensionless)
            b = project.zfactor("b", values=[1.0, 0.95, 0.90, 0.80])

            # Surface pressure (Pa) - 3D field
            ps = project.zfactor(
                "ps",
                values=surface_pressure_3d,
                dimensions=("time", "lat", "lon")
            )

            # Reference pressure (Pa) - scalar
            p0 = project.zfactor("p0", values=100000.0)

        Create orography term for ocean coordinates::

            orog = project.zfactor(
                "orog",
                values=ocean_depth,
                dimensions=("lat", "lon")
            )
        """

        data = self.formula_table.build({"name": name, **values})
        return ZFactor.model_validate(data)

    def validate_components(
        self,
        dataset: DatasetInfo | None,
        variable: Variable,
        axes: Sequence[Axis],
        *,
        grid: Grid | None = None,
        zfactors: Sequence[ZFactor] = (),
    ) -> None:
        """Validate metadata records and dataset configuration comprehensively.

        This is the complete validation check that ensures all components are
        consistent with each other and with the loaded project tables. It can
        be used both as the final check before dataset creation and as a
        user-facing validation function to verify metadata setup before
        writing data.

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

        Examples
        --------
        Validate components before creating a dataset::

            project = ProjectTables(...)
            dataset = project.dataset_info({...})
            variable = project.variable("tas")
            axes = [project.axis("time", ...), project.axis("lat", ...)]

            # Validate everything before attempting to create dataset
            project.validate_components(dataset, variable, axes)

            # If validation passes, safe to create dataset
            ds = create_dataset(dataset, variable, axes, data)
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
                    "parameter of create_dataset() / cmorize()."
                )

        # Variable validation: ensure stored attributes match table entry
        # Note: This may be redundant with validation in _dataset_for_variable,
        # but we validate again here to ensure consistency when called directly
        # by users or if variable was modified after _dataset_for_variable
        variable_entry = self.variable_table.resolve(variable.to_dict())
        self.variable_table.validate_against(variable, variable_entry)

        # Dataset-variable consistency checks
        if dataset is not None:
            self._validate_dataset_variable_consistency(
                dataset, variable, variable_entry
            )

        # Axis validation: only validate axes not prepared by this instance
        for axis in axes:
            if not self._is_prepared_axis(axis):
                # Check if this is a grid coordinate (auxiliary lat/lon)
                adict = axis.to_dict()
                if ae := self.coordinate_table.resolve_grid_coord(adict):
                    # Grid coordinates validated against grid coordinate table
                    _validate_table_metadata(
                        adict,
                        ae.name,
                        ae.entry,
                        ("units", "standard_name", "long_name"),
                        "grid coordinate",
                    )
                elif ae := self.coordinate_table.resolve_coord(adict):
                    # Regular coordinates validated against coordinate table
                    _validate_table_metadata(
                        adict,
                        ae.name,
                        ae.entry,
                        (
                            "units",
                            "standard_name",
                            "long_name",
                            "axis",
                            "positive",
                            "formula",
                        ),
                        "axis",
                    )

        # Dataset-axis consistency checks
        # (must_have_bounds, time interval,etc.)
        # Run unconditionally so that must_have_bounds and the variable's
        # table-declared frequency are enforced even when no dataset is
        # provided.
        _validate_axes(dataset, variable, axes)

        # Calendar validation: warn if the dataset specifies a calendar that
        # is technically valid per CF but inappropriate for MIP data.
        if dataset is not None:
            _validate_calendar(dataset)

        # Check that required scalar axes are present (if not auto-added)
        present = {
            str(value)
            for axis in axes
            for value in (
                axis.name,
                axis.table_entry,
                axis.axis_entry,
                axis.coordinate,
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
                if dimension_name in self.scalar_axis_entries:
                    raise TableValidationError(
                        f"Variable requires scalar axis {dimension_name!r} "
                        "but it was not provided. Use "
                        "ProjectTables.scalar_axes_for() or "
                        "ProjectTables.complete_axes() to get required "
                        "scalar axes."
                    )
                # Non-scalar dimension not found - will be caught elsewhere
                # (e.g., when building dataset if dimension truly missing)

        # Grid validation: ensure stored attributes match tables
        if grid is not None:
            requested = str(grid.table_entry or grid.mapping_entry or grid.name or "")
            _gm = self.grid_table.resolve_mapping(requested) if requested else None
            entry_name, entry = (_gm.name, _gm.entry) if _gm else (None, None)
            if entry is not None:
                for key, user_val in (
                    ("mapping_name", grid.mapping_name),
                    ("grid_mapping_name", grid.grid_mapping_name),
                ):
                    expected = entry.get(key)
                    if (
                        _is_table_value(expected)
                        and user_val is not None
                        and str(user_val) != str(expected)
                    ):
                        raise TableValidationError(
                            f"grid mapping {entry_name!r} {key}="
                            f"{user_val!r} does not match table value "
                            f"{expected!r}."
                        )

            # Validate dimensional axes and lat/lon shape consistency.
            grid_axes = getattr(grid, "axes", [])
            if grid_axes:
                _validate_grid_dimensions(grid, axes)
            elif grid.dimensions:
                _validate_grid_dimensions(grid, axes)

        # ZFactor validation: ensure stored attributes match tables
        for zfactor in zfactors:
            _ze = self.formula_table.resolve(zfactor.to_dict())
            entry_name, entry = (_ze.name, _ze.entry) if _ze else (None, None)
            if entry is None:
                continue

            # Units must be dimensionally convertible, not just equal.
            # Use the same cf_units-based check as Variable units validation.
            table_units = entry.get("units")
            user_units = zfactor.units
            if (
                _is_table_value(table_units)
                and str(table_units) != "?"
                and user_units not in (None, "")
                and str(user_units) != str(table_units)
                and not _units_are_convertible(str(user_units), str(table_units))
            ):
                raise TableValidationError(
                    f"formula term {entry_name!r} units={user_units!r} does "
                    f"not match table value {table_units!r} and the two are "
                    f"not dimensionally convertible."
                )

            # Validate remaining metadata (standard_name, long_name)
            # by exact match.
            _validate_table_metadata(
                zfactor.to_dict(),
                entry_name,
                entry,
                ("standard_name", "long_name"),
                "formula term",
            )

            # When the formula-term table entry has no declared
            # dimensions the term is expected to be a scalar.  Accept a
            # size-1 array (CMOR3-compatible) but reject larger arrays.
            entry_dims = entry.get("dimensions")
            if not _is_table_value(entry_dims) and zfactor.values is not None:
                arr = np.asarray(zfactor.values)
                if arr.ndim > 0 and arr.size != 1:
                    raise TableValidationError(
                        f"formula term {entry_name!r} has no declared "
                        "dimensions (expected a scalar value) but values "
                        f"with shape {arr.shape} were provided."
                    )

    def _validate_dataset_variable_consistency(
        self,
        dataset: DatasetInfo,
        variable: Variable,
        variable_entry: VariableEntry,
    ) -> None:
        """Validate consistency between dataset and variable metadata."""
        # Check frequency consistency
        if (
            "frequency" in dataset
            and variable.frequency is not None
            and str(dataset["frequency"]) != str(variable.frequency)
        ):
            raise TableValidationError(
                f"Dataset frequency={dataset['frequency']!r} does not match "
                f"variable {variable_entry.table_id}:{variable_entry.name} "
                f"frequency={variable.frequency!r}."
            )

    def validate_global_attributes(self, attrs: Mapping[str, Any]) -> None:
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

        self.validate_dataset(attrs)
        self.validate_source_attributes(attrs)
        self.validate_experiment(attrs)
        self.validate_parent_attributes(attrs)

    def _mark_prepared_axis(self, axis: Axis) -> Axis:
        object.__setattr__(axis, "_cmor4_project_tables", self)
        return axis

    def _is_prepared_axis(self, axis: Axis) -> bool:
        return getattr(axis, "_cmor4_project_tables", None) is self

    def _add_table_header_defaults(
        self, dataset: dict[str, Any], variable_entry: VariableEntry
    ) -> None:
        """Fill defaults from the loaded variable table header."""

        header = variable_entry.table_header or {}
        for key in ("Conventions", "data_specs_version"):
            value = header.get(key)
            if _is_table_value(value):
                dataset.setdefault(key, value)

        if "table_info" not in dataset and variable_entry.table_file is not None:
            dataset["table_info"] = _build_table_info(variable_entry.table_file)

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
            if _is_table_value(value):
                dataset.setdefault(key, _single_or_original(value))

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None:
        """Validate user-supplied controlled values against the project CV.

        This method performs controlled vocabulary validation on dataset-level
        metadata, checking that CV-controlled attribute values (like
        institution_id, source_id, experiment_id) are recognized and that
        required attributes are present.

        Parameters
        ----------
        dataset
            Dataset metadata dictionary containing global attributes to
            validate against the project's controlled vocabulary.

        Raises
        ------
        ControlledVocabularyError
            If required attributes are missing, if attribute values are not
            found in the CV, or if attribute combinations are invalid.

        Examples
        --------
        Validate dataset before creating variables::

            project = ProjectTables.from_directory(...)
            dataset_attrs = {
                "mip_era": "CMIP7",
                "institution_id": "NCAR",
                "source_id": "CESM2",
                "experiment_id": "historical"
            }
            project.validate_dataset(dataset_attrs)
            # Raises ControlledVocabularyError if any value is invalid
        """

        self.cv.validate_dataset(dataset)

    def validate_required_global_attributes(self, dataset: Mapping[str, Any]) -> None:
        """Require every CV-listed global attribute that CMOR4 can write.

        Parameters
        ----------
        dataset
            Dataset metadata to check.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if required attributes are
            missing.
        """

        self.cv.validate_required_global_attributes(dataset)

    def required_global_attributes(self) -> tuple[str, ...]:
        """Return CV-listed required global attributes.

        This method returns the list of global attribute names that are marked
        as required in the project's controlled vocabulary. These attributes
        must be present in dataset metadata before writing NetCDF output.

        Returns
        -------
        tuple[str, ...]
            Tuple of required global attribute names from the project CV.
            Common examples include "mip_era", "institution_id", "source_id",
            "experiment_id", "variant_label", "grid_label", etc.

        Examples
        --------
        Check which attributes are required::

            project = ProjectTables.from_directory(...)
            required = project.required_global_attributes()
            # Returns ("mip_era", "institution_id", "source_id", ...)

        Validate that dataset has required attributes::

            required = project.required_global_attributes()
            for attr in required:
                if attr not in dataset:
                    print(f"Missing required attribute: {attr}")
        """

        return self.cv.required_global_attributes()

    def validate_experiment(self, dataset: Mapping[str, Any]) -> None:
        """Validate experiment-specific CV attributes.

        Parameters
        ----------
        dataset
            Dataset metadata containing an ``experiment_id``.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if experiment metadata is
            inconsistent.
        """

        self.cv.validate_experiment(dataset)

    def validate_source_type(
        self,
        dataset: Mapping[str, Any],
        experiment_entry: Mapping[str, Any],
    ) -> None:
        """Validate experiment-specific required source_type tokens.

        Parameters
        ----------
        dataset
            Dataset metadata containing ``source_type``.
        experiment_entry
            Experiment CV entry with required and allowed source types.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if source types are missing
            or disallowed.
        """

        self.cv.validate_source_type(dataset, experiment_entry)

    def validate_source_attributes(self, dataset: Mapping[str, Any]) -> None:
        """Validate source_id-specific CV attributes.

        Parameters
        ----------
        dataset
            Dataset metadata containing a ``source_id``.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if source-specific metadata
            is inconsistent.
        """

        self.cv.validate_source_attributes(dataset)

    def validate_parent_attributes(self, dataset: Mapping[str, Any]) -> None:
        """Validate CMIP-style parent experiment attributes.

        Parameters
        ----------
        dataset
            Dataset metadata containing experiment and parent metadata.

        Returns
        -------
        None
            Raises ``ControlledVocabularyError`` if parent metadata is missing
            or inconsistent.
        """

        self.cv.validate_parent_attributes(dataset)

    @staticmethod
    def _read_entries(table_file: Path, key: str) -> dict[str, Mapping[str, Any]]:
        with table_file.open() as handle:
            data = json.load(handle)
        return {
            str(name): entry
            for name, entry in data.get(key, {}).items()
            if isinstance(entry, Mapping)
        }


def _validate_grid_dimensions(
    grid: Grid,
    axes: Sequence[Axis],
) -> None:
    """Validate that grid dimensions correspond to spatial axes in the right order.

    When ``grid.axes`` is non-empty the dimensional axes are already owned by
    the grid; validation checks that each axis carries coordinate values and
    that the lat/lon array shapes match the axis lengths in declared order.

    When ``grid.axes`` is empty (name-based path) every name in
    ``grid.dimensions`` must resolve to one of the supplied *axes* by any
    recognised name variant, and the same shape checks apply.

    Parameters
    ----------
    grid
        Grid whose dimensional axes are to be validated.
    axes
        The axis objects supplied alongside the grid (non-grid axes such as
        time).  Only consulted on the name-based path.

    Raises
    ------
    TableValidationError
        If a grid dimension name cannot be matched to any axis, or if the
        lat/lon array size along a given axis index does not match the axis
        length.
    """
    import numpy as np

    if getattr(grid, "axes", None):
        # --- Axis-based path: grid owns its dimensional Axis objects -------
        grid_axes = list(grid.axes)

        lat = grid.latitude
        lon = grid.longitude
        for array, label in ((lat, "latitude"), (lon, "longitude")):
            if array is None:
                continue
            arr = np.asarray(array)
            for i, axis in enumerate(grid_axes):
                axis_len = len(axis.values_array())
                if arr.shape[i] != axis_len:
                    dim_name = str(axis.out_name or axis.name)
                    raise TableValidationError(
                        f"Grid {label} array shape {arr.shape} does not match "
                        f"the axis lengths implied by grid.axes. "
                        f"Axis {dim_name!r} (index {i}) has {axis_len} "
                        f"coordinate values but {label} has size "
                        f"{arr.shape[i]} along that axis."
                    )
        return

    # --- Name-based path: match dimension strings to supplied axes ----------
    if not grid.dimensions:
        return

    # Build a lookup from every recognised name variant to the axis object.
    name_to_axis: dict[str, Axis] = {}
    for axis in axes:
        for value in (
            axis.name,
            axis.table_entry,
            axis.axis_entry,
            axis.coordinate,
            axis.out_name,
            axis.generic_level_name,
        ):
            if value:
                name_to_axis.setdefault(str(value), axis)

    grid_dims = [str(d) for d in grid.dimensions]

    for dim_name in grid_dims:
        if dim_name not in name_to_axis:
            raise TableValidationError(
                f"Grid dimension {dim_name!r} does not correspond to any "
                "of the supplied axes.  Each grid spatial dimension must "
                "match an axis by name, out_name, or table_entry."
            )

    lat = grid.latitude
    lon = grid.longitude
    for array, label in ((lat, "latitude"), (lon, "longitude")):
        if array is None:
            continue
        arr = np.asarray(array)
        for i, dim_name in enumerate(grid_dims):
            axis = name_to_axis[dim_name]
            axis_len = len(axis.values_array())
            if arr.shape[i] != axis_len:
                raise TableValidationError(
                    f"Grid {label} array shape {arr.shape} does not match "
                    f"the axis lengths implied by grid dimensions "
                    f"{grid_dims!r}. "
                    f"Dimension {dim_name!r} (index {i}) has {axis_len} "
                    f"coordinate values but {label} has size "
                    f"{arr.shape[i]} along that axis."
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


def _build_table_info(table_file: Path) -> str:
    """Build a table_info string with the filename, creation date, and MD5 hash.

    Matches the format written by CMOR3:
    ``Name: <file>; Creation Date:(<date>) MD5:<hash>``
    """
    try:
        raw = table_file.read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        mtime = datetime.fromtimestamp(table_file.stat().st_mtime, timezone.utc)
        creation_date_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"Name: {table_file.name};"
            f" Creation Date:({creation_date_str})"
            f" MD5:{md5}"
        )
    except OSError:
        return f"Name: {table_file.name};"
