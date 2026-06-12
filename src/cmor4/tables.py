from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._table_utils import (
    is_table_value as _is_table_value,
    single_or_original as _single_or_original,
)
from ._templates import is_unresolved_template as _is_unresolved_template
from .axis import Axis
from .cv import ControlledVocabulary
from .dataset import DatasetInfo
from .exceptions import TableValidationError
from .grid import Grid
from .variable import Variable, VariableEntry
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
        self.variable_table_files = tuple(
            Path(path) for path in variable_tables
        )
        self.coordinate_table_file = (
            Path(coordinate_table) if coordinate_table is not None else None
        )
        self.formula_table_file = (
            Path(formula_table) if formula_table is not None else None
        )
        self.grid_table_file = (
            Path(grid_table) if grid_table is not None else None
        )
        self.cv = ControlledVocabulary.from_file(self.cv_file)
        self.variable_entries: dict[str, VariableEntry] = {}
        self._variable_entries_by_name: dict[str, list[VariableEntry]] = {}
        for table_file in self.variable_table_files:
            self._load_variable_table(table_file)
        self.grid_axis_entries: dict[str, Mapping[str, Any]] = {}
        self.grid_coordinate_entries: dict[str, Mapping[str, Any]] = {}
        self.grid_mapping_entries: dict[str, Mapping[str, Any]] = {}
        if self.grid_table_file is not None:
            self.grid_axis_entries = self._read_entries(
                self.grid_table_file, "axis_entry"
            )
            self.grid_coordinate_entries = self._read_entries(
                self.grid_table_file, "variable_entry"
            )
            self.grid_mapping_entries = self._read_entries(
                self.grid_table_file, "mapping_entry"
            )
        coordinate_entries: dict[str, Mapping[str, Any]] = {}
        if self.coordinate_table_file is not None:
            coordinate_entries = self._read_entries(
                self.coordinate_table_file, "axis_entry"
            )
        self.coordinate_entries = _overlay_table_entries(
            coordinate_entries, self.grid_axis_entries
        )
        self.scalar_axis_entries = {
            name: entry
            for name, entry in self.coordinate_entries.items()
            if _is_table_value(entry.get("value"))
        }
        self.generic_level_entries = _generic_level_entries(
            self.coordinate_entries
        )
        self.formula_entries: dict[str, Mapping[str, Any]] = {}
        if self.formula_table_file is not None:
            self.formula_entries = self._read_entries(
                self.formula_table_file, "formula_entry"
            )

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
        resolved_grid_table = _resolve_optional_table(
            root_path, grid_table, "grids"
        )
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

        user_info = (
            dataset.user_info
            if isinstance(dataset, DatasetInfo)
            else dataset
        )
        normalized_dataset = self.cv.get_dataset_info(dataset)
        self.cv.validate_dataset_values(normalized_dataset)
        self.validate_source_attributes(normalized_dataset)
        self.validate_experiment(normalized_dataset)
        self.validate_parent_attributes(normalized_dataset)
        return DatasetInfo(
            normalized_dataset,
            project=self,
            user_info=user_info,
        )

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
        user_info = dataset.user_info
        normalized_dataset = self.cv.get_dataset_info(dataset)
        variable_entry = variable.resolve_table_entry(self)
        self._add_table_header_defaults(normalized_dataset, variable_entry)
        self._add_variable_global_defaults(
            normalized_dataset, variable
        )
        self.validate_dataset(normalized_dataset)
        self.validate_source_attributes(normalized_dataset)
        self.validate_experiment(normalized_dataset)
        self.validate_parent_attributes(normalized_dataset)

        # Note: Full variable and dataset-variable consistency validation
        # happens in validate_components, not here
        prepared_dataset = DatasetInfo(
            normalized_dataset,
            project=self,
            user_info=user_info,
        )

        # Quick validation check for dataset-variable consistency
        # This is duplicated in validate_components but done early for fast
        # failure
        variable.validate_against_entry(variable_entry)
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

        return Variable(name=name, project=self, **values)

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

        return self._mark_prepared_axis(
            Axis(name=name, project=self, **values)
        )

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
            axis if self._is_prepared_axis(axis)
            else axis._merge_table_entry(self)
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
        for dimension in variable.get("dimensions", ()):
            dimension_name = str(dimension)
            if dimension_name in present:
                continue
            if dimension_name not in self.scalar_axis_entries:
                continue
            axis = Axis(
                name=dimension_name,
                table_entry=dimension_name,
                scalar=True,
                project=self,
            )
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

    def grid(self, name: str | None = None, **values: Any) -> Grid:
        """Create a grid with metadata from the loaded grid table.

        This factory method creates a ``Grid`` metadata record for variables
        on non-rectilinear grids or with coordinate reference systems. It
        resolves grid mapping entries from the project's grid table and merges
        projection parameters with user-provided values.

        Parameters
        ----------
        name
            Optional grid mapping entry name (e.g., "lambert_conformal_conic",
            "rotated_latitude_longitude"). If None, the grid must specify
            mapping metadata via other parameters.
        **values
            User-supplied grid metadata. Common keywords include dimensions
            (for grid dimension override), mapping_name or grid_mapping_name,
            params (projection parameters dict), coordinates (auxiliary
            coordinate names), mapping_var (grid mapping variable name), and
            attrs for additional attributes.

        Returns
        -------
        Grid
            Grid metadata record with table values and projection parameters
            merged.

        Examples
        --------
        Create grid for Lambert Conformal Conic projection::

            project = ProjectTables.from_directory(...)
            grid = project.grid(
                "lambert_conformal_conic",
                params={
                    "standard_parallel": ([30.0, 60.0], "degrees_north"),
                    "longitude_of_central_meridian": (-100.0, "degrees_east")
                }
            )

        Create grid with dimension override for curvilinear ocean::

            grid = project.grid(
                dimensions=("j", "i"),
                coordinates=["nav_lat", "nav_lon"]
            )

        Create rotated pole grid::

            grid = project.grid(
                "rotated_latitude_longitude",
                params={
                    "grid_north_pole_latitude": (37.5, "degrees_north"),
                    "grid_north_pole_longitude": (-177.5, "degrees_east")
                }
            )
        """

        return Grid(name=name, project=self, **values)

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

        return ZFactor(name=name, project=self, **values)

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

        # Variable validation: ensure stored attributes match table entry
        # Note: This may be redundant with validation in _dataset_for_variable,
        # but we validate again here to ensure consistency when called directly
        # by users or if variable was modified after _dataset_for_variable
        variable_entry = variable.resolve_table_entry(self)
        variable.validate_against_entry(variable_entry)

        # Dataset-variable consistency checks
        if dataset is not None:
            self._validate_dataset_variable_consistency(
                dataset, variable, variable_entry
            )

        # Axis validation: only validate axes not prepared by this instance
        for axis in axes:
            if not self._is_prepared_axis(axis):
                # Validate axis against coordinate table entry
                entry_name, entry = axis.resolve_table_entry(self)
                if entry is not None:
                    axis._validate_metadata(
                        "axis",
                        entry_name,
                        entry,
                        (
                            "units",
                            "standard_name",
                            "long_name",
                            "axis",
                            "positive",
                            "formula",
                        ),
                    )
                # Validate axis against grid coordinate entry (if applicable)
                grid_entry_name, grid_entry = (
                    axis.resolve_grid_coordinate(self)
                )
                if grid_entry is not None:
                    axis._validate_metadata(
                        "grid coordinate",
                        grid_entry_name,
                        grid_entry,
                        ("units", "standard_name", "long_name"),
                    )

        # Dataset-axis consistency checks (e.g., time axis needs frequency)
        if dataset is not None:
            self._validate_dataset_axis_consistency(dataset, variable, axes)

        # Grid validation: ensure stored attributes match tables
        if grid is not None:
            entry_name, entry = grid.resolve_table_entry(self)
            if entry is not None:
                user_values = grid.to_dict()
                for key in ("mapping_name", "grid_mapping_name"):
                    expected = entry.get(key)
                    if (
                        _is_table_value(expected)
                        and key in user_values
                        and str(user_values[key]) != str(expected)
                    ):
                        raise TableValidationError(
                            f"grid mapping {entry_name!r} {key}="
                            f"{user_values[key]!r} does not match table value "
                            f"{expected!r}."
                        )

        # ZFactor validation: ensure stored attributes match tables
        for zfactor in zfactors:
            entry_name, entry = zfactor.resolve_table_entry(self)
            if entry is not None:
                zfactor._validate_metadata(
                    "formula term",
                    entry_name,
                    entry,
                    ("units", "standard_name", "long_name"),
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
            and "frequency" in variable
            and str(dataset["frequency"]) != str(variable["frequency"])
        ):
            raise TableValidationError(
                f"Dataset frequency={dataset['frequency']!r} does not match "
                f"variable {variable_entry.table_id}:{variable_entry.name} "
                f"frequency={variable['frequency']!r}."
            )

    def _validate_dataset_axis_consistency(
        self,
        dataset: DatasetInfo,
        variable: Variable,
        axes: Sequence[Axis],
    ) -> None:
        """Validate axes in context of dataset.

        For example, time axis with frequency. This is a placeholder for
        cross-component validation checks that require dataset context. The
        main time axis validation with frequency happens in
        validate_and_normalize_axes during dataset creation, but this could
        be extended to perform additional checks.
        """
        # Check that required scalar axes are present (if not auto-added)
        present_names = {
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

        for dimension in variable.get("dimensions", ()):
            dimension_name = str(dimension)
            if dimension_name not in present_names:
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
                if key not in dataset or _is_unresolved_template(
                    dataset[key]
                ):
                    dataset[key] = labels[key]
        for key in ("frequency", "realm", "table_id"):
            value = variable.get(key)
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

    def validate_required_global_attributes(
        self, dataset: Mapping[str, Any]
    ) -> None:
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
    def _read_entries(
        table_file: Path, key: str
    ) -> dict[str, Mapping[str, Any]]:
        with table_file.open() as handle:
            data = json.load(handle)
        return {
            str(name): entry
            for name, entry in data.get(key, {}).items()
            if isinstance(entry, Mapping)
        }

    def _load_variable_table(self, table_file: Path) -> None:
        with table_file.open() as handle:
            data = json.load(handle)
        entries = data.get("variable_entry", {})
        table_id = str(
            data.get("Header", {}).get("table_id") or table_file.stem
        )
        if table_id.startswith("Table "):
            table_id = table_id.removeprefix("Table ")
        for name, entry in entries.items():
            variable_entry = VariableEntry(
                name=name,
                table_id=str(table_id),
                entry=entry,
                table_file=table_file,
                table_header=data.get("Header", {}),
            )
            self.variable_entries.setdefault(name, variable_entry)
            self._variable_entries_by_name.setdefault(name, []).append(
                variable_entry
            )


def _generic_level_entries(
    entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Mapping[str, Any]]]:
    generic_entries: dict[str, dict[str, Mapping[str, Any]]] = {}
    for name, entry in entries.items():
        generic_level_name = entry.get("generic_level_name")
        if _is_table_value(generic_level_name):
            generic_entries.setdefault(str(generic_level_name), {})[
                str(name)
            ] = entry
    return generic_entries


def _overlay_table_entries(
    base_entries: Mapping[str, Mapping[str, Any]],
    overlay_entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    entries = {name: dict(entry) for name, entry in base_entries.items()}
    for name, overlay in overlay_entries.items():
        entry = dict(entries.get(name, {}))
        for key, value in overlay.items():
            if _is_table_value(value) or key not in entry:
                entry[key] = value
        entries[name] = entry
    return entries


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
