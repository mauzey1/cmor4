class AxisValidationError(ValueError):
    """Raised when axis values or bounds violate axis constraints.

    This exception is raised during axis validation when coordinate values,
    bounds, or metadata do not meet the requirements defined by the axis
    metadata or project coordinate tables. Common causes include non-monotonic
    coordinates, bounds that don't bracket values, out-of-range values, or
    inconsistent shapes between values and bounds.

    Parameters
    ----------
    *args
        Error message arguments forwarded to ``ValueError``.

    Examples
    --------
    Raised when axis values are not monotonically increasing::

        axis = Axis(name="lat", values=[90, 45, 0, -45, -90])
        # AxisValidationError: Axis 'lat' values must be monotonic

    Raised when bounds don't match values::

        axis = Axis(name="time", values=[0, 1, 2], bounds=[[0, 1]])
        # AxisValidationError: time bounds shape (1, 2) inconsistent
    """


class ControlledVocabularyError(ValueError):
    """Raised when a dataset attribute is not allowed by the project CV.

    This exception is raised when dataset-level metadata validation fails
    against the project's controlled vocabulary (CV). Common causes include
    missing required global attributes, invalid values for CV-controlled
    fields (e.g., unrecognized experiment_id, source_id), or inconsistent
    combinations of attributes (e.g., mismatched parent experiment metadata).

    Parameters
    ----------
    *args
        Error message arguments forwarded to ``ValueError``.

    Examples
    --------
    Raised when a required global attribute is missing::

        dataset = DatasetInfo({"mip_era": "CMIP7"})
        # ControlledVocabularyError: Required attribute 'institution_id'
        # is missing

    Raised when an attribute value is not in the CV::

        dataset = DatasetInfo({"experiment_id": "unknown-experiment"})
        # ControlledVocabularyError: experiment_id 'unknown-experiment'
        # not found in CV
    """


class TableValidationError(ValueError):
    """Raised when user input is not allowed by project tables.

    This exception is raised when metadata provided by the user does not match
    the requirements defined in project variable, coordinate, grid, or formula
    tables. Common causes include mismatched variable dimensions, incorrect
    units or standard_name values, ambiguous variable or coordinate names,
    or inconsistencies between user-supplied metadata and table entries.

    Parameters
    ----------
    *args
        Error message arguments forwarded to ``ValueError``.

    Examples
    --------
    Raised when variable dimensions don't match table entry::

        variable = Variable(name="tas", dimensions=("time", "lat"))
        # TableValidationError: dimensions=('time', 'lat') does not match
        # Amon:tas dimensions ('time', 'lat', 'lon')

    Raised when variable name is ambiguous across tables::

        variable = Variable(name="tas")
        # TableValidationError: Variable 'tas' is ambiguous across loaded
        # tables; specify table_id. Choices: Amon:tas, day:tas

    Raised when user-supplied metadata conflicts with table::

        axis = Axis(name="time", units="days since 2000-01-01",
                    standard_name="wrong_name")
        # TableValidationError: axis 'time' standard_name='wrong_name' does
        # not match table value 'time'
    """


class VariableValidationError(ValueError):
    """Raised when variable values violate validation checks.

    This exception is raised when the data values for a variable or formula
    term fail validation against constraints defined in the variable or
    formula table entry. Common causes include NaN or Inf values exceeding
    tolerance thresholds, values outside valid_min/valid_max ranges, or
    absolute mean values outside ok_min_mean_abs/ok_max_mean_abs ranges.

    Parameters
    ----------
    *args
        Error message arguments forwarded to ``ValueError``.

    Examples
    --------
    Raised when data contains too many NaN values::

        variable = Variable(name="tas", ...)
        data = np.array([np.nan, np.nan, 280.0])
        # VariableValidationError: Variable 'tas' contains 66.7% NaN values

    Raised when values exceed valid range::

        variable = Variable(name="tas", valid_min=200.0, valid_max=330.0)
        data = np.array([180.0, 280.0, 350.0])
        # VariableValidationError: Variable 'tas' contains values outside
        # valid range [200.0, 330.0]

    Raised when absolute mean is out of range::

        variable = Variable(name="tas", ok_min_mean_abs=250.0)
        data = np.array([10.0, 20.0, 30.0])
        # VariableValidationError: Variable 'tas' absolute mean 20.0 is
        # below ok_min_mean_abs=250.0
    """
