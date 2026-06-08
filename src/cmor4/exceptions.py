class AxisValidationError(ValueError):
    """Raised when axis values or bounds violate axis contraints.

    Parameters
    ----------
    *args:
        Error message arguments forwarded to ``ValueError``.
    """


class ControlledVocabularyError(ValueError):
    """Raised when a dataset attribute is not allowed by the project CV.

    Parameters
    ----------
    *args:
        Error message arguments forwarded to ``ValueError``.
    """


class TableValidationError(ValueError):
    """Raised when user input is not allowed by project tables.

    Parameters
    ----------
    *args:
        Error message arguments forwarded to ``ValueError``.
    """


class VariableValidationError(ValueError):
    """Raised when variable values violate checks.

    Parameters
    ----------
    *args:
        Error message arguments forwarded to ``ValueError``.
    """
