class AxisValidationError(ValueError):
    """Raised when axis values or bounds violate axis contraints."""


class ControlledVocabularyError(ValueError):
    """Raised when dataset attribute is not allowed by project CV."""


class TableValidationError(ValueError):
    """Raised when user input is not allowed by project tables."""


class VariableValidationError(ValueError):
    """Raised when variable values violate checks."""
