class TableValidationError(ValueError):
    """Raised when user input is not allowed by project tables."""


class AxisValidationError(ValueError):
    """Raised when axis values or bounds violate axis contraints."""
