from __future__ import annotations


def units_are_convertible(user_units: str, table_units: str) -> bool:
    """Return True if *user_units* and *table_units* are dimensionally
    compatible.

    Uses ``cf_units`` (a required dependency) for a proper udunits-based check,
    so that physically equivalent unit strings such as ``"degC"`` and ``"K"``
    or ``"hPa"`` and ``"Pa"`` are accepted even when the strings differ.

    If the unit strings cannot be parsed by ``cf_units`` (e.g. non-standard
    strings) the function falls back to requiring exact string equality.

    Parameters
    ----------
    user_units
        Units string supplied by the user.
    table_units
        Units string from the project table entry.

    Returns
    -------
    bool
        True when the units are the same string or are dimensionally
        convertible; False when they belong to different physical dimensions.

    Examples
    --------
    >>> units_are_convertible("degC", "K")
    True
    >>> units_are_convertible("hPa", "Pa")
    True
    >>> units_are_convertible("m s-1", "K")
    False
    >>> units_are_convertible("kg m-2 s-1", "kg m-2 s-1")
    True
    """
    if user_units == table_units:
        return True
    try:
        import cf_units

        a = cf_units.Unit(user_units)
        b = cf_units.Unit(table_units)
        return a.is_convertible(b)
    except Exception:
        # Unit strings not parseable by cf_units: require exact equality.
        return False
