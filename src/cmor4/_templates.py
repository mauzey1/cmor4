from __future__ import annotations

import re
from typing import Any, Mapping


def render_template(template: str,
                    tokens: Mapping[str, Any],
                    separator: str | None = None) -> str:
    """Render a template string by replacing tokens with values.

    Template tokens are placeholders in the form <token_name> that get replaced
    with corresponding values from the tokens mapping.

    This function is permissive and will match any content between angle
    brackets, including text with spaces like <Your Centre Name>. For strict
    detection of actual template tokens (no whitespace), use
    is_unresolved_template().

    Parameters
    ----------
    template:
        Template string containing tokens like <variable_id>, <mip_era>, etc.
    tokens:
        Mapping of token names to their replacement values.
    separator:
        Optional separator to insert between adjacent tokens
        (e.g., "-" or "_").

    Returns
    -------
    str
        Rendered template with tokens replaced by their values. Tokens not
        found in the mapping are replaced with empty strings.

    Notes
    -----
    This function uses a permissive pattern `<([^>]+)>` that matches any
    content between angle brackets, including whitespace. This allows it to
    handle placeholder text like `<Your Centre Name>` from CV files.
    """
    _template = template
    if separator:
        _template = re.sub("><", f">{separator}<", _template)
    return re.sub(
        r"<([^>]+)>",
        lambda match: str(tokens.get(match.group(1), "")),
        _template,
    )


def is_unresolved_template(value: Any) -> bool:
    """Check if a value contains unresolved template tokens.

    This function uses a strict pattern that only matches template tokens with
    no whitespace, e.g., <variable_id>, <mip_era>, <source_id>. This helps
    distinguish actual template tokens from other uses of angle brackets.

    For rendering templates (including those with whitespace), use
    render_template() which is more permissive.

    Parameters
    ----------
    value:
        Value to check for template tokens.

    Returns
    -------
    bool
        True if value is a string containing template tokens (no whitespace),
        False otherwise.

    Examples
    --------
    >>> is_unresolved_template("<variable_id>")
    True
    >>> is_unresolved_template("<mip_era>_<source_id>")
    True
    >>> is_unresolved_template("< >")
    False
    >>> is_unresolved_template("<Your Centre Name>")
    False
    >>> is_unresolved_template("x < y")
    False

    Notes
    -----
    This function uses the strict pattern r"<\\S+>" (non-whitespace only),
    while render_template() uses the permissive pattern r"<([^>]+)>"
    (any content).
    """
    return isinstance(value, str) and bool(re.search(r"<\S+>", value))
