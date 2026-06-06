from __future__ import annotations

import re
from typing import Any, Mapping


def render_template(template: str,
                    tokens: Mapping[str, Any],
                    separator: str | None = None) -> str:
    _template = template
    if separator:
        _template = re.sub("><", f">{separator}<", _template)
    return re.sub(
        r"<([^>]+)>",
        lambda match: str(tokens.get(match.group(1), "")),
        _template,
    )
