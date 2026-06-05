from __future__ import annotations

import re
from typing import Any, Mapping


def render_template(template: str, tokens: Mapping[str, Any]) -> str:
    return re.sub(
        r"<([^>]+)>",
        lambda match: str(tokens.get(match.group(1), "")),
        template,
    )
