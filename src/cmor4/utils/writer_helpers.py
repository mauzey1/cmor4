from __future__ import annotations

from typing import Sequence

from ..axis import Axis


def find_time_axis(axes: Sequence[Axis]) -> tuple[int, Axis]:
    """Return the time axis index and axis from a sequence of axes."""

    for index, axis in enumerate(axes):
        names = {
            str(value).lower()
            for value in (
                axis.name,
                axis.out_name,
                axis.table_entry,
                axis.axis_entry,
                axis.coordinate,
                axis.standard_name,
            )
            if value
        }
        if (
            str(axis.axis or "").upper() == "T"
            or "time" in names
            or any(name.startswith("time") for name in names)
        ):
            return index, axis
    raise ValueError(
        "No time axis was found. Pass an axis with axis='T' or time-like "
        "axis metadata."
    )
