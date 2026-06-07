from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import cftime


def decode_time_value(
    value: Any, units: Any, calendar: Any = "standard"
) -> Any | None:
    if np.issubdtype(np.asarray(value).dtype, np.datetime64):
        text = np.datetime_as_string(value, unit="s")
        return datetime.fromisoformat(text)
    if not units:
        return None
    units_text = normalize_time_units(str(units))
    if cftime is not None:
        try:
            return cftime.num2date(
                float(value),
                units_text,
                calendar=str(calendar or "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=False,
            )
        except Exception:
            pass
    match = re.match(
        r"^(days|hours|minutes|seconds) since "
        r"(\d{1,4})-(\d{1,2})-(\d{1,2})",
        units_text,
    )
    if not match:
        return None
    unit, year, month, day = match.groups()
    base = datetime(int(year), int(month), int(day))
    numeric = float(value)
    if unit == "days":
        return base + timedelta(days=numeric)
    if unit == "hours":
        return base + timedelta(hours=numeric)
    if unit == "minutes":
        return base + timedelta(minutes=numeric)
    return base + timedelta(seconds=numeric)


def cftime_interval_days(
    values: np.ndarray, units: str, calendar: str
) -> np.ndarray | None:
    if " since " not in units:
        return None
    try:
        dates = cftime.num2date(
            values.astype("f8"),
            normalize_time_units(units),
            calendar=calendar or "standard",
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
    except Exception:
        return None

    elapsed_days: list[float] = []
    for start, end in zip(dates[:-1], dates[1:]):
        seconds = _elapsed_seconds(start, end)
        if seconds is None:
            return None
        elapsed_days.append(abs(seconds) / 86400.0)
    return np.asarray(elapsed_days, dtype="f8")


def _elapsed_seconds(start: Any, end: Any) -> float | None:
    try:
        delta = end - start
    except TypeError:
        return None
    if hasattr(delta, "total_seconds"):
        return float(delta.total_seconds())
    days = getattr(delta, "days", None)
    seconds = getattr(delta, "seconds", 0)
    microseconds = getattr(delta, "microseconds", 0)
    if days is None:
        return None
    return float(days * 86400.0 + seconds + microseconds / 1.0e6)


def normalize_time_units(units: str) -> str:
    match = re.match(
        r"^(\w+) since (\d{1,4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?(.*)$",
        units,
    )
    if not match:
        return units
    unit, year, month, day, suffix = match.groups()
    return (
        f"{unit} since {int(year):04d}-{int(month or 1):02d}-"
        f"{int(day or 1):02d}{suffix or ''}"
    )


def add_time_delta(value: Any, delta: timedelta) -> Any:
    try:
        return value + delta
    except TypeError:
        return (
            datetime(
                value.year,
                value.month,
                value.day,
                int(value.hour),
                int(value.minute),
                int(value.second),
            )
            + delta
        )


def date_part(value: Any, precision: str) -> str:
    if precision == "minute":
        value = add_time_delta(value, timedelta(seconds=30))
    else:
        value = add_time_delta(value, timedelta(seconds=0.5))
    if precision == "year":
        return f"{value.year:04d}"
    if precision == "month":
        return f"{value.year:04d}{value.month:02d}"
    if precision == "day":
        return f"{value.year:04d}{value.month:02d}{value.day:02d}"
    if precision == "minute":
        return (
            f"{value.year:04d}{value.month:02d}{value.day:02d}"
            f"{value.hour:02d}{value.minute:02d}"
        )
    return (
        f"{value.year:04d}{value.month:02d}{value.day:02d}"
        f"{value.hour:02d}{value.minute:02d}{value.second:02d}"
    )
