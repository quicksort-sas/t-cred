from __future__ import annotations

from datetime import date

from tcred.dataset.models import TemporalInterval


def make_interval(
    start_year: int,
    end_year: int | None,
    *,
    start_month: int = 1,
    start_day: int = 1,
    end_month: int = 12,
    end_day: int = 31,
    interval_type: str | None = None,
) -> TemporalInterval:
    start = date(start_year, start_month, start_day)
    end = None if end_year is None else date(end_year, end_month, end_day)
    type_name = interval_type or ("open_interval" if end is None else "interval")
    return TemporalInterval(type=type_name, start=start, end=end)


def point(year: int, month: int = 6, day: int = 1) -> TemporalInterval:
    value = date(year, month, day)
    return TemporalInterval(type="point", start=value, end=value)


def unknown_interval() -> TemporalInterval:
    return TemporalInterval(type="unknown", start=None, end=None, granularity="unknown")


def fmt_date(value: date | None) -> str:
    if value is None:
        return "present"
    return value.strftime("%B %-d, %Y") if hasattr(value, "strftime") else str(value)


def human_interval(interval: TemporalInterval) -> str:
    if interval.start is None:
        return "an unknown period"
    start = interval.start.strftime("%B %d, %Y").replace(" 0", " ")
    if interval.end is None:
        return f"since {start}"
    end = interval.end.strftime("%B %d, %Y").replace(" 0", " ")
    if interval.start == interval.end:
        return f"on {start}"
    return f"from {start} to {end}"
