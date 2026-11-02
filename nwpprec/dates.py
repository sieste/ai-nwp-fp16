"""Initialisation dates: one per week of a year, and a seasonally balanced
subset of them for the component experiments."""

from datetime import datetime, timedelta

import numpy as np

INIT_HOURS = ("00", "06", "12", "18")


def weekly_dates(year: int = 2024, seed: int = 42) -> list:
    """52 (date "YYYY-MM-DD", init hour "HH") pairs, one per week of `year`.

    The weekday and the initialisation hour of each week's date are drawn
    from shuffled, balanced pools (each hour 13 times, each weekday 7 or 8
    times), so that neither is confounded with the season or with each
    other. Returned sorted by date.
    """
    n_weeks = 52
    rng = np.random.default_rng(seed)
    weekdays = (list(range(7)) * (n_weeks // 7 + 1))[:n_weeks]
    rng.shuffle(weekdays)
    hours = list(INIT_HOURS) * (n_weeks // len(INIT_HOURS))
    rng.shuffle(hours)
    dates = []
    for week in range(n_weeks):
        start = datetime(year, 1, 1) + timedelta(days=7 * week)
        end = datetime(year, 12, 31) if week == n_weeks - 1 else start + timedelta(days=6)
        date = min(start + timedelta(days=(weekdays[week] - start.weekday()) % 7), end)
        dates.append((date.strftime("%Y-%m-%d"), hours[week]))
    return sorted(dates)


def balanced_subset(dates: list, n: int = 5) -> list:
    """n of the sorted `dates`: the date nearest the centre of each of n equal
    parts of the year, skipping dates whose initialisation hour is already
    used until every hour is represented."""
    all_hours = {h for _, h in dates}
    picked, used = [], set()
    for k in range(n):
        target = (k + 0.5) * len(dates) / n
        for i in sorted(range(len(dates)), key=lambda i: abs(i - target)):
            if i in picked or (len(used) < len(all_hours) and dates[i][1] in used):
                continue
            picked.append(i)
            used.add(dates[i][1])
            break
    return [dates[i] for i in sorted(picked)]


def init_time(date: str, hour: str) -> datetime:
    return datetime.strptime(f"{date} {hour}", "%Y-%m-%d %H")
