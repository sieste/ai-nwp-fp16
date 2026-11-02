from collections import Counter

from nwpprec.dates import balanced_subset, weekly_dates


def test_weekly_dates_balanced():
    dates = weekly_dates(2024, 42)
    assert len(dates) == 52 and dates == sorted(dates)
    assert set(Counter(h for _, h in dates).values()) == {13}


def test_oat_dates():
    # the five component-experiment dates used throughout the study
    assert balanced_subset(weekly_dates(2024, 42)) == [
        ("2024-02-08", "00"), ("2024-04-26", "06"), ("2024-06-24", "18"), ("2024-08-29", "12"), ("2024-11-26", "12")]
