"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from src.config import DECLARED_FEATURES, FORBIDDEN, TARGET
from src.features import build_features, encode_weather, feature_columns
from src.ingest import load_raw


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_feature_set_is_declared(feats):
    """The model's inputs must be exactly DECLARED_FEATURES in src/config.py.

    test_no_forbidden_features cannot fail by construction once
    feature_columns() excludes FORBIDDEN. This one can: a leaked column
    copied under an innocent name (out["walkins"] = out.casual) is an
    undeclared input, and a feature that silently vanishes is a missing one.
    """
    cols = set(feature_columns(feats))
    assert cols == DECLARED_FEATURES, (
        f"undeclared inputs: {sorted(cols - DECLARED_FEATURES)}; "
        f"declared but missing: {sorted(DECLARED_FEATURES - cols)}"
    )


def test_no_feature_pair_restates_the_target(feats):
    """No feature, or pair of features, reproduces cnt exactly (least squares
    with an intercept). Catches casual + registered under any names, even
    if someone also adds the new names to DECLARED_FEATURES to quiet the
    test above."""
    cols = feature_columns(feats)
    y = feats[TARGET].to_numpy(dtype=float)
    hits = []
    for r in (1, 2):
        for subset in combinations(cols, r):
            X = np.column_stack(
                [feats[list(subset)].to_numpy(dtype=float), np.ones(len(feats))]
            )
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            if np.abs(X @ coef - y).max() < 1e-6:
                hits.append(subset)
    assert not hits, f"features that restate {TARGET} exactly: {hits}"


def test_lags_are_true_calendar_lags(feats):
    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
    by_date = feats.set_index("dteday")[TARGET]
    date = feats.dteday.iloc[200]
    window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
    if all(d in by_date.index for d in window):
        expected = sum(by_date[d] for d in window) / 7
        assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)


def test_weather_lag_1_is_yesterdays_weather(feats):
    """weather_severity_lag_1 on date d is the weather severity of d-1 in
    calendar days: strictly earlier information only.

    Named date, computed by hand from the raw file: 2011-04-16 was BAD (2),
    and 2011-04-17 itself was GOOD (0). A feature that read the same day,
    or a row-shift across the 2011-03-10 gap, gives a different number.
    """
    by_date = feats.set_index("dteday")
    assert by_date.loc[pd.Timestamp("2011-04-17"), "weather_severity_lag_1"] == 2.0

    # The same property on every row whose previous calendar day is in raw.
    raw = load_raw().set_index("dteday")
    severity = encode_weather(raw.weather)
    checked = 0
    for date, lag in by_date.weather_severity_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in severity.index:
            assert lag == severity[prev], f"bad weather lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique
