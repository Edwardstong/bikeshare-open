"""Stage 3: temporal split, fit, evaluate, write metrics.

Two baselines and two models, on purpose.

  baseline_mean         predict the 2011 average for every day
  baseline_persistence  predict yesterday's count (cnt_lag_1)
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence
is the baseline that actually matters: ridge's features are dominated by
lagged demand, so if it cannot beat "yesterday again" it has learned
nothing worth reporting. The booster is there because it fails, and the
reason it fails is the headline result.

Run: python -m src.train
Exercise 4 only: python -m src.train --target ratio --splits valid --out <file>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS, PROCESSED, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit."""

    def __init__(self, column: str = "cnt_lag_1"):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"persistence baseline needs column {self.column!r}")
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_persistence": PersistenceBaseline("cnt_lag_1"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=300),
}


# --- Exercise 4 only: target transformations ----------------------------
# reports/metrics.json is always the untransformed pipeline ("none").
# Each entry is (forward: frame -> training target, inverse: frame, pred ->
# rides). Every transform uses only quantities known at prediction time
# (cnt_roll_7 and cnt_lag_1 are shifted), and scores are always computed
# in rides. Baselines keep their definitions; only ridge and hgb change.
TARGET_TRANSFORMS = {
    "none": (lambda d: d[TARGET], lambda d, p: p),
    # Today relative to the trailing 7-day mean. Scale-free, so 2012's
    # growth stops pushing targets above anything a tree saw in 2011.
    "ratio": (
        lambda d: d[TARGET] / d["cnt_roll_7"],
        lambda d, p: p * d["cnt_roll_7"].to_numpy(),
    ),
    "logratio": (
        lambda d: np.log(d[TARGET] / d["cnt_roll_7"]),
        lambda d, p: np.exp(p) * d["cnt_roll_7"].to_numpy(),
    ),
    # Today minus yesterday: the model predicts the change from persistence.
    "diff": (
        lambda d: d[TARGET] - d["cnt_lag_1"],
        lambda d, p: p + d["cnt_lag_1"].to_numpy(),
    ),
}
TRANSFORMED_MODELS = {"ridge", "hgb"}


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test."""
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]
    return train, valid, test


def fit_and_score(target: str = "none", splits: tuple = ("valid", "test")) -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)
    parts = {"valid": valid, "test": test}
    forward, inverse = TARGET_TRANSFORMS[target]

    results = {
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    }
    if target != "none":
        results["target_transform"] = target
    for name, model in MODELS.items():
        transformed = target != "none" and name in TRANSFORMED_MODELS
        model.fit(train[cols], forward(train) if transformed else train[TARGET])

        def predict(part, model=model, transformed=transformed):
            pred = model.predict(part[cols])
            return inverse(part, pred) if transformed else pred

        results[name] = {
            split: {
                "r2": round(float(r2_score(parts[split][TARGET], predict(parts[split]))), 4),
                "mae": round(float(mean_absolute_error(parts[split][TARGET], predict(parts[split]))), 1),
            }
            for split in splits
        }
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGET_TRANSFORMS), default="none")
    ap.add_argument("--splits", default="valid,test",
                    help="comma-separated; use 'valid' while iterating")
    ap.add_argument("--out", type=Path, default=METRICS)
    ns = ap.parse_args(argv)
    splits = tuple(ns.splits.split(","))
    if ns.out.resolve() == METRICS.resolve() and (ns.target != "none" or splits != ("valid", "test")):
        raise SystemExit(
            "reports/metrics.json is the untransformed pipeline on both splits; "
            "write Exercise 4 runs elsewhere with --out"
        )
    res = fit_and_score(ns.target, splits)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
