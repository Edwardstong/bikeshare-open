"""Exercise 5: is ridge worth deploying over "yesterday again"?

Ridge beats persistence on R^2 and loses on MAE. Those two metrics weigh
errors differently (squared vs absolute), so this module looks at where
each model's errors come from instead of picking a metric to win on.

Uses the final pipeline only (no fitting choices are made here), so
reading the test split is evaluation, not tuning.

Run: python -m src.compare   ->  reports/ridge_vs_persistence.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import PROCESSED, ROOT, TARGET
from src.features import feature_columns
from src.train import MODELS, temporal_split

OUT = ROOT / "reports" / "ridge_vs_persistence.json"
SEED = 0
N_BOOT = 2000
BLOCK = 7  # days; errors are autocorrelated, so resample whole weeks


def _summary(y, pred_ridge, pred_pers) -> dict:
    er, ep = pred_ridge - y, pred_pers - y
    return {
        "n_days": int(len(y)),
        "ridge": {
            "r2": round(float(r2_score(y, pred_ridge)), 4),
            "mae": round(float(mean_absolute_error(y, pred_ridge)), 1),
            "rmse": round(float(np.sqrt(np.mean(er**2))), 1),
            "median_abs_err": round(float(np.median(np.abs(er))), 1),
            "mean_err": round(float(er.mean()), 1),
        },
        "persistence": {
            "r2": round(float(r2_score(y, pred_pers)), 4),
            "mae": round(float(mean_absolute_error(y, pred_pers)), 1),
            "rmse": round(float(np.sqrt(np.mean(ep**2))), 1),
            "median_abs_err": round(float(np.median(np.abs(ep))), 1),
            "mean_err": round(float(ep.mean()), 1),
        },
        "share_days_ridge_closer": round(float((np.abs(er) < np.abs(ep)).mean()), 3),
    }


def _block_bootstrap_ci(abs_diff: np.ndarray, sq_diff: np.ndarray) -> dict:
    """95% moving-block bootstrap CI for mean(|e_ridge| - |e_pers|) and for
    RMSE_ridge - RMSE_pers, resampling 7-day blocks."""
    rng = np.random.default_rng(SEED)
    n = len(abs_diff)
    starts = np.arange(n - BLOCK + 1)
    k = int(np.ceil(n / BLOCK))
    mae_d, rmse_d = [], []
    for _ in range(N_BOOT):
        idx = (rng.choice(starts, k)[:, None] + np.arange(BLOCK)).ravel()[:n]
        mae_d.append(abs_diff[idx].mean())
        sq_r, sq_p = sq_diff[0][idx], sq_diff[1][idx]
        rmse_d.append(np.sqrt(sq_r.mean()) - np.sqrt(sq_p.mean()))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 1), round(float(np.percentile(a, 97.5)), 1)]
    return {"mae_diff_ci95": q(mae_d), "rmse_diff_ci95": q(rmse_d),
            "block_days": BLOCK, "n_boot": N_BOOT, "seed": SEED}


def compare() -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)

    ridge = clone(MODELS["ridge"]).fit(train[cols], train[TARGET])
    pers = clone(MODELS["baseline_persistence"]).fit(train[cols], train[TARGET])

    # "Big move" days: |cnt - cnt_lag_1| above the 75th percentile of
    # TRAINING days, so the threshold is not chosen by looking at test.
    big_move = float(np.percentile(np.abs(train[TARGET] - train.cnt_lag_1), 75))

    out = {"big_move_threshold_rides": round(big_move, 1)}
    for name, part in [("valid", valid), ("test", test)]:
        y = part[TARGET].to_numpy(dtype=float)
        pr, pp = ridge.predict(part[cols]), pers.predict(part[cols])
        res = _summary(y, pr, pp)

        er, ep = np.abs(pr - y), np.abs(pp - y)
        groups = {
            "weather_changed": (part.weather_severity != part.weather_severity_lag_1).to_numpy(),
            "big_move": (np.abs(y - part.cnt_lag_1.to_numpy()) > big_move),
        }
        for g, mask in groups.items():
            res[g] = {
                "n_days": int(mask.sum()),
                "ridge_mae": round(float(er[mask].mean()), 1),
                "persistence_mae": round(float(ep[mask].mean()), 1),
                "ridge_mae_other_days": round(float(er[~mask].mean()), 1),
                "persistence_mae_other_days": round(float(ep[~mask].mean()), 1),
            }
        res["bootstrap"] = _block_bootstrap_ci(er - ep, ((pr - y) ** 2, (pp - y) ** 2))
        out[name] = res

    coefs = ridge[-1].coef_
    order = np.argsort(-np.abs(coefs))
    out["ridge_standardised_coefs"] = {cols[i]: round(float(coefs[i]), 1) for i in order}
    return out


if __name__ == "__main__":
    res = compare()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    t = res["test"]
    print(f"OK compare: test MAE ridge {t['ridge']['mae']} vs persistence "
          f"{t['persistence']['mae']}; ridge closer on {t['share_days_ridge_closer']:.0%} of days "
          f"-> {OUT.name}")
