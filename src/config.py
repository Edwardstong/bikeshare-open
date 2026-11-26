"""Single source of truth for paths, columns, and split dates."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bike_daily.csv"
PROCESSED = ROOT / "data" / "processed" / "features.parquet"
METRICS = ROOT / "reports" / "metrics.json"

# Daily Capital Bikeshare counts (Fanaee-T & Gama). Canonical home:
#   https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
# We use the processed copy from Molnar's "Interpretable Machine Learning"
# because the UCI host blocks some campus networks. It differs from UCI:
#   - 728 rows, 2011-01-03 to 2012-12-31 (UCI has 731 from 2011-01-01)
#   - 2011-03-10 is absent (the UCI row records humidity 0, a sensor
#     artefact, and the mirror's cleaning dropped the whole day)
#   - `weather` is the labels GOOD / MISTY / BAD, not UCI's codes 1-4
#   - two extra columns: `days_since_2011` and `cnt_2d_bfr`
DATA_URL = (
    "https://raw.githubusercontent.com/christophM/"
    "interpretable-ml-book/master/data/bike.csv"
)

# SHA-256 of the checked-in raw file. Raw data is immutable, but the
# permission rules only bind the Edit/Write tools: a Bash `cp` walks past
# them and past the PostToolUse hook. `make data` and the test suite
# compare against this, so any change to the bytes is loud.
RAW_SHA256 = "a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc"

TARGET = "cnt"

# --- Three reasons a column is kept out of the feature set --------------

# 1. Leakage: the target or an arithmetic restatement of it.
#    casual + registered == cnt exactly, on every row. A model handed these
#    scores R^2 ~ 1.0 and has learned addition. At prediction time you would
#    not know either input: they are components of the thing being predicted.
LEAKAGE_COLS = {"casual", "registered", "cnt"}

# 2. Time indexes: not leakage, but a bad bet. A tree splits on "after date
#    X" and predicts a flat constant for every future date; a linear model
#    extrapolates a two-year trend we have no reason to trust. Excluding
#    them keeps the model comparison about what each does with lagged demand.
TIME_INDEX_COLS = {"instant", "yr", "days_since_2011"}

# 3. Unverified: `cnt_2d_bfr` is a two-day lag of cnt that shipped with the
#    mirror. It is NOT leakage -- a two-day-old count is known at prediction
#    time -- but we did not build it and cannot test how it was aligned, so we
#    build our own lags from the target and test those instead.
UNVERIFIED_COLS = {"cnt_2d_bfr"}

# What feature_columns() and the tests actually enforce.
FORBIDDEN = LEAKAGE_COLS | TIME_INDEX_COLS | UNVERIFIED_COLS

# --- Every other raw column, classified --------------------------------
# FORBIDDEN is a denylist: a column nobody thought to forbid enters the
# model silently. So every raw column must also appear in exactly one of
# the sets below, and tests/ fail on any column that does not
# (test_every_raw_column_is_classified) or on a model input nobody
# declared (test_feature_set_is_declared).

DATE_COL = "dteday"

# Raw numeric columns used as features unchanged. Same-day weather is
# treated as known to a forecaster (a weather forecast), as in the starter.
RAW_FEATURE_COLS = {"temp", "atemp", "hum", "windspeed"}

# Raw label columns that src/features.py re-encodes; the raw string never
# reaches the model (holiday -> is_holiday, workday -> is_workday,
# weather -> weather_severity and weather_severity_lag_1).
ENCODED_COLS = {"holiday", "workday", "weather"}

# Raw label columns not used: dayofweek, month_sin and month_cos are
# derived from dteday instead, so these would only duplicate them.
REDUNDANT_LABEL_COLS = {"season", "mnth", "weekday"}

# Features src/features.py builds. Add a feature here when you add it.
ENGINEERED_COLS = {
    "dayofweek", "is_weekend", "is_holiday", "is_workday",
    "month_sin", "month_cos",
    "weather_severity", "weather_severity_lag_1",
    "cnt_lag_1", "cnt_lag_7", "cnt_roll_7",
}

# The model's inputs must be exactly these.
DECLARED_FEATURES = RAW_FEATURE_COLS | ENGINEERED_COLS

# Temporal split: train on 2011, validate on H1 2012, test on H2 2012.
TRAIN_END = "2011-12-31"
VALID_END = "2012-06-30"

# Longest lag window in src/features.py. The first LAG_WARMUP calendar days
# have no complete history and are dropped; tests/test_contract.py uses this
# to bound how many rows may legitimately disappear.
LAG_WARMUP = 7
