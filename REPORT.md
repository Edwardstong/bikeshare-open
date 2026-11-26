# Bikeshare demand: methodology report

Forecasting daily Capital Bikeshare rides (`cnt`) in Washington, DC. Train on
2011, validate on H1 2012, test on H2 2012 (184 days). All numbers below come
from `make all` (`reports/metrics.json`) unless a different file is named.

## Results

`reports/metrics.json` is the "after" row of each model. The "before" rows
are the unmodified starter (`reports/metrics_before_feature.json`, commit
"Track reports/"). The baselines do not use the new feature, so their rows
are unchanged.

| model | feature set | valid R^2 | valid MAE | test R^2 | test MAE |
|---|---|---|---|---|---|
| `baseline_mean` | before (starter, 14 features) | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| `baseline_mean` | after (+ weather_severity_lag_1, 15) | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| `baseline_persistence` | before (starter, 14 features) | 0.4507 | 900.6 | 0.5099 | 840.1 |
| `baseline_persistence` | after (+ weather_severity_lag_1, 15) | 0.4507 | 900.6 | 0.5099 | 840.1 |
| `ridge` | before (starter, 14 features) | 0.6475 | 832.9 | 0.5833 | 947.1 |
| `ridge` | after (+ weather_severity_lag_1, 15) | 0.667 | 803.9 | 0.6124 | 910.6 |
| `hgb` | before (starter, 14 features) | 0.2185 | 1296.6 | -0.4975 | 1972.6 |
| `hgb` | after (+ weather_severity_lag_1, 15) | 0.2437 | 1275.2 | -0.4687 | 1950.4 |

R^2 is computed against each split's own mean. MAE is in rides per day.

**The feature.** `weather_severity_lag_1` is yesterday's weather severity
(GOOD 0, MISTY 1, BAD 2), shifted on the gap-free calendar index. For date
d it is d-1's recorded weather, known before d starts. I picked it before
running anything. Persistence misses worst on the day after bad weather:
2011-04-16 was BAD with 795 rides, and 2011-04-17 was GOOD with 3,744. A
model that only sees `cnt_lag_1` reads that low count as falling demand. The
test was written first and failed for the right reason (`KeyError`, recorded
in `docs/SEEDED_FAULTS.md`). The decision used validation only: ridge
+0.0195 R^2 (MAE 832.9 -> 803.9), hgb +0.0252. That clears the 0.01 floor
in the `new-feature` skill, so the feature stays. With 182 validation days,
0.01 is a floor, not a significance test.

**Verification.** `make test` runs 21 tests, all green. Seven are new:
- the feature's own test;
- the required unclassified-raw-column test;
- an allowlist on the model's inputs;
- a least-squares check that no feature pair reproduces `cnt`;
- a test that the weather encoder raises on unknown labels (added after
  seeded fault 3 left the starter suite green);
- a test that the column classes don't overlap;
- a SHA-256 check on the raw file (added after seeded fault 4 changed the
  headline with every test green).

On a scratch copy I checked that each of these goes red: a column appended
to the raw CSV, a raw column renamed, `casual` copied into the features
under a new name, and `casual`/`registered` copied and declared.

**Batch audit.** `agent/audit.py` on this repo, run with the local
`qwen3.5:9b` model, flagged nothing (`reports/audit.jsonl`: no leakage,
temporal split valid, baselines reported, reproducible, no findings). I
count that as agreement with the suite, not as extra evidence: it never ran
its identity tool (`docs/AUDIT.md`).

## Headline finding

The naive evaluation gives R^2 = 1.000, and it is invalid:
`casual + registered == cnt` on every row (`python -m src.leakage_demo`).
Under a temporal split, the mean baseline scores -2.1441 on test, because
ridership grew about 60% from 2011 to 2012. Persistence ("yesterday again")
scores 0.5099 (MAE 840.1). Ridge reaches 0.6124: ahead of persistence by
0.10 R^2, but behind it on MAE (910.6 vs 840.1). Gradient boosting scores
-0.4687. It still beats the mean baseline by a wide margin, but it cannot
predict above the 2011 maximum of 6,043 rides, and 56.5% of test days are
above it. The new feature raised ridge's validation R^2 from 0.6475 to
0.667, and we kept it; its test R^2 moved from 0.5833 to 0.6124.

## Exercise 4: target transformation

**Question.** Ridge beats gradient boosting here because ridership grew. Can
a change of target let the tree compete?

**Method.** `python -m src.train --target {ratio,logratio,diff}` transforms
the target for ridge and hgb only. Predictions are mapped back to rides
before scoring, and the baselines are unchanged. Every transform divides by,
or subtracts, a quantity known at prediction time (`cnt_roll_7` and
`cnt_lag_1` are both shifted). I tried three transforms on validation
(`make ex4-valid`, `reports/exercise4/valid_*.json`; validation split only),
chose one, committed that decision, and then touched test once
(`make ex4-test`).

| target for ridge and hgb | ridge valid R^2 | hgb valid R^2 | hgb valid MAE |
|---|---|---|---|
| `cnt` (pipeline) | 0.667 | 0.2437 | 1275.2 |
| `cnt / cnt_roll_7` ("ratio") | 0.4305 | 0.7591 | 662.0 |
| `log(cnt / cnt_roll_7)` | 0.5472 | 0.7161 | 724.5 |
| `cnt - cnt_lag_1` | 0.6675 | 0.6254 | 804.3 |

**Mechanism.** 39% of validation days, and 56.5% of test days, are above
the largest count in the 2011 training data, and a tree cannot predict
above that. The ratio of today to the trailing 7-day mean ranged from 0.158
to 1.878 in 2011, and 0% of validation ratios fall outside that range. The
tree no longer has to extrapolate. It predicts a relative change, and the
known recent level carries the growth.

**Single test run** (`reports/exercise4/test_ratio.json`): hgb on the ratio
target scores **test R^2 0.6301, MAE 833.7**. That is ahead of untransformed
ridge (0.6124 / 910.6) and slightly ahead of persistence on MAE (840.1).
Ridge on the same target falls to 0.0877.

**Is it defensible?** As a forecasting setup, yes:
- it uses only information available at d;
- it is scored in rides against the same baselines;
- it was chosen on validation and fixed before test;
- it reflects a real property of the data: demand is roughly
  multiplicative around a moving level.

As evidence that "gradient boosting beats ridge", no. The same transform
hurts ridge badly, so what is being compared is two (model, target) pairs,
each with a target that suits it. I also chose among three transforms on
182 validation days, a small selection effect that the single test run
does not remove. The honest statement is narrower: hgb's failure in the
headline is a failure to extrapolate the target's level, not a lack of
signal. Once the level is supplied, the tree is competitive. The committed
`reports/metrics.json` stays the untransformed pipeline.

## Exercise 5: ridge vs persistence

Ridge is not worth deploying over "yesterday again" as it stands. On the
184 test days it is closer than persistence on only 41% of days, and its
median absolute error is 877.7 rides against 506.5. Most of that comes from
a level bias. Ridge's mean error is -522.6 rides/day on test (-479.2 on
validation), while persistence's is +16.1. Its standardised coefficients
split the weight between `cnt_roll_7` (480.7) and `cnt_lag_1` (469.6), with
weather terms on top (`weather_severity` -287.6, `weather_severity_lag_1`
+121.2). So it has learned more than "yesterday again": a smoothed recent
level adjusted for today's and yesterday's weather. But its intercept and
shrinkage are anchored to 2011, so it under-forecasts a growing system
every day. Its advantage is concentrated on the 70 "big move" days (days
where |cnt - cnt_lag_1| exceeds 756, the 75th percentile of 2011): MAE
1095.2 vs 1694.9. On the other 114 days it loses badly (797.3 vs 315.2).
A 7-day block bootstrap (`reports/ridge_vs_persistence.json`) puts the test
RMSE difference at [-235.4, -18.3], a real gain on large misses. The MAE
difference is [-13.2, 175.2], not distinguishable from zero. If large misses
cost much more than small ones, ridge's RMSE edge would matter. That is the
case if a missed demand rebound after a storm means empty docks. On a
typical day, though, persistence is closer and unbiased. I would deploy
ridge only with its level bias corrected (for example the `diff` target from
Exercise 4: ridge validation R^2 0.6675, about the same as untransformed)
and re-checked on fresh data.

## Attribution
