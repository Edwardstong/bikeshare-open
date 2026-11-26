# Seeded faults

Exercise 2, plus the recorded failing run of the Exercise 1 feature test.

## How these were produced

- Each fault was made as one edit to one file and committed in its broken
  state, so `git log` shows the fault and its fix as separate commits.
- **Hook stderr** is the real output of `.claude/hooks/check_leakage.sh`, fed
  the same PostToolUse payload Claude Code sends after an Edit:
  `printf '{"tool_name":"Edit","tool_input":{"file_path":"%s"}}' "$PWD/src/features.py" | CLAUDE_PROJECT_DIR="$PWD" .claude/hooks/check_leakage.sh`.
  The agent session for this assignment ran from outside the repo root (see
  Attribution in `REPORT.md`), so the hook was fed its payload by hand after
  each seeded edit. The script, matcher and exit code are the repo's own.
- **`make test`** output is pasted from the real run: the failure summary and
  the counts. Full tracebacks are omitted for length.
- **Line numbers** in tracebacks (e.g. `tests/test_leakage.py:46`) refer to
  the test files at each fault's own commit. Later commits added tests above
  them, so the same fault seeded on the final code prints the same test name
  and message at a different line, and higher pass counts (21 tests now).
- **Turns to fix**: each fix was done by a fresh agent with no context. It got
  only the hook's stderr, the repo's permission allowlist (Read/Grep/Glob/Edit,
  plus `make`, `python -m pytest`, `git diff`, `git status`), and a ban on
  editing `tests/`, `data/raw/` or using git to restore files. Because the
  broken state was committed first, `git diff` showed nothing, so the agent had
  to diagnose from the failure. A turn here means one tool call.

## Exercise 1: the feature test, written first

The test `test_weather_lag_1_is_yesterdays_weather` was committed before the
feature existed (commit "Failing test for weather_severity_lag_1"). It failed
for the right reason, a missing column:

```
python -m pytest tests/ -q
.........F.....                                                          [100%]
=================================== FAILURES ===================================
...
E           KeyError: 'weather_severity_lag_1'
...
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_weather_lag_1_is_yesterdays_weather - KeyE...
1 failed, 14 passed in 0.86s
make: *** [test] Error 1
```

After the feature was added: `15 passed in 0.73s`.

## Fault 1: a forbidden column added to `feature_columns`

Commits: "Seed fault 1" (broken), "Fix fault 1".

**Exact change**

```diff
diff --git a/src/features.py b/src/features.py
index 3d9ebac..72f4540 100644
--- a/src/features.py
+++ b/src/features.py
@@ -85,7 +85,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
 def feature_columns(df: pd.DataFrame) -> list[str]:
     """Numeric columns that are safe to feed the model."""
     numeric = df.select_dtypes(include="number").columns
-    return [c for c in numeric if c not in FORBIDDEN]
+    return [c for c in numeric if c not in FORBIDDEN] + ["casual"]
 
 
 if __name__ == "__main__":
```

**Hook stderr** (the last line is the exit code, echoed by the shell; exit 2 sends the stderr back to the agent)

```
Leakage tests failed after editing /Users/ac2/Documents/bikeshare-open/src/features.py:
        copied under an innocent name (out["walkins"] = out.casual) is an
        undeclared input, and a feature that silently vanishes is a missing one.
        """
        cols = set(feature_columns(feats))
>       assert cols == DECLARED_FEATURES, (
            f"undeclared inputs: {sorted(cols - DECLARED_FEATURES)}; "
            f"declared but missing: {sorted(DECLARED_FEATURES - cols)}"
        )
E       AssertionError: undeclared inputs: ['casual']; declared but missing: []
E       assert {'atemp', 'ca...yofweek', ...} == {'atemp', 'cn...', 'hum', ...}
E         
E         Extra items in the left set:
E         'casual'
E         Use -v to get more diff

tests/test_leakage.py:46: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
FAILED tests/test_leakage.py::test_feature_set_is_declared - AssertionError: ...
2 failed, 8 passed in 0.27s
hook exit=2
```

The hook prints the last 20 lines, so only the second failure's detail
shows. Both red tests are named in the summary.

**`make test`**

```
python -m pytest tests/ -q
...
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
FAILED tests/test_leakage.py::test_feature_set_is_declared - AssertionError: ...
2 failed, 17 passed in 0.80s
make: *** [test] Error 1
```

**Fix.** A fresh agent fixed it in **7 tool calls** (4 Read: `CLAUDE.md`,
`src/features.py`, `tests/test_leakage.py`, `src/config.py`; 1 Edit; 2 Bash:
`make test`, `git diff`). Its diagnosis: "adding it after the filter broke both
`test_no_forbidden_features` and `test_feature_set_is_declared`". It removed
`+ ["casual"]`. After the fix: `19 passed`, hook exit 0.

Note: `test_no_feature_pair_restates_the_target` stayed green here, as it
should. `casual` alone does not reproduce `cnt` exactly; it needs
`registered` as well. A lone leaked component is caught by the declared
feature set, not by the identity check.

## Fault 2: the `.shift(1)` deleted from `cnt_roll_7`

Commits: "Seed fault 2" (broken), "Fix fault 2".

**Exact change**

```diff
diff --git a/src/features.py b/src/features.py
index 3d9ebac..76d4369 100644
--- a/src/features.py
+++ b/src/features.py
@@ -77,7 +77,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
     # If you add a longer lag, raise LAG_WARMUP in src/config.py.
     out["cnt_lag_1"] = out[TARGET].shift(1)
     out["cnt_lag_7"] = out[TARGET].shift(LAG_WARMUP)
-    out["cnt_roll_7"] = out[TARGET].shift(1).rolling(LAG_WARMUP).mean()
+    out["cnt_roll_7"] = out[TARGET].rolling(LAG_WARMUP).mean()
 
     return out.dropna().reset_index(drop=True)
```

**Hook stderr** (the last line is the exit code, echoed by the shell)

```
Leakage tests failed after editing /Users/ac2/Documents/bikeshare-open/src/features.py:
[714 rows x 29 columns]

    def test_rolling_mean_excludes_today(feats):
        """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
        by_date = feats.set_index("dteday")[TARGET]
        date = feats.dteday.iloc[200]
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if all(d in by_date.index for d in window):
            expected = sum(by_date[d] for d in window) / 7
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4389.0) == 4414.857142857143 ± 0.00441486
E             
E             comparison failed
E             Obtained: 4389.0
E             Expected: 4414.857142857143 ± 0.00441486

tests/test_leakage.py:94: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 9 passed in 0.27s
hook exit=2
```

**`make test`**

```
python -m pytest tests/ -q
...
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 18 passed in 0.79s
make: *** [test] Error 1
```

**Fix.** A fresh agent fixed it in **7 tool calls** (harness count; its own
list was 3 Read, 1 Edit, 2 Bash: `git diff`, `make test`). Its diagnosis: the
window "covered d-6..d and included that day's own count, a target leak". It
restored `.shift(1)` before `.rolling()`. After the fix: `19 passed`, hook
exit 0.

Note: the row count stayed at 714 with the fault in place, because
`cnt_lag_7` already sets the warm-up. So `test_only_expected_rows_are_dropped`
cannot see this fault. Only the value check on a named row catches it.

## Fault 3: the weather encoder's `raise` removed

Commits: "Seed fault 3" (broken, suite green), "Test that encode_weather
fails loudly..." (fault still in place, now red), "Fix fault 3".

**Exact change**

```diff
diff --git a/src/features.py b/src/features.py
index 3d9ebac..4563654 100644
--- a/src/features.py
+++ b/src/features.py
@@ -26,8 +26,6 @@ def encode_weather(series: pd.Series) -> pd.Series:
     """
     known = series.dropna()
     unknown = set(known.unique()) - set(WEATHER_ORDER)
-    if unknown:
-        raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
     return series.map(WEATHER_ORDER).astype(float)
```

**First result: nothing fired.** The hook exited 0 with no stderr, and
`make test` stayed green:

```
hook exit=0
```

```
python -m pytest tests/ -q
...................                                                      [100%]
19 passed in 0.76s
```

This is the finding. The current data has only the three labels in
`WEATHER_ORDER`, so the `raise` never runs, and no test fed the encoder an
unknown label. Without it, a new label (a mirror that adds `SNOW`, or one
that reverts to UCI's codes 1-4) would become NaN and be deleted by
`dropna()`. That is the 20-row bug from the lesson, back with no warning.
The starter suite could not fail on this fault, so it was not verifying it.

**Added test:** `test_encode_weather_fails_loudly_on_unknown_labels` in
`tests/test_leakage.py`. It is in that file so the PostToolUse hook runs it.
It checks the encoder alone (`["GOOD", "SNOW", "MISTY"]` must raise) and end
to end (one raw day relabelled `RAINY` must stop `build_features`). With the
fault still in place:

**Hook stderr** (the last line is the exit code, echoed by the shell)

```
Leakage tests failed after editing /Users/ac2/Documents/bikeshare-open/src/features.py:
........F..                                                              [100%]
=================================== FAILURES ===================================
______________ test_encode_weather_fails_loudly_on_unknown_labels ______________

    def test_encode_weather_fails_loudly_on_unknown_labels():
        """A weather label outside WEATHER_ORDER must raise, not become a NaN
        that a later dropna() deletes. The starter suite never fed the encoder
        an unknown label, so removing the raise left every test green
        (docs/SEEDED_FAULTS.md, fault 3)."""
>       with pytest.raises(ValueError, match="unmapped weather labels"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

tests/test_leakage.py:125: Failed
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_encode_weather_fails_loudly_on_unknown_labels
1 failed, 10 passed in 0.26s
hook exit=2
```

**`make test`**

```
python -m pytest tests/ -q
...
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_encode_weather_fails_loudly_on_unknown_labels
1 failed, 19 passed in 0.79s
make: *** [test] Error 1
```

**Fix.** A fresh agent fixed it in **5 tool calls** (3 Read, 1 Edit, 1 Bash:
`make test`). It put the `raise` back. After the fix: `20 passed`, hook
exit 0.

## Fault 4: a `cp` into `data/raw/` from Bash (the hook cannot see it)

Not committed in its broken state: the graded raw file must keep its hash.
The tampered copy was made in a scratch directory. Every test-period row
(2012-07-01 on, 184 rows) got `registered` scaled by 1.10, with `cnt` reset
to `casual + registered`, so the leakage identity still holds. It was then
copied over the raw file from Bash.

**What happened.** Every guard in the starter missed it:

- The permission rules `Write(./data/raw/**)` and `Edit(./data/raw/**)`
  bind those two tools only. A Bash `cp` is not either.
- The PostToolUse hook's matcher is `Edit|Write`, so Claude Code does not run
  it for Bash. Fed the Bash payload by hand, it exits 0 because there is no
  `file_path`.
- All 20 tests passed. The tests check structure (columns, identity, lags,
  row counts), and the tamper kept all of it.
- `make all` ran clean and **the headline moved**.

```
$ cp "$SCRATCH/bike_daily_tampered.csv" data/raw/bike_daily.csv
cp exit=0
$ git status --short
 M data/raw/bike_daily.csv
?? docs/SEEDED_FAULTS.md
?? src/compare.py
$ shasum -a 256 data/raw/bike_daily.csv
3d2bbaadeafcbdda5180b6cd83887c71b8f9f322273243e129671d04fc5fb4d3  data/raw/bike_daily.csv
$ (PostToolUse hook fed the Bash tool payload)
hook exit=0 (no stderr)
$ make test
20 passed in 0.77s
$ make all
OK ingest: 728 rows, 2011-01-03 -> 2012-12-31
OK features: 714 rows, 15 features -> features.parquet
```

| test set (after vs before tampering) | before | after |
|---|---|---|
| ridge R^2 | 0.6124 | 0.5741 |
| ridge MAE | 910.6 | 1048.0 |
| persistence MAE | 840.1 | 907.1 |
| hgb R^2 | -0.4687 | -0.9104 |

**What would have caught it**

1. **A hash check in `make data`** (implemented). Store the file's SHA-256
   and refuse to run if the bytes differ. This catches the change whoever
   made it (agent, script or person) and however (`cp`, `sed -i`, a Python
   one-liner).
2. **A Stop hook that runs the full suite** (implemented). With the hash
   test in the suite, the agent cannot report "done" while the raw file
   differs. `stop_hook_active` lets it stop on the second try, so a red
   suite it cannot fix does not loop.
3. **A PreToolUse hook on Bash** (not implemented). It would deny commands
   that write under `data/raw/`. It acts earliest, but it is a pattern match
   on command text: `python -c "open(...).write(...)"` or a `cd` first gets
   past it. That is the same weakness as the deny rules.
4. **Filesystem permissions** (not implemented). `chmod -w data/raw` stops
   any writer that does not first `chmod` it back, but it is local to one
   checkout and does not travel with a clone.

**The guard, run against the tampered file** (before restoring it)

```
$ make data
ValueError: bike_daily.csv has been modified: sha256 3d2bbaadeafcbdda..., expected a6bed1577f6e4de5... Raw data is immutable; restore it with `git checkout -- data/raw/bike_daily.csv` and change things downstream.
make: *** [data] Error 1
$ make test
=========================== short test summary info ============================
FAILED tests/test_contract.py::test_raw_file_is_unmodified - AssertionError: ...
1 failed, 20 passed in 0.79s
make: *** [test] Error 1
$ (Stop hook, first stop)
Not done: make test is red.
FAILED tests/test_contract.py::test_raw_file_is_unmodified - AssertionError: ...
1 failed, 20 passed in 0.78s
stop hook exit=2
$ (Stop hook, stop_hook_active=true)
stop hook exit=0
$ git checkout -- data/raw/bike_daily.csv
a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc  data/raw/bike_daily.csv
$ make all && make test
OK ingest: 728 rows, 2011-01-03 -> 2012-12-31
OK features: 714 rows, 15 features -> features.parquet
21 passed in 0.76s
stop hook exit=0
```

**Diff of the guard** (commit "Fault 4 guard")

```diff
diff --git a/.claude/hooks/require_green.sh b/.claude/hooks/require_green.sh
new file mode 100755
index 0000000..a7d7379
--- /dev/null
+++ b/.claude/hooks/require_green.sh
@@ -0,0 +1,23 @@
+#!/usr/bin/env bash
+# Stop hook: the agent's "done" only counts with a green suite.
+#
+# The PostToolUse hook only sees Edit/Write under src/. A Bash command
+# (a `cp` into data/raw/, a script that rewrites a file) never triggers it.
+# This runs the whole suite, including the raw-data hash check, whenever
+# the agent tries to finish. Exit 2 blocks the stop and sends stderr back.
+set -uo pipefail
+
+payload=$(cat)
+active=$(printf '%s' "$payload" | python3 -c \
+  'import json,sys; print(json.load(sys.stdin).get("stop_hook_active", False))')
+
+# Already sent back once by this hook: let it stop, so it cannot loop.
+[ "$active" = "True" ] && exit 0
+
+cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
+if ! out=$(python -m pytest tests/ -q 2>&1); then
+  echo "Not done: make test is red." >&2
+  echo "$out" | grep -E "^(FAILED|ERROR)|passed|failed" | tail -15 >&2
+  exit 2
+fi
+exit 0
diff --git a/.claude/settings.json b/.claude/settings.json
index 5c849d8..a206399 100644
--- a/.claude/settings.json
+++ b/.claude/settings.json
@@ -19,6 +19,18 @@
           }
         ]
       }
+    ],
+    "Stop": [
+      {
+        "hooks": [
+          {
+            "type": "command",
+            "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/require_green.sh",
+            "timeout": 120,
+            "statusMessage": "Running the full suite before stopping..."
+          }
+        ]
+      }
     ]
   },
   "permissions": {
diff --git a/.gitattributes b/.gitattributes
new file mode 100644
index 0000000..59bf54b
--- /dev/null
+++ b/.gitattributes
@@ -0,0 +1,4 @@
+# The raw CSV is checked by SHA-256 (src/config.py RAW_SHA256). Never let
+# git rewrite its line endings on checkout, or the hash check fails on a
+# clean clone.
+data/raw/*.csv -text
diff --git a/src/config.py b/src/config.py
index 93dd6bc..497ec16 100644
--- a/src/config.py
+++ b/src/config.py
@@ -20,6 +20,12 @@ DATA_URL = (
     "interpretable-ml-book/master/data/bike.csv"
 )
 
+# SHA-256 of the checked-in raw file. Raw data is immutable, but the
+# permission rules only bind the Edit/Write tools: a Bash `cp` walks past
+# them and past the PostToolUse hook. `make data` and the test suite
+# compare against this, so any change to the bytes is loud.
+RAW_SHA256 = "a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc"
+
 TARGET = "cnt"
 
 # --- Three reasons a column is kept out of the feature set --------------
diff --git a/src/ingest.py b/src/ingest.py
index 6654075..578cf84 100644
--- a/src/ingest.py
+++ b/src/ingest.py
@@ -2,12 +2,13 @@
 
 Run: python -m src.ingest
 """
+import hashlib
 import sys
 import urllib.request
 
 import pandas as pd
 
-from src.config import DATA_URL, RAW
+from src.config import DATA_URL, RAW, RAW_SHA256
 
 EXPECTED_MIN_ROWS = 700
 REQUIRED_COLS = {"dteday", "cnt", "casual", "registered", "temp", "hum", "windspeed"}
@@ -20,6 +21,21 @@ def download(force: bool = False) -> None:
     urllib.request.urlretrieve(DATA_URL, RAW)
 
 
+def raw_sha256() -> str:
+    return hashlib.sha256(RAW.read_bytes()).hexdigest()
+
+
+def verify_raw() -> None:
+    """Fail if data/raw/bike_daily.csv is not the file the results came from."""
+    actual = raw_sha256()
+    if actual != RAW_SHA256:
+        raise ValueError(
+            f"{RAW.name} has been modified: sha256 {actual[:16]}..., expected "
+            f"{RAW_SHA256[:16]}... Raw data is immutable; restore it with "
+            f"`git checkout -- data/raw/{RAW.name}` and change things downstream."
+        )
+
+
 def load_raw() -> pd.DataFrame:
     df = pd.read_csv(RAW, parse_dates=["dteday"])
     missing = REQUIRED_COLS - set(df.columns)
@@ -32,5 +48,6 @@ def load_raw() -> pd.DataFrame:
 
 if __name__ == "__main__":
     download(force="--force" in sys.argv)
+    verify_raw()
     df = load_raw()
     print(f"OK ingest: {len(df)} rows, {df.dteday.min().date()} -> {df.dteday.max().date()}")
diff --git a/tests/test_contract.py b/tests/test_contract.py
index 76fdd92..38b4fb5 100644
--- a/tests/test_contract.py
+++ b/tests/test_contract.py
@@ -10,11 +10,12 @@ from src.config import (
     LAG_WARMUP,
     RAW,
     RAW_FEATURE_COLS,
+    RAW_SHA256,
     REDUNDANT_LABEL_COLS,
     TARGET,
 )
 from src.features import build_features, feature_columns, reindex_to_calendar
-from src.ingest import load_raw
+from src.ingest import load_raw, raw_sha256
 from src.train import MODELS, temporal_split
 
 
@@ -72,6 +73,18 @@ def test_every_raw_column_is_classified():
     )
 
 
+def test_raw_file_is_unmodified():
+    """The raw CSV must be byte-identical to the file the results came from.
+
+    Every other test checks structure; a tampered value that keeps
+    casual + registered == cnt passes all of them (docs/SEEDED_FAULTS.md,
+    fault 4: a Bash `cp` into data/raw/ changed the headline silently)."""
+    assert raw_sha256() == RAW_SHA256, (
+        "data/raw/bike_daily.csv has been modified; restore it with "
+        "`git checkout -- data/raw/bike_daily.csv`"
+    )
+
+
 def test_column_classes_do_not_overlap():
     """A column cannot be both forbidden and a feature."""
     sets = [FORBIDDEN, RAW_FEATURE_COLS, ENCODED_COLS, REDUNDANT_LABEL_COLS, {DATE_COL}]
```

## Sanity checks of the hook itself (Step 7)

- A `src/` edit with tests passing: hook exit 0, no output (after each fix
  above).
- A non-`src/` edit: the hook exits at the `case` statement, with no output.
  Feeding it `{"tool_input":{"file_path":"docs/SEEDED_FAULTS.md"}}` gives
  exit 0.
- A `src/` edit with the guard broken: faults 1-3 above, exit 2 each time.
