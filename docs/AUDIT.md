# Audit

Exercise 3. Two runs are asked for: the `/leakage-auditor` subagent on a
teammate's repository, and the Agent SDK batch auditor (`agent/audit.py`)
on this one.

## 1. `/leakage-auditor` on a teammate's repository

Not done. No teammate repository was available to me for this assignment,
so this part of Exercise 3 is left out rather than replaced with another
repo. The subagent is unchanged at `.claude/agents/leakage-auditor.md`.

## 2. `agent/audit.py` on this repository

**Run:** `python agent/audit.py . --out reports/audit.jsonl`, on the finished
repo (2026-11-25), with the starter's local model `bikeshare-coder`
(`qwen3.5:9b` via Ollama 0.40.18, 64k context; `make check-model`: OK). The
run took about 25 minutes and made 27 tool calls. stderr ended with
`audited 1 repos; 0 flagged -> reports/audit.jsonl`.

**Output (`reports/audit.jsonl`, verbatim)**

```
{"repo": "/Users/ac2/Documents/bikeshare-open", "leakage": "none", "temporal_ok": true, "baseline_reported": true, "reproducible": true, "findings": [], "confidence": "high"}
```

**What the auditor actually did** (from its session transcript, not from
its answer):
- Read `src/features.py`, `src/train.py`, `src/config.py`, the Makefile,
  `tests/test_leakage.py`, `REPORT.md` and `docs/DECISIONS.md`.
- Ran `pytest`, then `make clean && make all` twice, comparing output hashes.
- Wrote its own pandas checks:
  - `cnt_lag_1`, `cnt_roll_7` and `weather_severity_lag_1` against the
    calendar (712, 700 and 712 rows passed);
  - the split (348 / 182 / 184 rows, no overlapping dates);
  - the model definitions (hgb has `random_state=0`).
- Three of its scripts first failed (`NameError`, then
  `No module named 'src'`) before it fixed them. It launched one
  sub-agent, whose result never came back before it answered.
- It **did not call `check_identity`**, the deterministic leakage tool its
  instructions told it to use on the feature table.

**My verdict.** No finding changes a number, because there are no
findings. I agree with all four verdicts, but for my own reasons, not the
auditor's:
- *No leakage:* established by
  `test_no_feature_pair_restates_the_target` (least squares on every
  feature pair) and `test_feature_set_is_declared`, both green.
- *Temporal validity:* established by the split and calendar-lag tests.
- *Baselines:* established by `test_persistence_baseline_is_reported` and
  `reports/metrics.json`.
- *Reproducibility:* established by a fresh clone reproducing
  `metrics.json` byte for byte. The auditor's own `make clean && make all`
  also left the committed file unchanged.

What I do not accept is the "high" confidence on leakage. The auditor
skipped the one tool that tests for an arithmetic identity, and judged
leakage by reading `FORBIDDEN` and the tests. A repo whose tests were
wrong would have got the same answer. So this run is a screening result
that agrees with the suite, not independent evidence. A 9B model also
stops early and asserts confidently, which is the behaviour the lesson
warns about.
