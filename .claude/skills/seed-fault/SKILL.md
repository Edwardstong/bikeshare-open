---
name: seed-fault
description: Break the pipeline on purpose with one described fault, record the red test output and the hook's stderr in docs/SEEDED_FAULTS.md, then fix it back to green and count the turns. Use when asked to seed a fault, break something deliberately, or check that a guard really fires.
argument-hint: <fault, e.g. "delete the .shift(1) from cnt_roll_7">
allowed-tools: Read, Edit, Bash(make*), Bash(python -m pytest*), Bash(git diff*), Bash(git status*), Bash(printf*)
---

# Working tree now

!`git status --short`

# Seeding: $ARGUMENTS

1. **Start green.** Run `make test`. If anything already fails, stop and
   say so: a fault seeded on a red suite proves nothing.

2. **Make the fault as one Edit to one file under `src/`.** Show
   `git diff` for it; that diff is the "exact change". Faults in
   `data/raw/` are out of scope here (writes there are denied).

3. **Record what fires, verbatim.**
   - The PostToolUse hook runs by itself after the Edit. Copy its stderr.
     To capture it again on demand:
     `printf '{"tool_name":"Edit","tool_input":{"file_path":"%s"}}' "$PWD/<file>" | CLAUDE_PROJECT_DIR="$PWD" .claude/hooks/check_leakage.sh; echo "hook exit=$?"`
   - Run `make test` and keep the `FAILED ...` lines and the counts.
   - If nothing goes red, that is the finding. Write "suite stays green",
     name the test that is missing, and stop before fixing.

4. **Write it down.** Append to `docs/SEEDED_FAULTS.md` under
   `## Fault: $ARGUMENTS`: the diff, the hook stderr and the `make test`
   output, each in a fenced block. Paste real output, never a summary
   of it.

5. **Fix it from the failure output, not with `git restore`.** Count the
   turns until `make test` is green again and record the count and the
   final summary line. Never edit `tests/` to make a fault pass.
