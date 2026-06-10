# Agent Handoff Protocol

This protocol turns recurring ValleyScope workflow rules into checkable
handoff requirements.

## Roles

Codex is the methodology reviewer, merge reviewer, and task planner. After each
cc review, Codex must rewrite `.codex_cc_handoff.md` with either review fixes
for cc or the next approved cc goal-mode task.

cc is the implementation agent. cc works on local `cc/<task-name>` branches,
self-tests, self-reviews, and rewrites `.codex_cc_handoff.md` before asking for
Codex review.

## Branch Policy

cc must:

- create a local branch named `cc/<task-name>`;
- avoid remote feature branches;
- avoid upstream tracking for `cc/*` branches;
- avoid merging to `main`;
- leave the branch ready for Codex review.

Codex may merge reviewed work to `main`, push `main`, and delete local/remote
feature branches after review.

## Handoff Policy

`.codex_cc_handoff.md` is a working file for Codex and cc. It is gitignored on
purpose and should be rewritten for each handoff.

cc completion handoff must include:

- branch name;
- commit hash;
- `Remote feature branch: No`;
- exact changed files;
- exact test commands and exact outputs;
- exact `git diff --check HEAD` output;
- skipped tests or benchmarks and why;
- remaining risks in three bullets or fewer;
- confirmation that cc did not merge to `main`.

Codex review handoff must include:

- review verdict;
- blocking findings or merge result;
- exact evidence Codex checked;
- next cc goal-mode prompt when further work is needed;
- branch cleanup status when work was merged.

## Test Responsibility

cc owns task test execution. Codex should confirm that cc reported the requested
test outputs and inspect whether the results match the changed surface area.

Codex should not routinely repeat cc's test work. Codex may run targeted tests
or full `pytest -q` when:

- cc's handoff is missing or incomplete;
- the reported test scope does not cover the change;
- Codex is about to merge directly to `main`;
- a result looks stale or inconsistent;
- the user explicitly asks Codex to verify locally.

## Local Protocol Check

Run this before declaring a cc task complete:

```bash
python scripts/check_agent_protocol.py
```

The check fails when:

- `.codex_cc_handoff.md` lacks commit/test/diff-check evidence;
- `.codex_cc_handoff.md` does not state `Remote feature branch: No`;
- the current `cc/*` branch tracks an upstream;
- any `origin/cc/*` branch is visible locally.

The script is intentionally lightweight. It catches workflow drift; it does not
replace technical review.
