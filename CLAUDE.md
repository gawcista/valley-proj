# Claude Code Rules For ValleyScope

Claude Code (`cc`) is the implementation agent for this repository. Follow
`AGENTS.md` first, then this file for cc-specific workflow rules.

## Branch Rules

- Work on a local branch named `cc/<task-name>`.
- Do not create, push, or track remote feature branches.
- Do not run `git push -u origin cc/<task-name>`.
- Do not merge to `main`.
- Leave the branch ready for Codex review.

If a remote `origin/cc/*` branch exists, treat it as cleanup work for Codex or
the user; do not rely on it for handoff.

## Handoff Rules

`.codex_cc_handoff.md` is the working communication file between cc and Codex.
It is not a user-facing report.

At completion, rewrite `.codex_cc_handoff.md` from the current task state. The
handoff must include:

- branch name and commit hash;
- `Remote feature branch: No`;
- exact changed files;
- exact test commands and exact outputs;
- exact `git diff --check HEAD` output;
- skipped tests or benchmarks and why;
- remaining risks in three bullets or fewer;
- explicit statement that cc did not merge to `main`.

Use `.codex_cc_handoff.template.md` as the structure.

## Required Checks

Before reporting completion, run:

```bash
python scripts/check_agent_protocol.py
git diff --check HEAD
```

Also run the task-specific tests requested by Codex. For code or schema changes,
run at least the relevant targeted tests. For broad workflow changes, run
`pytest -q` unless Codex explicitly narrowed the test scope.

Codex should confirm the test results from the handoff. cc must not leave
testing for Codex to repeat.
