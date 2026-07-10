from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


COMMIT_RE = re.compile(r"\b[0-9a-f]{7,40}\b")


def check_handoff_text(text: str) -> list[str]:
    errors: list[str] = []
    lower = text.lower()

    if not COMMIT_RE.search(text):
        errors.append("handoff must include a commit hash")
    if "remote feature branch: no" not in lower:
        errors.append("handoff must state 'remote feature branch: No'")
    if "pytest -q" not in text:
        errors.append("handoff must include pytest command(s)")
    if not re.search(r"# .*?(passed|failed|skipped)", text):
        errors.append("handoff must include exact test result output")
    if "git diff --check HEAD" not in text:
        errors.append("handoff must include git diff --check HEAD")
    return errors


def check_branch_policy(branch: str, upstream: str) -> list[str]:
    if branch.startswith("cc/") and upstream:
        return [f"cc branch '{branch}' must not track remote upstream '{upstream}'"]
    return []


def check_remote_branches(remote_branches: list[str]) -> list[str]:
    return [
        f"remote cc branch exists: {branch}"
        for branch in remote_branches
        if branch.startswith("origin/cc/")
    ]


_TRACKED_MD_ALLOWED = {"README.md", "README.zh.md"}


def check_tracked_markdown(tracked_files: list[str]) -> list[str]:
    """Only README.md and README.zh.md are allowed as tracked Markdown.

    The handoff (.codex_cc_handoff.md), planning docs (CLAUDE.md,
    AGENTS.md, PLAN.md), and docs/*.md must remain local and untracked.
    """
    tracked_md = {f for f in tracked_files if f.endswith(".md")}
    unexpected = tracked_md - _TRACKED_MD_ALLOWED
    if unexpected:
        return [
            f"tracked Markdown file '{f}' is not allowed; "
            f"only {sorted(_TRACKED_MD_ALLOWED)} may be tracked"
            for f in sorted(unexpected)
        ]
    return []


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _current_branch() -> str:
    return _git(["branch", "--show-current"])


def _current_upstream() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _remote_branches() -> list[str]:
    output = _git(["branch", "--remotes", "--format=%(refname:short)"])
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _tracked_files() -> list[str]:
    output = _git(["ls-tree", "-r", "--name-only", "HEAD"])
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check ValleyScope agent handoff and branch protocol."
    )
    parser.add_argument(
        "--handoff",
        default=".codex_cc_handoff.md",
        help="Path to the Codex/cc handoff file.",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    handoff_path = Path(args.handoff)
    if not handoff_path.exists():
        errors.append(f"handoff file not found: {handoff_path}")
    else:
        errors.extend(check_handoff_text(handoff_path.read_text(encoding="utf-8")))

    try:
        errors.extend(check_branch_policy(_current_branch(), _current_upstream()))
        errors.extend(check_remote_branches(_remote_branches()))
        errors.extend(check_tracked_markdown(_tracked_files()))
    except subprocess.CalledProcessError as exc:
        errors.append(exc.stderr.strip() or str(exc))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("agent protocol check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
