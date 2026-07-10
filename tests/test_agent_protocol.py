from scripts.check_agent_protocol import (
    check_branch_policy,
    check_handoff_text,
    check_remote_branches,
    check_tracked_markdown,
)


def test_handoff_requires_commit_tests_and_remote_branch_statement():
    good = """
Branch: cc/example
Commit: 123abcd Fix example
Remote feature branch: No
pytest -q tests/test_io_and_workflow.py
# 68 passed in 2.35s
pytest -q
# 483 passed in 4.45s
git diff --check HEAD
# clean
Do not merge to `main`; leave the branch ready for Codex review.
"""

    assert check_handoff_text(good) == []

    bad = """
Branch: cc/example
Commit: 123abcd Fix example
pytest -q
"""

    errors = check_handoff_text(bad)
    assert any("remote feature branch" in error for error in errors)
    assert any("git diff --check HEAD" in error for error in errors)
    assert any("test result output" in error for error in errors)


def test_cc_branch_must_not_track_remote_upstream():
    assert check_branch_policy("cc/example", "") == []
    errors = check_branch_policy("cc/example", "origin/cc/example")
    assert errors == [
        "cc branch 'cc/example' must not track remote upstream 'origin/cc/example'"
    ]


def test_remote_cc_branches_are_rejected():
    errors = check_remote_branches(["origin/main", "origin/cc/example"])
    assert errors == ["remote cc branch exists: origin/cc/example"]


def test_tracked_markdown_allows_only_readme():
    assert check_tracked_markdown(["README.md", "README.zh.md"]) == []
    assert check_tracked_markdown(["README.md", "src/main.py"]) == []


def test_tracked_markdown_rejects_handoff_and_docs():
    errors = check_tracked_markdown([
        "README.md", ".codex_cc_handoff.md", "docs/schema.md", "src/main.py",
    ])
    assert len(errors) == 2
    assert any(".codex_cc_handoff.md" in e for e in errors)
    assert any("docs/schema.md" in e for e in errors)


def test_tracked_markdown_rejects_claude_md():
    errors = check_tracked_markdown(["CLAUDE.md", "README.md"])
    assert len(errors) == 1
    assert "CLAUDE.md" in errors[0]


def test_tracked_markdown_empty_list_is_clean():
    assert check_tracked_markdown([]) == []
