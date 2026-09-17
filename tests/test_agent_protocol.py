import subprocess
import sys
from pathlib import Path

from scripts.check_agent_protocol import (
    check_branch_policy,
    check_handoff_text,
    check_remote_branches,
    check_tracked_markdown,
)


_EXPECTED_HEAD = "705a7a2f0f751abcc4f0d1f49d7c62e984f60345"

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_agent_protocol.py"


def _handoff_text(author: str, branch: str, commit: str) -> str:
    return f"""
Updated by: {author}
Branch: {branch}
Commit: {commit}
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""


def test_handoff_requires_commit_tests_and_remote_branch_statement():
    good = """
Updated by: cc
Branch: cc/example
Commit: 705a7a2 Fix example
Remote feature branch: No
pytest -q tests/test_io_and_workflow.py
# 68 passed in 2.35s
pytest -q
# 483 passed in 4.45s
git diff --check HEAD
# clean
Do not merge to `main`; leave the branch ready for Codex review.
"""

    assert check_handoff_text(good, expected_head=_EXPECTED_HEAD) == []

    bad = """
Updated by: cc
Branch: cc/example
Commit: 705a7a2 Fix example
pytest -q
"""

    errors = check_handoff_text(bad, expected_head=_EXPECTED_HEAD)
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


def test_completed_handoff_rejects_not_yet_wired():
    """A handoff marked COMPLETED must not claim a required path is not yet wired."""
    text = """
Updated by: cc
Branch: cc/example
Commit: 705a7a2 Fix example
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
COMPLETED - Phase A1
Remaining Risks:
- State 2 (table_validation_passed) is not yet wired.
"""
    errors = check_handoff_text(text, expected_head=_EXPECTED_HEAD)
    assert any("not yet wired" in e for e in errors)


def test_completed_without_not_yet_wired_is_clean():
    text = """
Updated by: cc
Branch: cc/example
Commit: 705a7a2 Fix example
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
COMPLETED - Phase A1
Remaining Risks:
- Centered settings without explicit transform remain unresolved.
"""
    assert check_handoff_text(text, expected_head=_EXPECTED_HEAD) == []


def test_placeholder_output_is_rejected():
    text = """
Updated by: cc
Branch: cc/example
Commit: 705a7a2 Fix example
Remote feature branch: No
pytest -q tests/test_example.py
# [targeted counts]
git diff --check HEAD
# clean
"""
    errors = check_handoff_text(text, expected_head=_EXPECTED_HEAD)
    assert any("placeholder" in e for e in errors)


def test_stale_head_hash_is_detected():
    """A handoff referencing a hash not matching expected HEAD is stale."""
    text = """
Updated by: cc
Branch: cc/example
Commit: deadbeef Stale handoff
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""
    errors = check_handoff_text(text, expected_head=_EXPECTED_HEAD)
    assert errors == [
        "handoff hash(es) ['deadbeef'] do not match current HEAD "
        f"({_EXPECTED_HEAD}); the handoff is stale or referencing a wrong commit"
    ]


def test_handoff_requires_explicit_hash_line():
    text = """
Updated by: cc
Branch: cc/example
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# unrelated provenance 705a7a2
"""
    errors = check_handoff_text(text, expected_head=_EXPECTED_HEAD)
    assert errors == ["handoff must include an explicit Commit: or Reviewed HEAD: hash line"]


def test_short_and_full_matching_hashes_are_accepted():
    base = """
Updated by: cc
Branch: cc/example
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""
    short = f"Commit: {_EXPECTED_HEAD[:7]}\n{base}"
    full = f"Reviewed HEAD: {_EXPECTED_HEAD}\n{base}"
    assert check_handoff_text(short, expected_head=_EXPECTED_HEAD) == []
    assert check_handoff_text(full, expected_head=_EXPECTED_HEAD) == []


def test_conflicting_explicit_hash_lines_are_rejected():
    text = f"""
Updated by: cc
Branch: cc/example
Commit: {_EXPECTED_HEAD[:7]}
Reviewed HEAD: cafebabe
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""
    errors = check_handoff_text(text, expected_head=_EXPECTED_HEAD)
    assert any("conflicting" in error for error in errors)


def test_text_validation_without_expected_head_is_repository_independent():
    text = """
Updated by: cc
Branch: cc/example
Commit: deadbeef
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""
    assert check_handoff_text(text) == []


def test_handoff_requires_updater_identity():
    text = """
Branch: cc/example
Commit: deadbeef
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""
    errors = check_handoff_text(text)
    assert errors == [
        "handoff must state 'Updated by: Codex', 'Updated by: cc', "
        "or 'Updated by: GLM'"
    ]


def test_handoff_rejects_multiple_updater_identities():
    text = """
Updated by: Codex
Updated by: cc
Branch: cc/example
Commit: deadbeef
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""
    errors = check_handoff_text(text)
    assert errors == [
        "handoff must contain exactly one 'Updated by' author line"
    ]


def test_glm_is_a_valid_author():
    text = _handoff_text("GLM", "glm/example", _EXPECTED_HEAD)
    assert check_handoff_text(text, expected_head=_EXPECTED_HEAD) == []


def test_glm_author_line_with_cc_branch_is_rejected():
    text = _handoff_text("GLM", "cc/example", _EXPECTED_HEAD)
    errors = check_handoff_text(text, expected_head=_EXPECTED_HEAD)
    assert errors == [
        "GLM-authored handoff must name a local 'glm/*' branch; "
        "found branch line(s): ['cc/example']"
    ]


def test_glm_author_without_branch_line_is_rejected():
    text = """
Updated by: GLM
Commit: deadbeef
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""
    errors = check_handoff_text(text)
    assert errors == [
        "GLM-authored handoff must name a local 'glm/*' branch; "
        "found branch line(s): []"
    ]


def test_cc_author_with_glm_branch_is_rejected():
    text = _handoff_text("cc", "glm/example", _EXPECTED_HEAD)
    errors = check_handoff_text(text, expected_head=_EXPECTED_HEAD)
    assert errors == [
        "cc-authored handoff must name a local 'cc/*' branch; "
        "found branch line(s): ['glm/example']"
    ]


def test_codex_review_on_cc_and_glm_branches_is_legal():
    for branch in ("cc/example", "glm/example"):
        text = _handoff_text("Codex", branch, _EXPECTED_HEAD)
        assert check_handoff_text(text, expected_head=_EXPECTED_HEAD) == []


def test_repeated_identical_glm_author_lines_are_rejected():
    text = """
Updated by: GLM
Updated by: GLM
Branch: glm/example
Commit: deadbeef
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""
    errors = check_handoff_text(text)
    assert errors == [
        "handoff must contain exactly one 'Updated by' author line"
    ]


def test_glm_and_cc_author_lines_are_rejected():
    text = """
Updated by: GLM
Updated by: cc
Branch: glm/example
Commit: deadbeef
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""
    errors = check_handoff_text(text)
    assert errors == [
        "handoff must contain exactly one 'Updated by' author line"
    ]


def test_glm_and_unknown_author_lines_are_rejected():
    text = """
Updated by: GLM
Updated by: someone-else
Branch: glm/example
Commit: deadbeef
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
"""
    errors = check_handoff_text(text)
    assert errors == [
        "handoff must contain exactly one 'Updated by' author line"
    ]


def test_unknown_single_author_is_rejected():
    text = _handoff_text("someone-else", "glm/example", "deadbeef")
    errors = check_handoff_text(text)
    assert errors == [
        "unknown handoff author 'someone-else'; allowed authors are "
        "Codex, cc, GLM"
    ]


def test_existing_codex_and_cc_authors_remain_supported():
    for author, branch in (("Codex", "cc/example"), ("cc", "cc/example")):
        text = _handoff_text(author, branch, _EXPECTED_HEAD)
        assert check_handoff_text(text, expected_head=_EXPECTED_HEAD) == []


def test_glm_branch_must_not_track_remote_upstream():
    assert check_branch_policy("glm/example", "") == []
    errors = check_branch_policy("glm/example", "origin/glm/example")
    assert errors == [
        "glm branch 'glm/example' must not track remote upstream "
        "'origin/glm/example'"
    ]


def test_remote_glm_branches_are_rejected():
    errors = check_remote_branches(["origin/main", "origin/glm/example"])
    assert errors == ["remote glm branch exists: origin/glm/example"]


def _make_controlled_repo(tmp_path: Path, handoff_text: str) -> Path:
    """Create a throwaway git repository with a GLM branch and handoff."""

    repo = tmp_path / "protocol-repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    git("init")
    git("config", "user.email", "protocol-test@example.com")
    git("config", "user.name", "protocol-test")
    (repo / "README.md").write_text("# protocol test\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "init")
    git("checkout", "-b", "glm/example")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        stdout=subprocess.PIPE, check=True,
    ).stdout.strip()
    (repo / ".codex_cc_handoff.md").write_text(
        handoff_text.replace("{head}", head), encoding="utf-8",
    )
    return repo


def _run_checker(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--handoff", ".codex_cc_handoff.md"],
        cwd=repo, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def test_cli_accepts_glm_handoff_in_controlled_repository(tmp_path):
    repo = _make_controlled_repo(
        tmp_path,
        """
Updated by: GLM
Executor: GLM
Client: ZCODE
Branch: glm/example
Commit: {head}
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
""",
    )
    result = _run_checker(repo)
    assert result.returncode == 0, result.stderr
    assert "agent protocol check passed" in result.stdout


def test_cli_rejects_repeated_identical_glm_author(tmp_path):
    repo = _make_controlled_repo(
        tmp_path,
        """
Updated by: GLM
Updated by: GLM
Executor: GLM
Client: ZCODE
Branch: glm/example
Commit: {head}
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
""",
    )
    result = _run_checker(repo)
    assert result.returncode == 1
    assert "exactly one 'Updated by' author line" in result.stderr


def test_cli_rejects_glm_handoff_naming_cc_branch(tmp_path):
    repo = _make_controlled_repo(
        tmp_path,
        """
Updated by: GLM
Executor: GLM
Client: ZCODE
Branch: cc/example
Commit: {head}
Remote feature branch: No
pytest -q
# 100 passed in 1.00s
git diff --check HEAD
# clean
""",
    )
    result = _run_checker(repo)
    assert result.returncode == 1
    assert "GLM-authored handoff must name a local 'glm/*' branch" in result.stderr
