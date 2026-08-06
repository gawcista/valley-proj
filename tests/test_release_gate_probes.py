"""Focused negative probes for the release-gate snapshot/provenance contract.

These stay small and independent of real fixtures: each probe builds its own
tiny git checkout or wheel/sdist in a temporary directory.  They prove that
a dirty tracked mutation is rejected before the build, that an injected
user-site/PYTHONPATH ValleyScope cannot satisfy venv provenance, and that
artifact audits reject forbidden local material.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.release_gate import (
    _checkout_clean,
    _extract_head_archive,
    _audit_archive,
    _snapshot_matches_head,
)
from scripts.release_gate_installed_check import _module_in_venv

NEEDS_GIT = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is required for snapshot probes"
)


def _git(command: list[str], cwd: Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + command,
        cwd=cwd,
        capture_output=True,
        text=True,
        **kwargs,
    )


def _tiny_repo(root: Path, content: str = "committed") -> Path:
    """Init a git repo with one committed file; return the checkout root."""
    _git(["init", "-q", "-b", "main", str(root)], root.parent)
    tracked = root / "tracked.txt"
    tracked.write_text(content)
    _git(["add", "tracked.txt"], root)
    _git(
        [
            "-c", "user.name=probe", "-c", "user.email=probe@example.invalid",
            "commit", "-q", "-m", "seed",
        ],
        root,
    )
    assert _git(["rev-parse", "HEAD"], root).returncode == 0
    return root


@NEEDS_GIT
def test_dirty_tracked_mutation_rejected_before_build(tmp_path: Path) -> None:
    repo = _tiny_repo(tmp_path / "repo")
    assert _checkout_clean(repo) is None

    # Unstaged tracked mutation must be rejected.
    (repo / "tracked.txt").write_text("dirty")
    description = _checkout_clean(repo)
    assert description is not None
    assert "tracked.txt" in description

    # Staged tracked mutation must be rejected too.
    _git(["add", "tracked.txt"], repo)
    description = _checkout_clean(repo)
    assert description is not None
    assert "tracked.txt" in description


@NEEDS_GIT
def test_untracked_and_ignored_content(tmp_path: Path) -> None:
    repo = _tiny_repo(tmp_path / "repo")
    (repo / "untracked.txt").write_text("new")
    assert _checkout_clean(repo) is not None
    (repo / "untracked.txt").unlink()

    # Ignored development files must not fail the gate (they never enter
    # the snapshot).
    (repo / ".gitignore").write_text("ignored.md\n")
    _git(["add", ".gitignore"], repo)
    _git(
        [
            "-c", "user.name=probe", "-c", "user.email=probe@example.invalid",
            "commit", "-q", "-m", "gitignore",
        ],
        repo,
    )
    (repo / "ignored.md").write_text("local dev note")
    assert _checkout_clean(repo) is None


@NEEDS_GIT
def test_archive_snapshot_matches_head(tmp_path: Path) -> None:
    repo = _tiny_repo(tmp_path / "repo")
    srctree = tmp_path / "srctree"
    count = _extract_head_archive(repo, srctree)
    assert count == 1
    assert (srctree / "tracked.txt").read_text() == "committed"
    assert _snapshot_matches_head(repo, srctree) == []

    # A dirty worktree must not change the snapshot (archive comes from
    # HEAD, not from the mutable working tree).
    (repo / "tracked.txt").write_text("dirty")
    assert (srctree / "tracked.txt").read_text() == "committed"
    assert _snapshot_matches_head(repo, srctree) == []


def test_user_site_valleyscope_cannot_satisfy_provenance(tmp_path: Path) -> None:
    venv_purelib = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    user_site = tmp_path / "user-site"
    roots = [venv_purelib]
    assert _module_in_venv(user_site / "valleyscope" / "__init__.py", roots) is False
    assert _module_in_venv(
        venv_purelib / "valleyscope" / "__init__.py", roots
    ) is True


def test_pythonpath_injection_cannot_satisfy_provenance(tmp_path: Path) -> None:
    fake = tmp_path / "fake-pythonpath"
    (fake / "valleyscope").mkdir(parents=True)
    (fake / "valleyscope" / "__init__.py").write_text(
        "__version__ = '0.0.0'\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "from scripts.release_gate_installed_check import "
        "_check_import_provenance; "
        "raise SystemExit(0 if _check_import_provenance({}) else 1)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(fake), str(repo_root)])
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    # The injected fake wins the import but must fail provenance.
    assert result.returncode == 1, result.stdout + result.stderr


def _wheel_with(directory: Path, names: list[str]) -> Path:
    path = directory / "probe.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, "x")
    return path


def _sdist_with(directory: Path, names: list[str]) -> Path:
    path = directory / "probe.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            archive.addfile(tarfile.TarInfo(name), io.BytesIO(b"x"))
    return path


def test_audit_rejects_forbidden_local_material(tmp_path: Path) -> None:
    wheel = _wheel_with(
        tmp_path,
        [
            "valleyscope/__init__.py",
            "valleyscope/data/reduced_ebr/README.md",  # nested Markdown
            "real_tests/anything.py",  # local material
            "valleyscope/module.pyc",  # compiled bytecode
            "valleyscope/data/WAVECAR",  # local DFT output
        ],
    )
    violations = _audit_archive(wheel)
    assert violations == [
        "valleyscope/data/reduced_ebr/README.md",
        "real_tests/anything.py",
        "valleyscope/module.pyc",
        "valleyscope/data/WAVECAR",
    ]


def test_audit_allows_sdist_root_readme(tmp_path: Path) -> None:
    sdist = _sdist_with(
        tmp_path,
        [
            "valleyscope-0.1.0/README.md",
            "valleyscope-0.1.0/valleyscope/__init__.py",
        ],
    )
    assert _audit_archive(sdist) == []
