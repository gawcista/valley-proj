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
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.release_gate import (
    _checkout_clean,
    _extract_head_archive,
    _audit_archive,
    _prepare_workspace,
    _snapshot_matches_head,
)
from scripts.release_gate_installed_check import (
    _check_installed_acceptance,
    _check_import_provenance,
    _module_in_venv,
)


def test_installed_acceptance_requires_numerical_spinful_workflow():
    """An algebraic candidate fixture alone must not satisfy the gate."""
    report = {}
    assert _check_installed_acceptance(report)
    numerical = report["portable_numerical_acceptance"]
    assert numerical["required_operations_by_hsp"] == {"GM": 3, "K": 3, "KA": 3, "M": 1}
    assert numerical["observed_hsps"] == ["GM", "K", "KA", "M"]
    assert numerical["final_reduced_ebr_result_count"] == 2
    assert numerical["broken_coefficients_final_result_count"] == 0
    assert numerical["rotation_chern_positive"]["schema_version"] == "2.2.0"
    assert numerical["rotation_chern_positive"]["rows"] == [
        {"valley": "minus", "modulus": 3, "residue": 0,
         "status": "conditional", "subspace_rank": 2},
        {"valley": "plus", "modulus": 3, "residue": 0,
         "status": "conditional", "subspace_rank": 2},
    ]
    assert numerical["rotation_chern_broken_coefficients"]["status"] == "blocked"
    noncommuting = report["noncommuting_numerical_acceptance"]
    assert noncommuting["source_space_group"] == 99
    assert noncommuting["required_operations_by_hsp"] == {"GM": 8, "X": 4, "M": 8}
    assert noncommuting["final_reduced_ebr_result_count"] == 1
    assert noncommuting["broken_coefficients_final_result_count"] == 0
    assert noncommuting["rotation_chern_positive"]["rows"] == [
        {"valley": "center", "modulus": 4, "residue": 0,
         "status": "conditional", "subspace_rank": 2},
    ]
    for acceptance in (numerical, noncommuting):
        for control in ("rotation_chern_positive", "rotation_chern_broken_coefficients"):
            assert acceptance[control]["global_valley_subspace_status"] == "not_evaluated"
        assert all(row["residue"] is None and row["status"] == "blocked"
                   for row in acceptance["rotation_chern_broken_coefficients"]["rows"])


@pytest.fixture(scope="module", params=["p3", "p4mm"])
def numerical_chern_controls(request, tmp_path_factory):
    """Amortize real numerical runs across acceptance-validator unit probes."""
    from tests.noncommuting_numerical_chain import (
        assert_noncommuting_negative, assert_noncommuting_positive,
        run_noncommuting_workflow,
    )
    from tests.portable_numerical_chain import (
        assert_broken_coefficients_blocked, assert_numerical_positive,
        run_numerical_workflow,
    )

    runner, positive, negative = (
        (run_numerical_workflow, assert_numerical_positive, assert_broken_coefficients_blocked)
        if request.param == "p3" else
        (run_noncommuting_workflow, assert_noncommuting_positive, assert_noncommuting_negative)
    )
    root = tmp_path_factory.mktemp(f"gate_chern_{request.param}")
    return {
        "positive": (runner(root / "positive"), positive),
        "negative": (runner(root / "negative", broken_coefficients=True), negative),
    }


@pytest.mark.parametrize("mutation", [
    "missing_report", "missing_residue", "wrong_residue", "null_residue",
    "wrong_status", "wrong_modulus", "wrong_rank", "missing_evidence",
    "wrong_global_status", "old_schema", "missing_valley",
])
def test_numerical_acceptance_rejects_corrupted_chern_output(numerical_chern_controls, mutation):
    """Validator unit negatives; mutated reports are not new physics evidence."""
    original, validator = numerical_chern_controls["positive"]
    result = deepcopy(original)
    summary = result["reports"]["valley_summary_json"]
    report = summary["valley_chern_mod"]
    row = report["rows"][0]
    if mutation == "missing_report":
        del summary["valley_chern_mod"]
    elif mutation == "missing_residue":
        del row["residue"]
    elif mutation == "wrong_residue":
        row["residue"] = 1
    elif mutation == "null_residue":
        row["residue"] = None
    elif mutation == "wrong_status":
        row["status"] = "blocked"
    elif mutation == "wrong_modulus":
        row["modulus"] = 2
    elif mutation == "wrong_rank":
        row["subspace_rank"] = 1
    elif mutation == "missing_evidence":
        row["eigenvalue_evidence"] = []
    elif mutation == "wrong_global_status":
        report["global_valley_subspace_status"] = "validated"
    elif mutation == "old_schema":
        summary["schema_version"] = "2.1.0"
    else:
        report["rows"].pop()
    with pytest.raises((AssertionError, KeyError)):
        validator(result)


@pytest.mark.parametrize("mutation", ["zero_filled_residue", "conditional_status", "missing_blocker"])
def test_numerical_acceptance_rejects_unblocked_broken_chern_output(numerical_chern_controls, mutation):
    """A real closure failure must retain blocked/null in the checked output."""
    original, validator = numerical_chern_controls["negative"]
    result = deepcopy(original)
    row = result["reports"]["valley_summary_json"]["valley_chern_mod"]["rows"][0]
    if mutation == "zero_filled_residue":
        row["residue"] = 0
    elif mutation == "conditional_status":
        row["status"] = "conditional"
    else:
        row["blocking_reasons"] = []
    with pytest.raises((AssertionError, KeyError)):
        validator(result)


@pytest.fixture
def arithmetic_only_gate(monkeypatch):
    """Isolate arithmetic dispatch; the unpatched numerical gate is tested above."""
    for module, name in (
        ("tests.portable_acceptance_chain", "run_installed_portable_acceptance"),
        ("tests.portable_numerical_chain", "run_installed_numerical_acceptance"),
        ("tests.noncommuting_numerical_chain", "run_installed_noncommuting_acceptance"),
    ):
        monkeypatch.setattr(f"{module}.{name}", lambda: {"validation_errors": []})
    return _check_installed_acceptance


def test_installed_gate_executes_nonzero_rotation_arithmetic(arithmetic_only_gate):
    report = {}
    assert arithmetic_only_gate(report)
    assert report["rotation_chern_arithmetic_acceptance"] == {
        "scope": "arithmetic_only_not_numerical_trust",
        "rows": [
            {"modulus": order, "residue": 1, "spinful": spinful,
             "subspace_rank": 1 if spinful else 2}
            for spinful in (False, True) for order in (2, 3, 4, 6)
        ],
    }


def test_installed_gate_rejects_constant_zero_arithmetic(arithmetic_only_gate, monkeypatch):
    """Mutation unit probe: a zero-returning installed formula cannot pass."""
    monkeypatch.setattr(
        "valleyscope.analysis.rotation_chern_math.rotation_chern_residue",
        lambda *args, **kwargs: 0,
    )
    with pytest.raises(AssertionError):
        arithmetic_only_gate({})


@pytest.mark.parametrize("optimization", ["-O", "-OO"])
def test_installed_gate_rejects_disabled_acceptance_assertions(optimization):
    checker = Path(__file__).resolve().parents[1] / "scripts/release_gate_installed_check.py"
    result = subprocess.run(
        [sys.executable, optimization, str(checker), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "Python optimization disables acceptance assertions" in result.stderr

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

    # Extra files from a reused workspace must invalidate snapshot identity.
    (srctree / "stale.py").write_text("not from HEAD")
    assert _snapshot_matches_head(repo, srctree) == ["stale.py"]


def test_reused_nonempty_workspace_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "release-gate"
    _prepare_workspace(workspace)
    assert workspace.is_dir()

    (workspace / "stale-artifact.whl").write_text("stale")
    with pytest.raises(SystemExit, match="absent or empty"):
        _prepare_workspace(workspace)


def test_user_site_valleyscope_cannot_satisfy_provenance(tmp_path: Path) -> None:
    venv_purelib = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    user_site = tmp_path / "user-site"
    roots = [venv_purelib]
    assert _module_in_venv(user_site / "valleyscope" / "__init__.py", roots) is False
    assert _module_in_venv(
        venv_purelib / "valleyscope" / "__init__.py", roots
    ) is True


def test_site_packages_must_be_under_current_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import valleyscope

    source_root = Path(valleyscope.__file__).resolve().parent.parent
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(
        "scripts.release_gate_installed_check._venv_site_packages",
        lambda: [source_root],
    )
    assert _check_import_provenance({}) is False


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
