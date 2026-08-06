#!/usr/bin/env python3
"""Automated portable release gate (source tree -> installed artifact).

Builds wheel and sdist from a clean copy of the checkout, audits artifact
contents for forbidden local material (`real_tests`, WAVECAR/HDF5 outputs,
ignored development Markdown, local `irrep2`), installs the wheel into a
fresh isolated virtual environment, runs `pip check`, and proves import/CLI
provenance plus the portable production-chain acceptance against the
installed package.  Any failing step exits non-zero (fail closed).  Only
stdlib is required besides a Python with `build` and `pip` available.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ALLOWED_TOP_LEVEL_MARKDOWN = frozenset({"README.md", "README.zh.md"})
FORBIDDEN_COMPONENTS = frozenset({
    "real_tests",
    "WAVECAR",
    "CHGCAR",
    "CHG",
    "vasprun.xml",
    "OUTCAR",
    "EIGENVAL",
    "PROCAR",
    "irrep2",
    "valley_analysis",
    "__pycache__",
    ".pytest_cache",
})
FORBIDDEN_SUFFIXES = (".h5", ".hdf5", ".pyc")


def _run(command: list[str | Path], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    command = [str(part) for part in command]
    print(f"+ {' '.join(command)}")
    return subprocess.run(command, cwd=cwd, text=True)


def _git_tracked(checkout: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", "-z"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit("FAIL: git ls-files failed; is this a git checkout?")
    return [name for name in result.stdout.split("\0") if name]


def _copy_tracked_tree(checkout: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in _git_tracked(checkout):
        source = checkout / name
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_tracked_subtree(checkout: Path, rel_dir: str, target: Path) -> int:
    count = 0
    for name in _git_tracked(checkout):
        if name == rel_dir or name.startswith(rel_dir + "/"):
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkout / name, destination)
            count += 1
    return count


def _audit_archive(path: Path) -> list[str]:
    """Return a list of forbidden entries inside a wheel or sdist."""
    names: list[str]
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            names = archive.getnames()
    violations = []
    for name in names:
        components = Path(name).parts
        if any(component in FORBIDDEN_COMPONENTS for component in components):
            violations.append(name)
            continue
        lowered = name.lower()
        if lowered.endswith(FORBIDDEN_SUFFIXES):
            violations.append(name)
            continue
        # Only the sdist root may carry README; any other Markdown (including
        # the ignored `valleyscope/data/reduced_ebr/README.md`) is forbidden.
        if lowered.endswith(".md") and not (
            len(components) == 2
            and components[1] in ALLOWED_TOP_LEVEL_MARKDOWN
        ):
            violations.append(name)
    return violations


def _build_artifacts(python: str, srctree: Path, dist_dir: Path) -> list[Path]:
    if _run([python, "-m", "build", "--outdir", str(dist_dir), str(srctree)]).returncode != 0:
        raise SystemExit(
            "FAIL: wheel/sdist build failed (is the `build` frontend "
            "installed for this interpreter?)"
        )
    artifacts = sorted(dist_dir.iterdir())
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(
            f"FAIL: expected exactly one wheel and one sdist, got {artifacts}"
        )
    return wheels + sdists


def _create_fresh_venv(python: str, venv_dir: Path) -> Path:
    if _run([python, "-m", "venv", "--clear", str(venv_dir)]).returncode != 0:
        raise SystemExit("FAIL: venv creation failed")
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Automated portable release gate: build, audit, fresh-env "
            "install, provenance, and installed acceptance."
        )
    )
    parser.add_argument(
        "--checkout",
        required=True,
        type=Path,
        help="Repository checkout root (must contain pyproject.toml).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to build and create the venv.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Optional workspace directory (created if missing; kept on exit).",
    )
    args = parser.parse_args(argv)

    checkout = args.checkout.resolve()
    if not (checkout / "pyproject.toml").is_file():
        raise SystemExit(f"FAIL: no pyproject.toml under {checkout}")

    owned_workspace = args.work_dir is None
    workspace = (
        Path(tempfile.mkdtemp(prefix="valleyscope_release_gate_"))
        if owned_workspace
        else args.work_dir.resolve()
    )
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"workspace: {workspace}")

    srctree = workspace / "srctree"
    _copy_tracked_tree(checkout, srctree)
    print(f"copied {sum(1 for _ in srctree.rglob('*'))} tracked files to srctree")

    dist_dir = workspace / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _build_artifacts(args.python, srctree, dist_dir)

    for artifact in artifacts:
        violations = _audit_archive(artifact)
        if violations:
            raise SystemExit(
                f"FAIL: forbidden content in {artifact.name}: {violations}"
            )
        names = (
            zipfile.ZipFile(artifact).namelist()
            if artifact.suffix == ".whl"
            else tarfile.open(artifact).getnames()
        )
        print(
            f"audit ok: {artifact.name} ({len(names)} entries, "
            f"no forbidden local material)"
        )

    venv_python = _create_fresh_venv(args.python, workspace / "venv")
    wheel = next(path for path in artifacts if path.suffix == ".whl")
    if _run([
        venv_python, "-m", "pip", "install",
        "--disable-pip-version-check", str(wheel),
    ]).returncode != 0:
        raise SystemExit("FAIL: wheel install into fresh environment failed")

    pip_check = _run([venv_python, "-m", "pip", "check"])
    if pip_check.returncode != 0:
        raise SystemExit("FAIL: pip check reported broken dependencies")

    # Isolated sys.path root holding only the tracked tests package, so
    # `import tests` works without exposing the checkout or srctree.
    tests_root = workspace / "gatedeps"
    copied = _copy_tracked_subtree(checkout, "tests", tests_root)
    print(f"copied {copied} tracked files under tests/ to {tests_root}")

    installed_check = _run([
        venv_python, str(checkout / "scripts" / "release_gate_installed_check.py"),
        "--checkout", str(checkout),
        "--tests-dir", str(tests_root),
    ])
    if installed_check.returncode != 0:
        raise SystemExit("FAIL: installed-artifact check failed")

    print("release gate passed")
    if owned_workspace:
        shutil.rmtree(workspace, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
