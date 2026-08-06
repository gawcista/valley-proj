#!/usr/bin/env python3
"""Automated portable release gate (committed snapshot -> installed artifact).

The complete gate is bound to one exact commit: any staged or unstaged
tracked difference from `HEAD` is rejected before the build, and `srctree`
is extracted from `git archive HEAD` so file contents and symlink semantics
come from the commit, never from the mutable working tree.  Wheel and sdist
are built from that snapshot, artifact contents are audited for forbidden
local material (`real_tests`, WAVECAR/HDF5 outputs, ignored development
Markdown, local `irrep2`), the wheel is installed into a fresh isolated
virtual environment, `pip check` runs, and import/CLI provenance plus the
portable production-chain acceptance are proven against the installed
package.  The installed check and the portable tests are taken from the
same snapshot.  Any failing step exits non-zero (fail closed).  Only stdlib
is required besides a Python with `build` and `pip` available.
"""

from __future__ import annotations

import argparse
import io
import os
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


def _git_head_commit(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit("FAIL: cannot resolve HEAD; is this a git checkout?")
    return result.stdout.strip()


def _checkout_clean(checkout: Path) -> str | None:
    """Return a description of any tracked difference from HEAD, else None."""
    result = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit("FAIL: git status failed; is this a git checkout?")
    entries = [line for line in result.stdout.splitlines() if line.strip()]
    if entries:
        return "\n".join(entries)
    return None


def _extract_head_archive(checkout: Path, target: Path) -> int:
    """Extract `git archive HEAD` into target; return the file count."""
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", "HEAD"],
        capture_output=True,
    )
    if archive.returncode != 0:
        raise SystemExit("FAIL: git archive HEAD failed")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r") as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(target, filter="data")
        else:
            tar.extractall(target)
    return sum(1 for _ in target.rglob("*") if _.is_file() or _.is_symlink())


def _snapshot_matches_head(checkout: Path, srctree: Path) -> list[str]:
    """Prove every snapshot file is byte-identical to HEAD's blob."""
    tree = subprocess.run(
        ["git", "-C", str(checkout), "ls-tree", "-r", "HEAD"],
        capture_output=True,
        text=True,
    )
    if tree.returncode != 0:
        raise SystemExit("FAIL: git ls-tree HEAD failed")
    # ls-tree line: "<mode> blob <sha>\t<path>" — the blob id is index 2.
    expected = {
        name: mode_blob[2]
        for name, mode_blob in (
            (entry.split("\t")[1], entry.split("\t")[0].split(" "))
            for entry in tree.stdout.splitlines()
        )
    }
    hashed = subprocess.run(
        ["git", "-C", str(checkout), "hash-object", "--stdin-paths"],
        input="".join(str(srctree / name) + "\n" for name in expected),
        capture_output=True,
        text=True,
    )
    if hashed.returncode != 0:
        raise SystemExit("FAIL: git hash-object --stdin-paths failed")
    actual = hashed.stdout.splitlines()
    mismatches = [
        name
        for name, blob in zip(expected, actual)
        if blob != expected[name]
    ]
    return mismatches


def _copy_snapshot_subtree(srctree: Path, rel_dir: str, target: Path) -> int:
    count = 0
    for path in sorted(srctree.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(srctree).as_posix()
        if name == rel_dir or name.startswith(rel_dir + "/"):
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
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

    dirty = _checkout_clean(checkout)
    if dirty is not None:
        raise SystemExit(
            "FAIL: checkout differs from HEAD; release gate is bound to one "
            f"exact commit:\n{dirty}"
        )
    commit_identity = _git_head_commit(checkout)
    print(f"gated commit: {commit_identity}")

    owned_workspace = args.work_dir is None
    workspace = (
        Path(tempfile.mkdtemp(prefix="valleyscope_release_gate_"))
        if owned_workspace
        else args.work_dir.resolve()
    )
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"workspace: {workspace}")

    srctree = workspace / "srctree"
    file_count = _extract_head_archive(checkout, srctree)
    print(f"extracted {file_count} files from git archive HEAD into srctree")
    mismatches = _snapshot_matches_head(checkout, srctree)
    if mismatches:
        raise SystemExit(
            f"FAIL: {len(mismatches)} snapshot files differ from HEAD: "
            f"{mismatches[:5]}..."
        )
    print(
        f"snapshot identity verified: srctree matches HEAD "
        f"({len(mismatches)} blobs checked, 0 mismatches)"
    )

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

    # Isolated sys.path root holding only the snapshot's tests package, so
    # `import tests` works without exposing the checkout or srctree.
    tests_root = workspace / "gatedeps"
    copied = _copy_snapshot_subtree(srctree, "tests", tests_root)
    print(f"copied {copied} files under tests/ from srctree to {tests_root}")

    # Installed check and portable tests come from the same committed
    # snapshot that produced the wheel.
    installed_check = _run([
        venv_python,
        str(srctree / "scripts" / "release_gate_installed_check.py"),
        "--checkout", str(checkout),
        "--tests-dir", str(tests_root),
        "--commit", commit_identity,
    ])
    if installed_check.returncode != 0:
        raise SystemExit("FAIL: installed-artifact check failed")

    print("release gate passed")
    if owned_workspace:
        shutil.rmtree(workspace, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
