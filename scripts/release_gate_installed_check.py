"""Installed-artifact provenance and portable acceptance check.

Runs inside the fresh isolated environment with the installed wheel.  Proves
that `valleyscope` resolves from site-packages (not the checkout), that the
public CLI entry points used by analyze/reduced-EBR/collection load, and that
the portable production chain completes with no validation errors.  Every
failure exits non-zero (fail closed).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import site
import subprocess
import sys
from pathlib import Path

CLI_COMMANDS = (
    "analyze-hsp",
    "extract-wavecar",
    "map-reduced-ebr",
    "collect-database-record",
    "collect-database-index",
    "build-reduced-ebr-table",
    "inspect-ebr-source",
    "scaffold-spec",
    "validate-spec",
)
RUNTIME_DEPENDENCIES = (
    "numpy",
    "h5py",
    "PyYAML",
    "spglib",
    "irreptables",
    "irrep",
    "sympy",
)


def _site_packages_paths() -> list[Path]:
    paths = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        paths.append(Path(user_site))
    return paths


def _check_import_provenance(report: dict[str, object]) -> bool:
    import valleyscope

    module_path = Path(valleyscope.__file__).resolve()
    in_site_packages = any(
        module_path.is_relative_to(site_path)
        for site_path in _site_packages_paths()
    )
    if not in_site_packages:
        print(
            "FAIL: valleyscope resolves to "
            f"{module_path}, not the fresh environment's site-packages",
            file=sys.stderr,
        )
        return False
    report["valleyscope_module_path"] = str(module_path)
    report["valleyscope_version"] = valleyscope.__version__
    report["resolves_from_site_packages"] = True
    print(f"valleyscope {valleyscope.__version__} at {module_path}")
    return True


def _check_cli_provenance(report: dict[str, object], cwd: Path) -> bool:
    console = Path(sys.executable).parent / (
        "valleyscope.exe" if os.name == "nt" else "valleyscope"
    )
    if not console.is_file():
        print(
            f"FAIL: console script {console} missing (entry point not "
            "installed)",
            file=sys.stderr,
        )
        return False
    loaded: list[str] = []
    for command in CLI_COMMANDS:
        result = subprocess.run(
            [str(console), command, "--help"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.returncode != 0:
            print(
                f"FAIL: CLI entry point {command!r} did not load",
                file=sys.stderr,
            )
            print(result.stderr[-2000:], file=sys.stderr)
            return False
        loaded.append(command)
    report["cli_entry_points_loaded"] = loaded
    print(f"CLI entry points loaded: {', '.join(loaded)}")
    return True


def _check_installed_acceptance(report: dict[str, object]) -> bool:
    from tests.portable_acceptance_chain import (
        run_installed_portable_acceptance,
    )

    summary = run_installed_portable_acceptance()
    report["portable_acceptance"] = summary
    print(
        "portable production-chain acceptance: "
        + json.dumps(summary, sort_keys=True)
    )
    return summary["validation_errors"] == []


def _check_dependencies(report: dict[str, object]) -> None:
    versions = {}
    for name in RUNTIME_DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report["dependency_versions"] = versions
    report["irrep2_installed"] = (
        importlib.util.find_spec("irrep2") is not None
    )
    print(
        "dependency versions: "
        + json.dumps(versions, sort_keys=True)
    )
    print(f"irrep2 installed: {report['irrep2_installed']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prove installed-wheel provenance and run the portable "
            "production-chain acceptance (release gate, installed side)."
        )
    )
    parser.add_argument(
        "--checkout",
        required=True,
        type=Path,
        help="Repository checkout root; removed from sys.path (source-tree masking).",
    )
    parser.add_argument(
        "--tests-dir",
        required=True,
        type=Path,
        help=(
            "Directory that contains the tracked tests package (i.e. its "
            "parent) used to import the acceptance builders."
        ),
    )
    args = parser.parse_args(argv)

    checkout_root = args.checkout.resolve()
    sys.path[:] = [
        str(path)
        for path in map(Path, sys.path)
        if path.resolve() != checkout_root
    ]
    tests_dir = args.tests_dir.resolve()
    sys.path.insert(0, str(tests_dir))
    # Never let the checkout or its source copy mask site-packages.
    for excluded in (checkout_root, tests_dir / "srctree"):
        sys.path[:] = [
            str(path)
            for path in map(Path, sys.path)
            if path.resolve() != excluded
        ]

    report: dict[str, object] = {"check": "installed artifact"}
    if not _check_import_provenance(report):
        return 1
    if not _check_cli_provenance(report, cwd=tests_dir.parent):
        return 1
    if not _check_installed_acceptance(report):
        return 1
    _check_dependencies(report)
    print("installed-artifact check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
