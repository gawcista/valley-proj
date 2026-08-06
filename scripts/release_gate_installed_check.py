"""Installed-artifact provenance and portable acceptance check.

Runs inside the fresh isolated environment with the installed wheel.  Proves
that `valleyscope` resolves from this interpreter's own venv
`purelib`/`platlib` under `sys.prefix` (never user-site or PYTHONPATH), that
the public CLI entry points used by analyze/reduced-EBR/collection load from
the same venv, that every declared runtime dependency is present, that
`irrep2`/OR-Tools are absent from the environment and never enter
`sys.modules` during the acceptance, and that the portable production chain
completes with no validation errors.  Every failure exits non-zero (fail
closed).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import sysconfig
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
FORBIDDEN_MODULES = ("irrep.ebrs", "ortools", "irrep2")


def _module_in_venv(module_path: Path, site_packages: list[Path]) -> bool:
    return any(module_path.is_relative_to(site_path) for site_path in site_packages)


def _venv_site_packages() -> list[Path]:
    """This interpreter's own purelib/platlib under sys.prefix (no user-site)."""
    paths = []
    for key in ("purelib", "platlib"):
        path = sysconfig.get_path(key)
        if path:
            paths.append(Path(path).resolve())
    return paths


def _check_import_provenance(report: dict[str, object]) -> bool:
    import valleyscope

    module_path = Path(valleyscope.__file__).resolve()
    site_packages = _venv_site_packages()
    report["venv_purelib_platlib"] = [str(path) for path in site_packages]
    prefix = Path(sys.prefix).resolve()
    if not site_packages or any(
        not site_path.is_relative_to(prefix) for site_path in site_packages
    ):
        print(
            "FAIL: purelib/platlib are not contained under this interpreter's "
            f"sys.prefix {prefix}: {site_packages}",
            file=sys.stderr,
        )
        return False
    if not _module_in_venv(module_path, site_packages):
        print(
            "FAIL: valleyscope resolves to "
            f"{module_path}, not this venv's purelib/platlib "
            f"({site_packages})",
            file=sys.stderr,
        )
        return False
    report["valleyscope_module_path"] = str(module_path)
    report["valleyscope_version"] = valleyscope.__version__
    report["resolves_from_venv"] = True
    print(f"valleyscope {valleyscope.__version__} at {module_path}")
    print("venv purelib/platlib: " + ", ".join(str(path) for path in site_packages))
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
    # The console script must be this venv's own: its shebang must invoke
    # the interpreter under sys.prefix.
    with console.open("r", errors="replace") as handle:
        first_line = handle.readline()
    if os.name != "nt" and str(sys.prefix) not in first_line:
        print(
            f"FAIL: console script {console} does not belong to this venv "
            f"(sys.prefix {sys.prefix} not in shebang: {first_line.strip()})",
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
    report["console_script"] = str(console)
    print(f"CLI entry points loaded: {', '.join(loaded)}")
    print(f"console script: {console} (belongs to this venv)")
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


def _check_environment(report: dict[str, object]) -> bool:
    """Fail closed: every declared runtime dependency present, and no
    `irrep2`/OR-Tools installed in this fresh environment."""
    versions = {}
    missing = []
    for name in RUNTIME_DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            missing.append(name)
    report["dependency_versions"] = versions
    report["missing_dependencies"] = missing
    report["irrep2_installed"] = (
        importlib.util.find_spec("irrep2") is not None
    )
    report["ortools_installed"] = (
        importlib.util.find_spec("ortools") is not None
    )
    print(
        "dependency versions: "
        + json.dumps(versions, sort_keys=True)
    )
    print(f"irrep2 installed: {report['irrep2_installed']}")
    print(f"OR-Tools installed: {report['ortools_installed']}")
    if missing:
        print(
            f"FAIL: missing declared runtime dependencies: {missing}",
            file=sys.stderr,
        )
        return False
    if report["irrep2_installed"]:
        print("FAIL: irrep2 must not be installed in the fresh environment", file=sys.stderr)
        return False
    if report["ortools_installed"]:
        print("FAIL: OR-Tools must not be installed in the fresh environment", file=sys.stderr)
        return False
    return True


def _check_forbidden_imports(report: dict[str, object]) -> bool:
    """Fail if irrep.ebrs/OR-Tools/irrep2 entered sys.modules during the
    acceptance (public `irrep` itself remains expected)."""
    entered = sorted(
        module
        for module in sys.modules
        if module in FORBIDDEN_MODULES
        or any(module.startswith(name + ".") for name in FORBIDDEN_MODULES)
    )
    report["forbidden_modules_entered"] = entered
    print(
        f"forbidden modules in sys.modules after acceptance: "
        f"{entered if entered else 'none'}"
    )
    if entered:
        print(
            f"FAIL: forbidden modules entered sys.modules during acceptance: "
            f"{entered}",
            file=sys.stderr,
        )
        return False
    return True


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
    parser.add_argument(
        "--commit",
        required=True,
        help="Full commit hash the gate is bound to (provenance identity).",
    )
    args = parser.parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{40}", args.commit) is None:
        parser.error("--commit must be a full 40-character lowercase Git hash")

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

    print(f"gated commit: {args.commit}")
    print(f"interpreter: {sys.executable} (sys.prefix {sys.prefix})")
    report: dict[str, object] = {
        "check": "installed artifact",
        "gated_commit": args.commit,
    }
    if not _check_import_provenance(report):
        return 1
    if not _check_cli_provenance(report, cwd=tests_dir.parent):
        return 1
    if not _check_installed_acceptance(report):
        return 1
    if not _check_forbidden_imports(report):
        return 1
    if not _check_environment(report):
        return 1
    print("installed-artifact check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
