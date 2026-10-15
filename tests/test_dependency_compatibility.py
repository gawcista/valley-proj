"""Dependency-policy and installed-interface checks; no physics evidence is mocked."""

from __future__ import annotations

import subprocess
import sys
from email import message_from_string
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement

from scripts.release_gate_installed_check import _check_irrep_compatibility


@pytest.fixture(scope="session")
def built_requirements(tmp_path_factory):
    """Read the actual metadata an editable/wheel installer consumes."""
    output = tmp_path_factory.mktemp("dependency_metadata")
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; from setuptools.build_meta import prepare_metadata_for_build_wheel; "
            "prepare_metadata_for_build_wheel(sys.argv[1])"
        ), str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = next(output.glob("*.dist-info/METADATA"))
    return message_from_string(metadata.read_text())["Requires-Python"], [
        Requirement(line) for line in message_from_string(metadata.read_text()).get_all("Requires-Dist", [])
    ]


@pytest.mark.parametrize("python_version,allowed", [
    ("3.10", {"3.1.1", "3.2.0"}),
    ("3.11", {"3.1.1", "3.2.0"}),
    ("3.12", {"3.1.1", "3.2.0", "3.3.0", "3.3.0+valleyscope.1"}),
    ("3.13", {"3.1.1", "3.2.0", "3.3.0", "3.3.0+valleyscope.1"}),
])
def test_metadata_allows_usable_public_irrep_candidates(built_requirements, python_version, allowed):
    """Catch metadata that leaves old Python with only an unimportable/private candidate."""
    _, requirements = built_requirements
    active = [
        req for req in requirements if req.name.lower() == "irrep"
        and (req.marker is None or req.marker.evaluate({"python_version": python_version, "extra": ""}))
    ]
    assert active, f"missing irrep runtime dependency on Python {python_version}"
    candidates = {"2.6.3", "3.1.1", "3.2.0", "3.3.0", "3.3.0+valleyscope.1", "3.4.0"}
    accepted = {value for value in candidates if all(value in req.specifier for req in active)}
    assert accepted == allowed


def test_metadata_keeps_reviewed_table_data(built_requirements):
    _, requirements = built_requirements
    tables = [req for req in requirements if req.name.lower() == "irreptables"]
    assert tables
    assert all("3.1.0" in req.specifier for req in tables)
    assert not all("3.0.0" in req.specifier for req in tables)
    assert not all("3.2.0" in req.specifier for req in tables)


@pytest.fixture
def installed_interfaces(tmp_path, monkeypatch, built_requirements):
    """Explicit unit fixture for version/origin/export validation only."""
    import scripts.release_gate_installed_check as check

    site = tmp_path / "site-packages"
    versions = {"irrep": "3.1.1", "irreptables": "3.1.0"}

    class Distribution:
        def __init__(self, name):
            self.name = name

        @property
        def version(self):
            return versions[self.name]

        def locate_file(self, name):
            return site / name

    exports = {
        "irrep.utility": ("str2bool", "str2list_space", "str_", "log_message"),
        "irreptables.irreps": ("IrrepTable",),
        "irreptables.ebrs": ("load_ebr_data",),
    }
    modules = {
        name: SimpleNamespace(
            __file__=str(site / (name.replace(".", "/") + ".py")),
            **{symbol: lambda *args, **kwargs: None for symbol in symbols},
        ) for name, symbols in exports.items()
    }
    original_distribution = check.importlib.metadata.distribution
    original_requirements = check.importlib.metadata.requires
    original_import = check.importlib.import_module
    monkeypatch.setattr(check.importlib.metadata, "distribution", lambda name:
                        Distribution(name) if name in versions else original_distribution(name))
    monkeypatch.setattr(check.importlib.metadata, "requires", lambda name:
                        [str(req) for req in built_requirements[1]] if name == "valleyscope"
                        else original_requirements(name))
    monkeypatch.setattr(check.importlib, "import_module", lambda name:
                        modules[name] if name in modules else original_import(name))
    monkeypatch.setattr(check, "_venv_site_packages", lambda: [site])
    return versions, modules, site


@pytest.mark.parametrize("version", ["3.1.1", "3.2.0"])
def test_installed_public_irrep_interfaces_are_accepted(installed_interfaces, version):
    versions, _, _ = installed_interfaces
    versions["irrep"] = version
    report = {}
    assert _check_irrep_compatibility(report)
    assert report["symmetry_dependencies"]["versions"] == {"irrep": version, "irreptables": "3.1.0"}


@pytest.mark.parametrize("mutation", [
    "wrong_irrep", "unreviewed_tables", "missing_requirement", "missing_export",
    "noncallable_export", "shadow_utility", "shadow_tables", "outside_venv", "import_failure",
])
def test_installed_interfaces_reject_incompatible_or_untraceable_inputs(
    installed_interfaces, mutation, monkeypatch, tmp_path,
):
    import scripts.release_gate_installed_check as check

    versions, modules, site = installed_interfaces
    if mutation == "wrong_irrep":
        versions["irrep"] = "9.0.0"
    elif mutation == "unreviewed_tables":
        versions["irreptables"] = "3.2.0"
    elif mutation == "missing_requirement":
        monkeypatch.setattr(check.importlib.metadata, "requires", lambda name: [])
    elif mutation == "missing_export":
        del modules["irreptables.ebrs"].load_ebr_data
    elif mutation == "noncallable_export":
        modules["irreptables.irreps"].IrrepTable = None
    elif mutation in ("shadow_utility", "shadow_tables"):
        name = "irrep.utility" if mutation == "shadow_utility" else "irreptables.irreps"
        modules[name].__file__ = str(site / "shadow.py")
    elif mutation == "outside_venv":
        monkeypatch.setattr(check, "_venv_site_packages", lambda: [tmp_path / "other"])
    else:
        original = check.importlib.import_module

        def broken_import(name):
            if name == "irreptables.irreps":
                raise ImportError("broken installed interface")
            return original(name)

        monkeypatch.setattr(check.importlib, "import_module", broken_import)
    assert not _check_irrep_compatibility({})
