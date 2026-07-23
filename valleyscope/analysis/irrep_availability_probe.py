"""Diagnostic-only availability probe for public irrep data sources.

The probe reports import/version/path status for the public ``irrep`` package
and the companion ``irreptables`` package used by ``irrep`` for EBR data.  It
does not call raw 3D EBR decomposition routines and never imports private
``irrep2``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from typing import Any

_UNSAFE_NATIVE_PROBE_ENV = "VALLEYSCOPE_PROBE_UNSAFE_NATIVE"
_UNSAFE_NATIVE_MODULES = {
    "irrep.ebrs": (
        "unsafe optional native probe skipped; set "
        f"{_UNSAFE_NATIVE_PROBE_ENV}=1 to import this module in a subprocess"
    ),
}


def probe_irrep_runtime_sources(
    *,
    probe_unsafe_native: bool | None = None,
) -> dict[str, Any]:
    """Return structured diagnostic status for public irrep data sources."""
    unsafe_enabled = (
        _env_truthy(os.environ.get(_UNSAFE_NATIVE_PROBE_ENV))
        if probe_unsafe_native is None
        else bool(probe_unsafe_native)
    )
    result: dict[str, Any] = {
        "irrep": _probe_package("irrep"),
        "irreptables": _probe_package("irreptables"),
        "submodules": {},
        "errors": [],
        "unsafe_native_probe_enabled": unsafe_enabled,
    }

    for module_name in (
        "irrep.spacegroup_irreps",
        "irrep.ebrs",
        "irreptables.ebrs",
    ):
        if module_name in _UNSAFE_NATIVE_MODULES and not unsafe_enabled:
            status = _probe_module_metadata(
                module_name,
                probe_skipped_reason=_UNSAFE_NATIVE_MODULES[module_name],
            )
        else:
            status = _probe_module(module_name)
        result["submodules"][module_name] = status
        if status["error"] is not None:
            result["errors"].append(f"{module_name}: {status['error']}")

    load_ebr = result["submodules"]["irreptables.ebrs"]
    result["irreptables"]["load_ebr_data_available"] = bool(
        load_ebr["available"] and "load_ebr_data" in load_ebr["public_names"]
    )
    return result


def _probe_package(package_name: str) -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": False,
        "version": None,
        "path": None,
        "error": None,
    }
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        status["error"] = "not installed"
        return status
    status["path"] = spec.origin
    try:
        status["version"] = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        status["version"] = None
    try:
        importlib.import_module(package_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        status["error"] = f"{type(exc).__name__}: {exc}"
        return status
    status["available"] = True
    return status


def _probe_module(module_name: str) -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": False,
        "path": None,
        "public_names": [],
        "error": None,
        "probe_skipped_reason": None,
    }
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        spec = None
        status["error"] = f"find_spec {type(exc).__name__}: {exc}"
    if spec is not None:
        status["path"] = spec.origin
    script = (
        "import importlib, json\n"
        f"module = importlib.import_module({module_name!r})\n"
        "print(json.dumps({"
        "'public_names': [name for name in dir(module) if not name.startswith('_')]"
        "}))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except Exception as exc:
        status["error"] = f"subprocess {type(exc).__name__}: {exc}"
        return status
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if not stderr:
            stderr = f"subprocess exited with return code {proc.returncode}"
        status["error"] = stderr
        return status
    status["available"] = True
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        status["available"] = False
        status["error"] = f"invalid probe JSON: {exc}"
        return status
    status["public_names"] = payload.get("public_names", [])
    return status


def _probe_module_metadata(
    module_name: str,
    *,
    probe_skipped_reason: str,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": False,
        "path": None,
        "public_names": [],
        "error": None,
        "probe_skipped_reason": probe_skipped_reason,
    }
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        spec = None
        status["error"] = f"find_spec {type(exc).__name__}: {exc}"
    if spec is not None:
        status["path"] = spec.origin
    return status


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
