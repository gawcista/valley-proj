"""Package-data catalog for reduced-dimensional EBR tables.

No static reviewed physical reduced EBR tables are shipped here. The
manifest is empty. Production reduced EBR data comes from the irreptables
runtime reducer (``valleyscope/analysis/irreptables_runtime_table_builder.py``)
or from user-supplied external table files loaded through
``valleyscope.analysis.reduced_ebr_mapping.load_reduced_ebr_table()``.

This module retains minimal inert helpers for the manifest and package-data
root; no static table selector API is exposed.
"""

from __future__ import annotations

import json
from pathlib import Path


def package_data_root() -> Path:
    """Return the absolute path to the reduced_ebr package-data directory."""
    return Path(__file__).resolve().parent


def load_reduced_ebr_manifest() -> dict:
    """Load and validate the catalog manifest.

    Returns a dict with ``schema_version`` and ``description``.
    The ``tables`` and ``basis_maps`` lists are always empty.

    Raises ValueError if the manifest is malformed.
    """
    manifest_path = package_data_root() / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load reduced EBR manifest: {exc}") from exc
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: object) -> None:
    """Validate the catalog manifest structure."""
    if not isinstance(manifest, dict):
        raise ValueError("reduced EBR manifest must be a JSON object (dict)")

    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("manifest schema_version must be a non-empty string")

    tables = manifest.get("tables")
    if not isinstance(tables, list):
        raise ValueError("manifest 'tables' must be a list")

    basis_maps = manifest.get("basis_maps")
    if not isinstance(basis_maps, list):
        raise ValueError("manifest 'basis_maps' must be a list")
