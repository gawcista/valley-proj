"""Package-data catalog for reviewed reduced-dimensional EBR tables.

Currently contains no reviewed tables.  All accessor functions are typed
and will raise clear errors for missing data rather than returning
heuristic or fallback results.
"""

from __future__ import annotations

import json
from pathlib import Path


def package_data_root() -> Path:
    """Return the absolute path to the reduced_ebr package-data directory."""
    return Path(__file__).resolve().parent


def load_reduced_ebr_manifest() -> dict:
    """Load the catalog manifest.

    Returns a dict with ``schema_version``, ``description``, and ``tables``
    (a list of table entries, currently empty).
    """
    manifest_path = package_data_root() / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def list_reviewed_reduced_ebr_tables() -> list[dict]:
    """Return the list of reviewed reduced EBR table entries from the manifest.

    Each entry is the manifest table dict.  Returns an empty list when no
    reviewed tables exist.
    """
    manifest = load_reduced_ebr_manifest()
    tables = manifest.get("tables", [])
    if not isinstance(tables, list):
        return []
    return tables


def load_reviewed_reduced_ebr_table(name: str) -> dict:
    """Load a reviewed reduced EBR table by its manifest name.

    Parameters
    ----------
    name : str
        The table name as listed in the manifest (typically the JSON filename
        without extension, e.g. ``"C3_like_GammaM_KM"``).

    Returns
    -------
    dict
        The validated table dict.

    Raises
    ------
    ValueError
        If no reviewed table with that name exists in the manifest.
    FileNotFoundError
        If the manifest entry exists but the file is missing.
    """
    available = list_reviewed_reduced_ebr_tables()
    for entry in available:
        if entry.get("name") == name:
            filename = entry.get("filename")
            if not filename:
                raise ValueError(
                    f"manifest entry for {name!r} missing 'filename' key"
                )
            table_path = package_data_root() / filename
            if not table_path.is_file():
                raise FileNotFoundError(
                    f"reviewed table file not found: {table_path}"
                )
            return json.loads(table_path.read_text(encoding="utf-8"))

    available_names = [e.get("name", "?") for e in available]
    raise ValueError(
        f"no reviewed reduced EBR package table named {name!r}. "
        f"Available tables: {available_names}"
    )
