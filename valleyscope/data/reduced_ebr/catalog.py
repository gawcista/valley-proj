"""Package-data catalog for reviewed reduced-dimensional EBR tables.

Currently contains no reviewed tables.  All accessor functions are typed
and will raise clear errors for missing data rather than returning
heuristic or fallback results.

The manifest is validated at load time.  Individual tables are validated
through ``load_reduced_ebr_table`` from ``valleyscope.analysis.reduced_ebr_mapping``
so that packaged tables and user-supplied external tables pass through the
same validation path.
"""

from __future__ import annotations

import json
from pathlib import Path

from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def package_data_root() -> Path:
    """Return the absolute path to the reduced_ebr package-data directory."""
    return Path(__file__).resolve().parent


def load_reduced_ebr_manifest() -> dict:
    """Load and validate the catalog manifest.

    Returns a dict with ``schema_version``, ``description``, and ``tables``
    (a list of table entries, currently empty).

    Raises ValueError if the manifest is malformed.
    """
    manifest_path = package_data_root() / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load reduced EBR manifest: {exc}") from exc
    _validate_manifest(manifest)
    return manifest


def list_reviewed_reduced_ebr_tables() -> list[dict]:
    """Return the list of reviewed reduced EBR table entries from the manifest.

    Each entry is a dict with ``name`` and ``filename`` keys.  Returns an
    empty list when no reviewed tables exist.
    """
    manifest = load_reduced_ebr_manifest()
    return list(manifest.get("tables", []))


def load_reviewed_reduced_ebr_table(name: str) -> dict:
    """Load a reviewed reduced EBR table by its manifest name.

    Parameters
    ----------
    name : str
        The table name as listed in the manifest.

    Returns
    -------
    dict
        The validated table dict (via ``load_reduced_ebr_table``).

    Raises
    ------
    ValueError
        If no reviewed table with that name exists, or the table file fails
        external-table validation.
    FileNotFoundError
        If the manifest entry exists but the file is missing.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("table name must be a non-empty string")

    available = list_reviewed_reduced_ebr_tables()
    for entry in available:
        if entry.get("name") == name:
            filename = entry.get("filename")
            if not isinstance(filename, str) or not filename:
                raise ValueError(
                    f"manifest entry for {name!r} missing non-empty 'filename'"
                )
            table_path = _resolve_table_path(filename)
            if not table_path.is_file():
                raise FileNotFoundError(
                    f"reviewed table file not found: {table_path}"
                )
            # Route through the same validation as external tables.
            return load_reduced_ebr_table(table_path)

    available_names = [e.get("name", "?") for e in available]
    raise ValueError(
        f"no reviewed reduced EBR package table named {name!r}. "
        f"Available tables: {available_names}"
    )


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_manifest(manifest: object) -> None:
    """Validate the catalog manifest structure.

    Raises ValueError for malformed manifests.
    """
    if not isinstance(manifest, dict):
        raise ValueError("reduced EBR manifest must be a JSON object (dict)")

    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("manifest schema_version must be a non-empty string")

    tables = manifest.get("tables")
    if not isinstance(tables, list):
        raise ValueError("manifest 'tables' must be a list")

    seen_names: set[str] = set()
    for i, entry in enumerate(tables):
        if not isinstance(entry, dict):
            raise ValueError(
                f"manifest tables[{i}] must be a JSON object (dict)"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"manifest tables[{i}] must have a non-empty 'name' string"
            )
        if name in seen_names:
            raise ValueError(
                f"manifest tables[{i}] duplicate name {name!r}"
            )
        seen_names.add(name)

        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"manifest tables[{i}] ({name!r}) must have a non-empty 'filename'"
            )
        # Reject absolute paths and parent-directory traversal.
        _validate_filename_safe(filename, name)


def _validate_filename_safe(filename: str, name: str) -> None:
    """Reject unsafe filenames: absolute paths, .. traversal, empty segments."""
    if Path(filename).is_absolute():
        raise ValueError(
            f"manifest table {name!r} filename must be relative, "
            f"got {filename!r}"
        )
    parts = Path(filename).parts
    if not parts or parts[0] == "..":
        raise ValueError(
            f"manifest table {name!r} filename must not resolve outside "
            f"the package-data directory, got {filename!r}"
        )
    if ".." in parts:
        raise ValueError(
            f"manifest table {name!r} filename must not contain '..', "
            f"got {filename!r}"
        )


def _resolve_table_path(filename: str) -> Path:
    """Resolve a manifest filename to an absolute path inside the package-data dir."""
    root = package_data_root().resolve()
    resolved = (root / filename).resolve()
    # Double-check: resolved path must stay inside package_data_root.
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            f"table filename {filename!r} resolves outside "
            f"package-data directory: {resolved}"
        ) from None
    return resolved
