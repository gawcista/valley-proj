"""Package-data catalog for reviewed reduced-dimensional EBR tables.

The catalog contains only reviewed tables.  All accessor functions are typed
and will raise clear errors for missing or unreviewed data rather than
returning heuristic or fallback results.

The manifest is validated at load time and requires explicit review
metadata for every table entry.  Individual tables are validated through
``load_reduced_ebr_table`` from ``valleyscope.analysis.reduced_ebr_mapping``
for the basic schema contract; the catalog layer additionally requires
reviewed provenance on both the manifest entry and the loaded table's
``provenance`` block.

External tables loaded through
``valleyscope.analysis.reduced_ebr_mapping.load_reduced_ebr_table()`` are
NOT subject to reviewed-provenance checks.  Only the package-data catalog
path enforces reviewed provenance.
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
    (a list of reviewed table entries).

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

    Each entry is a dict with ``name`` and ``filename`` keys.
    """
    manifest = load_reduced_ebr_manifest()
    return list(manifest.get("tables", []))


def list_reviewed_reduced_ebr_basis_maps() -> list[dict]:
    """Return reviewed reduced EBR basis-map entries from the manifest."""
    manifest = load_reduced_ebr_manifest()
    return list(manifest.get("basis_maps", []))


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
            # Route through the same basic schema validation as external tables.
            table = load_reduced_ebr_table(table_path)
            # --- reviewed provenance gate (table content) ---
            _validate_table_reviewed_provenance(table, name)
            return table

    available_names = [e.get("name", "?") for e in available]
    raise ValueError(
        f"no reviewed reduced EBR package table named {name!r}. "
        f"Available tables: {available_names}"
    )


def load_reviewed_reduced_ebr_basis_map(name: str) -> dict:
    """Load a reviewed reduced EBR basis map by manifest name."""
    if not isinstance(name, str) or not name:
        raise ValueError("basis-map name must be a non-empty string")

    available = list_reviewed_reduced_ebr_basis_maps()
    for entry in available:
        if entry.get("name") == name:
            filename = entry.get("filename")
            if not isinstance(filename, str) or not filename:
                raise ValueError(
                    f"manifest basis-map entry for {name!r} missing "
                    "non-empty 'filename'"
                )
            map_path = _resolve_table_path(filename)
            if not map_path.is_file():
                raise FileNotFoundError(
                    f"reviewed basis-map file not found: {map_path}"
                )
            try:
                payload = json.loads(map_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"cannot load reviewed basis-map JSON {name!r}: {exc}"
                ) from exc
            _validate_basis_map_payload(payload, entry, name)
            return payload

    available_names = [e.get("name", "?") for e in available]
    raise ValueError(
        f"no reviewed reduced EBR basis map named {name!r}. "
        f"Available basis maps: {available_names}"
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
        # --- reviewed provenance gate (manifest entry) ---
        _validate_entry_review_metadata(entry, i, name)

        # Reject absolute paths and parent-directory traversal.
        _validate_filename_safe(filename, name)

    basis_maps = manifest.get("basis_maps", [])
    if not isinstance(basis_maps, list):
        raise ValueError("manifest 'basis_maps' must be a list")

    seen_basis_map_names: set[str] = set()
    for i, entry in enumerate(basis_maps):
        if not isinstance(entry, dict):
            raise ValueError(
                f"manifest basis_maps[{i}] must be a JSON object (dict)"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"manifest basis_maps[{i}] must have a non-empty 'name' string"
            )
        if name in seen_basis_map_names:
            raise ValueError(
                f"manifest basis_maps[{i}] duplicate name {name!r}"
            )
        seen_basis_map_names.add(name)

        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"manifest basis_maps[{i}] ({name!r}) must have a "
                "non-empty 'filename'"
            )
        table_name = entry.get("table_name")
        if not isinstance(table_name, str) or not table_name:
            raise ValueError(
                f"manifest basis_maps[{i}] ({name!r}) must have a "
                "non-empty 'table_name'"
            )
        _validate_entry_review_metadata(
            entry, i, name, collection="basis_maps",
        )
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


_REQUIRED_MANIFEST_REVIEW_KEYS = frozenset({
    "review_status", "reviewer", "review_date", "review_method",
    "source_reference",
})


def _validate_entry_review_metadata(
    entry: dict,
    index: int,
    name: str,
    *,
    collection: str = "tables",
) -> None:
    """Validate that a manifest table entry carries reviewed provenance.

    Every reviewed package-data table entry must carry explicit review
    metadata.  Missing keys, empty values, or a non-"reviewed" status
    raise ValueError.
    """
    for key in sorted(_REQUIRED_MANIFEST_REVIEW_KEYS):
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"manifest {collection}[{index}] ({name!r}) must have a non-empty "
                f"'{key}' string"
            )
    if entry["review_status"] != "reviewed":
        raise ValueError(
            f"manifest {collection}[{index}] ({name!r}) review_status must be "
            f"'reviewed', got {entry['review_status']!r}"
        )


_REQUIRED_TABLE_PROVENANCE_KEYS = frozenset({
    "review_status", "reviewer", "review_date", "review_method",
    "source_reference",
})


def _validate_table_reviewed_provenance(table: dict, name: str) -> None:
    """Validate that a loaded reviewed table carries explicit provenance.

    The table's top-level ``provenance`` block must be a dict with
    non-empty review metadata and ``valleyscope_reduction ==
    "sampled_hsp_valley_preserving"``.
    """
    provenance = table.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            f"reviewed table {name!r} must have a 'provenance' object "
            f"(dict), got {type(provenance).__name__}"
        )
    for key in sorted(_REQUIRED_TABLE_PROVENANCE_KEYS):
        value = provenance.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"reviewed table {name!r} provenance.{key} must be a "
                f"non-empty string"
            )
    if provenance["review_status"] != "reviewed":
        raise ValueError(
            f"reviewed table {name!r} provenance.review_status must be "
            f"'reviewed', got {provenance['review_status']!r}"
        )
    reduction = provenance.get("valleyscope_reduction")
    if reduction != "sampled_hsp_valley_preserving":
        raise ValueError(
            f"reviewed table {name!r} provenance.valleyscope_reduction "
            f"must be 'sampled_hsp_valley_preserving', got {reduction!r}"
        )

    # --- Physical identity provenance ---
    _validate_provenance_identity(table, provenance, name)


def _validate_provenance_identity(
    table: dict, provenance: dict, name: str,
) -> None:
    """Validate physical identity fields in the provenance block and
    cross-check against the table's top-level fields."""
    # data_source: non-empty string.
    ds = provenance.get("data_source")
    if not isinstance(ds, str) or not ds.strip():
        raise ValueError(
            f"reviewed table {name!r} provenance.data_source must be "
            f"a non-empty string"
        )

    # space_group_number: int or non-empty string.
    sg = provenance.get("space_group_number")
    if isinstance(sg, bool) or not isinstance(sg, (int, str)) or (
        isinstance(sg, str) and not sg.strip()
    ):
        raise ValueError(
            f"reviewed table {name!r} provenance.space_group_number must "
            f"be an int or non-empty string, got {sg!r}"
        )

    # spinful: bool.
    spinful = provenance.get("spinful")
    if not isinstance(spinful, bool):
        raise ValueError(
            f"reviewed table {name!r} provenance.spinful must be "
            f"a bool, got {spinful!r}"
        )

    # subspace_group_candidate: non-empty string matching table top-level.
    prov_sgc = provenance.get("subspace_group_candidate")
    if not isinstance(prov_sgc, str) or not prov_sgc.strip():
        raise ValueError(
            f"reviewed table {name!r} provenance.subspace_group_candidate "
            f"must be a non-empty string"
        )
    table_sgc = table.get("subspace_group_candidate")
    if prov_sgc != table_sgc:
        raise ValueError(
            f"reviewed table {name!r} provenance.subspace_group_candidate "
            f"({prov_sgc!r}) must match table top-level "
            f"subspace_group_candidate ({table_sgc!r})"
        )

    # expected_hsps: non-empty list of non-empty strings matching table.
    prov_hsps = provenance.get("expected_hsps")
    if not isinstance(prov_hsps, list) or not prov_hsps:
        raise ValueError(
            f"reviewed table {name!r} provenance.expected_hsps must be "
            f"a non-empty list"
        )
    for i, h in enumerate(prov_hsps):
        if not isinstance(h, str) or not h.strip():
            raise ValueError(
                f"reviewed table {name!r} provenance.expected_hsps[{i}] "
                f"must be a non-empty string"
            )
    table_hsps = table.get("expected_hsps")
    if prov_hsps != table_hsps:
        raise ValueError(
            f"reviewed table {name!r} provenance.expected_hsps "
            f"({prov_hsps!r}) must match table top-level "
            f"expected_hsps ({table_hsps!r})"
        )

    # central_sign_convention: non-empty string.
    csc = provenance.get("central_sign_convention")
    if not isinstance(csc, str) or not csc.strip():
        raise ValueError(
            f"reviewed table {name!r} provenance.central_sign_convention "
            f"must be a non-empty string"
        )


def _validate_basis_map_payload(payload: object, entry: dict, name: str) -> None:
    """Validate a reviewed package-data basis-map payload."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"reviewed basis map {name!r} must be a JSON object (dict)"
        )
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError(
            f"reviewed basis map {name!r} schema_version must be non-empty"
        )
    table_name = payload.get("table_name")
    if table_name != entry.get("table_name"):
        raise ValueError(
            f"reviewed basis map {name!r} table_name ({table_name!r}) must "
            f"match manifest table_name ({entry.get('table_name')!r})"
        )
    for key in sorted(_REQUIRED_MANIFEST_REVIEW_KEYS):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"reviewed basis map {name!r} must have a non-empty "
                f"'{key}' string"
            )
    if payload["review_status"] != "reviewed":
        raise ValueError(
            f"reviewed basis map {name!r} review_status must be 'reviewed', "
            f"got {payload['review_status']!r}"
        )
    subspace_group = payload.get("subspace_group_candidate")
    if not isinstance(subspace_group, str) or not subspace_group:
        raise ValueError(
            f"reviewed basis map {name!r} subspace_group_candidate must be "
            "a non-empty string"
        )
    sg_number = payload.get("source_space_group_number")
    if isinstance(sg_number, bool) or not isinstance(sg_number, int):
        raise ValueError(
            f"reviewed basis map {name!r} source_space_group_number must be "
            f"an integer, got {sg_number!r}"
        )
    spinful = payload.get("spinful")
    if not isinstance(spinful, bool):
        raise ValueError(
            f"reviewed basis map {name!r} spinful must be a bool, "
            f"got {spinful!r}"
        )
    basis_map = payload.get("basis_map")
    if not isinstance(basis_map, dict) or not basis_map:
        raise ValueError(
            f"reviewed basis map {name!r} basis_map must be a non-empty dict"
        )
    for key, value in basis_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"reviewed basis map {name!r} basis_map keys must be "
                "non-empty strings"
            )
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"reviewed basis map {name!r} basis_map[{key!r}] must be "
                "a non-empty string"
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
