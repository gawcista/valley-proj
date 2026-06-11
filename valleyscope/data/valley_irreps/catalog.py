"""Package-data catalog for valley-preserving irrep phase tables.

Provides validated access to spinful C3 and C2 one-dimensional irrep
phase tables shipped with ValleyScope.  These are irrep matching data,
not reduced EBR tables.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def package_data_root() -> Path:
    """Return the absolute path to the valley_irreps package-data directory."""
    return Path(__file__).resolve().parent


def load_valley_irrep_phase_table(name: str) -> dict:
    """Load and validate a valley-irrep phase table by its manifest name.

    Parameters
    ----------
    name : str
        One of ``"spinful_C3_phase_v1"`` or ``"spinful_C2_phase_v1"``.

    Returns
    -------
    dict
        Validated phase table with keys ``schema_version``, ``name``,
        ``spinful``, ``operation_order``, ``subspace_group_candidates``,
        ``phase_convention``, and ``irreps``.

    Raises
    ------
    ValueError
        If the table name is unknown or validation fails.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("phase table name must be a non-empty string")

    manifest = load_manifest()
    for entry in manifest.get("tables", []):
        if entry.get("name") == name:
            filename = entry.get("filename")
            table_path = _resolve_table_path(str(filename))
            if not table_path.is_file():
                raise FileNotFoundError(f"phase table not found: {table_path}")
            raw = json.loads(table_path.read_text(encoding="utf-8"))
            _validate_phase_table(raw, name)
            return raw
    available = [e.get("name", "?") for e in manifest.get("tables", [])]
    raise ValueError(
        f"no valley-irrep phase table named {name!r}. "
        f"Available: {available}"
    )


def load_manifest() -> dict:
    """Load the valley_irreps catalog manifest."""
    path = package_data_root() / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    return manifest


def get_irrep_phase_list(name: str) -> list[dict]:
    """Return the irrep phase list from a validated phase table."""
    table = load_valley_irrep_phase_table(name)
    return [
        {"label": str(entry["label"]), "phases": list(entry["phases"])}
        for entry in table.get("irreps", [])
    ]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "schema_version", "name", "spinful", "operation_order",
    "subspace_group_candidates", "phase_convention", "irreps",
}

_ORDER_TO_CANDIDATES = {
    3: {"C3_like", "P3"},
    2: {"C2_like", "P2"},
}

_FORBIDDEN_EBR_KEYS = {
    "ebrs",
    "ebr_decomposition",
    "compatibility_relations",
    "vector",
}


def _validate_manifest(manifest: object) -> None:
    """Validate the valley-irrep phase table manifest."""
    if not isinstance(manifest, dict):
        raise ValueError("valley-irrep manifest must be a JSON object")

    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("manifest schema_version must be a non-empty string")

    tables = manifest.get("tables")
    if not isinstance(tables, list):
        raise ValueError("manifest 'tables' must be a list")

    seen_names: set[str] = set()
    for i, entry in enumerate(tables):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest tables[{i}] must be a JSON object")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"manifest tables[{i}] must have a non-empty name")
        if name in seen_names:
            raise ValueError(f"manifest tables[{i}] duplicate name {name!r}")
        seen_names.add(name)

        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"manifest tables[{i}] ({name!r}) missing filename")
        _resolve_table_path(filename)


def _validate_phase_table(raw: object, name: str) -> None:
    """Validate a valley-irrep phase table structure."""
    if not isinstance(raw, dict):
        raise ValueError(f"phase table {name!r} must be a JSON object")

    _reject_ebr_payload(raw, name)

    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"phase table {name!r} missing keys: {sorted(missing)}")

    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError(
            f"phase table {name!r}: schema_version must be non-empty string"
        )

    if raw.get("name") != name:
        raise ValueError(
            f"phase table {name!r}: table name {raw.get('name')!r} "
            "does not match manifest"
        )

    if raw.get("spinful") is not True:
        raise ValueError(f"phase table {name!r}: spinful must be true")

    order = raw.get("operation_order")
    if not isinstance(order, int) or order < 2:
        raise ValueError(f"phase table {name!r}: operation_order must be int >= 2")
    if order not in _ORDER_TO_CANDIDATES:
        raise ValueError(f"phase table {name!r}: unsupported operation_order={order}")

    phase_convention = raw.get("phase_convention")
    if not isinstance(phase_convention, str) or not phase_convention:
        raise ValueError(
            f"phase table {name!r}: phase_convention must be non-empty string"
        )

    candidates = raw.get("subspace_group_candidates", [])
    if (
        not isinstance(candidates, list)
        or not candidates
        or not all(isinstance(c, str) and c for c in candidates)
    ):
        raise ValueError(
            f"phase table {name!r}: subspace_group_candidates must be "
            "non-empty strings"
        )
    if len(set(candidates)) != len(candidates):
        raise ValueError(
            f"phase table {name!r}: subspace_group_candidates must be unique"
        )
    expected = _ORDER_TO_CANDIDATES[order]
    if set(candidates) != expected:
        raise ValueError(
            f"phase table {name!r}: subspace_group_candidates {candidates} "
            f"must equal expected {sorted(expected)} for order {order}"
        )

    irreps = raw.get("irreps", [])
    if not isinstance(irreps, list) or not irreps:
        raise ValueError(f"phase table {name!r}: irreps must be a non-empty list")

    labels = []
    for entry in irreps:
        if not isinstance(entry, dict):
            raise ValueError(f"phase table {name!r}: each irrep must be a dict")
        _reject_ebr_payload(entry, name)
        label = entry.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(
                f"phase table {name!r}: each irrep must have a non-empty label"
            )
        labels.append(label)
        phases = entry.get("phases", [])
        if not isinstance(phases, list) or not phases:
            raise ValueError(
                f"phase table {name!r}: irrep {label!r} must have "
                "non-empty phases"
            )
        if len(phases) != 1:
            raise ValueError(
                f"phase table {name!r}: irrep {label!r} must be one-dimensional "
                f"(single phase), got {len(phases)} phases"
            )
        if not all(isinstance(p, (int, float)) for p in phases):
            raise ValueError(
                f"phase table {name!r}: irrep {label!r} phases must be numeric"
            )
        for p in phases:
            if isinstance(p, bool) or not math.isfinite(float(p)):
                raise ValueError(
                    f"phase table {name!r}: irrep {label!r} phases must be "
                    "finite numeric"
                )
            value = float(p)
            if not (-0.5 < value <= 0.5):
                raise ValueError(
                    f"phase table {name!r}: irrep {label!r} phase {p} "
                    f"not in canonical range (-0.5, 0.5]"
                )

    if len(set(labels)) != len(labels):
        raise ValueError(f"phase table {name!r}: irrep labels must be unique")


def _reject_ebr_payload(obj: dict, name: str) -> None:
    bad = sorted(k for k in obj if str(k).lower() in _FORBIDDEN_EBR_KEYS)
    if bad:
        raise ValueError(
            f"phase table {name!r}: forbidden EBR/decomposition keys: {bad}"
        )


def _resolve_table_path(filename: str) -> Path:
    """Resolve a manifest filename inside the package-data directory."""
    path = Path(filename)
    if path.is_absolute():
        raise ValueError(f"phase table filename must be relative, got {filename!r}")
    if ".." in path.parts:
        raise ValueError(f"phase table filename must not contain '..', got {filename!r}")
    root = package_data_root().resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"phase table filename {filename!r} resolves outside package-data directory"
        ) from exc
    if resolved.suffix != ".json":
        raise ValueError(f"phase table filename must end in .json, got {filename!r}")
    return resolved
