"""Package-data catalog for valley-preserving irrep phase tables.

Provides validated access to spinful C3 and C2 one-dimensional irrep
phase tables shipped with ValleyScope.  These are irrep matching data,
not reduced EBR tables.
"""

from __future__ import annotations

import json
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
    manifest = load_manifest()
    for entry in manifest.get("tables", []):
        if entry.get("name") == name:
            filename = entry.get("filename")
            if not filename:
                raise ValueError(f"manifest entry for {name!r} missing filename")
            table_path = package_data_root() / filename
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
    return json.loads(path.read_text(encoding="utf-8"))


def get_irrep_phase_list(name: str) -> list[dict]:
    """Return the irrep phase list from a validated phase table."""
    table = load_valley_irrep_phase_table(name)
    return list(table.get("irreps", []))


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


def _validate_phase_table(raw: dict, name: str) -> None:
    """Validate a valley-irrep phase table structure."""
    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"phase table {name!r} missing keys: {sorted(missing)}")

    if raw.get("spinful") is not True:
        raise ValueError(f"phase table {name!r}: spinful must be true")

    order = raw.get("operation_order")
    if not isinstance(order, int) or order < 2:
        raise ValueError(f"phase table {name!r}: operation_order must be int >= 2")

    candidates = raw.get("subspace_group_candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"phase table {name!r}: subspace_group_candidates must be non-empty list")
    expected = _ORDER_TO_CANDIDATES.get(order, set())
    if expected and not set(candidates) <= expected:
        raise ValueError(
            f"phase table {name!r}: subspace_group_candidates {candidates} "
            f"not subset of expected {sorted(expected)} for order {order}"
        )

    irreps = raw.get("irreps", [])
    if not isinstance(irreps, list) or not irreps:
        raise ValueError(f"phase table {name!r}: irreps must be a non-empty list")

    labels = []
    for entry in irreps:
        if not isinstance(entry, dict):
            raise ValueError(f"phase table {name!r}: each irrep must be a dict")
        label = entry.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"phase table {name!r}: each irrep must have a non-empty label")
        labels.append(label)
        phases = entry.get("phases", [])
        if not isinstance(phases, list) or not phases:
            raise ValueError(f"phase table {name!r}: irrep {label!r} must have non-empty phases")
        if len(phases) != 1:
            raise ValueError(
                f"phase table {name!r}: irrep {label!r} must be one-dimensional "
                f"(single phase), got {len(phases)} phases"
            )
        if not all(isinstance(p, (int, float)) for p in phases):
            raise ValueError(f"phase table {name!r}: irrep {label!r} phases must be numeric")
        # Verify phases are canonicalizable: each phase in (-0.5, 0.5].
        for p in phases:
            cp = float(p) % 1.0
            if cp > 0.5:
                cp -= 1.0
            if cp <= -0.5:
                cp += 1.0
            if abs(cp - float(p)) > 1e-10 and abs(cp % 1.0 - float(p) % 1.0) > 1e-10:
                # Allow phases already in canonical range.
                if not (-0.5 < float(p) <= 0.5 or abs(float(p) - 0.5) < 1e-12 or abs(float(p) + 0.5) < 1e-12):
                    raise ValueError(
                        f"phase table {name!r}: irrep {label!r} phase {p} "
                        f"not in canonical range (-0.5, 0.5]"
                    )

    if len(set(labels)) != len(labels):
        raise ValueError(f"phase table {name!r}: irrep labels must be unique")
