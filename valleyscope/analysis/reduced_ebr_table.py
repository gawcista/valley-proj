"""Reduced EBR JSON table loading and structural validation.

Loading preserves the complete payload; it does not establish reviewed
provenance or physical readiness. Promotion validates those independently.
This data-format boundary depends only on the Python standard library.
"""

from __future__ import annotations

import json
from pathlib import Path

_REQUIRED_TABLE_KEYS = {"schema_version", "subspace_group_candidate",
                         "expected_hsps", "irreps", "ebrs"}


def load_reduced_ebr_table(path: str | Path) -> dict:
    """Load and validate a reduced EBR table from JSON.

    Raises ValueError for missing keys, empty/malformed fields,
    non-unique labels, mismatched vector lengths, non-integer/nonnegative
    vector entries, or undocumented irrep key formats.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_reduced_ebr_table(raw)


def validate_reduced_ebr_table(raw: object) -> dict:
    """Validate table structure without coercing or mutating its payload.

    Return the original dict, including extra provenance fields. This does
    not establish reviewed provenance, physical compatibility or readiness.
    Boolean vector entries are not integer multiplicities. Malformed input
    raises ValueError, identically for JSON and direct-dict callers.
    """
    if not isinstance(raw, dict):
        raise ValueError("reduced EBR table must be a mapping")
    missing = _REQUIRED_TABLE_KEYS - set(raw)
    if missing:
        raise ValueError(f"reduced EBR table missing keys: {sorted(missing)}")

    # schema_version
    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("table schema_version must be a non-empty string")

    # subspace_group_candidate
    if not isinstance(raw.get("subspace_group_candidate"), str) or not raw["subspace_group_candidate"]:
        raise ValueError("table subspace_group_candidate must be a non-empty string")

    # expected_hsps
    expected_hsps = raw["expected_hsps"]
    if not isinstance(expected_hsps, list):
        raise ValueError("table expected_hsps must be a list")
    if not expected_hsps:
        raise ValueError("table expected_hsps must be a non-empty list")
    if not all(isinstance(h, str) and h for h in expected_hsps):
        raise ValueError("table expected_hsps entries must be non-empty strings")
    if len(set(expected_hsps)) != len(expected_hsps):
        raise ValueError("table expected_hsps must contain unique entries")

    # irreps
    irreps = raw["irreps"]
    if not isinstance(irreps, list) or not irreps:
        raise ValueError("table irreps must be a non-empty list")
    if not all(isinstance(label, str) and label for label in irreps):
        raise ValueError("table irreps must be non-empty strings")
    if len(set(irreps)) != len(irreps):
        raise ValueError("table irreps must be unique")
    _validate_irrep_key_format(irreps)

    # ebrs
    ebrs = raw["ebrs"]
    if not isinstance(ebrs, list) or not ebrs:
        raise ValueError("table ebrs must be a non-empty list")

    n_irreps = len(irreps)
    ebr_labels = []
    for ebr in ebrs:
        if not isinstance(ebr, dict):
            raise ValueError("each EBR entry must be a mapping")
        label = ebr.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("each EBR must define a non-empty label")
        ebr_labels.append(label)
        vector = ebr.get("vector")
        if not isinstance(vector, list):
            raise ValueError(f"EBR '{label}' missing vector")
        if not vector:
            raise ValueError(f"EBR '{label}' vector must be non-empty")
        if len(vector) != n_irreps:
            raise ValueError(
                f"EBR '{label}' vector length {len(vector)} "
                f"!= irrep count {n_irreps}"
            )
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in vector):
            raise ValueError(
                f"EBR '{label}' vector must be nonnegative integers"
            )
        if not any(v > 0 for v in vector):
            raise ValueError(
                f"EBR '{label}' vector must have at least one positive entry"
            )
    if len(set(ebr_labels)) != len(ebr_labels):
        raise ValueError("table EBR labels must be unique")

    return raw


_IRREP_KEY_RE = (
    r"\A"                         # start
    r"[A-Za-z][A-Za-z0-9_]*"      # kpoint label
    r":"                          # separator
    r"-?"                         # optional leading minus (spinor convention)
    r"[A-Za-z]"                   # irrep label must start with a letter
    r"[A-Za-z0-9_+/\-]*"         # rest of irrep label
    r"(?::op\d+)?"                # optional operation suffix
    r"\Z"                         # end
)


def _validate_irrep_key_format(irreps: list[str]) -> None:
    """Validate irrep keys match the documented format.

    Format: ``<kpoint>:<irrep_label>`` with optional ``:op<N>`` suffix.
    Example valid keys:
      ``GammaM:C3_spinor_phase_+1/2``
      ``KM:C3_spinor_phase_-1/6``
      ``GammaM:C3_spinor_phase_+1/2:op1``
    """
    import re
    pattern = re.compile(_IRREP_KEY_RE)
    for label in irreps:
        if not pattern.match(label):
            raise ValueError(
                f"invalid irrep key format: {label!r}. "
                f"Expected format: <kpoint>:<irrep_label> "
                f"with optional :op<N> suffix"
            )
