"""Canonical identities for reduced-EBR promotion inputs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


_IDENTITY_SCHEMA_VERSION = "1.0.0"
_IDENTITY_ALGORITHM = "sha256"


def build_promotion_input_identity(bundle: dict[str, Any]) -> dict[str, str]:
    """Return a deterministic identity for the exact bundle being promoted."""
    payload = {
        key: value
        for key, value in bundle.items()
        if key != "promotion_provenance"
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": _IDENTITY_SCHEMA_VERSION,
        "algorithm": _IDENTITY_ALGORITHM,
        "digest": hashlib.sha256(encoded).hexdigest(),
    }


def merge_table_input_provenance(
    table_provenance: dict[str, Any],
    reduced_ebr_input: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge physical table provenance with its current input source."""
    merged = (
        deepcopy(reduced_ebr_input)
        if isinstance(reduced_ebr_input, dict)
        else {}
    )
    merged.update(deepcopy(table_provenance))
    return merged


def normalize_operation_key(value: object) -> int | None:
    """Normalize an opaque operation ID key; reject aliases such as 0 and "0"."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = int(value)
    except ValueError:
        return None
    if str(normalized) != value:
        return None
    return normalized


def table_input_for_bundle(
    reduced_ebr_input: dict[str, Any] | None,
    bundle_id: str,
) -> dict[str, Any] | None:
    """Resolve per-bundle input provenance from an aggregate mapping."""
    if not isinstance(reduced_ebr_input, dict):
        return None
    per_bundle = reduced_ebr_input.get("table_input_provenance_by_bundle")
    if isinstance(per_bundle, dict):
        resolved = per_bundle.get(bundle_id)
        if isinstance(resolved, dict):
            return resolved
    return reduced_ebr_input
