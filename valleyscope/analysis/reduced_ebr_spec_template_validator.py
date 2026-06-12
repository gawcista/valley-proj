"""Mapping spec template builder and preflight validator.

Authoring aids that turn ``inspect-ebr-source`` output into a
non-buildable mapping-spec template, and validate completed specs
against the source basis before table construction.

Does NOT infer HSPs, ValleyScope irrep keys, or reduced EBR decomposition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_SCHEMA_VERSION = "1.0.0"
_DATA_SOURCE = "irreptables"
_PLACEHOLDER = "REQUIRED_FILL_BY_HUMAN"


# ---------------------------------------------------------------------------
# Template builder
# ---------------------------------------------------------------------------

def build_mapping_spec_template(source_basis_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a non-buildable mapping-spec template from a source-basis payload.

    Every source label gets placeholder mapping entries that will cause
    ``build_reduced_table_from_spec_file`` to fail with clear errors until
    a human replaces them.
    """
    _validate_source_basis_payload(source_basis_payload)

    sg = source_basis_payload["space_group_number"]
    spinful = source_basis_payload["spinful"]
    source_basis = list(source_basis_payload["source_basis"])

    source_hsp: dict[str, str] = {}
    valleyscope_key: dict[str, str] = {}
    for entry in source_basis:
        label = entry["source_label"]
        source_hsp[label] = _PLACEHOLDER
        valleyscope_key[label] = _PLACEHOLDER

    expected_hsps = [_PLACEHOLDER]
    allowed_irrep_keys = [_PLACEHOLDER]

    return {
        "schema_version": _SCHEMA_VERSION,
        "data_source": _DATA_SOURCE,
        "space_group_number": int(sg),
        "spinful": bool(spinful),
        "source_hsp_by_irrep": source_hsp,
        "valleyscope_key_by_source_irrep": valleyscope_key,
        "expected_hsps": expected_hsps,
        "allowed_irrep_keys": allowed_irrep_keys,
        "subspace_group_candidate": _PLACEHOLDER,
    }


# ---------------------------------------------------------------------------
# Preflight validator
# ---------------------------------------------------------------------------

def validate_mapping_spec_against_source_basis(
    spec: Mapping[str, Any],
    source_basis_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a completed mapping spec against a source-basis payload.

    Returns a dict with ``valid`` (bool), ``errors`` (list[str]), and
    ``summary`` (str).  The spec must match the source basis exactly;
    no labels may be missing, extra, or inferred.
    """
    errors: list[str] = []
    _validate_source_basis_payload(source_basis_payload)

    # --- Canonical fields ---
    for key in [
        "schema_version", "data_source", "space_group_number", "spinful",
        "source_hsp_by_irrep", "valleyscope_key_by_source_irrep",
        "expected_hsps", "allowed_irrep_keys", "subspace_group_candidate",
    ]:
        if key not in spec:
            errors.append(f"missing required field: {key!r}")
    if errors:
        return _result(False, errors)

    if spec.get("schema_version") != _SCHEMA_VERSION:
        errors.append(f"schema_version must be {_SCHEMA_VERSION!r}")
    if spec.get("data_source") != _DATA_SOURCE:
        errors.append(f"data_source must be {_DATA_SOURCE!r}")
    if spec.get("space_group_number") != source_basis_payload["space_group_number"]:
        errors.append("space_group_number mismatch")
    if not isinstance(spec.get("spinful"), bool):
        errors.append("spinful must be a boolean")

    # --- Check source label coverage ---
    source_labels = {e["source_label"] for e in source_basis_payload["source_basis"]}
    hsp_map = spec.get("source_hsp_by_irrep", {})
    key_map = spec.get("valleyscope_key_by_source_irrep", {})
    spec_labels = set(hsp_map.keys())
    if spec_labels != set(key_map.keys()):
        errors.append(
            "source_hsp_by_irrep and valleyscope_key_by_source_irrep "
            "must have identical key sets"
        )

    if spec_labels != source_labels:
        missing = sorted(source_labels - spec_labels)
        extra = sorted(spec_labels - source_labels)
        missing_names = sorted(source_labels - {e["source_label"] for e in source_basis_payload["source_basis"]})
        if missing:
            errors.append(f"missing source labels in spec: {missing}")
        if extra:
            errors.append(f"extra labels in spec not in source basis: {extra}")

    # --- Check mapping values are filled ---
    for label, hsp in sorted(hsp_map.items()):
        if hsp == _PLACEHOLDER or not isinstance(hsp, str) or not hsp:
            errors.append(f"source_hsp_by_irrep[{label!r}] must be a non-empty string")
        key = key_map.get(label)
        if key == _PLACEHOLDER or not isinstance(key, str) or not key:
            errors.append(
                f"valleyscope_key_by_source_irrep[{label!r}] must be a non-empty string"
            )

    # --- Check expected_hsps and allowed_irrep_keys are filled ---
    for field in ["expected_hsps", "allowed_irrep_keys"]:
        value = spec.get(field, [])
        if not isinstance(value, list) or not value:
            errors.append(f"{field} must be a non-empty list")
        elif any(v == _PLACEHOLDER for v in value):
            errors.append(f"{field} must not contain placeholder values")
        elif not all(isinstance(v, str) and v for v in value):
            errors.append(f"{field} entries must be non-empty strings")

    scg = spec.get("subspace_group_candidate")
    if scg == _PLACEHOLDER or not isinstance(scg, str) or not scg:
        errors.append("subspace_group_candidate must be a non-empty string")

    return _result(len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _validate_source_basis_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("source_basis_payload must be a mapping")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"source basis schema_version must be {_SCHEMA_VERSION!r}")
    source_basis = payload.get("source_basis")
    if not isinstance(source_basis, Sequence) or isinstance(source_basis, (str, bytes)):
        raise ValueError("source_basis_payload must contain a 'source_basis' list")
    if not source_basis:
        raise ValueError("source_basis_payload source_basis must be non-empty")


def _result(valid: bool, errors: list[str]) -> dict[str, Any]:
    return {
        "valid": valid,
        "errors": errors,
        "summary": "spec is valid" if valid else f"{len(errors)} validation error(s)",
    }
