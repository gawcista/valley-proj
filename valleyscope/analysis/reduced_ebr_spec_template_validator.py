"""Mapping spec template builder and preflight validator.

Authoring aids that turn ``inspect-ebr-source`` output into a
non-buildable mapping-spec template, and validate completed specs
against the source basis before table construction.

Does NOT infer HSPs, ValleyScope irrep keys, or reduced EBR decomposition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_SCHEMA_VERSION_V1 = "1.0.0"
_SCHEMA_VERSION_V1_1 = "1.1.0"
_SUPPORTED_SCHEMA_VERSIONS = frozenset([_SCHEMA_VERSION_V1, _SCHEMA_VERSION_V1_1])
_DATA_SOURCE = "irreptables"
_PLACEHOLDER = "REQUIRED_FILL_BY_HUMAN"


# ---------------------------------------------------------------------------
# Template builder
# ---------------------------------------------------------------------------

def build_mapping_spec_template(
    source_basis_payload: Mapping[str, Any],
    schema_version: str = _SCHEMA_VERSION_V1,
) -> dict[str, Any]:
    """Build a non-buildable mapping-spec template from a source-basis payload.

    Every source label gets placeholder mapping entries that will cause
    ``build_reduced_table_from_spec_file`` to fail with clear errors until
    a human replaces them.

    Parameters
    ----------
    source_basis_payload : Mapping
        Source basis payload from ``inspect-ebr-source``.
    schema_version : str
        ``"1.0.0"`` for legacy one-to-one specs, ``"1.1.0"`` for
        multiplicity-aware specs.  Default ``"1.0.0"``.
    """
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version {schema_version!r}; "
            f"supported: {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )
    _validate_source_basis_payload(source_basis_payload)

    sg = source_basis_payload["space_group_number"]
    spinful = source_basis_payload["spinful"]
    source_basis = _source_basis_entries(source_basis_payload)

    source_hsp: dict[str, str] = {}
    for entry in source_basis:
        label = entry["source_label"]
        source_hsp[label] = _PLACEHOLDER

    expected_hsps = [_PLACEHOLDER]
    allowed_irrep_keys = [_PLACEHOLDER]

    result: dict[str, Any] = {
        "data_source": _DATA_SOURCE,
        "space_group_number": int(sg),
        "spinful": bool(spinful),
        "source_hsp_by_irrep": source_hsp,
        "expected_hsps": expected_hsps,
        "allowed_irrep_keys": allowed_irrep_keys,
        "subspace_group_candidate": _PLACEHOLDER,
    }

    if schema_version == _SCHEMA_VERSION_V1_1:
        result["schema_version"] = _SCHEMA_VERSION_V1_1
        mult_map: dict[str, dict[str, int]] = {}
        for entry in source_basis:
            label = entry["source_label"]
            mult_map[label] = {_PLACEHOLDER: -1}
        result["valleyscope_irrep_multiplicity_by_source_irrep"] = mult_map
    else:
        result["schema_version"] = _SCHEMA_VERSION_V1
        valleyscope_key: dict[str, str] = {}
        for entry in source_basis:
            label = entry["source_label"]
            valleyscope_key[label] = _PLACEHOLDER
        result["valleyscope_key_by_source_irrep"] = valleyscope_key

    return result


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
    if not isinstance(spec, Mapping):
        return _result(False, ["spec must be a JSON object / mapping"])

    # --- Identify schema version ---
    sv = spec.get("schema_version")
    if sv not in _SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            f"Unsupported schema_version {sv!r}; "
            f"supported: {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )
        return _result(False, errors)

    is_v1_1 = sv == _SCHEMA_VERSION_V1_1

    # --- Canonical fields ---
    required = [
        "schema_version", "data_source", "space_group_number", "spinful",
        "source_hsp_by_irrep", "expected_hsps", "allowed_irrep_keys",
        "subspace_group_candidate",
    ]
    if is_v1_1:
        required.append("valleyscope_irrep_multiplicity_by_source_irrep")
    else:
        required.append("valleyscope_key_by_source_irrep")
    for key in required:
        if key not in spec:
            errors.append(f"missing required field: {key!r}")
    if errors:
        return _result(False, errors)

    if spec.get("data_source") != _DATA_SOURCE:
        errors.append(f"data_source must be {_DATA_SOURCE!r}")
    if spec.get("space_group_number") != source_basis_payload["space_group_number"]:
        errors.append("space_group_number mismatch")
    if not isinstance(spec.get("spinful"), bool):
        errors.append("spinful must be a boolean")

    # --- Check source label coverage ---
    source_labels = {e["source_label"] for e in _source_basis_entries(source_basis_payload)}
    hsp_map_raw = spec.get("source_hsp_by_irrep", {})
    if not isinstance(hsp_map_raw, Mapping):
        errors.append("source_hsp_by_irrep must be a mapping")
        return _result(False, errors)
    hsp_map = dict(hsp_map_raw)
    spec_labels = set(hsp_map.keys())

    if spec_labels != source_labels:
        missing = sorted(source_labels - spec_labels)
        extra = sorted(spec_labels - source_labels)
        if missing:
            errors.append(f"missing source labels in spec: {missing}")
        if extra:
            errors.append(f"extra labels in spec not in source basis: {extra}")

    # --- Validate HSP mappings ---
    for label, hsp in sorted(hsp_map.items()):
        if hsp == _PLACEHOLDER or not isinstance(hsp, str) or not hsp:
            errors.append(f"source_hsp_by_irrep[{label!r}] must be a non-empty string")

    # --- Parse expected_hsps and allowed_irrep_keys ---
    expected_hsps_set: set[str] = set()
    allowed_irrep_key_set: set[str] = set()
    for field in ["expected_hsps", "allowed_irrep_keys"]:
        value = spec.get(field, [])
        if not isinstance(value, list) or not value:
            errors.append(f"{field} must be a non-empty list")
        elif any(v == _PLACEHOLDER for v in value):
            errors.append(f"{field} must not contain placeholder values")
        elif not all(isinstance(v, str) and v for v in value):
            errors.append(f"{field} entries must be non-empty strings")
        elif field == "expected_hsps":
            expected_hsps_set = set(value)
        elif field == "allowed_irrep_keys":
            allowed_irrep_key_set = set(value)

    if is_v1_1:
        _validate_v1_1_multiplicities(
            spec, errors, source_labels, hsp_map,
            expected_hsps_set, allowed_irrep_key_set,
        )
    else:
        _validate_v1_0_key_map(
            spec, errors, hsp_map,
            expected_hsps_set, allowed_irrep_key_set,
        )

    scg = spec.get("subspace_group_candidate")
    if scg == _PLACEHOLDER or not isinstance(scg, str) or not scg:
        errors.append("subspace_group_candidate must be a non-empty string")

    return _result(len(errors) == 0, errors)


def _validate_v1_0_key_map(
    spec: Mapping[str, Any],
    errors: list[str],
    hsp_map: dict[str, str],
    expected_hsps_set: set[str],
    allowed_irrep_key_set: set[str],
) -> None:
    key_map = spec.get("valleyscope_key_by_source_irrep", {})
    if not isinstance(key_map, Mapping):
        errors.append("valleyscope_key_by_source_irrep must be a mapping")
        return
    spec_labels = set(hsp_map.keys())
    if spec_labels != set(key_map.keys()):
        errors.append(
            "source_hsp_by_irrep and valleyscope_key_by_source_irrep "
            "must have identical key sets"
        )
    for label in sorted(hsp_map):
        key = key_map.get(label)
        if key == _PLACEHOLDER or not isinstance(key, str) or not key:
            errors.append(
                f"valleyscope_key_by_source_irrep[{label!r}] "
                "must be a non-empty string"
            )
    if expected_hsps_set and allowed_irrep_key_set:
        for label, hsp in sorted(hsp_map.items()):
            key = key_map.get(label)
            if hsp in expected_hsps_set and key not in allowed_irrep_key_set:
                errors.append(
                    f"valleyscope_key_by_source_irrep[{label!r}] must be in "
                    "allowed_irrep_keys because its HSP is in expected_hsps"
                )
            if key in allowed_irrep_key_set and hsp not in expected_hsps_set:
                errors.append(
                    f"source_hsp_by_irrep[{label!r}] must be in expected_hsps "
                    "because its irrep key is in allowed_irrep_keys"
                )


def _validate_v1_1_multiplicities(
    spec: Mapping[str, Any],
    errors: list[str],
    source_labels: set[str],
    hsp_map: dict[str, str],
    expected_hsps_set: set[str],
    allowed_irrep_key_set: set[str],
) -> None:
    mult = spec.get("valleyscope_irrep_multiplicity_by_source_irrep", {})
    if not isinstance(mult, Mapping):
        errors.append(
            "valleyscope_irrep_multiplicity_by_source_irrep must be a mapping"
        )
        return
    mult_keys = set(mult.keys())
    if mult_keys - source_labels:
        extra = sorted(mult_keys - source_labels)
        errors.append(
            f"extra labels in valleyscope_irrep_multiplicity_by_source_irrep "
            f"not in source basis: {extra}"
        )

    for label in sorted(source_labels):
        hsp = hsp_map.get(label, "")
        in_sampled = hsp in expected_hsps_set
        has_mult = label in mult

        if in_sampled and not has_mult:
            errors.append(
                f"missing valleyscope_irrep_multiplicity_by_source_irrep "
                f"entry for sampled-HSP label {label!r} (HSP={hsp})"
            )
            continue
        if not in_sampled and has_mult:
            errors.append(
                f"valleyscope_irrep_multiplicity_by_source_irrep entry for "
                f"non-sampled-HSP label {label!r} (HSP={hsp}) — must be "
                f"omitted for labels outside expected_hsps"
            )
            continue
        if not in_sampled:
            continue

        sub = mult[label]
        if not isinstance(sub, Mapping) or not sub:
            errors.append(
                f"valleyscope_irrep_multiplicity_by_source_irrep[{label!r}] "
                f"must be a non-empty dict"
            )
            continue
        for key, val in sorted(sub.items()):
            if key == _PLACEHOLDER:
                errors.append(
                    f"valleyscope_irrep_multiplicity_by_source_irrep"
                    f"[{label!r}] contains placeholder key {_PLACEHOLDER!r}"
                )
            elif not isinstance(key, str) or not key:
                errors.append(
                    f"valleyscope_irrep_multiplicity_by_source_irrep"
                    f"[{label!r}] keys must be non-empty strings"
                )
            elif allowed_irrep_key_set and key not in allowed_irrep_key_set:
                errors.append(
                    f"valleyscope_irrep_multiplicity_by_source_irrep"
                    f"[{label!r}][{key!r}] not in allowed_irrep_keys"
                )
            if not isinstance(val, int) or isinstance(val, bool):
                errors.append(
                    f"valleyscope_irrep_multiplicity_by_source_irrep"
                    f"[{label!r}][{key!r}] must be an integer, got {val!r}"
                )
            elif val <= 0:
                errors.append(
                    f"valleyscope_irrep_multiplicity_by_source_irrep"
                    f"[{label!r}][{key!r}] must be a positive integer"
                )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _validate_source_basis_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("source_basis_payload must be a mapping")
    if payload.get("schema_version") != _SCHEMA_VERSION_V1:
        raise ValueError(f"source basis schema_version must be {_SCHEMA_VERSION_V1!r}")
    if payload.get("data_source") != _DATA_SOURCE:
        raise ValueError(f"source basis data_source must be {_DATA_SOURCE!r}")
    space_group_number = payload.get("space_group_number")
    if not isinstance(space_group_number, int) or isinstance(space_group_number, bool):
        raise ValueError("source basis space_group_number must be an integer")
    if not isinstance(payload.get("spinful"), bool):
        raise ValueError("source basis spinful must be a boolean")
    source_basis = payload.get("source_basis")
    if not isinstance(source_basis, Sequence) or isinstance(source_basis, (str, bytes)):
        raise ValueError("source_basis_payload must contain a 'source_basis' list")
    if not source_basis:
        raise ValueError("source_basis_payload source_basis must be non-empty")
    _source_basis_entries(payload)


def _source_basis_entries(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries_raw = payload.get("source_basis")
    if not isinstance(entries_raw, Sequence) or isinstance(entries_raw, (str, bytes)):
        raise ValueError("source_basis_payload must contain a 'source_basis' list")
    entries: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for idx, entry in enumerate(entries_raw):
        if not isinstance(entry, Mapping):
            raise ValueError(f"source_basis[{idx}] must be a mapping")
        label = entry.get("source_label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"source_basis[{idx}].source_label must be a non-empty string")
        if label in seen:
            raise ValueError(f"duplicate source_label in source_basis: {label!r}")
        seen.add(label)
        degeneracy = entry.get("degeneracy")
        if not isinstance(degeneracy, int) or isinstance(degeneracy, bool) or degeneracy <= 0:
            raise ValueError(
                f"source_basis[{idx}].degeneracy must be a positive integer"
            )
        entries.append(entry)
    return entries


def _result(valid: bool, errors: list[str]) -> dict[str, Any]:
    return {
        "valid": valid,
        "errors": errors,
        "summary": "spec is valid" if valid else f"{len(errors)} validation error(s)",
    }
