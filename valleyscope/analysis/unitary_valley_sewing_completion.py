"""Target-side irrep completion from directed unitary sewing evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np

from valleyscope.analysis.generic_irrep_matching import match_restricted_characters
from valleyscope.analysis.scoped_representation_evidence import (
    ScopedEvidenceValidation,
    _run_local_memoized,
    validate_directed_valley_sewing_evidence_record,
    validate_scoped_representation_evidence_record,
)
from valleyscope.io.wavefunction_convention import (
    canonical_identity,
    valid_sha256_identity,
)


SCHEMA_VERSION = "1.0.0"
_CPRIME_KEYS = (
    "spinor_source_basis_certificate_identity",
    "double_space_group_lift_certificate_identity",
    "scoped_representation_evidence_identity",
)


@_run_local_memoized
def build_unitary_valley_sewing_certificate(**raw: object) -> dict[str, object]:
    """Recompute one target irrep from producer-owned directed evidence."""
    reasons: list[str] = []
    source, target = _map(raw.get("source_candidate")), _map(raw.get("target_context"))
    directed = _map(raw.get("directed_scoped_evidence_context"))
    evidence, evidence_raw = directed.get("record"), directed.get("raw_inputs")
    if not isinstance(evidence, Mapping) or not isinstance(evidence_raw, Mapping):
        reasons.append("directed_scoped_evidence_context_missing")
        evidence = {}
    elif validate_directed_valley_sewing_evidence_record(
        evidence, **evidence_raw
    ).status != "passed":
        reasons.append("directed_scoped_evidence_revalidation_failed")
    scope = _map(evidence.get("scope"))
    source_ids = list(source.get("valley_preserving_operation_ids", []))
    target_ids = list(target.get("valley_preserving_operation_ids", []))
    links = (
        source.get("ready_for_ebr_input", True) is True,
        source.get("readiness_level") == "trusted",
        source.get("kpoint") == scope.get("source_kpoint_label"),
        source.get("valley") == scope.get("source_valley"),
        target.get("valley") == scope.get("target_valley"),
        source_ids == scope.get("source_little_group_operation_ids"),
        target_ids == scope.get("target_little_group_operation_ids"),
    )
    if not all(links):
        reasons.append("directed_scope_candidate_link_mismatch")
    _validate_table_links(raw, source, target, reasons)
    _validate_target_source_payload(target, reasons)
    cprime = _source_cprime(raw, source, evidence, reasons)
    transports = {
        side: _transport(raw, side, scope, reasons)
        for side in ("source", "target")
    }

    characters = {}
    for row in _map(evidence.get("directed_group_law")).get("rows", []):
        value = row.get("target_character") if isinstance(row, Mapping) else None
        operation_id = row.get("target_operation_id") if isinstance(row, Mapping) else None
        if (
            isinstance(operation_id, int)
            and isinstance(value, list)
            and len(value) == 2
            and row.get("passed") is True
        ):
            characters[operation_id] = complex(value[0], value[1])
        else:
            reasons.append("directed_intertwining_row_not_passed")
    if set(characters) != set(target_ids):
        reasons.append("target_character_vector_incomplete")
    try:
        matching = match_restricted_characters(
            computed_characters=characters,
            source_irrep_characters=target.get("source_irrep_characters", {}),
            valley_preserving_operation_ids=target_ids,
            source_operation_map=target.get("source_operation_map", {}),
            hsp_little_group_operation_ids=target_ids,
        )
    except (TypeError, ValueError):
        matching = {"matching_status": "blocked", "irrep_multiplicities": {}}
    multiplicities = matching.get("irrep_multiplicities")
    if matching.get("matching_status") != "matched" or not _multiplicities(
        multiplicities
    ):
        reasons.append("target_irrep_matching_not_unique")

    source_certificate = _map(raw.get("source_standard_setting_certificate"))
    target_certificate = _map(raw.get("target_standard_setting_certificate"))
    target_source = {
        "table_number": getattr(raw.get("target_table"), "number", None),
        "table_spinor": bool(getattr(raw.get("target_table"), "spinor", False)),
        "payload_provenance": target.get("source_payload_provenance", {}),
    }
    content = {
        "schema_version": SCHEMA_VERSION,
        "completion_kind": "inferred_by_unitary_valley_sewing",
        "source": {
            "sampled_kpoint": source.get("kpoint"),
            "valley": source.get("valley"),
            "source_hsp_label": _nested(
                source, "irrep_source_provenance", "source_hsp_label"
            ),
            "irrep": source.get("matched_irrep"),
            "multiplicity": source.get("irrep_multiplicity"),
            "candidate_identity": canonical_identity(_json(source)),
            "cprime": cprime,
        },
        "target": {
            "valley": target.get("valley"),
            "source_hsp_label": target.get("source_hsp_label"),
            "frame_label": scope.get("target_kpoint_label"),
            "irrep_multiplicities": dict(multiplicities or {}),
            "matching_status": matching.get("matching_status"),
            "matching_reason": matching.get("reason", ""),
            "reviewed_irrep_source_identity": canonical_identity(_json(target_source)),
        },
        "full_valley_orbit": scope.get("full_valley_orbit", []),
        "hsp_mapping": {
            key: scope.get(key)
            for key in ("source_k_frac", "target_k_frac", "reciprocal_lattice_shift")
        },
        "sewing_operation_id": scope.get("sewing_operation_id"),
        "directed_scoped_evidence_identity": evidence.get("evidence_identity"),
        "source_target_standard_setting_transport": {
            **transports,
            "source_certificate_identity": canonical_identity(_json(source_certificate)),
            "target_certificate_identity": canonical_identity(_json(target_certificate)),
        },
        "target_character_vector": {
            str(key): [float(value.real), float(value.imag)]
            for key, value in sorted(characters.items())
        },
        "grey_group_matching_allowed": False,
    }
    record = deepcopy(content)
    record.update(
        status="passed" if not reasons else "blocked_unknown",
        reason_codes=list(dict.fromkeys(reasons)),
        certificate_identity=canonical_identity(content),
    )
    return record


def validate_unitary_valley_sewing_certificate(
    certificate: Mapping[str, object], **raw: object
) -> ScopedEvidenceValidation:
    """Rebuild a completion certificate from raw producer contexts."""
    reasons = []
    try:
        rebuilt = build_unitary_valley_sewing_certificate(**raw)
    except (KeyError, TypeError, ValueError):
        rebuilt = None
        reasons.append("certificate_recomputation_failed")
    if rebuilt is not None and dict(certificate) != rebuilt:
        reasons.append("recomputed_certificate_mismatch")
    content = {
        key: deepcopy(value) for key, value in certificate.items()
        if key not in {"status", "reason_codes", "certificate_identity"}
    }
    try:
        identity = canonical_identity(content)
    except (TypeError, ValueError):
        identity = None
    if not valid_sha256_identity(certificate.get("certificate_identity")):
        reasons.append("certificate_identity_malformed")
    if identity != certificate.get("certificate_identity"):
        reasons.append("certificate_identity_mismatch")
    return ScopedEvidenceValidation(
        "blocked" if reasons else str(certificate.get("status")),
        tuple(dict.fromkeys(reasons or certificate.get("reason_codes", []))),
    )


def validate_unitary_valley_sewing_certificate_context(
    certificate: Mapping[str, object],
    context: Mapping[str, object],
) -> bool:
    """Require the exact producer certificate and recompute its raw evidence."""
    raw_inputs = context.get("raw_inputs")
    return bool(
        context.get("certificate") == certificate
        and isinstance(raw_inputs, Mapping)
        and validate_unitary_valley_sewing_certificate(
            certificate, **raw_inputs
        ).status == "passed"
    )


def build_unitary_valley_sewing_completion_report(
    *, attempts: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Collect agreeing paths without fabricating a sampled target row."""
    certificates, raw_by_identity = [], {}
    for raw in attempts:
        certificate = build_unitary_valley_sewing_certificate(**raw)
        certificates.append(certificate)
        identity = certificate.get("certificate_identity")
        if valid_sha256_identity(identity):
            raw_by_identity[str(identity)] = raw
    grouped = {}
    for certificate in certificates:
        target = _map(certificate.get("target"))
        if certificate.get("status") == "passed":
            key = tuple(str(value) for value in (
                target.get("frame_label", ""),
                target.get("valley", ""),
                target.get("source_hsp_label", ""),
            ))
            grouped.setdefault(key, []).append(certificate)

    inferred, blocked, contexts = [], [], {}
    for (_, target_valley, target_hsp), rows in grouped.items():
        vectors = {
            tuple(sorted(
                (str(label), int(count))
                for label, count in _map(
                    row["target"].get("irrep_multiplicities")
                ).items()
            ))
            for row in rows
        }
        if len(vectors) != 1:
            blocked.append({
                "target_valley": target_valley,
                "target_source_hsp_label": target_hsp,
                "reason": "multiple_unitary_sewing_paths_disagree",
                "certificate_identities": [
                    row.get("certificate_identity") for row in rows
                ],
            })
            continue
        first = rows[0]
        raw = raw_by_identity[str(first["certificate_identity"])]
        source = first["source"]
        source_vector = sorted({
            (str(row["source"].get("irrep", "")), int(row["source"].get("multiplicity", 1)))
            for row in rows
        })
        common = _inferred_common(raw, rows, source_vector, target_valley, target_hsp)
        for irrep, multiplicity in dict(next(iter(vectors))).items():
            inferred.append({
                **common,
                "matched_irrep": irrep,
                "irrep_multiplicity": multiplicity,
                "irrep_source_provenance": _inferred_provenance(
                    raw, rows, source, target_hsp
                ),
            })
        for row in rows:
            identity = str(row["certificate_identity"])
            contexts[identity] = {
                "certificate": deepcopy(row),
                "raw_inputs": raw_by_identity[identity],
            }
    return {
        "status": (
            "blocked_unknown" if blocked or (certificates and not inferred)
            else "has_inferred_rows" if inferred else "no_inferred_rows"
        ),
        "inferred_candidate_count": len(inferred),
        "blocked_attempt_count": sum(
            row.get("status") != "passed" for row in certificates
        ),
        "inferred_candidates": inferred,
        "blocked_targets": blocked,
        "attempts": certificates,
        "_validation_contexts": contexts,
    }


def _inferred_common(raw, rows, source_vector, target_valley, target_hsp):
    target = _map(raw.get("target_context"))
    source = rows[0]["source"]
    subspace_group = deepcopy(target.get("subspace_space_group", {}))
    if isinstance(subspace_group, dict):
        subspace_group.setdefault(
            "candidate_space_group_number", subspace_group.get("number")
        )
        subspace_group.setdefault(
            "candidate_space_group_symbol", subspace_group.get("symbol")
        )
    return {
        "valley": target_valley,
        "workflow_path": "unitary_valley_sewing_completion",
        "readiness_level": "trusted",
        "subspace_group_candidate": getattr(raw.get("target_table"), "name", ""),
        "subspace_space_group": subspace_group,
        "matching_strategy": "bilbao_restricted_character",
        "valley_preserving_operation_ids": list(
            target.get("valley_preserving_operation_ids", [])
        ),
        "source_operation_map": {},
        "source": f"unitary_valley_sewing_completion/{target_hsp}/{target_valley}",
        "ready_for_ebr_input": True,
        "completion_kind": "inferred_by_unitary_valley_sewing",
        "evidence_sampled_kpoint": source.get("sampled_kpoint"),
        "evidence_valley": source.get("valley"),
        "evidence_source_hsp_label": source.get("source_hsp_label"),
        "evidence_irrep_vector": [
            {"irrep": label, "multiplicity": count}
            for label, count in source_vector
        ],
        "unitary_valley_sewing_certificates": deepcopy(rows),
    }


def _inferred_provenance(raw, rows, source, target_hsp):
    cprime = _map(source.get("cprime"))
    return {
        "source_hsp_label": target_hsp,
        "source_table_sg_number": getattr(raw.get("target_table"), "number", None),
        "source_table_spinor": bool(getattr(raw.get("target_table"), "spinor", False)),
        "standard_setting_hsp_mapping": {
            "standard_setting_certificate": _json(
                raw.get("target_standard_setting_certificate")
            )
        },
        "cprime": {key: cprime.get(key) for key in _CPRIME_KEYS},
        "unitary_valley_sewing_certificate_identities": [
            row.get("certificate_identity") for row in rows
        ],
    }


def _source_cprime(raw, source, evidence, reasons):
    context = _map(raw.get("source_cprime_context"))
    record, inputs = context.get("record"), context.get("raw_inputs")
    if not isinstance(record, Mapping) or not isinstance(inputs, Mapping):
        reasons.append("source_cprime_context_missing")
        return {}
    validation = validate_scoped_representation_evidence_record(record, **inputs)
    expected = {
        _CPRIME_KEYS[0]: record.get("source_basis_certificate_identity"),
        _CPRIME_KEYS[1]: record.get("double_space_group_lift_certificate_identity"),
        _CPRIME_KEYS[2]: record.get("evidence_identity"),
    }
    if validation.status != "passed":
        reasons.append("source_cprime_revalidation_failed")
    if _nested(source, "irrep_source_provenance", "cprime") != expected:
        reasons.append("source_cprime_identity_link_mismatch")
    if record.get("source_basis_certificate_identity") != evidence.get(
        "source_basis_certificate_identity"
    ):
        reasons.append("directed_source_basis_identity_mismatch")
    if context.get("standard_setting_certificate") != raw.get(
        "source_standard_setting_certificate"
    ):
        reasons.append("source_cprime_standard_setting_mismatch")
    return {**expected, "revalidation_status": validation.status}


def _validate_table_links(raw, source, target, reasons):
    source_table, target_table = raw.get("source_table"), raw.get("target_table")
    source_provenance = _map(source.get("irrep_source_provenance"))
    target_provenance = _map(target.get("source_payload_provenance"))
    target_group = _map(target.get("subspace_space_group"))
    checks = (
        source_provenance.get("source_table_sg_number")
        == getattr(source_table, "number", None),
        source_provenance.get("source_table_spinor")
        is bool(getattr(source_table, "spinor", False)),
        target_provenance.get("table_sg_number")
        == getattr(target_table, "number", None),
        target_provenance.get("table_spinor")
        is bool(getattr(target_table, "spinor", False)),
        target_provenance.get("source_hsp_label")
        == target.get("source_hsp_label"),
        target_group.get(
            "candidate_space_group_number", target_group.get("number")
        )
        == getattr(target_table, "number", None),
    )
    if not all(checks):
        reasons.append("reviewed_source_table_identity_mismatch")


def _validate_target_source_payload(target, reasons):
    from valleyscope.irreps.source_payload import (
        build_source_payload_for_projected_hsp_matching,
    )

    context = _map(target.get("source_payload_context"))
    record, inputs = context.get("record"), context.get("raw_inputs")
    if not isinstance(record, Mapping) or not isinstance(inputs, Mapping):
        reasons.append("target_source_payload_context_missing")
        return
    try:
        rebuilt = build_source_payload_for_projected_hsp_matching(**inputs)
    except (KeyError, TypeError, ValueError):
        rebuilt = None
    links = (
        rebuilt is not None and _json(record) == _json(rebuilt),
        target.get("source_operation_map") == record.get("source_operation_map"),
        target.get("source_irrep_characters")
        == record.get("source_irrep_characters"),
        target.get("source_payload_provenance") == record.get("provenance"),
    )
    if not all(links):
        reasons.append("target_source_payload_revalidation_failed")


def _transport(raw, side, scope, reasons):
    from valleyscope.irreps.source_payload import (
        build_certified_standard_operation_transport,
    )

    certificate = _map(raw.get(f"{side}_standard_setting_certificate"))
    operation_ids = list(certificate.get("parent_basis_operation_ids", []))
    try:
        transform = np.asarray(
            certificate["parent_to_standard_direct_transform"], dtype=float
        )
        standard_k = np.linalg.inv(transform).T @ np.asarray(
            scope[f"{side}_k_frac"], dtype=float
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        standard_k = None
    view = build_certified_standard_operation_transport(
        table=raw.get(f"{side}_table"),
        certificate=certificate,
        detected_operations=[
            dict(row) for row in raw.get("detected_operations", [])
            if isinstance(row, Mapping)
            and row.get("operation_id") in set(operation_ids)
        ],
        operation_ids=operation_ids,
        standard_k_frac=standard_k,
        tol=1.0e-8,
    )
    if view.get("status") != "validated":
        reasons.append(f"{side}_standard_setting_transport_blocked")
    return _json(view)


def _nested(mapping, outer, inner):
    value = mapping.get(outer)
    return value.get(inner) if isinstance(value, Mapping) else None


def _map(value):
    return value if isinstance(value, Mapping) else {}


def _multiplicities(value):
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(label, str) and label and isinstance(count, int)
        and not isinstance(count, bool) and count > 0
        for label, count in value.items()
    )


def _json(value):
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if hasattr(value, "tolist"):
        return _json(value.tolist())
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return value
