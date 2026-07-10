"""EBR problem-instance collector from trusted EBR input candidates.

Groups trusted candidate irreps into per-valley/per-subspace-group EBR
problem instances with a certificate-aware physical identity key.

State model
-----------
1. ``sampled_basis`` — trusted irrep rows collected on a sampled HSP basis.
   ``ready_for_reduced_table_validation`` = true, ``ready_for_ebr_decomposition`` = false.
2. ``table_validated`` — HSP basis validated against a reviewed
   irreptables-derived reduced table (requires external table; not yet wired).
3. ``validated_basis`` — basis and certificate confirmed; instance is ready
   for exact reduced EBR decomposition.
4. Solve attempted/completed (downstream in ``reduced_ebr_mapping.py``).

Does NOT implement reduced EBR decomposition, EBR table matching,
compatibility relations, or new physics.
"""

from __future__ import annotations

from typing import Any


def build_ebr_problem_instances(
    *,
    ebr_input_candidates: dict[str, object] | None,
) -> dict[str, object]:
    """Build EBR problem instances from trusted input candidates.

    Returns a dict with instances grouped by certificate-aware physical
    identity (SG number, SG symbol, Hall number, certificate validation
    status, valley).
    """
    if ebr_input_candidates is None:
        return _empty_report("no EBR input candidates available")

    candidates: list[dict[str, object]] = []
    raw = ebr_input_candidates.get("candidates")
    if isinstance(raw, list):
        for c in raw:
            if isinstance(c, dict) and c.get("ready_for_ebr_input") is True:
                candidates.append(c)

    if not candidates:
        return _empty_report("no trusted EBR input candidates")

    # Group by certificate-aware physical identity.
    # Candidates with inconsistent or unresolved setting certificates
    # must not be silently merged into one final-ready EBR problem.
    groups: dict[tuple[int, str, int, str, str], list[dict[str, object]]] = {}
    for c in candidates:
        ssg = c.get("subspace_space_group", {})
        sg_symbol = (
            ssg.get("candidate_space_group_symbol")
            if isinstance(ssg, dict) else None
        )
        sg_number = _int_or_zero(
            ssg.get("candidate_space_group_number") if isinstance(ssg, dict) else None,
        )
        sg = str(sg_symbol) if sg_symbol else str(c.get("subspace_group_candidate", ""))
        valley = str(c.get("valley", ""))
        cert_fp = _certificate_fingerprint(c)
        groups.setdefault(
            (sg_number, sg, cert_fp.hall_number, cert_fp.validation_status, valley),
            [],
        ).append(c)

    instances: list[dict[str, object]] = []
    instance_counter = 0

    for (_sg_num, sg, _hall, _cert_status, valley), cands in groups.items():
        instance_counter += 1
        instance_id = f"ebr_instance_{instance_counter:03d}"

        # --- Canonical subgroup identity ---
        first_candidate_ssg = _first_subspace_space_group(cands)
        subspace_space_group: dict[str, object] = (
            dict(first_candidate_ssg) if isinstance(first_candidate_ssg, dict) else {}
        )
        canonical_sg = (
            subspace_space_group.get("candidate_space_group_symbol")
            or sg
        )
        canonical_sg_number = (
            subspace_space_group.get("candidate_space_group_number")
            or _sg_num
        )

        # --- Certificate identity from merged candidates ---
        cert_identity = _certificate_identity(cands)

        # --- Aggregate provenance ---
        workflow_paths = sorted({
            str(c.get("workflow_path", ""))
            for c in cands if c.get("workflow_path")
        })
        readiness_levels = sorted({
            str(c.get("readiness_level", ""))
            for c in cands if c.get("readiness_level")
        })
        workflow_path = str(cands[0].get("workflow_path", ""))
        readiness_level = str(cands[0].get("readiness_level", ""))

        irreps_by_kpoint: dict[str, list[str]] = {}
        operations_by_kpoint: dict[str, list[object]] = {}
        irrep_records_by_kpoint: dict[str, list[dict[str, object]]] = {}
        for c in cands:
            kp = str(c.get("kpoint", ""))
            irrep = c.get("matched_irrep")
            op_id = c.get("operation_id")
            if irrep:
                multiplicity = _positive_multiplicity(c.get("irrep_multiplicity"))
                irreps_by_kpoint.setdefault(kp, []).extend(
                    [str(irrep)] * multiplicity
                )
            if op_id is not None:
                operations_by_kpoint.setdefault(kp, []).append(op_id)
            record: dict[str, object] = {
                "valley": valley,
                "operation_id": c.get("operation_id"),
                "operation_order": c.get("operation_order"),
                "matched_irrep": c.get("matched_irrep"),
                "irrep_multiplicity": _positive_multiplicity(
                    c.get("irrep_multiplicity")
                ),
                "character": c.get("character"),
                "eigenphases": c.get("eigenphases", []),
                "workflow_path": str(c.get("workflow_path", "")),
                "readiness_level": str(c.get("readiness_level", "")),
                "source": c.get("source", ""),
            }
            for key in (
                "matching_strategy",
                "subspace_space_group",
                "valley_preserving_operation_ids",
                "source_operation_map",
                "irrep_source_provenance",
            ):
                if key in c:
                    record[key] = c[key]
            irrep_records_by_kpoint.setdefault(kp, []).append(record)

        # --- HSP basis ---
        actual_hsps = sorted(irreps_by_kpoint.keys())
        expected_hsps = list(actual_hsps)
        optional_hsps: list[str] = []

        has_hsps = bool(actual_hsps)
        # State 1 (sampled_basis): trusted irrep rows collected, not yet
        # validated against a reviewed reduced table.
        hsp_basis_status = "sampled_basis" if has_hsps else "no_data"
        status = "sampled_basis" if has_hsps else "no_data"

        instances.append({
            "instance_id": instance_id,
            "valley": valley,
            "subspace_group_candidate": canonical_sg,
            "subspace_sg_number": canonical_sg_number,
            "subspace_space_group": subspace_space_group,
            "certificate_identity": cert_identity,
            "workflow_path": workflow_path,
            "workflow_paths": workflow_paths,
            "readiness_level": readiness_level,
            "readiness_evidence": readiness_levels,
            "irreps_by_kpoint": {k: v for k, v in sorted(irreps_by_kpoint.items())},
            "operations_by_kpoint": {
                k: sorted(v, key=_sort_key)
                for k, v in sorted(operations_by_kpoint.items())
            },
            "irrep_records_by_kpoint": {
                k: sorted(v, key=lambda r: (_sort_key(r.get("operation_id"))))
                for k, v in sorted(irrep_records_by_kpoint.items())
            },
            "candidate_count": len(cands),
            "status": status,
            "ready_for_reduced_table_validation": has_hsps,
            "ready_for_ebr_decomposition": False,
            "blocked_by": [],
            "expected_hsps": expected_hsps,
            "expected_hsp_policy_source": "sampled_irrep_basis",
            "hsp_basis_status": hsp_basis_status,
            "optional_hsps": optional_hsps,
            "actual_hsps": actual_hsps,
            "missing_optional_hsps": [],
        })

    overall_status = "has_instances" if instances else "no_instances"
    return {
        "status": overall_status,
        "instance_count": len(instances),
        "reduced_ebr_decomposition_status": "not_implemented",
        "interpretation": (
            "Per-valley/per-subspace-group EBR problem instances grouped from "
            "trusted input candidates by certificate-aware (SG number, SG symbol, "
            "Hall number, certificate validation status, valley) identity.  "
            "State 1 (sampled_basis): irrep rows collected on a sampled HSP basis, "
            "ready_for_reduced_table_validation=true, "
            "ready_for_ebr_decomposition=false.  Promotion to state 2 "
            "(table_validated) and state 3 (validated_basis) requires a reviewed "
            "irreptables-derived reduced table."
        ),
        "instances": instances,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _first_subspace_space_group(
    cands: list[dict[str, object]],
) -> dict[str, object]:
    """Return the canonical subspace_space_group from the first candidate."""
    for c in cands:
        ssg = c.get("subspace_space_group")
        if isinstance(ssg, dict) and ssg:
            return ssg
    return {}


def _sort_key(op_id: object) -> tuple[int, object]:
    try:
        return (0, int(str(op_id)))
    except (TypeError, ValueError):
        return (1, str(op_id))


def _positive_multiplicity(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 1


def _int_or_zero(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _empty_report(reason: str) -> dict[str, object]:
    return {
        "status": "no_instances",
        "instance_count": 0,
        "reduced_ebr_decomposition_status": "not_implemented",
        "interpretation": reason,
        "instances": [],
    }


# ---------------------------------------------------------------------------
# Certificate-aware identity
# ---------------------------------------------------------------------------

class _CertFingerprint:
    """Lightweight hashable fingerprint of per-candidate certificate data."""

    __slots__ = ("hall_number", "validation_status")

    def __init__(self, hall_number: int, validation_status: str):
        self.hall_number = hall_number
        self.validation_status = validation_status


def _certificate_fingerprint(candidate: dict[str, object]) -> _CertFingerprint:
    """Extract a certificate fingerprint from one candidate."""
    prov = candidate.get("irrep_source_provenance")
    hall_number = 0
    validation_status = "not_evaluated"
    if isinstance(prov, dict):
        kmap = prov.get("standard_setting_hsp_mapping")
        if isinstance(kmap, dict):
            cert = kmap.get("standard_setting_certificate")
            if isinstance(cert, dict):
                hn = cert.get("hall_number")
                if isinstance(hn, int) and not isinstance(hn, bool):
                    hall_number = int(hn)
                vs = cert.get("validation_status")
                if isinstance(vs, str) and vs:
                    validation_status = str(vs)
    return _CertFingerprint(hall_number, validation_status)


def _certificate_identity(
    cands: list[dict[str, object]],
) -> dict[str, object]:
    """Build certificate-identity dict from merged candidates.

    Records the set of unique Hall numbers and validation statuses across
    all candidates, plus whether any certificate is unresolved.
    """
    hall_numbers: set[int] = set()
    validation_statuses: set[str] = set()
    for c in cands:
        fp = _certificate_fingerprint(c)
        if fp.hall_number:
            hall_numbers.add(fp.hall_number)
        validation_statuses.add(fp.validation_status)
    return {
        "hall_numbers": sorted(hall_numbers),
        "certificate_validation_statuses": sorted(validation_statuses),
        "any_unresolved": "unresolved" in validation_statuses
                          or "not_evaluated" in validation_statuses,
    }
