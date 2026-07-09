"""EBR-input candidate collector from generic Bilbao restricted-character
valley irrep matching results.

Aggregates only trusted, matched valley-preserving irreps from the
generic ``generic_matches_by_kpoint`` path into a compact machine-readable
EBR input candidate schema.  Legacy phase-table matching is removed;
generic restricted-character matching is the sole authoritative source.

Does NOT implement reduced EBR decomposition, compatibility relations,
or new character tables.
"""

from __future__ import annotations

from typing import Any


def build_ebr_input_candidates(
    *,
    irrep_workflow_decisions: dict[str, object] | None,
    valley_irrep_matching: dict[str, object] | None,
    symmetry_adapted_valley_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Collect trusted matched irreps as EBR input candidates.

    Only rows from ``generic_matches_by_kpoint`` satisfying ALL of:
      - workflow decision readiness == "trusted"
      - workflow path not "blocked"
      - matching_status == "matched"
      - matched_irrep is present

    are included as candidates.  Everything else is collected as
    blocked/not_ready with explicit reasons.

    Legacy phase-table by_kpoint matches are no longer processed;
    generic_matches_by_kpoint is the sole authoritative source.
    """
    candidates: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []

    if irrep_workflow_decisions is None or valley_irrep_matching is None:
        return _empty_report("missing input reports")

    decisions_by_kp = irrep_workflow_decisions.get("by_kpoint", {})
    generic_by_kp = valley_irrep_matching.get("generic_matches_by_kpoint", {})

    # --- Generic restricted-character matches ---
    if isinstance(generic_by_kp, dict):
        for kp_name, valleys in generic_by_kp.items():
            if not isinstance(valleys, dict):
                continue
            for v_name, gm in valleys.items():
                if not isinstance(gm, dict):
                    continue
                decision = decisions_by_kp.get(kp_name, {}).get(v_name, {})
                readiness = str(decision.get("readiness_level", "")) if isinstance(decision, dict) else ""
                path = str(decision.get("workflow_path", "")) if isinstance(decision, dict) else ""
                match_status = str(gm.get("matching_status", ""))
                diag = bool(gm.get("diagnostic_only", False))
                mults = gm.get("irrep_multiplicities", {})
                op_map = gm.get("source_operation_map", {})
                vp_ids = gm.get("valley_preserving_operation_ids", [])
                positive_mults, invalid_mults = _validated_multiplicities(mults)
                # Canonical subgroup identity: subspace_space_group is the
                # primary physical object.  The flat derived scalar key is
                # kept for compact exports and external-table interfaces.
                subspace_space_group = (
                    dict(gm.get("subspace_space_group", {}))
                    if isinstance(gm.get("subspace_space_group", {}), dict)
                    else {}
                )
                subspace_group_candidate = (
                    subspace_space_group.get("candidate_space_group_symbol")
                    or gm.get("subspace_group_candidate")
                )
                # Defensive gate: unresolved/null subgroup identity cannot
                # become a physical EBR candidate.
                sg_number = subspace_space_group.get(
                    "candidate_space_group_number"
                )
                sg_resolved = (
                    subspace_space_group.get("status") == "resolved"
                    and isinstance(sg_number, int)
                    and not isinstance(sg_number, bool)
                    and sg_number > 0
                    and isinstance(subspace_group_candidate, str)
                    and bool(subspace_group_candidate)
                    and subspace_group_candidate != "None"
                )

                # Candidate gate
                if (
                    sg_resolved
                    and readiness == "trusted"
                    and path != "blocked"
                    and match_status == "matched"
                    and not diag
                    and positive_mults
                    and not invalid_mults
                ):
                    for irrep_label, mult in positive_mults:
                        candidates.append({
                            "kpoint": kp_name,
                            "valley": v_name,
                            "workflow_path": path,
                            "readiness_level": readiness,
                            "subspace_group_candidate": subspace_group_candidate,
                            "subspace_space_group": subspace_space_group,
                            "matching_strategy": "bilbao_restricted_character",
                            "matched_irrep": irrep_label,
                            "irrep_multiplicity": mult,
                            "valley_preserving_operation_ids": _list_or_empty(vp_ids),
                            "source_operation_map": dict(op_map) if isinstance(op_map, dict) else {},
                            "irrep_source_provenance": _build_irrep_source_provenance(gm),
                            "source": f"valley_irrep_matching/generic/{kp_name}/{v_name}",
                            "ready_for_ebr_input": True,
                        })
                    continue

                # Not a candidate — explain why.
                reasons: list[str] = []
                if not sg_resolved:
                    reasons.append("subspace_group_candidate_unresolved")
                if readiness != "trusted":
                    reasons.append(f"readiness={readiness}")
                if path == "blocked":
                    reasons.append("path=blocked")
                if match_status != "matched":
                    reasons.append(f"matching_status={match_status}")
                if diag:
                    reasons.append("diagnostic_only=true")
                if not isinstance(mults, dict) or not positive_mults:
                    reasons.append("no irrep_multiplicities")
                if invalid_mults:
                    reasons.append(
                        f"invalid irrep_multiplicities: {invalid_mults}"
                    )
                gm_reason = gm.get("reason", "")
                if isinstance(gm_reason, str) and gm_reason:
                    reasons.append(gm_reason)

                blocked.append(_blocked_row(
                    kpoint=kp_name, valley=v_name,
                    readiness=readiness, path=path,
                    matching_status=match_status,
                    reason="; ".join(reasons) if reasons else "generic match blocked",
                ))

    by_kpoint: dict[str, dict[str, list[dict[str, object]]]] = {}
    for c in candidates:
        by_kpoint.setdefault(str(c["kpoint"]), {}).setdefault(str(c["valley"]), []).append(c)

    status = "has_candidates" if candidates else "no_candidates"
    return {
        "status": status,
        "candidate_count": len(candidates),
        "blocked_count": len(blocked),
        "reduced_ebr_decomposition_status": "not_implemented",
        "interpretation": (
            "Trusted matched valley-preserving irreps collected as EBR input "
            "candidates. Reduced EBR decomposition is not implemented here. "
            "Rows with usable_with_caution, blocked, or diagnostic_only status "
            "are excluded from candidates and listed in the blocked section "
            "with explicit reasons."
        ),
        "by_kpoint": by_kpoint,
        "candidates": candidates,
        "blocked": blocked,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _build_irrep_source_provenance(gm: dict[str, object]) -> dict[str, object]:
    """Extract compact irrep source provenance from a generic match entry.

    Captures the convention basis that produced the irrep label:
    subspace SG, source table, HSP label, operation mapping, and
    G_k^(a) operation IDs.
    """
    src_prov = gm.get("source_payload_provenance", {})
    if not isinstance(src_prov, dict):
        src_prov = {}
    ssg = gm.get("subspace_space_group", {})
    if not isinstance(ssg, dict):
        ssg = {}
    provenance: dict[str, object] = {
        "matching_strategy": str(gm.get("matching_strategy", "")),
        "subspace_space_group_number": ssg.get("candidate_space_group_number"),
        "subspace_space_group_symbol": ssg.get("candidate_space_group_symbol"),
        "source_table_sg_number": src_prov.get("table_sg_number"),
        "source_table_name": src_prov.get("table_name"),
        "source_table_spinor": src_prov.get("table_spinor"),
        "source_hsp_label": src_prov.get("source_hsp_label"),
        "valley_preserving_operation_ids": _list_or_empty(
            gm.get("valley_preserving_operation_ids")
        ),
        "source_table_operation_indices": _list_or_empty(
            src_prov.get("source_table_operation_indices")
        ),
    }
    op_mapping = gm.get("operation_mapping_provenance")
    if isinstance(op_mapping, str) and op_mapping:
        provenance["operation_mapping_provenance"] = op_mapping
    standard_mapping = src_prov.get("standard_setting_hsp_mapping")
    if isinstance(standard_mapping, dict) and standard_mapping:
        provenance["standard_setting_hsp_mapping"] = dict(standard_mapping)
    return provenance


def _validated_multiplicities(
    mults: object,
) -> tuple[list[tuple[str, int]], list[str]]:
    if not isinstance(mults, dict):
        return [], ["not a mapping"]
    positive: list[tuple[str, int]] = []
    invalid: list[str] = []
    for label, value in mults.items():
        if not isinstance(label, str) or not label:
            invalid.append(f"{label!r}: invalid label")
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            invalid.append(f"{label}: {value!r}")
            continue
        positive.append((label, value))
    return positive, invalid


def _list_or_empty(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _blocked_row(
    *,
    kpoint: str,
    valley: str,
    op_id: object = None,
    readiness: str = "",
    path: str = "",
    matching_status: str = "",
    reason: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "kpoint": kpoint,
        "valley": valley,
        "readiness_level": readiness,
        "workflow_path": path,
        "reason": reason,
        "ready_for_ebr_input": False,
    }
    if op_id is not None:
        row["operation_id"] = op_id
    if matching_status:
        row["matching_status"] = matching_status
    return row


def _empty_report(reason: str) -> dict[str, object]:
    return {
        "status": "no_candidates",
        "candidate_count": 0,
        "blocked_count": 0,
        "reduced_ebr_decomposition_status": "not_implemented",
        "interpretation": reason,
        "by_kpoint": {},
        "candidates": [],
        "blocked": [],
    }
