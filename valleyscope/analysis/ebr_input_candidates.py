"""EBR-input candidate collector from trusted valley irrep matching results.

Aggregates only trusted, matched valley-preserving irreps into a compact
machine-readable EBR input candidate schema.  Does NOT implement reduced
EBR decomposition, compatibility relations, or new character tables.
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

    Only rows satisfying ALL of:
      - workflow decision readiness == "trusted"
      - workflow path not "blocked"
      - matching_status == "matched"
      - matched_irrep is present

    are included as candidates.  Everything else is collected as
    blocked/not_ready with explicit reasons.
    """
    candidates: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []

    if irrep_workflow_decisions is None or valley_irrep_matching is None:
        return _empty_report("missing input reports")

    decisions_by_kp = irrep_workflow_decisions.get("by_kpoint", {})
    matching_by_kp = valley_irrep_matching.get("by_kpoint", {})
    generic_by_kp = valley_irrep_matching.get("generic_matches_by_kpoint", {})

    # Collect character data for candidates
    char_data = _collect_character_lookup(symmetry_adapted_valley_report)

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

                # Candidate gate
                if (
                    readiness == "trusted"
                    and path != "blocked"
                    and match_status == "matched"
                    and not diag
                    and isinstance(mults, dict)
                    and mults
                ):
                    for irrep_label, mult in mults.items():
                        if not isinstance(mult, int) or mult <= 0:
                            continue
                        for _ in range(mult):
                            candidates.append({
                                "kpoint": kp_name,
                                "valley": v_name,
                                "workflow_path": path,
                                "readiness_level": readiness,
                                "matching_strategy": "bilbao_restricted_character",
                                "matched_irrep": irrep_label,
                                "irrep_multiplicity": mult,
                                "valley_preserving_operation_ids": list(vp_ids),
                                "source_operation_map": dict(op_map) if isinstance(op_map, dict) else {},
                                "source": f"valley_irrep_matching/generic/{kp_name}/{v_name}",
                                "ready_for_ebr_input": True,
                            })
                    continue

                # Not a candidate — explain why.
                reasons: list[str] = []
                if readiness != "trusted":
                    reasons.append(f"readiness={readiness}")
                if path == "blocked":
                    reasons.append("path=blocked")
                if match_status != "matched":
                    reasons.append(f"matching_status={match_status}")
                if diag:
                    reasons.append("diagnostic_only=true")
                if not isinstance(mults, dict) or not mults:
                    reasons.append("no irrep_multiplicities")
                gm_reason = gm.get("reason", "")
                if isinstance(gm_reason, str) and gm_reason:
                    reasons.append(gm_reason)

                blocked.append(_blocked_row(
                    kpoint=kp_name, valley=v_name,
                    readiness=readiness, path=path,
                    matching_status=match_status,
                    reason="; ".join(reasons) if reasons else "generic match blocked",
                ))

    for kp_name in sorted(set(decisions_by_kp) | set(matching_by_kp)):
        kp_decisions = decisions_by_kp.get(kp_name, {})
        kp_matches = matching_by_kp.get(kp_name, {})

        for v_name in sorted(set(kp_decisions) | set(kp_matches)):
            decision = kp_decisions.get(v_name)
            matches = kp_matches.get(v_name, {})

            if not isinstance(decision, dict):
                blocked.append(_blocked_row(
                    kpoint=kp_name, valley=v_name,
                    reason="no workflow decision",
                ))
                continue

            readiness = str(decision.get("readiness_level", ""))
            path = str(decision.get("workflow_path", ""))

            if not isinstance(matches, dict) or not matches:
                blocked.append(_blocked_row(
                    kpoint=kp_name, valley=v_name,
                    readiness=readiness, path=path,
                    reason="no irrep matching results",
                ))
                continue

            for op_id, match in matches.items():
                if not isinstance(match, dict):
                    continue
                match_status = str(match.get("matching_status", ""))
                matched_irrep = match.get("matched_irrep")
                diagnostic_only = bool(match.get("diagnostic_only", False))

                # Candidate gate
                if (
                    readiness == "trusted"
                    and path != "blocked"
                    and match_status == "matched"
                    and not diagnostic_only
                    and matched_irrep is not None
                ):
                    sg_candidate = match.get("subspace_group_candidate")
                    eigenphases = match.get("eigenphases", [])
                    char = char_data.get((kp_name, v_name, _coerce_id(op_id)))
                    candidates.append({
                        "kpoint": kp_name,
                        "valley": v_name,
                        "workflow_path": path,
                        "readiness_level": readiness,
                        "subspace_group_candidate": sg_candidate,
                        "operation_id": op_id,
                        "operation_order": match.get("operation_order"),
                        "matched_irrep": matched_irrep,
                        "character": char,
                        "eigenphases": eigenphases,
                        "source": f"valley_irrep_matching/{kp_name}/{v_name}",
                        "ready_for_ebr_input": True,
                    })
                    continue

                # Not a candidate — explain why
                reasons: list[str] = []
                if readiness != "trusted":
                    reasons.append(f"readiness={readiness}")
                if path == "blocked":
                    reasons.append("path=blocked")
                if match_status != "matched":
                    reasons.append(f"matching_status={match_status}")
                if diagnostic_only:
                    reasons.append("diagnostic_only=true")
                if matched_irrep is None:
                    reasons.append("no matched_irrep")

                blocked.append(_blocked_row(
                    kpoint=kp_name, valley=v_name,
                    op_id=op_id,
                    readiness=readiness, path=path,
                    matching_status=match_status,
                    reason="; ".join(reasons),
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

def _collect_character_lookup(
    report: dict[str, object] | None,
) -> dict[tuple[str, str, object], dict[str, object]]:
    """Extract per-(kpoint, valley, op) character data from SA report."""
    result: dict[tuple[str, str, object], dict[str, object]] = {}
    if report is None:
        return result
    for kp_name, kp_data in report.get("by_kpoint", {}).items():
        if not isinstance(kp_data, dict):
            continue
        for subspace in kp_data.get("valley_preserving_subspaces", []):
            if not isinstance(subspace, dict):
                continue
            orbit = subspace.get("orbit", [])
            if not orbit:
                continue
            v = str(orbit[0])
            cd = subspace.get("valley_preserving_character_diagnostics", {})
            if not isinstance(cd, dict):
                continue
            for _, items in cd.get("per_valley", {}).items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    op_id = item.get("operation_id")
                    char = item.get("character")
                    if op_id is not None and char is not None:
                        result[(kp_name, v, _coerce_id(op_id))] = char
    return result


def _coerce_id(op_id: object) -> object:
    try:
        return int(str(op_id))
    except (TypeError, ValueError):
        return op_id


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
