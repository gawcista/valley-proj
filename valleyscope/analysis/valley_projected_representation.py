"""Generic valley-projected representation report.

Builds a serializable representation summary from existing workflow data
without reimplementing projector or symmetry-adapted math.  The primary
physical identifier is ``subspace_space_group``; legacy C2_like/C3_like
hints are carried under ``legacy_subspace_group_candidate`` and must not
be treated as final physical labels.
"""

from __future__ import annotations

from typing import Any


def build_valley_projected_representation_report(
    *,
    kpoint_names: list[str],
    valley_names: list[str],
    symmetry_eigenvalue_rows: list[dict[str, Any]] | None = None,
    symmetry_adapted_valley_report: dict[str, Any] | None = None,
    irrep_workflow_decisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a representation report from computed per-row data.

    Consumes the existing symmetry eigenvalue CSV rows, symmetry-adapted
    valley report (for subspace_space_group data), and irrep workflow
    decisions (for readiness/blocker info).  Produces per-row records
    with ``subspace_space_group`` as the primary identifier.
    """
    rows: list[dict[str, Any]] = []
    subspace_space_group_counts: dict[str, int] = {}
    legacy_subspace_counts: dict[str, int] = {}
    trusted_count: int = 0
    blocked_count: int = 0
    diagnostic_count: int = 0

    # Build a lookup from (kpoint, valley) to subspace data.
    subspace_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if symmetry_adapted_valley_report is not None:
        by_kp = symmetry_adapted_valley_report.get("by_kpoint", {})
        if isinstance(by_kp, dict):
            for kp_name, kp_data in by_kp.items():
                if not isinstance(kp_data, dict):
                    continue
                vp_subspaces = kp_data.get("valley_preserving_subspaces", [])
                if isinstance(vp_subspaces, list):
                    for vs in vp_subspaces:
                        if not isinstance(vs, dict):
                            continue
                        ref = vs.get("reference_valley")
                        if not ref:
                            orbit = vs.get("orbit", [])
                            if isinstance(orbit, list) and orbit:
                                ref = orbit[0]
                        subspace_lookup[(kp_name, ref)] = vs

    # Build a workflow lookup.
    wf_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if irrep_workflow_decisions is not None:
        by_kp_wf = irrep_workflow_decisions.get("by_kpoint", {})
        if isinstance(by_kp_wf, dict):
            for kp_name, valleys in by_kp_wf.items():
                if not isinstance(valleys, dict):
                    continue
                for v_name, wf_info in valleys.items():
                    if isinstance(wf_info, dict):
                        wf_lookup[(kp_name, v_name)] = wf_info

    # Process eigenvalue rows.
    if symmetry_eigenvalue_rows is not None:
        for row in symmetry_eigenvalue_rows:
            if not isinstance(row, dict):
                continue
            kp = row.get("kpoint", "?")
            valley = row.get("target_valley", "?")
            op_id = row.get("operation_id", "?")

            # Extract subspace data.
            subspace_data = subspace_lookup.get((kp, valley), {})
            subspace_space_group_data = subspace_data.get(
                "subspace_space_group", {}
            )
            if not isinstance(subspace_space_group_data, dict):
                subspace_space_group_data = {}
            subspace_group_data = subspace_data.get("subspace_group", {})
            if not isinstance(subspace_group_data, dict):
                subspace_group_data = {}

            # Extract workflow data.
            wf_data = wf_lookup.get((kp, valley), {})

            # Build representation record.
            diag_only = bool(row.get("diagnostic_only", False))
            topology_ready = bool(row.get("topology_input_ready", False))
            rec: dict[str, Any] = {
                "kpoint": kp,
                "valley": valley,
                "operation_id": op_id,
                "operation_order": row.get("order", "?"),
                "subspace_space_group": _compact_subspace_space_group(
                    subspace_space_group_data
                ),
                "hsp_little_group_operation_ids": _list_field(
                    subspace_data, "hsp_preserving_operation_ids"
                ),
                "valley_preserving_operation_ids": _list_field(
                    subspace_space_group_data,
                    "valley_preserving_operation_ids",
                ),
                "valley_changing_operation_ids": _list_field(
                    subspace_space_group_data,
                    "valley_changing_operation_ids",
                ),
                "readiness_level": wf_data.get("readiness_level", "?"),
                "workflow_path": wf_data.get("workflow_path", "?"),
                "diagnostic_only": diag_only,
                "topology_input_ready": topology_ready,
                "blocking_reasons": _blocking_reasons(row, wf_data, diag_only),
                "legacy_subspace_group_candidate": subspace_group_data.get(
                    "subspace_group_candidate",
                ),
            }
            rows.append(rec)

            # Counts.
            sg_symbol = subspace_space_group_data.get("candidate_space_group_symbol")
            if sg_symbol:
                key = str(sg_symbol)
                subspace_space_group_counts[key] = (
                    subspace_space_group_counts.get(key, 0) + 1
                )
            sgc = subspace_group_data.get("subspace_group_candidate", "")
            if sgc:
                key = str(sgc)
                legacy_subspace_counts[key] = legacy_subspace_counts.get(key, 0) + 1
            if diag_only:
                diagnostic_count += 1
            elif wf_data.get("readiness_level") == "trusted":
                trusted_count += 1
            else:
                blocked_count += 1

    return {
        "rows": rows,
        "subspace_space_group_counts": subspace_space_group_counts,
        "legacy_subspace_group_candidate_counts": legacy_subspace_counts,
        "trusted_representation_count": trusted_count,
        "blocked_representation_count": blocked_count,
        "diagnostic_only_count": diagnostic_count,
        "valley_labels": sorted(set(r["valley"] for r in rows)),
        "kpoint_labels": sorted(set(r["kpoint"] for r in rows)),
    }


def _compact_subspace_space_group(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_space_group_symbol": data.get(
            "candidate_space_group_symbol"
        ),
        "candidate_space_group_number": data.get(
            "candidate_space_group_number"
        ),
        "valley_preserving_operation_ids": data.get(
            "valley_preserving_operation_ids", []
        ),
        "status": data.get("status", "?"),
    }


def _list_field(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if isinstance(value, list):
        return value
    return []


def _blocking_reasons(
    row: dict[str, Any],
    wf_data: dict[str, Any],
    diagnostic_only: bool,
) -> list[str]:
    reasons: list[str] = []
    if diagnostic_only:
        reason = row.get("reason", "")
        if isinstance(reason, str) and reason:
            reasons.append(reason)
        else:
            reasons.append("diagnostic_only")
    else:
        if not bool(row.get("rotation_ready", False)):
            reasons.append("rotation_not_ready")
        if not bool(row.get("topology_input_ready", False)):
            reasons.append("topology_input_not_ready")
        wf_path = wf_data.get("workflow_path", "")
        if wf_path == "blocked":
            reasons.append("workflow_blocked")
    return reasons
