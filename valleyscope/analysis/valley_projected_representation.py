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
    valley_irrep_matching: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a representation report from computed per-row data.

    Consumes the existing symmetry eigenvalue CSV rows, symmetry-adapted
    valley report (for subspace_space_group data), irrep workflow
    decisions (for readiness/blocker info), and optional valley irrep
    matching data (for per-group irrep status).  Produces per-row records
    with ``subspace_space_group`` as the primary identifier, plus
    per-``(kpoint, valley)`` grouped ``representation_records``.
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
            legacy_sgc = (
                subspace_group_data.get("legacy_subspace_group_candidate")
                or subspace_group_data.get("subspace_group_candidate")
            )
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
                "state_index": row.get("state_index"),
                "character_raw": row.get("character_raw", ""),
                "character_valley": row.get("character_valley", ""),
                "eigenvalue_real": row.get("eigenvalue_real"),
                "eigenvalue_imag": row.get("eigenvalue_imag"),
                "phase_2pi": row.get("phase_2pi"),
                "root_deviation": row.get("root_deviation"),
                "basis": row.get("basis", ""),
                "blocking_reasons": _blocking_reasons(row, wf_data, diag_only),
                "legacy_subspace_group_candidate": legacy_sgc,
            }
            rows.append(rec)

            # Counts.
            sg_symbol = subspace_space_group_data.get("candidate_space_group_symbol")
            if sg_symbol:
                key = str(sg_symbol)
                subspace_space_group_counts[key] = (
                    subspace_space_group_counts.get(key, 0) + 1
                )
            if legacy_sgc:
                key = str(legacy_sgc)
                legacy_subspace_counts[key] = legacy_subspace_counts.get(key, 0) + 1
            if diag_only:
                diagnostic_count += 1
            elif wf_data.get("readiness_level") == "trusted":
                trusted_count += 1
            else:
                blocked_count += 1

    # --- Per-(kpoint, valley) grouped representation records ---
    representation_records = _build_representation_records(
        rows=rows,
        valley_irrep_matching=valley_irrep_matching,
    )

    return {
        "rows": rows,
        "representation_records": representation_records,
        "grouped_record_count": len(representation_records),
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


def _build_representation_records(
    *,
    rows: list[dict[str, Any]],
    valley_irrep_matching: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Group per-operation rows into per-(kpoint, valley) representation records.

    Each record aggregates all operations for a single (kpoint, valley) pair
    and includes irrep matching data when available.
    """
    # Group rows by (kpoint, valley).
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("kpoint", "")), str(row.get("valley", "")))
        groups.setdefault(key, []).append(row)

    # Collect irrep matching data.
    matching_by_kp: dict[str, Any] = {}
    generic_by_kp: dict[str, Any] = {}
    if isinstance(valley_irrep_matching, dict):
        by_kp = valley_irrep_matching.get("by_kpoint")
        if isinstance(by_kp, dict):
            matching_by_kp = by_kp
        gen_by_kp = valley_irrep_matching.get("generic_matches_by_kpoint")
        if isinstance(gen_by_kp, dict):
            generic_by_kp = gen_by_kp

    records: list[dict[str, Any]] = []
    for (kpoint, valley), group_rows in sorted(groups.items()):
        first = group_rows[0]
        ssg = first.get("subspace_space_group", {})
        if not isinstance(ssg, dict):
            ssg = {}

        # Per-operation entries.  Flat rows are per eigenstate, so aggregate
        # rows with the same operation ID into one representation entry.
        op_groups: dict[str, list[dict[str, Any]]] = {}
        for row in group_rows:
            key = str(row.get("operation_id"))
            op_groups.setdefault(key, []).append(row)

        operations: list[dict[str, Any]] = []
        for _, op_rows in sorted(
            op_groups.items(),
            key=lambda item: _operation_sort_key(
                item[1][0].get("operation_id")
            ),
        ):
            row = op_rows[0]
            sorted_rows = sorted(
                op_rows,
                key=lambda r: _operation_sort_key(r.get("state_index")),
            )
            eigenphases = [
                phase for phase in (
                    _optional_float(r.get("phase_2pi")) for r in sorted_rows
                )
                if phase is not None
            ]
            eigenvalues = [
                ev for ev in (
                    _eigenvalue_entry(r) for r in sorted_rows
                )
                if ev is not None
            ]
            blockers = _unique_strings(
                blocker
                for r in sorted_rows
                for blocker in (
                    r.get("blocking_reasons")
                    if isinstance(r.get("blocking_reasons"), list)
                    else []
                )
            )
            root_deviations = [
                value for value in (
                    _optional_float(r.get("root_deviation")) for r in sorted_rows
                )
                if value is not None
            ]
            op_entry: dict[str, Any] = {
                "operation_id": row.get("operation_id"),
                "operation_order": row.get("operation_order"),
                "character_raw": _nonempty_or_none(row.get("character_raw")),
                "character_valley": _nonempty_or_none(row.get("character_valley")),
                "eigenphases": eigenphases,
                "eigenvalues": eigenvalues,
                "diagnostic_only": any(
                    bool(r.get("diagnostic_only", False)) for r in sorted_rows
                ),
                "topology_input_ready": all(
                    bool(r.get("topology_input_ready", False)) for r in sorted_rows
                ),
                "source_row_count": len(sorted_rows),
            }
            if root_deviations:
                op_entry["max_root_deviation"] = max(root_deviations)
            basis = _nonempty_or_none(row.get("basis"))
            if basis is not None:
                op_entry["basis"] = basis
            if blockers:
                op_entry["blocking_reasons"] = blockers
            operations.append(op_entry)

        # Irrep matching data for this (kpoint, valley).
        # In generic mode, only the generic restricted-character path is
        # authoritative; legacy phase-table matches are never attached.
        matching_mode = (
            str(valley_irrep_matching.get("matching_mode", "not_evaluated"))
            if isinstance(valley_irrep_matching, dict)
            else "not_evaluated"
        )
        irrep_matching: dict[str, Any] | None = None
        gm = generic_by_kp.get(kpoint, {}).get(valley)
        if isinstance(gm, dict):
            irrep_matching = {
                "matching_status": gm.get("matching_status"),
                "matching_strategy": gm.get("matching_strategy"),
                "irrep_multiplicities": gm.get("irrep_multiplicities"),
                "source_operation_map": gm.get("source_operation_map"),
            }
        elif matching_mode != "generic":
            lm = matching_by_kp.get(kpoint, {}).get(valley)
            if isinstance(lm, dict):
                statuses = {
                    str(op_result.get("matching_status"))
                    for op_result in lm.values()
                    if isinstance(op_result, dict)
                }
                irrep_matching = {
                    "matching_status": (
                        "matched"
                        if statuses == {"matched"}
                        else "incomplete"
                        if "matched" in statuses
                        else "not_matched"
                    ),
                    "matching_strategy": (
                        next(
                            (
                                str(op_result.get("matching_strategy", ""))
                                for op_result in lm.values()
                                if isinstance(op_result, dict)
                                and op_result.get("matching_strategy")
                            ),
                            None,
                        )
                    ),
                }

        record: dict[str, Any] = {
            "kpoint": kpoint,
            "valley": valley,
            "subspace_space_group": ssg,
            "hsp_little_group_operation_ids": first.get(
                "hsp_little_group_operation_ids", []
            ),
            "valley_preserving_operation_ids": first.get(
                "valley_preserving_operation_ids", []
            ),
            "valley_changing_operation_ids": first.get(
                "valley_changing_operation_ids", []
            ),
            "valley_preserving_operations": operations,
            "readiness_level": first.get("readiness_level", "?"),
            "workflow_path": first.get("workflow_path", "?"),
            "blocking_reasons": first.get("blocking_reasons", []),
            "irrep_matching": irrep_matching,
            "legacy_subspace_group_candidate": first.get(
                "legacy_subspace_group_candidate"
            ),
        }
        records.append(record)

    return records


def _operation_sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, int(str(value)))
    except (TypeError, ValueError):
        return (1, str(value))


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _eigenvalue_entry(row: dict[str, Any]) -> dict[str, float] | None:
    real = _optional_float(row.get("eigenvalue_real"))
    imag = _optional_float(row.get("eigenvalue_imag"))
    if real is None or imag is None:
        return None
    return {"real": real, "imag": imag}


def _nonempty_or_none(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str) and not value:
        return None
    return value


def _unique_strings(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
