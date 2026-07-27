"""Generic valley-projected representation report.

Builds a serializable representation summary from existing workflow data
without reimplementing projector or symmetry-adapted math.  The primary
physical identifier is ``subspace_space_group``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def build_valley_projected_representation_report(
    *,
    kpoint_names: list[str],
    valley_names: list[str],
    symmetry_eigenvalue_rows: list[dict[str, Any]] | None = None,
    symmetry_adapted_valley_report: dict[str, Any] | None = None,
    irrep_workflow_decisions: dict[str, Any] | None = None,
    valley_irrep_matching: dict[str, Any] | None = None,
    symmetry_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one debug representation record per ``(kpoint, valley)``.

    Records include blocked and identity-only pairs without eigenvalue rows,
    while keeping the parent HSP little group, ``G_k^(a)``, and
    valley-changing operations distinct.
    """
    rows: list[dict[str, Any]] = []

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

    # Canonical matched subgroup identity overrides an unresolved
    # symmetry-adapted placeholder while preserving its operation inventory.
    matching_ssg_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(valley_irrep_matching, dict):
        generic_by_kp = valley_irrep_matching.get(
            "generic_matches_by_kpoint", {}
        )
        if isinstance(generic_by_kp, dict):
            for kp_name, valleys in generic_by_kp.items():
                if not isinstance(valleys, dict):
                    continue
                for valley_name, match in valleys.items():
                    if not isinstance(match, dict):
                        continue
                    ssg = match.get("subspace_space_group")
                    if isinstance(ssg, dict):
                        matching_ssg_lookup[(kp_name, valley_name)] = ssg

    # Per-(kpoint, valley) group-theoretic inventory from symmetry_analysis.
    # The public irrep object is defined on the subspace HSP little group
    # G_k^(a) (valley-preserving ops in the parent HSP little group).
    # Parent full HSP little group and valley-sewing ops are provenance only.
    sym_analysis_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if symmetry_analysis is not None:
        subgroup_report = symmetry_analysis.get(
            "valley_preserving_subgroup_report", {}
        )
        if isinstance(subgroup_report, dict):
            by_kp_sa = subgroup_report.get("by_kpoint", {})
            if isinstance(by_kp_sa, dict):
                for kp_name, kp_data in by_kp_sa.items():
                    if not isinstance(kp_data, dict):
                        continue
                    for v_name, pv_data in kp_data.items():
                        if not isinstance(pv_data, dict):
                            continue
                        # Skip legacy kpoint-level keys.
                        if v_name in (
                            "hsp_little_group_operation_ids",
                            "all_valley_intersection_operation_ids",
                        ):
                            continue
                        sym_analysis_lookup[(str(kp_name), str(v_name))] = {
                            "subspace_hsp_little_group_operation_ids": (
                                pv_data.get("allowed_operation_ids", [])
                            ),
                            "parent_hsp_little_group_operation_ids": (
                                pv_data.get("little_group_operation_ids", [])
                            ),
                            "valley_sewing_operation_ids": (
                                pv_data.get("valley_changing_operation_ids", [])
                            ),
                            "identity_operation_id": pv_data.get(
                                "identity_operation_id"
                            ),
                        }

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
            else:
                subspace_space_group_data = dict(subspace_space_group_data)
            matched_ssg = matching_ssg_lookup.get((kp, valley))
            if matched_ssg:
                subspace_space_group_data.update(matched_ssg)
            subspace_group_data = subspace_data.get("subspace_group", {})
            if not isinstance(subspace_group_data, dict):
                subspace_group_data = {}

            # Extract workflow data.
            wf_data = wf_lookup.get((kp, valley), {})

            # Build representation record.
            diag_only = bool(row.get("diagnostic_only", False))
            topology_ready = bool(row.get("topology_input_ready", False))
            # Public irrep group = subspace HSP little group G_k^(a).
            # Parent full HSP little group and valley-sewing ops are
            # provenance only — they must not feed irrep matching or EBR.
            sa_data = sym_analysis_lookup.get((str(kp), str(valley)), {})
            subspace_hsp_lg = (
                sa_data.get("subspace_hsp_little_group_operation_ids")
                or _list_field(subspace_data, "hsp_preserving_operation_ids")
            )
            parent_hsp_lg = sa_data.get(
                "parent_hsp_little_group_operation_ids", []
            )
            valley_sewing = (
                sa_data.get("valley_sewing_operation_ids")
                or _list_field(
                    subspace_space_group_data, "valley_changing_operation_ids"
                )
            )
            rec: dict[str, Any] = {
                "kpoint": kp,
                "valley": valley,
                "operation_id": op_id,
                "operation_order": row.get("order", "?"),
                "subspace_space_group": _compact_subspace_space_group(
                    subspace_space_group_data
                ),
                "subspace_hsp_little_group_operation_ids": list(subspace_hsp_lg),
                "hsp_little_group_operation_ids": list(subspace_hsp_lg),
                "valley_preserving_operation_ids": list(subspace_hsp_lg),
                "parent_hsp_little_group_operation_ids": list(parent_hsp_lg),
                "valley_sewing_operation_ids": list(valley_sewing),
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
                "basis": row.get("basis", ""),
                "blocking_reasons": _blocking_reasons(row, wf_data, diag_only),
            }
            rows.append(rec)

    # --- Per-(kpoint, valley) grouped representation records ---
    representation_records = _build_representation_records(
        rows=rows,
        valley_irrep_matching=valley_irrep_matching,
    )

    # --- Completeness: ensure every sampled (kpoint, valley) with
    #     symmetry_analysis or workflow data produces a record,
    #     including blocked / identity-only cases with no eigenvalue rows.
    existing_pairs: set[tuple[str, str]] = set()
    for rec in representation_records:
        existing_pairs.add((str(rec.get("kpoint", "")), str(rec.get("valley", ""))))

    for kp_name in kpoint_names:
        for v_name in valley_names:
            pair = (str(kp_name), str(v_name))
            if pair in existing_pairs:
                continue
            sa_data = sym_analysis_lookup.get(pair, {})
            wf_data = wf_lookup.get(pair, {})
            if not sa_data and not wf_data:
                # No symmetry or workflow data for this pair — skip.
                continue

            subspace_hsp_lg = sa_data.get(
                "subspace_hsp_little_group_operation_ids", []
            )
            parent_hsp_lg = sa_data.get(
                "parent_hsp_little_group_operation_ids", []
            )
            valley_sewing = sa_data.get("valley_sewing_operation_ids", [])

            # Subspace space group — prefer matched (resolved) identity.
            matched_ssg = matching_ssg_lookup.get(pair)
            subspace_ssg: dict[str, Any] = {}
            if matched_ssg:
                subspace_ssg = dict(matched_ssg)
            else:
                subspace_data = subspace_lookup.get(pair, {})
                raw_ssg = subspace_data.get("subspace_space_group", {})
                if isinstance(raw_ssg, dict):
                    subspace_ssg = dict(raw_ssg)

            identity_id = sa_data.get("identity_operation_id")
            non_identity_vp = [
                op for op in subspace_hsp_lg if op != identity_id
            ]
            is_identity_only = not non_identity_vp and len(subspace_hsp_lg) >= 1

            readiness = wf_data.get("readiness_level", "not_evaluated")
            wf_path = wf_data.get("workflow_path", "blocked")

            # Blocking reasons — identity-only G_k^(a) is a valid
            # physical local result, not a code defect.
            blockers: list[str] = []
            if is_identity_only:
                # G_k^(a)={E} is a physical property, not a blocker.
                # The note goes into irrep_matching.reason, not here.
                pass
            else:
                if readiness == "blocked" or wf_path == "blocked":
                    blockers.append("workflow_blocked")
                wf_blockers = wf_data.get("blocking_reasons", [])
                if isinstance(wf_blockers, list):
                    blockers.extend(wf_blockers)

            # Irrep matching data — may exist even for identity-only cases.
            irrep_matching: dict[str, Any] | None = None
            gbkp = (
                valley_irrep_matching.get("generic_matches_by_kpoint", {})
                if isinstance(valley_irrep_matching, dict)
                else {}
            )
            gm = gbkp.get(kp_name, {}).get(v_name)
            if isinstance(gm, dict):
                irrep_matching = {
                    "matching_status": gm.get("matching_status"),
                    "matching_strategy": gm.get("matching_strategy"),
                    "irrep_multiplicities": gm.get("irrep_multiplicities"),
                    "local_representation_dimension": gm.get(
                        "local_representation_dimension"
                    ),
                }
                if is_identity_only:
                    irrep_matching["reason"] = (
                        "G_k^(a) contains only the identity operation; "
                        "no non-identity valley-preserving operation in "
                        "the HSP little group"
                    )

            record: dict[str, Any] = {
                "kpoint": kp_name,
                "valley": v_name,
                "subspace_space_group": _compact_subspace_space_group(
                    subspace_ssg
                ),
                "subspace_hsp_little_group_operation_ids": list(subspace_hsp_lg),
                "hsp_little_group_operation_ids": list(subspace_hsp_lg),
                "valley_preserving_operation_ids": list(subspace_hsp_lg),
                "parent_hsp_little_group_operation_ids": list(parent_hsp_lg),
                "valley_sewing_operation_ids": list(valley_sewing),
                "valley_preserving_operations": [],
                "readiness_level": readiness,
                "workflow_path": wf_path,
                "blocking_reasons": blockers,
                "irrep_matching": irrep_matching,
            }
            representation_records.append(record)

    # Sort records for deterministic output.
    representation_records.sort(
        key=lambda r: (str(r.get("kpoint", "")), str(r.get("valley", "")))
    )

    readiness_level_counts = Counter(
        str(record.get("readiness_level", "not_evaluated"))
        for record in representation_records
    )
    space_group_symbols = [
        ssg.get("candidate_space_group_symbol")
        for record in representation_records
        if isinstance((ssg := record.get("subspace_space_group")), dict)
        and ssg.get("candidate_space_group_symbol")
    ]

    return {
        "representation_records": representation_records,
        "record_count": len(representation_records),
        "readiness_level_counts": dict(readiness_level_counts),
        "subspace_space_group_record_counts": dict(Counter(space_group_symbols)),
        "valley_labels": sorted(set(r["valley"] for r in representation_records)),
        "kpoint_labels": sorted(set(r["kpoint"] for r in representation_records)),
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
    generic_by_kp: dict[str, Any] = {}
    if isinstance(valley_irrep_matching, dict):
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
            basis = _nonempty_or_none(row.get("basis"))
            if basis is not None:
                op_entry["basis"] = basis
            if blockers:
                op_entry["blocking_reasons"] = blockers
            operations.append(op_entry)

        # Irrep matching data for this (kpoint, valley).
        # Only the generic restricted-character path is authoritative.
        irrep_matching: dict[str, Any] | None = None
        gm = generic_by_kp.get(kpoint, {}).get(valley)
        if isinstance(gm, dict):
            irrep_matching = {
                "matching_status": gm.get("matching_status"),
                "matching_strategy": gm.get("matching_strategy"),
                "irrep_multiplicities": gm.get("irrep_multiplicities"),
                "source_operation_map": gm.get("source_operation_map"),
                "local_representation_dimension": gm.get(
                    "local_representation_dimension"
                ),
            }

        record: dict[str, Any] = {
            "kpoint": kpoint,
            "valley": valley,
            "subspace_space_group": ssg,
            "subspace_hsp_little_group_operation_ids": first.get(
                "subspace_hsp_little_group_operation_ids", []
            ),
            "hsp_little_group_operation_ids": first.get(
                "hsp_little_group_operation_ids", []
            ),
            "valley_preserving_operation_ids": first.get(
                "valley_preserving_operation_ids", []
            ),
            "parent_hsp_little_group_operation_ids": first.get(
                "parent_hsp_little_group_operation_ids", []
            ),
            "valley_sewing_operation_ids": first.get(
                "valley_sewing_operation_ids", []
            ),
            "valley_preserving_operations": operations,
            "readiness_level": first.get("readiness_level", "?"),
            "workflow_path": first.get("workflow_path", "?"),
            "blocking_reasons": _unique_strings(
                blocker
                for row in group_rows
                for blocker in (
                    row.get("blocking_reasons")
                    if isinstance(row.get("blocking_reasons"), list)
                    else []
                )
            ),
            "irrep_matching": irrep_matching,
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
