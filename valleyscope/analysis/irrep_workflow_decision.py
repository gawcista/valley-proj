"""Workflow decision layer: choose direct-qcut, symmetry-adapted, or blocked path.

This is a pure decision helper that does NOT modify existing diagnostics or
readiness gates.  It inspects the three independent diagnostic streams
(q-cut seed symmetry, target-subspace closure, symmetry-adapted projector
quality) and produces a compact per-(valley) decision record.
"""

from __future__ import annotations

from typing import Any


# Public readiness levels (only three)
READINESS_TRUSTED = "trusted"
READINESS_USABLE_WITH_CAUTION = "usable_with_caution"
READINESS_BLOCKED = "blocked"

# Public workflow paths
PATH_DIRECT_QCUT = "direct_qcut"
PATH_SYMMETRY_ADAPTED = "symmetry_adapted"
PATH_BLOCKED = "blocked"


def decide_irrep_workflow(
    *,
    # --- q-cut seed projector symmetry ---
    seed_symmetry_status: str = "not_evaluated",
    seed_symmetry_max_epsilon: float | None = None,
    seed_symmetry_failed_count: int = 0,
    seed_symmetry_warn_count: int = 0,
    # --- target-subspace closure ---
    closure_quality: str = "not_evaluated",
    closure_max_raw_unitarity: float | None = None,
    closure_max_residual: float | None = None,
    # --- q-cut symmetry eigenvalue readiness ---
    qcut_eigenvalue_ready_count: int = 0,
    qcut_eigenvalue_total_count: int = 0,
    # --- symmetry-adapted projector quality ---
    sym_adapted_proj_status: str = "not_evaluated",
    sym_adapted_max_seed_overlap: float | None = None,
    sym_adapted_min_seed_overlap: float | None = None,
    sym_adapted_local_irrep_ready: bool = False,
    sym_adapted_diagnostic_only: bool = True,
    # --- common ---
    spinor_convention_verified: bool = False,
) -> dict[str, object]:
    """Determine workflow path and readiness level for a single valley subspace.

    Returns a compact decision record.  All three diagnostic streams are
    inspected independently; the most favourable feasible path is chosen.
    """
    reasons: list[str] = []

    # --- Direct q-cut path ---
    # Seed projector symmetry must pass for valley-preserving operations.
    seed_clean = (
        seed_symmetry_status == "passed"
        and seed_symmetry_failed_count == 0
        and seed_symmetry_warn_count == 0
    )
    seed_usable = (
        seed_symmetry_status not in ("failed", "not_evaluated")
        and seed_symmetry_failed_count == 0
        and (seed_symmetry_max_epsilon is None or seed_symmetry_max_epsilon <= 0.1)
    )
    # Target subspace must not be blocked.
    closure_ok = closure_quality not in ("blocked", "not_evaluated")
    closure_usable = closure_quality in ("usable_with_caution", "ok", "clean")
    closure_clean = closure_quality in ("ok", "clean")
    # Q-cut eigenvalue readiness: all relevant rows must be ready.
    qcut_ok = (
        qcut_eigenvalue_total_count > 0
        and qcut_eigenvalue_ready_count == qcut_eigenvalue_total_count
    )

    if seed_usable and closure_ok and qcut_ok:
        if seed_clean and closure_clean and spinor_convention_verified:
            return _decision(
                workflow_path=PATH_DIRECT_QCUT,
                readiness=READINESS_TRUSTED,
                reason="q-cut seed basis is symmetry-consistent; target-subspace closure ok; all readiness gates passed",
                uses_symmetry_adapted_projector=False,
                direct_qcut_allowed=True,
            )
        followups: list[str] = []
        caution_reasons: list[str] = []
        if not seed_clean:
            caution_reasons.append(
                f"seed projector symmetry has warnings "
                f"(status={seed_symmetry_status}, warn={seed_symmetry_warn_count})"
            )
        if not closure_clean:
            caution_reasons.append(f"target-subspace closure quality={closure_quality}")
        if not spinor_convention_verified:
            caution_reasons.append("spinor convention unverified")
            followups.append("verify spinor convention against benchmark")
        return _decision(
            workflow_path=PATH_DIRECT_QCUT,
            readiness=READINESS_USABLE_WITH_CAUTION,
            reason="; ".join(caution_reasons) if caution_reasons else "direct q-cut path usable with caution",
            required_followup="; ".join(followups),
            uses_symmetry_adapted_projector=False,
            direct_qcut_allowed=True,
        )

    if not seed_usable:
        reasons.append(
            f"seed projector symmetry not clean "
            f"(status={seed_symmetry_status}, failed={seed_symmetry_failed_count})"
        )
    if not closure_ok:
        reasons.append(f"target-subspace closure quality={closure_quality}")
    if not qcut_ok:
        reasons.append(
            f"q-cut eigenvalue readiness insufficient "
            f"(ready={qcut_eigenvalue_ready_count}/{qcut_eigenvalue_total_count})"
        )

    # --- Symmetry-adapted path ---
    # Symmetry-adapted projector construction must be usable.
    sa_proj_usable = sym_adapted_proj_status in ("ok", "warn")
    sa_ready = sym_adapted_local_irrep_ready and not sym_adapted_diagnostic_only

    if closure_quality == "blocked":
        return _decision(
            workflow_path=PATH_BLOCKED,
            readiness=READINESS_BLOCKED,
            reason=f"target-subspace closure blocked ({closure_max_raw_unitarity or 'N/A'})",
            required_followup="expand target bands or verify plane-wave mapping",
            uses_symmetry_adapted_projector=bool(sa_proj_usable),
            direct_qcut_allowed=False,
        )

    if sa_proj_usable and closure_usable:
        if sa_ready and spinor_convention_verified:
            return _decision(
                workflow_path=PATH_SYMMETRY_ADAPTED,
                readiness=READINESS_TRUSTED,
                reason="symmetry-adapted projector path: local irrep ready, closure usable",
                uses_symmetry_adapted_projector=True,
                direct_qcut_allowed=False,
            )
        elif sa_ready:
            return _decision(
                workflow_path=PATH_SYMMETRY_ADAPTED,
                readiness=READINESS_USABLE_WITH_CAUTION,
                reason="symmetry-adapted projector path: local irrep ready but spinor convention unverified",
                required_followup="verify spinor convention against benchmark",
                uses_symmetry_adapted_projector=True,
                direct_qcut_allowed=False,
            )
        else:
            return _decision(
                workflow_path=PATH_SYMMETRY_ADAPTED,
                readiness=READINESS_USABLE_WITH_CAUTION,
                reason=(
                    f"symmetry-adapted projector constructed but local irrep not ready "
                    f"(diagnostic_only={sym_adapted_diagnostic_only}, "
                    f"local_irrep_ready={sym_adapted_local_irrep_ready})"
                ),
                required_followup="improve seed overlap or D_raw unitarity at source HSP",
                uses_symmetry_adapted_projector=True,
                direct_qcut_allowed=False,
            )

    # --- Blocked ---
    if not sa_proj_usable:
        reasons.append(f"symmetry-adapted projector status={sym_adapted_proj_status}")
    if not closure_usable and closure_quality != "blocked":
        reasons.append(f"closure quality={closure_quality}")

    return _decision(
        workflow_path=PATH_BLOCKED,
        readiness=READINESS_BLOCKED,
        reason="; ".join(reasons) if reasons else "all paths blocked",
        required_followup="check projector construction, seed overlap, or D_raw closure",
        uses_symmetry_adapted_projector=False,
        direct_qcut_allowed=False,
    )


def _decision(
    *,
    workflow_path: str,
    readiness: str,
    reason: str,
    uses_symmetry_adapted_projector: bool = False,
    direct_qcut_allowed: bool = False,
    required_followup: str = "",
) -> dict[str, object]:
    result: dict[str, object] = {
        "workflow_path": workflow_path,
        "readiness_level": readiness,
        "reason": reason,
        "uses_symmetry_adapted_projector": uses_symmetry_adapted_projector,
        "direct_qcut_allowed": direct_qcut_allowed,
    }
    if required_followup:
        result["required_followup"] = required_followup
    return result


def build_irrep_workflow_decisions(
    *,
    projector_symmetry_report: dict[str, object] | None,
    target_subspace_closure_report: dict[str, object] | None,
    symmetry_adapted_valley_report: dict[str, object] | None,
    symmetry_rows: list[dict[str, object]],
    valley_names: list[str],
    spinor_convention_verified: bool = False,
) -> dict[str, object]:
    """Build per-(kpoint, valley) workflow decision records.

    Aggregates the three diagnostic streams and calls decide_irrep_workflow
    for each (kpoint, valley) pair.
    """
    by_kpoint: dict[str, object] = {}

    # Collect per-(kpoint, valley) diagnostics from each stream.
    seed_by_kp: dict[str, dict[str, dict[str, Any]]] = {}
    if isinstance(projector_symmetry_report, dict):
        for kp_name, kp_data in projector_symmetry_report.get("by_kpoint", {}).items():
            if not isinstance(kp_data, dict):
                continue
            for row in kp_data.get("seed_projector_symmetry", []):
                if not isinstance(row, dict):
                    continue
                src_v = str(row.get("source_valley", ""))
                mapped_v = str(row.get("mapped_valley", ""))
                # Only count valley-preserving rows.
                if src_v and mapped_v == src_v:
                    seed_by_kp.setdefault(kp_name, {}).setdefault(src_v, []).append(row)

    # Extract valley-preserving operation sets per (kpoint, valley) from
    # the symmetry_adapted_valley_report to filter closure quality.
    vp_ops_by_kp_valley: dict[str, dict[str, set[object]]] = {}
    if isinstance(symmetry_adapted_valley_report, dict):
        for kp_name, kp_data in symmetry_adapted_valley_report.get("by_kpoint", {}).items():
            if not isinstance(kp_data, dict):
                continue
            for subspace in kp_data.get("valley_preserving_subspaces", []):
                if not isinstance(subspace, dict):
                    continue
                orbit = subspace.get("orbit", [])
                if not orbit:
                    continue
                v = str(orbit[0])
                hsp_ops = subspace.get("hsp_preserving_operation_ids")
                if not isinstance(hsp_ops, list):
                    sg = subspace.get("subspace_group", {})
                    hsp_ops = sg.get("valley_preserving_operation_ids", []) if isinstance(sg, dict) else []
                if not isinstance(hsp_ops, list):
                    hsp_ops = []
                for op_id in hsp_ops:
                    vp_ops_by_kp_valley.setdefault(kp_name, {}).setdefault(v, set()).add(op_id)

    closure_by_kp: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if isinstance(target_subspace_closure_report, dict):
        for kp_name, rows in target_subspace_closure_report.get("by_kpoint", {}).items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if not bool(row.get("little_group_passed", True)):
                    continue
                op_id = row.get("operation_id")
                for v in valley_names:
                    vp_ops = vp_ops_by_kp_valley.get(kp_name, {}).get(v, set())
                    # Only include closure rows for valley-preserving ops.
                    if op_id in vp_ops:
                        closure_by_kp.setdefault(kp_name, {}).setdefault(v, []).append(row)

    # Extract eigenvalue readiness by (kpoint, valley)
    qcut_ready: dict[str, dict[str, dict[str, bool]]] = {}
    for row in symmetry_rows:
        if not isinstance(row, dict):
            continue
        kp = str(row.get("kpoint", ""))
        v = str(row.get("target_valley", ""))
        if not kp or not v:
            continue
        entry = qcut_ready.setdefault(kp, {}).setdefault(v, {"ready": 0, "total": 0})
        entry["total"] += 1
        if bool(row.get("topology_input_ready", False)):
            entry["ready"] += 1

    # Extract symmetry-adapted diagnostics per valley
    sa_by_kp: dict[str, dict[str, dict[str, Any]]] = {}
    if isinstance(symmetry_adapted_valley_report, dict):
        for kp_name, kp_data in symmetry_adapted_valley_report.get("by_kpoint", {}).items():
            if not isinstance(kp_data, dict):
                continue
            for subspace in kp_data.get("valley_preserving_subspaces", []):
                if not isinstance(subspace, dict):
                    continue
                orbit = subspace.get("orbit", [])
                if not orbit:
                    continue
                v = str(orbit[0])
                proj = subspace.get("symmetry_adapted_projectors", {})
                if not isinstance(proj, dict):
                    proj = {}
                sa_by_kp.setdefault(kp_name, {})[v] = {
                    "proj_status": str(proj.get("status", "not_evaluated")),
                    "max_seed_overlap": max(proj.get("seed_overlap", {}).values()) if proj.get("seed_overlap") else None,
                    "min_seed_overlap": min(proj.get("seed_overlap", {}).values()) if proj.get("seed_overlap") else None,
                    "local_irrep_ready": bool(subspace.get("local_irrep_ready", False)),
                    "diagnostic_only": bool(subspace.get("diagnostic_only", True)),
                }

    # Build decisions
    for kp_name in sorted(set(list(seed_by_kp) + list(closure_by_kp) + list(qcut_ready) + list(sa_by_kp))):
        kp_decisions: dict[str, object] = {}
        for v in valley_names:
            # Seed projector symmetry
            seed_rows = seed_by_kp.get(kp_name, {}).get(v, [])
            seed_failed = sum(1 for r in seed_rows if r.get("status") in ("failed",))
            seed_warn = sum(1 for r in seed_rows if r.get("status") == "warn")
            seed_status = (
                "failed" if seed_failed > 0
                else "warn" if seed_warn > 0
                else "passed" if seed_rows else "not_evaluated"
            )
            seed_epsilons = [
                float(r.get("epsilon_seed", 0.0)) for r in seed_rows
                if r.get("epsilon_seed") is not None
            ]
            seed_max_eps = max(seed_epsilons) if seed_epsilons else None

            # Closure quality: worst among VP operations
            closure_rows = closure_by_kp.get(kp_name, {}).get(v, [])
            closure_qualities = [
                str(r.get("closure_quality", "not_evaluated")) for r in closure_rows
            ]
            worst_closure = "not_evaluated"
            for cq in ("blocked", "usable_with_caution", "clean", "ok", "not_evaluated"):
                if cq in closure_qualities:
                    worst_closure = cq
                    break
            closure_unitarities = [
                float(r.get("raw_unitarity_error", 0.0)) for r in closure_rows
                if r.get("raw_unitarity_error") is not None
            ]
            closure_max_unit = max(closure_unitarities) if closure_unitarities else None
            closure_residuals = [
                float(r.get("max_closure_residual", 0.0)) for r in closure_rows
                if r.get("max_closure_residual") is not None
            ]
            closure_max_res = max(closure_residuals) if closure_residuals else None

            # Q-cut eigenvalue readiness
            qcut = qcut_ready.get(kp_name, {}).get(v, {"ready": 0, "total": 0})

            # Symmetry-adapted diagnostics
            sa = sa_by_kp.get(kp_name, {}).get(v, {})

            decision = decide_irrep_workflow(
                seed_symmetry_status=seed_status,
                seed_symmetry_max_epsilon=seed_max_eps,
                seed_symmetry_failed_count=seed_failed,
                seed_symmetry_warn_count=seed_warn,
                closure_quality=worst_closure,
                closure_max_raw_unitarity=closure_max_unit,
                closure_max_residual=closure_max_res,
                qcut_eigenvalue_ready_count=qcut["ready"],
                qcut_eigenvalue_total_count=qcut["total"],
                sym_adapted_proj_status=sa.get("proj_status", "not_evaluated"),
                sym_adapted_min_seed_overlap=sa.get("min_seed_overlap"),
                sym_adapted_local_irrep_ready=sa.get("local_irrep_ready", False),
                sym_adapted_diagnostic_only=sa.get("diagnostic_only", True),
                spinor_convention_verified=spinor_convention_verified,
            )
            kp_decisions[v] = decision
        if kp_decisions:
            by_kpoint[kp_name] = kp_decisions

    return {
        "status": "ok",
        "readiness_levels": [READINESS_TRUSTED, READINESS_USABLE_WITH_CAUTION, READINESS_BLOCKED],
        "workflow_paths": [PATH_DIRECT_QCUT, PATH_SYMMETRY_ADAPTED, PATH_BLOCKED],
        "by_kpoint": by_kpoint,
    }
