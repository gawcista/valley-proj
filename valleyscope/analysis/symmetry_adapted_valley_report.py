"""Serialize symmetry-adapted valley analysis into a compact JSON-safe report.

This module chains:
  q-cut valley seed projectors P_a^0
  -> symmetry-adapted projectors P_a^sym
  -> valley-preserving representations D_a(h)
  -> valley sewing matrices B_{ba}(g)
  -> character / eigenphase diagnostics

This module provides the formal symmetry-adapted valley analysis layer. Passing
readiness means the staged report is internally consistent and suitable as
input to later table matching; it does not by itself assign final EBR labels.
"""

from __future__ import annotations

import numpy as np

from valleyscope.analysis.symmetry_adapted_character_diagnostics import (
    build_valley_preserving_character_diagnostics,
    summarize_valley_preserving_character_diagnostics,
)
from valleyscope.analysis.symmetry_adapted_representations import (
    build_symmetry_adapted_representation_diagnostics,
    summarize_symmetry_adapted_representations,
)
from valleyscope.subspace.symmetry_adapted_projectors import (
    build_symmetry_adapted_projectors_for_orbit,
)


def build_symmetry_adapted_valley_report(
    *,
    seed_projectors: dict[str, np.ndarray],
    representations: dict[object, np.ndarray],
    valley_mappings: dict[object, dict[str, str]],
    orbit: list[str],
    reference_valley: str,
    rank: int | None = None,
    rank_method: str = "gap",
    rank_tol: float = 0.5,
    expected_total_projector: np.ndarray | None = None,
    orthonormality_tol: float = 1e-8,
    unitarity_tol: float = 1e-8,
    modulus_tol: float = 1e-8,
    closure_mapping: dict[tuple[object, object], object] | None = None,
    spinor_wavefunction: bool = False,
    spinor_convention_verified: bool | None = None,
    operation_orders: dict[object, int] | None = None,
    seed_overlap_warn_tol: float = 0.8,
    seed_overlap_fail_tol: float = 0.5,
    projector_symmetry_warn_tol: float = 1e-2,
    projector_symmetry_fail_tol: float = 1e-1,
    ebr_seed_overlap_min: float = 0.8,
    ebr_unitarity_max: float = 1e-3,
) -> dict[str, object]:
    """Build a full symmetry-adapted valley analysis report.

    local_irrep_ready is True only when every pipeline stage passes readiness
    checks.  The separate irrep_matching_input_ready gate controls whether the
    result may be passed to valley irrep matching and EBR input layers.
    """
    # Stage 1: symmetry-adapted projectors
    try:
        sym_result = build_symmetry_adapted_projectors_for_orbit(
            seed_projectors=seed_projectors,
            representations=representations,
            valley_mappings=valley_mappings,
            orbit=orbit,
            reference_valley=reference_valley,
            rank=rank,
            rank_method=rank_method,
            rank_tol=rank_tol,
            expected_total_projector=expected_total_projector,
            seed_overlap_warn_tol=seed_overlap_warn_tol,
            seed_overlap_fail_tol=seed_overlap_fail_tol,
            projector_symmetry_warn_tol=projector_symmetry_warn_tol,
            projector_symmetry_fail_tol=projector_symmetry_fail_tol,
        )
    except ValueError as exc:
        return _failed_report(
            orbit=orbit,
            reference_valley=reference_valley,
            reason=f"projector_input_invalid: {exc}",
        )
    proj_diag = sym_result.diagnostics
    proj_failed = proj_diag.status == "failed"
    proj_warn = proj_diag.status == "warn"
    projector_summary = _projector_summary_from_diagnostics(proj_diag)

    if proj_failed:
        reason = f"projector_construction_failed: {proj_diag.reason}"
        return {
            "status": "diagnostic_only",
            "reason": reason,
            "feature_status": "formal",
            "workflow_integration_status": "integrated",
            "trusted_irrep_label": False,
            "local_irrep_ready": False,
            "diagnostic_only": True,
            "irrep_matching_input_ready": False,
            "irrep_matching_input_status": "blocked",
            "irrep_matching_input_reason": "projector construction failed",
            "orbit": orbit,
            "reference_valley": reference_valley,
            "symmetry_adapted_projectors": projector_summary,
            "valley_preserving_representations": _not_evaluated_representation_summary(
                orbit,
                "not evaluated because projector construction failed",
            ),
            "valley_sewing_matrices": _not_evaluated_sewing_summary(
                "not evaluated because projector construction failed"
            ),
            "valley_preserving_character_diagnostics": _not_evaluated_character_summary(
                "not evaluated because projector construction failed"
            ),
            "subspace_group": _blocked_subspace_group(
                reason="projector_construction_failed",
                spinor_convention_verified=spinor_convention_verified,
            ),
            "subspace_space_group": _not_evaluated_subspace_space_group(
                "not evaluated in orbit-level report"
            ),
            "ebr_mapping_input": _blocked_ebr_mapping_input(
                blocked_by=["projector_construction_failed"],
                spinor_convention_verified=spinor_convention_verified,
                notes="Projector construction failed before EBR input could be assembled.",
            ),
        }

    # Stage 2: VP representations + sewing
    valley_bases = dict(sym_result.eigenvectors)
    # Store raw matrices for downstream diagnostics (e.g. subspace_representation_quality).
    # These are stripped by summarize_symmetry_adapted_valley_report and do not
    # appear in the public JSON schema.
    _internal_raw_eigenvectors = {
        str(valley): np.asarray(vecs, dtype=np.complex128)
        for valley, vecs in sym_result.eigenvectors.items()
    }
    _internal_raw_projectors = {
        str(valley): np.asarray(p, dtype=np.complex128)
        for valley, p in sym_result.projectors.items()
    }
    rep_diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases=valley_bases,
        representations=representations,
        valley_mappings=valley_mappings,
        orbit=orbit,
        orthonormality_tol=orthonormality_tol,
        unitarity_tol=unitarity_tol,
        closure_mapping=closure_mapping,
    )

    # Stage 3: character / eigenphase diagnostics
    vp_reps: dict[str, object] = {
        "status": rep_diag["status"],
        "reason": rep_diag["reason"],
        "representations": rep_diag.get("valley_preserving_representations", {}).get(
            "representations", {}
        ),
        "unitarity_error": rep_diag.get("valley_preserving_representations", {}).get(
            "unitarity_error", {}
        ),
    }
    char_diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp_reps,
        orbit=orbit,
        input_diagnostic_only=proj_failed or not rep_diag["local_irrep_ready"],
        input_local_irrep_ready=(
            False if proj_failed else rep_diag["local_irrep_ready"]
        ),
        unitarity_tol=unitarity_tol,
        modulus_tol=modulus_tol,
    )

    # --- Aggregate readiness ---
    diagnostic_only = (
        proj_failed
        or rep_diag["diagnostic_only"]
        or char_diag["diagnostic_only"]
    )
    local_irrep_ready = (
        not diagnostic_only
        and char_diag["local_irrep_ready"]
    )
    (
        irrep_matching_input_ready,
        irrep_matching_input_status,
        irrep_matching_input_reason,
    ) = _resolve_irrep_matching_input_gate(
        local_irrep_ready=local_irrep_ready,
        diagnostic_only=diagnostic_only,
        spinor_wavefunction=spinor_wavefunction,
        spinor_convention_verified=spinor_convention_verified,
    )

    reasons: list[str] = []
    if proj_failed:
        reasons.append(f"projector_construction_failed: {proj_diag.reason}")
    elif proj_warn:
        reasons.append(f"projector_warning: {proj_diag.reason}")
    if rep_diag["diagnostic_only"]:
        reasons.append(f"representation_diagnostics: {rep_diag['reason']}")
    if char_diag["diagnostic_only"]:
        reasons.append(f"character_diagnostics: {char_diag['reason']}")

    rep_summary_full = summarize_symmetry_adapted_representations(rep_diag)
    sewing_summary = {
        "status": rep_summary_full.get("status"),
        "max_sewing_unitarity_error": rep_summary_full.get(
            "max_sewing_unitarity_error"
        ),
        "items": rep_summary_full.get("valley_sewing_matrices_summary", []),
    }
    rep_summary = {
        key: value for key, value in rep_summary_full.items()
        if key != "valley_sewing_matrices_summary"
    }
    char_summary = summarize_valley_preserving_character_diagnostics(char_diag)

    return {
        "status": (
            "diagnostic_only"
            if diagnostic_only else "warn" if proj_warn else "ok"
        ),
        "reason": "; ".join(reasons) if reasons else "all stages passed",
        "feature_status": "formal",
        "workflow_integration_status": "integrated",
        "trusted_irrep_label": False,
        "local_irrep_ready": local_irrep_ready,
        "diagnostic_only": diagnostic_only,
        "irrep_matching_input_ready": irrep_matching_input_ready,
        "irrep_matching_input_status": irrep_matching_input_status,
        "irrep_matching_input_reason": irrep_matching_input_reason,
        "orbit": orbit,
        "reference_valley": reference_valley,
        "symmetry_adapted_projectors": projector_summary,
        "valley_preserving_representations": rep_summary,
        "valley_sewing_matrices": sewing_summary,
        "valley_preserving_character_diagnostics": char_summary,
        "subspace_group": _build_subspace_group(
            rep_diag=rep_diag,
            char_diag=char_diag,
            proj_status=proj_diag.status,
            spinor_convention_verified=spinor_convention_verified,
            operation_orders=operation_orders,
        ),
        "subspace_space_group": _not_evaluated_subspace_space_group(
            "not evaluated in orbit-level report"
        ),
        "ebr_mapping_input": _build_ebr_mapping_input(
            local_irrep_ready=local_irrep_ready,
            diagnostic_only=diagnostic_only,
            spinor_convention_verified=spinor_convention_verified,
            proj_diag=proj_diag,
            char_diag=char_diag,
            subspace_group_candidate=_subspace_group_candidate_from_orders(
                rep_diag=rep_diag,
                operation_orders=operation_orders,
            ),
            ebr_seed_overlap_min=ebr_seed_overlap_min,
            ebr_unitarity_max=ebr_unitarity_max,
        ),
        # Internal raw matrices for downstream quality diagnostics.
        # Stripped from public JSON by summarize_symmetry_adapted_valley_report.
        "_internal_raw_eigenvectors": _internal_raw_eigenvectors,
        "_internal_raw_projectors": _internal_raw_projectors,
    }


def summarize_symmetry_adapted_valley_report(
    report: dict[str, object],
) -> dict[str, object]:
    """Produce the outermost compact summary for inclusion in valley_summary.json.

    This is the public-facing schema; it omits raw matrices and exposes
    status / readiness / error summaries only.
    """
    return {
        "status": report.get("status"),
        "reason": report.get("reason"),
        "feature_status": report.get("feature_status", "formal"),
        "workflow_integration_status":
            report.get("workflow_integration_status", "integrated"),
        "trusted_irrep_label": report.get("trusted_irrep_label", False),
        "local_irrep_ready": report.get("local_irrep_ready"),
        "diagnostic_only": report.get("diagnostic_only"),
        "irrep_matching_input_ready": report.get("irrep_matching_input_ready"),
        "irrep_matching_input_status": report.get("irrep_matching_input_status"),
        "irrep_matching_input_reason": report.get("irrep_matching_input_reason"),
        "orbit": report.get("orbit"),
        "reference_valley": report.get("reference_valley"),
        "symmetry_adapted_projectors": report.get("symmetry_adapted_projectors"),
        "valley_preserving_representations":
            report.get("valley_preserving_representations"),
        "valley_sewing_matrices": report.get("valley_sewing_matrices"),
        "valley_preserving_character_diagnostics":
            report.get("valley_preserving_character_diagnostics"),
        "subspace_group": report.get("subspace_group"),
        "subspace_space_group": report.get("subspace_space_group"),
        "ebr_mapping_input": report.get("ebr_mapping_input"),
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _projector_summary_from_diagnostics(proj_diag) -> dict[str, object]:
    return {
        "status": proj_diag.status,
        "selected_rank": proj_diag.selected_rank,
        "rank_source": proj_diag.rank_source,
        "purification_gap": _safe_float(proj_diag.purification_gap),
        "seed_overlap": {str(k): float(v) for k, v in proj_diag.seed_overlap.items()},
        "orthogonality_error": float(proj_diag.orthogonality_error),
        "total_projector_idempotency_error": float(proj_diag.total_projector_idempotency_error),
        "completeness_error": _safe_float(proj_diag.completeness_error),
        "completeness_source": proj_diag.completeness_source,
        "max_projector_symmetry_error": (
            max(proj_diag.projector_symmetry_error.values())
            if proj_diag.projector_symmetry_error else 0.0
        ),
        "reason": proj_diag.reason,
        "representative_resolution": proj_diag.representative_resolution,
        "representative_candidates": list(proj_diag.representative_candidates),
        "representative_resolution_by_valley": dict(
            proj_diag.representative_resolution_by_valley
        ),
        "representative_candidates_by_valley": {
            str(k): list(v)
            for k, v in proj_diag.representative_candidates_by_valley.items()
        },
        "representative_projector_difference_by_valley": {
            str(k): float(v)
            for k, v in proj_diag.representative_projector_difference_by_valley.items()
        },
        "representative_selection_policy_by_valley": dict(
            proj_diag.representative_selection_policy_by_valley
        ),
        "selected_representative_by_valley": dict(
            proj_diag.selected_representative_by_valley
        ),
        "representative_auto_selected_by_valley": {
            str(k): bool(v)
            for k, v in proj_diag.representative_auto_selected_by_valley.items()
        },
        "representative_candidate_projector_differences_by_valley": {
            str(k): [
                {
                    "candidate_a": item.get("candidate_a"),
                    "candidate_b": item.get("candidate_b"),
                    "projector_difference": float(item.get("projector_difference", 0.0)),
                }
                for item in values
            ]
            for k, values in (
                proj_diag.representative_candidate_projector_differences_by_valley.items()
            )
        },
    }


def _not_evaluated_representation_summary(
    orbit: list[str],
    reason: str,
) -> dict[str, object]:
    return {
        "status": "diagnostic_only",
        "reason": reason,
        "local_irrep_ready": False,
        "diagnostic_only": True,
        "orbit": orbit,
        "selected_rank_by_valley": {},
        "valley_preserving_operations": {},
        "valley_changing_operations": {},
        "max_valley_preserving_unitarity_error": None,
        "max_sewing_unitarity_error": None,
        "representation_closure_status": "not_evaluated",
        "representation_closure_violations": [],
        "valley_preserving_representations": {},
    }


def _not_evaluated_sewing_summary(reason: str) -> dict[str, object]:
    return {
        "status": "diagnostic_only",
        "reason": reason,
        "max_sewing_unitarity_error": None,
        "items": [],
    }


def _not_evaluated_character_summary(reason: str) -> dict[str, object]:
    return {
        "status": "diagnostic_only",
        "reason": reason,
        "local_irrep_ready": False,
        "diagnostic_only": True,
        "irrep_matching_status": "failed_input_readiness",
        "max_valley_preserving_unitarity_error": None,
        "max_eigenvalue_modulus_deviation": None,
        "per_valley": {},
    }


def _resolve_irrep_matching_input_gate(
    *,
    local_irrep_ready: bool,
    diagnostic_only: bool,
    spinor_wavefunction: bool,
    spinor_convention_verified: bool | None,
) -> tuple[bool, str, str]:
    if diagnostic_only or not local_irrep_ready:
        return False, "blocked", "local_irrep_ready=false or diagnostic_only=true"
    if spinor_wavefunction and spinor_convention_verified is not True:
        return False, "blocked", "spinor convention unverified"
    return True, "ready", "all irrep matching input gates passed"


def _failed_report(
    *,
    orbit: list[str],
    reference_valley: str,
    reason: str,
) -> dict[str, object]:
    return {
        "status": "diagnostic_only",
        "reason": reason,
        "feature_status": "formal",
        "workflow_integration_status": "integrated",
        "trusted_irrep_label": False,
        "local_irrep_ready": False,
        "diagnostic_only": True,
        "irrep_matching_input_ready": False,
        "irrep_matching_input_status": "blocked",
        "irrep_matching_input_reason": reason,
        "orbit": orbit,
        "reference_valley": reference_valley,
        "symmetry_adapted_projectors": {
            "status": "failed",
            "selected_rank": 0,
            "rank_source": "failure",
            "purification_gap": None,
            "seed_overlap": {},
            "orthogonality_error": None,
            "total_projector_idempotency_error": None,
            "completeness_error": None,
            "completeness_source": "not_evaluated",
            "max_projector_symmetry_error": None,
            "reason": reason,
            "representative_resolution": "",
            "representative_candidates": [],
            "representative_resolution_by_valley": {},
            "representative_candidates_by_valley": {},
            "representative_projector_difference_by_valley": {},
            "representative_selection_policy_by_valley": {},
            "selected_representative_by_valley": {},
            "representative_auto_selected_by_valley": {},
            "representative_candidate_projector_differences_by_valley": {},
        },
        "valley_preserving_representations": {
            "status": "diagnostic_only",
            "reason": "not evaluated because projector input is invalid",
            "local_irrep_ready": False,
            "diagnostic_only": True,
            "orbit": orbit,
            "selected_rank_by_valley": {},
            "valley_preserving_operations": {},
            "valley_changing_operations": {},
            "max_valley_preserving_unitarity_error": None,
            "max_sewing_unitarity_error": None,
            "representation_closure_status": "not_evaluated",
            "representation_closure_violations": [],
            "valley_preserving_representations": {},
        },
        "valley_sewing_matrices": {
            "status": "diagnostic_only",
            "max_sewing_unitarity_error": None,
            "items": [],
        },
        "valley_preserving_character_diagnostics": {
            "status": "diagnostic_only",
            "reason": "not evaluated because projector input is invalid",
            "local_irrep_ready": False,
            "diagnostic_only": True,
            "irrep_matching_status": "failed_input_readiness",
            "max_valley_preserving_unitarity_error": None,
            "max_eigenvalue_modulus_deviation": None,
            "per_valley": {},
        },
        "subspace_group": _blocked_subspace_group(
            reason=reason,
            spinor_convention_verified=False,
        ),
        "subspace_space_group": _not_evaluated_subspace_space_group(
            "not evaluated in orbit-level report"
        ),
        "ebr_mapping_input": _blocked_ebr_mapping_input(
            blocked_by=["projector_input_invalid"],
            spinor_convention_verified=False,
            notes="Projector input is invalid; EBR input was not assembled.",
        ),
    }


# ---------------------------------------------------------------------------
# Subspace group / EBR readiness helpers
# ---------------------------------------------------------------------------

def _build_subspace_group(
    *,
    rep_diag: dict[str, object],
    char_diag: dict[str, object],
    proj_status: str,
    spinor_convention_verified: bool,
    operation_orders: dict[object, int] | None = None,
) -> dict[str, object]:
    vp_ops = rep_diag.get("valley_preserving_operations", {})
    vc_ops = rep_diag.get("valley_changing_operations", {})
    # Flatten across valleys for the orbit-level report
    all_vp = sorted(set(op for ops in vp_ops.values() if isinstance(ops, list) for op in ops))
    all_vc = sorted(set(op for ops in vc_ops.values() if isinstance(ops, list) for op in ops))

    operation_orders = operation_orders or {}
    vp_orders = [
        int(operation_orders[op_id])
        for op_id in all_vp
        if op_id in operation_orders and int(operation_orders[op_id]) > 1
    ]
    effective_order = max(vp_orders) if vp_orders else 1

    # The valley-projected subspace space group is not determined from
    # max operation order alone.  Report operation-level data; the
    # subspace-space-group identity is unresolved without a reviewed
    # or generic identification source.
    blocked = proj_status == "failed" or char_diag.get("diagnostic_only", True)
    if not spinor_convention_verified:
        blocked = True
    # EBR readiness blocked unless a subspace-space-group identification exists.
    ready_for_ebr = not blocked and effective_order > 1

    reason_parts = []
    if proj_status == "failed":
        reason_parts.append("projector_construction_failed")
    if char_diag.get("diagnostic_only", True):
        reason_parts.append("character_diagnostics_not_ready")
    if not spinor_convention_verified:
        reason_parts.append("spinor_convention_unverified")
    if effective_order <= 1:
        reason_parts.append("trivial_valley_preserving_subgroup")
    reason = "; ".join(reason_parts) if reason_parts else "ready"

    return {
        "status": "candidate" if ready_for_ebr else "blocked",
        "hsp_little_group_operation_ids": all_vp + all_vc,
        "valley_preserving_operation_ids": all_vp,
        "valley_changing_operation_ids": all_vc,
        "operation_orders": {str(k): int(v) for k, v in operation_orders.items()},
        "effective_point_group": None,
        "subspace_group_candidate": None,
        "spinor_convention_verified": spinor_convention_verified,
        "ready_for_ebr_mapping": ready_for_ebr,
        "reason": reason,
    }

def _build_ebr_mapping_input(
    *,
    local_irrep_ready: bool,
    diagnostic_only: bool,
    spinor_convention_verified: bool,
    proj_diag,
    char_diag: dict[str, object],
    subspace_group_candidate: str | None,
    ebr_seed_overlap_min: float,
    ebr_unitarity_max: float,
) -> dict[str, object]:
    blocked_by: list[str] = []
    ready = True

    if not local_irrep_ready:
        ready = False
        blocked_by.append("local_irrep_not_ready")
    if diagnostic_only:
        ready = False
        blocked_by.append("diagnostic_only")
    if not spinor_convention_verified:
        ready = False
        blocked_by.append("spinor_convention_unverified")
    min_overlap = min(proj_diag.seed_overlap.values()) if proj_diag.seed_overlap else 0.0
    if min_overlap < ebr_seed_overlap_min:
        ready = False
        blocked_by.append(f"low_seed_overlap_min={min_overlap:.3f}")
    max_unitarity = char_diag.get("max_valley_preserving_unitarity_error", 0.0) or 0.0
    if max_unitarity > ebr_unitarity_max:
        ready = False
        blocked_by.append(f"representation_unitarity={max_unitarity:.1e}")
    if subspace_group_candidate is None:
        ready = False
        blocked_by.append("subspace_group_candidate_missing")
    chars_available = bool(
        char_diag.get("per_valley") and not char_diag.get("diagnostic_only", True)
    )

    return {
        "ready": ready,
        "blocked_by": blocked_by,
        "required_tables": ["external_reduced_ebr_table"],
        "subspace_group_candidate": subspace_group_candidate,
        "valley_preserving_characters_available": chars_available,
        "spinor_convention_verified": spinor_convention_verified,
        "notes": (
            "Valley irrep matching is downstream in valley_irrep_matching; "
            "reduced EBR decomposition requires a user-supplied validated "
            "external table (see required_tables)."
        ),
    }


def _subspace_group_candidate_from_orders(
    *,
    rep_diag: dict[str, object],
    operation_orders: dict[object, int] | None,
) -> str | None:
    """Returns None — subspace-space-group identity is not inferred from
    operation orders.  A reviewed or generic identification source is required."""
    return None


def _blocked_subspace_group(
    *,
    reason: str,
    spinor_convention_verified: bool | None,
) -> dict[str, object]:
    return {
        "status": "blocked",
        "hsp_little_group_operation_ids": [],
        "valley_preserving_operation_ids": [],
        "valley_changing_operation_ids": [],
        "operation_orders": {},
        "effective_point_group": "C1",
        "subspace_group_candidate": None,
        "legacy_subspace_group_candidate": None,
        "spinor_convention_verified": bool(spinor_convention_verified),
        "ready_for_ebr_mapping": False,
        "reason": reason,
    }


def _not_evaluated_subspace_space_group(reason: str) -> dict[str, object]:
    return {
        "status": "not_evaluated",
        "candidate_space_group_symbol": None,
        "candidate_point_group": None,
        "valley_preserving_operation_ids": [],
        "valley_changing_operation_ids": [],
        "operation_orders": {},
        "source": "full_space_group_valley_mapping",
        "reason": reason,
    }


def _blocked_ebr_mapping_input(
    *,
    blocked_by: list[str],
    spinor_convention_verified: bool | None,
    notes: str,
) -> dict[str, object]:
    return {
        "ready": False,
        "blocked_by": blocked_by,
        "required_tables": ["external_reduced_ebr_table"],
        "subspace_group_candidate": None,
        "valley_preserving_characters_available": False,
        "spinor_convention_verified": bool(spinor_convention_verified),
        "notes": notes,
    }
