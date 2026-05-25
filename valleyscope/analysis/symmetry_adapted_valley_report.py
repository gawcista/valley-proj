"""Experimental scaffold: serialize symmetry-adapted valley analysis pipeline
into a compact JSON-safe report.

This module chains:
  q-cut valley seed projectors P_a^0
  -> symmetry-adapted projectors P_a^sym
  -> valley-preserving representations D_a(h)
  -> valley sewing matrices B_{ba}(g)
  -> character / eigenphase diagnostics

This module does NOT feed into production irrep matching. Passing readiness
only means that the experimental staged report is internally consistent; it
does not promote any production irrep label.
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
) -> dict[str, object]:
    """Build a full symmetry-adapted valley analysis report.

    Returns an experimental report. local_irrep_ready is True only when every
    pipeline stage passes readiness checks, but this helper does not feed into
    production irrep matching.
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
            "experimental": True,
            "workflow_integration_status": "not_integrated",
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
        }

    # Stage 2: VP representations + sewing
    valley_bases = dict(sym_result.eigenvectors)
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
        "experimental": True,
        "workflow_integration_status": "not_integrated",
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
        ),
        "ebr_mapping_input": _build_ebr_mapping_input(
            local_irrep_ready=local_irrep_ready,
            diagnostic_only=diagnostic_only,
            spinor_convention_verified=spinor_convention_verified,
            proj_diag=proj_diag,
            char_diag=char_diag,
        ),
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
        "experimental": report.get("experimental", True),
        "workflow_integration_status":
            report.get("workflow_integration_status", "not_integrated"),
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
        "experimental": True,
        "workflow_integration_status": "not_integrated",
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
) -> dict[str, object]:
    vp_ops = rep_diag.get("valley_preserving_operations", {})
    vc_ops = rep_diag.get("valley_changing_operations", {})
    # Flatten across valleys for the orbit-level report
    all_vp = sorted(set(op for ops in vp_ops.values() if isinstance(ops, list) for op in ops))
    all_vc = sorted(set(op for ops in vc_ops.values() if isinstance(ops, list) for op in ops))

    # Effective point group: max order among VP ops
    vp_orders = set()
    vp_reps = rep_diag.get("valley_preserving_representations", {}).get("representations", {})
    for valley_reps in vp_reps.values():
        if isinstance(valley_reps, dict):
            vp_orders.update(int(r.shape[0]) if hasattr(r, 'shape') else 0 for r in valley_reps.values())
    effective_order = max(vp_orders) if vp_orders else 0
    epg = f"C{effective_order}" if effective_order > 1 else "C1"

    # Subspace group candidate
    if effective_order == 2:
        candidate = "C2_like"
    elif effective_order >= 3:
        candidate = f"C{effective_order}_like"
    else:
        candidate = None

    # Readiness: blocked if projector failed or char not ready
    blocked = proj_status == "failed" or char_diag.get("diagnostic_only", True)
    if not spinor_convention_verified:
        blocked = True
    ready_for_ebr = not blocked and candidate is not None

    reason_parts = []
    if proj_status == "failed":
        reason_parts.append("projector_construction_failed")
    if char_diag.get("diagnostic_only", True):
        reason_parts.append("character_diagnostics_not_ready")
    if not spinor_convention_verified:
        reason_parts.append("spinor_convention_unverified")
    if candidate is None:
        reason_parts.append("no_subgroup_candidate")
    reason = "; ".join(reason_parts) if reason_parts else "ready"

    return {
        "status": "candidate" if ready_for_ebr else "blocked",
        "hsp_little_group_operation_ids": all_vp + all_vc,
        "valley_preserving_operation_ids": all_vp,
        "valley_changing_operation_ids": all_vc,
        "effective_point_group": epg,
        "subspace_group_candidate": candidate,
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
    if min_overlap < 0.8:
        ready = False
        blocked_by.append(f"low_seed_overlap_min={min_overlap:.3f}")
    max_unitarity = char_diag.get("max_valley_preserving_unitarity_error", 0.0) or 0.0
    if max_unitarity > 1e-3:
        ready = False
        blocked_by.append(f"representation_unitarity={max_unitarity:.1e}")
    chars_available = bool(
        char_diag.get("per_valley") and not char_diag.get("diagnostic_only", True)
    )

    return {
        "ready": ready,
        "blocked_by": blocked_by,
        "required_tables": ["unknown — character table matching not yet implemented"],
        "subspace_group_candidate": None,
        "valley_preserving_characters_available": chars_available,
        "spinor_convention_verified": spinor_convention_verified,
        "notes": (
            "EBR mapping requires character table matching for valley-preserving "
            "subgroup irreps.  Not implemented in current V1.1 experimental pipeline."
        ),
    }
