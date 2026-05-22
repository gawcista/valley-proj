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

    reasons: list[str] = []
    if proj_failed:
        reasons.append(f"projector_construction_failed: {proj_diag.reason}")
    elif proj_warn:
        reasons.append(f"projector_warning: {proj_diag.reason}")
    if rep_diag["diagnostic_only"]:
        reasons.append(f"representation_diagnostics: {rep_diag['reason']}")
    if char_diag["diagnostic_only"]:
        reasons.append(f"character_diagnostics: {char_diag['reason']}")

    # Compact projector summary
    projector_summary = {
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
    }

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
        "orbit": orbit,
        "reference_valley": reference_valley,
        "symmetry_adapted_projectors": projector_summary,
        "valley_preserving_representations": rep_summary,
        "valley_sewing_matrices": sewing_summary,
        "valley_preserving_character_diagnostics": char_summary,
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
        "orbit": report.get("orbit"),
        "reference_valley": report.get("reference_valley"),
        "symmetry_adapted_projectors": report.get("symmetry_adapted_projectors"),
        "valley_preserving_representations":
            report.get("valley_preserving_representations"),
        "valley_sewing_matrices": report.get("valley_sewing_matrices"),
        "valley_preserving_character_diagnostics":
            report.get("valley_preserving_character_diagnostics"),
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
