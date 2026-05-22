"""Toy-only construction of symmetry-adapted valley projectors.

This module builds symmetry-adapted valley projectors P_a^sym from q-cut
valley seed projectors P_a^0.  It does NOT integrate into the production
analyze_hsp workflow.  All inputs are synthetic matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ProjectorQualityDiagnostics:
    selected_rank: int
    eigenvalues: np.ndarray
    purification_gap: float | None
    rank_source: str
    seed_overlap: dict[str, float]
    projector_symmetry_error: dict[str, float]
    orthogonality_error: float
    total_projector_idempotency_error: float
    completeness_error: float | None
    completeness_source: str
    projector_overlap_matrices: dict[tuple[str, str], np.ndarray] | None
    projector_overlap_deviation: dict[tuple[str, str], float] | None
    valley_sewing_matrices: dict[tuple[str, object], np.ndarray] | None
    sewing_unitarity_error: dict[tuple[str, object], float] | None
    status: str
    reason: str


@dataclass(frozen=True)
class SymmetryAdaptedProjectors:
    projectors: dict[str, np.ndarray]
    eigenvectors: dict[str, np.ndarray]
    reference_valley: str
    diagnostics: ProjectorQualityDiagnostics


def build_symmetry_adapted_projectors_for_orbit(
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
    seed_overlap_warn_tol: float = 0.8,
    seed_overlap_fail_tol: float = 0.5,
) -> SymmetryAdaptedProjectors:
    """Build symmetry-adapted valley projectors for a single valley orbit.

    Parameters
    ----------
    seed_projectors : {valley_name: P_a^0 matrix}
    representations : {operation_id: D_g matrix}
    valley_mappings : {operation_id: {source: mapped}}
    orbit : list[str]
    reference_valley : str
    rank : int or None
    rank_method : "gap", "threshold", or "fixed"
    rank_tol : float
    expected_total_projector : ndarray or None
        Expected total projector I_expected = sum_a P_a^sym.
        Used for completeness_error.  If None, completeness is not_evaluated.
    """
    if reference_valley not in orbit:
        raise ValueError(f"reference_valley {reference_valley} not in orbit {orbit}")
    for v in orbit:
        if v not in seed_projectors:
            raise ValueError(f"seed projector for {v} not provided")

    _validate_orbit_closure(orbit, valley_mappings)
    n = next(iter(seed_projectors.values())).shape[0]

    # 1. Identify valley-preserving subgroup for reference valley
    preserving_ops: dict[object, np.ndarray] = {}
    for op_id, mapping in valley_mappings.items():
        if op_id not in representations:
            continue
        mapped = mapping.get(reference_valley)
        if mapped is not None and str(mapped) == str(reference_valley):
            preserving_ops[op_id] = np.asarray(representations[op_id], dtype=np.complex128)

    if not preserving_ops:
        return _failure(orbit, reference_valley, n,
                        "no valley-preserving operation found for reference valley")

    # 2. Reference seed symmetrization
    p_ref = np.asarray(seed_projectors[reference_valley], dtype=np.complex128)
    p_avg = np.zeros_like(p_ref)
    for d_h in preserving_ops.values():
        p_avg += d_h @ p_ref @ d_h.conj().T
    p_avg /= len(preserving_ops)

    # 3. Hermitian symmetrization
    p_avg = (p_avg + p_avg.conj().T) / 2.0

    # 4. Spectral decomposition and rank selection
    eigvals, eigvecs = np.linalg.eigh(p_avg)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    selected_rank, gap, rank_source = select_projector_rank(
        eigenvalues=eigvals, rank=rank, method=rank_method, tol=rank_tol,
    )

    if selected_rank < 1:
        return _failure(orbit, reference_valley, n,
                        f"rank selection failed: selected_rank={selected_rank}",
                        rank_source=rank_source, purification_gap=gap,
                        eigenvalues=eigvals)

    if rank_source == "gap_insufficient":
        return _failure(orbit, reference_valley, n,
                        f"failed_rank_selection: rank gap insufficient (max_gap={gap:.4f} < tol={rank_tol})",
                        rank_source=rank_source, purification_gap=gap,
                        eigenvalues=eigvals)

    # Build P_a0^sym and extract eigenvectors U_a0
    top_vecs = eigvecs[:, :selected_rank]
    p_ref_sym = top_vecs @ top_vecs.conj().T
    eigenvectors: dict[str, np.ndarray] = {reference_valley: top_vecs.copy()}

    # 5. Generate other valley projectors with explicit representative checks
    projectors: dict[str, np.ndarray] = {reference_valley: p_ref_sym}
    rep_ops: dict[str, object] = {}

    for valley in orbit:
        if valley == reference_valley:
            continue
        result = _resolve_representative_operation(
            reference_valley, valley, valley_mappings, representations
        )
        if result is None:
            return _failure(orbit, reference_valley, n,
                            f"no representative operation found mapping "
                            f"{reference_valley} -> {valley}")
        if isinstance(result, list):
            return _failure(orbit, reference_valley, n,
                            f"ambiguous representative operation for "
                            f"{reference_valley} -> {valley}: candidates={result}")
        rep_op_id = result
        rep_ops[valley] = rep_op_id
        d_rep = np.asarray(representations[rep_op_id], dtype=np.complex128)
        u_a = d_rep @ top_vecs
        projectors[valley] = u_a @ u_a.conj().T
        eigenvectors[valley] = u_a

    # 6. Quality diagnostics
    diag = compute_projector_quality_diagnostics(
        projectors=projectors,
        eigenvectors=eigenvectors,
        seed_projectors=seed_projectors,
        representations=representations,
        valley_mappings=valley_mappings,
        orbit=orbit,
        reference_valley=reference_valley,
        selected_rank=selected_rank,
        eigenvalues=eigvals,
        purification_gap=gap,
        rank_source=rank_source,
        expected_total_projector=expected_total_projector,
        seed_overlap_warn_tol=seed_overlap_warn_tol,
        seed_overlap_fail_tol=seed_overlap_fail_tol,
    )

    return SymmetryAdaptedProjectors(
        projectors=projectors,
        eigenvectors=eigenvectors,
        reference_valley=reference_valley,
        diagnostics=diag,
    )


def select_projector_rank(
    *,
    eigenvalues: np.ndarray,
    rank: int | None = None,
    method: str = "gap",
    tol: float = 0.5,
) -> tuple[int, float | None, str]:
    """Select projector rank from eigenvalue spectrum (sorted descending)."""
    if rank is not None:
        return rank, None, "user_specified"
    if method == "fixed":
        return len(eigenvalues), None, "fixed_full_rank"
    if method == "threshold":
        count = int(np.sum(eigenvalues > tol))
        if count == 0:
            count = 1
        gap = float(eigenvalues[count - 1] - eigenvalues[count]) if count < len(eigenvalues) else None
        return count, gap, "threshold"
    # Default: gap method
    gaps = np.diff(eigenvalues)
    if len(gaps) == 0:
        return len(eigenvalues), None, "gap_single_eigenvalue"
    max_gap_idx = int(np.argmax(np.abs(gaps)))
    gap_value = float(abs(gaps[max_gap_idx]))
    if gap_value < tol:
        return len(eigenvalues), gap_value, "gap_insufficient"
    return max_gap_idx + 1, gap_value, "gap"


def compute_projector_quality_diagnostics(
    *,
    projectors: dict[str, np.ndarray],
    eigenvectors: dict[str, np.ndarray],
    seed_projectors: dict[str, np.ndarray],
    representations: dict[object, np.ndarray],
    valley_mappings: dict[object, dict[str, str]],
    orbit: list[str],
    reference_valley: str,
    selected_rank: int,
    eigenvalues: np.ndarray,
    purification_gap: float | None,
    rank_source: str,
    expected_total_projector: np.ndarray | None = None,
    seed_overlap_warn_tol: float = 0.8,
    seed_overlap_fail_tol: float = 0.5,
) -> ProjectorQualityDiagnostics:
    """Compute quality diagnostics for symmetry-adapted projectors."""

    # Seed overlap
    seed_overlap: dict[str, float] = {}
    for v in orbit:
        p_sym = np.asarray(projectors[v], dtype=np.complex128)
        p_seed = np.asarray(seed_projectors[v], dtype=np.complex128)
        overlap = float(np.real(np.trace(p_sym @ p_seed)))
        seed_overlap[v] = overlap / max(selected_rank, 1)

    # Projector symmetry error
    symmetry_errors: dict[str, float] = {}
    for op_id, mapping in valley_mappings.items():
        if op_id not in representations:
            continue
        d_g = np.asarray(representations[op_id], dtype=np.complex128)
        for src in orbit:
            tgt = mapping.get(src)
            if tgt is None or tgt not in projectors:
                continue
            p_src = np.asarray(projectors[src], dtype=np.complex128)
            p_tgt = np.asarray(projectors[tgt], dtype=np.complex128)
            transformed = d_g @ p_src @ d_g.conj().T
            err = float(np.linalg.norm(transformed - p_tgt, ord="fro"))
            key = f"op_{op_id}_{src}->{tgt}"
            symmetry_errors[key] = err / max(selected_rank, 1)

    # Orthogonality
    ortho_err = 0.0
    for i, a in enumerate(orbit):
        for b in orbit[i + 1:]:
            p_a = np.asarray(projectors[a], dtype=np.complex128)
            p_b = np.asarray(projectors[b], dtype=np.complex128)
            overlap = float(np.abs(np.trace(p_a @ p_b)))
            ortho_err = max(ortho_err, overlap / max(selected_rank, 1))

    # Total projector idempotency: ||(sum P_a^sym)^2 - sum P_a^sym||_F / sqrt(rank*|orbit|)
    total = sum(np.asarray(p, dtype=np.complex128) for p in projectors.values())
    total_sq = total @ total
    idempotency_err = float(np.linalg.norm(total_sq - total, ord="fro"))
    n_valleys = len(orbit)
    idempotency_err /= max(np.sqrt(float(selected_rank * n_valleys)), 1.0)

    # Completeness: ||sum P_a^sym - I_expected||_F / sqrt(dim)
    dim = next(iter(projectors.values())).shape[0]
    completeness_err: float | None = None
    completeness_source: str = "not_evaluated"
    if expected_total_projector is not None:
        expected = np.asarray(expected_total_projector, dtype=np.complex128)
        completeness_err = float(np.linalg.norm(total - expected, ord="fro"))
        completeness_err /= max(np.sqrt(float(dim)), 1.0)
        completeness_source = "expected_total_projector"
    else:
        # Fallback: check trace consistency
        expected_trace = selected_rank * n_valleys
        actual_trace = float(np.real(np.trace(total)))
        if abs(actual_trace - expected_trace) > 1e-3 * expected_trace:
            completeness_err = abs(actual_trace - expected_trace) / max(float(dim), 1.0)
            completeness_source = "trace_mismatch"

    # Projector overlap matrices (replaces old "sewing" naming)
    overlap_matrices, overlap_deviation = _compute_projector_overlap(
        projectors=projectors, orbit=orbit
    )

    # True valley sewing matrices: B_{ba}(g) = U_b^dag D_g U_a, b = pi_g(a)
    sewing, sewing_err = _compute_valley_sewing(
        eigenvectors=eigenvectors,
        representations=representations,
        valley_mappings=valley_mappings,
        orbit=orbit,
    )

    # Status — ordered by priority
    status = "ok"
    reason = "all diagnostics within tolerance"

    min_seed_overlap = min(seed_overlap.values()) if seed_overlap else 0.0
    worst_valley = min(seed_overlap, key=lambda v: seed_overlap[v]) if seed_overlap else "?"

    # 1. Hard failures: orthogonality / idempotency / completeness
    if ortho_err > 0.1 or idempotency_err > 0.2:
        status = "failed"
        reason = f"orthogonality_error={ortho_err:.4f}, total_projector_idempotency_error={idempotency_err:.4f}"
    elif completeness_err is not None and completeness_err > 0.2:
        status = "failed"
        reason = f"completeness_error={completeness_err:.4f}"
    elif min_seed_overlap <= seed_overlap_fail_tol:
        status = "failed"
        reason = (
            f"low seed overlap: {worst_valley}={min_seed_overlap:.4f} "
            f"<= fail_tol={seed_overlap_fail_tol:.4f}"
        )
    # 2. Warnings
    elif any(e > 0.05 for e in symmetry_errors.values()):
        status = "warn"
        reason = "symmetry error elevated"
    elif min_seed_overlap < seed_overlap_warn_tol:
        status = "warn"
        reason = (
            f"low seed overlap: {worst_valley}={min_seed_overlap:.4f} "
            f"< warn_tol={seed_overlap_warn_tol:.4f}"
        )

    return ProjectorQualityDiagnostics(
        selected_rank=selected_rank,
        eigenvalues=eigenvalues,
        purification_gap=purification_gap,
        rank_source=rank_source,
        seed_overlap=seed_overlap,
        projector_symmetry_error=symmetry_errors,
        orthogonality_error=ortho_err,
        total_projector_idempotency_error=idempotency_err,
        completeness_error=completeness_err,
        completeness_source=completeness_source,
        projector_overlap_matrices=overlap_matrices if overlap_matrices else None,
        projector_overlap_deviation=overlap_deviation if overlap_deviation else None,
        valley_sewing_matrices=sewing if sewing else None,
        sewing_unitarity_error=sewing_err if sewing_err else None,
        status=status,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Internal: sewing and overlap
# ---------------------------------------------------------------------------

def _compute_projector_overlap(
    *,
    projectors: dict[str, np.ndarray],
    orbit: list[str],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float]]:
    """Projector overlap O_ab = P_a^sym P_b^sym and deviation from delta_ab P_a."""
    matrices: dict[tuple[str, str], np.ndarray] = {}
    deviation: dict[tuple[str, str], float] = {}
    for a in orbit:
        for b in orbit:
            p_a = np.asarray(projectors[a], dtype=np.complex128)
            p_b = np.asarray(projectors[b], dtype=np.complex128)
            o = p_a @ p_b
            matrices[(a, b)] = o
            expected = p_a if a == b else np.zeros_like(p_a)
            deviation[(a, b)] = float(np.linalg.norm(o - expected, ord="fro"))
    return matrices, deviation


def _compute_valley_sewing(
    *,
    eigenvectors: dict[str, np.ndarray],
    representations: dict[object, np.ndarray],
    valley_mappings: dict[object, dict[str, str]],
    orbit: list[str],
) -> tuple[dict[tuple[str, object], np.ndarray], dict[tuple[str, object], float]]:
    """Valley sewing matrices B_{ba}(g) = U_b^dag D_g U_a for b = pi_g(a).

    These are r x r matrices where r = selected_rank.  For exact symmetry-
    adapted projectors, each B_{ba}(g) should be unitary.
    """
    sewing: dict[tuple[str, object], np.ndarray] = {}
    unitarity_err: dict[tuple[str, object], float] = {}
    orbit_set = set(orbit)

    for op_id, d_g in representations.items():
        mapping = valley_mappings.get(op_id, {})
        for src in orbit:
            tgt = mapping.get(src)
            if tgt is None or str(tgt) not in orbit_set:
                continue
            if src not in eigenvectors or str(tgt) not in eigenvectors:
                continue
            u_src = np.asarray(eigenvectors[src], dtype=np.complex128)
            u_tgt = np.asarray(eigenvectors[str(tgt)], dtype=np.complex128)
            d_mat = np.asarray(d_g, dtype=np.complex128)
            b = u_tgt.conj().T @ d_mat @ u_src
            key = (str(tgt), op_id)
            sewing[key] = b
            # Unitarity check: B^dag B should be identity
            r = b.shape[0]
            unitarity_err[key] = float(
                np.linalg.norm(b.conj().T @ b - np.eye(r, dtype=np.complex128), ord="fro")
            )
    return sewing, unitarity_err


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_orbit_closure(
    orbit: list[str],
    valley_mappings: dict[object, dict[str, str]],
) -> None:
    orbit_set = set(orbit)
    for mapping in valley_mappings.values():
        for src, tgt in mapping.items():
            if src in orbit_set and tgt not in orbit_set:
                raise ValueError(
                    f"orbit not closed: {src} -> {tgt} but {tgt} not in {orbit}"
                )


def _resolve_representative_operation(
    src: str,
    tgt: str,
    valley_mappings: dict[object, dict[str, str]],
    representations: dict[object, np.ndarray],
) -> object | list[object] | None:
    """Find an operation mapping src -> tgt with a representation.

    Returns
    -------
    op_id : unique candidate
    None : no candidate
    list[op_id] : multiple candidates (ambiguity)
    """
    candidates = []
    for op_id, mapping in valley_mappings.items():
        if op_id not in representations:
            continue
        mapped = mapping.get(src)
        if mapped is not None and str(mapped) == str(tgt):
            candidates.append(op_id)
    if not candidates:
        return None
    if len(candidates) > 1:
        return candidates
    return candidates[0]


def _failure(
    orbit: list[str],
    reference_valley: str,
    dim: int,
    reason: str,
    *,
    rank_source: str = "failure",
    purification_gap: float | None = None,
    eigenvalues: np.ndarray | None = None,
) -> SymmetryAdaptedProjectors:
    eig = eigenvalues if eigenvalues is not None else np.zeros(dim)
    return SymmetryAdaptedProjectors(
        projectors={v: np.zeros((dim, dim), dtype=np.complex128) for v in orbit},
        eigenvectors={v: np.zeros((dim, 0), dtype=np.complex128) for v in orbit},
        reference_valley=reference_valley,
        diagnostics=ProjectorQualityDiagnostics(
            selected_rank=0,
            eigenvalues=eig,
            purification_gap=purification_gap,
            rank_source=rank_source,
            seed_overlap={v: 0.0 for v in orbit},
            projector_symmetry_error={},
            orthogonality_error=float("inf"),
            total_projector_idempotency_error=float("inf"),
            completeness_error=None,
            completeness_source="not_evaluated",
            projector_overlap_matrices=None,
            projector_overlap_deviation=None,
            valley_sewing_matrices=None,
            sewing_unitarity_error=None,
            status="failed",
            reason=reason,
        ),
    )
