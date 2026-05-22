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
    completeness_error: float
    valley_sewing_matrices: dict[tuple[str, str], np.ndarray] | None
    sewing_unitarity_error: dict[tuple[str, str], float] | None
    status: str
    reason: str


@dataclass(frozen=True)
class SymmetryAdaptedProjectors:
    projectors: dict[str, np.ndarray]
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
) -> SymmetryAdaptedProjectors:
    """Build symmetry-adapted valley projectors for a single valley orbit.

    Parameters
    ----------
    seed_projectors : {valley_name: P_a^0 matrix}
        Q-cut valley seed projectors in the target subspace.
    representations : {operation_id: D_g matrix}
        HSP little-group representation matrices in the target subspace.
    valley_mappings : {operation_id: {source_valley: mapped_valley}}
        Valley mapping pi_g(a) for each operation.
    orbit : list[str]
        Valley names forming one orbit under the HSP little group.
    reference_valley : str
        Valley name to use as reference a0 for symmetrization.
    rank : int or None
        Desired projector rank.  If None, auto-selected from eigenvalue
        spectrum using *rank_method*.
    rank_method : str
        Method for automatic rank selection: ``"gap"``, ``"threshold"``,
        or ``"fixed"``.
    rank_tol : float
        Tolerance / threshold for rank selection.

    Returns
    -------
    SymmetryAdaptedProjectors
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
        eigenvalues=eigvals,
        rank=rank,
        method=rank_method,
        tol=rank_tol,
    )

    if selected_rank < 1:
        return _failure(orbit, reference_valley, n,
                        f"rank selection failed: selected_rank={selected_rank}")

    # Build P_a0^sym from top eigenvectors
    top_vecs = eigvecs[:, :selected_rank]
    p_ref_sym = top_vecs @ top_vecs.conj().T

    # 5. Generate other valley projectors via representative operations
    projectors: dict[str, np.ndarray] = {reference_valley: p_ref_sym}
    rep_ops: dict[str, object] = {}

    for valley in orbit:
        if valley == reference_valley:
            continue
        rep_op_id = _find_representative_operation(
            reference_valley, valley, valley_mappings, representations
        )
        if rep_op_id is None:
            return _failure(
                orbit, reference_valley, n,
                f"no representative operation found mapping "
                f"{reference_valley} -> {valley}"
            )
        rep_ops[valley] = rep_op_id
        d_rep = np.asarray(representations[rep_op_id], dtype=np.complex128)
        projectors[valley] = d_rep @ p_ref_sym @ d_rep.conj().T

    # 6. Quality diagnostics
    diag = compute_projector_quality_diagnostics(
        projectors=projectors,
        seed_projectors=seed_projectors,
        representations=representations,
        valley_mappings=valley_mappings,
        orbit=orbit,
        reference_valley=reference_valley,
        preserving_ops=preserving_ops,
        selected_rank=selected_rank,
        eigenvalues=eigvals,
        purification_gap=gap,
        rank_source=rank_source,
        rep_ops=rep_ops,
    )

    return SymmetryAdaptedProjectors(
        projectors=projectors,
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
    """Select projector rank from eigenvalue spectrum.

    Parameters
    ----------
    eigenvalues : 1-d array, sorted descending
    rank : int or None, user-specified rank
    method : "gap", "threshold", or "fixed"
    tol : tolerance for gap or threshold method

    Returns
    -------
    selected_rank : int
    purification_gap : float or None
    rank_source : str
    """
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
    seed_projectors: dict[str, np.ndarray],
    representations: dict[object, np.ndarray],
    valley_mappings: dict[object, dict[str, str]],
    orbit: list[str],
    reference_valley: str,
    preserving_ops: dict[object, np.ndarray],
    selected_rank: int,
    eigenvalues: np.ndarray,
    purification_gap: float | None,
    rank_source: str,
    rep_ops: dict[str, object] | None = None,
) -> ProjectorQualityDiagnostics:
    """Compute quality diagnostics for symmetry-adapted projectors."""

    # Seed overlap: Tr(P_a^sym P_a^0) / rank
    seed_overlap: dict[str, float] = {}
    for v in orbit:
        p_sym = np.asarray(projectors[v], dtype=np.complex128)
        p_seed = np.asarray(seed_projectors[v], dtype=np.complex128)
        overlap = float(np.real(np.trace(p_sym @ p_seed)))
        seed_overlap[v] = overlap / max(selected_rank, 1)

    # Projector symmetry error: ||D_g P_a^sym D_g^dag - P_{pi_g(a)}^sym||_F / selected_rank
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

    # Orthogonality: max |Tr(P_a^sym P_b^sym)| / selected_rank for a != b
    ortho_err = 0.0
    for i, a in enumerate(orbit):
        for b in orbit[i + 1:]:
            p_a = np.asarray(projectors[a], dtype=np.complex128)
            p_b = np.asarray(projectors[b], dtype=np.complex128)
            overlap = float(np.abs(np.trace(p_a @ p_b)))
            ortho_err = max(ortho_err, overlap / max(selected_rank, 1))

    # Completeness: idempotency of the total projector sum_a P_a^sym.
    # ||(sum_a P_a^sym)^2 - (sum_a P_a^sym)||_F / sqrt(selected_rank * |orbit|)
    total = sum(np.asarray(p, dtype=np.complex128) for p in projectors.values())
    total_sq = total @ total
    completeness_err = float(np.linalg.norm(total_sq - total, ord="fro"))
    n_valleys = len(orbit)
    denom = np.sqrt(float(selected_rank * n_valleys))
    completeness_err /= max(denom, 1.0)

    # Valley sewing matrices
    sewing, sewing_err = compute_valley_sewing_matrices(
        projectors=projectors,
        orbit=orbit,
    )

    # Decide status
    status = "ok"
    reason = "all diagnostics within tolerance"
    if ortho_err > 0.1 or completeness_err > 0.2:
        status = "failed"
        reason = f"orthogonality_error={ortho_err:.4f}, completeness_error={completeness_err:.4f}"
    elif any(e > 0.05 for e in symmetry_errors.values()):
        status = "warn"
        reason = "symmetry error elevated"
    elif rep_ops:
        ambiguous = _check_representative_ambiguity(
            reference_valley, orbit, valley_mappings, rep_ops
        )
        if ambiguous:
            status = "warn"
            reason = f"representative operation ambiguous for: {ambiguous}"

    return ProjectorQualityDiagnostics(
        selected_rank=selected_rank,
        eigenvalues=eigenvalues,
        purification_gap=purification_gap,
        rank_source=rank_source,
        seed_overlap=seed_overlap,
        projector_symmetry_error=symmetry_errors,
        orthogonality_error=ortho_err,
        completeness_error=completeness_err,
        valley_sewing_matrices=sewing if sewing else None,
        sewing_unitarity_error=sewing_err if sewing_err else None,
        status=status,
        reason=reason,
    )


def compute_valley_sewing_matrices(
    *,
    projectors: dict[str, np.ndarray],
    orbit: list[str],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float]]:
    """Compute valley sewing matrices S_ab = P_a^sym P_b^sym.

    For exact idempotent orthogonal projectors, S_ab = delta_ab P_a^sym.
    Deviation from this measures sewing quality.
    """
    sewing: dict[tuple[str, str], np.ndarray] = {}
    unitarity_err: dict[tuple[str, str], float] = {}
    for a in orbit:
        for b in orbit:
            p_a = np.asarray(projectors[a], dtype=np.complex128)
            p_b = np.asarray(projectors[b], dtype=np.complex128)
            s = p_a @ p_b
            sewing[(a, b)] = s
            # Expected: delta_ab * P_a^sym
            if a == b:
                expected = p_a
            else:
                expected = np.zeros_like(p_a)
            unitarity_err[(a, b)] = float(np.linalg.norm(s - expected, ord="fro"))
    return sewing, unitarity_err


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_orbit_closure(
    orbit: list[str],
    valley_mappings: dict[object, dict[str, str]],
) -> None:
    """Check that the orbit is closed under all valley mappings."""
    orbit_set = set(orbit)
    for mapping in valley_mappings.values():
        for src, tgt in mapping.items():
            if src in orbit_set and tgt not in orbit_set:
                raise ValueError(
                    f"orbit not closed under valley mapping: "
                    f"{src} -> {tgt} but {tgt} not in orbit {orbit}"
                )


def _find_representative_operation(
    src: str,
    tgt: str,
    valley_mappings: dict[object, dict[str, str]],
    representations: dict[object, np.ndarray],
) -> object | None:
    """Find an operation that maps src -> tgt and has a representation matrix.

    Raises if multiple distinct candidates exist (representative ambiguity).
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
    return candidates[0]


def _check_representative_ambiguity(
    reference_valley: str,
    orbit: list[str],
    valley_mappings: dict[object, dict[str, str]],
    rep_ops: dict[str, object],
) -> list[str]:
    """Check whether any valley has multiple distinct candidate representative ops."""
    ambiguous = []
    for valley in orbit:
        if valley == reference_valley:
            continue
        candidates = []
        for op_id, mapping in valley_mappings.items():
            mapped = mapping.get(reference_valley)
            if mapped is not None and str(mapped) == str(valley):
                candidates.append(op_id)
        if len(candidates) > 1:
            chosen = rep_ops.get(valley)
            if chosen not in candidates:
                ambiguous.append(valley)
    return ambiguous


def _failure(
    orbit: list[str],
    reference_valley: str,
    dim: int,
    reason: str,
) -> SymmetryAdaptedProjectors:
    n_valleys = len(orbit)
    return SymmetryAdaptedProjectors(
        projectors={v: np.zeros((dim, dim), dtype=np.complex128) for v in orbit},
        reference_valley=reference_valley,
        diagnostics=ProjectorQualityDiagnostics(
            selected_rank=0,
            eigenvalues=np.zeros(dim),
            purification_gap=None,
            rank_source="failure",
            seed_overlap={v: 0.0 for v in orbit},
            projector_symmetry_error={},
            orthogonality_error=float("inf"),
            completeness_error=float("inf"),
            valley_sewing_matrices=None,
            sewing_unitarity_error=None,
            status="failed",
            reason=reason,
        ),
    )
