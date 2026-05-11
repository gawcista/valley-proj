from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwoValleyBasisResult:
    transform: np.ndarray
    eta: np.ndarray
    s_matrix: np.ndarray
    v_matrix: np.ndarray


@dataclass(frozen=True)
class MultiValleyDiagnostic:
    stably_separable: bool
    reason: str
    eigenvalues: dict[str, np.ndarray]
    max_commutator_norm: float


def _projector_matrix(coefficients: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape [nb,nspinor,nG]")
    flat = coeffs.reshape(coeffs.shape[0], -1)
    expanded_mask = np.tile(mask.astype(bool), coeffs.shape[1])
    selected = flat[:, expanded_mask]
    return selected @ selected.conj().T


def build_two_valley_adapted_basis(
    coefficients: np.ndarray,
    sector_masks: dict[str, np.ndarray],
    first_sector: str,
    second_sector: str,
) -> TwoValleyBasisResult:
    if first_sector not in sector_masks or second_sector not in sector_masks:
        raise ValueError("Both valley sectors must be present")
    p_first = _projector_matrix(coefficients, sector_masks[first_sector])
    p_second = _projector_matrix(coefficients, sector_masks[second_sector])
    s_matrix = p_first + p_second
    v_matrix = p_first - p_second
    eigvals, eigvecs = np.linalg.eigh(v_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    s_diag = np.real(np.diag(eigvecs.conj().T @ s_matrix @ eigvecs))
    eta = np.divide(eigvals, s_diag, out=np.zeros_like(eigvals), where=np.abs(s_diag) > 1e-14)
    return TwoValleyBasisResult(transform=eigvecs, eta=eta, s_matrix=s_matrix, v_matrix=v_matrix)


def diagnose_multivalley_subspace(
    sector_matrices: dict[str, np.ndarray],
    *,
    eig_tol: float = 1e-6,
    commutator_tol: float = 1e-6,
) -> MultiValleyDiagnostic:
    eigenvalues: dict[str, np.ndarray] = {}
    max_non_idempotent = 0.0
    for name, matrix in sector_matrices.items():
        mat = np.asarray(matrix, dtype=np.complex128)
        eigenvalues[name] = np.linalg.eigvalsh(mat)
        max_non_idempotent = max(max_non_idempotent, float(np.linalg.norm(mat @ mat - mat)))
    names = list(sector_matrices)
    max_commutator = 0.0
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            a = np.asarray(sector_matrices[name_a], dtype=np.complex128)
            b = np.asarray(sector_matrices[name_b], dtype=np.complex128)
            max_commutator = max(max_commutator, float(np.linalg.norm(a @ b - b @ a)))
    if max_non_idempotent > eig_tol:
        return MultiValleyDiagnostic(False, "non_idempotent_sector_projector", eigenvalues, max_commutator)
    if max_commutator > commutator_tol:
        return MultiValleyDiagnostic(False, "non_commuting_sector_projectors", eigenvalues, max_commutator)
    return MultiValleyDiagnostic(True, "stably_separable", eigenvalues, max_commutator)
