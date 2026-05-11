from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneWaveRepresentationResult:
    matrix: np.ndarray
    mapping: np.ndarray
    mapping_miss_count: int


def spin_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    vec = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("rotation axis must be non-zero")
    nx, ny, nz = vec / norm
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    sigma = nx * sigma_x + ny * sigma_y + nz * sigma_z
    return np.cos(angle / 2.0) * np.eye(2) - 1.0j * np.sin(angle / 2.0) * sigma


def unitarity_deviation(matrix: np.ndarray) -> float:
    mat = np.asarray(matrix, dtype=np.complex128)
    return float(np.linalg.norm(mat.conj().T @ mat - np.eye(mat.shape[1])))


def build_plane_wave_representation(
    coefficients: np.ndarray,
    q_cart: np.ndarray,
    rotation_cart: np.ndarray,
    translation_cart: np.ndarray,
    *,
    spin_rotation: np.ndarray | None = None,
    tolerance: float = 1e-8,
) -> PlaneWaveRepresentationResult:
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    q = np.asarray(q_cart, dtype=float)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape [nb,nspinor,nG]")
    if q.shape != (coeffs.shape[2], 3):
        raise ValueError("q_cart must have shape [nG,3] matching coefficients")
    rot = np.asarray(rotation_cart, dtype=float)
    trans = np.asarray(translation_cart, dtype=float)
    n_bands, n_spinor, n_g = coeffs.shape
    if spin_rotation is None:
        spin = np.eye(n_spinor, dtype=np.complex128)
    else:
        spin = np.asarray(spin_rotation, dtype=np.complex128)
        if spin.shape != (n_spinor, n_spinor):
            raise ValueError("spin_rotation shape must match nspinor")

    mapping = np.full(n_g, -1, dtype=int)
    for source_idx, q_source in enumerate(q):
        q_target = rot @ q_source
        distances = np.linalg.norm(q - q_target[None, :], axis=1)
        target_idx = int(np.argmin(distances))
        if distances[target_idx] <= tolerance:
            mapping[source_idx] = target_idx

    transformed = np.zeros_like(coeffs)
    for source_idx, target_idx in enumerate(mapping):
        if target_idx < 0:
            continue
        q_target = rot @ q[source_idx]
        phase = np.exp(-1.0j * float(q_target @ trans))
        for band in range(n_bands):
            transformed[band, :, target_idx] += phase * (spin @ coeffs[band, :, source_idx])

    flat_original = coeffs.reshape(n_bands, -1)
    flat_transformed = transformed.reshape(n_bands, -1)
    matrix = flat_original.conj() @ flat_transformed.T
    return PlaneWaveRepresentationResult(
        matrix=matrix,
        mapping=mapping,
        mapping_miss_count=int(np.sum(mapping < 0)),
    )
