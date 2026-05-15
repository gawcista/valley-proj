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
    tolerance: float = 1e-6,
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

    lookup = _q_vector_lookup(q, tolerance)
    mapping = np.full(n_g, -1, dtype=int)
    q_rotated = np.empty_like(q)
    for source_idx, q_source in enumerate(q):
        q_rotated[source_idx] = rot @ q_source
        mapping[source_idx] = _lookup_q_vector(q_rotated[source_idx], q, lookup, tolerance)

    transformed = np.zeros_like(coeffs)
    for source_idx, target_idx in enumerate(mapping):
        if target_idx < 0:
            continue
        phase = np.exp(-1.0j * float(q_rotated[source_idx] @ trans))
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


def _q_vector_lookup(q_cart: np.ndarray, tolerance: float) -> dict[tuple[int, int, int], list[int]]:
    lookup: dict[tuple[int, int, int], list[int]] = {}
    for idx, vector in enumerate(q_cart):
        lookup.setdefault(_q_key(vector, tolerance), []).append(idx)
    return lookup


def _lookup_q_vector(
    target: np.ndarray,
    q_cart: np.ndarray,
    lookup: dict[tuple[int, int, int], list[int]],
    tolerance: float,
) -> int:
    base_key = _q_key(target, tolerance)
    best_idx = -1
    best_distance_sq = float("inf")
    tol_sq = tolerance * tolerance
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                key = (base_key[0] + dx, base_key[1] + dy, base_key[2] + dz)
                for idx in lookup.get(key, []):
                    diff = q_cart[idx] - target
                    distance_sq = float(diff @ diff)
                    if distance_sq <= tol_sq and distance_sq < best_distance_sq:
                        best_idx = idx
                        best_distance_sq = distance_sq
    return best_idx


def _q_key(vector: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    scaled = np.rint(np.asarray(vector, dtype=float) / tolerance).astype(np.int64)
    return (int(scaled[0]), int(scaled[1]), int(scaled[2]))
