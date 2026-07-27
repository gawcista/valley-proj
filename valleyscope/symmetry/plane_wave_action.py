from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valleyscope.io.wavefunction_convention import canonical_identity


RECIPROCAL_GRID_ACTION_CONVENTION = "q_target = rotation_cart @ q_source"
DEFAULT_RECIPROCAL_GRID_MAPPING_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class ReciprocalGridMapResult:
    mapping: np.ndarray
    mapping_miss_count: int


@dataclass(frozen=True)
class ReciprocalGridPermutationValidation:
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class PlaneWaveRepresentationResult:
    matrix: np.ndarray
    mapping: np.ndarray
    mapping_miss_count: int
    norm_preservation_residual: float
    relative_norm_preservation_residual: float


@dataclass(frozen=True)
class PlaneWaveActionResult:
    transformed_coefficients: np.ndarray
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
    action = apply_plane_wave_action(
        coefficients,
        q_cart,
        rotation_cart,
        translation_cart,
        spin_rotation=spin_rotation,
        tolerance=tolerance,
    )
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    flat_original = coeffs.reshape(coeffs.shape[0], -1)
    flat_transformed = action.transformed_coefficients.reshape(
        coeffs.shape[0], -1
    )
    matrix = flat_original.conj() @ flat_transformed.T
    original_norm = float(np.linalg.norm(flat_original))
    transformed_norm = float(np.linalg.norm(flat_transformed))
    absolute_norm_residual = abs(transformed_norm - original_norm)
    relative_norm_residual = (
        absolute_norm_residual / original_norm
        if original_norm > 0.0
        else absolute_norm_residual
    )
    return PlaneWaveRepresentationResult(
        matrix=matrix,
        mapping=action.mapping,
        mapping_miss_count=action.mapping_miss_count,
        norm_preservation_residual=absolute_norm_residual,
        relative_norm_preservation_residual=relative_norm_residual,
    )


def apply_plane_wave_action(
    coefficients: np.ndarray,
    q_cart: np.ndarray,
    rotation_cart: np.ndarray,
    translation_cart: np.ndarray,
    *,
    spin_rotation: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> PlaneWaveActionResult:
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

    reciprocal_grid_map = build_reciprocal_grid_map(
        q,
        rot,
        tolerance=tolerance,
    )
    mapping = reciprocal_grid_map.mapping
    q_rotated = np.empty_like(q)
    for source_idx, q_source in enumerate(q):
        q_rotated[source_idx] = rot @ q_source

    transformed = np.zeros_like(coeffs)
    for source_idx, target_idx in enumerate(mapping):
        if target_idx < 0:
            continue
        phase = np.exp(-1.0j * float(q_rotated[source_idx] @ trans))
        for band in range(n_bands):
            transformed[band, :, target_idx] += phase * (spin @ coeffs[band, :, source_idx])

    return PlaneWaveActionResult(
        transformed_coefficients=transformed,
        mapping=mapping,
        mapping_miss_count=reciprocal_grid_map.mapping_miss_count,
    )


def build_reciprocal_grid_map(
    q_cart: np.ndarray,
    rotation_cart: np.ndarray,
    *,
    tolerance: float = DEFAULT_RECIPROCAL_GRID_MAPPING_TOLERANCE,
) -> ReciprocalGridMapResult:
    """Map every source reciprocal-grid vector under one spatial operation."""
    q = np.asarray(q_cart, dtype=float)
    rotation = np.asarray(rotation_cart, dtype=float)
    if q.ndim != 2 or q.shape[1:] != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("q_cart must be a finite array with shape [nG,3]")
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
    ):
        raise ValueError("rotation_cart must be a finite 3x3 matrix")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    lookup = _q_vector_lookup(q, tolerance)
    mapping = np.full(len(q), -1, dtype=int)
    for source_idx, q_source in enumerate(q):
        mapping[source_idx] = _lookup_q_vector(
            rotation @ q_source,
            q,
            lookup,
            tolerance,
        )
    return ReciprocalGridMapResult(
        mapping=mapping,
        mapping_miss_count=int(np.count_nonzero(mapping < 0)),
    )


def validate_reciprocal_grid_permutation(
    mapping: object,
    *,
    dimension: int,
) -> ReciprocalGridPermutationValidation:
    """Require an exact permutation of ``range(dimension)``."""
    reasons: list[str] = []
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 0
    ):
        return ReciprocalGridPermutationValidation(
            "blocked",
            ("reciprocal_grid_dimension_malformed",),
        )

    if isinstance(mapping, np.ndarray):
        if mapping.ndim != 1 or mapping.dtype.kind not in "iu":
            candidates: object = None
        else:
            candidates = mapping.tolist()
    elif isinstance(mapping, (list, tuple)):
        candidates = list(mapping)
    else:
        candidates = None
    if candidates is None:
        return ReciprocalGridPermutationValidation(
            "blocked",
            ("mapping_collection_malformed",),
        )

    if len(candidates) != dimension:
        reasons.append("source_coverage_incomplete")
    valid_indices: list[int] = []
    for value in candidates:
        if not isinstance(value, int) or isinstance(value, bool):
            reasons.append("mapping_index_malformed")
            continue
        if value < 0:
            reasons.append("source_coverage_incomplete")
            continue
        if value >= dimension:
            reasons.append("target_index_out_of_range")
            continue
        valid_indices.append(value)
    if len(set(valid_indices)) != len(valid_indices):
        reasons.append("target_index_collision")
    if set(valid_indices) != set(range(dimension)):
        reasons.append("target_coverage_incomplete")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return ReciprocalGridPermutationValidation(
        "passed" if not unique_reasons else "blocked",
        unique_reasons,
    )


def reciprocal_grid_identity(q_cart: np.ndarray) -> str:
    """Return an order-sensitive identity for one finite Cartesian q grid."""
    q = np.asarray(q_cart, dtype=float)
    if q.ndim != 2 or q.shape[1:] != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("q_cart must be a finite array with shape [nG,3]")
    return canonical_identity(
        {
            "coordinate_system": "cartesian_inverse_angstrom",
            "dimension": int(len(q)),
            "q_cart": q.tolist(),
        }
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
