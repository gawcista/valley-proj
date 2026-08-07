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
    relative_target_subspace_residual: float = 0.0


@dataclass(frozen=True)
class PlaneWaveActionResult:
    transformed_coefficients: np.ndarray
    mapping: np.ndarray
    mapping_miss_count: int



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
    target_coefficients: np.ndarray | None = None,
    target_q_cart: np.ndarray | None = None,
    action: PlaneWaveActionResult | None = None,
) -> PlaneWaveRepresentationResult:
    if action is None:
        action = apply_plane_wave_action(
            coefficients,
            q_cart,
            rotation_cart,
            translation_cart,
            spin_rotation=spin_rotation,
            tolerance=tolerance,
            target_q_cart=target_q_cart,
        )
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    target = (
        coeffs
        if target_coefficients is None
        else np.asarray(target_coefficients, dtype=np.complex128)
    )
    flat_original = coeffs.reshape(coeffs.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    flat_transformed = action.transformed_coefficients.reshape(
        coeffs.shape[0], -1
    )
    matrix = flat_target.conj() @ flat_transformed.T
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
        relative_target_subspace_residual=float(
            np.linalg.norm(flat_transformed - matrix.T @ flat_target)
            / max(original_norm, np.finfo(float).tiny)
        ),
    )


def apply_plane_wave_action(
    coefficients: np.ndarray,
    q_cart: np.ndarray,
    rotation_cart: np.ndarray,
    translation_cart: np.ndarray,
    *,
    spin_rotation: np.ndarray | None = None,
    tolerance: float = 1e-6,
    target_q_cart: np.ndarray | None = None,
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

    target_q = q if target_q_cart is None else np.asarray(target_q_cart, dtype=float)
    reciprocal_grid_map = build_reciprocal_grid_map(
        q,
        rot,
        tolerance=tolerance,
        target_q_cart=target_q,
    )
    mapping = reciprocal_grid_map.mapping
    transformed = np.zeros(
        (n_bands, n_spinor, len(target_q)), dtype=np.complex128
    )
    valid = mapping >= 0
    if np.any(valid):
        source_idx = np.flatnonzero(valid)
        target_idx = mapping[source_idx]
        phase = np.exp(-1.0j * ((q[source_idx] @ rot.T) @ trans))
        # Explicit spinor-row accumulation mirrors the per-band reference
        # ``spin @ coeffs[band, :, source]`` exactly (ascending spinor index,
        # scalar complex multiply-accumulate), instead of a BLAS batched
        # matmul whose kernel summation order can differ in the last bit.
        spin_coeffs = np.zeros_like(coeffs[:, :, source_idx])
        for spinor_row in range(n_spinor):
            row = np.zeros_like(spin_coeffs[:, 0, :])
            for spinor_col in range(n_spinor):
                row = row + spin[spinor_row, spinor_col] * coeffs[
                    :, spinor_col, source_idx
                ]
            spin_coeffs[:, spinor_row, :] = row
        np.add.at(
            transformed,
            (slice(None), slice(None), target_idx),
            phase[None, None, :] * spin_coeffs,
        )

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
    target_q_cart: np.ndarray | None = None,
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

    target_q = q if target_q_cart is None else np.asarray(target_q_cart, dtype=float)
    if (
        target_q.ndim != 2
        or target_q.shape[1:] != (3,)
        or not np.all(np.isfinite(target_q))
    ):
        raise ValueError("target_q_cart must be finite with shape [nG,3]")
    rot_q = q @ rotation.T
    base_source = np.rint(rot_q / tolerance).astype(np.int64)
    base_target = np.rint(target_q / tolerance).astype(np.int64)
    lookup = _q_vector_lookup(base_target)
    q_flat = target_q.tolist()
    tol_sq = tolerance * tolerance
    mapping = np.full(len(q), -1, dtype=int)
    for source_idx, ((bx, by, bz), (qx, qy, qz)) in enumerate(
        zip(base_source.tolist(), rot_q.tolist())
    ):
        mapping[source_idx] = _lookup_q_vector(
            bx, by, bz, qx, qy, qz, lookup, q_flat, tol_sq,
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
            in_range = (
                mapping[(mapping >= 0) & (mapping < dimension)]
                if mapping.dtype.kind == "i"
                else mapping[mapping < dimension]
            )
            reasons: list[str] = []
            if len(mapping) != dimension:
                reasons.append("source_coverage_incomplete")
            if mapping.dtype.kind == "i" and bool(np.count_nonzero(mapping < 0)):
                reasons.append("source_coverage_incomplete")
            if bool(np.count_nonzero(mapping >= dimension)):
                reasons.append("target_index_out_of_range")
            if len(np.unique(in_range)) != len(in_range):
                reasons.append("target_index_collision")
            if len(np.unique(in_range)) != dimension:
                reasons.append("target_coverage_incomplete")
            return ReciprocalGridPermutationValidation(
                "passed" if not reasons else "blocked",
                tuple(dict.fromkeys(reasons)),
            )
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


def _q_vector_lookup(base: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
    lookup: dict[tuple[int, int, int], list[int]] = {}
    for idx, key in enumerate(map(tuple, base.tolist())):
        lookup.setdefault(key, []).append(idx)
    return lookup


def _lookup_q_vector(
    bx: int,
    by: int,
    bz: int,
    qx: float,
    qy: float,
    qz: float,
    lookup: dict[tuple[int, int, int], list[int]],
    q_flat: list[list[float]],
    tol_sq: float,
) -> int:
    """Nearest candidate within ``sqrt(tol_sq)`` over the 27-cell neighborhood."""
    best_idx = -1
    best_distance_sq = float("inf")
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cell = lookup.get((bx + dx, by + dy, bz + dz))
                if cell is None:
                    continue
                for idx in cell:
                    vx, vy, vz = q_flat[idx]
                    diff_x = vx - qx
                    diff_y = vy - qy
                    diff_z = vz - qz
                    distance_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                    if distance_sq <= tol_sq and distance_sq < best_distance_sq:
                        best_idx = idx
                        best_distance_sq = distance_sq
    return best_idx
