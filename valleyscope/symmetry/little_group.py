from __future__ import annotations

import numpy as np

# Default absolute tolerance for HSP little-group k-point residual
# (max componentwise deviation of (R^-T k - k) from an integer).
# Chosen to admit the measured ~1e-6 reciprocal-lattice residual that
# occurs when a HSP fractional coordinate is stored with six-decimal
# precision in a HDF5 file, with a small explicit margin.
DEFAULT_HSP_LITTLE_GROUP_K_RESIDUAL_TOLERANCE: float = 5e-6


def reciprocal_transform(rotation: np.ndarray, k_frac: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotation, dtype=float)
    k = np.asarray(k_frac, dtype=float)
    return np.linalg.inv(rot).T @ k


def is_integer_vector(values: np.ndarray, tolerance: float = 1e-8) -> bool:
    arr = np.asarray(values, dtype=float)
    # Strict absolute comparison: a relative tolerance would accept ~1e-5
    # residue on integer components of magnitude one, exceeding the
    # unitary-completion absolute-norm bound of 1e-8.
    return bool(np.allclose(arr, np.rint(arr), rtol=0.0, atol=tolerance))


def _validate_crystallographic_rotation(rotation: np.ndarray) -> np.ndarray | None:
    """Return *rotation* cast to float64 if it is a finite, nonsingular 3x3
    matrix whose entries satisfy the integer crystallographic operation
    contract (every entry within 1e-12 of an integer).  Return ``None`` for
    malformed, non-integer, non-3x3, singular, or non-finite input."""
    rot = np.asarray(rotation, dtype=float)
    if rot.shape != (3, 3):
        return None
    if not np.isfinite(rot).all():
        return None
    if abs(np.linalg.det(rot)) < 1e-12:
        return None
    # All entries must be indistinguishable from integers.
    if not np.allclose(rot, np.rint(rot), rtol=0.0, atol=1e-12):
        return None
    return rot


def little_group_residual_max_abs(rotation: np.ndarray, k_frac: np.ndarray) -> float:
    """Return ``max_i |(R^-T k - k - G)_i|`` for integer rotation *R*.

    This is the raw reciprocal-lattice residual used for HSP little-group
    membership decisions.  The rotation must pass
    :func:`_validate_crystallographic_rotation`; otherwise ``inf`` is returned.
    """
    rot = _validate_crystallographic_rotation(rotation)
    if rot is None:
        return float("inf")
    k = np.asarray(k_frac, dtype=float)
    transformed = reciprocal_transform(rot, k)
    residue = transformed - k
    nearest_int = np.rint(residue)
    return float(np.max(np.abs(residue - nearest_int)))


def is_little_group_operation(
    rotation: np.ndarray,
    k_frac: np.ndarray,
    tolerance: float = DEFAULT_HSP_LITTLE_GROUP_K_RESIDUAL_TOLERANCE,
) -> bool:
    """Return ``True`` when the integer rotation *R* leaves the k-point
    invariant up to a reciprocal-lattice vector within the given absolute
    fractional-k residual tolerance.

    The tolerance is an absolute per-component threshold on
    ``max_i |(R^-T k - k - G)_i|`` and is distinct from exact
    reciprocal-grid permutation, projector covariance, representation
    closure, and standard-setting matching tolerances.
    """
    return little_group_residual_max_abs(rotation, k_frac) <= float(tolerance)
