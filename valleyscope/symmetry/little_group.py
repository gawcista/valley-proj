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


def _validate_gl3z_rotation(rotation: np.ndarray) -> np.ndarray | None:
    """Return *rotation* cast to float64 if it is a valid GL(3,Z) fractional
    rotation matrix, or ``None``.

    A crystallographic fractional rotation must be:
    - exactly (3, 3) shape;
    - finite in every entry;
    - integer-valued in every entry (exact equality, not within a
      tolerance — the producer representation *is* integer);
    - unimodular (det = ±1, within float64 precision).
    """
    rot = np.asarray(rotation, dtype=float)
    if rot.shape != (3, 3):
        return None
    if not np.isfinite(rot).all():
        return None
    # Every entry must equal its integer rounding exactly.
    rot_int = np.rint(rot)
    if not np.array_equal(rot, rot_int):
        return None
    det = float(np.linalg.det(rot))
    if not np.isclose(abs(det), 1.0, rtol=0.0, atol=1e-12):
        return None
    return rot


def _validate_k_frac(k_frac: np.ndarray) -> np.ndarray | None:
    """Return *k_frac* as a finite float64 3-vector, or ``None``."""
    k = np.asarray(k_frac, dtype=float)
    if k.shape != (3,):
        return None
    if not np.isfinite(k).all():
        return None
    return k


def _validate_tolerance(tolerance: float, label: str) -> float:
    """Return *tolerance* as float, or raise ``ValueError``."""
    t = float(tolerance)
    if not np.isfinite(t) or t < 0:
        raise ValueError(f"{label} must be finite and nonnegative, got {t!r}")
    return t


def hsp_little_group_evidence(
    rotation: np.ndarray,
    k_frac: np.ndarray,
    tolerance: float = DEFAULT_HSP_LITTLE_GROUP_K_RESIDUAL_TOLERANCE,
) -> dict[str, object]:
    """Return a per-operation evidence dict with keys *passed*,
    *residual_max_abs*, *tolerance*, and *reason*.

    Validates the rotation as GL(3,Z), validates k and tolerance, and
    computes ``max_i |R^-T k - k - G|`` against the absolute fractional-k
    tolerance.  The raw k-coordinate is never mutated.

    This is the single trust-boundary helper for HSP little-group
    membership evidence; all production inventory builders should use it.
    """
    tol = _validate_tolerance(tolerance, "hsp_little_group_k_residual_tolerance")
    rot = _validate_gl3z_rotation(rotation)
    if rot is None:
        return {
            "passed": False,
            "residual_max_abs": float("inf"),
            "tolerance": tol,
            "reason": "rotation_not_gl3z",
        }
    k = _validate_k_frac(k_frac)
    if k is None:
        return {
            "passed": False,
            "residual_max_abs": float("inf"),
            "tolerance": tol,
            "reason": "k_frac_malformed",
        }
    transformed = reciprocal_transform(rot, k)
    residue = transformed - k
    nearest_int = np.rint(residue)
    residual = float(np.max(np.abs(residue - nearest_int)))
    return {
        "passed": residual <= tol,
        "residual_max_abs": residual,
        "tolerance": tol,
        "reason": "",
    }


def little_group_residual_max_abs(rotation: np.ndarray, k_frac: np.ndarray) -> float:
    """Return ``max_i |(R^-T k - k - G)_i|`` for a valid GL(3,Z) rotation,
    or ``inf`` when the rotation or k-point are invalid."""
    try:
        ev = hsp_little_group_evidence(rotation, k_frac, tolerance=1.0)
        return float(ev["residual_max_abs"])
    except ValueError:
        return float("inf")


def is_little_group_operation(
    rotation: np.ndarray,
    k_frac: np.ndarray,
    tolerance: float = DEFAULT_HSP_LITTLE_GROUP_K_RESIDUAL_TOLERANCE,
) -> bool:
    """Return ``True`` when the GL(3,Z) rotation *R* leaves the k-point
    invariant up to a reciprocal-lattice vector within the given absolute
    fractional-k residual tolerance.

    Thin wrapper around :func:`hsp_little_group_evidence`.
    """
    return bool(hsp_little_group_evidence(rotation, k_frac, tolerance)["passed"])
