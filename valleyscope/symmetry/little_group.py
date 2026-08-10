from __future__ import annotations

import numpy as np

# Common high-symmetry-point denominators for reciprocal-lattice fractional
# coordinates (hexagonal: 2, 3, 6; cubic: 2, 4; general: 2, 3, 4, 6).
_COMMON_HSP_DENOMINATORS = (2, 3, 4, 6)
# Snap tolerance for k-point fractions that lost precision during HDF5
# storage or moiré reciprocal-lattice computation.
_K_FRAC_SNAP_TOLERANCE = 5e-4


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


def snap_hsp_kpoint(k_frac: np.ndarray) -> np.ndarray:
    """Return *k_frac* with each component snapped to the nearest rational
    fraction n/d (d = 2, 3, 4, 6) when within ``_K_FRAC_SNAP_TOLERANCE``.

    This corrects the precision loss that occurs when high-symmetry-point
    fractional coordinates are stored with truncated precision in HDF5 files
    or computed through float-heavy moiré reciprocal-lattice transforms.
    The snapping is deterministic and conservative: a component that is not
    recognisably close to a common-HSP rational is left unchanged.
    """
    k = np.asarray(k_frac, dtype=float)
    out = k.copy()
    for i in range(k.shape[0]):
        best = float(k[i])
        best_dist = float("inf")
        for denom in _COMMON_HSP_DENOMINATORS:
            for num in range(denom + 1):
                candidate = float(num) / float(denom)
                dist = abs(float(k[i]) - candidate)
                if dist < best_dist:
                    best_dist = dist
                    best = candidate
        if best_dist < _K_FRAC_SNAP_TOLERANCE:
            out[i] = best
    return out


def is_little_group_operation(rotation: np.ndarray, k_frac: np.ndarray, tolerance: float = 1e-8) -> bool:
    # Round the fractional rotation matrix to the nearest integers before
    # computing the inverse transpose.  Spglib symmetry operations are
    # crystallographic rotations that are integer-valued in fractional
    # coordinates, but floating-point detection and closure products may
    # introduce small deviations.
    rot_int = np.rint(np.asarray(rotation, dtype=float)).astype(np.int64)
    transformed = reciprocal_transform(rot_int, k_frac)
    return is_integer_vector(transformed - np.asarray(k_frac, dtype=float), tolerance=tolerance)
