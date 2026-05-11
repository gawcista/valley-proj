from __future__ import annotations

from itertools import product

import numpy as np


def reciprocal_grid(reciprocal_cart: np.ndarray, shell: int = 2, use_2d: bool = True) -> np.ndarray:
    basis = np.asarray(reciprocal_cart, dtype=float)
    if basis.shape != (3, 3):
        raise ValueError("reciprocal_cart must have shape [3,3]")
    if shell < 0:
        raise ValueError("shell must be non-negative")
    if use_2d:
        coeffs = [(i, j, 0) for i, j in product(range(-shell, shell + 1), repeat=2)]
    else:
        coeffs = list(product(range(-shell, shell + 1), repeat=3))
    return np.asarray(coeffs, dtype=float) @ basis


def minimum_periodic_distance(
    q_cart: np.ndarray,
    center_cart: np.ndarray,
    reciprocal_cart: np.ndarray,
    shell: int = 2,
    use_2d: bool = True,
) -> np.ndarray:
    q = np.asarray(q_cart, dtype=float)
    center = np.asarray(center_cart, dtype=float)
    basis = np.asarray(reciprocal_cart, dtype=float)
    if basis.shape != (3, 3):
        raise ValueError("reciprocal_cart must have shape [3,3]")
    if use_2d:
        basis_2d = basis[:2, :2]
        if abs(float(np.linalg.det(basis_2d))) > 1e-14:
            deltas = q[:, :2] - center[:2]
            frac = deltas @ np.linalg.inv(basis_2d)
            wrapped = frac - np.rint(frac)
            return np.linalg.norm(wrapped @ basis_2d, axis=1)
    elif abs(float(np.linalg.det(basis))) > 1e-14:
        deltas = q - center
        frac = deltas @ np.linalg.inv(basis)
        wrapped = frac - np.rint(frac)
        return np.linalg.norm(wrapped @ basis, axis=1)

    shifts = reciprocal_grid(basis, shell=shell, use_2d=use_2d)
    deltas = q[:, None, :] - (center[None, None, :] + shifts[None, :, :])
    if use_2d:
        deltas = deltas.copy()
        deltas[:, :, 2] = 0.0
    return np.linalg.norm(deltas, axis=2).min(axis=1)


def equivalent_mod_reciprocal(
    q_cart: np.ndarray,
    center_cart: np.ndarray,
    reciprocal_cart: np.ndarray,
    tolerance: float,
    shell: int = 2,
    use_2d: bool = True,
) -> bool:
    q = np.asarray(q_cart, dtype=float).reshape(1, 3)
    distance = minimum_periodic_distance(q, center_cart, reciprocal_cart, shell=shell, use_2d=use_2d)[0]
    return bool(distance <= tolerance)
