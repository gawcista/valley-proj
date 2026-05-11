from __future__ import annotations

import numpy as np


def reciprocal_transform(rotation: np.ndarray, k_frac: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotation, dtype=float)
    k = np.asarray(k_frac, dtype=float)
    return np.linalg.inv(rot).T @ k


def is_integer_vector(values: np.ndarray, tolerance: float = 1e-8) -> bool:
    arr = np.asarray(values, dtype=float)
    return bool(np.allclose(arr, np.rint(arr), atol=tolerance))


def is_little_group_operation(rotation: np.ndarray, k_frac: np.ndarray, tolerance: float = 1e-8) -> bool:
    transformed = reciprocal_transform(rotation, k_frac)
    return is_integer_vector(transformed - np.asarray(k_frac, dtype=float), tolerance=tolerance)
