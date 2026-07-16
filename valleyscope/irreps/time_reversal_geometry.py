"""Shared reciprocal-space geometry for time-reversal source reduction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_centering_vectors(
    values: Sequence[Sequence[float]],
) -> list[np.ndarray] | None:
    """Validate and normalize direct-space centering vectors."""
    if not values:
        return None
    out: list[np.ndarray] = []
    for value in values:
        try:
            vector = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            return None
        out.append(vector)
    return out


def centered_k_equivalent(
    left: np.ndarray,
    right: np.ndarray,
    centering_vectors: Sequence[np.ndarray],
    *,
    tolerance: float,
) -> bool:
    """Test reciprocal equivalence including conventional centering phases."""
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    shift = np.rint(delta).astype(int)
    if np.linalg.norm(delta - shift) > tolerance:
        return False
    return all(
        abs(float(np.dot(shift, vector)) - round(float(np.dot(shift, vector))))
        <= tolerance
        for vector in centering_vectors
    )
