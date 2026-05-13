from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valleyscope.symmetry.plane_wave_action import unitarity_deviation


@dataclass(frozen=True)
class RotationEigenvalueResult:
    matrix: np.ndarray
    eigenvalues: np.ndarray
    phases_2pi: np.ndarray
    modulus_deviation: np.ndarray
    unitarity_deviation: float
    spinor_convention_verified: bool


def nearest_root_of_unity(value: complex, order: int) -> tuple[int, complex, float]:
    if order <= 0:
        raise ValueError("order must be positive")
    roots = np.exp(2.0j * np.pi * np.arange(order) / order)
    distances = np.abs(np.asarray(value, dtype=np.complex128) - roots)
    index = int(np.argmin(distances))
    return index, complex(roots[index]), float(distances[index])


def extract_rotation_eigenvalues(
    representation_matrix: np.ndarray,
    *,
    spinor_convention_verified: bool = False,
) -> RotationEigenvalueResult:
    matrix = np.asarray(representation_matrix, dtype=np.complex128)
    eigenvalues = np.linalg.eigvals(matrix)
    phases = np.angle(eigenvalues) / (2.0 * np.pi)
    return RotationEigenvalueResult(
        matrix=matrix,
        eigenvalues=eigenvalues,
        phases_2pi=phases,
        modulus_deviation=np.abs(np.abs(eigenvalues) - 1.0),
        unitarity_deviation=unitarity_deviation(matrix),
        spinor_convention_verified=spinor_convention_verified,
    )
