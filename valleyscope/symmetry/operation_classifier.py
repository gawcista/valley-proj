from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperationInfo:
    rotation: np.ndarray
    translation: np.ndarray
    det: int
    order: int | None
    kind: str
    allowed_for_rotation_workflow: bool


def operation_order(rotation: np.ndarray, max_order: int = 12) -> int | None:
    rot = np.asarray(rotation, dtype=int)
    eye = np.eye(3, dtype=int)
    power = np.eye(3, dtype=int)
    for order in range(1, max_order + 1):
        power = power @ rot
        if np.array_equal(power, eye):
            return order
    return None


def classify_operation(
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    allowed_orders: list[int] | tuple[int, ...] = (2, 3, 4, 6),
) -> OperationInfo:
    rot = np.asarray(rotation, dtype=int)
    trans = np.asarray(translation, dtype=float)
    det = int(round(np.linalg.det(rot)))
    order = operation_order(rot)
    if det == 1 and order in allowed_orders and order != 1:
        kind = f"C{order}"
        allowed = True
    elif det == 1:
        kind = "proper_rotation"
        allowed = False
    else:
        kind = "improper_or_reflection"
        allowed = False
    return OperationInfo(rot, trans, det, order, kind, allowed)


def rotation_axis_angle(rotation_cart: np.ndarray, *, tolerance: float = 1e-8) -> tuple[np.ndarray, float]:
    rotation = np.asarray(rotation_cart, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_cart must have shape [3,3]")
    trace = float(np.trace(rotation))
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if abs(angle) < tolerance:
        raise ValueError("identity rotation has no unique axis")
    if abs(angle - np.pi) < tolerance:
        eigvals, eigvecs = np.linalg.eig(rotation)
        matches = np.where(np.isclose(eigvals.real, 1.0, atol=tolerance) & np.isclose(eigvals.imag, 0.0, atol=tolerance))[0]
        if len(matches) == 0:
            raise ValueError("could not extract rotation axis")
        axis = eigvecs[:, matches[0]].real
    else:
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=float,
        )
        axis /= 2.0 * np.sin(angle)
    norm = float(np.linalg.norm(axis))
    if norm < tolerance:
        raise ValueError("could not extract rotation axis")
    axis = axis / norm
    return axis, angle
