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
    if det == 1 and order is not None and order != 1:
        kind = f"C{order}"
    elif det == 1:
        kind = "proper_rotation"
    else:
        kind = "improper_or_reflection"
    allowed = bool(det == 1 and order in allowed_orders and order != 1)
    return OperationInfo(rot, trans, det, order, kind, allowed)
