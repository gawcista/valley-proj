from __future__ import annotations

from typing import Iterable

import numpy as np


RotationOrder = int | str | None

SUPPORTED_ROTATION_ORDERS = {2, 3, 4, 6}


def parse_rotation_order(value: object, *, default: RotationOrder = "auto") -> RotationOrder:
    if value is _MISSING:
        return default
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("rotation_order must be an integer, 'auto', or 'none'")
    if isinstance(value, int):
        return _validate_order(value)
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered == "auto":
            return "auto"
        if lowered == "none":
            return None
        if lowered.startswith("c") and lowered[1:].isdigit():
            return _validate_order(int(lowered[1:]))
        if lowered.isdigit():
            return _validate_order(int(lowered))
    raise ValueError("rotation_order must be an integer, 'auto', or 'none'")


def resolve_rotation_order(
    requested: RotationOrder,
    *,
    international: str,
    candidate_orders: Iterable[int],
) -> int | None:
    if requested is None:
        return None
    if isinstance(requested, int):
        return requested
    lowered = str(requested).strip().lower()
    if lowered == "none":
        return None
    if lowered != "auto":
        raise ValueError("requested rotation order must be an integer, 'auto', or None")
    available = set(candidate_orders)
    for order in (6, 4, 3, 2):
        if order in available:
            return order
    return None


def mark_rotation_generators(operations: list[dict[str, object]]) -> None:
    generators: dict[tuple[tuple[int, ...], ...], int] = {}
    generator_matrices: dict[int, np.ndarray] = {}
    for operation in operations:
        if not operation.get("candidate_rotation", False):
            continue
        operation_id = operation.get("operation_id")
        if not isinstance(operation_id, int) or isinstance(operation_id, bool):
            raise ValueError(
                "candidate rotation operation_id must be an exact "
                "non-Boolean Python integer"
            )
        rotation = np.asarray(operation["rotation_frac"], dtype=int)
        order = int(operation["order"])
        subgroup_key = _cyclic_subgroup_key(rotation, order)
        if subgroup_key not in generators:
            generators[subgroup_key] = operation_id
            generator_matrices[operation_id] = rotation
            operation["rotation_generator_operation_id"] = operation_id
            operation["rotation_power_of_generator"] = 1
            operation["candidate_rejection_reason"] = ""
            continue
        generator_id = generators[subgroup_key]
        operation["candidate_rotation"] = False
        operation["candidate_rejection_reason"] = "power_of_rotation_generator"
        operation["rotation_generator_operation_id"] = generator_id
        operation["rotation_power_of_generator"] = _rotation_power(generator_matrices[generator_id], rotation, order)


def _validate_order(value: int) -> int:
    if value not in SUPPORTED_ROTATION_ORDERS:
        raise ValueError("rotation_order currently supports only 2, 3, 4, or 6")
    return value


def _cyclic_subgroup_key(rotation: np.ndarray, order: int) -> tuple[tuple[int, ...], ...]:
    powers: list[tuple[int, ...]] = []
    power = np.eye(3, dtype=int)
    for _ in range(1, order):
        power = power @ rotation
        powers.append(tuple(int(value) for value in power.reshape(-1)))
    return tuple(sorted(powers))


def _rotation_power(generator: np.ndarray, rotation: np.ndarray, order: int) -> int:
    power = np.eye(3, dtype=int)
    for exponent in range(1, order):
        power = power @ generator
        if np.array_equal(power, rotation):
            return exponent
    return 0


class _Missing:
    pass


_MISSING = _Missing()
