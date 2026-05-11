from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValleyCenter:
    name: str
    cart: np.ndarray
    layer: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cart", np.asarray(self.cart, dtype=float))
        if self.cart.shape != (3,):
            raise ValueError("ValleyCenter.cart must have shape [3]")


@dataclass(frozen=True)
class ValleySector:
    name: str
    centers: list[str]

    def __post_init__(self) -> None:
        if not self.centers:
            raise ValueError(f"Valley sector {self.name} must contain at least one center")


def centers_by_name(centers: list[ValleyCenter]) -> dict[str, ValleyCenter]:
    result: dict[str, ValleyCenter] = {}
    for center in centers:
        if center.name in result:
            raise ValueError(f"Duplicate valley center name: {center.name}")
        result[center.name] = center
    return result
