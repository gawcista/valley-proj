from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from valley_proj.geometry.reciprocal import minimum_periodic_distance
from valley_proj.geometry.valley_centers import ValleyCenter, ValleySector, centers_by_name


@dataclass(frozen=True)
class SectorProjectors:
    sector_masks: dict[str, np.ndarray]
    center_masks: dict[str, np.ndarray]
    ambiguous_mask: np.ndarray
    qcut: float
    warnings: list[str]

    @property
    def sector_names(self) -> list[str]:
        return list(self.sector_masks)


def build_sector_projectors(
    q_cart: np.ndarray,
    centers: list[ValleyCenter],
    sectors: list[ValleySector],
    monolayer_reciprocal_cart: np.ndarray,
    qcut: float,
    *,
    use_2d: bool = True,
    g_search_shell: int = 2,
    ambiguous_policy: str = "warn_exclude",
) -> SectorProjectors:
    q = np.asarray(q_cart, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q_cart must have shape [nG,3]")
    if qcut <= 0.0:
        raise ValueError("qcut must be positive")
    center_map = centers_by_name(centers)
    sector_masks: dict[str, np.ndarray] = {}
    center_masks: dict[str, np.ndarray] = {}

    for center in centers:
        center_reciprocal = center.reciprocal_cart
        if center_reciprocal is None:
            center_reciprocal = monolayer_reciprocal_cart
        distances = minimum_periodic_distance(
            q,
            center.cart,
            center_reciprocal,
            shell=g_search_shell,
            use_2d=use_2d,
        )
        center_masks[center.name] = distances < qcut

    for sector in sectors:
        missing = [name for name in sector.centers if name not in center_map]
        if missing:
            raise ValueError(f"Valley sector {sector.name} references unknown centers: {missing}")
        mask = np.zeros(q.shape[0], dtype=bool)
        for center_name in sector.centers:
            mask |= center_masks[center_name]
        sector_masks[sector.name] = mask

    memberships = np.zeros(q.shape[0], dtype=int)
    for mask in sector_masks.values():
        memberships += mask.astype(int)
    ambiguous = memberships > 1
    messages: list[str] = []
    if ambiguous.any():
        message = (
            f"{int(ambiguous.sum())} plane-wave components are ambiguous across valley sectors; "
            "try smaller qcut or check valley centers and monolayer lattice"
        )
        messages.append(message)
        if ambiguous_policy == "error":
            raise ValueError(message)
        if ambiguous_policy == "warn_exclude":
            warnings.warn(message, UserWarning, stacklevel=2)
            for name in list(sector_masks):
                sector_masks[name] = sector_masks[name] & ~ambiguous
        elif ambiguous_policy == "include":
            pass
        else:
            raise ValueError(f"Unsupported ambiguous_cross_sector policy: {ambiguous_policy}")

    return SectorProjectors(
        sector_masks=sector_masks,
        center_masks=center_masks,
        ambiguous_mask=ambiguous,
        qcut=float(qcut),
        warnings=messages,
    )
