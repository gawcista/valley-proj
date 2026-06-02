from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from valleyscope.geometry.reciprocal import minimum_periodic_distance
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector, centers_by_name
from valleyscope.projection.folded_center import fold_center_into_moire_bz


@dataclass(frozen=True)
class SectorProjectors:
    sector_masks: dict[str, np.ndarray]
    center_masks: dict[str, np.ndarray]
    overlap_mask: np.ndarray
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
    g_search_shell: int = 3,
    overlap_policy: str = "warn_exclude",
    emit_warnings: bool = True,
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
    overlap = memberships > 1
    messages: list[str] = []
    if overlap.any():
        message = (
            f"{int(overlap.sum())} plane-wave components overlap across valleys; "
            "try smaller qcut or check valley centers and monolayer lattice"
        )
        messages.append(message)
        if overlap_policy == "error":
            raise ValueError(message)
        if overlap_policy == "warn_exclude":
            if emit_warnings:
                warnings.warn(message, UserWarning, stacklevel=2)
            for name in list(sector_masks):
                sector_masks[name] = sector_masks[name] & ~overlap
        elif overlap_policy == "include":
            pass
        else:
            raise ValueError(f"Unsupported overlap_policy: {overlap_policy}")

    return SectorProjectors(
        sector_masks=sector_masks,
        center_masks=center_masks,
        overlap_mask=overlap,
        qcut=float(qcut),
        warnings=messages,
    )


def adjust_centers_for_parent_valley(
    centers: list[ValleyCenter],
    moire_k_cart: np.ndarray,
    moire_reciprocal_cart: np.ndarray,
    *,
    use_2d: bool = True,
) -> list[ValleyCenter]:
    """Create k-dependent dynamic centers for k-resolved parent-valley projection.

    Each monolayer valley center Q_a is folded into the moire BZ:
        Q_a = k_a^fold + G_a^M

    For a given moire k-point k_M, the dynamic center is:
        Q_a(k_M) = Q_a + (k_M - k_a^fold) = k_M + G_a^M

    This is momentum-space parent-valley projection, not full monolayer
    Bloch-state unfolding.

    Parameters
    ----------
    centers : list of ValleyCenter
        Original monolayer valley centers.
    moire_k_cart : (3,) ndarray
        Cartesian coordinate of the current moire k-point.
    moire_reciprocal_cart : (3, 3) ndarray
        Moire reciprocal lattice basis.
    use_2d : bool
        If True, only in-plane components are used.

    Returns
    -------
    list of ValleyCenter
        New centers with cart adjusted to Q_a(k_M).
    """
    k_m = np.asarray(moire_k_cart, dtype=float)
    adjusted: list[ValleyCenter] = []
    for center in centers:
        folded_frac, g_int, folded_cart = fold_center_into_moire_bz(
            center.cart, moire_reciprocal_cart, use_2d=use_2d,
        )
        # Q_a(k_M) = Q_a + (k_M - k_a^fold)
        # k_a^fold = folded_cart (cartesian of folded frac position)
        k_a_fold = np.asarray(folded_cart, dtype=float)
        dynamic_center_cart = np.asarray(center.cart, dtype=float) + (k_m - k_a_fold)
        if use_2d:
            dynamic_center_cart[2] = 0.0
        adjusted.append(
            ValleyCenter(
                name=center.name,
                cart=dynamic_center_cart,
                layer=center.layer,
                reciprocal_cart=center.reciprocal_cart,
            )
        )
    return adjusted


# Deprecated alias — use adjust_centers_for_parent_valley.
adjust_centers_for_folded_family = adjust_centers_for_parent_valley
