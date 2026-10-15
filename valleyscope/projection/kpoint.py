"""Shared source-coefficient reporting projection for one k point."""
from __future__ import annotations

import numpy as np

from valleyscope.geometry.reciprocal import RunLocalPeriodicDistanceCache
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.projection.sector_projectors import SectorProjectors, build_sector_projectors
from valleyscope.projection.weights import ValleyWeightResult, compute_valley_weights


def project_kpoint(
    q_cart: np.ndarray,
    source_coefficients: np.ndarray,
    centers: list[ValleyCenter],
    valleys: list[ValleySector],
    monolayer_reciprocal_cart: np.ndarray,
    qcut: float,
    *,
    use_2d: bool = True,
    overlap_policy: str = "warn_exclude",
    distance_cache: RunLocalPeriodicDistanceCache | None = None,
) -> tuple[SectorProjectors, list[ValleyWeightResult]]:
    """Return window masks and raw weights from the original source states.

    The caller supplies reporting centers in its selected mode. Readiness seed
    construction and representation trust are separate consumers.
    """
    projectors = build_sector_projectors(
        q_cart, centers, valleys, monolayer_reciprocal_cart, qcut,
        use_2d=use_2d, overlap_policy=overlap_policy, emit_warnings=False,
        distance_cache=distance_cache,
    )
    return projectors, compute_valley_weights(source_coefficients, projectors)
