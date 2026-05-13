from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valleyscope.geometry.reciprocal import equivalent_mod_reciprocal
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector, centers_by_name


@dataclass(frozen=True)
class ValleyMappingResult:
    sector_mapping: dict[str, str | None]
    preserved: dict[str, bool]
    center_mapping: dict[str, str | None]


def map_valley_sectors(
    rotation_frac: np.ndarray,
    rotation_cart: np.ndarray,
    centers: list[ValleyCenter],
    sectors: list[ValleySector],
    monolayer_reciprocal_cart: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> ValleyMappingResult:
    del rotation_frac
    center_map = centers_by_name(centers)
    center_to_sector = {center_name: sector.name for sector in sectors for center_name in sector.centers}
    for sector in sectors:
        for center_name in sector.centers:
            if center_name not in center_map:
                raise ValueError(f"Unknown center in sector {sector.name}: {center_name}")

    center_mapping: dict[str, str | None] = {}
    for center in centers:
        rotated = np.asarray(rotation_cart, dtype=float) @ center.cart
        target_name = None
        for candidate in centers:
            reciprocal = candidate.reciprocal_cart
            if reciprocal is None:
                reciprocal = center.reciprocal_cart
            if reciprocal is None:
                reciprocal = monolayer_reciprocal_cart
            if equivalent_mod_reciprocal(
                rotated,
                candidate.cart,
                reciprocal,
                tolerance=tolerance,
                shell=2,
                use_2d=True,
            ):
                target_name = candidate.name
                break
        center_mapping[center.name] = target_name

    sector_mapping: dict[str, str | None] = {}
    preserved: dict[str, bool] = {}
    for sector in sectors:
        mapped_sector_names: set[str] = set()
        all_centers_mapped = True
        for center_name in sector.centers:
            target_center = center_mapping.get(center_name)
            if target_center is None:
                all_centers_mapped = False
            else:
                mapped_sector_names.add(center_to_sector[target_center])
        if all_centers_mapped and len(mapped_sector_names) == 1:
            target_sector = next(iter(mapped_sector_names))
        else:
            target_sector = None
        sector_mapping[sector.name] = target_sector
        preserved[sector.name] = target_sector == sector.name
    return ValleyMappingResult(sector_mapping, preserved, center_mapping)
