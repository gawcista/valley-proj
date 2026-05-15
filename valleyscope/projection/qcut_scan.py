from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valleyscope.geometry.reciprocal import minimum_periodic_distance
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector, centers_by_name
from valleyscope.projection.sector_projectors import build_sector_projectors
from valleyscope.projection.weights import ValleyWeightResult, compute_valley_weights


@dataclass(frozen=True)
class QcutScanEntry:
    qcut: float
    weights: list[ValleyWeightResult]
    overlap_count: int


@dataclass(frozen=True)
class QcutScanResult:
    entries: list[QcutScanEntry]
    has_plateau: bool


def scan_qcut(
    q_cart: np.ndarray,
    coefficients: np.ndarray,
    centers: list[ValleyCenter],
    sectors: list[ValleySector],
    monolayer_reciprocal_cart: np.ndarray,
    qcuts: list[float],
    *,
    use_2d: bool = True,
    overlap_policy: str = "warn_exclude",
    plateau_tol: float = 1e-2,
) -> QcutScanResult:
    entries: list[QcutScanEntry] = []
    for qcut in qcuts:
        projectors = build_sector_projectors(
            q_cart,
            centers,
            sectors,
            monolayer_reciprocal_cart,
            qcut,
            use_2d=use_2d,
            overlap_policy=overlap_policy,
        )
        entries.append(
            QcutScanEntry(
                qcut=float(qcut),
                weights=compute_valley_weights(coefficients, projectors),
                overlap_count=int(projectors.overlap_mask.sum()),
            )
        )
    has_plateau = False
    if len(entries) >= 3:
        tail = entries[-3:]
        values = np.array([[band.w_val for band in entry.weights] for entry in tail], dtype=float)
        has_plateau = bool(np.max(np.ptp(values, axis=0)) <= plateau_tol)
    return QcutScanResult(entries=entries, has_plateau=has_plateau)


def qcut_from_moire_shell(moire_reciprocal_cart: np.ndarray, shell: float) -> float:
    basis = np.asarray(moire_reciprocal_cart, dtype=float)
    return float(shell * max(np.linalg.norm(basis[0]), np.linalg.norm(basis[1])))


def qcut_from_min_sector_distance(
    centers: list[ValleyCenter],
    sectors: list[ValleySector],
    fraction: float,
    reciprocal_cart: np.ndarray | None = None,
    *,
    use_2d: bool = True,
) -> float:
    center_map = centers_by_name(centers)
    sector_points: list[tuple[str, np.ndarray]] = []
    for sector in sectors:
        for center_name in sector.centers:
            sector_points.append((sector.name, center_map[center_name].cart))
    distances: list[float] = []
    for idx, (sector_a, point_a) in enumerate(sector_points):
        for sector_b, point_b in sector_points[idx + 1 :]:
            if sector_a != sector_b:
                basis_a = _basis_for_point(point_a, centers, reciprocal_cart)
                basis_b = _basis_for_point(point_b, centers, reciprocal_cart)
                candidates = [float(np.linalg.norm(point_a[:2] - point_b[:2]))]
                if basis_a is not None:
                    candidates.append(
                        float(
                            minimum_periodic_distance(
                                point_a.reshape(1, 3),
                                point_b,
                                basis_a,
                                use_2d=use_2d,
                            )[0]
                        )
                    )
                if basis_b is not None:
                    candidates.append(
                        float(
                            minimum_periodic_distance(
                                point_a.reshape(1, 3),
                                point_b,
                                basis_b,
                                use_2d=use_2d,
                            )[0]
                        )
                    )
                distances.append(min(candidates))
    if not distances:
        raise ValueError("relative_min_sector_distance requires at least two sectors")
    return float(fraction * min(distances))


def _basis_for_point(
    point: np.ndarray,
    centers: list[ValleyCenter],
    fallback: np.ndarray | None,
) -> np.ndarray | None:
    for center in centers:
        if np.allclose(center.cart, point, atol=1e-8):
            return center.reciprocal_cart if center.reciprocal_cart is not None else fallback
    return fallback
