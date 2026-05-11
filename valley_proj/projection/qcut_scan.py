from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valley_proj.geometry.valley_centers import ValleyCenter, ValleySector
from valley_proj.projection.sector_projectors import build_sector_projectors
from valley_proj.projection.weights import ValleyWeightResult, compute_valley_weights


@dataclass(frozen=True)
class QcutScanEntry:
    qcut: float
    weights: list[ValleyWeightResult]
    ambiguous_count: int


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
    ambiguous_policy: str = "warn_exclude",
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
            ambiguous_policy=ambiguous_policy,
        )
        entries.append(
            QcutScanEntry(
                qcut=float(qcut),
                weights=compute_valley_weights(coefficients, projectors),
                ambiguous_count=int(projectors.ambiguous_mask.sum()),
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
) -> float:
    center_map = {center.name: center for center in centers}
    sector_points: list[tuple[str, np.ndarray]] = []
    for sector in sectors:
        for center_name in sector.centers:
            sector_points.append((sector.name, center_map[center_name].cart))
    distances: list[float] = []
    for idx, (sector_a, point_a) in enumerate(sector_points):
        for sector_b, point_b in sector_points[idx + 1 :]:
            if sector_a != sector_b:
                distances.append(float(np.linalg.norm(point_a[:2] - point_b[:2])))
    if not distances:
        raise ValueError("relative_min_sector_distance requires at least two sectors")
    return float(fraction * min(distances))
