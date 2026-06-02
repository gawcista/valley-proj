"""Folded-center report: fold valley centers into the moire BZ.

Provides diagnostic information about where monolayer valley centers land
inside the moire Brillouin zone and how far they are from each sampled
moiré k-point.  This is a companion diagnostic for the folded_family
projector mode but is always computed when moire reciprocal lattice data
is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valleyscope.geometry.valley_centers import ValleyCenter


@dataclass(frozen=True)
class FoldedCenterEntry:
    """A single valley center folded into the moire BZ."""

    center_name: str
    layer: str | None
    cart: np.ndarray
    """Original monolayer valley center in Cartesian (A^-1)."""
    folded_frac: np.ndarray
    """Folded moiré fractional coordinate, centered into [-0.5, 0.5)."""
    g_moire_int: np.ndarray
    """Integer moiré reciprocal lattice shift G_a^M = round(Q_a in moiré frac)."""
    folded_cart: np.ndarray
    """Folded center position in Cartesian (A^-1)."""


@dataclass(frozen=True)
class FoldedCenterReport:
    """Folded-center report across all centers and sampled k-points."""

    entries: list[FoldedCenterEntry]
    kpoint_distances: dict[str, list[float]]
    """Per-center-name dict: distance from folded center to each sampled kpoint."""


def fold_center_into_moire_bz(
    center_cart: np.ndarray,
    moire_reciprocal_cart: np.ndarray,
    use_2d: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold a Cartesian valley center into the moire BZ.

    Parameters
    ----------
    center_cart : (3,) ndarray
        Monolayer valley center in Cartesian coordinates (A^-1).
    moire_reciprocal_cart : (3, 3) ndarray
        Moiré reciprocal lattice basis, rows are basis vectors.
    use_2d : bool
        If True, use only in-plane components (default).

    Returns
    -------
    folded_frac : (3,) ndarray
        Fractional coordinate in moiré BZ, centered into [-0.5, 0.5).
    g_int : (3,) ndarray
        Integer G-vector index G_a^M = round(frac_raw).
    folded_cart : (3,) ndarray
        Folded Cartesian position (approximately k_a^fold).
    """
    q = np.asarray(center_cart, dtype=float)
    basis = np.asarray(moire_reciprocal_cart, dtype=float)
    if basis.shape != (3, 3):
        raise ValueError("moire_reciprocal_cart must have shape [3,3]")
    if use_2d:
        basis_2d = basis[:2, :2]
        q_2d = q[:2]
        frac_2d = q_2d @ np.linalg.inv(basis_2d)
        g_int_2d = np.rint(frac_2d)
        folded_frac_2d = frac_2d - g_int_2d
        folded_cart_2d = folded_frac_2d @ basis_2d
        folded_frac = np.zeros(3, dtype=float)
        folded_frac[:2] = folded_frac_2d
        g_int = np.zeros(3, dtype=float)
        g_int[:2] = g_int_2d
        folded_cart = np.zeros(3, dtype=float)
        folded_cart[:2] = folded_cart_2d
    else:
        frac = q @ np.linalg.inv(basis)
        g_int = np.rint(frac)
        folded_frac = frac - g_int
        folded_cart = folded_frac @ basis
    return folded_frac, g_int, folded_cart


def build_folded_center_report(
    centers: list[ValleyCenter],
    moire_reciprocal_cart: np.ndarray,
    sampled_k_frac: dict[str, np.ndarray],
    *,
    use_2d: bool = True,
) -> FoldedCenterReport:
    """Build a folded-center diagnostic report.

    Parameters
    ----------
    centers : list of ValleyCenter
        Monolayer valley centers.
    moire_reciprocal_cart : (3, 3) ndarray
        Moiré reciprocal lattice basis.
    sampled_k_frac : dict[str, (3,) ndarray]
        Map from k-point name to fractional moiré coordinate.
    use_2d : bool
        If True, use only in-plane components.

    Returns
    -------
    FoldedCenterReport
    """
    entries: list[FoldedCenterEntry] = []
    kpoint_distances: dict[str, list[float]] = {}

    basis = np.asarray(moire_reciprocal_cart, dtype=float)

    for center in centers:
        folded_frac, g_int, folded_cart = fold_center_into_moire_bz(
            center.cart, moire_reciprocal_cart, use_2d=use_2d,
        )
        entries.append(
            FoldedCenterEntry(
                center_name=center.name,
                layer=center.layer,
                cart=np.asarray(center.cart, dtype=float).copy(),
                folded_frac=folded_frac,
                g_moire_int=g_int.astype(int),
                folded_cart=folded_cart,
            )
        )

        # Distance from folded center to each sampled k-point.
        distances: list[float] = []
        for k_name, k_frac in sampled_k_frac.items():
            kf = np.asarray(k_frac, dtype=float)
            if use_2d:
                basis_2d = basis[:2, :2]
                delta_frac = kf[:2] - folded_frac[:2]
                delta_frac -= np.rint(delta_frac)
                delta_cart = delta_frac @ basis_2d
                dist = float(np.linalg.norm(delta_cart))
            else:
                delta_frac = kf - folded_frac
                delta_frac -= np.rint(delta_frac)
                delta_cart = delta_frac @ basis
                dist = float(np.linalg.norm(delta_cart))
            distances.append(dist)
        kpoint_distances[center.name] = distances

    return FoldedCenterReport(
        entries=entries,
        kpoint_distances=kpoint_distances,
    )


def folded_center_report_to_dict(
    report: FoldedCenterReport,
    kpoint_names: list[str],
) -> dict[str, object]:
    """Serialize a FoldedCenterReport to a JSON-compatible dict."""
    entries_payload: list[dict[str, object]] = []
    for entry in report.entries:
        entries_payload.append({
            "center_name": entry.center_name,
            "layer": entry.layer,
            "cart": entry.cart.tolist(),
            "folded_frac": entry.folded_frac.tolist(),
            "g_moire_int": entry.g_moire_int.tolist(),
            "folded_cart": entry.folded_cart.tolist(),
        })

    distances_payload: dict[str, object] = {}
    for center_name, dists in report.kpoint_distances.items():
        distances_payload[center_name] = {
            kp: dist for kp, dist in zip(kpoint_names, dists)
        }

    return {
        "folded_centers": entries_payload,
        "kpoint_distances": distances_payload,
    }
