from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from valley_proj.projection.sector_projectors import SectorProjectors


DEFAULT_THRESHOLDS = {
    "W_val_min": 0.8,
    "P_v_clean": 0.95,
    "P_v_approx": 0.85,
}


@dataclass(frozen=True)
class ValleyWeightResult:
    band_position: int
    sector_weights: dict[str, float]
    w_val: float
    purity: float
    leakage: float
    ambiguous_weight: float
    eta: float | None
    norm: float
    diagnostics: dict[str, object] = field(default_factory=dict)


def _band_probabilities(coefficients: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape [nb,nspinor,nG]")
    return np.sum(np.abs(coeffs) ** 2, axis=1)


def compute_valley_weights(coefficients: np.ndarray, projectors: SectorProjectors) -> list[ValleyWeightResult]:
    probabilities = _band_probabilities(coefficients)
    n_g = probabilities.shape[1]
    for name, mask in projectors.sector_masks.items():
        if mask.shape != (n_g,):
            raise ValueError(f"Sector mask {name} does not match coefficient nG")
    if projectors.ambiguous_mask.shape != (n_g,):
        raise ValueError("ambiguous_mask does not match coefficient nG")

    results: list[ValleyWeightResult] = []
    sector_names = list(projectors.sector_masks)
    for band_pos, probs in enumerate(probabilities):
        norm = float(np.sum(probs))
        sector_weights = {
            name: float(np.sum(probs[mask]))
            for name, mask in projectors.sector_masks.items()
        }
        w_val = float(sum(sector_weights.values()))
        ambiguous_weight = float(np.sum(probs[projectors.ambiguous_mask]))
        leakage = float(max(0.0, norm - w_val - ambiguous_weight))
        purity = float(max(sector_weights.values()) / w_val) if w_val > 0.0 else 0.0
        eta = None
        if len(sector_names) == 2 and w_val > 0.0:
            eta = float((sector_weights[sector_names[0]] - sector_weights[sector_names[1]]) / w_val)
        results.append(
            ValleyWeightResult(
                band_position=band_pos,
                sector_weights=sector_weights,
                w_val=w_val,
                purity=purity,
                leakage=leakage,
                ambiguous_weight=ambiguous_weight,
                eta=eta,
                norm=norm,
            )
        )
    return results


def classify_valley_weights(
    *,
    w_val: float,
    purity: float,
    thresholds: dict[str, float] | None,
) -> dict[str, object]:
    values = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        values.update(thresholds)
    if purity > values["P_v_clean"]:
        clean = "clean"
    elif purity > values["P_v_approx"]:
        clean = "approximate"
    else:
        clean = "mixed"
    return {
        "valley_derived": bool(w_val > values["W_val_min"]),
        "valley_clean": clean,
    }
