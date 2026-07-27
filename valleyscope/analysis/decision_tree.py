from __future__ import annotations

import warnings

import numpy as np

DEFAULT_THRESHOLDS = {
    "W_val_min": 0.8,
    "valley_concentration_clean": 0.95,
    "valley_concentration_approx": 0.85,
    "P_v_clean": 0.95,
    "P_v_approx": 0.85,
    "overlap_warn": 0.05,
    "residual_warn": 0.20,
}


def _resolve_concentration_thresholds(
    thresholds: dict[str, float] | None,
) -> tuple[float, float]:
    """Resolve valley concentration clean/approx thresholds.

    Prefers ``valley_concentration_clean`` / ``valley_concentration_approx``
    over legacy ``P_v_clean`` / ``P_v_approx``.  Returns (clean, approx).
    """
    if thresholds is None:
        return (
            float(DEFAULT_THRESHOLDS["valley_concentration_clean"]),
            float(DEFAULT_THRESHOLDS["valley_concentration_approx"]),
        )
    has_new_clean = "valley_concentration_clean" in thresholds
    has_new_approx = "valley_concentration_approx" in thresholds
    has_legacy_clean = "P_v_clean" in thresholds
    has_legacy_approx = "P_v_approx" in thresholds

    if (has_new_clean or has_new_approx) and (has_legacy_clean or has_legacy_approx):
        warnings.warn(
            "Both valley_concentration_* and legacy P_v_* thresholds are set; "
            "using valley_concentration_* values where provided",
            DeprecationWarning, stacklevel=3,
        )

    clean = float(
        thresholds.get(
            "valley_concentration_clean",
            thresholds.get("P_v_clean", DEFAULT_THRESHOLDS["valley_concentration_clean"]),
        )
    )
    approx = float(
        thresholds.get(
            "valley_concentration_approx",
            thresholds.get("P_v_approx", DEFAULT_THRESHOLDS["valley_concentration_approx"]),
        )
    )
    return clean, approx


def _resolve_eta_thresholds(thresholds: dict[str, float] | None) -> tuple[float, float]:
    values = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        values.update(thresholds)
    if "eta_clean" in values:
        eta_clean = float(values["eta_clean"])
    else:
        pv_clean, _ = _resolve_concentration_thresholds(thresholds)
        eta_clean = 2.0 * pv_clean - 1.0
    if "eta_approx" in values:
        eta_approx = float(values["eta_approx"])
    else:
        _, pv_approx = _resolve_concentration_thresholds(thresholds)
        eta_approx = 2.0 * pv_approx - 1.0
    return eta_clean, eta_approx


def _resolve_pv_thresholds(thresholds: dict[str, float] | None) -> tuple[float, float]:
    return _resolve_concentration_thresholds(thresholds)


def derive_valley_status(
    *,
    analysis_level: str,
    derived_score: float,
    polarization_score: float,
    w_overlap: float,
    w_res: float,
    thresholds: dict[str, float] | None = None,
    two_sector: bool = True,
) -> str:
    values = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        values.update(thresholds)

    if derived_score < values["W_val_min"]:
        return "not_valley_derived"
    if w_overlap > values.get("overlap_warn", 0.05) or w_res > values.get("residual_warn", 0.20):
        return "projector_unreliable"

    if two_sector:
        eta_clean, eta_approx = _resolve_eta_thresholds(thresholds)
        clean_threshold = eta_clean
        approx_threshold = eta_approx
    else:
        pv_clean, pv_approx = _resolve_pv_thresholds(thresholds)
        clean_threshold = pv_clean
        approx_threshold = pv_approx

    if analysis_level == "raw_state":
        if polarization_score >= clean_threshold:
            return "raw_valley_clean"
        if polarization_score >= approx_threshold:
            return "raw_valley_approx"
        return "raw_valley_mixed"

    if analysis_level == "adapted_subspace":
        if polarization_score >= clean_threshold:
            return "valley_separable_subspace"
        if polarization_score >= approx_threshold:
            return "valley_approximately_separable_subspace"
        return "valley_mixed_subspace"

    return "not_valley_derived"


def derive_symmetry_status(
    *,
    symmetry_skipped: bool,
    little_group_passed: bool | None = None,
    valley_preserving: bool | None = None,
    local_irrep_ready: bool | None = None,
) -> str:
    if symmetry_skipped:
        return "not_requested"
    if little_group_passed is False:
        return "rejected_not_little_group"
    if valley_preserving is False:
        return "rejected_not_valley_preserving"
    if local_irrep_ready is True:
        return "local_irrep_ready"
    return "diagnostic_only"


def derive_derived_score(
    *,
    analysis_level: str,
    w_val: float | None = None,
    s_min: float | None = None,
) -> float:
    if analysis_level == "adapted_subspace" and s_min is not None:
        return float(s_min)
    if w_val is not None:
        return float(w_val)
    return 0.0


def derive_polarization_score(
    *,
    analysis_level: str,
    eta_raw: float | None = None,
    eta_adapted: np.ndarray | None = None,
    purity: float | None = None,
) -> float:
    if analysis_level == "adapted_subspace" and eta_adapted is not None and len(eta_adapted) > 0:
        return float(min(abs(float(v)) for v in eta_adapted))
    if eta_raw is not None:
        return float(abs(eta_raw))
    if purity is not None:
        return float(purity)
    return 0.0
