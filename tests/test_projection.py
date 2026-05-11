import numpy as np
import pytest

from valley_proj.geometry.valley_centers import ValleyCenter, ValleySector
from valley_proj.projection.sector_projectors import build_sector_projectors
from valley_proj.projection.weights import classify_valley_weights, compute_valley_weights
from valley_proj.projection.qcut_scan import scan_qcut


RECIP = np.array(
    [
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def coeffs_for(amplitudes):
    arr = np.asarray(amplitudes, dtype=np.complex128).reshape(1, 1, -1)
    norm = np.linalg.norm(arr)
    if norm:
        arr = arr / norm
    return arr


def two_sector_setup():
    centers = [
        ValleyCenter("K", np.array([0.0, 0.0, 0.0])),
        ValleyCenter("Kp", np.array([5.0, 0.0, 0.0])),
    ]
    sectors = [ValleySector("K_sector", ["K"]), ValleySector("Kp_sector", ["Kp"])]
    return centers, sectors


def test_pure_target_sector_state_has_unit_weight():
    centers, sectors = two_sector_setup()
    q_cart = np.array([[0.1, 0.0, 0.0], [8.0, 0.0, 0.0]])

    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.5)
    result = compute_valley_weights(coeffs_for([1.0, 0.0]), projectors)[0]

    assert result.sector_weights["K_sector"] == pytest.approx(1.0)
    assert result.sector_weights["Kp_sector"] == pytest.approx(0.0)
    assert result.w_val == pytest.approx(1.0)
    assert result.purity == pytest.approx(1.0)
    assert result.leakage == pytest.approx(0.0)


def test_opposite_sector_state_has_negative_two_valley_eta():
    centers, sectors = two_sector_setup()
    q_cart = np.array([[0.0, 0.0, 0.0], [5.1, 0.0, 0.0]])

    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.5)
    result = compute_valley_weights(coeffs_for([0.0, 1.0]), projectors)[0]

    assert result.sector_weights["K_sector"] == pytest.approx(0.0)
    assert result.sector_weights["Kp_sector"] == pytest.approx(1.0)
    assert result.eta == pytest.approx(-1.0)


def test_equal_weight_mixed_state_has_half_purity_and_zero_eta():
    centers, sectors = two_sector_setup()
    q_cart = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.5)
    result = compute_valley_weights(coeffs_for([1.0, 1.0]), projectors)[0]

    assert result.w_val == pytest.approx(1.0)
    assert result.purity == pytest.approx(0.5)
    assert result.eta == pytest.approx(0.0)


def test_leakage_state_reports_non_valley_weight():
    centers, sectors = two_sector_setup()
    q_cart = np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 0.0]])

    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.5)
    result = compute_valley_weights(coeffs_for([1.0, 1.0]), projectors)[0]

    assert result.w_val == pytest.approx(0.5)
    assert result.leakage == pytest.approx(0.5)


def test_same_sector_overlapping_windows_are_counted_once():
    centers = [
        ValleyCenter("top_K", np.array([0.0, 0.0, 0.0])),
        ValleyCenter("bottom_K", np.array([0.2, 0.0, 0.0])),
    ]
    sectors = [ValleySector("K_sector", ["top_K", "bottom_K"])]
    q_cart = np.array([[0.1, 0.0, 0.0]])

    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.3)
    result = compute_valley_weights(coeffs_for([1.0]), projectors)[0]

    assert projectors.sector_masks["K_sector"].tolist() == [True]
    assert result.sector_weights["K_sector"] == pytest.approx(1.0)
    assert result.w_val == pytest.approx(1.0)


def test_cross_sector_overlap_is_excluded_and_reported():
    centers = [
        ValleyCenter("A", np.array([0.0, 0.0, 0.0])),
        ValleyCenter("B", np.array([0.2, 0.0, 0.0])),
    ]
    sectors = [ValleySector("A_sector", ["A"]), ValleySector("B_sector", ["B"])]
    q_cart = np.array([[0.1, 0.0, 0.0]])

    with pytest.warns(UserWarning, match="ambiguous"):
        projectors = build_sector_projectors(
            q_cart,
            centers,
            sectors,
            RECIP,
            qcut=0.3,
            ambiguous_policy="warn_exclude",
        )
    result = compute_valley_weights(coeffs_for([1.0]), projectors)[0]

    assert projectors.ambiguous_mask.tolist() == [True]
    assert result.ambiguous_weight == pytest.approx(1.0)
    assert result.w_val == pytest.approx(0.0)


def test_classification_thresholds_are_explicit():
    result = classify_valley_weights(w_val=0.9, purity=0.9, thresholds=None)
    assert result["valley_derived"] is True
    assert result["valley_clean"] == "approximate"


def test_qcut_scan_reports_plateau_for_stable_weights():
    centers, sectors = two_sector_setup()
    q_cart = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    coefficients = coeffs_for([1.0, 0.0])

    results = scan_qcut(
        q_cart,
        coefficients,
        centers,
        sectors,
        RECIP,
        qcuts=[0.2, 0.3, 0.4],
    )

    assert [entry.qcut for entry in results.entries] == [0.2, 0.3, 0.4]
    assert results.has_plateau is True
