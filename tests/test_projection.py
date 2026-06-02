import numpy as np
import pytest

from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.projection.sector_projectors import (
    adjust_centers_for_folded_family,
    build_sector_projectors,
)
from valleyscope.projection.weights import classify_valley_weights, compute_valley_weights
from valleyscope.projection.qcut_scan import scan_qcut
from valleyscope.projection.folded_center import (
    build_folded_center_report,
    fold_center_into_moire_bz,
)


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
    assert result.residual_weight == pytest.approx(0.0)


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


def test_residual_weight_reports_non_valley_weight():
    centers, sectors = two_sector_setup()
    q_cart = np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 0.0]])

    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.5)
    result = compute_valley_weights(coeffs_for([1.0, 1.0]), projectors)[0]

    assert result.w_val == pytest.approx(0.5)
    assert result.residual_weight == pytest.approx(0.5)


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

    with pytest.warns(UserWarning, match="overlap"):
        projectors = build_sector_projectors(
            q_cart,
            centers,
            sectors,
            RECIP,
            qcut=0.3,
            overlap_policy="warn_exclude",
        )
    result = compute_valley_weights(coeffs_for([1.0]), projectors)[0]

    assert projectors.overlap_mask.tolist() == [True]
    assert result.overlap_weight == pytest.approx(1.0)
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
    assert [entry.overlap_count for entry in results.entries] == [0, 0, 0]
    assert results.has_plateau is True


def test_projector_uses_reciprocal_torus_wrapping_for_far_g_vectors():
    centers = [ValleyCenter("K", np.array([1.0, 1.0, 0.0]))]
    sectors = [ValleySector("K_sector", ["K"])]
    q_cart = np.array([[1.0 + 40.0, 1.0 - 30.0, 0.0]])

    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.2)
    result = compute_valley_weights(coeffs_for([1.0]), projectors)[0]

    assert projectors.center_masks["K"].tolist() == [True]
    assert result.w_val == pytest.approx(1.0)


# --- folded_family / k-dependent projector tests ---

MOIRE_RECIP = np.array([
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 1.0],
])


def test_fixed_point_gives_zero_weight_for_far_k():
    """fixed_point W_val=0 when k is far from all fixed centers."""
    # Center at origin, k-point far away at (5, 0) in frac -> (2.5, 0) cart
    centers = [ValleyCenter("V0", np.array([0.0, 0.0, 0.0]))]
    sectors = [ValleySector("V0_sector", ["V0"])]
    # k at (5.0, 0.0) frac in moire BZ => cart = (2.5, 0.0, 0.0)
    k_cart = np.array([2.5, 0.0, 0.0])
    # G=0 only plane-wave component
    q_cart = k_cart.reshape(1, 3)

    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.3)
    result = compute_valley_weights(coeffs_for([1.0]), projectors)[0]

    # fixed_point: center at (0,0,0), q at (2.5,0,0) — distance 2.5 >> 0.3
    assert result.w_val == pytest.approx(0.0)
    assert result.center_weights["V0"] == pytest.approx(0.0)


def test_folded_family_recovers_valley_weight_for_far_k():
    """folded_family recovers valley-family weight for a state far from fixed center."""
    # Center V0 at (0,0) cart. Folded into moire BZ: V0 = (0,0) + 0*G_moire
    # k_M at (2.5, 0, 0) cart. G_a^M = (0,0)
    # Dynamic center: Q_a(k_M) = k_M + G_a^M = (2.5, 0, 0)
    centers = [ValleyCenter("V0", np.array([0.0, 0.0, 0.0]))]
    sectors = [ValleySector("V0_sector", ["V0"])]
    k_cart = np.array([2.5, 0.0, 0.0])
    q_cart = k_cart.reshape(1, 3)

    effective = adjust_centers_for_folded_family(centers, k_cart, MOIRE_RECIP)
    projectors = build_sector_projectors(q_cart, effective, sectors, RECIP, qcut=0.3)
    result = compute_valley_weights(coeffs_for([1.0]), projectors)[0]

    # folded_family: dynamic center = k_M + 0 = (2.5, 0, 0), q = same => distance 0
    assert result.w_val == pytest.approx(1.0)
    assert result.center_weights["V0"] == pytest.approx(1.0)


def test_center_weights_sum_to_sector_weights_non_overlapping():
    """Center-resolved weights equal sector weights for non-overlapping centers."""
    centers = [
        ValleyCenter("A", np.array([0.0, 0.0, 0.0])),
        ValleyCenter("B", np.array([2.0, 0.0, 0.0])),
    ]
    sectors = [ValleySector("AB_sector", ["A", "B"])]
    # q at center A, not near B
    q_cart = np.array([[0.0, 0.0, 0.0]])
    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.5)
    result = compute_valley_weights(coeffs_for([1.0]), projectors)[0]

    assert result.center_weights["A"] == pytest.approx(1.0)
    assert result.center_weights["B"] == pytest.approx(0.0)
    assert result.sector_weights["AB_sector"] == pytest.approx(1.0)
    # Sum of center weights = sector weight for non-overlapping centers
    assert result.center_weights["A"] + result.center_weights["B"] == pytest.approx(
        result.sector_weights["AB_sector"]
    )


def test_folded_center_handles_moire_reciprocal_periodic_equivalence():
    """Folded-center report handles centers separated by moire reciprocal vectors."""
    # Center at (0.6, 0, 0) in moire frac coords.
    # In MOIRE_RECIP: frac = cart @ inv(MOIRE_RECIP[:2,:2])
    # MOIRE_RECIP[:2,:2] = [[0.5, 0], [0, 0.5]] -> inv = [[2, 0], [0, 2]]
    # cart (0.3, 0) -> frac = (0.6, 0) -> folded = (0.6-1, 0) = (-0.4, 0)
    center_cart = np.array([0.3, 0.0, 0.0])
    folded_frac, g_int, folded_cart = fold_center_into_moire_bz(center_cart, MOIRE_RECIP)

    assert folded_frac[0] == pytest.approx(-0.4, abs=1e-10)
    assert folded_frac[1] == pytest.approx(0.0)
    assert g_int[0] == 1  # wrapped by one moire reciprocal vector
    assert folded_cart[0] == pytest.approx(-0.2, abs=1e-10)


def test_sampled_k_coverage_detects_one_sided_branch():
    """Sampled-k coverage detects one-sided sampling in a simple 1D path."""
    centers = [ValleyCenter("V0", np.array([0.0, 0.0, 0.0]))]
    # Sampled k on one side only (all positive x)
    k_frac = {
        "k1": np.array([0.1, 0.0, 0.0]),
        "k2": np.array([0.3, 0.0, 0.0]),
        "k3": np.array([0.5, 0.0, 0.0]),
    }
    report = build_folded_center_report(centers, MOIRE_RECIP, k_frac)

    # Distances should be computed
    assert "V0" in report.kpoint_distances
    assert len(report.kpoint_distances["V0"]) == 3

    # All k-points are on one side of folded center V0 (which is at origin)
    # in fractional space: k_frac values are (0.1, 0.3, 0.5) all positive.
    from valleyscope.workflows.analyze_hsp import _build_sampled_k_coverage

    coverage = _build_sampled_k_coverage(
        folded_center_report=report,
        kpoint_names=["k1", "k2", "k3"],
        kpoint_frac_by_name=k_frac,
    )
    assert len(coverage["one_sided_branch_warnings"]) >= 1
