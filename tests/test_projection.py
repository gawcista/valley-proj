import numpy as np
import pytest

from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.projection.sector_projectors import (
    adjust_centers_for_parent_valley,
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


# --- k_resolved_parent_valley / parent-valley projector tests ---

MOIRE_RECIP = np.array([
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 1.0],
])


def test_fixed_center_gives_zero_weight_for_far_k():
    """fixed_center W_val=0 when k is far from all fixed centers."""
    centers = [ValleyCenter("V0", np.array([0.0, 0.0, 0.0]))]
    sectors = [ValleySector("V0_sector", ["V0"])]
    k_cart = np.array([2.5, 0.0, 0.0])
    q_cart = k_cart.reshape(1, 3)

    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.3)
    result = compute_valley_weights(coeffs_for([1.0]), projectors)[0]

    # fixed_center: center at (0,0,0), q at (2.5,0,0) — distance 2.5 >> 0.3
    assert result.w_val == pytest.approx(0.0)
    assert result.center_weights["V0"] == pytest.approx(0.0)


def test_k_resolved_parent_valley_recovers_weight_for_far_k():
    """k_resolved_parent_valley recovers parent-valley weight for a state far from fixed center."""
    centers = [ValleyCenter("V0", np.array([0.0, 0.0, 0.0]))]
    sectors = [ValleySector("V0_sector", ["V0"])]
    k_cart = np.array([2.5, 0.0, 0.0])
    q_cart = k_cart.reshape(1, 3)

    effective = adjust_centers_for_parent_valley(centers, k_cart, MOIRE_RECIP)
    projectors = build_sector_projectors(q_cart, effective, sectors, RECIP, qcut=0.3)
    result = compute_valley_weights(coeffs_for([1.0]), projectors)[0]

    # k_resolved_parent_valley: dynamic center = k_M + 0 = (2.5, 0, 0)
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


def test_projector_mode_invalid_raises_config_error():
    """Config rejects invalid projector_mode strings."""
    from valleyscope.io.config import _normalize_projector_mode
    import pytest
    with pytest.raises(ValueError, match="projector_mode"):
        _normalize_projector_mode("bogus")


def test_projector_mode_aliases_normalize():
    """Deprecated aliases normalize to canonical values."""
    from valleyscope.io.config import _normalize_projector_mode
    assert _normalize_projector_mode("fixed_point") == "fixed_center"
    assert _normalize_projector_mode("folded_family") == "k_resolved_parent_valley"
    assert _normalize_projector_mode("fixed_center") == "fixed_center"
    assert _normalize_projector_mode("k_resolved_parent_valley") == "k_resolved_parent_valley"


def test_fold_boundary_plus_half_wraps_to_minus_half():
    """+0.5 fractional boundary wraps to -0.5 with correct G-shift."""
    # Center at frac=(0.5, 0, 0). In moire BZ coord, this is on the boundary.
    # MOIRE_RECIP[:2,:2] = [[0.5,0],[0,0.5]] -> inv = [[2,0],[0,2]]
    # cart (0.25, 0) -> frac = (0.5, 0) -> should wrap to (-0.5, 0), g_int = (1, 0)
    center_cart = np.array([0.25, 0.0, 0.0])
    folded_frac, g_int, folded_cart = fold_center_into_moire_bz(center_cart, MOIRE_RECIP)

    assert folded_frac[0] == pytest.approx(-0.5, abs=1e-10)
    assert folded_frac[1] == pytest.approx(0.0)
    assert g_int[0] == 1
    assert folded_cart[0] == pytest.approx(-0.25, abs=1e-10)


def test_fold_boundary_minus_half_unchanged():
    """-0.5 fractional boundary stays -0.5."""
    # cart (-0.25, 0) -> frac = (-0.5, 0) -> already in [-0.5, 0.5)
    center_cart = np.array([-0.25, 0.0, 0.0])
    folded_frac, g_int, folded_cart = fold_center_into_moire_bz(center_cart, MOIRE_RECIP)

    assert folded_frac[0] == pytest.approx(-0.5, abs=1e-10)
    assert folded_frac[1] == pytest.approx(0.0)
    # No wrapping needed: frac_raw = -0.5, rint(-0.5) = 0
    assert g_int[0] == 0


def test_center_weights_are_raw_window_weights():
    """Center weights are raw window weights, not exclusive after overlap handling."""
    # Two overlapping centers in the same sector
    centers = [
        ValleyCenter("A", np.array([0.0, 0.0, 0.0])),
        ValleyCenter("B", np.array([0.2, 0.0, 0.0])),
    ]
    sectors = [ValleySector("AB", ["A", "B"])]
    q_cart = np.array([[0.15, 0.0, 0.0]])

    projectors = build_sector_projectors(q_cart, centers, sectors, RECIP, qcut=0.3)
    result = compute_valley_weights(coeffs_for([1.0]), projectors)[0]

    # Both centers' raw masks include the q-point (distance 0.15 < 0.3 for A, 0.05 < 0.3 for B).
    # Sector weight after overlap exclusion is 1.0 (single-sector, overlap only across sectors).
    # But center_weights are raw: both nonzero.
    assert result.center_weights["A"] > 0.0
    assert result.center_weights["B"] > 0.0
    assert result.sector_weights["AB"] == pytest.approx(1.0)


def test_deprecated_alias_still_works():
    """adjust_centers_for_folded_family is callable alias."""
    from valleyscope.projection.sector_projectors import adjust_centers_for_folded_family
    centers = [ValleyCenter("V0", np.array([0.0, 0.0, 0.0]))]
    k_cart = np.array([0.0, 0.0, 0.0])
    result = adjust_centers_for_folded_family(centers, k_cart, MOIRE_RECIP)
    assert result[0].name == "V0"
