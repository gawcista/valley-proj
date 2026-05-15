import numpy as np
import pytest

from valleyscope.analysis.decision_tree import (
    derive_valley_status,
    derive_symmetry_status,
    derive_derived_score,
    derive_polarization_score,
    DEFAULT_THRESHOLDS,
)


class TestDeriveValleyStatus:
    def test_not_valley_derived_when_derived_score_below_threshold(self):
        status = derive_valley_status(
            analysis_level="raw_state",
            derived_score=0.3,
            polarization_score=0.9,
            w_overlap=0.0,
            w_res=0.0,
        )
        assert status == "not_valley_derived"

    def test_projector_unreliable_when_overlap_too_large(self):
        status = derive_valley_status(
            analysis_level="raw_state",
            derived_score=0.95,
            polarization_score=0.9,
            w_overlap=0.1,
            w_res=0.0,
        )
        assert status == "projector_unreliable"

    def test_projector_unreliable_when_residual_too_large(self):
        status = derive_valley_status(
            analysis_level="raw_state",
            derived_score=0.95,
            polarization_score=0.9,
            w_overlap=0.01,
            w_res=0.3,
        )
        assert status == "projector_unreliable"

    def test_raw_valley_clean_when_polarization_high(self):
        status = derive_valley_status(
            analysis_level="raw_state",
            derived_score=0.95,
            polarization_score=0.98,
            w_overlap=0.0,
            w_res=0.0,
        )
        assert status == "raw_valley_clean"

    def test_raw_valley_mixed_when_polarization_low(self):
        status = derive_valley_status(
            analysis_level="raw_state",
            derived_score=0.95,
            polarization_score=0.5,
            w_overlap=0.0,
            w_res=0.0,
        )
        assert status == "raw_valley_mixed"

    def test_valley_separable_subspace_when_s_min_and_eta_high(self):
        status = derive_valley_status(
            analysis_level="adapted_subspace",
            derived_score=0.95,
            polarization_score=0.98,
            w_overlap=0.0,
            w_res=0.0,
        )
        assert status == "valley_separable_subspace"

    def test_valley_mixed_subspace_when_polarization_low(self):
        status = derive_valley_status(
            analysis_level="adapted_subspace",
            derived_score=0.9,
            polarization_score=0.5,
            w_overlap=0.0,
            w_res=0.0,
        )
        assert status == "valley_mixed_subspace"

    def test_custom_thresholds_override_defaults(self):
        custom = {"W_val_min": 0.6, "overlap_warn": 0.15}
        status = derive_valley_status(
            analysis_level="raw_state",
            derived_score=0.7,
            polarization_score=0.5,
            w_overlap=0.12,
            w_res=0.0,
            thresholds=custom,
        )
        assert status == "raw_valley_mixed"

    def test_all_statuses_cover_controlled_vocabulary(self):
        valid = {
            "not_valley_derived",
            "projector_unreliable",
            "raw_valley_clean",
            "raw_valley_mixed",
            "valley_separable_subspace",
            "valley_mixed_subspace",
        }
        test_cases = [
            ("not_valley_derived", "raw_state", 0.3, 0.9, 0.0, 0.0),
            ("projector_unreliable", "raw_state", 0.9, 0.9, 0.1, 0.0),
            ("raw_valley_clean", "raw_state", 0.95, 0.98, 0.0, 0.0),
            ("raw_valley_mixed", "raw_state", 0.9, 0.5, 0.0, 0.0),
            ("valley_separable_subspace", "adapted_subspace", 0.95, 0.98, 0.0, 0.0),
            ("valley_mixed_subspace", "adapted_subspace", 0.9, 0.5, 0.0, 0.0),
        ]
        for expected, level, derived, pol, overlap, res in test_cases:
            result = derive_valley_status(
                analysis_level=level,
                derived_score=derived,
                polarization_score=pol,
                w_overlap=overlap,
                w_res=res,
            )
            assert result == expected, f"Expected {expected} for {level}"
            assert result in valid


class TestDeriveSymmetryStatus:
    def test_not_requested_when_symmetry_skipped(self):
        status = derive_symmetry_status(symmetry_skipped=True)
        assert status == "not_requested"

    def test_rejected_not_little_group(self):
        status = derive_symmetry_status(
            symmetry_skipped=False, little_group_passed=False,
        )
        assert status == "rejected_not_little_group"

    def test_rejected_not_valley_preserving(self):
        status = derive_symmetry_status(
            symmetry_skipped=False, little_group_passed=True,
            valley_preserving=False,
        )
        assert status == "rejected_not_valley_preserving"

    def test_topology_input_ready(self):
        status = derive_symmetry_status(
            symmetry_skipped=False, little_group_passed=True,
            valley_preserving=True, topology_input_ready=True,
        )
        assert status == "topology_input_ready"

    def test_diagnostic_only_when_passes_checks_but_not_ready(self):
        status = derive_symmetry_status(
            symmetry_skipped=False, little_group_passed=True,
            valley_preserving=True, topology_input_ready=False,
        )
        assert status == "diagnostic_only"

    def test_all_statuses_cover_controlled_vocabulary(self):
        valid = {
            "not_requested",
            "rejected_not_little_group",
            "rejected_not_valley_preserving",
            "diagnostic_only",
            "topology_input_ready",
        }
        cases = [
            ("not_requested", True, None, None, None),
            ("rejected_not_little_group", False, False, None, None),
            ("rejected_not_valley_preserving", False, True, False, None),
            ("diagnostic_only", False, True, True, False),
            ("topology_input_ready", False, True, True, True),
        ]
        for expected, skipped, lg, vp, ready in cases:
            result = derive_symmetry_status(
                symmetry_skipped=skipped, little_group_passed=lg,
                valley_preserving=vp, topology_input_ready=ready,
            )
            assert result == expected, f"Expected {expected}"
            assert result in valid


class TestDerivedScore:
    def test_raw_state_uses_w_val(self):
        score = derive_derived_score(analysis_level="raw_state", w_val=0.85)
        assert score == pytest.approx(0.85)

    def test_subspace_uses_s_min(self):
        score = derive_derived_score(analysis_level="adapted_subspace", w_val=0.5, s_min=0.92)
        assert score == pytest.approx(0.92)

    def test_returns_zero_when_no_value_available(self):
        score = derive_derived_score(analysis_level="raw_state")
        assert score == 0.0


class TestPolarizationScore:
    def test_raw_state_uses_abs_eta(self):
        score = derive_polarization_score(analysis_level="raw_state", eta_raw=-0.95)
        assert score == pytest.approx(0.95)

    def test_raw_state_eta_none_returns_zero(self):
        score = derive_polarization_score(analysis_level="raw_state", eta_raw=None)
        assert score == 0.0

    def test_subspace_uses_max_abs_eta(self):
        eta = np.array([0.98, -0.96, 0.5])
        score = derive_polarization_score(analysis_level="adapted_subspace", eta_adapted=eta)
        assert score == pytest.approx(0.98)

    def test_subspace_empty_returns_zero(self):
        score = derive_polarization_score(analysis_level="adapted_subspace", eta_adapted=np.array([]))
        assert score == 0.0


class TestDecisionTreeIntegration:
    """End-to-end test of the 4-step decision tree per AGENTS.md."""

    def test_full_four_step_clean_raw_state(self):
        """A clean isolated band: derived > threshold, |eta| > threshold, no overlap."""
        derived = derive_derived_score(analysis_level="raw_state", w_val=0.97)
        polarization = derive_polarization_score(analysis_level="raw_state", eta_raw=0.98)
        v_status = derive_valley_status(
            analysis_level="raw_state",
            derived_score=derived,
            polarization_score=polarization,
            w_overlap=0.0,
            w_res=0.03,
        )
        assert derived == pytest.approx(0.97)
        assert polarization == pytest.approx(0.98)
        assert v_status == "raw_valley_clean"

    def test_full_four_step_mixed_subspace(self):
        """A mixed subspace: s_min good but |eta| too low."""
        derived = derive_derived_score(analysis_level="adapted_subspace", s_min=0.92)
        polarization = derive_polarization_score(
            analysis_level="adapted_subspace", eta_adapted=np.array([0.5, -0.3])
        )
        v_status = derive_valley_status(
            analysis_level="adapted_subspace",
            derived_score=derived,
            polarization_score=polarization,
            w_overlap=0.0,
            w_res=0.0,
        )
        assert derived == pytest.approx(0.92)
        assert polarization == pytest.approx(0.5)
        assert v_status == "valley_mixed_subspace"

    def test_full_four_step_with_symmetry(self):
        """A separable subspace with verified rotation eigenvalue."""
        s_status = derive_symmetry_status(
            symmetry_skipped=False, little_group_passed=True,
            valley_preserving=True, topology_input_ready=True,
        )
        assert s_status == "topology_input_ready"

    def test_overlap_short_circuits_before_polarization_check(self):
        """Step 3 (projector reliability) gates before Step 2 (separation)."""
        status = derive_valley_status(
            analysis_level="raw_state",
            derived_score=0.95,
            polarization_score=0.99,
            w_overlap=0.08,
            w_res=0.0,
        )
        assert status == "projector_unreliable"


class TestEtaThresholdConversion:
    """P_v thresholds are converted to |eta| thresholds internally:
       |eta| = 2 * P_v - 1, so P_v_clean=0.95 -> |eta| >= 0.90."""

    def test_pv_clean_095_boundary_passes_eta_091(self):
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.91, w_overlap=0.0, w_res=0.0,
        )
        assert status == "raw_valley_clean"

    def test_pv_clean_095_eta_089_passes_approx_070(self):
        """|eta|=0.89 < 0.90 (eta_clean) but >= 0.70 (eta_approx), so still clean."""
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.89, w_overlap=0.0, w_res=0.0,
        )
        assert status == "raw_valley_clean"

    def test_eta_below_both_thresholds_is_mixed(self):
        """|eta|=0.60 < 0.70 (eta_approx) and < 0.90 (eta_clean), so mixed."""
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.60, w_overlap=0.0, w_res=0.0,
        )
        assert status == "raw_valley_mixed"

    def test_pv_approx_085_passes_eta_075(self):
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.75, w_overlap=0.0, w_res=0.0,
        )
        assert status == "raw_valley_clean"

    def test_explicit_eta_clean_overrides_pv_clean(self):
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.85, w_overlap=0.0, w_res=0.0,
            thresholds={"eta_clean": 0.80},
        )
        assert status == "raw_valley_clean"

    def test_explicit_eta_approx_with_default_pv_clean(self):
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.65, w_overlap=0.0, w_res=0.0,
            thresholds={"eta_approx": 0.60, "P_v_clean": 0.99},
        )
        assert status == "raw_valley_clean"

    def test_subspace_uses_converted_thresholds(self):
        status = derive_valley_status(
            analysis_level="adapted_subspace", derived_score=0.92,
            polarization_score=0.91, w_overlap=0.0, w_res=0.0,
        )
        assert status == "valley_separable_subspace"

    def test_custom_pv_clean_changes_eta_boundary(self):
        """Custom P_v_clean=0.80 -> |eta| >= 0.60 is clean."""
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.61, w_overlap=0.0, w_res=0.0,
            thresholds={"P_v_clean": 0.80},
        )
        assert status == "raw_valley_clean"


class TestMultiSectorPolarization:
    """For >2 sectors |eta| is undefined; use P_v directly as polarization_score."""

    def test_three_sector_high_purity_clean(self):
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.96, w_overlap=0.0, w_res=0.0,
            two_sector=False,
        )
        assert status == "raw_valley_clean"

    def test_three_sector_low_purity_mixed(self):
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.50, w_overlap=0.0, w_res=0.0,
            two_sector=False,
        )
        assert status == "raw_valley_mixed"

    def test_three_sector_pv_approx_boundary(self):
        """P_v=0.93 > 0.85 but < 0.95, passes P_v_approx so clean."""
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.93, w_overlap=0.0, w_res=0.0,
            two_sector=False,
        )
        assert status == "raw_valley_clean"

    def test_two_sector_still_uses_eta_conversion(self):
        status = derive_valley_status(
            analysis_level="raw_state", derived_score=0.95,
            polarization_score=0.91, w_overlap=0.0, w_res=0.0,
            two_sector=True,
        )
        assert status == "raw_valley_clean"

    def test_purity_fallback_when_eta_none(self):
        score = derive_polarization_score(
            analysis_level="raw_state", eta_raw=None, purity=0.96,
        )
        assert score == pytest.approx(0.96)

    def test_eta_priority_over_purity(self):
        score = derive_polarization_score(
            analysis_level="raw_state", eta_raw=-0.85, purity=0.96,
        )
        assert score == pytest.approx(0.85)
