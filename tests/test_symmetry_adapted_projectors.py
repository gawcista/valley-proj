import numpy as np
import pytest

from valleyscope.subspace.symmetry_adapted_projectors import (
    SymmetryAdaptedProjectors,
    build_symmetry_adapted_projectors_for_orbit,
    compute_projector_quality_diagnostics,
    select_projector_rank,
)


# -----------------------------------------------------------------------
# A. Exact identity case
# -----------------------------------------------------------------------

def test_identity_single_valley_exact():
    p_seed = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    d_e = np.eye(2, dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"V1": p_seed},
        representations={0: d_e},
        valley_mappings={0: {"V1": "V1"}},
        orbit=["V1"],
        reference_valley="V1",
        rank=1,
    )

    assert result.diagnostics.status == "ok"
    np.testing.assert_allclose(result.projectors["V1"], p_seed, atol=1e-12)
    assert result.diagnostics.selected_rank == 1
    assert result.diagnostics.rank_source == "user_specified"
    assert result.diagnostics.seed_overlap["V1"] == pytest.approx(1.0)
    assert result.diagnostics.completeness_source == "not_evaluated"


def test_identity_preserves_two_independent_valleys():
    p_a = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: d_e, 1: d_swap},
        valley_mappings={
            0: {"VA": "VA", "VB": "VB"},
            1: {"VA": "VB", "VB": "VA"},
        },
        orbit=["VA", "VB"],
        reference_valley="VA",
        rank=1,
    )

    assert result.diagnostics.status == "ok"
    np.testing.assert_allclose(result.projectors["VA"], p_a, atol=1e-12)
    np.testing.assert_allclose(result.projectors["VB"], p_b, atol=1e-12)


# -----------------------------------------------------------------------
# B. C3 three-valley cyclic mapping
# -----------------------------------------------------------------------

def _c3_three_valley_setup():
    p_m1 = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_m2 = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    p_m3 = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3
    d_e = np.eye(3, dtype=np.complex128)
    seeds = {"M1": p_m1, "M2": p_m2, "M3": p_m3}
    reps = {0: d_e, 1: d_c3, 2: d_c3sq}
    mappings = {
        0: {"M1": "M1", "M2": "M2", "M3": "M3"},
        1: {"M1": "M2", "M2": "M3", "M3": "M1"},
        2: {"M1": "M3", "M2": "M1", "M3": "M2"},
    }
    orbit = ["M1", "M2", "M3"]
    return seeds, reps, mappings, orbit


def test_c3_three_valley_symmetry_adapted_exact():
    seeds, reps, mappings, orbit = _c3_three_valley_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds,
        representations=reps,
        valley_mappings=mappings,
        orbit=orbit,
        reference_valley="M1",
        rank=1,
    )

    assert result.diagnostics.status == "ok"
    np.testing.assert_allclose(result.projectors["M1"], seeds["M1"], atol=1e-12)
    expected_m2 = reps[1] @ seeds["M1"] @ reps[1].conj().T
    np.testing.assert_allclose(result.projectors["M2"], expected_m2, atol=1e-12)
    expected_m3 = reps[1] @ result.projectors["M2"] @ reps[1].conj().T
    np.testing.assert_allclose(result.projectors["M3"], expected_m3, atol=1e-12)


def test_c3_symmetry_consistency_condition_holds():
    seeds, reps, mappings, orbit = _c3_three_valley_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )

    for op_id, d_g in reps.items():
        mapping = mappings[op_id]
        for src in orbit:
            tgt = mapping[src]
            transformed = d_g @ result.projectors[src] @ d_g.conj().T
            np.testing.assert_allclose(transformed, result.projectors[tgt], atol=1e-12)

    for err in result.diagnostics.projector_symmetry_error.values():
        assert err < 1e-12


def test_c3_orthogonality_and_idempotency():
    seeds, reps, mappings, orbit = _c3_three_valley_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )

    assert result.diagnostics.orthogonality_error < 1e-12
    assert result.diagnostics.total_projector_idempotency_error < 1e-12
    total = sum(result.projectors.values())
    np.testing.assert_allclose(total @ total, total, atol=1e-12)
    np.testing.assert_allclose(total, np.eye(3, dtype=np.complex128), atol=1e-12)


def test_c3_valley_sewing_matrices_unitary():
    """B_{ba}(g) = U_b^dag D_g U_a should be unitary in the exact case."""
    seeds, reps, mappings, orbit = _c3_three_valley_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )

    sewing = result.diagnostics.valley_sewing_matrices
    assert sewing is not None
    # B_{M2,M1}(C3): U_M2^dag D_C3 U_M1
    b_m2_m1 = sewing.get((1, "M1", "M2"))
    assert b_m2_m1 is not None
    assert b_m2_m1.shape == (1, 1)
    np.testing.assert_allclose(
        b_m2_m1.conj().T @ b_m2_m1, np.eye(1, dtype=np.complex128), atol=1e-12
    )
    # B_{M3,M1}(C3^2): U_M3^dag D_C3^2 U_M1
    b_m3_m1 = sewing.get((2, "M1", "M3"))
    assert b_m3_m1 is not None

    for key, err in result.diagnostics.sewing_unitarity_error.items():
        assert err < 1e-12, f"sewing unitarity error for {key}: {err}"


# -----------------------------------------------------------------------
# C. Full M-star toy: E, C3, C3^2, C2_M1, C2_M2, C2_M3
# -----------------------------------------------------------------------

def _full_mstar_setup():
    """4 operations: E, C3, C3^2, C2_M1.
    Orbit {M1, M2, M3}. C3 cycles, C2_M1 fixes M1 and swaps M2<->M3.
    Unique representatives: C3 for M1->M2, C3^2 for M1->M3.
    """
    p_m1 = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_m2 = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    p_m3 = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3
    d_c2_m1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex128)

    seeds = {"M1": p_m1, "M2": p_m2, "M3": p_m3}
    reps = {0: d_e, 1: d_c3, 2: d_c3sq, 3: d_c2_m1}
    mappings = {
        0: {"M1": "M1", "M2": "M2", "M3": "M3"},
        1: {"M1": "M2", "M2": "M3", "M3": "M1"},
        2: {"M1": "M3", "M2": "M1", "M3": "M2"},
        3: {"M1": "M1", "M2": "M3", "M3": "M2"},
    }
    orbit = ["M1", "M2", "M3"]
    return seeds, reps, mappings, orbit


def test_full_mstar_valley_preserving_subgroup_contains_e_and_c2_m1():
    """Reference M1: ops preserving M1 are E(0) and C2_M1(3)."""
    seeds, reps, mappings, orbit = _full_mstar_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )

    assert result.diagnostics.status == "ok"
    np.testing.assert_allclose(result.projectors["M1"], seeds["M1"], atol=1e-12)
    # M2 should be generated by C3 (op 1): D_C3 @ P_M1^sym @ D_C3^dag
    np.testing.assert_allclose(
        result.projectors["M2"],
        reps[1] @ result.projectors["M1"] @ reps[1].conj().T,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.projectors["M3"],
        reps[2] @ result.projectors["M1"] @ reps[2].conj().T,
        atol=1e-12,
    )


def test_full_mstar_symmetry_consistency_all_ops():
    seeds, reps, mappings, orbit = _full_mstar_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )

    for op_id, d_g in reps.items():
        mapping = mappings[op_id]
        for src in orbit:
            tgt = mapping[src]
            transformed = d_g @ result.projectors[src] @ d_g.conj().T
            np.testing.assert_allclose(transformed, result.projectors[tgt], atol=1e-12)


def test_full_mstar_c3_sewing_unitary():
    """B_{M2,M1}(C3) should be 1x1 unitary for valley-changing C3."""
    seeds, reps, mappings, orbit = _full_mstar_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )

    sewing = result.diagnostics.valley_sewing_matrices
    # C3: M1->M2 (op 1)
    b = sewing.get((1, "M1", "M2"))
    assert b is not None, "B_{M2,M1}(C3) should exist"
    np.testing.assert_allclose(
        b.conj().T @ b, np.eye(1, dtype=np.complex128), atol=1e-12
    )


# -----------------------------------------------------------------------
# D. C2 fixes one valley, swaps two (M1 as its own orbit)
# -----------------------------------------------------------------------

def _c2_fix_m1_setup():
    p_m1 = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_m2 = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    p_m3 = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_c2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    seeds = {"M1": p_m1, "M2": p_m2, "M3": p_m3}
    reps = {0: d_e, 3: d_c2}
    mappings = {
        0: {"M1": "M1", "M2": "M2", "M3": "M3"},
        3: {"M1": "M1", "M2": "M3", "M3": "M2"},
    }
    orbit = ["M1", "M2", "M3"]
    return seeds, reps, mappings, orbit


def test_c2_fixes_m1_valley_preserving_subgroup_symmetrizes():
    """M1's VPS = {E, C2}. P_M1^sym = (P_M1^0 + D_C2 P_M1^0 D_C2^dag)/2."""
    seeds, reps, mappings, orbit = _c2_fix_m1_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=["M1"],
        reference_valley="M1", rank=1,
    )

    assert result.diagnostics.status == "ok"
    np.testing.assert_allclose(result.projectors["M1"], seeds["M1"], atol=1e-12)


# -----------------------------------------------------------------------
# E. Non-symmetry-consistent seed → failed diagnostic
# -----------------------------------------------------------------------

def test_seed_inconsistent_with_mapping_gives_low_seed_overlap_warning():
    """Seed P_VB^0 = [[0.5,0.5,0],[0.5,0.5,0],[0,0,0]] is a projector orthogonal
    to the symmetry-adapted P_VB^sym = D_swap P_VA^sym D_swap^dag = [[0,0,0],[0,1,0],[0,0,0]].
    Tr(P_VB^sym P_VB^0) / rank = 0.5, triggering low seed overlap."""
    p_a = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: d_e, 1: d_swap},
        valley_mappings={
            0: {"VA": "VA", "VB": "VB"},
            1: {"VA": "VB", "VB": "VA"},
        },
        orbit=["VA", "VB"], reference_valley="VA", rank=1,
    )

    diag = result.diagnostics
    assert diag.status in {"warn", "failed"}
    assert "seed overlap" in diag.reason.lower()
    assert diag.seed_overlap["VB"] == pytest.approx(0.5, abs=0.01)


# -----------------------------------------------------------------------
# F. Rank selection
# -----------------------------------------------------------------------

def test_rank_selection_gap_method():
    eigvals = np.array([0.99, 0.98, 0.12, 0.11, 0.01])
    rank, gap, source = select_projector_rank(eigenvalues=eigvals, method="gap", tol=0.1)
    assert rank == 2
    assert gap == pytest.approx(0.86)
    assert source == "gap"


def test_rank_selection_threshold_method():
    eigvals = np.array([0.95, 0.92, 0.08, 0.03, 0.01])
    rank, gap, source = select_projector_rank(eigenvalues=eigvals, method="threshold", tol=0.5)
    assert rank == 2
    assert source == "threshold"


def test_rank_selection_gap_insufficient():
    eigvals = np.array([0.52, 0.48, 0.45, 0.42, 0.40])
    rank, gap, source = select_projector_rank(eigenvalues=eigvals, method="gap", tol=0.5)
    assert rank == 5
    assert source == "gap_insufficient"


def test_rank_ambiguity_reported_as_failed_in_orbit():
    """gap_insufficient must produce status='failed' in the full construction."""
    p_m1 = np.diag([0.52, 0.48, 0.45]).astype(np.complex128)
    p_m2 = np.diag([0.45, 0.52, 0.48]).astype(np.complex128)
    p_m3 = np.diag([0.48, 0.45, 0.52]).astype(np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"M1": p_m1, "M2": p_m2, "M3": p_m3},
        representations={0: d_e, 1: d_c3, 2: d_c3sq},
        valley_mappings={
            0: {"M1": "M1", "M2": "M2", "M3": "M3"},
            1: {"M1": "M2", "M2": "M3", "M3": "M1"},
            2: {"M1": "M3", "M2": "M1", "M3": "M2"},
        },
        orbit=["M1", "M2", "M3"], reference_valley="M1",
        rank_method="gap", rank_tol=0.5,
    )

    assert result.diagnostics.status == "failed"
    assert "rank gap insufficient" in result.diagnostics.reason
    assert result.diagnostics.rank_source == "gap_insufficient"


def test_user_specified_rank_overrides_auto():
    eigvals = np.array([0.99, 0.12, 0.11])
    rank, gap, source = select_projector_rank(eigenvalues=eigvals, rank=2)
    assert rank == 2
    assert source == "user_specified"


# -----------------------------------------------------------------------
# G. Missing / ambiguous representative operation
# -----------------------------------------------------------------------

def test_missing_representative_operation_fails():
    p_a = np.diag([1.0, 0.0]).astype(np.complex128)
    p_b = np.diag([0.0, 1.0]).astype(np.complex128)
    d_e = np.eye(2, dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: d_e},
        valley_mappings={0: {"VA": "VA"}},
        orbit=["VA", "VB"], reference_valley="VA", rank=1,
    )

    assert result.diagnostics.status == "failed"
    assert "no representative operation" in result.diagnostics.reason


def test_ambiguous_representative_operation_fails():
    """Two different ops map M1 -> M2 → must fail."""
    p_m1 = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_m2 = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    p_m3 = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_alt = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"M1": p_m1, "M2": p_m2, "M3": p_m3},
        representations={0: d_e, 1: d_c3, 99: d_alt},
        valley_mappings={
            0: {"M1": "M1", "M2": "M2", "M3": "M3"},
            1: {"M1": "M2", "M2": "M3", "M3": "M1"},
            99: {"M1": "M2", "M2": "M1", "M3": "M3"},  # also maps M1 -> M2!
        },
        orbit=["M1", "M2", "M3"], reference_valley="M1", rank=1,
    )

    assert result.diagnostics.status == "failed"
    assert "inequivalent" in result.diagnostics.reason.lower()


# -----------------------------------------------------------------------
# H. Orbit closure validation
# -----------------------------------------------------------------------

def test_orbit_not_closed_raises():
    p_a = np.diag([1.0, 0.0]).astype(np.complex128)
    p_b = np.diag([0.0, 1.0]).astype(np.complex128)
    d_e = np.eye(2, dtype=np.complex128)

    with np.testing.assert_raises(ValueError):
        build_symmetry_adapted_projectors_for_orbit(
            seed_projectors={"VA": p_a, "VB": p_b},
            representations={0: d_e},
            valley_mappings={0: {"VA": "VC"}},
            orbit=["VA", "VB"], reference_valley="VA", rank=1,
        )


# -----------------------------------------------------------------------
# I. No valley-preserving operation → explicit failure
# -----------------------------------------------------------------------

def test_no_valley_preserving_operation_for_reference_fails():
    p_a = np.diag([1.0, 0.0]).astype(np.complex128)
    p_b = np.diag([0.0, 1.0]).astype(np.complex128)
    d_swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={1: d_swap},
        valley_mappings={1: {"VA": "VB", "VB": "VA"}},
        orbit=["VA", "VB"], reference_valley="VA", rank=1,
    )

    assert result.diagnostics.status == "failed"
    assert "no valley-preserving operation" in result.diagnostics.reason


# -----------------------------------------------------------------------
# J. Completeness and idempotency diagnostics
# -----------------------------------------------------------------------

def test_completeness_with_expected_total_projector():
    p1 = np.diag([1.0, 0.0, 0.0, 0.0]).astype(np.complex128)
    p2 = np.diag([0.0, 1.0, 0.0, 0.0]).astype(np.complex128)
    d_e = np.eye(4, dtype=np.complex128)
    d_swap = np.eye(4, dtype=np.complex128)
    d_swap[0, 0] = 0; d_swap[1, 1] = 0; d_swap[0, 1] = 1; d_swap[1, 0] = 1

    i_expected = np.diag([1.0, 1.0, 0.0, 0.0]).astype(np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"V1": p1, "V2": p2},
        representations={0: d_e, 1: d_swap},
        valley_mappings={
            0: {"V1": "V1", "V2": "V2"},
            1: {"V1": "V2", "V2": "V1"},
        },
        orbit=["V1", "V2"], reference_valley="V1", rank=1,
        expected_total_projector=i_expected,
    )

    assert result.diagnostics.completeness_source == "expected_total_projector"
    assert result.diagnostics.completeness_error is not None
    assert result.diagnostics.completeness_error < 1e-12
    assert result.diagnostics.total_projector_idempotency_error < 1e-12


def test_quality_diagnostics_on_perfect_projectors():
    n = 4
    p1 = np.diag([1.0, 0.0, 0.0, 0.0]).astype(np.complex128)
    p2 = np.diag([0.0, 1.0, 0.0, 0.0]).astype(np.complex128)
    u1 = np.zeros((n, 1), dtype=np.complex128); u1[0, 0] = 1.0
    u2 = np.zeros((n, 1), dtype=np.complex128); u2[1, 0] = 1.0

    diag = compute_projector_quality_diagnostics(
        projectors={"V1": p1, "V2": p2},
        eigenvectors={"V1": u1, "V2": u2},
        seed_projectors={"V1": p1, "V2": p2},
        representations={0: np.eye(n, dtype=np.complex128)},
        valley_mappings={0: {"V1": "V1", "V2": "V2"}},
        orbit=["V1", "V2"], reference_valley="V1",
        selected_rank=1, eigenvalues=np.ones(n),
        purification_gap=1.0, rank_source="user_specified",
    )

    assert diag.orthogonality_error < 1e-12
    assert diag.total_projector_idempotency_error < 1e-12
    assert diag.completeness_source == "not_evaluated"
    assert diag.seed_overlap["V1"] == pytest.approx(1.0)
    # Projector overlap check
    assert diag.projector_overlap_matrices is not None
    assert diag.projector_overlap_deviation[("V1", "V2")] < 1e-12


# -----------------------------------------------------------------------
# K. Representative ambiguity resolution
# -----------------------------------------------------------------------

def test_equivalent_candidates_resolved_with_representative_resolution():
    """Two different operation_ids that produce identical projectors →
    accepted with representative_resolution='equivalent_candidates'.
    Orbit has only 2 valleys to avoid needing a 3rd representative."""
    p_a = np.diag([1.0, 0.0]).astype(np.complex128)
    p_b = np.diag([0.0, 1.0]).astype(np.complex128)
    d_e = np.eye(2, dtype=np.complex128)
    d_swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: d_e, 1: d_swap, 99: d_swap.copy()},
        valley_mappings={
            0: {"VA": "VA", "VB": "VB"},
            1: {"VA": "VB", "VB": "VA"},
            99: {"VA": "VB", "VB": "VA"},
        },
        orbit=["VA", "VB"], reference_valley="VA", rank=1,
    )

    assert result.diagnostics.status == "ok"
    assert result.diagnostics.representative_resolution == "equivalent_candidates"
    assert len(result.diagnostics.representative_candidates) == 2


def test_equivalent_candidates_preserves_inequivalent_failure():
    """Verify the old ambiguous test still fails for genuinely inequivalent ops."""
    p_m1 = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_m2 = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    p_m3 = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_diff = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"M1": p_m1, "M2": p_m2, "M3": p_m3},
        representations={0: d_e, 1: d_c3, 2: d_c3sq, 99: d_diff},
        valley_mappings={
            0: {"M1": "M1", "M2": "M2", "M3": "M3"},
            1: {"M1": "M2", "M2": "M3", "M3": "M1"},
            2: {"M1": "M3", "M2": "M1", "M3": "M2"},
            99: {"M1": "M2", "M2": "M1", "M3": "M3"},
        },
        orbit=["M1", "M2", "M3"], reference_valley="M1", rank=1,
    )

    assert result.diagnostics.status == "failed"
    assert "inequivalent" in result.diagnostics.reason.lower()


def test_representative_resolution_field_in_diagnostics():
    """Both unique and resolved paths populate the field."""
    p_m1 = np.diag([1.0, 0.0]).astype(np.complex128)
    p_m2 = np.diag([0.0, 1.0]).astype(np.complex128)
    d_swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"VA": p_m1, "VB": p_m2},
        representations={0: np.eye(2, dtype=np.complex128), 1: d_swap},
        valley_mappings={
            0: {"VA": "VA", "VB": "VB"},
            1: {"VA": "VB", "VB": "VA"},
        },
        orbit=["VA", "VB"], reference_valley="VA", rank=1,
    )

    assert result.diagnostics.status == "ok"
    assert result.diagnostics.representative_resolution == "unique"
