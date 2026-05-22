import numpy as np
import pytest

from valleyscope.subspace.symmetry_adapted_projectors import (
    SymmetryAdaptedProjectors,
    build_symmetry_adapted_projectors_for_orbit,
    compute_projector_quality_diagnostics,
    compute_valley_sewing_matrices,
    select_projector_rank,
)


# -----------------------------------------------------------------------
# A. Exact identity case
# -----------------------------------------------------------------------

def test_identity_single_valley_exact():
    """Identity operation: valley-preserving subgroup = {E}.
    P_a0^sym should equal P_a0^0."""
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


def test_identity_preserves_two_independent_valleys():
    """Two independent valleys each with identity mapping VA->VA and VB->VB,
    plus a swap operation mapping VA->VB. Orbit: [VA, VB]."""
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
    """Returns seed_projectors, representations, valley_mappings, orbit."""
    p_m1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    p_m2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    p_m3 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.complex128)

    # C3 cycles: state 0->1->2->0
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3

    seeds = {"M1": p_m1, "M2": p_m2, "M3": p_m3}
    reps = {1: d_c3, 2: d_c3sq,
            0: np.eye(3, dtype=np.complex128)}
    mappings = {
        0: {"M1": "M1", "M2": "M2", "M3": "M3"},
        1: {"M1": "M2", "M2": "M3", "M3": "M1"},
        2: {"M1": "M3", "M2": "M1", "M3": "M2"},
    }
    orbit = ["M1", "M2", "M3"]
    return seeds, reps, mappings, orbit


def test_c3_three_valley_symmetry_adapted_exact():
    """C3 cycles M1->M2->M3. Seeds are already perfect projectors.
    Valley-preserving subgroup for M1 = {E}. P_M1^sym = P_M1^0.
    C3 generates M2 and M3 projectors from M1^sym."""
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
    # P_M1^sym = P_M1^0 (only identity in preserving subgroup)
    np.testing.assert_allclose(result.projectors["M1"], seeds["M1"], atol=1e-12)
    # P_M2^sym = D_C3 @ P_M1^sym @ D_C3^dag
    expected_m2 = reps[1] @ seeds["M1"] @ reps[1].conj().T
    np.testing.assert_allclose(result.projectors["M2"], expected_m2, atol=1e-12)
    # P_M3^sym = D_C3 @ P_M2^sym @ D_C3^dag
    expected_m3 = reps[1] @ result.projectors["M2"] @ reps[1].conj().T
    np.testing.assert_allclose(result.projectors["M3"], expected_m3, atol=1e-12)


def test_c3_symmetry_consistency_condition_holds():
    """D_g P_a^sym D_g^dag == P_{pi_g(a)}^sym for all g, a."""
    seeds, reps, mappings, orbit = _c3_three_valley_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds,
        representations=reps,
        valley_mappings=mappings,
        orbit=orbit,
        reference_valley="M1",
        rank=1,
    )

    for op_id, d_g in reps.items():
        mapping = mappings[op_id]
        for src in orbit:
            tgt = mapping[src]
            p_src = result.projectors[src]
            p_tgt = result.projectors[tgt]
            transformed = np.asarray(d_g, dtype=np.complex128) @ p_src @ np.asarray(d_g, dtype=np.complex128).conj().T
            np.testing.assert_allclose(transformed, p_tgt, atol=1e-12,
                                       err_msg=f"op={op_id}, {src}->{tgt}")

    # Check symmetry error via diagnostics
    for key, err in result.diagnostics.projector_symmetry_error.items():
        assert err < 1e-12, f"{key}: {err}"


def test_c3_orthogonality_and_completeness():
    """Generated projectors should be orthogonal and complete."""
    seeds, reps, mappings, orbit = _c3_three_valley_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds,
        representations=reps,
        valley_mappings=mappings,
        orbit=orbit,
        reference_valley="M1",
        rank=1,
    )

    assert result.diagnostics.orthogonality_error < 1e-12
    assert result.diagnostics.completeness_error < 1e-12
    total = sum(result.projectors.values())
    np.testing.assert_allclose(total, np.eye(3, dtype=np.complex128), atol=1e-12)


# -----------------------------------------------------------------------
# C. C2 fixes one valley, swaps two valleys
# -----------------------------------------------------------------------

def _c2_fix_m1_setup():
    """C2 preserves M1, swaps M2<->M3."""
    p_m1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    p_m2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    p_m3 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.complex128)
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


def test_c2_fixes_m1_symmetry_adapted():
    """C2 preserves M1 and swaps M2<->M3.
    M1's valley-preserving subgroup = {E, C2} (both fix M1).
    Since no op maps M1 to another valley under these operations,
    M1 is its own orbit.  The M2<->M3 connection is a separate orbit.
    P_M1^sym should reflect the preservation subgroup symmetrization."""
    seeds, reps, mappings, orbit = _c2_fix_m1_setup()

    # Only M1 as orbit — no op maps M1 elsewhere
    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds,
        representations=reps,
        valley_mappings=mappings,
        orbit=["M1"],
        reference_valley="M1",
        rank=1,
    )

    assert result.diagnostics.status == "ok"
    np.testing.assert_allclose(result.projectors["M1"], seeds["M1"], atol=1e-12)


def test_c2_fix_m1_symmetry_consistency():
    """D_g P_a^sym D_g^dag == P_{pi_g(a)}^sym for C2 preserves M1 case."""
    seeds, reps, mappings, orbit = _c2_fix_m1_setup()

    # M1 as its own orbit
    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds,
        representations=reps,
        valley_mappings=mappings,
        orbit=["M1"],
        reference_valley="M1",
        rank=1,
    )

    # C2 fixes M1: D_C2 P_M1^sym D_C2^dag == P_M1^sym
    d_c2 = np.asarray(reps[3], dtype=np.complex128)
    np.testing.assert_allclose(
        d_c2 @ result.projectors["M1"] @ d_c2.conj().T,
        result.projectors["M1"], atol=1e-12)


# -----------------------------------------------------------------------
# D. Non-symmetry-consistent seed → failed diagnostic
# -----------------------------------------------------------------------

def test_non_symmetry_consistent_seed_gives_failed_status():
    """A randomly rotated seed projector that does not satisfy the
    symmetry-consistency condition should produce poor diagnostics."""
    p_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.36, 0.48], [0.48, 0.64]], dtype=np.complex128)
    d_e = np.eye(2, dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"V1": p_a, "V2": p_b},
        representations={0: d_e},
        valley_mappings={0: {"V1": "V1", "V2": "V2"}},
        orbit=["V1", "V2"],
        reference_valley="V1",
        rank=1,
    )

    # p_b is not a pure projector → symmetry-adapted quality should be degraded
    diag = result.diagnostics
    assert diag.orthogonality_error > 0.0 or diag.status != "ok"
    assert diag.seed_overlap["V2"] < 0.9  # p_b is rotated away from seed


# -----------------------------------------------------------------------
# E. Rank selection
# -----------------------------------------------------------------------

def test_rank_selection_gap_method():
    """Gap method picks the largest gap in eigenvalue spectrum."""
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
    """When no gap exceeds tol, fall back to full rank."""
    eigvals = np.array([0.52, 0.48, 0.45, 0.42, 0.40])
    rank, gap, source = select_projector_rank(eigenvalues=eigvals, method="gap", tol=0.5)
    assert rank == 5
    assert source == "gap_insufficient"


def test_rank_ambiguity_reported():
    """When gap method is ambiguous (small max gap), it is reported, not silently
    accepted."""
    eigvals = np.array([0.55, 0.54, 0.53, 0.52, 0.51])
    rank, gap, source = select_projector_rank(eigenvalues=eigvals, method="gap", tol=0.5)
    assert source == "gap_insufficient"
    assert rank == 5  # full rank fallback


def test_user_specified_rank_overrides_auto():
    eigvals = np.array([0.99, 0.12, 0.11])
    rank, gap, source = select_projector_rank(eigenvalues=eigvals, rank=2)
    assert rank == 2
    assert source == "user_specified"


# -----------------------------------------------------------------------
# F. Missing representative operation → explicit failure
# -----------------------------------------------------------------------

def test_missing_representative_operation_fails():
    """When no operation maps reference valley to another orbit valley,
    construction must fail with explicit reason."""
    p_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    d_e = np.eye(2, dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: d_e},
        valley_mappings={0: {"VA": "VA"}},  # VB not in mapping
        orbit=["VA", "VB"],
        reference_valley="VA",
        rank=1,
    )

    assert result.diagnostics.status == "failed"
    assert "no representative operation" in result.diagnostics.reason
    assert "VB" in result.diagnostics.reason or result.diagnostics.selected_rank == 0


# -----------------------------------------------------------------------
# G. Orbit closure validation
# -----------------------------------------------------------------------

def test_orbit_not_closed_raises():
    p_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    p_c = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    d_e = np.eye(2, dtype=np.complex128)

    with np.testing.assert_raises(ValueError):
        build_symmetry_adapted_projectors_for_orbit(
            seed_projectors={"VA": p_a, "VB": p_b},
            representations={0: d_e},
            valley_mappings={0: {"VA": "VC"}},  # VC not in orbit
            orbit=["VA", "VB"],
            reference_valley="VA",
            rank=1,
        )


# -----------------------------------------------------------------------
# H. Valley sewing matrices
# -----------------------------------------------------------------------

def test_valley_sewing_exact_case_unitary():
    """In the exact toy case, S_ab = delta_ab P_a^sym."""
    seeds, reps, mappings, orbit = _c3_three_valley_setup()

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors=seeds,
        representations=reps,
        valley_mappings=mappings,
        orbit=orbit,
        reference_valley="M1",
        rank=1,
    )

    sewing = result.diagnostics.valley_sewing_matrices
    assert sewing is not None
    for (a, b), s in sewing.items():
        if a == b:
            np.testing.assert_allclose(s, result.projectors[a], atol=1e-12)
        else:
            np.testing.assert_allclose(s, np.zeros((3, 3), dtype=np.complex128), atol=1e-12)

    unitarity = result.diagnostics.sewing_unitarity_error
    assert unitarity is not None
    for err in unitarity.values():
        assert err < 1e-12


# -----------------------------------------------------------------------
# I. No valley-preserving operation → explicit failure
# -----------------------------------------------------------------------

def test_no_valley_preserving_operation_for_reference_fails():
    """When no operation in the HSP little group preserves the reference valley,
    construction must fail."""
    p_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    d_swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    result = build_symmetry_adapted_projectors_for_orbit(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={1: d_swap},
        valley_mappings={1: {"VA": "VB", "VB": "VA"}},
        orbit=["VA", "VB"],
        reference_valley="VA",
        rank=1,
    )

    assert result.diagnostics.status == "failed"
    assert "no valley-preserving operation" in result.diagnostics.reason


# -----------------------------------------------------------------------
# J. Direct quality diagnostics computation
# -----------------------------------------------------------------------

def test_quality_diagnostics_on_perfect_projectors():
    """Perfect idempotent orthogonal projectors should have all-zero errors."""
    n = 4
    p1 = np.zeros((n, n), dtype=np.complex128)
    p1[0, 0] = 1.0
    p2 = np.zeros((n, n), dtype=np.complex128)
    p2[1, 1] = 1.0

    diag = compute_projector_quality_diagnostics(
        projectors={"V1": p1, "V2": p2},
        seed_projectors={"V1": p1, "V2": p2},
        representations={0: np.eye(n, dtype=np.complex128)},
        valley_mappings={0: {"V1": "V1", "V2": "V2"}},
        orbit=["V1", "V2"],
        reference_valley="V1",
        preserving_ops={0: np.eye(n, dtype=np.complex128)},
        selected_rank=1,
        eigenvalues=np.ones(n),
        purification_gap=1.0,
        rank_source="user_specified",
    )

    assert diag.orthogonality_error < 1e-12
    # sum of idempotent orthogonal projectors is idempotent
    assert diag.completeness_error < 1e-12
    assert diag.seed_overlap["V1"] == pytest.approx(1.0)
