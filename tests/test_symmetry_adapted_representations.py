import numpy as np
import pytest

from valleyscope.analysis.symmetry_adapted_representations import (
    build_symmetry_adapted_representation_diagnostics,
    build_valley_preserving_representations,
    build_valley_sewing_matrices,
    summarize_symmetry_adapted_representations,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _c3_setup():
    """3-valley C3 cyclic orbit with rank-1 orthonormal bases."""
    orbit = ["M1", "M2", "M3"]
    dim = 3
    u1 = np.zeros((dim, 1), dtype=np.complex128); u1[0, 0] = 1.0
    u2 = np.zeros((dim, 1), dtype=np.complex128); u2[1, 0] = 1.0
    u3 = np.zeros((dim, 1), dtype=np.complex128); u3[2, 0] = 1.0
    bases = {"M1": u1, "M2": u2, "M3": u3}
    d_e = np.eye(dim, dtype=np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3
    reps = {0: d_e, 1: d_c3, 2: d_c3sq}
    mappings = {
        0: {"M1": "M1", "M2": "M2", "M3": "M3"},
        1: {"M1": "M2", "M2": "M3", "M3": "M1"},
        2: {"M1": "M3", "M2": "M1", "M3": "M2"},
    }
    return bases, reps, mappings, orbit


def _mstar_setup():
    """4-ops: E, C3, C3^2, C2_M1. M1 preserving: {E, C2_M1}."""
    orbit = ["M1", "M2", "M3"]
    dim = 3
    u1 = np.zeros((dim, 1), dtype=np.complex128); u1[0, 0] = 1.0
    u2 = np.zeros((dim, 1), dtype=np.complex128); u2[1, 0] = 1.0
    u3 = np.zeros((dim, 1), dtype=np.complex128); u3[2, 0] = 1.0
    bases = {"M1": u1, "M2": u2, "M3": u3}
    d_e = np.eye(dim, dtype=np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3
    d_c2_m1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    reps = {0: d_e, 1: d_c3, 2: d_c3sq, 3: d_c2_m1}
    mappings = {
        0: {"M1": "M1", "M2": "M2", "M3": "M3"},
        1: {"M1": "M2", "M2": "M3", "M3": "M1"},
        2: {"M1": "M3", "M2": "M1", "M3": "M2"},
        3: {"M1": "M1", "M2": "M3", "M3": "M2"},
    }
    return bases, reps, mappings, orbit


# -----------------------------------------------------------------------
# 1. Identity single-valley representation
# -----------------------------------------------------------------------

def test_identity_single_valley_representation():
    u = np.eye(2, 1, dtype=np.complex128)  # rank 1, dim 2
    reps = {0: np.eye(2, dtype=np.complex128)}
    mappings = {0: {"V1": "V1"}}
    orbit = ["V1"]

    diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases={"V1": u}, representations=reps,
        valley_mappings=mappings, orbit=orbit,
    )

    assert diag["local_irrep_ready"] is True
    assert diag["diagnostic_only"] is False
    assert diag["max_valley_preserving_unitarity_error"] < 1e-12
    assert diag["max_sewing_unitarity_error"] == pytest.approx(0.0)
    assert diag["valley_preserving_operations"]["V1"] == [0]


def test_identity_valley_preserving_rep_is_identity():
    u = np.eye(2, 1, dtype=np.complex128)
    vp = build_valley_preserving_representations(
        valley_bases={"V1": u},
        representations={0: np.eye(2, dtype=np.complex128)},
        valley_mappings={0: {"V1": "V1"}},
        orbit=["V1"],
    )

    d_a = vp["representations"]["V1"][0]
    np.testing.assert_allclose(d_a, np.eye(1, dtype=np.complex128), atol=1e-12)
    assert vp["unitarity_error"]["V1"][0] < 1e-12
    assert vp["status"] == "ok"


# -----------------------------------------------------------------------
# 2. C3 three-valley cyclic orbit
# -----------------------------------------------------------------------

def test_c3_sewing_matrices_unitary():
    bases, reps, mappings, orbit = _c3_setup()

    sewing = build_valley_sewing_matrices(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
    )

    assert sewing["status"] == "ok"
    # C3: M1->M2, M2->M3, M3->M1
    b_12 = sewing["sewing_matrices"].get((1, "M1", "M2"))
    b_23 = sewing["sewing_matrices"].get((1, "M2", "M3"))
    b_31 = sewing["sewing_matrices"].get((1, "M3", "M1"))
    assert b_12 is not None
    assert b_23 is not None
    assert b_31 is not None
    for b in [b_12, b_23, b_31]:
        assert b.shape == (1, 1)
        np.testing.assert_allclose(b.conj().T @ b, np.eye(1), atol=1e-12)

    for err in sewing["unitarity_error"].values():
        assert err < 1e-12


def test_c3_no_valley_preserving_except_identity():
    """C3 maps M1->M2, M2->M3, M3->M1. Only E preserves each valley."""
    bases, reps, mappings, orbit = _c3_setup()

    vp = build_valley_preserving_representations(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
    )

    for valley in orbit:
        assert list(vp["representations"][valley].keys()) == [0]  # only E


def test_c3_diagnostics_local_irrep_ready():
    bases, reps, mappings, orbit = _c3_setup()

    diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
    )

    assert diag["local_irrep_ready"] is True
    assert diag["diagnostic_only"] is False
    assert diag["max_sewing_unitarity_error"] < 1e-12


# -----------------------------------------------------------------------
# 3. M-star toy with C2 fixing M1
# -----------------------------------------------------------------------

def test_mstar_c2_m1_is_valley_preserving_for_m1():
    bases, reps, mappings, orbit = _mstar_setup()

    vp = build_valley_preserving_representations(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
    )

    m1_reps = vp["representations"]["M1"]
    assert 0 in m1_reps  # identity
    assert 3 in m1_reps  # C2_M1
    d_a = m1_reps[3]
    np.testing.assert_allclose(d_a, np.eye(1, dtype=np.complex128), atol=1e-12)
    assert vp["unitarity_error"]["M1"][3] < 1e-12


def test_mstar_c2_m1_is_valley_changing_for_m2_m3():
    bases, reps, mappings, orbit = _mstar_setup()

    sewing = build_valley_sewing_matrices(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
    )

    # C2_M1 (op 3) swaps M2<->M3
    b_23 = sewing["sewing_matrices"].get((3, "M2", "M3"))
    b_32 = sewing["sewing_matrices"].get((3, "M3", "M2"))
    assert b_23 is not None
    assert b_32 is not None
    for b in [b_23, b_32]:
        np.testing.assert_allclose(b.conj().T @ b, np.eye(1), atol=1e-12)


def test_mstar_diagnostics_ready_with_vp_and_sewing():
    bases, reps, mappings, orbit = _mstar_setup()

    diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
    )

    assert diag["local_irrep_ready"] is True
    assert diag["diagnostic_only"] is False
    vp_ops = diag["valley_preserving_operations"]
    assert vp_ops["M1"] == [0, 3]
    assert vp_ops["M2"] == [0]
    assert vp_ops["M3"] == [0]
    vc_ops = diag["valley_changing_operations"]
    assert 1 in vc_ops["M1"]  # C3
    assert 3 in vc_ops["M2"]  # C2_M1
    assert 3 in vc_ops["M3"]  # C2_M1


# -----------------------------------------------------------------------
# 4. Missing pi_g(a)
# -----------------------------------------------------------------------

def test_missing_valley_mapping_fails():
    u = np.eye(2, 1, dtype=np.complex128)
    # mapping missing for V2
    vp = build_valley_preserving_representations(
        valley_bases={"V1": u, "V2": u},
        representations={0: np.eye(2, dtype=np.complex128)},
        valley_mappings={0: {"V1": "V1"}},  # V2 not in mapping
        orbit=["V1", "V2"],
    )

    assert vp["status"] == "partial"
    assert any("V2" in m for m in vp["missing_mapping"])


def test_missing_mapping_diagnostics_not_ready():
    u = np.eye(2, 1, dtype=np.complex128)
    diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases={"V1": u, "V2": u},
        representations={0: np.eye(2, dtype=np.complex128)},
        valley_mappings={0: {"V1": "V1"}},
        orbit=["V1", "V2"],
    )

    # Not failed diagnostics, just partial — but sewing has missing mapping
    assert diag["diagnostic_only"] is False  # sewing partial ≠ shape failure
    assert "partial" in str(diag.get("valley_sewing_matrices", {}).get("status", ""))


# -----------------------------------------------------------------------
# 5. Shape / rank mismatch
# -----------------------------------------------------------------------

def test_rank_mismatch_fails_sewing():
    u1 = np.eye(3, 1, dtype=np.complex128)  # rank 1
    u2 = np.eye(3, 2, dtype=np.complex128)  # rank 2 (wrong!)
    d_swap = np.eye(3, dtype=np.complex128)
    d_swap[0, 1] = 1; d_swap[1, 0] = 1; d_swap[0, 0] = 0; d_swap[1, 1] = 0

    sewing = build_valley_sewing_matrices(
        valley_bases={"V1": u1, "V2": u2},
        representations={1: d_swap},
        valley_mappings={1: {"V1": "V2", "V2": "V1"}},
        orbit=["V1", "V2"],
    )

    assert sewing["status"] == "failed"
    assert any("rank mismatch" in m for m in sewing["shape_mismatch"])


def test_missing_basis_fails():
    u1 = np.eye(2, 1, dtype=np.complex128)
    vp = build_valley_preserving_representations(
        valley_bases={"V1": u1},  # V2 missing
        representations={0: np.eye(2, dtype=np.complex128)},
        valley_mappings={0: {"V1": "V1", "V2": "V2"}},
        orbit=["V1", "V2"],
    )

    assert vp["status"] == "failed"
    assert any("V2" in m for m in vp["shape_mismatch"])


def test_non_orthonormal_basis_fails():
    # U not orthonormal: columns have norm > 1
    u = np.ones((3, 1), dtype=np.complex128)  # not orthonormal
    vp = build_valley_preserving_representations(
        valley_bases={"V1": u},
        representations={0: np.eye(3, dtype=np.complex128)},
        valley_mappings={0: {"V1": "V1"}},
        orbit=["V1"],
    )

    assert vp["status"] == "failed"
    assert any("orthonormal" in m for m in vp["shape_mismatch"])


# -----------------------------------------------------------------------
# 6. Non-unitary D_g → elevated error → not ready
# -----------------------------------------------------------------------

def test_non_unitary_dg_elevates_unitarity_error():
    u = np.eye(2, 1, dtype=np.complex128)
    d_bad = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.complex128)  # not unitary

    diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases={"V1": u},
        representations={0: d_bad},
        valley_mappings={0: {"V1": "V1"}},
        orbit=["V1"],
    )

    assert diag["max_valley_preserving_unitarity_error"] > 0.1
    assert diag["local_irrep_ready"] is False
    assert diag["diagnostic_only"] is True
    assert "unitarity" in diag["reason"]


# -----------------------------------------------------------------------
# 7. Schema test
# -----------------------------------------------------------------------

def test_compact_summary_field_names():
    bases, reps, mappings, orbit = _mstar_setup()

    diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
    )
    summary = summarize_symmetry_adapted_representations(diag)

    # Required fields
    for key in [
        "status", "reason", "local_irrep_ready", "diagnostic_only", "orbit",
        "selected_rank_by_valley", "valley_preserving_operations",
        "valley_changing_operations", "max_valley_preserving_unitarity_error",
        "max_sewing_unitarity_error", "representation_closure_status",
        "valley_preserving_representations", "valley_sewing_matrices_summary",
    ]:
        assert key in summary, f"missing key: {key}"

    # Forbidden terms in summary and diagnostics
    import json
    encoded = json.dumps(summary, default=str)
    for forbidden in ["covariance", "equivariant", "equivariance", "valley_little_group"]:
        assert forbidden not in encoded.lower(), f"forbidden term: {forbidden}"

    # Valley preserving ops use correct naming
    vp = diag["valley_preserving_operations"]
    assert isinstance(vp, dict)
    vc = diag["valley_changing_operations"]
    assert isinstance(vc, dict)

    # Compact sewing uses correct key structure
    sewing_summary = summary["valley_sewing_matrices_summary"]
    assert isinstance(sewing_summary, list)
    if sewing_summary:
        row = sewing_summary[0]
        assert "operation_id" in row
        assert "source_valley" in row
        assert "target_valley" in row
        assert "sewing_unitarity_error" in row


def test_compact_summary_serializable():
    bases, reps, mappings, orbit = _mstar_setup()
    diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
    )
    summary = summarize_symmetry_adapted_representations(diag)

    import json
    encoded = json.dumps(summary, default=str)
    assert len(encoded) > 0
    # No numpy types in encoded
    assert "dtype" not in encoded


# -----------------------------------------------------------------------
# 8. Representation closure (with explicit closure_mapping)
# -----------------------------------------------------------------------

def test_closure_check_with_explicit_mapping():
    """M-star: C2_M1 @ C2_M1 = E. When closure_mapping provided, closure
    should be checked."""
    bases, reps, mappings, orbit = _mstar_setup()

    closure = {(3, 3): 0}  # C2_M1 @ C2_M1 = E
    diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
        closure_mapping=closure,
    )

    assert diag["representation_closure_status"] == "closed"
    assert diag["representation_closure_missing_products"] == []


def test_closure_fails_with_wrong_mapping():
    """Provide wrong closure mapping: product should mismatch."""
    bases, reps, mappings, orbit = _mstar_setup()

    closure = {(3, 3): 1}  # C2_M1 @ C2_M1 → C3 (wrong product)
    diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
        closure_mapping=closure,
    )

    assert diag["representation_closure_status"] == "not_closed"
    assert len(diag["representation_closure_missing_products"]) > 0


def test_closure_not_evaluated_without_mapping():
    bases, reps, mappings, orbit = _c3_setup()
    diag = build_symmetry_adapted_representation_diagnostics(
        valley_bases=bases, representations=reps,
        valley_mappings=mappings, orbit=orbit,
    )

    assert diag["representation_closure_status"] == "not_evaluated"
