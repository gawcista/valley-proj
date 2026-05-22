import json
import numpy as np
import pytest

from valleyscope.analysis.symmetry_adapted_character_diagnostics import (
    build_valley_preserving_character_diagnostics,
    compute_valley_preserving_characters,
    compute_valley_preserving_eigenphases,
    summarize_valley_preserving_character_diagnostics,
)
from valleyscope.analysis.symmetry_adapted_representations import (
    build_valley_preserving_representations,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _vp_reps_for_orbit(valley_bases, reps_dict, mappings_dict, orbit):
    return build_valley_preserving_representations(
        valley_bases=valley_bases,
        representations=reps_dict,
        valley_mappings=mappings_dict,
        orbit=orbit,
    )


def _eigenphases_for_op(diag, valley, op_id):
    """Get sorted eigenphases list for a specific valley/op."""
    for item in diag["per_valley"][valley]:
        if item["operation_id"] == op_id:
            return item["eigenphases"]
    return None


# -----------------------------------------------------------------------
# 1. Identity rank-1: chi=1, phase=0
# -----------------------------------------------------------------------

def test_identity_rank1_character_is_one():
    u = np.eye(2, 1, dtype=np.complex128)
    vp = _vp_reps_for_orbit({"V1": u}, {0: np.eye(2, dtype=np.complex128)},
                             {0: {"V1": "V1"}}, ["V1"])

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp, orbit=["V1"],
    )

    item = diag["per_valley"]["V1"][0]
    assert item["character"] == pytest.approx(1.0 + 0.0j)
    assert item["eigenphases"] == pytest.approx([0.0], abs=1e-10)
    assert diag["local_irrep_ready"] is True
    assert diag["diagnostic_only"] is False


# -----------------------------------------------------------------------
# 2. C2 spinless rank-1: eigenvalue ±1, phase 0 or 0.5
# -----------------------------------------------------------------------

def test_c2_spinless_eigenphases():
    u = np.eye(2, 1, dtype=np.complex128)
    d_c2z = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    vp = _vp_reps_for_orbit({"V1": u}, {0: np.eye(2, dtype=np.complex128), 3: d_c2z},
                             {0: {"V1": "V1"}, 3: {"V1": "V1"}}, ["V1"])

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp, orbit=["V1"],
    )

    chars = compute_valley_preserving_characters(
        valley_preserving_representations=vp, orbit=["V1"])
    assert chars["V1"][3] == pytest.approx(-1.0 + 0.0j)

    phases = _eigenphases_for_op(diag, "V1", 3)
    assert phases is not None
    # D_a(C2) = U^dag D(C2) U = -1 → lambda=-1 → phase=-0.5 (wrapped)
    assert phases[0] == pytest.approx(0.5, abs=1e-10) or phases[0] == pytest.approx(-0.5, abs=1e-10)


# -----------------------------------------------------------------------
# 3. C3 rank-1: eigenvalue exp(i 2π/3), phase 1/3
# -----------------------------------------------------------------------

def test_c3_eigenphase_one_third():
    """Use eigenvector of C3 with eigenvalue exp(2πi/3)."""
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    # eigenvector of C3z with eigenvalue exp(+2πi/3): [1, -i, 0]/sqrt2
    u = np.array([[inv_sqrt2], [-inv_sqrt2 * 1j], [0.0]], dtype=np.complex128)
    u /= np.sqrt(float(np.real(u.conj().T @ u).item()))
    angle = 2.0 * np.pi / 3.0
    d_c3 = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.complex128)

    vp = _vp_reps_for_orbit({"V1": u}, {0: np.eye(3, dtype=np.complex128), 1: d_c3},
                             {0: {"V1": "V1"}, 1: {"V1": "V1"}}, ["V1"])

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp, orbit=["V1"],
    )

    phases = _eigenphases_for_op(diag, "V1", 1)
    assert phases is not None
    found = any(abs(p - 1.0/3.0) < 1e-10 for p in phases)
    assert found, f"Expected eigenphase ~1/3 in {phases}"


# -----------------------------------------------------------------------
# 4. Rank-2 diagonal representation: two eigenphases
# -----------------------------------------------------------------------

def test_rank2_diagonal_two_eigenphases():
    u = np.eye(4, 2, dtype=np.complex128)
    d_c2 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
    ], dtype=np.complex128)

    vp = _vp_reps_for_orbit({"V1": u}, {0: np.eye(4, dtype=np.complex128), 3: d_c2},
                             {0: {"V1": "V1"}, 3: {"V1": "V1"}}, ["V1"])

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp, orbit=["V1"],
    )

    chars = compute_valley_preserving_characters(
        valley_preserving_representations=vp, orbit=["V1"])
    assert chars["V1"][3] == pytest.approx(0.0 + 0.0j)  # Tr = 1 + (-1) = 0

    phases = _eigenphases_for_op(diag, "V1", 3)
    assert phases is not None
    assert len(phases) == 2  # two eigenphases


# -----------------------------------------------------------------------
# 5. Non-unitary representation → diagnostic_only
# -----------------------------------------------------------------------

def test_non_unitary_dg_makes_character_diagnostics_diagnostic_only():
    u = np.eye(2, 1, dtype=np.complex128)
    d_bad = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    vp = _vp_reps_for_orbit({"V1": u}, {0: np.eye(2, dtype=np.complex128), 99: d_bad},
                             {0: {"V1": "V1"}, 99: {"V1": "V1"}}, ["V1"])

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp, orbit=["V1"],
    )

    assert diag["diagnostic_only"] is True
    assert diag["local_irrep_ready"] is False
    assert diag["irrep_matching_status"] == "failed_unitarity"


def test_direct_non_unitary_representation_without_upstream_error_fails():
    vp = {
        "status": "ok",
        "representations": {
            "V1": {
                7: np.array(
                    [[1.0, 1.0], [0.0, 1.0]],
                    dtype=np.complex128,
                )
            }
        },
        "unitarity_error": {},
    }

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp,
        orbit=["V1"],
    )

    assert diag["diagnostic_only"] is True
    assert diag["local_irrep_ready"] is False
    assert diag["irrep_matching_status"] == "failed_unitarity"
    assert diag["max_valley_preserving_unitarity_error"] > 0.1


def test_non_square_representation_fails_input_readiness():
    vp = {
        "status": "ok",
        "representations": {"V1": {0: np.ones((2, 1), dtype=np.complex128)}},
        "unitarity_error": {},
    }

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp,
        orbit=["V1"],
    )

    assert diag["diagnostic_only"] is True
    assert diag["local_irrep_ready"] is False
    assert diag["irrep_matching_status"] == "failed_input_readiness"
    assert "must be square" in diag["reason"]


# -----------------------------------------------------------------------
# 6. Inherited input diagnostic_only → output diagnostic_only
# -----------------------------------------------------------------------

def test_inherited_diagnostic_only_propagates():
    u = np.eye(2, 1, dtype=np.complex128)
    vp = _vp_reps_for_orbit({"V1": u}, {0: np.eye(2, dtype=np.complex128)},
                             {0: {"V1": "V1"}}, ["V1"])

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp,
        orbit=["V1"],
        input_diagnostic_only=True,
        input_local_irrep_ready=False,
    )

    assert diag["diagnostic_only"] is True
    assert diag["local_irrep_ready"] is False
    assert diag["irrep_matching_status"] == "failed_input_readiness"


def test_inherited_diagnostic_only_without_ready_flag_propagates():
    u = np.eye(2, 1, dtype=np.complex128)
    vp = _vp_reps_for_orbit({"V1": u}, {0: np.eye(2, dtype=np.complex128)},
                             {0: {"V1": "V1"}}, ["V1"])

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp,
        orbit=["V1"],
        input_diagnostic_only=True,
    )

    assert diag["diagnostic_only"] is True
    assert diag["local_irrep_ready"] is False
    assert diag["irrep_matching_status"] == "failed_input_readiness"
    assert "diagnostic_only=True" in diag["reason"]


def test_partial_input_status_propagates_to_readiness():
    vp = {
        "status": "partial",
        "reason": "missing valley mapping",
        "representations": {"V1": {0: np.eye(1, dtype=np.complex128)}},
        "unitarity_error": {"V1": {0: 0.0}},
    }

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp,
        orbit=["V1"],
    )

    assert diag["diagnostic_only"] is True
    assert diag["local_irrep_ready"] is False
    assert diag["irrep_matching_status"] == "failed_input_readiness"
    assert "status=partial" in diag["reason"]


# -----------------------------------------------------------------------
# 7. Summary JSON serializable without default=str
# -----------------------------------------------------------------------

def test_summary_json_serializable():
    u = np.eye(2, 1, dtype=np.complex128)
    vp = _vp_reps_for_orbit({"V1": u}, {0: np.eye(2, dtype=np.complex128)},
                             {0: {"V1": "V1"}}, ["V1"])

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp, orbit=["V1"],
    )
    summary = summarize_valley_preserving_character_diagnostics(diag)

    encoded = json.dumps(summary)
    assert len(encoded) > 0
    # must not require default=str
    assert "dtype" not in encoded
    assert "ndarray" not in encoded
    assert summary["phase_convention"] == "phase_over_2pi_in_interval_minus_half_to_half"


# -----------------------------------------------------------------------
# 8. Public schema check: no forbidden terms
# -----------------------------------------------------------------------

def test_schema_no_forbidden_terms():
    u = np.eye(2, 1, dtype=np.complex128)
    vp = _vp_reps_for_orbit({"V1": u}, {0: np.eye(2, dtype=np.complex128)},
                             {0: {"V1": "V1"}}, ["V1"])

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp, orbit=["V1"],
    )
    summary = summarize_valley_preserving_character_diagnostics(diag)

    encoded = json.dumps(summary)
    for forbidden in [
        "covariance",
        "equivariant",
        "equivariance",
        "stabilizer",
        "valley_little_group",
    ]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"

    # Required fields in summary
    for key in [
        "status", "reason", "local_irrep_ready", "diagnostic_only",
        "irrep_matching_status", "max_valley_preserving_unitarity_error",
        "max_eigenvalue_modulus_deviation", "per_valley",
    ]:
        assert key in summary, f"missing key: {key}"


# -----------------------------------------------------------------------
# 9. C3 M-star: all three valleys and C3 (valley-preserving only for M1 if
#    C3 doesn't preserve each valley — here using C2_M1 for M1)
# -----------------------------------------------------------------------

def test_mstar_c2_m1_character_diagnostics():
    dim = 3
    u1 = np.zeros((dim, 1), dtype=np.complex128); u1[0, 0] = 1.0
    u2 = np.zeros((dim, 1), dtype=np.complex128); u2[1, 0] = 1.0
    u3 = np.zeros((dim, 1), dtype=np.complex128); u3[2, 0] = 1.0

    d_c2_m1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex128)

    vp = _vp_reps_for_orbit(
        {"M1": u1, "M2": u2, "M3": u3},
        {0: np.eye(3, dtype=np.complex128), 3: d_c2_m1},
        {0: {"M1": "M1", "M2": "M2", "M3": "M3"},
         3: {"M1": "M1", "M2": "M3", "M3": "M2"}},
        ["M1", "M2", "M3"],
    )

    diag = build_valley_preserving_character_diagnostics(
        valley_preserving_representations=vp, orbit=["M1", "M2", "M3"],
    )

    # M1: C2_M1 is VP → D_M1(C2) = 1 → chi=1, phase=0
    item_m1 = next(i for i in diag["per_valley"]["M1"] if i["operation_id"] == 3)
    assert item_m1["character"] == pytest.approx(1.0 + 0.0j)

    # M2: C2_M1 is valley-changing → not in VP reps
    assert not any(i["operation_id"] == 3 for i in diag["per_valley"]["M2"])
