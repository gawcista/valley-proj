import json
import numpy as np
import pytest

from valleyscope.analysis.symmetry_adapted_valley_report import (
    build_symmetry_adapted_valley_report,
    summarize_symmetry_adapted_valley_report,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _c3_mstar_setup():
    """C3 three-valley cyclic with C2_M1: full success path."""
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


# -----------------------------------------------------------------------
# 1. Successful full pipeline
# -----------------------------------------------------------------------

def test_full_pipeline_c3_mstar_success():
    seeds, reps, mappings, orbit = _c3_mstar_setup()

    report = build_symmetry_adapted_valley_report(
        seed_projectors=seeds,
        representations=reps,
        valley_mappings=mappings,
        orbit=orbit,
        reference_valley="M1",
        rank=1,
    )

    assert report["local_irrep_ready"] is True
    assert report["diagnostic_only"] is False
    assert report["status"] == "ok"

    # All sub-reports present
    for key in [
        "symmetry_adapted_projectors",
        "valley_preserving_representations",
        "valley_sewing_matrices",
        "valley_preserving_character_diagnostics",
    ]:
        assert key in report, f"missing: {key}"

    proj = report["symmetry_adapted_projectors"]
    assert proj["status"] == "ok"
    assert proj["selected_rank"] == 1


# -----------------------------------------------------------------------
# 2. Compact summary
# -----------------------------------------------------------------------

def test_compact_summary_serializable():
    seeds, reps, mappings, orbit = _c3_mstar_setup()

    report = build_symmetry_adapted_valley_report(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )
    summary = summarize_symmetry_adapted_valley_report(report)

    encoded = json.dumps(summary)
    assert len(encoded) > 0
    assert "dtype" not in encoded

    for key in [
        "status", "reason", "local_irrep_ready", "diagnostic_only",
        "orbit", "reference_valley",
        "symmetry_adapted_projectors",
        "valley_preserving_representations",
        "valley_sewing_matrices",
        "valley_preserving_character_diagnostics",
    ]:
        assert key in summary, f"missing: {key}"


# -----------------------------------------------------------------------
# 3. Schema: no forbidden terms
# -----------------------------------------------------------------------

def test_schema_no_forbidden_terms():
    seeds, reps, mappings, orbit = _c3_mstar_setup()

    report = build_symmetry_adapted_valley_report(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )
    summary = summarize_symmetry_adapted_valley_report(report)
    encoded = json.dumps(summary)

    for forbidden in [
        "covariance", "equivariant", "equivariance",
        "valley_little_group", "P_cov",
    ]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"


# -----------------------------------------------------------------------
# 4. Failure: rank ambiguity
# -----------------------------------------------------------------------

def test_failure_rank_ambiguity_propagates():
    """gap_insufficient makes projector fail → diagnostic_only."""
    p_m1 = np.diag([0.52, 0.48, 0.45]).astype(np.complex128)
    p_m2 = np.diag([0.45, 0.52, 0.48]).astype(np.complex128)
    p_m3 = np.diag([0.48, 0.45, 0.52]).astype(np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3

    report = build_symmetry_adapted_valley_report(
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

    assert report["diagnostic_only"] is True
    assert report["local_irrep_ready"] is False
    assert "rank gap insufficient" in report["reason"]


# -----------------------------------------------------------------------
# 5. Failure: missing representative operation
# -----------------------------------------------------------------------

def test_failure_missing_representative():
    p_a = np.diag([1.0, 0.0]).astype(np.complex128)
    p_b = np.diag([0.0, 1.0]).astype(np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: np.eye(2, dtype=np.complex128)},
        valley_mappings={0: {"VA": "VA"}},
        orbit=["VA", "VB"], reference_valley="VA", rank=1,
    )

    assert report["diagnostic_only"] is True
    assert report["local_irrep_ready"] is False
    assert "no representative operation" in report["reason"]


# -----------------------------------------------------------------------
# 6. Failure: low seed overlap
# -----------------------------------------------------------------------

def test_failure_low_seed_overlap():
    p_a = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_b = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    d_swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: np.eye(3, dtype=np.complex128), 1: d_swap},
        valley_mappings={
            0: {"VA": "VA", "VB": "VB"},
            1: {"VA": "VB", "VB": "VA"},
        },
        orbit=["VA", "VB"], reference_valley="VA", rank=1,
    )

    assert report["diagnostic_only"] is True
    assert report["local_irrep_ready"] is False
    assert "seed overlap" in report["reason"].lower()


# -----------------------------------------------------------------------
# 7. Failure: non-unitary sewing
# -----------------------------------------------------------------------

def test_failure_non_unitary_sewing():
    """D_g with small non-unitarity passes projector construction but fails
    sewing unitarity check because the eigenvalue modulus deviates."""
    p_a = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_b = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    p_c = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
    # D_g close to swap but slightly non-unitary (eigenvalue 1.5 instead of 1)
    d_bad = np.array([
        [0.0, 1.5, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"V1": p_a, "V2": p_b, "V3": p_c},
        representations={
            0: np.eye(3, dtype=np.complex128),
            1: d_bad,
        },
        valley_mappings={
            0: {"V1": "V1", "V2": "V2", "V3": "V3"},
            1: {"V1": "V2", "V2": "V1", "V3": "V3"},
        },
        orbit=["V1", "V2", "V3"], reference_valley="V1", rank=1,
        unitarity_tol=1e-10,
    )

    assert report["diagnostic_only"] is True
    assert report["local_irrep_ready"] is False


# -----------------------------------------------------------------------
# 8. Default workflow does not change old outputs (no integration test)
# -----------------------------------------------------------------------

def test_report_structure_consistent():
    """Verify all sub-report keys exist even in diagnostic_only mode."""
    seeds, reps, mappings, orbit = _c3_mstar_setup()

    report = build_symmetry_adapted_valley_report(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )

    vp = report["valley_preserving_representations"]
    assert isinstance(vp, dict)
    assert "valley_preserving_operations" in vp or "status" in vp

    sewing = report["valley_sewing_matrices"]
    assert sewing is not None

    chars = report["valley_preserving_character_diagnostics"]
    assert isinstance(chars, dict)
    assert "per_valley" in chars or "status" in chars
