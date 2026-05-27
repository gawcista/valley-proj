import json
import numpy as np

from valleyscope.analysis.subspace_representation_quality import (
    build_subspace_representation_quality_report,
)


def _make_identity_bases_valleys(dim=3, valleys=None, rank=1):
    if valleys is None:
        valleys = ["VA"]
    bases = {}
    for v in valleys:
        u = np.zeros((dim, rank), dtype=np.complex128)
        for i in range(rank):
            u[i, i] = 1.0
        bases[v] = u
    return bases


def _make_projectors(valleys, dim, rank):
    proj = {}
    for v in valleys:
        p = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(rank):
            p[i, i] = 1.0
        proj[v] = p
    return proj


# -----------------------------------------------------------------------
# 1. Exact invariant subspace
# -----------------------------------------------------------------------

def test_exact_invariant_subspace_ok():
    dim = 3
    bases = _make_identity_bases_valleys(dim=dim)
    proj = _make_projectors(["VA"], dim, 1)
    d_e = np.eye(dim, dtype=np.complex128)
    d_c2 = np.diag([1, -1, -1]).astype(np.complex128)

    report = build_subspace_representation_quality_report(
        valley_bases=bases,
        projectors=proj,
        representations={0: d_e, 1: d_c2},
        valley_mappings={0: {"VA": "VA"}, 1: {"VA": "VA"}},
        operation_orders={0: 1, 1: 2},
        spinor_wavefunction=False,
    )

    assert report["status"] == "ok"
    rows = report["rows"]
    vp_rows = [r for r in rows if r["is_valley_preserving"]]
    for r in vp_rows:
        assert r["diagnosis"] == "ok"
        assert r["basis_orthonormality_error"] < 1e-12
        assert r["D_raw_unitarity_error"] < 1e-12
        assert r["projector_invariance_error"] < 1e-12
        assert r["local_representation_unitarity_error"] < 1e-12


# -----------------------------------------------------------------------
# 2. Non-orthonormal basis
# -----------------------------------------------------------------------

def test_non_orthonormal_basis():
    dim = 3
    u_bad = np.ones((dim, 1), dtype=np.complex128) * 2.0
    d_e = np.eye(dim, dtype=np.complex128)
    proj = _make_projectors(["VA"], dim, 1)

    report = build_subspace_representation_quality_report(
        valley_bases={"VA": u_bad},
        projectors=proj,
        representations={0: d_e},
        valley_mappings={0: {"VA": "VA"}},
    )

    rows = [r for r in report["rows"] if r["is_valley_preserving"]]
    assert rows[0]["diagnosis"] == "non_orthonormal_basis"
    assert rows[0]["basis_orthonormality_error"] > 1e-6


# -----------------------------------------------------------------------
# 3. Raw representation nonunitary
# -----------------------------------------------------------------------

def test_raw_representation_nonunitary():
    dim = 2
    bases = _make_identity_bases_valleys(dim=dim)
    proj = _make_projectors(["VA"], dim, 1)
    d_bad = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    report = build_subspace_representation_quality_report(
        valley_bases=bases,
        projectors=proj,
        representations={1: d_bad},
        valley_mappings={1: {"VA": "VA"}},
        operation_orders={1: 2},
    )

    rows = [r for r in report["rows"] if r["is_valley_preserving"]]
    assert rows[0]["diagnosis"] == "raw_representation_nonunitary"
    assert rows[0]["D_raw_unitarity_error"] > 1e-3


# -----------------------------------------------------------------------
# 4. Projector not invariant
# -----------------------------------------------------------------------

def test_projector_not_invariant():
    dim = 3
    bases = _make_identity_bases_valleys(dim=dim, valleys=["VA"], rank=1)
    d_good = np.diag([1, -1, -1]).astype(np.complex128)
    p_bad = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.complex128)

    report = build_subspace_representation_quality_report(
        valley_bases=bases,
        projectors={"VA": p_bad},
        representations={1: d_good},
        valley_mappings={1: {"VA": "VA"}},
        operation_orders={1: 2},
    )

    rows = [r for r in report["rows"] if r["is_valley_preserving"]]
    assert rows[0]["diagnosis"] == "projector_not_invariant_under_valley_preserving_operation"
    assert rows[0]["projector_invariance_error"] > 1e-1


# -----------------------------------------------------------------------
# 5. Spinful C2 group relation
# -----------------------------------------------------------------------

def test_spinful_c2_group_relation():
    dim = 2
    bases = _make_identity_bases_valleys(dim=dim, rank=2)
    proj = _make_projectors(["VA"], dim, 2)
    d_c2_spinful = np.diag([1j, -1j]).astype(np.complex128)

    report = build_subspace_representation_quality_report(
        valley_bases=bases,
        projectors=proj,
        representations={1: d_c2_spinful},
        valley_mappings={1: {"VA": "VA"}},
        operation_orders={1: 2},
        spinor_wavefunction=True,
    )

    rows = [r for r in report["rows"] if r["is_valley_preserving"]]
    assert rows[0]["diagnosis"] == "ok"
    assert rows[0]["local_group_relation_error"] < 1e-10
    assert "spinful" in rows[0]["local_group_relation_label"]


# -----------------------------------------------------------------------
# 6. JSON serializability
# -----------------------------------------------------------------------

def test_json_serializable():
    dim = 3
    bases = _make_identity_bases_valleys(dim=dim)
    proj = _make_projectors(["VA"], dim, 1)
    d_c2 = np.diag([1, -1, -1]).astype(np.complex128)

    report = build_subspace_representation_quality_report(
        valley_bases=bases,
        projectors=proj,
        representations={0: np.eye(dim, dtype=np.complex128), 1: d_c2},
        valley_mappings={0: {"VA": "VA"}, 1: {"VA": "VA"}},
        operation_orders={0: 1, 1: 2},
    )

    encoded = json.dumps(report)
    assert len(encoded) > 0
    assert "dtype" not in encoded
    assert "default=str" not in encoded


# -----------------------------------------------------------------------
# 7. Schema terminology
# -----------------------------------------------------------------------

def test_schema_no_forbidden_terms():
    dim = 3
    bases = _make_identity_bases_valleys(dim=dim)
    proj = _make_projectors(["VA"], dim, 1)
    d_c2 = np.diag([1, -1, -1]).astype(np.complex128)

    report = build_subspace_representation_quality_report(
        valley_bases=bases,
        projectors=proj,
        representations={0: np.eye(dim, dtype=np.complex128), 1: d_c2},
        valley_mappings={0: {"VA": "VA"}, 1: {"VA": "VA"}},
        operation_orders={0: 1, 1: 2},
    )

    encoded = json.dumps(report)
    for forbidden in [
        "covariance", "equivariance", "stabilizer", "valley_little_group",
    ]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"


# -----------------------------------------------------------------------
# 8. Valley-changing operation skipped
# -----------------------------------------------------------------------

def test_valley_changing_op_not_valley_preserving():
    dim = 3
    bases = _make_identity_bases_valleys(dim=dim, valleys=["VA", "VB"], rank=1)
    proj = _make_projectors(["VA", "VB"], dim, 1)
    d_swap = np.array([
        [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
    ], dtype=np.complex128)

    report = build_subspace_representation_quality_report(
        valley_bases=bases,
        projectors=proj,
        representations={1: d_swap},
        valley_mappings={1: {"VA": "VB", "VB": "VA"}},
    )

    # VA: op=1 maps VA->VB (valley-changing) → not_valley_preserving
    va_rows = [r for r in report["rows"] if r["valley"] == "VA" and r["operation_id"] == 1]
    assert va_rows[0]["diagnosis"] == "not_valley_preserving"

    # Same for VB
    vb_rows = [r for r in report["rows"] if r["valley"] == "VB" and r["operation_id"] == 1]
    assert vb_rows[0]["diagnosis"] == "not_valley_preserving"


# -----------------------------------------------------------------------
# 9. Multiple valleys with mixed quality
# -----------------------------------------------------------------------

def test_mixed_quality_across_valleys():
    dim = 3
    # VA: clean
    u_va = np.zeros((dim, 1), dtype=np.complex128)
    u_va[0, 0] = 1.0
    # VB: bad basis
    u_vb = np.ones((dim, 1), dtype=np.complex128) * 2.0

    proj = _make_projectors(["VA", "VB"], dim, 1)
    d_c2 = np.diag([1, -1, -1]).astype(np.complex128)

    report = build_subspace_representation_quality_report(
        valley_bases={"VA": u_va, "VB": u_vb},
        projectors=proj,
        representations={1: d_c2},
        valley_mappings={1: {"VA": "VA", "VB": "VB"}},
        operation_orders={1: 2},
    )

    va_rows = [r for r in report["rows"] if r["valley"] == "VA" and r["is_valley_preserving"]]
    vb_rows = [r for r in report["rows"] if r["valley"] == "VB" and r["is_valley_preserving"]]

    assert va_rows[0]["diagnosis"] == "ok"
    assert vb_rows[0]["diagnosis"] == "non_orthonormal_basis"
    assert report["status"] == "quality_issues_detected"


# -----------------------------------------------------------------------
# 10. Singular values output
# -----------------------------------------------------------------------

def test_singular_values_present():
    dim = 3
    bases = _make_identity_bases_valleys(dim=dim, rank=2)
    p = np.zeros((dim, dim), dtype=np.complex128)
    p[0, 0] = 1.0
    p[1, 1] = 1.0
    d_c2 = np.diag([1, -1, -1]).astype(np.complex128)

    report = build_subspace_representation_quality_report(
        valley_bases=bases,
        projectors={"VA": p},
        representations={1: d_c2},
        valley_mappings={1: {"VA": "VA"}},
        operation_orders={1: 2},
    )

    rows = [r for r in report["rows"] if r["is_valley_preserving"]]
    assert "singular_values_of_UdagDU" in rows[0]
    assert len(rows[0]["singular_values_of_UdagDU"]) == 2


# -----------------------------------------------------------------------
# 11. Diagnostic-only fields marked
# -----------------------------------------------------------------------

def test_diagnostic_only_fields_are_marked():
    dim = 3
    bases = _make_identity_bases_valleys(dim=dim)
    proj = _make_projectors(["VA"], dim, 1)
    d_c2 = np.diag([1, -1, -1]).astype(np.complex128)

    report = build_subspace_representation_quality_report(
        valley_bases=bases,
        projectors=proj,
        representations={0: np.eye(dim, dtype=np.complex128), 1: d_c2},
        valley_mappings={0: {"VA": "VA"}, 1: {"VA": "VA"}},
        operation_orders={0: 1, 1: 2},
    )

    rows = [r for r in report["rows"] if r["is_valley_preserving"] and r["operation_id"] != 0]
    assert "closest_unitary_eigenphases_diagnostic_only" in rows[0]
    assert "polar_unitarity_distance_diagnostic_only" in rows[0]
