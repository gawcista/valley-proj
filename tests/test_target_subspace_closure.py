import json
import numpy as np

from valleyscope.analysis.target_subspace_closure import (
    build_target_subspace_closure_report,
    check_target_subspace_closure_blocked,
    check_target_subspace_closure_blocked_for_operation,
)


def _make_raw_reps(d_raw_dict, little_group_passed=True):
    result = {}
    for op_id, d_raw in d_raw_dict.items():
        result[op_id] = {
            "D_raw": np.asarray(d_raw, dtype=np.complex128),
            "kind": "C2" if op_id > 0 else "E",
            "order": 2 if op_id > 0 else 1,
            "little_group_passed": little_group_passed,
            "sector_mapping": {},
        }
    return {"MM": result}


def test_exact_unitary_passes():
    d_e = np.eye(6, dtype=np.complex128)
    d_c2 = np.diag([1, -1, 1, -1, 1, -1]).astype(np.complex128)

    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=_make_raw_reps({0: d_e, 5: d_c2}),
        operation_orders={0: 1, 5: 2},
    )

    assert report["status"] == "ok"
    rows = report["by_kpoint"]["MM"]
    assert all(row["status"] == "ok" for row in rows)


def test_nonunitary_d_fails():
    d_bad = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=_make_raw_reps({1: d_bad}),
        operation_orders={1: 2},
        unitarity_tol=1e-10,
    )

    row = report["by_kpoint"]["MM"][0]
    assert row["status"] == "failed"
    assert "raw_unitarity_error" in row["reason"]


def test_c2_spinless_group_relation():
    d_good = np.diag([1, -1]).astype(np.complex128)
    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=_make_raw_reps({0: d_good}),
        operation_orders={0: 2},
        spinor_wavefunction=False,
    )

    row = report["by_kpoint"]["MM"][0]
    assert row["status"] == "ok"
    assert row["group_relation_label"] == "D^2 - I (spinless)"
    assert row["group_relation_error"] < 1e-10


def test_c2_spinful_group_relation():
    d_good = np.diag([1j, -1j]).astype(np.complex128)
    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=_make_raw_reps({0: d_good}),
        operation_orders={0: 2},
        spinor_wavefunction=True,
    )

    row = report["by_kpoint"]["MM"][0]
    assert row["status"] == "ok"
    assert row["group_relation_label"] == "D^2 + I (spinful)"
    assert row["group_relation_error"] < 1e-10


def test_group_relation_violation_fails():
    d_bad = np.diag([2, 2]).astype(np.complex128)
    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=_make_raw_reps({0: d_bad}),
        operation_orders={0: 2},
        spinor_wavefunction=False,
        group_relation_tol=1e-10,
    )

    row = report["by_kpoint"]["MM"][0]
    assert row["status"] == "failed"


def test_summary_json_exposes_diagnostic():
    d_c2 = np.diag([1, -1, 1, -1, 1, -1]).astype(np.complex128)
    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=_make_raw_reps({5: d_c2}),
        operation_orders={5: 2},
    )

    encoded = json.dumps(report)
    assert "target_subspace_closure" not in encoded.lower()
    assert "raw_unitarity_error" in encoded
    assert "group_relation_error" in encoded


def test_check_blockers_returns_failed():
    d_bad = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=_make_raw_reps({1: d_bad}),
        operation_orders={1: 2},
        unitarity_tol=1e-10,
    )

    blockers = check_target_subspace_closure_blocked(report)
    assert "target_subspace_closure_failed" in blockers


def test_check_blockers_empty_when_ok():
    d_c2 = np.diag([1, -1]).astype(np.complex128)
    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=_make_raw_reps({5: d_c2}),
        operation_orders={5: 2},
    )

    blockers = check_target_subspace_closure_blocked(report)
    assert blockers == []


def test_identity_skips_group_relation():
    d_e = np.eye(6, dtype=np.complex128)
    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=_make_raw_reps({0: d_e}),
        operation_orders={0: 1},
    )

    row = report["by_kpoint"]["MM"][0]
    assert row["status"] == "ok"
    assert "group_relation_error" not in row


def test_not_little_group_not_evaluated():
    d_c2 = np.diag([1, -1]).astype(np.complex128)
    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=_make_raw_reps({5: d_c2}, little_group_passed=False),
        operation_orders={5: 2},
    )

    row = report["by_kpoint"]["MM"][0]
    assert row["status"] == "not_evaluated"
    assert "not in little group" in row["reason"]


def test_mapping_miss_count_fails():
    reps = _make_raw_reps({5: np.diag([1, -1]).astype(np.complex128)})
    reps["MM"][5]["mapping_miss_count"] = 3

    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint=reps,
        operation_orders={5: 2},
    )

    row = report["by_kpoint"]["MM"][0]
    assert row["status"] == "failed"
    assert "mapping_miss_count=3" in row["reason"]


def test_no_data_returns_no_data_status():
    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint={},
    )
    assert report["status"] == "no_data"


def test_closure_gate_per_operation():
    """closure failure on op=5 (valley-changing) should NOT block op=4 (valley-preserving)."""
    d_good = np.diag([1, -1, 1, -1, 1, -1]).astype(np.complex128)
    d_bad = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint={
            "MM": {
                4: {
                    "D_raw": d_good,
                    "kind": "C2",
                    "order": 2,
                    "little_group_passed": True,
                    "sector_mapping": {},
                },
                5: {
                    "D_raw": d_bad,
                    "kind": "C2",
                    "order": 2,
                    "little_group_passed": True,
                    "sector_mapping": {},
                },
            },
        },
        operation_orders={4: 2, 5: 2},
        unitarity_tol=1e-10,
    )

    # op=5 fails
    assert check_target_subspace_closure_blocked_for_operation(report, "MM", 5) is True
    # op=4 passes
    assert check_target_subspace_closure_blocked_for_operation(report, "MM", 4) is False


def test_closure_gate_valley_changing_op_fails_does_not_block():
    """A failed valley-changing op should not block a valley-preserving subspace."""
    d_bad = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    d_good = np.diag([1, -1]).astype(np.complex128)

    report = build_target_subspace_closure_report(
        raw_representations_by_kpoint={
            "MM": {
                3: {  # valley-changing
                    "D_raw": d_bad,
                    "kind": "C2",
                    "order": 2,
                    "little_group_passed": True,
                    "sector_mapping": {},
                },
                4: {  # valley-preserving
                    "D_raw": d_good,
                    "kind": "C2",
                    "order": 2,
                    "little_group_passed": True,
                    "sector_mapping": {},
                },
            },
        },
        operation_orders={3: 2, 4: 2},
        unitarity_tol=1e-10,
    )

    assert check_target_subspace_closure_blocked_for_operation(report, "MM", 3) is True
    assert check_target_subspace_closure_blocked_for_operation(report, "MM", 4) is False
