from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from valleyscope.analysis.time_reversal_sewing import (
    build_time_reversal_sewing_report,
    select_trusted_valley_projectors,
    validate_time_reversal_sewing_report,
)


def _report(
    coefficients: np.ndarray,
    *,
    spinor: bool,
    spinor_convention_verified: bool = True,
) -> dict[str, object]:
    n_bands = coefficients.shape[0]
    return build_time_reversal_sewing_report(
        kpoint_frac_by_name={"G": np.zeros(3)},
        g_vectors_frac_by_kpoint={"G": np.zeros((1, 3), dtype=int)},
        coefficients_by_kpoint={"G": coefficients},
        band_indices_by_kpoint={"G": np.arange(1, n_bands + 1)},
        valley_projectors_by_kpoint={
            "G": {"v": np.eye(n_bands, dtype=complex)},
        },
        valley_projector_provenance_by_kpoint={
            "G": {
                "v": {
                    "workflow_path": "direct_qcut",
                    "projector_kind": "fixed_center_seed",
                },
            },
        },
        projector_selection_blockers=[],
        time_reversal_valley_mapping={"v": "v"},
        spinor=spinor,
        spinor_convention_verified=spinor_convention_verified,
    )


def test_scalar_self_mapped_subspace_has_theta_square_plus_one():
    report = _report(
        np.asarray([[[1.0 + 0.0j]]]),
        spinor=False,
    )

    assert report["status"] == "validated"
    row = report["rows"][0]
    assert row["reciprocal_shift"] == [0, 0, 0]
    assert row["overlap_singular_values"] == [1.0]
    assert row["theta_square_residual"] == 0.0
    assert row["projector_covariance"]["v"]["status"] == "validated"


def test_spinful_complete_kramers_pair_has_theta_square_minus_one():
    coefficients = np.asarray([
        [[1.0 + 0.0j], [0.0 + 0.0j]],
        [[0.0 + 0.0j], [1.0 + 0.0j]],
    ])

    report = _report(coefficients, spinor=True)

    assert report["status"] == "validated"
    row = report["rows"][0]
    assert row["overlap_singular_values"] == [1.0, 1.0]
    assert row["theta_square_residual"] == 0.0
    assert row["blockers"] == []


def test_spinful_window_cutting_kramers_pair_fails_closed():
    report = _report(
        np.asarray([[[1.0 + 0.0j], [0.0 + 0.0j]]]),
        spinor=True,
    )

    assert report["status"] == "blocked"
    assert "incomplete_kramers_subspace_odd_dimension:G" in report["blockers"]
    assert any(
        blocker.startswith("time_reversal_target_subspace_not_closed:G")
        for blocker in report["blockers"]
    )


def test_unverified_spinor_convention_blocks_numerical_sewing():
    coefficients = np.asarray([
        [[1.0 + 0.0j], [0.0 + 0.0j]],
        [[0.0 + 0.0j], [1.0 + 0.0j]],
    ])

    report = _report(
        coefficients,
        spinor=True,
        spinor_convention_verified=False,
    )

    assert report["status"] == "blocked"
    assert "spinor_convention_unverified_for_time_reversal" in report[
        "blockers"
    ]


def test_fractional_g_vectors_are_rejected_without_integer_coercion():
    coefficients = np.asarray([[[1.0 + 0.0j]]])
    report = build_time_reversal_sewing_report(
        kpoint_frac_by_name={"G": np.zeros(3)},
        g_vectors_frac_by_kpoint={"G": np.asarray([[0.25, 0.0, 0.0]])},
        coefficients_by_kpoint={"G": coefficients},
        band_indices_by_kpoint={"G": np.asarray([1])},
        valley_projectors_by_kpoint={"G": {"v": np.eye(1)}},
        valley_projector_provenance_by_kpoint={
            "G": {"v": {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }},
        },
        projector_selection_blockers=[],
        time_reversal_valley_mapping={"v": "v"},
        spinor=False,
        spinor_convention_verified=True,
    )

    assert report["status"] == "blocked"
    assert "malformed_time_reversal_g_vectors:G" in report["blockers"]


def test_nonbijective_kpoint_and_valley_sewing_evidence_fails_closed():
    coefficients = np.asarray([[[1.0 + 0.0j]]])
    report = build_time_reversal_sewing_report(
        kpoint_frac_by_name={"G1": np.zeros(3), "G2": np.zeros(3)},
        g_vectors_frac_by_kpoint={
            "G1": np.zeros((1, 3), dtype=int),
            "G2": np.zeros((1, 3), dtype=int),
        },
        coefficients_by_kpoint={"G1": coefficients, "G2": coefficients},
        band_indices_by_kpoint={"G1": np.asarray([1]), "G2": np.asarray([1])},
        valley_projectors_by_kpoint={
            "G1": {"v": np.eye(1)},
            "G2": {"v": np.eye(1)},
        },
        valley_projector_provenance_by_kpoint={
            name: {"v": {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }}
            for name in ("G1", "G2")
        },
        projector_selection_blockers=[],
        time_reversal_valley_mapping={"v": "missing"},
        spinor=False,
        spinor_convention_verified=True,
    )

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("ambiguous_time_reversal_kpoint_partner:")
        for blocker in report["blockers"]
    )
    assert "incomplete_or_nonbijective_time_reversal_valley_mapping" in report[
        "blockers"
    ]


@pytest.mark.parametrize(
    "tamper",
    [
        "top_level_blocker",
        "mapping_miss",
        "orthonormality",
        "nan_singular_value",
        "nan_covariance",
        "missing_reciprocal_shift",
        "wrong_partner_valley",
        "missing_band_inventory",
        "row_reciprocal_shift",
        "negative_residual",
        "projector_fingerprint",
    ],
)
def test_serialized_sewing_certificate_rejects_tampering(tamper: str):
    evidence = deepcopy(_report(np.asarray([[[1.0 + 0.0j]]]), spinor=False))
    row = evidence["rows"][0]
    if tamper == "top_level_blocker":
        evidence["blockers"] = ["injected"]
    elif tamper == "mapping_miss":
        row["mapping_miss_count"] = 1
    elif tamper == "orthonormality":
        row["source_orthonormality_residual"] = 9.0
    elif tamper == "nan_singular_value":
        row["overlap_singular_values"] = [float("nan")]
    elif tamper == "nan_covariance":
        row["projector_covariance"]["v"]["covariance_residual"] = float(
            "nan"
        )
    elif tamper == "missing_reciprocal_shift":
        evidence["reciprocal_shifts_by_kpoint"] = {}
    elif tamper == "wrong_partner_valley":
        row["projector_covariance"]["v"]["partner_valley"] = "other"
    elif tamper == "missing_band_inventory":
        row["source_band_indices_vasp"] = []
    elif tamper == "row_reciprocal_shift":
        row["reciprocal_shift"] = [1, 0, 0]
    elif tamper == "negative_residual":
        row["target_subspace_closure_residual"] = -1.0
    elif tamper == "projector_fingerprint":
        row["projector_covariance"]["v"][
            "source_projector_provenance"
        ]["projector_fingerprint"] = "sha256:" + "0" * 64

    assert not validate_time_reversal_sewing_report(
        evidence,
        valley_members=["v"],
        theta_square=1,
    )


def test_spinful_serialized_certificate_rechecks_convention_and_kramers_rank():
    coefficients = np.asarray([
        [[1.0 + 0.0j], [0.0 + 0.0j]],
        [[0.0 + 0.0j], [1.0 + 0.0j]],
    ])
    evidence = _report(coefficients, spinor=True)

    unverified = deepcopy(evidence)
    unverified["spinor_convention_verified"] = False
    assert not validate_time_reversal_sewing_report(
        unverified, valley_members=["v"], theta_square=-1
    )

    odd_rank = deepcopy(evidence)
    row = odd_rank["rows"][0]
    row["source_band_indices_vasp"] = [1]
    row["target_band_indices_vasp"] = [1]
    row["overlap_singular_values"] = [1.0]
    assert not validate_time_reversal_sewing_report(
        odd_rank, valley_members=["v"], theta_square=-1
    )


def test_projector_selection_uses_workflow_exactly_without_fallback():
    seed = np.diag([1.0, 0.0])
    adapted = np.diag([0.0, 1.0])

    selected, provenance, blockers = select_trusted_valley_projectors(
        workflow_decisions={
            "by_kpoint": {
                "G": {
                    "direct": {
                        "workflow_path": "direct_qcut",
                        "readiness_level": "trusted",
                    },
                    "adapted": {
                        "workflow_path": "symmetry_adapted",
                        "readiness_level": "trusted",
                    },
                    "blocked": {
                        "workflow_path": "blocked",
                        "readiness_level": "blocked",
                    },
                    "missing_adapted": {
                        "workflow_path": "symmetry_adapted",
                        "readiness_level": "trusted",
                    },
                },
            },
        },
        seed_projectors_by_kpoint={"G": {
            "direct": seed,
            "adapted": seed,
            "blocked": seed,
            "missing_adapted": seed,
        }},
        symmetry_adapted_projectors_by_kpoint={"G": {
            "adapted": adapted,
        }},
    )

    assert selected["G"]["direct"] is seed
    assert selected["G"]["adapted"] is adapted
    assert "blocked" not in selected["G"]
    assert "missing_adapted" not in selected["G"]
    assert {
        key: provenance["G"]["direct"][key]
        for key in ("workflow_path", "projector_kind")
    } == {
        "workflow_path": "direct_qcut",
        "projector_kind": "fixed_center_seed",
    }
    assert {
        key: provenance["G"]["adapted"][key]
        for key in ("workflow_path", "projector_kind")
    } == {
        "workflow_path": "symmetry_adapted",
        "projector_kind": "symmetry_adapted",
    }
    assert provenance["G"]["direct"]["projector_shape"] == [2, 2]
    assert provenance["G"]["adapted"]["projector_shape"] == [2, 2]
    assert provenance["G"]["direct"]["projector_fingerprint"] != (
        provenance["G"]["adapted"]["projector_fingerprint"]
    )
    assert "trusted_projector_workflow_blocked:G:blocked" in blockers
    assert (
        "trusted_projector_missing:G:missing_adapted:symmetry_adapted"
        in blockers
    )


def test_required_trim_ignores_unrelated_failed_sample():
    coefficients = np.asarray([[[1.0 + 0.0j]]])
    report = build_time_reversal_sewing_report(
        kpoint_frac_by_name={
            "G": np.zeros(3),
            "unrelated": np.asarray([0.5, 0.0, 0.0]),
        },
        g_vectors_frac_by_kpoint={
            "G": np.zeros((1, 3), dtype=int),
            "unrelated": np.asarray([[0, 0, 0], [-1, 0, 0]]),
        },
        coefficients_by_kpoint={
            "G": coefficients,
            "unrelated": np.asarray([[[2.0 + 0.0j, 0.0 + 0.0j]]]),
        },
        band_indices_by_kpoint={
            name: np.asarray([1]) for name in ("G", "unrelated")
        },
        valley_projectors_by_kpoint={
            name: {"v": np.eye(1)} for name in ("G", "unrelated")
        },
        valley_projector_provenance_by_kpoint={
            name: {"v": {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }}
            for name in ("G", "unrelated")
        },
        projector_selection_blockers=[],
        time_reversal_valley_mapping={"v": "v"},
        spinor=False,
        spinor_convention_verified=True,
    )

    assert report["status"] == "blocked"
    assert next(
        row for row in report["rows"]
        if row["source_kpoint"] == "unrelated"
    )["status"] == "blocked"
    assert validate_time_reversal_sewing_report(
        report,
        valley_members=["v"],
        theta_square=1,
        required_kpoints=["G"],
        required_projector_workflows={"G": {"v": "direct_qcut"}},
    )


def test_required_valley_orbit_ignores_unrelated_valley_covariance_failure():
    coefficients = np.asarray([[[1.0 + 0.0j]]])
    report = build_time_reversal_sewing_report(
        kpoint_frac_by_name={"G": np.zeros(3)},
        g_vectors_frac_by_kpoint={"G": np.zeros((1, 3), dtype=int)},
        coefficients_by_kpoint={"G": coefficients},
        band_indices_by_kpoint={"G": np.asarray([1])},
        valley_projectors_by_kpoint={"G": {"v1": np.eye(1)}},
        valley_projector_provenance_by_kpoint={
            "G": {"v1": {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }},
        },
        projector_selection_blockers=[
            "trusted_projector_workflow_blocked:G:v2"
        ],
        time_reversal_valley_mapping={"v1": "v1", "v2": "v2"},
        spinor=False,
        spinor_convention_verified=True,
    )

    assert report["status"] == "blocked"
    covariance = report["rows"][0]["projector_covariance"]
    assert covariance["v1"]["status"] == "validated"
    assert covariance["v2"]["status"] == "blocked"
    assert validate_time_reversal_sewing_report(
        report,
        valley_members=["v1"],
        theta_square=1,
        required_kpoints=["G"],
        required_projector_workflows={"G": {"v1": "direct_qcut"}},
    )
    assert not validate_time_reversal_sewing_report(
        report,
        valley_members=["v2"],
        theta_square=1,
        required_kpoints=["G"],
        required_projector_workflows={"G": {"v2": "direct_qcut"}},
    )


def _nontrim_pair_report() -> dict[str, object]:
    coefficients = np.asarray([[[1.0 + 0.0j]]])
    return build_time_reversal_sewing_report(
        kpoint_frac_by_name={
            "Q": np.asarray([0.25, 0.0, 0.0]),
            "QA": np.asarray([0.75, 0.0, 0.0]),
        },
        g_vectors_frac_by_kpoint={
            "Q": np.asarray([[0, 0, 0]]),
            "QA": np.asarray([[-1, 0, 0]]),
        },
        coefficients_by_kpoint={"Q": coefficients, "QA": coefficients},
        band_indices_by_kpoint={
            "Q": np.asarray([1]), "QA": np.asarray([1]),
        },
        valley_projectors_by_kpoint={
            "Q": {"v": np.eye(1)}, "QA": {"v": np.eye(1)},
        },
        valley_projector_provenance_by_kpoint={
            name: {"v": {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }}
            for name in ("Q", "QA")
        },
        projector_selection_blockers=[],
        time_reversal_valley_mapping={"v": "v"},
        spinor=False,
        spinor_convention_verified=True,
    )


def test_required_nontrim_closes_scope_with_distinct_partner():
    report = _nontrim_pair_report()

    assert report["status"] == "validated"
    assert validate_time_reversal_sewing_report(
        report,
        valley_members=["v"],
        theta_square=1,
        required_kpoints=["Q"],
        required_projector_workflows={
            "Q": {"v": "direct_qcut"},
            "QA": {"v": "direct_qcut"},
        },
    )


def test_required_nontrim_missing_partner_blocks():
    report = _nontrim_pair_report()
    report["time_reversal_kpoint_mapping"].pop("QA")

    assert not validate_time_reversal_sewing_report(
        report,
        valley_members=["v"],
        theta_square=1,
        required_kpoints=["Q"],
        required_projector_workflows={
            "Q": {"v": "direct_qcut"},
            "QA": {"v": "direct_qcut"},
        },
    )


def test_required_scope_rejects_numerically_blocked_row():
    report = _nontrim_pair_report()
    row = next(
        value for value in report["rows"]
        if value["source_kpoint"] == "Q"
    )
    row["status"] = "blocked"
    row["blockers"] = ["duplicate_time_reversal_partner_g_vector:Q"]

    assert not validate_time_reversal_sewing_report(
        report,
        valley_members=["v"],
        theta_square=1,
        required_kpoints=["Q"],
        required_projector_workflows={
            "Q": {"v": "direct_qcut"},
            "QA": {"v": "direct_qcut"},
        },
    )
