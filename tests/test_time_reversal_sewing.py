from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from valleyscope.analysis.time_reversal_sewing import (
    build_time_reversal_sewing_report,
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
