from __future__ import annotations

import numpy as np

from valleyscope.analysis.time_reversal_sewing import (
    build_time_reversal_sewing_report,
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
