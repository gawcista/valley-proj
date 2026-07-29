from __future__ import annotations

import importlib
from copy import deepcopy

import numpy as np
import pytest

from valleyscope.projection.sector_projectors import SectorProjectors
from valleyscope.projection.weights import compute_valley_weights
from valleyscope.subspace.valley_basis import _projector_matrix


def _build_target_frame(
    coefficients: np.ndarray,
    *,
    wavecar_rtag: int | None,
):
    try:
        module = importlib.import_module(
            "valleyscope.analysis.target_frame"
        )
    except ModuleNotFoundError:
        pytest.fail("target-frame producer module is missing")
    builder = getattr(module, "build_target_frame", None)
    assert callable(builder), "target-frame producer is missing"
    return builder(coefficients, wavecar_rtag=wavecar_rtag)


def test_exact_orthonormal_target_frame_needs_no_precision_correction():
    coefficients = np.eye(2, dtype=np.complex128).reshape(2, 1, 2)

    result = _build_target_frame(coefficients, wavecar_rtag=None)

    assert result.status == "passed"
    assert result.reason_codes == ()
    assert np.array_equal(result.coefficients, coefficients)
    assert np.array_equal(result.transform, np.eye(2))
    assert result.record["canonicalization"]["applied"] is False
    assert result.record["canonicalization"]["post_gram_error"] == 0.0


def test_rtag45200_small_nonorthogonality_gets_canonical_row_frame():
    overlap = 2.0e-5
    coefficients = np.array(
        [
            [[1.0 + 0.0j, 0.0 + 0.0j]],
            [[overlap, np.sqrt(1.0 - overlap**2)]],
        ],
        dtype=np.complex128,
    )

    result = _build_target_frame(coefficients, wavecar_rtag=45200)

    assert result.status == "passed"
    assert result.record["source_precision"]["complex_dtype"] == "complex64"
    assert result.record["canonicalization"]["method"] == (
        "symmetric_inverse_square_root"
    )
    assert result.record["canonicalization"]["applied"] is True
    assert result.record["canonicalization"]["correction_size"] < (
        result.record["admissibility"]["correction_size_limit"]
    )
    flattened = result.coefficients.reshape(2, -1)
    assert np.allclose(
        flattened @ flattened.conj().T,
        np.eye(2),
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("coefficients", "wavecar_rtag", "reason"),
    [
        (
            np.array([[[1.0, 0.0]], [[1.0, 0.0]]]),
            45200,
            "target_frame_rank_deficient",
        ),
        (
            np.array([[[1.0, 0.0]], [[0.5, np.sqrt(0.75)]]]),
            45200,
            "target_frame_correction_exceeds_precision_bound",
        ),
        (
            np.array([[[1.0, 0.0]], [[0.0, 0.01]]]),
            45200,
            "target_frame_ill_conditioned",
        ),
        (
            np.array([[[1.0, 0.0]], [[2.0e-5, 1.0]]]),
            None,
            "target_frame_source_precision_missing",
        ),
        (
            np.array([[[np.nan, 0.0]], [[0.0, 1.0]]]),
            45200,
            "target_coefficients_malformed",
        ),
    ],
)
def test_target_frame_fail_closed_cases(
    coefficients: np.ndarray,
    wavecar_rtag: int | None,
    reason: str,
):
    result = _build_target_frame(
        coefficients.astype(np.complex128),
        wavecar_rtag=wavecar_rtag,
    )

    assert result.status == "blocked"
    assert result.coefficients is None
    assert reason in result.reason_codes


def test_target_frame_validator_recomputes_and_rejects_tampering():
    module = importlib.import_module("valleyscope.analysis.target_frame")
    coefficients = np.array(
        [
            [[1.0, 0.0]],
            [[2.0e-5, np.sqrt(1.0 - (2.0e-5) ** 2)]],
        ],
        dtype=np.complex128,
    )
    result = module.build_target_frame(
        coefficients,
        wavecar_rtag=45200,
    )
    tampered = deepcopy(result.record)
    tampered["canonicalization"]["correction_size"] = 0.0

    validation = module.validate_target_frame_record(
        tampered,
        coefficients,
        wavecar_rtag=45200,
    )

    assert validation.status == "blocked"
    assert "target_frame_recomputation_mismatch" in validation.reason_codes


def test_canonical_frame_keeps_group_projector_and_valley_block_algebra():
    overlap = 2.0e-5j
    source = np.array(
        [
            [[1.0, 0.0]],
            [[overlap, np.sqrt(1.0 - abs(overlap) ** 2)]],
        ],
        dtype=np.complex128,
    )
    result = _build_target_frame(source, wavecar_rtag=45200)
    canonical = result.coefficients
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    transformed = canonical.reshape(2, 2) @ swap.T
    representation = (
        canonical.reshape(2, 2).conj() @ transformed.T
    )
    projector_0 = _projector_matrix(
        canonical,
        np.array([True, False]),
    )
    projector_1 = _projector_matrix(
        canonical,
        np.array([False, True]),
    )
    basis_0 = np.linalg.eigh(projector_0)[1][:, -1:]
    basis_1 = np.linalg.eigh(projector_1)[1][:, -1:]

    assert np.allclose(
        representation.conj().T @ representation,
        np.eye(2),
    )
    assert np.allclose(representation @ representation, np.eye(2))
    assert np.allclose(
        representation @ projector_0 @ representation.conj().T,
        projector_1,
    )
    assert np.allclose(
        np.abs(basis_1.conj().T @ representation @ basis_0),
        np.ones((1, 1)),
        atol=1.0e-12,
    )


def test_target_frame_does_not_change_raw_qcut_reporting_weights():
    source = np.array(
        [
            [[1.0, 0.0]],
            [[2.0e-5, np.sqrt(1.0 - (2.0e-5) ** 2)]],
        ],
        dtype=np.complex128,
    )
    original = source.copy()
    projectors = SectorProjectors(
        sector_masks={"v0": np.array([True, False])},
        center_masks={"c0": np.array([True, False])},
        overlap_mask=np.array([False, False]),
        qcut=0.1,
        warnings=[],
    )
    before = compute_valley_weights(source, projectors)

    result = _build_target_frame(source, wavecar_rtag=45200)
    after = compute_valley_weights(source, projectors)

    assert np.array_equal(source, original)
    assert [row.sector_weights for row in after] == [
        row.sector_weights for row in before
    ]
    assert not np.array_equal(result.coefficients, source)
