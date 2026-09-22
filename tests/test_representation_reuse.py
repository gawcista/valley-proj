"""Numerical reuse must not reuse trust decisions or stale input data."""

from copy import deepcopy

import numpy as np
import pytest

from valleyscope.symmetry import plane_wave_action as action
from valleyscope.analysis import symmetry_eigenvalue_diagnostic as diagnostic


def _inputs():
    return {
        "coefficients": np.array([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]], dtype=complex),
        "q_cart": np.array([[1., 0., 0.], [-1., 0., 0.]]),
        "rotation_cart": np.eye(3),
        "translation_cart": np.zeros(3),
        "spin_rotation": np.eye(2, dtype=complex),
        "tolerance": 1e-6,
    }


def _assert_same(actual, expected):
    np.testing.assert_array_equal(actual.matrix, expected.matrix)
    np.testing.assert_array_equal(actual.mapping, expected.mapping)
    for field in ("mapping_miss_count", "norm_preservation_residual",
                  "relative_norm_preservation_residual", "relative_target_subspace_residual"):
        assert getattr(actual, field) == getattr(expected, field)


def test_reuse_returns_immutable_numerical_result_and_recomputes_after_eviction(monkeypatch):
    inputs = _inputs()
    calls = []
    original = action.build_plane_wave_representation

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(action, "build_plane_wave_representation", counted)
    cache = action.RunLocalPlaneWaveRepresentationCache(max_entries=1)
    first = cache.build(**inputs)
    _assert_same(first, original(**inputs))
    _assert_same(cache.build(**deepcopy(inputs)), first)
    assert len(calls) == 1
    for array in (first.matrix, first.mapping):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    inputs["translation_cart"][0] = 0.25
    _assert_same(cache.build(**inputs), original(**inputs))
    inputs["translation_cart"][0] = 0.
    _assert_same(cache.build(**inputs), first)
    assert len(calls) == 3


@pytest.mark.parametrize("field", [
    "coefficients", "q_cart", "rotation_cart", "translation_cart",
    "spin_rotation", "tolerance",
])
def test_reuse_binds_actual_inputs_after_in_place_mutation(field, monkeypatch):
    inputs = _inputs()
    calls = []
    original = action.build_plane_wave_representation

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(action, "build_plane_wave_representation", counted)
    cache = action.RunLocalPlaneWaveRepresentationCache()
    cache.build(**inputs)
    if field == "tolerance":
        inputs[field] *= 0.5
    else:
        inputs[field].flat[0] += 0.25
    _assert_same(cache.build(**inputs), original(**inputs))
    assert len(calls) == 2


def test_grid_identity_reuse_rechecks_order_and_in_place_changes(monkeypatch):
    q = _inputs()["q_cart"]
    original = action.reciprocal_grid_identity
    calls = []

    def counted(q):
        calls.append(1)
        return original(q)

    monkeypatch.setattr(action, "reciprocal_grid_identity", counted)
    cache = action.RunLocalPlaneWaveRepresentationCache()
    assert cache.grid_identity(q) == original(q)
    assert cache.grid_identity(q.copy()) == original(q)
    assert len(calls) == 1
    q[:] = q[::-1]
    assert cache.grid_identity(q) == original(q)
    assert len(calls) == 2
    q[0, 0] = np.nan
    with pytest.raises(ValueError):
        cache.grid_identity(q)


@pytest.mark.parametrize("field", ["rotation_cart", "spin_rotation"])
def test_same_bytes_different_shape_cannot_hit_valid_cached_result(field):
    inputs = _inputs()
    cache = action.RunLocalPlaneWaveRepresentationCache()
    cache.build(**inputs)
    inputs[field] = inputs[field].reshape(-1)
    with pytest.raises(ValueError):
        cache.build(**inputs)


def test_grid_identity_reuse_preserves_signed_zero_serialization():
    q = np.array([[0., 1., 2.]])
    cache = action.RunLocalPlaneWaveRepresentationCache()
    positive = cache.grid_identity(q)
    q[0, 0] = -0.
    negative = cache.grid_identity(q)
    assert negative == action.reciprocal_grid_identity(q)
    assert negative != positive


@pytest.mark.parametrize("raw_mode", ["normal", "missing_mapping", "mapping_miss", "no_raw"])
def test_shared_raw_and_diagnostics_keep_independent_failure_semantics(raw_mode, monkeypatch):
    inputs = _inputs()
    operation = {
        "operation_id": 7, "order": 1, "kind": "identity",
        "rotation_frac": np.eye(3), "rotation_cart": np.eye(3),
        "translation_frac": np.zeros(3), "translation_cart": np.zeros(3),
        "sector_mapping": {"a": "a", "b": "b"},
        "preserved": {"a": True, "b": True},
    }
    if raw_mode == "missing_mapping":
        operation["sector_mapping"] = {}
    if raw_mode == "mapping_miss":
        operation["rotation_cart"] = np.diag([-1., -1., 1.])
        inputs["q_cart"][1, 0] = 2.
    arguments = {
        "kpoint_name": "GM", "k_frac": np.zeros(3),
        "q_cart": inputs["q_cart"], "coefficients": inputs["coefficients"],
        "symmetry_payload": {"detected_operations": [operation]},
    }
    original = action.build_plane_wave_representation
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(action, "build_plane_wave_representation", counted)
    cache = action.RunLocalPlaneWaveRepresentationCache()
    if raw_mode != "no_raw":
        raw = diagnostic.build_raw_representations_for_kpoint(
            **arguments, numerical_cache=cache,
        )
        assert (raw[7]["D_raw"] is None) == (raw_mode != "normal")
    payload = {}
    rows = diagnostic.symmetry_eigenvalue_diagnostics_for_kpoint(
        **arguments, basis_payload=None, representation_payload=payload,
        valley_names=["a", "b"], numerical_cache=cache,
    )
    assert len(rows) == 4
    assert {row["target_valley"] for row in rows} == {"a", "b"}
    assert all(row["diagnostic_only"] and not row["local_irrep_ready"] for row in rows)
    assert all(row["plane_wave_mapping_complete"] == (raw_mode != "mapping_miss") for row in rows)
    assert len(calls) == 1
    expected = original(
        inputs["coefficients"], inputs["q_cart"], operation["rotation_cart"],
        operation["translation_cart"],
        spin_rotation=diagnostic.spin_lift_from_orthogonal(operation["rotation_cart"]),
    )
    for record in payload["GM"].values():
        np.testing.assert_array_equal(record["D_raw"], expected.matrix)


def test_workflow_builds_each_k_operation_only_once(tmp_path, monkeypatch):
    from tests.portable_numerical_chain import run_numerical_workflow, assert_numerical_positive

    calls = []
    original = action.build_plane_wave_representation

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    # Both imports are covered so the pre-reuse production path is counted.
    monkeypatch.setattr(action, "build_plane_wave_representation", counted)
    monkeypatch.setattr(diagnostic, "build_plane_wave_representation", counted, raising=False)
    assert_numerical_positive(run_numerical_workflow(tmp_path))
    assert len(calls) == 10  # Three ops at GM/K/KA and identity at M.


def test_scoped_frame_recomputed_once_per_validation_and_raw_mutation_blocks(monkeypatch):
    from tests.test_scoped_representation_evidence import _raw_inputs
    from valleyscope.analysis import target_frame, scoped_representation_evidence as scoped

    inputs = _raw_inputs(scope_kind="local_irrep", required_operation_ids=(2,))
    original = target_frame.build_target_frame
    inputs["target_frame_record"] = original(inputs["target_coefficients"], wavecar_rtag=None).record
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(target_frame, "build_target_frame", counted)
    monkeypatch.setattr(scoped, "build_target_frame", counted)
    record = scoped.build_scoped_representation_evidence(**inputs).to_record()
    assert record["status"] == "passed"
    assert len(calls) == 1
    assert scoped.validate_scoped_representation_evidence_record(record, **inputs).status == "passed"
    assert len(calls) == 2
    inputs["target_coefficients"].flat[0] += 0.25
    assert scoped.validate_scoped_representation_evidence_record(record, **inputs).status == "blocked"
    assert len(calls) == 3


def test_grid_identity_shared_only_within_one_raw_validation(monkeypatch):
    from tests.test_scoped_representation_evidence import _raw_inputs
    from valleyscope.analysis import scoped_representation_evidence as scoped

    inputs = _raw_inputs(scope_kind="local_irrep", required_operation_ids=(2, 5))
    evidence = inputs["plane_wave_evidence"]
    original = scoped.reciprocal_grid_identity
    calls = []

    def counted(q):
        calls.append(1)
        return original(q)

    monkeypatch.setattr(scoped, "reciprocal_grid_identity", counted)
    reasons = []
    rows, maps = scoped._plane_wave_rows([2, 5], evidence, 1e-8, reasons)
    assert not reasons
    assert all(row["passed"] for row in rows)
    assert maps == {2: [0, 1], 5: [0, 1]}
    assert len(calls) == 1
    evidence[5]["q_cart"] = evidence[5]["q_cart"].copy()
    evidence[5]["q_cart"][0, 0] = -0.
    reasons = []
    rows, _ = scoped._plane_wave_rows([2, 5], evidence, 1e-8, reasons)
    assert "plane_wave_grid_identity_mismatch" in reasons
    assert rows[0]["passed"] and not rows[1]["passed"]
    assert len(calls) == 3  # New validation recomputes both actual identities.


def test_cached_zero_miss_collision_remains_blocked_by_real_scoped_validator():
    from tests.test_scoped_representation_evidence import _raw_inputs
    from valleyscope.analysis.scoped_representation_evidence import build_scoped_representation_evidence

    inputs = _raw_inputs(scope_kind="local_irrep", required_operation_ids=(2,))
    row = inputs["plane_wave_evidence"][2]
    cache = action.RunLocalPlaneWaveRepresentationCache()
    arguments = (inputs["target_coefficients"], row["q_cart"], np.eye(3), np.zeros(3))
    assert cache.build(*arguments).mapping.tolist() == [0, 1]
    row["q_cart"][1] = row["q_cart"][0]
    result = cache.build(*arguments)
    _assert_same(cache.build(*arguments), result)
    assert result.mapping_miss_count == 0
    assert result.relative_norm_preservation_residual == 0.
    assert result.mapping.tolist() == [0, 0]
    inputs["representations"][2] = result.matrix
    row["reciprocal_grid_identity"] = cache.grid_identity(row["q_cart"])
    row["source_to_target_map"] = result.mapping.tolist()
    record = build_scoped_representation_evidence(**inputs).to_record()
    assert record["status"] == "blocked"
    assert "plane_wave_mapping_not_bijective" in record["reason_codes"]


@pytest.mark.parametrize("mutation", ["source", "precision", "record"])
def test_corrected_frame_revalidation_binds_raw_source_precision_and_record(mutation):
    from tests.test_scoped_representation_evidence import _raw_inputs
    from valleyscope.analysis.target_frame import build_target_frame
    from valleyscope.analysis.scoped_representation_evidence import (
        build_scoped_representation_evidence, validate_scoped_representation_evidence_record,
    )

    inputs = _raw_inputs(scope_kind="local_irrep", required_operation_ids=(2,))
    overlap = 2e-5
    source = np.array([[[1., 0.]], [[overlap, np.sqrt(1 - overlap**2)]]], dtype=complex)
    frame = build_target_frame(source, wavecar_rtag=45200)
    inputs.update(source_target_coefficients=source, target_coefficients=frame.coefficients,
                  wavecar_rtag=45200, target_frame_record=deepcopy(frame.record))
    record = build_scoped_representation_evidence(**inputs).to_record()
    assert record["status"] == "passed"
    if mutation == "source":
        source[0, 0, 0] += 0.2
    elif mutation == "precision":
        inputs["wavecar_rtag"] = 45210
    else:
        inputs["target_frame_record"]["contract_identity"] = "sha256:" + "0" * 64
    validation = validate_scoped_representation_evidence_record(record, **inputs)
    assert validation.status == "blocked"
