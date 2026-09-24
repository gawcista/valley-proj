"""End-to-end numerical trust checks, without patched producers."""

import h5py
import numpy as np
import pytest

from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

from tests.portable_numerical_chain import (
    assert_broken_coefficients_blocked,
    assert_numerical_positive,
    run_numerical_workflow,
    write_numerical_fixture,
)


def test_generated_fixture_respects_nonmagnetic_spinor_time_reversal(tmp_path):
    """The synthetic input must obey its V1 domain, including the M TRIM."""
    write_numerical_fixture(tmp_path)
    with h5py.File(tmp_path / "wavefunctions.h5") as h5:
        rows = {group["name"].asstr()[()]: group for group in h5["kpoints"].values()}
        for source_name, target_name in {"GM": "GM", "K": "KA", "KA": "K", "M": "M"}.items():
            source, target = rows[source_name], rows[target_name]
            shift = np.rint(source["frac"][()] + target["frac"][()]).astype(int)
            lookup = {tuple(g): i for i, g in enumerate(target["g_vectors_frac"][()])}
            partners = [-g - shift for g in source["g_vectors_frac"][()]]
            assert all(tuple(g) in lookup for g in partners), source_name
            permutation = [lookup[tuple(g)] for g in partners]
            assert sorted(permutation) == list(range(6))
            coefficients = source["coefficients"][()]
            transformed = np.zeros_like(coefficients)
            transformed[:, 0, permutation] = coefficients[:, 1, :].conj()
            transformed[:, 1, permutation] = -coefficients[:, 0, :].conj()
            target_frame = target["coefficients"][()].reshape(4, -1)
            transformed = transformed.reshape(4, -1)
            sewing = transformed @ target_frame.conj().T
            np.testing.assert_allclose(sewing @ target_frame, transformed, atol=1e-12)
            np.testing.assert_allclose(sewing @ sewing.conj().T, np.eye(4), atol=1e-12)
            projector_plus = np.diag([1, 1, 0, 0])
            np.testing.assert_allclose(
                sewing.conj().T @ projector_plus @ sewing,
                np.diag([0, 0, 1, 1]), atol=1e-12,
            )


def test_generated_spinful_wavefunctions_reach_exact_reduced_ebr(tmp_path):
    result = run_numerical_workflow(tmp_path)
    ingestion = result["ingestion"]
    assert ingestion["validation_errors"] == []
    assert ingestion["final_reduced_ebr_result_count"] == 2, (
        result["reports"]["valley_ebr_problem_instances_json"]
    )
    assert_numerical_positive(result)
    reports = result["reports"]
    candidates = reports["valley_ebr_input_candidates_json"]
    reversed_report = build_ebr_problem_instances(
        ebr_input_candidates={**candidates, "candidates": list(reversed(candidates["candidates"]))},
        projected_hsp_coverage=reports["sampled_k_coverage_json"]["projected_subspace_hsp_coverage"],
    )
    assert reversed_report["ready_instance_count"] == 2
    assert all(row["expected_hsps"] == ["GM", "K", "KA", "M"] for row in reversed_report["instances"])


def test_normalized_valley_pure_states_without_symmetry_closure_are_blocked(tmp_path):
    result = run_numerical_workflow(tmp_path, broken_coefficients=True)
    assert_broken_coefficients_blocked(result)


def test_generated_numerical_chain_reaches_public_standard_outputs(tmp_path):
    result = run_numerical_workflow(tmp_path, profile="standard")
    assert "diagnostics_h5" not in result["outputs"]
    assert result["ingestion"]["validation_errors"] == []
    assert result["ingestion"]["final_reduced_ebr_result_count"] == 2


@pytest.mark.parametrize("invalid_entry", [True, False])
def test_malformed_auto_table_blocks_only_affected_numerical_bundle(
    tmp_path, monkeypatch, invalid_entry,
):
    """Corrupt one real reduced table, never the numerical evidence or readiness."""
    import valleyscope.analysis.irreptables_runtime_table_builder as builder
    original_builder = builder.build_auto_canonical_reduced_ebr_table
    table_count = 0

    def corrupt_first_table(**kwargs):
        nonlocal table_count
        table = original_builder(**kwargs)
        table_count += 1
        if table_count == 1:
            table["ebrs"][0]["vector"][0] = invalid_entry
        return table

    monkeypatch.setattr(builder, "build_auto_canonical_reduced_ebr_table", corrupt_first_table)
    result = run_numerical_workflow(tmp_path)
    mapping = result["reports"]["valley_reduced_ebr_mapping_json"]
    assert table_count == 2
    assert mapping["status"] == "partial"
    assert mapping["table_status"] == "partial"
    assert len(mapping["solutions"]) == 1
    assert mapping["solutions"][0]["status"] == "solved_exact"
    assert len(mapping["excluded_bundles"]) == 1
    assert "vector must be nonnegative integers" in mapping["excluded_bundles"][0]["reason"]
    assert mapping["solutions"][0]["bundle_id"] != mapping["excluded_bundles"][0]["bundle_id"]
    assert result["ingestion"]["validation_errors"] == []
    assert result["ingestion"]["final_reduced_ebr_result_count"] == 1
    assert result["ingestion"]["final_mapping_excluded_bundle_count"] == 1
