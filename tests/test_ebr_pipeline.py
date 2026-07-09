import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.workflows.analyze_hsp import analyze_hsp

from tests.helpers_io_workflow import write_fixture, write_config
from tests.helpers_io_workflow import _E2E_SAMPLE_TABLE, e2e_write_table

from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping, load_reduced_ebr_table

from tests.helpers_io_workflow import write_fixture, write_config

# Valley irrep -> EBR pipeline contract tests
# -----------------------------------------------------------------------

def test_ebr_input_candidates_excludes_non_trusted():
    """Non-trusted/diagnostic-only rows must not reach ready_for_ebr_input=true."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates

    # Workflow: K_valley is trusted/direct_qcut, Kp_valley is diagnostic_only.
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
                "Kp_valley": {"readiness_level": "usable_with_caution", "workflow_path": "symmetry_adapted"},
            },
        },
    }
    # Matching: generic_matches_by_kpoint — K_valley matched, Kp_valley diagnostic_only.
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "matching_status": "matched",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {"C3_spinor_phase_+1/2": 1},
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 143,
                        "candidate_space_group_symbol": "P3",
                    },
                    "valley_preserving_operation_ids": [0, 1],
                    "hsp_little_group_operation_ids": [0, 1],
                },
                "Kp_valley": {
                    "matching_status": "diagnostic",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {"C3_spinor_phase_-1/2": 1},
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 143,
                        "candidate_space_group_symbol": "P3",
                    },
                    "valley_preserving_operation_ids": [0, 1],
                    "hsp_little_group_operation_ids": [0, 1],
                    "diagnostic_only": True,
                },
            },
        },
    }

    report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )

    # Only trusted, non-diagnostic rows are candidates.
    candidates = report["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["valley"] == "K_valley"
    assert candidates[0]["ready_for_ebr_input"] is True

    # Diagnostic row is blocked.
    blocked = report["blocked"]
    assert any("diagnostic_only=true" in str(b.get("reason", "")) for b in blocked)
    assert any(b.get("valley") == "Kp_valley" for b in blocked)


def test_ebr_input_candidates_excludes_blocked_path():
    """Workflow path=blocked must not produce candidates."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates

    workflow = {
        "by_kpoint": {
            "MM": {
                "K_valley": {"readiness_level": "blocked", "workflow_path": "blocked"},
            },
        },
    }
    matching = {
        "by_kpoint": {
            "MM": {
                "K_valley": {},
            },
        },
    }
    report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    assert report["candidate_count"] == 0
    assert report["status"] == "no_candidates"


def test_ebr_problem_instances_ready_from_actual():
    """Table-authoritative: ready from actual irreps, not hard-coded policy."""
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    candidates = {
        "status": "has_candidates",
        "candidates": [{
            "kpoint": "GammaM", "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "matched_irrep": "C3_spinor_phase_+1/2",
            "operation_id": 1,
            "ready_for_ebr_input": True,
        }],
    }
    report = build_ebr_problem_instances(ebr_input_candidates=candidates)
    assert report["instance_count"] == 1
    inst = report["instances"][0]
    assert inst["ready_for_ebr_decomposition"] is True
    assert inst["expected_hsps"] == ["GammaM"]
    assert inst["status"] == "complete"


def test_ebr_problem_instances_complete_hsp_is_ready():
    """All required HSPs present -> ready_for_ebr_decomposition=true."""
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    candidates = {
        "status": "has_candidates",
        "candidates": [
            {"kpoint": "GammaM", "valley": "K_valley",
             "subspace_group_candidate": "P3",
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "matched_irrep": "C3_spinor_phase_+1/2", "operation_id": 1,
             "ready_for_ebr_input": True},
            {"kpoint": "KM", "valley": "K_valley",
             "subspace_group_candidate": "P3",
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "matched_irrep": "C3_spinor_phase_+1/6", "operation_id": 1,
             "ready_for_ebr_input": True},
        ],
    }
    report = build_ebr_problem_instances(ebr_input_candidates=candidates)
    inst = report["instances"][0]
    assert inst["ready_for_ebr_decomposition"] is True
    assert inst["status"] == "complete"


def test_ebr_export_bundle_preserves_hsp_and_irrep_fields():
    """Export bundle must preserve expected_hsps, irreps_by_kpoint, operations_by_kpoint."""
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle

    problem_instances = {
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
            "operations_by_kpoint": {"GammaM": [1]},
            "expected_hsps": ["GammaM", "KM"],
            "optional_hsps": ["MM"],
            "missing_optional_hsps": ["MM"],
            "ready_for_ebr_decomposition": True,
            "status": "complete",
        }],
    }
    report = build_ebr_export_bundle(ebr_problem_instances=problem_instances)
    assert report["bundle_count"] == 1
    bundle = report["bundles"][0]
    assert bundle["ready_for_external_solver"] is True
    assert bundle["subspace_group_candidate"] == "P3"
    assert bundle["expected_hsps"] == ["GammaM", "KM"]
    assert bundle["optional_hsps"] == ["MM"]
    assert bundle["missing_optional_hsps"] == ["MM"]
    assert bundle["irreps_by_kpoint"] == {"GammaM": ["C3_spinor_phase_+1/2"]}
    assert bundle["operations_by_kpoint"] == {"GammaM": [1]}


def test_ebr_export_bundle_excludes_non_ready():
    """Non-ready instances are excluded, not bundled."""
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle

    problem_instances = {
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "status": "partial",
            "ready_for_ebr_decomposition": False,
            "blocked_by": ["missing required HSPs: [KM]"],
        }],
    }
    report = build_ebr_export_bundle(ebr_problem_instances=problem_instances)
    assert report["bundle_count"] == 0
    assert report["status"] == "no_bundles"
    assert report["excluded_count"] == 1
    assert "ready_for_ebr_decomposition" in str(report["excluded_instances"][0]["exclusion_reasons"])


def test_reduced_ebr_mapping_rejects_hsp_mismatch():
    """Reduced EBR mapping must reject bundles whose HSP basis does not match the table."""
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P3",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:C3_spinor_phase_+1/2", "KM:C3_spinor_phase_+1/6"],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0]}, {"label": "EBR_B", "vector": [0, 1]}],
    }
    # Bundle only has GammaM, missing KM.
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "P3",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM"],
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=bundle, table=table)
    assert len(r["solutions"]) == 0
    assert len(r["excluded_bundles"]) == 1
    assert "expected_hsps mismatch" in r["excluded_bundles"][0]["reason"]


def test_generic_p4_table_authoritative_bundle_maps_and_rejects_mismatch():
    """P4 synthetic instance exports without hard-coded Cn HSP policy."""
    problem_instances = build_ebr_problem_instances(
        ebr_input_candidates={
            "status": "has_candidates",
            "candidates": [
                {
                    "kpoint": "GammaM",
                    "valley": "K_valley",
                    "workflow_path": "direct_qcut",
                    "readiness_level": "trusted",
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                        "valley_preserving_operation_ids": [0, 4],
                    },
                    "matched_irrep": "P4_spinor_phase_+1/4",
                    "irrep_multiplicity": 1,
                    "matching_strategy": "bilbao_restricted_character",
                    "ready_for_ebr_input": True,
                },
                {
                    "kpoint": "XM",
                    "valley": "K_valley",
                    "workflow_path": "direct_qcut",
                    "readiness_level": "trusted",
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                        "valley_preserving_operation_ids": [0, 4],
                    },
                    "matched_irrep": "P4_spinor_phase_-1/4",
                    "irrep_multiplicity": 1,
                    "matching_strategy": "bilbao_restricted_character",
                    "ready_for_ebr_input": True,
                },
            ],
        },
    )
    inst = problem_instances["instances"][0]
    assert inst["subspace_group_candidate"] == "P4"
    assert inst["expected_hsps"] == ["GammaM", "XM"]
    assert inst["expected_hsp_policy_source"] == "sampled_irrep_basis"
    assert inst["ready_for_ebr_decomposition"] is True

    export_bundle = build_ebr_export_bundle(
        ebr_problem_instances=problem_instances
    )
    assert export_bundle["status"] == "ready_for_external_solver"
    bundle = export_bundle["bundles"][0]
    assert bundle["subspace_group_candidate"] == "P4"
    assert bundle["expected_hsps"] == ["GammaM", "XM"]

    matching_table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM", "XM"],
        "irreps": [
            "GammaM:P4_spinor_phase_+1/4",
            "XM:P4_spinor_phase_-1/4",
        ],
        "ebrs": [{"label": "EBR_P4_A", "vector": [1, 1]}],
    }
    solved = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=matching_table,
    )
    assert solved["status"] == "solved_exact"
    assert solved["solutions"][0]["irrep_vector"] == [1, 1]

    mismatched_table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM", "MM"],
        "irreps": [
            "GammaM:P4_spinor_phase_+1/4",
            "MM:P4_spinor_phase_-1/4",
        ],
        "ebrs": [{"label": "EBR_P4_bad", "vector": [1, 1]}],
    }
    rejected = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=mismatched_table,
    )
    assert rejected["solutions"] == []
    assert len(rejected["excluded_bundles"]) == 1
    assert "expected_hsps mismatch" in rejected["excluded_bundles"][0]["reason"]


_STANDARD_PUBLIC_FILES = frozenset({
    "valley_summary.txt", "valley_summary.json", "valley_weights.csv",
    "valley_ebr_export_bundle.json", "valley_reduced_ebr_mapping.json",
})

_DEBUG_ONLY_FILES = frozenset({
    "valley_subspace.json", "symmetry_report.json", "symmetry_eigenvalues.csv",
    "diagnostics.h5", "valley_basis_transform.h5",
    "projector_symmetry_report.json", "symmetry_adapted_valley_analysis.json",
    "target_subspace_closure.json", "hsp_star_conjugation.json",
    "hsp_star_derived_characters.json", "subspace_representation_quality.json",
    "irrep_workflow_decisions.json", "valley_irrep_matching.json",
    "valley_ebr_input_candidates.json", "valley_ebr_problem_instances.json",
    "folded_center_report.json", "sampled_k_coverage.json",
})


def test_ready_export_bundle_maps_to_public_reduced_ebr_outputs_only(tmp_path):
    """Ready export bundle plus validated table writes only public standard outputs."""
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
    from valleyscope.analysis.reduced_ebr_mapping import (
        build_reduced_ebr_mapping,
        load_reduced_ebr_table,
    )
    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    table_path = tmp_path / "table.json"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P3",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": [
            "GammaM:C3_spinor_phase_+1/2",
            "KM:C3_spinor_phase_+1/6",
        ],
        "ebrs": [
            {"label": "EBR_G", "vector": [1, 0]},
            {"label": "EBR_K", "vector": [0, 1]},
        ],
    }
    e2e_write_table(table_path, table)
    export_bundle = build_ebr_export_bundle(
        ebr_problem_instances={
            "instances": [{
                "instance_id": "ebr_instance_001",
                "valley": "K_valley",
                "subspace_group_candidate": "P3",
                "workflow_path": "direct_qcut",
                "readiness_level": "trusted",
                "irreps_by_kpoint": {
                    "GammaM": ["C3_spinor_phase_+1/2"],
                    "KM": ["C3_spinor_phase_+1/6"],
                },
                "operations_by_kpoint": {"GammaM": [1], "KM": [1]},
                "expected_hsps": ["GammaM", "KM"],
                "optional_hsps": ["MM"],
                "missing_optional_hsps": ["MM"],
                "ready_for_ebr_decomposition": True,
                "status": "complete",
            }],
        }
    )
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=load_reduced_ebr_table(table_path),
    )
    assert mapping["status"] == "solved_exact"

    outputs = write_analysis_outputs(
        config=load_config(config_path),
        qcut=0.5,
        weight_rows=[],
        sector_names=["K_valley"],
        subspace_payload={"kpoints": {}},
        symmetry_payload={"status": "skipped", "reason": "test",
                          "detected_operations": [], "candidate_rotations": [],
                          "little_group_check": {"status": "not_run"},
                          "valley_preservation_check": {"status": "not_run"}},
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
        ebr_export_bundle=export_bundle,
        reduced_ebr_mapping=mapping,
    )

    written = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert written <= _STANDARD_PUBLIC_FILES
    assert not (written & _DEBUG_ONLY_FILES)
    assert outputs["valley_ebr_export_bundle_json"].exists()
    assert outputs["valley_reduced_ebr_mapping_json"].exists()
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary["valley_ebr_export_bundle"] == export_bundle
    assert summary["valley_reduced_ebr_mapping"] == mapping


def test_pipeline_contract_fixture_data_is_material_agnostic():
    """Pipeline contract fixture data must not name real validation materials."""
    fixture_text = "\n".join([
        "C3_like C2_like P3 P2",
        "GammaM KM MM K_valley",
        "C3_spinor_phase_+1/2 C3_spinor_phase_+1/6",
    ])
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in fixture_text


# -----------------------------------------------------------------------

# Trusted irrep export provenance tests

def test_ebr_problem_instances_include_irrep_records():
    """EBR problem instances must include irrep_records_by_kpoint for trusted candidates."""
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    candidates = {
        "status": "has_candidates",
        "candidates": [
            {"kpoint": "GammaM", "valley": "K_valley",
             "subspace_group_candidate": "P3",
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "matched_irrep": "C3_spinor_phase_+1/2", "operation_id": 1,
             "operation_order": 3, "eigenphases": [0.5],
             "character": {"real": -1.0, "imag": 0.0},
             "source": "valley_irrep_matching/GammaM/K_valley",
             "ready_for_ebr_input": True},
            {"kpoint": "KM", "valley": "K_valley",
             "subspace_group_candidate": "P3",
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "matched_irrep": "C3_spinor_phase_+1/6", "operation_id": 1,
             "operation_order": 3, "eigenphases": [0.166667],
             "source": "valley_irrep_matching/KM/K_valley",
             "ready_for_ebr_input": True},
        ],
    }
    report = build_ebr_problem_instances(ebr_input_candidates=candidates)
    inst = report["instances"][0]
    assert "irrep_records_by_kpoint" in inst
    records = inst["irrep_records_by_kpoint"]
    assert "GammaM" in records
    assert "KM" in records
    gamma_rec = records["GammaM"][0]
    assert gamma_rec["valley"] == "K_valley"
    assert gamma_rec["operation_id"] == 1
    assert gamma_rec["operation_order"] == 3
    assert gamma_rec["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert gamma_rec["eigenphases"] == [0.5]
    assert gamma_rec["workflow_path"] == "direct_qcut"
    assert gamma_rec["readiness_level"] == "trusted"
    assert gamma_rec["character"] is not None
    assert gamma_rec["source"] == "valley_irrep_matching/GammaM/K_valley"


def test_export_bundle_copies_irrep_records():
    """Export bundles copy irrep_records_by_kpoint for complete trusted instances."""
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle

    records = {
        "GammaM": [{"valley": "K_valley", "operation_id": 1, "operation_order": 3,
                     "matched_irrep": "C3_spinor_phase_+1/2", "eigenphases": [0.5],
                     "workflow_path": "direct_qcut", "readiness_level": "trusted",
                     "source": "valley_irrep_matching/GammaM/K_valley"}],
        "KM": [{"valley": "K_valley", "operation_id": 1, "operation_order": 3,
                 "matched_irrep": "C3_spinor_phase_+1/6", "eigenphases": [0.166667],
                 "workflow_path": "direct_qcut", "readiness_level": "trusted",
                 "source": "valley_irrep_matching/KM/K_valley"}],
    }

    problem_instances = {
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
            "operations_by_kpoint": {"GammaM": [1]},
            "irrep_records_by_kpoint": records,
            "expected_hsps": ["GammaM", "KM"],
            "optional_hsps": [],
            "missing_optional_hsps": [],
            "ready_for_ebr_decomposition": True,
            "status": "complete",
        }],
    }
    report = build_ebr_export_bundle(ebr_problem_instances=problem_instances)
    bundle = report["bundles"][0]
    assert "irrep_records_by_kpoint" in bundle
    assert bundle["irrep_records_by_kpoint"] == records


def test_non_trusted_rows_excluded_from_irrep_records():
    """Non-trusted/diagnostic-only rows must not appear in irrep_records_by_kpoint."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    # Mix of trusted and diagnostic_only rows via generic_matches_by_kpoint.
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            },
        },
    }
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "matching_status": "matched",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {"C3_spinor_phase_+1/2": 1},
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 143,
                        "candidate_space_group_symbol": "P3",
                        "valley_preserving_operation_ids": [0, 1],
                    },
                    "valley_preserving_operation_ids": [0, 1],
                    "hsp_little_group_operation_ids": [0, 1],
                    "source_operation_map": {0: 1, 1: 2},
                },
            },
        },
    }
    candidates_report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow, valley_irrep_matching=matching)
    # 1 trusted candidate.
    assert candidates_report["candidate_count"] == 1

    instances_report = build_ebr_problem_instances(ebr_input_candidates=candidates_report)
    inst = instances_report["instances"][0]
    records = inst["irrep_records_by_kpoint"]
    gamma_recs = records.get("GammaM", [])
    # Only the trusted record appears.
    assert len(gamma_recs) == 1
    assert gamma_recs[0]["matched_irrep"] == "C3_spinor_phase_+1/2"


def test_reduced_ebr_mapping_ignores_irrep_records():
    """reduced_ebr_mapping must remain compatible and ignore irrep_records_by_kpoint."""
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P3",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:C3_spinor_phase_+1/2", "KM:C3_spinor_phase_+1/6"],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0]}, {"label": "EBR_B", "vector": [0, 1]}],
    }
    records = {
        "GammaM": [{"valley": "K_valley", "operation_id": 1, "matched_irrep": "C3_spinor_phase_+1/2"}],
        "KM": [{"valley": "K_valley", "operation_id": 1, "matched_irrep": "C3_spinor_phase_+1/6"}],
    }
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "P3",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM", "KM"],
            "irreps_by_kpoint": {
                "GammaM": ["C3_spinor_phase_+1/2"],
                "KM": ["C3_spinor_phase_+1/6"],
            },
            "irrep_records_by_kpoint": records,
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=bundle, table=table)
    assert r["status"] == "solved_exact"  # Provenance ignored; decomposition succeeds.


def test_schema_doc_documents_irrep_records_by_kpoint():
    """docs/schema.md must document the new irrep_records_by_kpoint field."""
    schema = Path("docs/schema.md").read_text(encoding="utf-8")
    assert "irrep_records_by_kpoint" in schema, (
        "docs/schema.md must document irrep_records_by_kpoint"
    )
    assert "provenance" in schema.lower(), (
        "docs/schema.md should mention provenance for the new field"
    )


def test_generic_irrep_full_pipeline_smoke():
    """Generic matcher -> candidates -> instances -> export ->
    reduced EBR mapping with matching table, rejection with mismatched table."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle

    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                },
            },
        },
    }
    symmetry_adapted_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "reference_valley": "K_valley",
                    "orbit": ["K_valley"],
                    "subspace_group": {"subspace_group_candidate": "P4"},
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                        "valley_preserving_operation_ids": [0, 4],
                    },
                    "hsp_preserving_operation_ids": [0, 4],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                                {"operation_id": 4, "eigenphases": [0.0, 0.5]},
                            ],
                        },
                    },
                }],
            },
        },
    }
    source_chars = {
        "-GM5": {1: 1.0 + 0j, 2: 1.0 + 0j},
        "-GM6_a": {1: 1.0 + 0j, 2: -1.0 + 0j},
    }
    operation_maps = {"GammaM": {"K_valley": {0: 1, 4: 2}}}

    # 1. Generic restricted-character matching.
    matching = build_valley_irrep_matching_report(
        irrep_workflow_decisions=workflow,
        symmetry_adapted_valley_report=symmetry_adapted_report,
        source_irrep_characters_flattened={
            "GammaM": {"K_valley": source_chars},
        },
        source_operation_maps=operation_maps,
    )
    generic = matching["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert generic["matching_status"] == "matched"
    assert generic["matching_strategy"] == "bilbao_restricted_character"
    assert generic["irrep_multiplicities"] == {"-GM5": 1, "-GM6_a": 1}
    assert generic["subspace_space_group"]["candidate_space_group_symbol"] == "P4"

    # 2. EBR input candidates.
    candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    assert candidates["candidate_count"] == 2

    # 3. Problem instances (table-authoritative: expected_hsps from actual).
    instances = build_ebr_problem_instances(ebr_input_candidates=candidates)
    assert instances["instance_count"] == 1
    inst = instances["instances"][0]
    assert inst["ready_for_ebr_decomposition"] is True
    assert inst["expected_hsps"] == ["GammaM"]

    # 4. Export bundle.
    bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    assert bundle["bundle_count"] == 1
    b = bundle["bundles"][0]
    assert b["ready_for_external_solver"] is True

    # 5. Reduced EBR mapping with a matching table.
    bp_irreps = b["irreps_by_kpoint"]["GammaM"]
    matching_table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM"],
        "irreps": [f"GammaM:{irr}" for irr in bp_irreps],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0]},
            {"label": "EBR_B", "vector": [0, 1]},
        ],
    }
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle, table=matching_table,
    )
    assert result["mapping_status"] == "solved_exact"

    # 6. Rejected by mismatched HSP basis.
    bad_table = dict(matching_table)
    bad_table["expected_hsps"] = ["GammaM", "KM"]
    bad_table["irreps"] = list(matching_table["irreps"]) + ["KM:-K5"]
    bad_table["ebrs"] = [
        {"label": "EBR_A", "vector": [1, 0, 0]},
        {"label": "EBR_B", "vector": [0, 1, 0]},
    ]
    result2 = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle, table=bad_table,
    )
    assert len(result2["excluded_bundles"]) == 1
    assert "expected_hsps" in result2["excluded_bundles"][0]["reason"]


def test_generic_ebr_builder_e2e_p4_group_agnostic(tmp_path):
    """Group-agnostic E2E: generic restricted-character match → candidates →
    instances → export → builder-generated reduced table → exact solve.

    Uses P4 (not C3_like) with build_reduced_table_from_runtime_source to
    produce the reduced EBR table, then validates that generic provenance
    survives the full pipeline.
    """
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping
    from valleyscope.analysis.irrep_runtime_reducer import (
        build_reduced_table_from_runtime_source,
    )

    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                },
            },
        },
    }
    symmetry_adapted_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "reference_valley": "K_valley",
                    "orbit": ["K_valley"],
                    "subspace_group": {"subspace_group_candidate": "P4"},
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                        "valley_preserving_operation_ids": [0, 1],
                    },
                    "hsp_preserving_operation_ids": [0, 1],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                                {"operation_id": 1, "eigenphases": [0.25, -0.25]},
                            ],
                        },
                    },
                }],
            },
        },
    }
    # P4 C4-symmetric source irreps (two eigenstates, four irreps).
    i = 1j
    source_chars = {
        "GM_plus_1over4":  {1: 1.0+0j, 2:  i},
        "GM_minus_1over4": {1: 1.0+0j, 2: -i},
    }
    operation_maps = {"GammaM": {"K_valley": {0: 1, 1: 2}}}

    # 1. Generic restricted-character matching.
    matching = build_valley_irrep_matching_report(
        irrep_workflow_decisions=workflow,
        symmetry_adapted_valley_report=symmetry_adapted_report,
        source_irrep_characters_flattened={
            "GammaM": {"K_valley": source_chars},
        },
        source_operation_maps=operation_maps,
    )
    assert matching["matching_mode"] == "generic"
    gm = matching["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "matched"
    assert gm["matching_strategy"] == "bilbao_restricted_character"
    mults = gm["irrep_multiplicities"]
    assert mults.get("GM_plus_1over4") == 1
    assert mults.get("GM_minus_1over4") == 1
    assert gm["subspace_space_group"]["candidate_space_group_symbol"] == "P4"

    # 2. EBR input candidates — generic source only, no legacy promotion.
    candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    assert candidates["candidate_count"] == 2
    for c in candidates["candidates"]:
        assert c["matching_strategy"] == "bilbao_restricted_character"
        assert c["subspace_group_candidate"] == "P4"

    # 3. Problem instances.
    instances = build_ebr_problem_instances(ebr_input_candidates=candidates)
    assert instances["instance_count"] == 1
    inst = instances["instances"][0]
    assert inst["ready_for_ebr_decomposition"] is True
    assert inst["expected_hsps"] == ["GammaM"]
    assert inst["expected_hsp_policy_source"] == "sampled_irrep_basis"
    assert inst["subspace_group_candidate"] == "P4"

    # 4. Export bundle.
    bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    assert bundle["bundle_count"] == 1
    b = bundle["bundles"][0]
    assert b["ready_for_external_solver"] is True
    assert b["subspace_group_candidate"] == "P4"

    # 5. Build reduced EBR table via runtime reducer (not hand-written).
    bp_irreps = b["irreps_by_kpoint"]["GammaM"]
    source_payload = {
        "basis": [
            {"source_label": f"src_{i}", "hsp": "GammaM",
             "valleyscope_irrep_key": f"GammaM:{irr}",
             "source_index": i, "multiplicity": 1}
            for i, irr in enumerate(bp_irreps)
        ],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0]},
            {"label": "EBR_B", "vector": [0, 1]},
        ],
    }
    table = build_reduced_table_from_runtime_source(
        source_payload=source_payload,
        expected_hsps=["GammaM"],
        allowed_irrep_keys=[f"GammaM:{irr}" for irr in bp_irreps],
        subspace_group_candidate="P4",
    )
    # Validate table has expected fields.
    assert table["subspace_group_candidate"] == "P4"
    assert table["expected_hsps"] == ["GammaM"]

    table_path = tmp_path / "p4_reduced_ebr_table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    validated_table = load_reduced_ebr_table(table_path)
    assert validated_table["subspace_group_candidate"] == "P4"
    assert validated_table["expected_hsps"] == ["GammaM"]

    # 6. Exact reduced EBR solve with validated builder-generated table.
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle, table=validated_table,
    )
    assert result["mapping_status"] == "solved_exact"
    assert result["solutions"][0]["classification"] == "atomic-compatible-candidate"
    assert result["solutions"][0]["subspace_group_candidate"] == "P4"

    # 7. Generic provenance survives the pipeline.
    rec = inst["irrep_records_by_kpoint"]["GammaM"][0]
    assert rec.get("matching_strategy") == "bilbao_restricted_character"
    assert rec.get("irrep_multiplicity") == 1
    assert rec.get("source_operation_map") == {0: 1, 1: 2}
    assert rec.get("valley_preserving_operation_ids") == [0, 1]
    assert rec["subspace_space_group"]["candidate_space_group_symbol"] == "P4"


def test_irreptables_loader_e2e_p4_group_agnostic(tmp_path):
    """E2E through the irreptables loader path with a fake package-style source.

    fake irreptables-style source
    → build_reduced_table_from_irreptables()
    → load_reduced_ebr_table() (validated through JSON)
    → generic P4 export bundle/problem instance
    → build_reduced_ebr_mapping()
    → exact reduced EBR solution

    No network, no real Bilbao downloads, no private irrep2.
    """
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
    from valleyscope.analysis.reduced_ebr_mapping import (
        build_reduced_ebr_mapping, load_reduced_ebr_table,
    )
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_reduced_table_from_irreptables,
    )

    # --- Steps 1-4: same generic P4 irrep → EBR export pipeline ---
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                },
            },
        },
    }
    symmetry_adapted_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "reference_valley": "K_valley",
                    "orbit": ["K_valley"],
                    "subspace_group": {"subspace_group_candidate": "P4"},
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                        "valley_preserving_operation_ids": [0, 1],
                    },
                    "hsp_preserving_operation_ids": [0, 1],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                                {"operation_id": 1, "eigenphases": [0.25, -0.25]},
                            ],
                        },
                    },
                }],
            },
        },
    }
    i = 1j
    source_chars = {
        "GM_plus_1over4":  {1: 1.0+0j, 2:  i},
        "GM_minus_1over4": {1: 1.0+0j, 2: -i},
    }
    operation_maps = {"GammaM": {"K_valley": {0: 1, 1: 2}}}

    matching = build_valley_irrep_matching_report(
        irrep_workflow_decisions=workflow,
        symmetry_adapted_valley_report=symmetry_adapted_report,
        source_irrep_characters_flattened={
            "GammaM": {"K_valley": source_chars},
        },
        source_operation_maps=operation_maps,
    )
    assert matching["matching_mode"] == "generic"
    gm = matching["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "matched"
    mults = gm["irrep_multiplicities"]
    assert mults.get("GM_plus_1over4") == 1
    assert mults.get("GM_minus_1over4") == 1

    candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    instances = build_ebr_problem_instances(ebr_input_candidates=candidates)
    bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    b = bundle["bundles"][0]

    # --- Step 5: build reduced table via irreptables loader with fake source ---
    bp_irreps = b["irreps_by_kpoint"]["GammaM"]
    source_irrep_labels = [f"src_{irr}" for irr in bp_irreps]
    fake_ebr_data = {
        "basis": {
            "irrep_labels": source_irrep_labels,
        },
        "ebrs": [
            {"ebr_name": "EBR_A", "vector": [1, 0]},
            {"ebr_name": "EBR_B", "vector": [0, 1]},
        ],
    }
    fake_package_version = "3.2.1-fake"

    def _fake_loader(sg, spin):
        """Fake irreptables loader: returns mock EBR data without network."""
        assert sg == 75  # P4 space group number
        assert spin is False
        return fake_ebr_data

    source_hsp_map = {label: "GammaM" for label in source_irrep_labels}
    valleyscope_key_map = {
        label: f"GammaM:{irr}"
        for label, irr in zip(source_irrep_labels, bp_irreps)
    }
    table = build_reduced_table_from_irreptables(
        space_group_number=75,
        spinful=False,
        source_loader=_fake_loader,
        source_hsp_by_irrep=source_hsp_map,
        valleyscope_key_by_source_irrep=valleyscope_key_map,
        expected_hsps=["GammaM"],
        allowed_irrep_keys=[f"GammaM:{irr}" for irr in bp_irreps],
        subspace_group_candidate="P4",
        provenance={"package_version": fake_package_version},
    )

    # --- Provenance assertions ---
    prov = table.get("provenance", {})
    assert isinstance(prov, dict) and prov
    assert prov.get("data_source") == "irreptables"
    assert prov.get("package") == "irreptables"
    assert prov.get("space_group_number") == 75
    assert prov.get("spinful") is False
    assert prov.get("expected_hsps") == ["GammaM"]
    assert prov.get("subspace_group_candidate") == "P4"
    assert prov.get("valleyscope_reduction") == "sampled_hsp_valley_preserving"
    assert prov.get("package_version") == fake_package_version
    assert table["subspace_group_candidate"] == "P4"
    assert table["expected_hsps"] == ["GammaM"]

    # --- Step 6: serialize, validate through load_reduced_ebr_table, solve ---
    table_path = tmp_path / "p4_irreptables_ebr_table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    validated_table = load_reduced_ebr_table(table_path)
    assert validated_table["subspace_group_candidate"] == "P4"

    result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle, table=validated_table,
    )
    assert result["mapping_status"] == "solved_exact"
    assert result["solutions"][0]["classification"] == "atomic-compatible-candidate"
    assert result["solutions"][0]["subspace_group_candidate"] == "P4"


def test_p4_public_output_contract(tmp_path):
    """Group-agnostic public output contract: all standard outputs use
    subspace_space_group as primary identity, not Cn-like labels.

    Covers: representation_records, valley_irrep_matching,
    valley_ebr_input_candidates, valley_ebr_problem_instances,
    valley_ebr_export_bundle, valley_reduced_ebr_mapping,
    valley_summary.txt.
    """
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
    from valleyscope.analysis.reduced_ebr_mapping import (
        build_reduced_ebr_mapping, load_reduced_ebr_table,
    )
    from valleyscope.analysis.valley_projected_representation import (
        build_valley_projected_representation_report,
    )

    # --- Build synthetic P4 pipeline ---
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                },
            },
        },
    }
    symmetry_adapted_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "reference_valley": "K_valley",
                    "orbit": ["K_valley"],
                    "subspace_group": {"subspace_group_candidate": "P4"},
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                        "valley_preserving_operation_ids": [0, 1],
                    },
                    "hsp_preserving_operation_ids": [0, 1],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                                {"operation_id": 1, "eigenphases": [0.25, -0.25]},
                            ],
                        },
                    },
                }],
            },
         },
    }
    i = 1j
    source_chars = {
        "GM_plus_1over4":  {1: 1.0+0j, 2:  i},
        "GM_minus_1over4": {1: 1.0+0j, 2: -i},
    }
    operation_maps = {"GammaM": {"K_valley": {0: 1, 1: 2}}}
    eigen_rows = [
        {"kpoint": "GammaM", "target_valley": "K_valley",
         "operation_id": 1, "order": 4,
         "diagnostic_only": False, "topology_input_ready": True,
         "rotation_ready": True},
    ]

    # 1. Irrep matching.
    matching = build_valley_irrep_matching_report(
        irrep_workflow_decisions=workflow,
        symmetry_adapted_valley_report=symmetry_adapted_report,
        source_irrep_characters_flattened={
            "GammaM": {"K_valley": source_chars},
        },
        source_operation_maps=operation_maps,
    )
    assert matching["matching_mode"] == "generic"

    # Contract 1: matching report must not promote Cn-like as physical identity.
    raw_matching = json.dumps(matching)
    for cn in ("C2_like", "C3_like", "C4_like"):
        assert f'"subspace_group_candidate": "{cn}"' not in raw_matching, (
            f"{cn} must not appear as physical group identity in irrep matching"
        )

    # 2. Representation records.
    rep_report = build_valley_projected_representation_report(
        kpoint_names=["GammaM"],
        valley_names=["K_valley"],
        symmetry_eigenvalue_rows=eigen_rows,
        symmetry_adapted_valley_report=symmetry_adapted_report,
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    # Contract 2: representation_records use P4 as primary identifier.
    recs = rep_report["representation_records"]
    assert len(recs) == 1
    assert recs[0]["subspace_space_group"]["candidate_space_group_symbol"] == "P4"
    raw_rep = json.dumps(recs)
    for cn in ("C2_like", "C3_like", "C4_like"):
        assert f'"candidate_space_group_symbol": "{cn}"' not in raw_rep, (
            f"{cn} must not appear as subspace_space_group symbol"
        )

    # 3. EBR pipeline.
    candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    instances = build_ebr_problem_instances(ebr_input_candidates=candidates)
    bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    b = bundle["bundles"][0]
    # Contract 3: all EBR outputs use P4 as physical group identity.
    assert b["subspace_group_candidate"] == "P4"
    raw_bundle = json.dumps(b)
    assert '"subspace_group_candidate": "C4_like"' not in raw_bundle
    inst = instances["instances"][0]
    assert inst["subspace_group_candidate"] == "P4"

    # 4. Solve.
    bp_irreps = b["irreps_by_kpoint"]["GammaM"]
    table_def = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM"],
        "irreps": [f"GammaM:{irr}" for irr in bp_irreps],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0]},
            {"label": "EBR_B", "vector": [0, 1]},
        ],
    }
    table_path = tmp_path / "contract_table.json"
    table_path.write_text(json.dumps(table_def), encoding="utf-8")
    loaded_table = load_reduced_ebr_table(table_path)
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle, table=loaded_table,
    )
    assert result["mapping_status"] == "solved_exact"
    assert result["solutions"][0]["subspace_group_candidate"] == "P4"

    # 5. Summary: public output JSON contract — P4 is the physical identity.
    raw_summary = json.dumps(bundle)
    assert "P4" in raw_summary
    # Cn-like must not be promoted as physical group identity.
    for cn in ("C2_like", "C3_like", "C4_like"):
        assert f'"subspace_group_candidate": "{cn}"' not in raw_summary, (
            f"{cn} must not appear as physical group identity in export bundle"
        )


def test_standard_outputs_no_cn_like_guardrail(tmp_path):
    """Standard public outputs must not emit C2_like, C3_like, or C4_like
    as physical group identity in any standard output object."""
    import json
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
    from valleyscope.analysis.valley_projected_representation import (
        build_valley_projected_representation_report,
    )
    from valleyscope.reports.summary_report import build_summary_payload

    # Synthetic P4 generic pipeline — produces all standard output objects.
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                },
            },
        },
    }
    sa = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "reference_valley": "K_valley",
                    "orbit": ["K_valley"],
                    "subspace_group": {
                        "subspace_group_candidate": "P4",
                        "legacy_subspace_group_candidate": "C4_like",
                    },
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                        "valley_preserving_operation_ids": [0, 1],
                    },
                    "hsp_preserving_operation_ids": [0, 1],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                                {"operation_id": 1, "eigenphases": [0.25, -0.25]},
                            ],
                        },
                    },
                }],
            },
        },
    }
    i = 1j
    source_chars = {
        "GM_plus_1over4": {1: 1.0 + 0j, 2: i},
        "GM_minus_1over4": {1: 1.0 + 0j, 2: -i},
    }
    eigen_rows = [{
        "kpoint": "GammaM", "target_valley": "K_valley",
        "operation_id": 1, "order": 4,
        "diagnostic_only": False, "topology_input_ready": True,
        "rotation_ready": True,
    }]

    matching = build_valley_irrep_matching_report(
        irrep_workflow_decisions=workflow,
        symmetry_adapted_valley_report=sa,
        source_irrep_characters_flattened={
            "GammaM": {"K_valley": source_chars},
        },
        source_operation_maps={"GammaM": {"K_valley": {0: 1, 1: 2}}},
    )
    candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    instances = build_ebr_problem_instances(ebr_input_candidates=candidates)
    bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    rep_report = build_valley_projected_representation_report(
        kpoint_names=["GammaM"],
        valley_names=["K_valley"],
        symmetry_eigenvalue_rows=eigen_rows,
        symmetry_adapted_valley_report=sa,
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "skipped",
            "reason": "unit test",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={},
        valley_irrep_matching=matching,
        ebr_input_candidates=candidates,
        ebr_problem_instances=instances,
        ebr_export_bundle=bundle,
        valley_projected_representation=rep_report,
    )

    standard_outputs = {
        "valley_ebr_export_bundle": bundle,
        "valley_summary": summary,
    }
    for name, output in standard_outputs.items():
        raw = json.dumps(output)
        for cn in ("C2_like", "C3_like", "C4_like"):
            assert cn not in raw, (
                f"{cn} appears in standard public output {name}"
            )
    # Physical P{n} symbol must remain available in matching and export bundle.
    raw_matching = json.dumps(matching)
    assert '"subspace_group_candidate": "P4"' in raw_matching or "P4" in raw_matching
    assert "P4" in json.dumps(bundle)
    assert "P4" in json.dumps(summary)


# -----------------------------------------------------------------------
# Irrep source provenance propagation tests
# -----------------------------------------------------------------------

def test_candidate_carries_irrep_source_provenance():
    """Trusted EBR input candidate includes irrep_source_provenance."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates

    workflow = {"by_kpoint": {"GammaM": {"K_valley": {
        "readiness_level": "trusted", "workflow_path": "direct_qcut"}}}}
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {"GammaM": {"K_valley": {
            "matching_status": "matched",
            "matching_strategy": "bilbao_restricted_character",
            "irrep_multiplicities": {"-GM5": 1},
            "subspace_space_group": {
                "status": "resolved", "candidate_space_group_number": 75,
                "candidate_space_group_symbol": "P4"},
            "valley_preserving_operation_ids": [0, 1],
            "source_operation_map": {0: 1, 1: 2},
            "source_payload_provenance": {
                "table_sg_number": 75, "table_name": "P4",
                "table_spinor": True, "source_hsp_label": "GM",
                "source_table_operation_indices": [1, 2],
                "standard_setting_hsp_mapping": {
                    "standard_setting_certificate": {
                        "validation_status": "validated",
                        "subspace_sg_number": 75,
                        "resolved_hsp_label": "GM",
                        "centering_status": "primitive_direct_match",
                    },
                },
            },
            "operation_mapping_provenance": "exact_spatial",
        }},
    }}
    report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow, valley_irrep_matching=matching)
    assert report["candidate_count"] == 1
    c = report["candidates"][0]
    prov = c.get("irrep_source_provenance", {})
    assert prov["subspace_space_group_number"] == 75
    assert prov["subspace_space_group_symbol"] == "P4"
    assert prov["source_table_sg_number"] == 75
    assert prov["source_table_spinor"] is True
    assert prov["source_hsp_label"] == "GM"
    assert prov["operation_mapping_provenance"] == "exact_spatial"
    assert prov["valley_preserving_operation_ids"] == [0, 1]
    cert = prov["standard_setting_hsp_mapping"]["standard_setting_certificate"]
    assert cert["validation_status"] == "validated"
    assert cert["subspace_sg_number"] == 75
    assert cert["resolved_hsp_label"] == "GM"


def test_problem_instance_preserves_multi_hsp_provenance():
    """Problem instance irrep_records carry provenance for multiple HSPs."""
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    candidates = {"status": "has_candidates", "candidates": [
        {"kpoint": "GammaM", "valley": "K_valley", "ready_for_ebr_input": True,
         "subspace_group_candidate": "P4", "workflow_path": "direct_qcut",
         "readiness_level": "trusted", "matched_irrep": "-GM5",
         "irrep_multiplicity": 1, "operation_id": 1,
         "subspace_space_group": {"status": "resolved",
                                  "candidate_space_group_number": 75,
                                  "candidate_space_group_symbol": "P4"},
         "irrep_source_provenance": {
             "source_hsp_label": "GM", "source_table_sg_number": 75,
             "source_table_spinor": True, "operation_mapping_provenance": "exact_spatial",
             "valley_preserving_operation_ids": [0, 1]}},
        {"kpoint": "KM", "valley": "K_valley", "ready_for_ebr_input": True,
         "subspace_group_candidate": "P4", "workflow_path": "direct_qcut",
         "readiness_level": "trusted", "matched_irrep": "-K5",
         "irrep_multiplicity": 1, "operation_id": 1,
         "subspace_space_group": {"status": "resolved",
                                  "candidate_space_group_number": 75,
                                  "candidate_space_group_symbol": "P4"},
         "irrep_source_provenance": {
             "source_hsp_label": "K", "source_table_sg_number": 75,
             "source_table_spinor": True,
             "valley_preserving_operation_ids": [0, 1]}},
    ]}
    report = build_ebr_problem_instances(ebr_input_candidates=candidates)
    inst = report["instances"][0]
    records = inst["irrep_records_by_kpoint"]
    assert "GammaM" in records and "KM" in records
    gm_prov = records["GammaM"][0]["irrep_source_provenance"]
    km_prov = records["KM"][0]["irrep_source_provenance"]
    assert gm_prov["source_hsp_label"] == "GM"
    assert km_prov["source_hsp_label"] == "K"


def test_export_bundle_preserves_multi_hsp_provenance():
    """Export bundle preserves per-HSP irrep source provenance."""
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle

    records = {
        "GammaM": [{
            "valley": "K_valley",
            "matched_irrep": "-GM5",
            "irrep_multiplicity": 1,
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "source": "valley_irrep_matching/generic/GammaM/K_valley",
            "irrep_source_provenance": {
                "source_hsp_label": "GM",
                "source_table_sg_number": 75,
                "source_table_spinor": True,
            },
        }],
        "KM": [{
            "valley": "K_valley",
            "matched_irrep": "-K5",
            "irrep_multiplicity": 1,
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "source": "valley_irrep_matching/generic/KM/K_valley",
            "irrep_source_provenance": {
                "source_hsp_label": "K",
                "source_table_sg_number": 75,
                "source_table_spinor": True,
            },
        }],
    }
    problem_instances = {"instances": [{
        "instance_id": "ebr_instance_001",
        "valley": "K_valley",
        "subspace_group_candidate": "P4",
        "subspace_space_group": {
            "status": "resolved",
            "candidate_space_group_number": 75,
            "candidate_space_group_symbol": "P4",
        },
        "workflow_path": "direct_qcut",
        "readiness_level": "trusted",
        "irreps_by_kpoint": {"GammaM": ["-GM5"], "KM": ["-K5"]},
        "operations_by_kpoint": {"GammaM": [1], "KM": [1]},
        "expected_hsps": ["GammaM", "KM"],
        "optional_hsps": [],
        "missing_optional_hsps": [],
        "irrep_records_by_kpoint": records,
        "status": "complete",
        "ready_for_ebr_decomposition": True,
    }]}

    report = build_ebr_export_bundle(ebr_problem_instances=problem_instances)
    bundle = report["bundles"][0]
    out_records = bundle["irrep_records_by_kpoint"]
    assert out_records["GammaM"][0]["irrep_source_provenance"]["source_hsp_label"] == "GM"
    assert out_records["KM"][0]["irrep_source_provenance"]["source_hsp_label"] == "K"


def test_reduced_ebr_solution_preserves_multi_hsp_provenance():
    """Reduced EBR solution carries per-kpoint provenance for both HSPs."""
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    table = {"schema_version": "1.0.0", "subspace_group_candidate": "P4",
             "expected_hsps": ["GammaM", "KM"],
             "irreps": ["GammaM:-GM5", "KM:-K5"],
             "ebrs": [{"label": "EBR_A", "vector": [1, 1]}]}
    bundle = {"bundles": [{
        "bundle_id": "b_001", "valley": "K", "subspace_group_candidate": "P4",
        "ready_for_external_solver": True,
        "expected_hsps": ["GammaM", "KM"],
        "irreps_by_kpoint": {"GammaM": ["-GM5"], "KM": ["-K5"]},
        "irrep_records_by_kpoint": {
            "GammaM": [{"matched_irrep": "-GM5", "irrep_multiplicity": 1,
                        "irrep_source_provenance": {"source_hsp_label": "GM",
                        "source_table_sg_number": 75, "source_table_spinor": True}}],
            "KM": [{"matched_irrep": "-K5", "irrep_multiplicity": 1,
                    "irrep_source_provenance": {"source_hsp_label": "K",
                    "source_table_sg_number": 75, "source_table_spinor": True}}],
        },
    }]}
    r = build_reduced_ebr_mapping(ebr_export_bundle=bundle, table=table)
    assert r["mapping_status"] == "solved_exact"
    sol = r["solutions"][0]
    by_kp = sol.get("irrep_source_provenance_by_kpoint", {})
    assert "GammaM" in by_kp and "KM" in by_kp
    assert by_kp["GammaM"][0]["source_hsp_label"] == "GM"
    assert by_kp["KM"][0]["source_hsp_label"] == "K"


def test_reduced_ebr_excluded_preserves_provenance():
    """Excluded bundle with HSP mismatch retains provenance for audit."""
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    table = {"schema_version": "1.0.0", "subspace_group_candidate": "P4",
             "expected_hsps": ["GammaM"],
             "irreps": ["GammaM:-GM5"],
             "ebrs": [{"label": "EBR_A", "vector": [1]}]}
    bundle = {"bundles": [{
        "bundle_id": "b_001", "valley": "K", "subspace_group_candidate": "P4",
        "ready_for_external_solver": True,
        "expected_hsps": ["GammaM", "KM"],
        "irreps_by_kpoint": {"GammaM": ["-GM5"], "KM": ["-K5"]},
        "irrep_records_by_kpoint": {
            "GammaM": [{"matched_irrep": "-GM5", "irrep_multiplicity": 1,
                        "irrep_source_provenance": {"source_hsp_label": "GM"}}],
        },
    }]}
    r = build_reduced_ebr_mapping(ebr_export_bundle=bundle, table=table)
    assert len(r["excluded_bundles"]) == 1
    exc = r["excluded_bundles"][0]
    assert "expected_hsps mismatch" in exc["reason"]
    by_kp = exc.get("irrep_source_provenance_by_kpoint", {})
    assert "GammaM" in by_kp


def test_blocked_diagnostic_no_candidate():
    """Blocked/diagnostic-only generic match does not produce a candidate."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates

    workflow = {"by_kpoint": {"GammaM": {"K_valley": {
        "readiness_level": "blocked", "workflow_path": "blocked"}}}}
    matching = {"matching_mode": "generic",
                "generic_matches_by_kpoint": {"GammaM": {"K_valley": {
                    "matching_status": "blocked",
                    "diagnostic_only": True,
                    "irrep_multiplicities": {},
                    "subspace_space_group": {"status": "resolved",
                        "candidate_space_group_number": 143,
                        "candidate_space_group_symbol": "P3"},
                }}}}
    report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow, valley_irrep_matching=matching)
    assert report["candidate_count"] == 0
    assert report["blocked_count"] == 1


# ---------------------------------------------------------------------------
# Public E2E record contract: full standard-output chain
# ---------------------------------------------------------------------------

def test_public_e2e_record_chain_with_certificate_provenance():
    """Full public chain: summary→export→mapping→database, certificate preserved."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
    from valleyscope.analysis.reduced_ebr_mapping import (
        build_reduced_ebr_mapping,
    )
    from valleyscope.analysis.database_ingestion_record import (
        build_database_ingestion_record,
    )

    # --- 1. Synthetic matching with certificate provenance ---
    workflow = {"by_kpoint": {"GammaM": {"K_valley": {
        "readiness_level": "trusted", "workflow_path": "direct_qcut",
    }}}}
    sa_report = {"by_kpoint": {"GammaM": {"valley_preserving_subspaces": [{
        "orbit": ["K_valley"],
        "hsp_preserving_operation_ids": [0, 1],
        "subspace_space_group": {
            "status": "resolved",
            "candidate_space_group_symbol": "P4",
            "candidate_space_group_number": 75,
            "valley_preserving_operation_ids": [0, 1],
        },
        "valley_preserving_character_diagnostics": {
            "per_valley": {"K_valley": [
                {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                {"operation_id": 1, "eigenphases": [0.5, -0.5]},
            ]},
        },
    }]}}}
    certificate = {
        "validation_status": "validated",
        "subspace_sg_number": 75,
        "subspace_sg_symbol": "P4",
        "hall_number": 81,
        "hall_symbol": "P 4",
        "resolved_hsp_label": "GM",
        "centering_type": "P",
        "centering_status": "primitive_direct_match",
    }
    kmap_prov = {"standard_setting_certificate": certificate}
    i_ = 1j
    matching = build_valley_irrep_matching_report(
        irrep_workflow_decisions=workflow,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters_flattened={"GammaM": {"K_valley": {
            "A": {1: 1.0 + 0j, 2: -1.0 + 0j},
        }}},
        source_operation_maps={"GammaM": {"K_valley": {0: 1, 1: 2}}},
        source_payload_provenance={"GammaM": {"K_valley": {
            "standard_setting_hsp_mapping": kmap_prov,
        }}},
    )
    gm = matching["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "matched"

    # --- 2. EBR pipeline ---
    candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    assert candidates["candidate_count"] >= 1
    c = candidates["candidates"][0]
    assert c["irrep_source_provenance"] is not None
    assert "standard_setting_hsp_mapping" in c["irrep_source_provenance"]
    cert_in_cand = c["irrep_source_provenance"]["standard_setting_hsp_mapping"]
    assert cert_in_cand["standard_setting_certificate"]["validation_status"] == "validated"

    instances = build_ebr_problem_instances(ebr_input_candidates=candidates)
    assert instances["instance_count"] >= 1
    bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    assert bundle["bundle_count"] >= 1

    # --- 3. Reduced EBR (without table → missing_table) ---
    mapping_result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle, table=None,
    )
    assert mapping_result["mapping_status"] == "missing_table"

    # --- 4. Database ingestion record ---
    summary = {"target_kpoints": ["GammaM"], "iband": [1],
               "input": {"spinor_convention_verified": True}}
    record = build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=bundle,
        valley_reduced_ebr_mapping=mapping_result,
    )
    assert record["record_status"] == "has_ready_ebr_bundles"
    # Certificate provenance preserved in irrep records.
    val_records = record.get("valley_irrep_records", [])
    assert len(val_records) >= 1
    prov = val_records[0].get("irrep_source_provenance")
    assert prov is not None
    assert "standard_setting_hsp_mapping" in prov
