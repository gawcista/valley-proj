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
    # Matching: K_valley matched, Kp_valley diagnostic_only.
    matching = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "1": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [0.5], "diagnostic_only": False},
                },
                "Kp_valley": {
                    "1": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_-1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [-0.5], "diagnostic_only": True},
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


def test_ebr_problem_instances_missing_hsp_blocked():
    """Missing required HSPs block instance readiness."""
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    # C3_like requires GammaM+KM. Only GammaM has data -> blocked.
    candidates = {
        "status": "has_candidates",
        "candidates": [{
            "kpoint": "GammaM", "valley": "K_valley",
            "subspace_group_candidate": "C3_like",
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
    assert inst["ready_for_ebr_decomposition"] is False
    assert "missing required HSPs" in str(inst["blocked_by"])


def test_ebr_problem_instances_complete_hsp_is_ready():
    """All required HSPs present -> ready_for_ebr_decomposition=true."""
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    candidates = {
        "status": "has_candidates",
        "candidates": [
            {"kpoint": "GammaM", "valley": "K_valley",
             "subspace_group_candidate": "C3_like",
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "matched_irrep": "C3_spinor_phase_+1/2", "operation_id": 1,
             "ready_for_ebr_input": True},
            {"kpoint": "KM", "valley": "K_valley",
             "subspace_group_candidate": "C3_like",
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
            "subspace_group_candidate": "C3_like",
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
    assert bundle["subspace_group_candidate"] == "C3_like"
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
            "subspace_group_candidate": "C3_like",
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
        "subspace_group_candidate": "C3_like",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:C3_spinor_phase_+1/2", "KM:C3_spinor_phase_+1/6"],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0]}, {"label": "EBR_B", "vector": [0, 1]}],
    }
    # Bundle only has GammaM, missing KM.
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM"],
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=bundle, table=table)
    assert len(r["solutions"]) == 0
    assert len(r["excluded_bundles"]) == 1
    assert "expected_hsps mismatch" in r["excluded_bundles"][0]["reason"]


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
        "subspace_group_candidate": "C3_like",
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
                "subspace_group_candidate": "C3_like",
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
             "subspace_group_candidate": "C3_like",
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "matched_irrep": "C3_spinor_phase_+1/2", "operation_id": 1,
             "operation_order": 3, "eigenphases": [0.5],
             "character": {"real": -1.0, "imag": 0.0},
             "source": "valley_irrep_matching/GammaM/K_valley",
             "ready_for_ebr_input": True},
            {"kpoint": "KM", "valley": "K_valley",
             "subspace_group_candidate": "C3_like",
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
            "subspace_group_candidate": "C3_like",
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

    # Mix of trusted and diagnostic_only rows
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            },
        },
    }
    matching = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "1": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [0.5], "diagnostic_only": False},
                    "2": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [-0.5], "diagnostic_only": True},
                },
            },
        },
    }
    candidates_report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow, valley_irrep_matching=matching)
    # Only 1 trusted candidate (op=1), op=2 is diagnostic_only -> blocked.
    assert candidates_report["candidate_count"] == 1

    instances_report = build_ebr_problem_instances(ebr_input_candidates=candidates_report)
    inst = instances_report["instances"][0]
    records = inst["irrep_records_by_kpoint"]
    gamma_recs = records.get("GammaM", [])
    # Only the trusted record appears.
    assert len(gamma_recs) == 1
    assert gamma_recs[0]["operation_id"] == "1"


def test_reduced_ebr_mapping_ignores_irrep_records():
    """reduced_ebr_mapping must remain compatible and ignore irrep_records_by_kpoint."""
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C3_like",
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
            "subspace_group_candidate": "C3_like",
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
