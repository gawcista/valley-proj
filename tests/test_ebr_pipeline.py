import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.workflows.analyze_hsp import analyze_hsp

from tests.helpers_io_workflow import write_fixture, write_config
from tests.helpers_io_workflow import _E2E_SAMPLE_TABLE, e2e_write_table
from tests.reduced_ebr_promo_helpers import (
    attach_real_certificate, add_real_certificate_to_candidates,
    complete_table_provenance, real_primitive_certificate_identity)

from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
from valleyscope.analysis.database_ingestion_record import (
    build_database_ingestion_record,
)
from valleyscope.analysis.reduced_ebr_mapping import (
    build_reduced_ebr_mapping,
    load_reduced_ebr_table,
    promote_bundle_for_solve,
)

from tests.helpers_io_workflow import write_fixture, write_config


def _complete_coverage_for_candidates(report):
    by_valley = {}
    for candidate in report.get("candidates", []):
        valley = candidate["valley"]
        kpoint = candidate["kpoint"]
        by_valley.setdefault(valley, {})[kpoint] = kpoint
    return {
        "by_valley": {
            valley: {
                "required_source_hsp_labels": list(mapping),
                "covered_source_hsp_labels": list(mapping),
                "missing_source_hsp_labels": [],
                "trusted_matched_source_hsp_labels": list(mapping),
                "trusted_missing_source_hsp_labels": [],
                "source_hsp_to_sampled_kpoint": mapping,
                "complete": True,
                "ready_for_ebr_promotion": True,
                "source_basis_provenance": {"data_source": "irreptables"},
            }
            for valley, mapping in by_valley.items()
        }
    }

# Valley irrep -> EBR pipeline contract tests
# -----------------------------------------------------------------------





def test_generic_projected_subspace_k_is_excluded_without_becoming_blocker():
    workflow = {
        "by_kpoint": {
            "KM": {
                "M1_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                },
            },
        },
    }
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "KM": {
                "M1_valley": {
                    "matching_status": "not_applicable",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {},
                    "diagnostic_only": False,
                    "reason": "generic_projected_subspace_k",
                    "source_hsp_membership": False,
                    "projected_hsp_classification": {
                        "classification": "generic",
                        "validation_status": "validated",
                    },
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 5,
                        "candidate_space_group_symbol": "C2",
                    },
                },
            },
        },
    }

    report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )

    assert report["candidate_count"] == 0
    assert report["blocked_count"] == 0
    assert report["non_source_count"] == 1
    assert report["non_source_rows"][0]["kpoint"] == "KM"
    assert report["non_source_rows"][0]["local_representation_ready"] is True






def test_incomplete_per_valley_source_hsp_coverage_blocks_ebr_promotion():
    candidates = {
        "status": "has_candidates",
        "candidates": [{
            "kpoint": "GammaM",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "matched_irrep": "-GM4",
            "irrep_multiplicity": 1,
            "irrep_source_provenance": {"source_hsp_label": "GM"},
            "ready_for_ebr_input": True,
        }],
    }
    coverage = {
        "by_valley": {
            "K_valley": {
                "required_source_hsp_labels": ["GM", "K"],
                "covered_source_hsp_labels": ["GM"],
                "missing_source_hsp_labels": ["K"],
                "trusted_matched_source_hsp_labels": ["GM"],
                "trusted_missing_source_hsp_labels": ["K"],
                "source_hsp_to_sampled_kpoint": {"GM": "GammaM"},
                "complete": False,
                "trusted_matching_complete": False,
                "ready_for_ebr_promotion": False,
                "source_basis_provenance": {"source": "irreptables"},
            },
        },
    }

    report = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        projected_hsp_coverage=coverage,
    )

    instance = report["instances"][0]
    assert instance["status"] == "incomplete_canonical_hsp_vector"
    assert instance["canonical_hsp_vector_complete"] is False
    assert instance["required_source_hsp_labels"] == ["GM", "K"]
    assert instance["covered_source_hsp_labels"] == ["GM"]
    assert instance["missing_source_hsp_labels"] == ["K"]
    assert "missing trusted source HSPs: ['K']" in instance["blocked_by"]








def test_generic_p4_table_authoritative_bundle_maps_and_rejects_mismatch():
    """P4 synthetic instance exports without hard-coded Cn HSP policy."""
    problem_instances = build_ebr_problem_instances(
        ebr_input_candidates=add_real_certificate_to_candidates({
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
                    "source": "fixture/K_valley/GammaM",
                    "irrep_source_provenance": {
                        "source_hsp_label": "GammaM",
                    },
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
                    "source": "fixture/K_valley/XM",
                    "irrep_source_provenance": {
                        "source_hsp_label": "XM",
                    },
                    "ready_for_ebr_input": True,
                },
            ],
        }, 75, "P4", spinor=True),
        projected_hsp_coverage={
            "by_valley": {
                "K_valley": {
                    "required_source_hsp_labels": ["GammaM", "XM"],
                    "covered_source_hsp_labels": ["GammaM", "XM"],
                    "missing_source_hsp_labels": [],
                    "trusted_matched_source_hsp_labels": ["GammaM", "XM"],
                    "trusted_missing_source_hsp_labels": [],
                    "source_hsp_to_sampled_kpoint": {
                        "GammaM": "GammaM", "XM": "XM"
                    },
                    "complete": True,
                    "ready_for_ebr_promotion": True,
                    "source_basis_provenance": {
                        "data_source": "irreptables"
                    },
                }
            }
        },
    )
    inst = problem_instances["instances"][0]
    assert inst["subspace_group_candidate"] == "P4"
    assert inst["expected_hsps"] == ["GammaM", "XM"]
    assert inst["expected_hsp_policy_source"] == "certified_source_hsp_basis"
    assert inst["canonical_hsp_vector_complete"] is True

    export_bundle = build_ebr_export_bundle(
        ebr_problem_instances=problem_instances
    )
    assert export_bundle["status"] == "ready_for_reduced_table_validation"
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
        "provenance": {"space_group_number": 75, "spinful": True},
    }
    complete_table_provenance(matching_table, 75, spinful=True)
    solved = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=matching_table
    )
    # Sampled-basis bundles do not reach the solver.
    assert solved["status"] == "solved_exact"

    missing_construction = dict(bundle)
    missing_construction.pop("unitary_vector_construction")
    promotion = promote_bundle_for_solve(
        bundle=missing_construction,
        table=matching_table,
    )
    assert promotion["promoted"] is False
    assert {
        row["code"] for row in promotion["blocker_reasons"]
    } == {"unitary_construction_provenance_invalid"}

    missing_physical_object = dict(bundle)
    missing_physical_object.pop("physical_object_kind")
    promotion = promote_bundle_for_solve(
        bundle=missing_physical_object,
        table=matching_table,
    )
    assert promotion["promoted"] is False
    assert "problem_physical_object_kind_mismatch" in {
        row["code"] for row in promotion["blocker_reasons"]
    }
    assert solved["solutions"][0]["irrep_vector"] == [1, 1]

    direct_record = bundle["irrep_records_by_kpoint"]["GammaM"][0]
    assert direct_record["sampled_kpoint"] == "GammaM"
    assert direct_record["source_hsp_label"] == "GammaM"
    assert direct_record["certificate_identity"] == bundle[
        "certificate_identity"
    ]
    assert bundle["spinor"] is True

    direct_mutations = {
        "irrep": lambda record: record.__setitem__(
            "matched_irrep", "forged_irrep"
        ),
        "multiplicity": lambda record: record.__setitem__(
            "irrep_multiplicity", 2
        ),
        "source_hsp": lambda record: record.__setitem__(
            "source_hsp_label", "XM"
        ),
        "valley": lambda record: record.__setitem__(
            "valley", "forged_valley"
        ),
        "sample": lambda record: record.__setitem__(
            "sampled_kpoint", "XM"
        ),
        "spin": lambda record: record[
            "irrep_source_provenance"
        ].__setitem__("source_table_spinor", False),
        "readiness": lambda record: record.__setitem__(
            "readiness_level", "blocked"
        ),
        "workflow": lambda record: record.__setitem__(
            "workflow_path", "time_reversal_valley_orbit"
        ),
        "workflow_identity_mismatch": lambda record: record.__setitem__(
            "workflow_path", "symmetry_adapted"
        ),
        "source_identity_mismatch": lambda record: record.__setitem__(
            "source", "forged/source"
        ),
    }
    for mutation, mutate in direct_mutations.items():
        forged = deepcopy(bundle)
        mutate(forged["irrep_records_by_kpoint"]["GammaM"][0])
        rejected_promotion = promote_bundle_for_solve(
            bundle=forged,
            table=matching_table,
        )
        assert not rejected_promotion["promoted"], mutation
        assert "unitary_construction_provenance_invalid" in {
            row["code"] for row in rejected_promotion["blocker_reasons"]
        }

        ingestion = build_database_ingestion_record(
            valley_summary={},
            valley_ebr_export_bundle={
                "status": "ready_for_reduced_table_validation",
                "bundles": [forged],
                "excluded_instances": [],
            },
        )
        assert any(
            "invalid direct unitary construction provenance" in error
            for error in ingestion["validation_errors"]
        ), mutation

    mismatched_table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM", "MM"],
        "irreps": [
            "GammaM:P4_spinor_phase_+1/4",
            "MM:P4_spinor_phase_-1/4",
        ],
        "ebrs": [{"label": "EBR_P4_bad", "vector": [1, 1]}],
        "provenance": {"space_group_number": 75, "spinful": True},
    }
    complete_table_provenance(mismatched_table, 75, spinful=True)
    rejected = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=mismatched_table
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
    certificate_identity = real_primitive_certificate_identity(
        143, "P3", spinor=True
    )

    def direct_record(sampled_kpoint, source_hsp, matched_irrep):
        identity = {
            "source": f"fixture/K_valley/{source_hsp}",
            "workflow_path": "direct_qcut",
            "valley": "K_valley",
            "source_hsp_label": source_hsp,
            "sampled_kpoint": sampled_kpoint,
            "irrep": matched_irrep,
            "multiplicity": 1,
        }
        provenance = {
            "source": identity["source"],
            "workflow_path": "direct_qcut",
            "irrep_source_provenance": {
                "source_hsp_label": source_hsp,
                "source_table_spinor": True,
            },
        }
        return {
            "valley": "K_valley",
            "sampled_kpoint": sampled_kpoint,
            "source_hsp_label": source_hsp,
            "matched_irrep": matched_irrep,
            "irrep_multiplicity": 1,
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "source": identity["source"],
            "certificate_identity": certificate_identity,
            "irrep_source_provenance": provenance[
                "irrep_source_provenance"
            ],
            "source_candidate_identity": identity,
            "source_candidate_provenance": provenance,
        }

    export_bundle = build_ebr_export_bundle(
        ebr_problem_instances={
            "instances": [{
                "instance_id": "ebr_instance_001",
                "problem_kind": "unitary_valley_reduced_ebr",
                "physical_object_kind": "unitary_valley_projected_subspace",
                "valley": "K_valley",
                "valley_orbit": [],
                "subspace_group_candidate": "P3",
                "subspace_sg_number": 143,
                "spinor": True,
                "workflow_path": "direct_qcut",
                "unitary_vector_construction": {
                    "kind": "direct_observed_unitary_rows",
                    "source": "trusted_ebr_input_candidates",
                },
                "readiness_level": "trusted",
                "irreps_by_kpoint": {
                    "GammaM": ["C3_spinor_phase_+1/2"],
                    "KM": ["C3_spinor_phase_+1/6"],
                },
                "operations_by_kpoint": {"GammaM": [1], "KM": [1]},
                "expected_hsps": ["GammaM", "KM"],
                "optional_hsps": ["MM"],
                "missing_optional_hsps": ["MM"],
                "canonical_hsp_vector_complete": True,
                "canonical_hsp_vector_ready": True,
                "status": "canonical_hsp_vector_ready",
                "certificate_identity": certificate_identity,
                "irrep_records_by_kpoint": {
                    "GammaM": [direct_record(
                        "GammaM", "GM", "C3_spinor_phase_+1/2"
                    )],
                    "KM": [direct_record(
                        "KM", "K", "C3_spinor_phase_+1/6"
                    )],
                },
                "required_source_hsp_labels": ["GM", "K"],
                "source_hsp_to_sampled_kpoint": {
                    "GM": "GammaM", "K": "KM",
                },
            }],
        }
    )
    _loaded_414 = load_reduced_ebr_table(table_path)
    complete_table_provenance(_loaded_414, 143, spinful=True)
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=_loaded_414
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
    report = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        projected_hsp_coverage=_complete_coverage_for_candidates(candidates),
    )
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
                "canonical_hsp_vector_complete": True,
                "canonical_hsp_vector_ready": True,
                "status": "canonical_hsp_vector_ready",
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

    instances_report = build_ebr_problem_instances(
        ebr_input_candidates=candidates_report,
        projected_hsp_coverage=_complete_coverage_for_candidates(
            candidates_report
        ),
    )
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
            "ready_for_reduced_table_validation": True,
            "expected_hsps": ["GammaM", "KM"],
            "irreps_by_kpoint": {
                "GammaM": ["C3_spinor_phase_+1/2"],
                "KM": ["C3_spinor_phase_+1/6"],
            },
            "irrep_records_by_kpoint": records,
        }],
    }
    complete_table_provenance(table, 143, spinful=True)
    attach_real_certificate(bundle, table)
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
    candidates = add_real_certificate_to_candidates(candidates, 75, "P4", spinor=True)
    instances = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        projected_hsp_coverage=_complete_coverage_for_candidates(candidates),
    )
    assert instances["instance_count"] == 1
    inst = instances["instances"][0]
    assert inst["canonical_hsp_vector_complete"] is True
    assert inst["expected_hsps"] == ["GammaM"]
    assert inst["expected_hsp_policy_source"] == "certified_source_hsp_basis"
    assert inst["subspace_group_candidate"] == "P4"

    # 4. Export bundle.
    bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    assert bundle["bundle_count"] == 1
    b = bundle["bundles"][0]
    # Exported for reviewed-table validation, not yet solver-ready.
    assert b["ready_for_reduced_table_validation"] is True
    assert "ready_for_external_solver" not in b
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
    complete_table_provenance(validated_table, 75, spinful=True)
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle, table=validated_table
    )
    assert result["status"] == "solved_exact"
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
    candidates = add_real_certificate_to_candidates(candidates, 75, "P4", spinor=True)
    instances = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        projected_hsp_coverage=_complete_coverage_for_candidates(candidates),
    )
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

    complete_table_provenance(validated_table, 75, spinful=True)
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle, table=validated_table
    )
    assert result["status"] == "solved_exact"
    assert result["solutions"][0]["classification"] == "atomic-compatible-candidate"
    assert result["solutions"][0]["subspace_group_candidate"] == "P4"


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
    candidates = add_real_certificate_to_candidates(candidates, 75, "P4", spinor=True)
    instances = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        projected_hsp_coverage=_complete_coverage_for_candidates(candidates),
    )
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


def test_centered_validated_certificate_reaches_ebr_candidate_provenance():
    """Centered standard-setting certificate is preserved in EBR candidates."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates

    centered_kmap = {
        "standard_setting_certificate": {
            "validation_status": "validated",
            "subspace_sg_number": 5,
            "subspace_sg_symbol": "C2",
            "hall_number": 9,
            "hall_symbol": "C 2y",
            "standard_setting_source": "explicit_transform",
            "primitive_conventional_relation": "explicit_transform",
            "resolved_hsp_label": "GM",
            "centering_type": "C",
            "centering_vectors": [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
            "translation_validation_status": "passed",
            "matched_affine_operations": 1,
            "total_parent_operations": 1,
        },
        "transform_candidate": {
            "validation_status": "validated",
            "operation_mapping_status": "operation_basis_verification_passed",
            "affine_validation_status": "passed",
            "centering_type": "C",
            "centering_vectors": [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
        },
    }
    workflow = {"by_kpoint": {"GammaM": {"K_valley": {
        "readiness_level": "trusted",
        "workflow_path": "direct_qcut",
    }}}}
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {"GammaM": {"K_valley": {
            "matching_status": "matched",
            "matching_strategy": "bilbao_restricted_character",
            "irrep_multiplicities": {"GM1": 1},
            "subspace_space_group": {
                "status": "resolved",
                "candidate_space_group_number": 5,
                "candidate_space_group_symbol": "C2",
            },
            "valley_preserving_operation_ids": [0],
            "source_payload_provenance": {
                "table_sg_number": 5,
                "table_name": "C2",
                "table_spinor": False,
                "source_hsp_label": "GM",
                "source_table_operation_indices": [1],
                "standard_setting_hsp_mapping": centered_kmap,
            },
            "operation_mapping_provenance": "exact_spatial",
        }}},
    }

    report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )

    assert report["candidate_count"] == 1
    provenance = report["candidates"][0]["irrep_source_provenance"]
    assert provenance["subspace_space_group_number"] == 5
    assert provenance["source_hsp_label"] == "GM"
    kmap = provenance["standard_setting_hsp_mapping"]
    cert = kmap["standard_setting_certificate"]
    assert cert["validation_status"] == "validated"
    assert cert["centering_type"] == "C"
    assert cert["centering_vectors"] == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
    assert cert["translation_validation_status"] == "passed"
    tc = kmap["transform_candidate"]
    assert tc["validation_status"] == "validated"
    assert tc["affine_validation_status"] == "passed"


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
    report = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        projected_hsp_coverage=_complete_coverage_for_candidates(candidates),
    )
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
        "status": "canonical_hsp_vector_ready",
        "canonical_hsp_vector_complete": True,
        "canonical_hsp_vector_ready": True,
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
            "ready_for_reduced_table_validation": True,
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
    complete_table_provenance(table, 75, spinful=True)
    attach_real_certificate(bundle, table)
    r = build_reduced_ebr_mapping(ebr_export_bundle=bundle, table=table)
    assert r["status"] == "solved_exact"
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
            "ready_for_reduced_table_validation": True,
        "expected_hsps": ["GammaM", "KM"],
        "irreps_by_kpoint": {"GammaM": ["-GM5"], "KM": ["-K5"]},
        "irrep_records_by_kpoint": {
            "GammaM": [{"matched_irrep": "-GM5", "irrep_multiplicity": 1,
                        "irrep_source_provenance": {"source_hsp_label": "GM"}}],
        },
    }]}
    attach_real_certificate(bundle, table)
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

    candidates = add_real_certificate_to_candidates(candidates, 75, "P4", spinor=True)
    instances = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        projected_hsp_coverage=_complete_coverage_for_candidates(candidates),
    )
    assert instances["instance_count"] >= 1
    bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    assert bundle["bundle_count"] >= 1

    # --- 3. Reduced EBR with reviewed reduced table ---
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:A"],
        "ebrs": [{"label": "EBR_P4_A", "vector": [2]}],
    }
    complete_table_provenance(table, 75, spinful=True)
    mapping_result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle, table=table
    )
    assert mapping_result["status"] == "solved_exact"
    solution = mapping_result["solutions"][0]
    assert solution["classification"] == "atomic-compatible-candidate"
    assert solution["ebr_decomposition"] == [
        {"label": "EBR_P4_A", "coefficient": 1}
    ]
    solution_prov = solution["irrep_source_provenance_by_kpoint"]["GammaM"][0]
    solution_cert = (
        solution_prov["standard_setting_hsp_mapping"]
        ["standard_setting_certificate"]
    )
    assert solution_cert["validation_status"] == "validated"
    assert solution_cert["subspace_sg_number"] == 75

    # --- 4. Database ingestion record ---
    summary = {"target_kpoints": ["GammaM"], "iband": [1],
               "input": {"spinor_convention_verified": True}}
    record = build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=bundle,
        valley_reduced_ebr_mapping=mapping_result,
    )
    assert record["record_status"] == "has_final_reduced_ebr_results"
    # Certificate provenance preserved in irrep records.
    val_records = record.get("valley_irrep_records", [])
    assert len(val_records) >= 1
    prov = val_records[0].get("irrep_source_provenance")
    assert prov is not None
    assert "standard_setting_hsp_mapping" in prov
    reduced_records = record.get("reduced_ebr_records", [])
    assert len(reduced_records) == 1
    reduced_record = reduced_records[0]
    assert reduced_record["classification"] == "atomic-compatible-candidate"
    reduced_cert = (
        reduced_record["irrep_source_provenance_by_kpoint"]
        ["GammaM"][0]
        ["standard_setting_hsp_mapping"]
        ["standard_setting_certificate"]
    )
    assert reduced_cert["resolved_hsp_label"] == "GM"
