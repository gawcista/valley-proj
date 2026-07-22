import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.workflows.analyze_hsp import analyze_hsp

from valleyscope.analysis.database_ingestion_record import (
    build_database_ingestion_record,
    load_database_ingestion_record_from_directory,
)

from tests.helpers_io_workflow import write_fixture, write_config

# Database ingestion record tests
# -----------------------------------------------------------------------

def test_ingestion_record_requires_summary():
    """Missing valley_summary.json produces invalid record."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    record = build_database_ingestion_record()
    assert record["record_status"] == "invalid_missing_summary"
    assert len(record["validation_errors"]) > 0


def test_ingestion_uses_stage_owned_counts_and_final_result_status():
    summary = {"target_kpoints": ["GammaM"], "iband": [1], "input": {}}
    export = {
        "status": "partial_export",
        "bundles": [{
            "bundle_id": "b_candidate",
            "ready_for_reduced_table_validation": True,
            "irrep_records_by_kpoint": {
                "GammaM": [{"valley": "K", "matched_irrep": "A"}],
            },
        }],
        "excluded_instances": [{
            "source_instance_id": "i_blocked",
            "status": "canonical_hsp_vector_complete_but_untrusted",
            "canonical_hsp_vector_complete": True,
            "canonical_hsp_vector_ready": False,
            "exclusion_reasons": ["source_hsp_coverage_not_ready"],
        }],
    }
    mapping = {
        "status": "partial",
        "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_candidate",
            "status": "no_exact_solution",
            "classification": "in_integer_span_no_nonnegative_witness",
        }],
        "excluded_bundles": [{
            "bundle_id": "b_other",
            "reason": "validation blocked",
            "blocker_reasons": [{"code": "certificate_unresolved"}],
        }],
    }

    record = build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )

    assert record["record_status"] == "has_final_reduced_ebr_results"
    assert record["reduced_table_validation_candidate_bundle_count"] == 1
    assert record["final_reduced_ebr_result_count"] == 1
    assert record["final_mapping_excluded_bundle_count"] == 1
    assert record["input_excluded_instance_count"] == 1
    assert len(record["valley_irrep_records"]) == 1
    assert record["reduced_ebr_records"][0]["status"] == "no_exact_solution"
    assert record["input_excluded_ebr_records"][0][
        "canonical_hsp_vector_complete"
    ] is True
    assert record["input_excluded_ebr_records"][0][
        "canonical_hsp_vector_ready"
    ] is False
    assert record["final_mapping_excluded_records"][0]["bundle_id"] == (
        "b_other"
    )
    for removed in (
        "ready_bundle_count",
        "validation_candidate_count",
        "decomposition_ready_count",
        "excluded_bundle_count",
        "excluded_ebr_records",
    ):
        assert removed not in record


def test_ingestion_candidate_without_mapping_has_candidate_status():
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": [], "iband": [], "input": {}},
        valley_ebr_export_bundle={
            "bundles": [{
                "bundle_id": "b",
                "ready_for_reduced_table_validation": True,
                "irrep_records_by_kpoint": {},
            }],
        },
    )

    assert record["record_status"] == (
        "has_reduced_table_validation_candidates"
    )
    assert record["reduced_table_validation_candidate_bundle_count"] == 1
    assert record["final_reduced_ebr_result_count"] == 0


@pytest.mark.parametrize(
    ("solution_status", "classification", "search_status"),
    [
        (
            "no_exact_solution",
            "in_integer_span_no_nonnegative_witness",
            None,
        ),
        (
            "indeterminate_truncated",
            "indeterminate_truncated",
            "truncated_by_max_coefficient",
        ),
    ],
)
def test_evaluated_nonexact_solution_is_a_final_result(
    solution_status, classification, search_status,
):
    solution = {
        "bundle_id": "b",
        "status": solution_status,
        "classification": classification,
    }
    if search_status is not None:
        solution["search_status"] = search_status
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": [], "iband": [], "input": {}},
        valley_reduced_ebr_mapping={
            "status": solution_status,
            "table_status": "loaded",
            "solutions": [solution],
            "excluded_bundles": [],
        },
    )

    assert record["record_status"] == "has_final_reduced_ebr_results"
    assert record["final_reduced_ebr_result_count"] == 1
    counts = record["reduced_ebr_classification_counts"]
    assert counts[classification] == 1
    assert sum(counts.values()) == record["final_reduced_ebr_result_count"]
    if search_status is not None:
        assert record["reduced_ebr_records"][0]["search_status"] == (
            search_status
        )


def _tr_validation_candidate_bundle():
    return {
        "bundle_id": "tr",
        "source_instance_id": "orbit",
        "problem_kind": "valley_orbit_reduced_ebr",
        "subspace_group_candidate": "P3",
        "spinor": True,
        "ready_for_reduced_table_validation": True,
        "workflow_path": "time_reversal_valley_orbit",
        "readiness_level": "trusted",
        "source_hsp_to_sampled_kpoint": {
            "GM": "GammaM", "K": "KM",
        },
        "time_reversal": {
            "representative_valley": "K",
            "source_hsp_to_sampled_kpoint_by_valley": {
                "K": {"GM": "GammaM", "K": "KM"},
                "Kp": {"GM": "GammaM_Kp", "K": "KM_Kp"},
            },
        },
        "unitary_valley_irreps": {
            "K": {"GM": {"A": 1}, "K": {"B": 2}},
            "Kp": {"GM": {"A": 1}},
        },
        "irrep_records_by_kpoint": {},
    }


def test_tr_validation_candidate_unitary_irreps_survive_without_mapping():
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": [], "iband": [], "input": {}},
        valley_ebr_export_bundle={
            "bundles": [_tr_validation_candidate_bundle()],
        },
    )

    assert record["record_status"] == (
        "has_reduced_table_validation_candidates"
    )
    assert len(record["valley_irrep_records"]) == 3
    assert record["valley_irrep_records"] == [
        {
            "kpoint": "GammaM",
            "source_hsp_label": "GM",
            "valley": "K",
            "subspace_group_candidate": "P3",
            "matched_irrep": "A",
            "irrep_multiplicity": 1,
            "workflow_path": "time_reversal_valley_orbit",
            "readiness_level": "trusted",
            "source": "unitary_valley_irreps",
            "source_bundle_id": "tr",
            "source_instance_id": "orbit",
            "certificate_identity": {},
        },
        {
            "kpoint": "KM",
            "source_hsp_label": "K",
            "valley": "K",
            "subspace_group_candidate": "P3",
            "matched_irrep": "B",
            "irrep_multiplicity": 2,
            "workflow_path": "time_reversal_valley_orbit",
            "readiness_level": "trusted",
            "source": "unitary_valley_irreps",
            "source_bundle_id": "tr",
            "source_instance_id": "orbit",
            "certificate_identity": {},
        },
        {
            "kpoint": "GammaM_Kp",
            "source_hsp_label": "GM",
            "valley": "Kp",
            "subspace_group_candidate": "P3",
            "matched_irrep": "A",
            "irrep_multiplicity": 1,
            "workflow_path": "time_reversal_valley_orbit",
            "readiness_level": "trusted",
            "source": "unitary_valley_irreps",
            "source_bundle_id": "tr",
            "source_instance_id": "orbit",
            "certificate_identity": {},
        },
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_valley_resolved_contract",
        "malformed_valley_resolved_contract",
        "missing_nonrepresentative_component",
        "missing_nonrepresentative_source_hsp",
        "representative_flat_map_conflict",
    ],
)
def test_tr_ingestion_fallback_requires_complete_valley_resolved_binding(
    mutation,
):
    from valleyscope.analysis.database_index import build_database_index

    bundle = _tr_validation_candidate_bundle()
    if mutation == "missing_valley_resolved_contract":
        bundle["time_reversal"].pop(
            "source_hsp_to_sampled_kpoint_by_valley"
        )
    elif mutation == "malformed_valley_resolved_contract":
        bundle["time_reversal"][
            "source_hsp_to_sampled_kpoint_by_valley"
        ] = []
    elif mutation == "missing_nonrepresentative_component":
        bundle["time_reversal"][
            "source_hsp_to_sampled_kpoint_by_valley"
        ].pop("Kp")
    elif mutation == "missing_nonrepresentative_source_hsp":
        bundle["time_reversal"][
            "source_hsp_to_sampled_kpoint_by_valley"
        ]["Kp"].pop("GM")
    else:
        bundle["source_hsp_to_sampled_kpoint"]["GM"] = "wrong_GammaM"

    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": [], "iband": [], "input": {}},
        valley_ebr_export_bundle={"bundles": [bundle]},
    )

    assert record["valley_irrep_records"] == []
    assert len(record["validation_errors"]) == 1
    assert "tr" in record["validation_errors"][0]
    assert "source-HSP/sample binding" in record["validation_errors"][0]
    index = build_database_index([record])
    assert index["valley_irrep_records"] == []


def _tr_completed_unitary_bundle():
    observed_identity = {
        "source": "fixture/K/GM",
        "valley": "K",
        "source_hsp_label": "GM",
        "sampled_kpoint": "GammaM",
        "irrep": "A",
        "multiplicity": 1,
    }
    provenance = {
        "source": "fixture/K/GM",
        "workflow_path": "direct_qcut",
        "irrep_source_provenance": {
            "source_hsp_label": "GM",
            "source_table_spinor": True,
        },
    }
    return {
        "bundle_id": "unitary_K",
        "source_instance_id": "unitary_K_instance",
        "problem_kind": "unitary_valley_reduced_ebr",
        "physical_object_kind": "unitary_valley_projected_subspace",
        "valley": "K",
        "valley_orbit": ["K", "Kp"],
        "subspace_group_candidate": "P3",
        "spinor": True,
        "workflow_path": "time_reversal_completed_unitary_valley",
        "unitary_vector_construction": {
            "kind": "time_reversal_completed_unitary_rows",
            "source": "validated_time_reversal_valley_orbit",
            "orbit_id": "time_reversal_valley_orbit_001",
        },
        "readiness_level": "trusted",
        "ready_for_reduced_table_validation": True,
        "expected_hsps": ["GM", "K", "KA"],
        "irreps_by_kpoint": {"GM": ["A"], "K": ["B"], "KA": ["B"]},
        "source_hsp_to_sampled_kpoint": {
            "GM": "GammaM",
            "K": "KM_K",
        },
        "independent_source_hsp_to_sampled_kpoint": {
            "GM": "GammaM",
            "K": "KM_K",
        },
        "observed_source_hsp_to_sampled_kpoint": {
            "GM": "GammaM",
            "K": "KM_K",
        },
        "time_reversal": {
            "theta_square": -1,
            "mapping_type": "exchanged",
            "valley_orbit": ["K", "Kp"],
            "time_reversal_valley_mapping": {"K": "Kp", "Kp": "K"},
            "time_reversal_hsp_orbits": [
                {
                    "representative": "GM",
                    "members": ["GM"],
                    "self_mapped": True,
                },
                {
                    "representative": "K",
                    "members": ["K", "KA"],
                    "self_mapped": False,
                },
            ],
            "full_unitary_source_hsp_labels": ["GM", "K", "KA"],
            "independent_time_reversal_hsp_labels": ["GM", "K"],
            "time_reversal_irrep_pairing": {
                "A": "A",
                "B": "Bp",
                "Bp": "B",
            },
        },
        "irrep_records_by_kpoint": {},
        "unitary_irrep_completion_records_by_hsp": {
            "GM": [{
                "completion_kind": "observed_at_sampled_kpoint",
                "target_valley": "K",
                "target_source_hsp_label": "GM",
                "irrep": "A",
                "multiplicity": 1,
                "sampled_kpoint": "GammaM",
                "source_candidate_identity": observed_identity,
                "source_candidate_provenance": provenance,
                "structural_status": "validated",
                "readiness_status": "trusted",
                "blockers": [],
            }],
            "K": [{
                "completion_kind": "observed_at_sampled_kpoint",
                "target_valley": "K",
                "target_source_hsp_label": "K",
                "irrep": "B",
                "multiplicity": 1,
                "sampled_kpoint": "KM_K",
                "source_candidate_identity": {
                    **observed_identity,
                    "source": "fixture/K/K",
                    "source_hsp_label": "K",
                    "sampled_kpoint": "KM_K",
                    "irrep": "B",
                },
                "source_candidate_provenance": {
                    **provenance,
                    "source": "fixture/K/K",
                    "irrep_source_provenance": {
                        "source_hsp_label": "K",
                        "source_table_spinor": True,
                    },
                },
                "structural_status": "validated",
                "readiness_status": "trusted",
                "blockers": [],
            }],
            "KA": [{
                "completion_kind": "inferred_by_time_reversal",
                "target_valley": "K",
                "target_source_hsp_label": "KA",
                "irrep": "B",
                "multiplicity": 1,
                "evidence_valley": "Kp",
                "evidence_source_hsp_label": "K",
                "evidence_sampled_kpoint": "KM_Kp",
                "reviewed_time_reversal_relation": {
                    "evidence_valley": "Kp",
                    "target_valley": "K",
                    "evidence_source_hsp_label": "K",
                    "target_source_hsp_label": "KA",
                    "evidence_irrep": "Bp",
                    "target_irrep": "B",
                },
                "source_candidate_identity": {
                    **observed_identity,
                    "source": "fixture/Kp/K",
                    "valley": "Kp",
                    "source_hsp_label": "K",
                    "sampled_kpoint": "KM_Kp",
                    "irrep": "Bp",
                },
                "source_candidate_provenance": {
                    **provenance,
                    "source": "fixture/Kp/K",
                    "irrep_source_provenance": {
                        "source_hsp_label": "K",
                        "source_table_spinor": True,
                    },
                },
                "structural_status": "validated",
                "readiness_status": "trusted",
                "blockers": [],
            }],
        },
    }


def test_tr_unitary_ingestion_preserves_observed_and_inferred_rows():
    unitary = _tr_completed_unitary_bundle()
    legacy_joint = _tr_validation_candidate_bundle()
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": [], "iband": [], "input": {}},
        valley_ebr_export_bundle={"bundles": [unitary, legacy_joint]},
    )

    assert len(record["valley_irrep_records"]) == 3
    observed, observed_k, inferred = record["valley_irrep_records"]
    assert observed["completion_kind"] == "observed_at_sampled_kpoint"
    assert observed["kpoint"] == "GammaM"
    assert observed_k["completion_kind"] == "observed_at_sampled_kpoint"
    assert observed_k["kpoint"] == "KM_K"
    assert inferred["completion_kind"] == "inferred_by_time_reversal"
    assert "kpoint" not in inferred
    assert inferred["evidence_sampled_kpoint"] == "KM_Kp"
    assert inferred["source_bundle_id"] == "unitary_K"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_evidence",
        "missing_tr_metadata",
        "missing_construction",
        "changed_workflow_and_missing_completion",
        "forged_evidence_hsp",
    ],
)
def test_tr_unitary_ingestion_never_uses_joint_representative_fallback(
    mutation,
):
    unitary = _tr_completed_unitary_bundle()
    if mutation == "missing_evidence":
        unitary["unitary_irrep_completion_records_by_hsp"]["KA"][0].pop(
            "evidence_sampled_kpoint"
        )
    elif mutation == "missing_tr_metadata":
        unitary.pop("time_reversal")
    elif mutation == "missing_construction":
        unitary.pop("unitary_vector_construction")
    elif mutation == "changed_workflow_and_missing_completion":
        unitary["workflow_path"] = "direct_qcut"
        unitary.pop("unitary_vector_construction")
        unitary.pop("unitary_irrep_completion_records_by_hsp")
    else:
        inferred = unitary[
            "unitary_irrep_completion_records_by_hsp"
        ]["KA"][0]
        inferred["evidence_source_hsp_label"] = "GM"
        inferred["reviewed_time_reversal_relation"][
            "evidence_source_hsp_label"
        ] = "GM"
        inferred["source_candidate_identity"]["source_hsp_label"] = "GM"
        inferred["source_candidate_provenance"][
            "irrep_source_provenance"
        ]["source_hsp_label"] = "GM"
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": [], "iband": [], "input": {}},
        valley_ebr_export_bundle={
            "bundles": [unitary, _tr_validation_candidate_bundle()],
        },
    )

    assert record["valley_irrep_records"] == []
    assert record["validation_errors"] == [
        "bundle unitary_K: invalid TR-completed unitary provenance"
    ]


def test_ingestion_record_with_ready_bundle():
    """Ready export bundle produces trusted irrep records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": ["GammaM"], "iband": [1],
               "input": {"spinor_convention_verified": True},
               "symmetry_analysis": {"international": "P321", "spacegroup_number": 150}}
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "source_instance_id": "ebr_001",
            "subspace_group_candidate": "P3",
                "ready_for_reduced_table_validation": True,
            "irrep_records_by_kpoint": {
                "GammaM": [{"valley": "K_valley", "operation_id": 1,
                            "operation_order": 3,
                            "matched_irrep": "C3_spinor_phase_+1/2",
                            "eigenphases": [0.5],
                            "workflow_path": "direct_qcut",
                            "readiness_level": "trusted",
                            "source": "valley_irrep_matching/GammaM/K_valley"}],
            },
        }],
    }

    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle)

    assert record["record_status"] == (
        "has_reduced_table_validation_candidates"
    )
    assert record["final_reduced_ebr_result_count"] == 0
    assert record["reduced_table_validation_candidate_bundle_count"] == 1
    assert record["space_group_international"] == "P321"
    records = record["valley_irrep_records"]
    assert len(records) == 1
    r = records[0]
    assert r["kpoint"] == "GammaM"
    assert r["valley"] == "K_valley"
    assert r["subspace_group_candidate"] == "P3"
    assert r["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert r["source_bundle_id"] == "b_001"


def test_ingestion_record_excludes_non_ready_bundles():
    """Non-ready bundles do not contribute trusted irrep records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": [], "iband": [], "input": {}}
    bundle = {
        "bundles": [{
            "bundle_id": "b_001",
                "ready_for_reduced_table_validation": False,
            "irrep_records_by_kpoint": {},
        }],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle)
    assert record["reduced_table_validation_candidate_bundle_count"] == 0
    assert record["valley_irrep_records"] == []


def test_ingestion_record_with_reduced_ebr_mapping():
    """Reduced EBR mapping adds status and classification counts."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": [], "iband": [], "input": {}}
    mapping = {
        "status": "solved_exact",
        "table_status": "loaded",
        "solutions": [
            {
                "classification": "atomic-compatible-candidate",
                "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            },
            {"classification": "atomic-compatible-candidate"},
            {"classification": "in_integer_span_no_nonnegative_witness"},
        ],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_reduced_ebr_mapping=mapping)
    assert record["reduced_ebr_mapping_status"] == "solved_exact"
    counts = record["reduced_ebr_classification_counts"]
    assert counts["atomic_compatible"] == 2
    assert counts["in_integer_span_no_nonnegative_witness"] == 1
    assert counts["outside_integer_span"] == 0


def test_ingestion_record_missing_reduced_ebr_is_not_an_error():
    """Missing reduced EBR mapping is not an error."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(
        valley_summary=summary, valley_reduced_ebr_mapping=None)
    assert record["reduced_ebr_mapping_status"] == "not_available"
    assert record["reduced_ebr_classification_counts"] == {
        "atomic_compatible": 0,
        "in_integer_span_no_nonnegative_witness": 0,
        "outside_integer_span": 0,
        "indeterminate_truncated": 0,
    }
    assert len(record["validation_errors"]) == 0


def test_ingestion_record_from_directory(tmp_path):
    """load_database_ingestion_record_from_directory reads files from dir."""
    from valleyscope.analysis.database_ingestion_record import (
        load_database_ingestion_record_from_directory,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {"target_kpoints": ["GammaM"], "iband": [1],
               "input": {"spinor_convention_verified": False}}
    (run_dir / "valley_summary.json").write_text(json.dumps(summary))

    record = load_database_ingestion_record_from_directory(str(run_dir))
    assert record["summary_status"] == "present"
    assert record["final_reduced_ebr_result_count"] == 0


def test_cli_collect_database_record(tmp_path, capsys):
    """CLI writes ingestion record to requested output path."""
    from valleyscope.cli import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {"target_kpoints": ["GammaM"], "iband": [1],
               "input": {"spinor_convention_verified": False}}
    (run_dir / "valley_summary.json").write_text(json.dumps(summary))

    out_path = tmp_path / "nested" / "record.json"
    rc = main(["collect-database-record", str(run_dir), "-o", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert record["summary_status"] == "present"
    captured = capsys.readouterr().out
    assert "no_reduced_ebr_input" in captured
    assert "validation candidates:" in captured
    assert "final EBR results:" in captured


def test_cli_collect_database_record_returns_nonzero_for_invalid_record(tmp_path):
    """CLI writes invalid record but exits nonzero when summary is missing."""
    from valleyscope.cli import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_path = tmp_path / "record.json"

    rc = main(["collect-database-record", str(run_dir), "-o", str(out_path)])
    assert rc == 1
    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert record["record_status"] == "invalid_missing_summary"
    assert record["validation_errors"]


def test_schema_doc_documents_database_ingestion_record():
    """Public schema documents the explicit offline ingestion-record CLI."""
    schema = Path("docs/schema.md").read_text(encoding="utf-8")
    assert "database_ingestion_record.json" in schema
    assert "collect-database-record" in schema
    assert "valley_irrep_records" in schema
    assert "not a default `analyze-hsp` output" in schema


def test_ingestion_record_no_material_names():
    """Ingestion record module must not contain real material names."""
    src = Path("valleyscope/analysis/database_ingestion_record.py").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src, f"database_ingestion_record.py must not contain {name!r}"


def test_ingestion_record_from_public_outputs_with_reduced_ebr_mapping(tmp_path):
    """Synthetic C3-like public outputs preserve reduced EBR ingestion fields."""
    from valleyscope.analysis.database_ingestion_record import (
        load_database_ingestion_record_from_directory,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {"target_kpoints": ["GammaM", "KM"], "iband": [101, 102], "input": {}}
    c3_records = {
        "GammaM": [
            {"valley": "K_valley", "operation_id": "C3", "operation_order": 3,
             "matched_irrep": "C3_spinor_phase_+1/2", "eigenphases": [0.5],
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "source": "valley_irrep_matching/GammaM/K_valley/C3"},
            {"valley": "K_valley", "operation_id": "C3^2", "operation_order": 3,
             "matched_irrep": "C3_spinor_phase_+1/2", "eigenphases": [0.5],
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "source": "valley_irrep_matching/GammaM/K_valley/C3^2"},
        ],
        "KM": [
            {"valley": "K_valley", "operation_id": "C3", "operation_order": 3,
             "matched_irrep": "C3_spinor_phase_+1/6", "eigenphases": [1 / 6],
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "source": "valley_irrep_matching/KM/K_valley/C3"},
            {"valley": "K_valley", "operation_id": "C3^2", "operation_order": 3,
             "matched_irrep": "C3_spinor_phase_-1/6", "eigenphases": [-1 / 6],
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "source": "valley_irrep_matching/KM/K_valley/C3^2"},
        ],
    }
    c3p_records = {
        kpoint: [
            {**record, "valley": "Kp_valley",
             "source": record["source"].replace("K_valley", "Kp_valley")}
            for record in records
        ]
        for kpoint, records in c3_records.items()
    }
    bundle = {
        "status": "ready_for_reduced_table_validation",
        "bundle_count": 2,
        "excluded_count": 0,
        "bundles": [
            {
                "bundle_id": "b_001", "source_instance_id": "ebr_001",
                "valley": "K_valley",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                "ready_for_reduced_table_validation": True,
                "irrep_records_by_kpoint": c3_records,
            },
            {
                "bundle_id": "b_002", "source_instance_id": "ebr_002",
                "valley": "Kp_valley",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                "ready_for_reduced_table_validation": True,
                "irrep_records_by_kpoint": c3p_records,
            },
        ],
        "excluded_instances": [],
    }
    mapping = {
        "status": "solved_exact", "table_status": "loaded",
        "solutions": [
            {
                "bundle_id": "b_001", "valley": "K_valley",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                "status": "solved_exact",
                "classification": "atomic-compatible-candidate",
                "integer_span_status": "in_integer_span",
                "nonnegative_solution_status": "solved_exact",
                "irrep_vector": [0, 2, 0, 1, 0, 1],
                "ebr_decomposition": [{"label": "-E↑G(2)", "coefficient": 1}],
            },
            {
                "bundle_id": "b_002", "valley": "Kp_valley",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                "status": "solved_exact",
                "classification": "atomic-compatible-candidate",
                "integer_span_status": "in_integer_span",
                "nonnegative_solution_status": "solved_exact",
                "irrep_vector": [0, 2, 0, 1, 0, 1],
                "ebr_decomposition": [{"label": "-E↑G(2)", "coefficient": 1}],
            },
        ],
    }
    (run_dir / "valley_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "valley_ebr_export_bundle.json").write_text(
        json.dumps(bundle), encoding="utf-8"
    )
    (run_dir / "valley_reduced_ebr_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )

    record = load_database_ingestion_record_from_directory(run_dir)

    assert record["schema_version"] == "1.8.0"
    assert record["record_status"] == "has_final_reduced_ebr_results"
    assert record["reduced_table_validation_candidate_bundle_count"] == 2
    assert record["final_reduced_ebr_result_count"] == 2
    assert len(record["valley_irrep_records"]) == 8
    assert record["valley_irrep_records"][0]["valley"] == "K_valley"
    assert record["valley_irrep_records"][0]["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert record["valley_irrep_records"][0]["source_bundle_id"] == "b_001"
    assert record["reduced_ebr_mapping_status"] == "solved_exact"
    assert record["reduced_ebr_table_status"] == "loaded"
    counts = record["reduced_ebr_classification_counts"]
    assert counts["atomic_compatible"] == 2
    assert counts["in_integer_span_no_nonnegative_witness"] == 0
    assert counts["outside_integer_span"] == 0
    assert set(record["source_files"]) == {
        "valley_summary",
        "valley_ebr_export_bundle",
        "valley_reduced_ebr_mapping",
    }
    assert all(Path(path).is_absolute() for path in record["source_files"].values())
    # Per-bundle reduced EBR records
    recs = record["reduced_ebr_records"]
    assert len(recs) == 2
    for r in recs:
        assert r["valley"] in ("K_valley", "Kp_valley")
        assert r["status"] == "solved_exact"
        assert r["classification"] == "atomic-compatible-candidate"
        assert r["integer_span_status"] == "in_integer_span"
        assert r["nonnegative_solution_status"] == "solved_exact"
        assert r["irrep_vector"] == [0, 2, 0, 1, 0, 1]
        assert r["subspace_space_group"] == {
            "candidate_space_group_symbol": "P3"
        }
        assert len(r["ebr_decomposition"]) == 1
        assert r["ebr_decomposition"][0]["coefficient"] == 1


def test_reduced_ebr_records_empty_when_mapping_missing():
    """Missing mapping gives empty reduced_ebr_records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(valley_summary=summary)
    assert record["reduced_ebr_records"] == []


# -----------------------------------------------------------------------
# Multi-run database index collector
# -----------------------------------------------------------------------

def _make_ingestion_record(
    status="has_final_reduced_ebr_results", run_id="run_0000"
):
    return {
        "schema_version": "1.3.0",
        "record_status": status,
        "space_group_international": "P321",
        "space_group_number": 150,
        "reduced_table_validation_candidate_bundle_count": 2,
        "final_reduced_ebr_result_count": 1,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 0,
        "input_excluded_ebr_records": [],
        "final_mapping_excluded_records": [],
        "valley_irrep_records": [
            {"kpoint": "GammaM", "valley": "K_valley",
             "subspace_group_candidate": "P3"},
        ],
        "reduced_ebr_records": [
            {"bundle_id": "b_001", "valley": "K_valley",
             "subspace_group_candidate": "P3",
             "status": "solved_exact",
             "classification": "atomic-compatible-candidate",
             "irrep_vector": [0, 2, 0, 1, 0, 1],
             "ebr_decomposition": [{"label": "-E↑G(2)", "coefficient": 1}]},
        ],
        "reduced_ebr_classification_counts": {
            "atomic_compatible": 1, "in_integer_span_no_nonnegative_witness": 0, "outside_integer_span": 0,
        },
        "reduced_ebr_mapping_status": "solved_exact",
        "reduced_ebr_table_status": "loaded",
        "validation_errors": [],
    }


def test_database_index_builder_two_records():
    """Pure builder with final-result and no-input records."""
    from valleyscope.analysis.database_index import build_database_index
    rec1 = _make_ingestion_record("has_final_reduced_ebr_results")
    rec2 = _make_ingestion_record("no_reduced_ebr_input")
    index = build_database_index(
        [rec1, rec2],
        source_files=[
            "/tmp/run_a/database_ingestion_record.json",
            "/tmp/run_b/database_ingestion_record.json",
        ],
    )
    assert index["record_count"] == 2
    assert index["status_counts"]["has_final_reduced_ebr_results"] == 1
    assert index["status_counts"]["no_reduced_ebr_input"] == 1
    assert index[
        "reduced_table_validation_candidate_bundle_count_total"
    ] == 4
    assert index["final_reduced_ebr_result_count_total"] == 2
    assert index["reduced_ebr_classification_counts_total"]["atomic_compatible"] == 2
    # Flattened records have run_id provenance.
    assert index["runs"][0]["run_id"] == "run_0000"
    assert index["runs"][1]["run_id"] == "run_0001"
    assert index["runs"][0]["source"].endswith("/run_a/database_ingestion_record.json")
    for ir in index["valley_irrep_records"]:
        assert "run_id" in ir
        assert "source_record" in ir
    for rr in index["reduced_ebr_records"]:
        assert "run_id" in rr
        assert "source_record" in rr
    assert len(index["reduced_ebr_records"]) == 2


def test_database_index_aggregates_indeterminate_truncated_classification():
    from valleyscope.analysis.database_index import build_database_index

    record = _make_ingestion_record()
    record["final_reduced_ebr_result_count"] = 2
    record["reduced_ebr_classification_counts"] = {
        "atomic_compatible": 1,
        "in_integer_span_no_nonnegative_witness": 0,
        "outside_integer_span": 0,
        "indeterminate_truncated": 1,
    }

    index = build_database_index([record])

    assert index["final_reduced_ebr_result_count_total"] == 2
    assert index["reduced_ebr_classification_counts_total"] == {
        "atomic_compatible": 1,
        "in_integer_span_no_nonnegative_witness": 0,
        "outside_integer_span": 0,
        "indeterminate_truncated": 1,
    }


def test_database_index_uses_stage_owned_aggregates_only():
    from valleyscope.analysis.database_index import build_database_index

    record = {
        "record_status": "has_final_reduced_ebr_results",
        "reduced_table_validation_candidate_bundle_count": 2,
        "final_reduced_ebr_result_count": 1,
        "final_mapping_excluded_bundle_count": 1,
        "input_excluded_instance_count": 3,
        "valley_irrep_records": [],
        "reduced_ebr_records": [{"bundle_id": "b"}],
        "input_excluded_ebr_records": [{"source_instance_id": "i"}],
        "final_mapping_excluded_records": [{"bundle_id": "blocked"}],
        "reduced_ebr_classification_counts": {},
        "validation_errors": [],
    }

    index = build_database_index([record])

    assert index["schema_version"] == "1.1.0"
    assert index["status_counts"]["has_final_reduced_ebr_results"] == 1
    assert index[
        "reduced_table_validation_candidate_bundle_count_total"
    ] == 2
    assert index["final_reduced_ebr_result_count_total"] == 1
    assert index["final_mapping_excluded_bundle_count_total"] == 1
    assert index["input_excluded_instance_count_total"] == 3
    assert index["input_excluded_ebr_records"][0]["run_id"] == "run_0000"
    assert index["final_mapping_excluded_records"][0]["run_id"] == (
        "run_0000"
    )
    assert "ready_bundle_count_total" not in index
    assert "excluded_ebr_records" not in index


def test_database_index_cli_writes_json(tmp_path):
    """CLI collect-database-index writes database_index.json."""
    from valleyscope.cli import main
    rec1_path = tmp_path / "rec1.json"
    rec1_path.write_text(json.dumps(_make_ingestion_record(
        "has_final_reduced_ebr_results"
    )))
    rec2_path = tmp_path / "rec2.json"
    rec2_path.write_text(json.dumps(_make_ingestion_record(
        "no_reduced_ebr_input"
    )))
    out = tmp_path / "index.json"
    rc = main(["collect-database-index", str(rec1_path), str(rec2_path),
               "-o", str(out)])
    assert rc == 0
    assert out.exists()
    idx = json.loads(out.read_text())
    assert idx["record_count"] == 2
    assert idx[
        "reduced_table_validation_candidate_bundle_count_total"
    ] == 4
    assert idx["final_reduced_ebr_result_count_total"] == 2


def test_database_index_cli_invalid_input(tmp_path):
    """CLI returns nonzero on missing input file."""
    from valleyscope.cli import main
    rec1_path = tmp_path / "rec1.json"
    rec1_path.write_text(json.dumps(_make_ingestion_record(
        "has_final_reduced_ebr_results"
    )))
    out = tmp_path / "index.json"
    rc = main(["collect-database-index", str(rec1_path), "/nonexistent/path.json",
               "-o", str(out)])
    assert rc != 0
    assert out.exists()
    idx = json.loads(out.read_text())
    assert idx["record_count"] == 2
    assert idx["status_counts"]["has_final_reduced_ebr_results"] == 1
    assert idx["status_counts"]["invalid_missing_summary"] == 1
    assert idx["validation_errors"]
    assert "FileNotFoundError" in idx["validation_errors"][0]


def test_database_index_module_no_material_names():
    """Index module must not contain real material names."""
    src = Path("valleyscope/analysis/database_index.py").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src


def test_ingestion_record_includes_input_excluded_ebr_records():
    """Export exclusions remain distinct input-stage records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": [], "iband": [], "input": {}}
    bundle = {
        "status": "partial_export",
        "interpretation": "one blocked instance",
        "bundles": [],
        "excluded_instances": [
            {
                "source_instance_id": "ebr_001", "valley": "M3_valley",
                "subspace_group_candidate": "P2",
                "subspace_space_group": {"candidate_space_group_symbol": "P2"},
                "status": "blocked",
                "canonical_hsp_vector_complete": False,
                "exclusion_reasons": [
                    "spinor_convention_unverified",
                    "low_seed_projector_symmetry",
                ],
            },
        ],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle,
    )
    assert record["ebr_export_status"] == "partial_export"
    assert record["ebr_export_interpretation"] == "one blocked instance"
    exclude = record["input_excluded_ebr_records"]
    assert len(exclude) == 1
    assert exclude[0]["source_instance_id"] == "ebr_001"
    assert exclude[0]["valley"] == "M3_valley"
    assert exclude[0]["subspace_space_group"] == {
        "candidate_space_group_symbol": "P2"
    }
    assert exclude[0]["exclusion_reasons"] == [
        "spinor_convention_unverified", "low_seed_projector_symmetry",
    ]


def test_input_excluded_ebr_records_empty_when_not_present():
    """Missing bundle gives empty input_excluded_ebr_records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(valley_summary=summary)
    assert record["input_excluded_ebr_records"] == []
    assert record["ebr_export_status"] == "not_available"


def test_database_index_input_excluded_records_aggregated():
    """Index aggregates input exclusions with run provenance."""
    from valleyscope.analysis.database_index import build_database_index
    rec = {
        "schema_version": "1.3.0",
        "record_status": "no_reduced_ebr_input",
        "reduced_table_validation_candidate_bundle_count": 0,
        "final_reduced_ebr_result_count": 0,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 1,
        "valley_irrep_records": [],
        "reduced_ebr_records": [],
        "reduced_ebr_classification_counts": {},
        "reduced_ebr_mapping_status": "?",
        "reduced_ebr_table_status": "?",
        "ebr_export_status": "partial_export",
        "input_excluded_ebr_records": [
            {"source_instance_id": "ebr_001", "valley": "M3_valley",
             "exclusion_reasons": ["spinor_convention_unverified"]},
        ],
        "validation_errors": [],
    }
    idx = build_database_index([rec])
    assert idx["input_excluded_instance_count_total"] == 1
    assert idx["ebr_export_status_counts"]["partial_export"] == 1
    assert idx["input_excluded_ebr_records"][0]["run_id"] == "run_0000"


def test_database_index_input_exclusions_have_source_record():
    """Input exclusions carry source_record when source_files are provided."""
    from valleyscope.analysis.database_index import build_database_index
    rec = {
        "record_status": "no_reduced_ebr_input",
        "reduced_table_validation_candidate_bundle_count": 0,
        "final_reduced_ebr_result_count": 0,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 1,
        "valley_irrep_records": [],
        "reduced_ebr_records": [],
        "reduced_ebr_classification_counts": {},
        "reduced_ebr_mapping_status": "?",
        "reduced_ebr_table_status": "?",
        "ebr_export_status": "no_bundles",
        "input_excluded_ebr_records": [
            {"source_instance_id": "ebr_x", "valley": "M1_valley",
             "exclusion_reasons": ["low_seed_overlap"]},
        ],
    }
    idx = build_database_index([rec], source_files=["/tmp/rec.json"])
    er = idx["input_excluded_ebr_records"][0]
    assert er["run_id"] == "run_0000"
    assert er["source_record"] == "/tmp/rec.json"


def test_ingestion_record_schema_version_is_1_8_0():
    """Ingestion record schema_version is now 1.8.0."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(valley_summary=summary)
    assert record["schema_version"] == "1.8.0"


# -----------------------------------------------------------------------


def test_irrep_records_preserve_generic_fields():
    """Generic irrep provenance fields survive ingestion flattening."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": ["GammaM"], "iband": [1], "input": {}}
    bundle = {
        "bundles": [{
            "bundle_id": "b_001",
            "subspace_group_candidate": "P3",
                "ready_for_reduced_table_validation": True,
            "irrep_records_by_kpoint": {
                "GammaM": [{
                    "valley": "K_valley",
                    "operation_id": 2,
                    "operation_order": 3,
                    "matched_irrep": "-GM5",
                    "eigenphases": [0.5],
                    "workflow_path": "direct_qcut",
                    "readiness_level": "trusted",
                    "source": "generic/GammaM/K_valley",
                    "irrep_multiplicity": 2,
                    "matching_strategy": "bilbao_restricted_character",
                    "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                    "legacy_subspace_group_candidate": "C3_like",
                    "valley_preserving_operation_ids": [0, 2, 3],
                    "source_operation_map": {0: 1, 2: 2, 3: 3},
                    "irrep_source_provenance": {
                        "source_hsp_label": "GM",
                        "source_table_sg_number": 143,
                        "standard_setting_hsp_mapping": {
                            "standard_setting_certificate": {
                                "validation_status": "validated",
                                "subspace_sg_number": 143,
                                "resolved_hsp_label": "GM",
                                "centering_status": "primitive_direct_match",
                            },
                        },
                    },
                }],
            },
        }],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle,
    )
    recs = record["valley_irrep_records"]
    assert len(recs) == 1
    r = recs[0]
    assert r["irrep_multiplicity"] == 2
    assert r["matching_strategy"] == "bilbao_restricted_character"
    assert r["subspace_space_group"] == {"candidate_space_group_symbol": "P3"}
    assert r["valley_preserving_operation_ids"] == [0, 2, 3]
    assert r["source_operation_map"] == {0: 1, 2: 2, 3: 3}
    cert = (
        r["irrep_source_provenance"]
        ["standard_setting_hsp_mapping"]
        ["standard_setting_certificate"]
    )
    assert cert["validation_status"] == "validated"
    assert cert["subspace_sg_number"] == 143
    assert cert["resolved_hsp_label"] == "GM"


def test_ingestion_preserves_centered_certificate_identity_from_bundle():
    from valleyscope.analysis.database_ingestion_record import (
        build_database_ingestion_record,
    )

    centered_map = [{
        "parent_operation_id": -3,
        "centering_coset_index": 0,
        "standard_operation_index": 0,
    }, {
        "parent_operation_id": -3,
        "centering_coset_index": 1,
        "standard_operation_index": 1,
    }]
    certificate_identity = {
        "sg_number": 5,
        "hall_number": 9,
        "centering_type": "C",
        "centered_affine_operation_map": centered_map,
        "affine_unmatched_centered_operation_pairs": [],
    }
    bundle = {
        "bundles": [{
            "bundle_id": "b_centered",
            "source_instance_id": "i_centered",
            "subspace_group_candidate": "C2",
                "ready_for_reduced_table_validation": True,
            "certificate_identity": certificate_identity,
            "irrep_records_by_kpoint": {
                "GM": [{
                    "valley": "K_valley",
                    "operation_id": -3,
                    "matched_irrep": "GM1",
                }],
            },
        }],
    }
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": ["GM"], "iband": [1], "input": {}},
        valley_ebr_export_bundle=bundle,
    )

    assert record["schema_version"] == "1.8.0"
    assert record["valley_irrep_records"][0]["certificate_identity"] == (
        certificate_identity
    )


def test_legacy_records_still_ingest_without_generic_fields():
    """Legacy records without generic fields ingest successfully."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": ["GammaM"], "iband": [1], "input": {}}
    bundle = {
        "bundles": [{
            "bundle_id": "b_001",
            "subspace_group_candidate": "P3",
                "ready_for_reduced_table_validation": True,
            "irrep_records_by_kpoint": {
                "GammaM": [{
                    "valley": "K_valley",
                    "operation_id": 1,
                    "operation_order": 3,
                    "matched_irrep": "C3_spinor_phase_+1/2",
                    "eigenphases": [0.5],
                    "workflow_path": "direct_qcut",
                    "readiness_level": "trusted",
                    "source": "legacy/GammaM/K_valley",
                }],
            },
        }],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle,
    )
    assert record["final_reduced_ebr_result_count"] == 0
    assert record["reduced_table_validation_candidate_bundle_count"] == 1
    r = record["valley_irrep_records"][0]
    assert r["matched_irrep"] == "C3_spinor_phase_+1/2"
    for key in [
        "irrep_multiplicity",
        "matching_strategy",
        "subspace_space_group",
        "legacy_subspace_group_candidate",
        "valley_preserving_operation_ids",
        "source_operation_map",
    ]:
        assert key not in r


def test_database_index_preserves_generic_irrep_fields_with_run_provenance():
    """Database index keeps generic irrep fields with run provenance."""
    from valleyscope.analysis.database_index import build_database_index

    record = {
        "schema_version": "1.3.0",
        "record_status": "has_reduced_table_validation_candidates",
        "reduced_table_validation_candidate_bundle_count": 1,
        "final_reduced_ebr_result_count": 0,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 0,
        "input_excluded_ebr_records": [],
        "final_mapping_excluded_records": [],
        "valley_irrep_records": [{
            "kpoint": "GammaM",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "matched_irrep": "-GM5",
            "irrep_multiplicity": 2,
            "matching_strategy": "bilbao_restricted_character",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "legacy_subspace_group_candidate": "C3_like",
            "valley_preserving_operation_ids": [0, 2, 3],
            "source_operation_map": {0: 1, 2: 2, 3: 3},
        }],
        "reduced_ebr_records": [],
        "reduced_ebr_classification_counts": {
            "atomic_compatible": 0,
            "in_integer_span_no_nonnegative_witness": 0,
            "outside_integer_span": 0,
        },
        "reduced_ebr_mapping_status": "not_available",
        "reduced_ebr_table_status": "not_available",
        "validation_errors": [],
    }

    index = build_database_index(
        [record],
        source_files=["/tmp/database_ingestion_record.json"],
    )
    ir = index["valley_irrep_records"][0]
    assert ir["run_id"] == "run_0000"
    assert ir["source_record"] == "/tmp/database_ingestion_record.json"
    assert ir["irrep_multiplicity"] == 2
    assert ir["matching_strategy"] == "bilbao_restricted_character"
    assert ir["subspace_space_group"] == {"candidate_space_group_symbol": "P3"}
    assert ir["legacy_subspace_group_candidate"] == "C3_like"
    assert ir["valley_preserving_operation_ids"] == [0, 2, 3]
    assert ir["source_operation_map"] == {0: 1, 2: 2, 3: 3}


# ---------------------------------------------------------------------------
# Compact reduced EBR table provenance in ingestion records
# ---------------------------------------------------------------------------

def _auto_table_provenance():
    """Minimal auto-canonical table_provenance dict."""
    return {
        "source": "auto_canonical",
        "auto_canonical": True,
        "subspace_group_candidate": "P3",
        "space_group_number": 143,
        "spinful": True,
        "data_source": "irreptables",
        "package": "irreptables",
        "package_version": "3.1.0",
        "expected_hsps": ["GammaM", "KM"],
        "valleyscope_reduction": "sampled_hsp_valley_preserving",
        "source_basis_count": 20,
        "reduction_basis_count": 6,
        "dropped_source_row_count": 14,
        "dropped_source_rows": ["label1", "label2"],
    }


def test_reduced_ebr_records_pick_up_table_provenance():
    """Compact ingestion records carry table_provenance fields when present."""
    mapping = {
        "status": "solved_exact",
        "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_001", "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "status": "solved_exact",
            "classification": "atomic-compatible-candidate",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "solved_exact",
            "irrep_vector": [1, 0],
            "ebr_decomposition": [{"label": "E@1a", "coefficient": 1}],
            "table_provenance": _auto_table_provenance(),
            "table_status": "loaded",
        }],
    }
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": ["GammaM", "KM"], "iband": [1, 2],
                        "input": {}},
        valley_reduced_ebr_mapping=mapping,
    )
    recs = record["reduced_ebr_records"]
    assert len(recs) == 1
    r = recs[0]
    assert r["table_source"] == "auto_canonical"
    assert r["data_source"] == "irreptables"
    assert r["package"] == "irreptables"
    assert r["package_version"] == "3.1.0"
    assert r["space_group_number"] == 143
    assert r["spinful"] is True
    assert r["expected_hsps"] == ["GammaM", "KM"]
    assert r["valleyscope_reduction"] == "sampled_hsp_valley_preserving"
    assert r["source_basis_count"] == 20
    assert r["reduction_basis_count"] == 6
    assert r["dropped_source_row_count"] == 14
    assert r["table_status"] == "loaded"
    assert r["dropped_source_rows"] == ["label1", "label2"]
    assert r["filtered_zero_vector_ebr_count"] == 0
    assert r["filtered_zero_vector_ebrs"] == []
    assert "auto_canonical" not in r


def test_reduced_ebr_records_preserve_joint_valley_orbit_identity():
    """Compact ingestion keeps the physical identity of a joint TR problem."""
    time_reversal = {
        "theta_square": -1,
        "time_reversal_valley_mapping": {
            "valley_a": "valley_b",
            "valley_b": "valley_a",
        },
        "representative_valley": "valley_a",
        "source_hsp_to_sampled_kpoint_by_valley": {
            "valley_a": {"K": "K_a"},
            "valley_b": {"K": "K_b"},
        },
    }
    unitary_valley_irreps = {
        "valley_a": {"K": {"rho_a": 1}},
        "valley_b": {"K": {"rho_b": 1}},
    }
    mapping = {
        "status": "no_exact_solution",
        "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_orbit",
            "valley": "",
            "problem_kind": "valley_orbit_reduced_ebr",
            "valley_orbit": ["valley_a", "valley_b"],
            "unitary_valley_irreps": unitary_valley_irreps,
            "time_reversal": time_reversal,
            "subspace_group_candidate": "P1",
            "status": "no_exact_solution",
            "classification": "in_integer_span_no_nonnegative_witness",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "no_nonnegative_solution",
            "irrep_vector": [1],
        }],
    }

    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": ["K"], "iband": [1], "input": {}},
        valley_reduced_ebr_mapping=mapping,
    )

    assert record["reduced_ebr_records"] == [{
        "bundle_id": "b_orbit",
        "valley": "",
        "problem_kind": "valley_orbit_reduced_ebr",
        "valley_orbit": ["valley_a", "valley_b"],
        "unitary_valley_irreps": unitary_valley_irreps,
        "time_reversal": time_reversal,
        "subspace_group_candidate": "P1",
        "subspace_space_group": {},
        "status": "no_exact_solution",
        "classification": "in_integer_span_no_nonnegative_witness",
        "integer_span_status": "in_integer_span",
        "nonnegative_solution_status": "no_nonnegative_solution",
        "irrep_vector": [1],
    }]


def test_reduced_ebr_records_no_table_provenance_still_works():
    """Records without table_provenance work (backward compat)."""
    mapping = {
        "status": "solved_exact", "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_001", "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "classification": "atomic-compatible-candidate",
            "status": "solved_exact",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "solved_exact",
            "irrep_vector": [1],
        }],
    }
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": [], "iband": [], "input": {}},
        valley_reduced_ebr_mapping=mapping,
    )
    r = record["reduced_ebr_records"][0]
    assert r["bundle_id"] == "b_001"
    assert "table_source" not in r
    assert "table_provenance" not in r


def test_reduced_ebr_records_from_directory_with_table_provenance(tmp_path):
    """Directory loading preserves compact table_provenance in records."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {"target_kpoints": ["GammaM", "KM"], "iband": [1, 2], "input": {}}
    mapping = {
        "status": "solved_exact", "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_001", "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "status": "solved_exact",
            "classification": "atomic-compatible-candidate",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "solved_exact",
            "irrep_vector": [1, 0],
            "ebr_decomposition": [{"label": "E@1a", "coefficient": 1}],
            "table_provenance": _auto_table_provenance(),
            "table_status": "loaded",
        }],
    }
    (run_dir / "valley_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "valley_reduced_ebr_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )

    record = load_database_ingestion_record_from_directory(run_dir)
    recs = record["reduced_ebr_records"]
    assert len(recs) == 1
    r = recs[0]
    assert r["table_source"] == "auto_canonical"
    assert r["space_group_number"] == 143
    assert r["expected_hsps"] == ["GammaM", "KM"]
    assert r["reduction_basis_count"] == 6
    assert r["dropped_source_rows"] == ["label1", "label2"]


def test_tmote2_ingestion_compact_reduced_ebr_records():
    """tMoTe2 fixture retains two unitary and one joint TR EBR records."""
    ing = load_database_ingestion_record_from_directory(
        Path(__file__).parent.parent / "real_tests" / "tMoTe2" / "output"
        / "valley_analysis_wave",
    )
    recs = ing.get("reduced_ebr_records", [])
    if not recs:
        pytest.skip("tMoTe2 fixture output not found or no reduced EBR records")

    assert len(recs) == 3
    r = next(
        row for row in recs
        if row["problem_kind"] == "valley_orbit_reduced_ebr"
    )
    assert r["problem_kind"] == "valley_orbit_reduced_ebr"
    assert r["valley_orbit"] == ["K_valley", "Kp_valley"]
    assert set(r["unitary_valley_irreps"]) == {"K_valley", "Kp_valley"}
    assert r["time_reversal"]["time_reversal_valley_mapping"] == {
        "K_valley": "Kp_valley",
        "Kp_valley": "K_valley",
    }
    assert r["subspace_group_candidate"] == "P3"
    assert r["classification"] == "in_integer_span_no_nonnegative_witness"
    assert r["table_source"] == "auto_time_reversal_grey"
    assert r["data_source"] == "irreptables"
    assert r["space_group_number"] == 143
    assert r["spinful"] is True
    assert r["expected_hsps"] == ["GM", "K", "M"]
    assert r["valleyscope_reduction"] == "sampled_hsp_valley_preserving"
    assert r["source_basis_count"] > r["reduction_basis_count"] > 0
    assert r["table_status"] == "loaded"
    assert isinstance(r["dropped_source_rows"], list)
    unitary = {
        row["valley"]: row for row in recs
        if row["problem_kind"] == "unitary_valley_reduced_ebr"
    }
    assert set(unitary) == {"K_valley", "Kp_valley"}
    assert all(
        row["physical_object_kind"] == (
            "unitary_valley_projected_subspace"
        )
        and row["classification"] == "outside_integer_span"
        and row["table_source"] == "auto_canonical"
        for row in unitary.values()
    )
    irrep_rows = ing["valley_irrep_records"]
    assert len(irrep_rows) == 8
    inferred = [
        row for row in irrep_rows
        if row.get("completion_kind") == "inferred_by_time_reversal"
    ]
    assert {
        (row["valley"], row["source_hsp_label"], row["matched_irrep"])
        for row in inferred
    } == {
        ("K_valley", "KA", "-KA5"),
        ("Kp_valley", "KA", "-KA6"),
    }
    assert all("kpoint" not in row for row in inferred)
    assert all(row["evidence_sampled_kpoint"] == "KM" for row in inferred)
