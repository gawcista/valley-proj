"""Offline collection boundaries, including genuine numerical output controls."""

from copy import deepcopy
import json

import pytest

from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
from valleyscope.analysis.database_index import build_database_index
from valleyscope.cli import main
from tests.portable_numerical_chain import run_numerical_workflow


@pytest.fixture(scope="module")
def numerical_outputs(tmp_path_factory):
    result = run_numerical_workflow(tmp_path_factory.mktemp("database-boundary"), profile="standard")
    assert result["ingestion"]["validation_errors"] == []
    assert result["ingestion"]["final_reduced_ebr_result_count"] == 2
    reports = result["reports"]
    return (
        reports["valley_summary_json"],
        reports["valley_ebr_export_bundle_json"],
        reports["valley_reduced_ebr_mapping_json"],
    )


def _collect(outputs, summary=None):
    return build_database_ingestion_record(
        valley_summary=outputs[0] if summary is None else summary,
        valley_ebr_export_bundle=outputs[1],
        valley_reduced_ebr_mapping=outputs[2],
    )


@pytest.mark.parametrize("edit", ["source_identity", "source_status", "missing_scope", "scope_identity", "duplicate_scope"])
def test_summary_conflicts_exclude_affected_final_results(numerical_outputs, edit):
    summary = deepcopy(numerical_outputs[0])
    cprime = summary["cprime"]
    expected = 0
    if edit == "source_identity":
        cprime["spinor_source_basis"]["identity"] = "sha256:" + "0" * 64
    elif edit == "source_status":
        cprime["spinor_source_basis"]["status"] = "blocked"
    else:
        expected = 1
        row = next(row for row in cprime["acceptance_matrix"] if row["valley"] == "plus")
        if edit == "missing_scope":
            cprime["acceptance_matrix"].remove(row)
        elif edit == "duplicate_scope":
            cprime["acceptance_matrix"].append(deepcopy(row))
        else:
            row["scoped_representation_evidence_identity"] = "sha256:" + "0" * 64
    record = _collect(numerical_outputs, summary)
    assert record["validation_errors"]
    assert record["final_reduced_ebr_result_count"] == expected
    assert len(record["reduced_ebr_records"]) == expected
    assert record["reduced_table_validation_candidate_bundle_count"] == expected
    assert record["record_status"] == (
        "has_final_reduced_ebr_results" if expected else "no_reduced_ebr_input"
    )
    if expected:
        assert {row["valley"] for row in record["reduced_ebr_records"]} == {"minus"}
    index = build_database_index([record])
    assert index["final_reduced_ebr_result_count_total"] == expected
    assert index["validation_errors"]


def test_genuine_final_records_survive_index_validation(numerical_outputs):
    index = build_database_index([_collect(numerical_outputs)])
    assert index["validation_errors"] == []
    assert index["final_reduced_ebr_result_count_total"] == 2
    assert len(index["reduced_ebr_records"]) == 2


@pytest.mark.parametrize("bad_status", [[], {}, ["has_final_reduced_ebr_results"], None, True])
def test_malformed_status_isolated_from_valid_neighbor(bad_status):
    valid = build_database_ingestion_record(valley_summary={})
    bad = deepcopy(valid)
    bad["record_status"] = bad_status
    index = build_database_index([bad, valid])
    assert index["record_count"] == 2
    assert [row["record_status"] for row in index["runs"]] == [
        "invalid_missing_summary", "no_reduced_ebr_input",
    ]
    assert index["validation_errors"]
    assert index["final_reduced_ebr_result_count_total"] == 0


@pytest.mark.parametrize("edit", [
    "final_count", "classification_count", "classification_key", "row_classification",
    "row_status", "record_status", "missing_final_rows", "final_exclusions", "input_exclusions",
    "invalid_final_row", "invalid_irrep_row", "invalid_input_row", "invalid_excluded_row",
    "error_entry", "candidate_count",
    "duplicate_final_id", "missing_final_id", "malformed_final_id",
])
def test_inconsistent_record_cannot_pollute_index(numerical_outputs, edit):
    valid = _collect(numerical_outputs)
    bad = deepcopy(valid)
    if edit == "final_count":
        bad["final_reduced_ebr_result_count"] = 999
    elif edit == "classification_count":
        bad["reduced_ebr_classification_counts"]["atomic_compatible"] = 999
    elif edit == "classification_key":
        bad["reduced_ebr_classification_counts"]["unknown"] = 1
    elif edit == "row_classification":
        bad["reduced_ebr_records"][0]["classification"] = []
    elif edit == "row_status":
        bad["reduced_ebr_records"][0]["status"] = "blocked"
    elif edit == "record_status":
        bad["record_status"] = "no_reduced_ebr_input"
    elif edit == "missing_final_rows":
        bad["reduced_ebr_records"] = []
    elif edit == "final_exclusions":
        bad["final_mapping_excluded_bundle_count"] = 1
    elif edit == "input_exclusions":
        bad["input_excluded_instance_count"] = 1
    elif edit.startswith("invalid_"):
        field = {
            "invalid_final_row": "reduced_ebr_records",
            "invalid_irrep_row": "valley_irrep_records",
            "invalid_input_row": "input_excluded_ebr_records",
            "invalid_excluded_row": "final_mapping_excluded_records",
        }[edit]
        bad[field].append(None)
    elif edit == "error_entry":
        bad["validation_errors"] = [{}]
    elif edit == "candidate_count":
        bad["reduced_table_validation_candidate_bundle_count"] = 0
    elif edit == "duplicate_final_id":
        bad["reduced_ebr_records"][1] = deepcopy(bad["reduced_ebr_records"][0])
    elif edit == "missing_final_id":
        bad["reduced_ebr_records"][0].pop("bundle_id")
    elif edit == "malformed_final_id":
        bad["reduced_ebr_records"][0]["bundle_id"] = []
    index = build_database_index([bad, valid])
    assert index["validation_errors"]
    assert index["runs"][0]["record_status"] == "invalid_missing_summary"
    assert index["final_reduced_ebr_result_count_total"] == 2
    assert len(index["reduced_ebr_records"]) == 2
    assert {row["run_id"] for row in index["reduced_ebr_records"]} == {"run_0001"}


def test_cli_bad_record_preserves_valid_input_and_writes_errors(tmp_path, numerical_outputs):
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    output = tmp_path / "index.json"
    record = _collect(numerical_outputs)
    good.write_text(json.dumps(record))
    record["record_status"] = []
    bad.write_text(json.dumps(record))
    assert main(["collect-database-index", str(bad), str(good), "--output", str(output)]) == 1
    index = json.loads(output.read_text())
    assert index["validation_errors"]
    assert index["final_reduced_ebr_result_count_total"] == 2
    assert index["status_counts"]["invalid_missing_summary"] == 1


def test_zero_rows_cannot_declare_final_results():
    record = build_database_ingestion_record(valley_summary={})
    record["record_status"] = "has_final_reduced_ebr_results"
    record["final_reduced_ebr_result_count"] = 999
    record["reduced_ebr_classification_counts"]["atomic_compatible"] = 999
    index = build_database_index([record])
    assert index["validation_errors"]
    assert index["final_reduced_ebr_result_count_total"] == 0
    assert index["reduced_ebr_records"] == []


@pytest.mark.parametrize("bad_id", [[], {}, None, ""])
def test_malformed_mapping_solution_id_keeps_valid_neighbor(numerical_outputs, bad_id):
    summary, export, mapping = deepcopy(numerical_outputs)
    mapping["solutions"][0]["bundle_id"] = bad_id
    record = _collect((summary, export, mapping))
    assert record["validation_errors"]
    assert record["final_reduced_ebr_result_count"] == 1
    assert len(record["reduced_ebr_records"]) == 1


def test_duplicate_export_id_still_blocks_its_final_result(numerical_outputs):
    summary, export, mapping = deepcopy(numerical_outputs)
    duplicate = deepcopy(export["bundles"][0])
    duplicate["cprime_identity_by_kpoint"] = {}
    export["bundles"].append(duplicate)
    record = _collect((summary, export, mapping))
    assert record["validation_errors"]
    assert record["final_reduced_ebr_result_count"] == 1
    assert all(row["bundle_id"] != duplicate["bundle_id"] for row in record["reduced_ebr_records"])
