"""Real producer-chain regression for TR-completed reduced EBR output."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from valleyscope.analysis.database_ingestion_record import (
    build_database_ingestion_record,
)
from valleyscope.workflows.analyze_hsp import analyze_hsp


_ROOT = Path(__file__).resolve().parent.parent
_TMO_CONFIG = _ROOT / "real_tests" / "tMoTe2" / "analyze.yaml"


def test_real_tmo_chain_needs_no_synthetic_cprime_or_sampled_ka():
    outputs = analyze_hsp(_TMO_CONFIG)
    summary = json.loads(
        outputs["valley_summary_json"].read_text(encoding="utf-8")
    )
    export = json.loads(
        outputs["valley_ebr_export_bundle_json"].read_text(encoding="utf-8")
    )
    mapping = json.loads(
        outputs["valley_reduced_ebr_mapping_json"].read_text(
            encoding="utf-8"
        )
    )

    assert summary["cprime"]["spinor_source_basis"]["status"] == "passed"
    local_rows = [
        row for row in summary["cprime"]["acceptance_matrix"]
        if row.get("scope_kind") == "local_irrep"
    ]
    tr_rows = [
        row for row in summary["cprime"]["acceptance_matrix"]
        if row.get("scope_kind") == "tr_completed"
    ]
    assert len(local_rows) == 6
    assert all(row["scoped_representation_status"] == "passed" for row in local_rows)
    assert tr_rows
    assert all(row["scoped_representation_status"] == "passed" for row in tr_rows)

    problems = summary["valley_ebr_problem_instances"]
    assert problems["status"] == "canonical_hsp_vectors_ready"
    assert problems["instance_count"] == 3
    assert problems["ready_instance_count"] == 3
    assert all(instance["blocked_by"] == [] for instance in problems["instances"])
    assert "missing trusted source HSPs" not in json.dumps(problems)

    assert len(export["bundles"]) == 3
    assert all(
        bundle["ready_for_reduced_table_validation"] is True
        for bundle in export["bundles"]
    )
    unitary = [
        bundle for bundle in export["bundles"]
        if bundle["problem_kind"] == "unitary_valley_reduced_ebr"
    ]
    assert len(unitary) == 2
    assert all(
        set(bundle["cprime_identity_by_kpoint"])
        == {"GM", "K", "KA", "M"}
        for bundle in unitary
    )

    assert mapping["table_status"] == "loaded"
    assert len(mapping["solutions"]) == 3
    assert mapping["excluded_bundles"] == []
    assert all(
        solution["table_provenance"]["data_source"] == "irreptables"
        for solution in mapping["solutions"]
    )

    ingestion = build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )
    assert ingestion["record_status"] == "has_final_reduced_ebr_results"
    assert ingestion["final_reduced_ebr_result_count"] == 3
    assert ingestion["validation_errors"] == []

    tampered = deepcopy(export)
    tampered["bundles"][0]["cprime_identity_by_kpoint"]["GM"][
        "scoped_representation_evidence_identity"
    ] = "sha256:" + "0" * 64
    rejected = build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=tampered,
        valley_reduced_ebr_mapping=mapping,
    )
    assert rejected["final_reduced_ebr_result_count"] == 2
    assert any(
        "promotion input identity does not match current export bundle" in error
        for error in rejected["validation_errors"]
    )
