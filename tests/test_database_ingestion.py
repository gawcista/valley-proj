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
            "ready_for_external_solver": True,
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

    assert record["record_status"] == "has_ready_ebr_bundles"
    assert record["ready_bundle_count"] == 1
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
            "ready_for_external_solver": False,
            "irrep_records_by_kpoint": {},
        }],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle)
    assert record["ready_bundle_count"] == 0
    assert record["valley_irrep_records"] == []


def test_ingestion_record_with_reduced_ebr_mapping():
    """Reduced EBR mapping adds status and classification counts."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": [], "iband": [], "input": {}}
    mapping = {
        "status": "solved_exact",
        "table_status": "loaded",
        "solutions": [
            {"classification": "atomic-compatible-candidate"},
            {"classification": "atomic-compatible-candidate"},
            {"classification": "fragile-topology-candidate"},
        ],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_reduced_ebr_mapping=mapping)
    assert record["reduced_ebr_mapping_status"] == "solved_exact"
    counts = record["reduced_ebr_classification_counts"]
    assert counts["atomic_compatible"] == 2
    assert counts["fragile_topology"] == 1
    assert counts["stable_topology"] == 0


def test_ingestion_record_missing_reduced_ebr_is_not_an_error():
    """Missing reduced EBR mapping is not an error."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(
        valley_summary=summary, valley_reduced_ebr_mapping=None)
    assert record["reduced_ebr_mapping_status"] == "not_available"
    assert record["reduced_ebr_classification_counts"] == {
        "atomic_compatible": 0,
        "fragile_topology": 0,
        "stable_topology": 0,
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
    assert record["ready_bundle_count"] == 0


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
    assert "no_ready_ebr_bundles" in captured


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
        "status": "ready_for_external_solver",
        "bundle_count": 2,
        "excluded_count": 0,
        "bundles": [
            {
                "bundle_id": "b_001", "source_instance_id": "ebr_001",
                "valley": "K_valley",
                "subspace_group_candidate": "P3",
                "ready_for_external_solver": True,
                "irrep_records_by_kpoint": c3_records,
            },
            {
                "bundle_id": "b_002", "source_instance_id": "ebr_002",
                "valley": "Kp_valley",
                "subspace_group_candidate": "P3",
                "ready_for_external_solver": True,
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

    assert record["schema_version"] == "1.2.0"
    assert record["record_status"] == "has_ready_ebr_bundles"
    assert record["ready_bundle_count"] == 2
    assert len(record["valley_irrep_records"]) == 8
    assert record["valley_irrep_records"][0]["valley"] == "K_valley"
    assert record["valley_irrep_records"][0]["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert record["valley_irrep_records"][0]["source_bundle_id"] == "b_001"
    assert record["reduced_ebr_mapping_status"] == "solved_exact"
    assert record["reduced_ebr_table_status"] == "loaded"
    counts = record["reduced_ebr_classification_counts"]
    assert counts["atomic_compatible"] == 2
    assert counts["fragile_topology"] == 0
    assert counts["stable_topology"] == 0
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

def _make_ingestion_record(status="has_ready_ebr_bundles", run_id="run_0000"):
    return {
        "schema_version": "1.2.0",
        "record_status": status,
        "space_group_international": "P321",
        "space_group_number": 150,
        "ready_bundle_count": 2,
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
            "atomic_compatible": 1, "fragile_topology": 0, "stable_topology": 0,
        },
        "reduced_ebr_mapping_status": "solved_exact",
        "reduced_ebr_table_status": "loaded",
        "validation_errors": [],
    }


def test_database_index_builder_two_records():
    """Pure builder with has_ready + no_ready records."""
    from valleyscope.analysis.database_index import build_database_index
    rec1 = _make_ingestion_record("has_ready_ebr_bundles")
    rec2 = _make_ingestion_record("no_ready_ebr_bundles")
    index = build_database_index(
        [rec1, rec2],
        source_files=[
            "/tmp/run_a/database_ingestion_record.json",
            "/tmp/run_b/database_ingestion_record.json",
        ],
    )
    assert index["record_count"] == 2
    assert index["status_counts"]["has_ready_ebr_bundles"] == 1
    assert index["status_counts"]["no_ready_ebr_bundles"] == 1
    assert index["ready_bundle_count_total"] == 4
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
    assert index["reduced_ebr_record_count_total"] == 2


def test_database_index_cli_writes_json(tmp_path):
    """CLI collect-database-index writes database_index.json."""
    from valleyscope.cli import main
    rec1_path = tmp_path / "rec1.json"
    rec1_path.write_text(json.dumps(_make_ingestion_record("has_ready_ebr_bundles")))
    rec2_path = tmp_path / "rec2.json"
    rec2_path.write_text(json.dumps(_make_ingestion_record("no_ready_ebr_bundles")))
    out = tmp_path / "index.json"
    rc = main(["collect-database-index", str(rec1_path), str(rec2_path),
               "-o", str(out)])
    assert rc == 0
    assert out.exists()
    idx = json.loads(out.read_text())
    assert idx["record_count"] == 2
    assert idx["ready_bundle_count_total"] == 4


def test_database_index_cli_invalid_input(tmp_path):
    """CLI returns nonzero on missing input file."""
    from valleyscope.cli import main
    rec1_path = tmp_path / "rec1.json"
    rec1_path.write_text(json.dumps(_make_ingestion_record("has_ready_ebr_bundles")))
    out = tmp_path / "index.json"
    rc = main(["collect-database-index", str(rec1_path), "/nonexistent/path.json",
               "-o", str(out)])
    assert rc != 0
    assert out.exists()
    idx = json.loads(out.read_text())
    assert idx["record_count"] == 2
    assert idx["status_counts"]["has_ready_ebr_bundles"] == 1
    assert idx["status_counts"]["invalid_missing_summary"] == 1
    assert idx["validation_errors"]
    assert "FileNotFoundError" in idx["validation_errors"][0]


def test_database_index_module_no_material_names():
    """Index module must not contain real material names."""
    src = Path("valleyscope/analysis/database_index.py").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src


def test_ingestion_record_includes_excluded_ebr_records():
    """Excluded instances from export bundle become excluded_ebr_records."""
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
                "status": "blocked",
                "ready_for_ebr_decomposition": False,
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
    exclude = record["excluded_ebr_records"]
    assert len(exclude) == 1
    assert exclude[0]["source_instance_id"] == "ebr_001"
    assert exclude[0]["valley"] == "M3_valley"
    assert exclude[0]["exclusion_reasons"] == [
        "spinor_convention_unverified", "low_seed_projector_symmetry",
    ]


def test_excluded_ebr_records_empty_when_not_present():
    """Missing bundle gives empty excluded_ebr_records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(valley_summary=summary)
    assert record["excluded_ebr_records"] == []
    assert record["ebr_export_status"] == "not_available"


def test_database_index_excluded_ebr_records_aggregated():
    """Index aggregates excluded EBR records with run_id provenance."""
    from valleyscope.analysis.database_index import build_database_index
    rec = {
        "schema_version": "1.2.0",
        "record_status": "has_ready_ebr_bundles",
        "ready_bundle_count": 1,
        "valley_irrep_records": [],
        "reduced_ebr_records": [],
        "reduced_ebr_classification_counts": {},
        "reduced_ebr_mapping_status": "?",
        "reduced_ebr_table_status": "?",
        "ebr_export_status": "partial_export",
        "excluded_ebr_records": [
            {"source_instance_id": "ebr_001", "valley": "M3_valley",
             "exclusion_reasons": ["spinor_convention_unverified"]},
        ],
        "validation_errors": [],
    }
    idx = build_database_index([rec])
    assert idx["excluded_ebr_record_count_total"] == 1
    assert idx["ebr_export_status_counts"]["partial_export"] == 1
    assert idx["excluded_ebr_records"][0]["run_id"] == "run_0000"


def test_database_index_excluded_ebr_records_have_source_record():
    """Excluded EBR records carry source_record when source_files provided."""
    from valleyscope.analysis.database_index import build_database_index
    rec = {
        "record_status": "has_ready_ebr_bundles",
        "ready_bundle_count": 0,
        "valley_irrep_records": [],
        "reduced_ebr_records": [],
        "reduced_ebr_classification_counts": {},
        "reduced_ebr_mapping_status": "?",
        "reduced_ebr_table_status": "?",
        "ebr_export_status": "no_bundles",
        "excluded_ebr_records": [
            {"source_instance_id": "ebr_x", "valley": "M1_valley",
             "exclusion_reasons": ["low_seed_overlap"]},
        ],
    }
    idx = build_database_index([rec], source_files=["/tmp/rec.json"])
    er = idx["excluded_ebr_records"][0]
    assert er["run_id"] == "run_0000"
    assert er["source_record"] == "/tmp/rec.json"


def test_ingestion_record_schema_version_is_1_2_0():
    """Ingestion record schema_version is now 1.2.0."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(valley_summary=summary)
    assert record["schema_version"] == "1.2.0"


# -----------------------------------------------------------------------


def test_irrep_records_preserve_generic_fields():
    """Generic irrep provenance fields survive ingestion flattening."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": ["GammaM"], "iband": [1], "input": {}}
    bundle = {
        "bundles": [{
            "bundle_id": "b_001",
            "subspace_group_candidate": "P3",
            "ready_for_external_solver": True,
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


def test_legacy_records_still_ingest_without_generic_fields():
    """Legacy records without generic fields ingest successfully."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": ["GammaM"], "iband": [1], "input": {}}
    bundle = {
        "bundles": [{
            "bundle_id": "b_001",
            "subspace_group_candidate": "P3",
            "ready_for_external_solver": True,
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
    assert record["ready_bundle_count"] == 1
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
        "schema_version": "1.2.0",
        "record_status": "has_ready_ebr_bundles",
        "ready_bundle_count": 1,
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
            "fragile_topology": 0,
            "stable_topology": 0,
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
