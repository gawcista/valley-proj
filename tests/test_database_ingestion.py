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
            "subspace_group_candidate": "C3_like",
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
    assert r["subspace_group_candidate"] == "C3_like"
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


def test_ingestion_record_with_reduced_ebr_bundle_and_mapping():
    """Synthetic C3-like ingestion record with ready bundles, reduced EBR
    mapping, and classification counts."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": ["GammaM", "KM"], "iband": [101, 102], "input": {}}
    bundle = {
        "status": "ready_for_external_solver",
        "bundle_count": 2,
        "excluded_count": 0,
        "bundles": [
            {
                "bundle_id": "b_001", "valley": "K_valley",
                "subspace_group_candidate": "C3_like",
                "ready_for_external_solver": True,
                "irrep_records_by_kpoint": {
                    "GammaM": [{"valley_irrep_key": "C3_spinor_phase_+1/2"}],
                    "KM": [{"valley_irrep_key": "C3_spinor_phase_+1/6"},
                           {"valley_irrep_key": "C3_spinor_phase_-1/6"}],
                },
            },
            {
                "bundle_id": "b_002", "valley": "Kp_valley",
                "subspace_group_candidate": "C3_like",
                "ready_for_external_solver": True,
                "irrep_records_by_kpoint": {
                    "GammaM": [{"valley_irrep_key": "C3_spinor_phase_+1/2"}],
                    "KM": [{"valley_irrep_key": "C3_spinor_phase_+1/6"},
                           {"valley_irrep_key": "C3_spinor_phase_-1/6"}],
                },
            },
        ],
        "excluded_instances": [],
    }
    mapping = {
        "status": "solved_exact", "table_status": "loaded",
        "solutions": [
            {"classification": "atomic-compatible-candidate"},
            {"classification": "atomic-compatible-candidate"},
        ],
    }
    record = build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=bundle,
        valley_reduced_ebr_mapping=mapping,
    )
    assert record["record_status"] == "has_ready_ebr_bundles"
    assert record["ready_bundle_count"] == 2
    assert len(record["valley_irrep_records"]) == 6  # 3 per bundle, 2 bundles
    assert record["reduced_ebr_mapping_status"] == "solved_exact"
    assert record["reduced_ebr_table_status"] == "loaded"
    counts = record["reduced_ebr_classification_counts"]
    assert counts["atomic_compatible"] == 2
    assert counts["fragile_topology"] == 0
    assert counts["stable_topology"] == 0
    # source_files records public output file paths
    assert isinstance(record.get("source_files"), dict)


# -----------------------------------------------------------------------
