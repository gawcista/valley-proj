"""Multi-run database index builder.

Builds a compact cross-run index from existing database ingestion records.
This is a pure offline plumbing module: it does not run analyze-hsp,
valley projection, irrep matching, or reduced EBR solving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = "1.0.0"


def build_database_index(
    records: list[dict[str, Any]],
    *,
    source_files: list[str] | None = None,
) -> dict[str, Any]:
    """Build a compact cross-run index from ingestion records.

    Parameters
    ----------
    records : list[dict]
        List of parsed ``database_ingestion_record.json`` payloads.
    source_files : list[str] or None
        Optional list of source record file paths for provenance.

    Returns
    -------
    dict
        Index payload with ``schema_version``, ``record_count``,
        status/classification aggregates, flattened per-run records,
        and ``validation_errors``.
    """
    errors: list[str] = []
    status_counts: dict[str, int] = {
        "has_ready_ebr_bundles": 0,
        "no_ready_ebr_bundles": 0,
        "invalid_missing_summary": 0,
    }
    total_ready: int = 0
    total_irrep: int = 0
    total_reduced_ebr: int = 0
    total_classification: dict[str, int] = {
        "atomic_compatible": 0,
        "fragile_topology": 0,
        "stable_topology": 0,
    }

    runs: list[dict[str, Any]] = []
    all_irrep_records: list[dict[str, Any]] = []
    all_reduced_ebr_records: list[dict[str, Any]] = []
    all_excluded_ebr_records: list[dict[str, Any]] = []
    ebr_export_status_counts: dict[str, int] = {}

    for idx, record in enumerate(records):
        run_id = f"run_{idx:04d}"
        source = None
        if source_files and idx < len(source_files):
            source = source_files[idx]

        run_entry: dict[str, Any] = {
            "run_id": run_id,
            "record_status": record.get("record_status", "?"),
            "space_group_international": record.get("space_group_international"),
            "space_group_number": record.get("space_group_number"),
            "ready_bundle_count": record.get("ready_bundle_count", 0),
            "valley_irrep_record_count": len(record.get("valley_irrep_records", [])),
            "reduced_ebr_record_count": len(record.get("reduced_ebr_records", [])),
            "reduced_ebr_mapping_status": record.get("reduced_ebr_mapping_status", "?"),
            "reduced_ebr_table_status": record.get("reduced_ebr_table_status", "?"),
            "ebr_export_status": record.get("ebr_export_status", "not_available"),
            "excluded_ebr_record_count": len(record.get("excluded_ebr_records", [])),
        }
        if source is not None:
            run_entry["source"] = source

        record_errors = record.get("validation_errors", [])
        if isinstance(record_errors, list):
            for err in record_errors:
                if source is not None:
                    errors.append(f"{source}: {err}")
                else:
                    errors.append(str(err))

        status = record.get("record_status", "?")
        if status in status_counts:
            status_counts[status] += 1

        total_ready += record.get("ready_bundle_count", 0)
        total_irrep += len(record.get("valley_irrep_records", []))
        total_reduced_ebr += len(record.get("reduced_ebr_records", []))

        counts = record.get("reduced_ebr_classification_counts", {})
        if isinstance(counts, dict):
            total_classification["atomic_compatible"] += counts.get("atomic_compatible", 0)
            total_classification["fragile_topology"] += counts.get("fragile_topology", 0)
            total_classification["stable_topology"] += counts.get("stable_topology", 0)

        # Flatten valley irrep records with run provenance.
        for ir in record.get("valley_irrep_records", []):
            if isinstance(ir, dict):
                flat = {**ir, "run_id": run_id}
                if source is not None:
                    flat["source_record"] = source
                all_irrep_records.append(flat)

        # Flatten reduced EBR records with run provenance.
        for rr in record.get("reduced_ebr_records", []):
            if isinstance(rr, dict):
                flat = {**rr, "run_id": run_id}
                if source is not None:
                    flat["source_record"] = source
                all_reduced_ebr_records.append(flat)

        # Aggregate ebr_export_status counts.
        ebr_status = record.get("ebr_export_status", "not_available")
        ebr_export_status_counts[ebr_status] = ebr_export_status_counts.get(ebr_status, 0) + 1

        # Flatten excluded EBR records with run provenance.
        for er in record.get("excluded_ebr_records", []):
            if isinstance(er, dict):
                flat = {**er, "run_id": run_id}
                if source is not None:
                    flat["source_record"] = source
                all_excluded_ebr_records.append(flat)

        runs.append(run_entry)

    # Validate at least one valid record.
    if not any(r.get("record_status") != "invalid_missing_summary" for r in records):
        errors.append("no valid ingestion record found")

    return {
        "schema_version": _SCHEMA_VERSION,
        "record_count": len(records),
        "source_files": list(source_files) if source_files else [],
        "status_counts": status_counts,
        "ready_bundle_count_total": total_ready,
        "valley_irrep_record_count_total": total_irrep,
        "reduced_ebr_record_count_total": total_reduced_ebr,
        "reduced_ebr_classification_counts_total": total_classification,
        "excluded_ebr_record_count_total": len(all_excluded_ebr_records),
        "ebr_export_status_counts": ebr_export_status_counts,
        "runs": runs,
        "valley_irrep_records": all_irrep_records,
        "reduced_ebr_records": all_reduced_ebr_records,
        "excluded_ebr_records": all_excluded_ebr_records,
        "validation_errors": errors,
    }


def load_database_index_from_files(
    record_paths: list[str],
) -> dict[str, Any]:
    """Load ingestion records from file paths and build a database index.

    Invalid JSON files are recorded as validation errors with a synthetic
    ``invalid_missing_summary`` run entry.  Missing files are recorded
    similarly.
    """
    records: list[dict[str, Any]] = []
    source_files: list[str] = []

    for path in record_paths:
        source_files.append(str(Path(path).resolve()))
        try:
            records.append(json.loads(Path(path).read_text(encoding="utf-8")))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            records.append({
                "schema_version": "?",
                "record_status": "invalid_missing_summary",
                "validation_errors": [f"{path}: {type(exc).__name__}: {exc}"],
            })

    return build_database_index(records, source_files=source_files)
