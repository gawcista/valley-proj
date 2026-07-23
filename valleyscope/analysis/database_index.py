"""Multi-run database index builder.

Builds a compact cross-run index from existing database ingestion records.
This is a pure offline plumbing module: it does not run analyze-hsp,
valley projection, irrep matching, or reduced EBR solving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = "1.1.0"


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
        "has_final_reduced_ebr_results": 0,
        "has_reduced_table_validation_candidates": 0,
        "no_reduced_ebr_input": 0,
        "invalid_missing_summary": 0,
    }
    total_validation_candidates: int = 0
    total_final_results: int = 0
    total_final_mapping_excluded: int = 0
    total_input_excluded: int = 0
    total_irrep: int = 0
    total_classification: dict[str, int] = {
        "atomic_compatible": 0,
        "in_integer_span_no_nonnegative_witness": 0,
        "outside_integer_span": 0,
        "indeterminate_truncated": 0,
    }

    runs: list[dict[str, Any]] = []
    all_irrep_records: list[dict[str, Any]] = []
    all_reduced_ebr_records: list[dict[str, Any]] = []
    all_input_excluded_ebr_records: list[dict[str, Any]] = []
    all_final_mapping_excluded_records: list[dict[str, Any]] = []
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
            "reduced_table_validation_candidate_bundle_count": record.get(
                "reduced_table_validation_candidate_bundle_count", 0
            ),
            "final_reduced_ebr_result_count": record.get(
                "final_reduced_ebr_result_count", 0
            ),
            "final_mapping_excluded_bundle_count": record.get(
                "final_mapping_excluded_bundle_count", 0
            ),
            "input_excluded_instance_count": record.get(
                "input_excluded_instance_count", 0
            ),
            "valley_irrep_record_count": len(record.get("valley_irrep_records", [])),
            "reduced_ebr_mapping_status": record.get("reduced_ebr_mapping_status", "?"),
            "reduced_ebr_table_status": record.get("reduced_ebr_table_status", "?"),
            "ebr_export_status": record.get("ebr_export_status", "not_available"),
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

        total_validation_candidates += record.get(
            "reduced_table_validation_candidate_bundle_count", 0
        )
        total_final_results += record.get(
            "final_reduced_ebr_result_count", 0
        )
        total_final_mapping_excluded += record.get(
            "final_mapping_excluded_bundle_count", 0
        )
        total_input_excluded += record.get(
            "input_excluded_instance_count", 0
        )
        total_irrep += len(record.get("valley_irrep_records", []))

        counts = record.get("reduced_ebr_classification_counts", {})
        if isinstance(counts, dict):
            for classification in total_classification:
                total_classification[classification] += counts.get(
                    classification, 0
                )

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

        # Flatten input-stage exclusions with run provenance.
        for er in record.get("input_excluded_ebr_records", []):
            if isinstance(er, dict):
                flat = {**er, "run_id": run_id}
                if source is not None:
                    flat["source_record"] = source
                all_input_excluded_ebr_records.append(flat)

        # Flatten final mapping exclusions with run provenance.
        for er in record.get("final_mapping_excluded_records", []):
            if isinstance(er, dict):
                flat = {**er, "run_id": run_id}
                if source is not None:
                    flat["source_record"] = source
                all_final_mapping_excluded_records.append(flat)

        runs.append(run_entry)

    # Validate at least one valid record.
    if not any(r.get("record_status") != "invalid_missing_summary" for r in records):
        errors.append("no valid ingestion record found")

    return {
        "schema_version": _SCHEMA_VERSION,
        "record_count": len(records),
        "source_files": list(source_files) if source_files else [],
        "status_counts": status_counts,
        "reduced_table_validation_candidate_bundle_count_total": (
            total_validation_candidates
        ),
        "final_reduced_ebr_result_count_total": total_final_results,
        "final_mapping_excluded_bundle_count_total": (
            total_final_mapping_excluded
        ),
        "input_excluded_instance_count_total": total_input_excluded,
        "valley_irrep_record_count_total": total_irrep,
        "reduced_ebr_classification_counts_total": total_classification,
        "ebr_export_status_counts": ebr_export_status_counts,
        "runs": runs,
        "valley_irrep_records": all_irrep_records,
        "reduced_ebr_records": all_reduced_ebr_records,
        "input_excluded_ebr_records": all_input_excluded_ebr_records,
        "final_mapping_excluded_records": all_final_mapping_excluded_records,
        "validation_errors": errors,
    }


def load_database_index_from_files(
    record_paths: list[str],
) -> dict[str, Any]:
    """Load ingestion records from file paths and build a database index.

    This compatibility entry point delegates to the mixed-input loader.
    """
    return load_database_index_from_inputs(record_paths)


def load_database_index_from_inputs(
    input_paths: list[str],
) -> dict[str, Any]:
    """Load explicit ingestion-record files or analyze-hsp output directories.

    Invalid JSON files are recorded as validation errors with a synthetic
    ``invalid_missing_summary`` run entry. Missing inputs and malformed public
    output directories are recorded similarly. Duplicate resolved inputs are
    rejected without adding a second run.
    """
    records: list[dict[str, Any]] = []
    source_inputs: list[str] = []
    loader_errors: list[str] = []
    seen: set[Path] = set()

    for raw_path in input_paths:
        resolved = Path(raw_path).resolve()
        if resolved in seen:
            loader_errors.append(f"duplicate resolved input: {resolved}")
            continue
        seen.add(resolved)
        source_inputs.append(str(resolved))
        try:
            if resolved.is_dir():
                from valleyscope.analysis.database_ingestion_record import (
                    load_database_ingestion_record_from_directory,
                )

                record = load_database_ingestion_record_from_directory(
                    resolved
                )
            else:
                record = json.loads(resolved.read_text(encoding="utf-8"))
                if not isinstance(record, dict):
                    raise ValueError(
                        "ingestion record JSON must be an object"
                    )
            records.append(record)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            records.append({
                "schema_version": "?",
                "record_status": "invalid_missing_summary",
                "validation_errors": [f"{type(exc).__name__}: {exc}"],
            })

    index = build_database_index(records, source_files=source_inputs)
    index["validation_errors"].extend(loader_errors)
    return index
