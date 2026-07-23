"""Multi-run database index builder.

Builds a compact cross-run index from existing database ingestion records.
This is a pure offline plumbing module: it does not run analyze-hsp,
valley projection, irrep matching, or reduced EBR solving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = "1.2.0"
_RECORD_STATUSES = {
    "has_final_reduced_ebr_results",
    "has_reduced_table_validation_candidates",
    "no_reduced_ebr_input",
    "invalid_missing_summary",
}
_COUNT_FIELDS = (
    "reduced_table_validation_candidate_bundle_count",
    "final_reduced_ebr_result_count",
    "final_mapping_excluded_bundle_count",
    "input_excluded_instance_count",
)
_COLLECTION_FIELDS = (
    "valley_irrep_records",
    "reduced_ebr_records",
    "input_excluded_ebr_records",
    "final_mapping_excluded_records",
)
_CLASSIFICATION_FIELDS = (
    "atomic_compatible",
    "in_integer_span_no_nonnegative_witness",
    "outside_integer_span",
    "indeterminate_truncated",
)
_SOURCE_INPUT_KINDS = {
    "ingestion_record_file",
    "analyze_output_directory",
}


def _is_nonnegative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


def _validate_source_inputs(
    source_inputs: list[dict[str, str]],
    *,
    record_count: int,
) -> list[dict[str, str]]:
    if len(source_inputs) != record_count:
        raise ValueError(
            "source_inputs must contain exactly one entry per ingestion record"
        )

    validated: list[dict[str, str]] = []
    for idx, source_input in enumerate(source_inputs):
        if not isinstance(source_input, dict):
            raise ValueError(f"source_inputs[{idx}] must be an object")
        kind = source_input.get("kind")
        path = source_input.get("path")
        if kind not in _SOURCE_INPUT_KINDS:
            raise ValueError(
                f"source_inputs[{idx}].kind is not recognized: {kind!r}"
            )
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                f"source_inputs[{idx}].path must be a nonempty string"
            )
        validated.append({"kind": kind, "path": path})
    return validated


def _validate_ingestion_record(record: object) -> list[str]:
    if not isinstance(record, dict):
        return ["ingestion record JSON must be an object"]

    errors: list[str] = []
    schema_version = record.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        errors.append("schema_version must be a nonempty string")

    record_status = record.get("record_status")
    if record_status not in _RECORD_STATUSES:
        errors.append(f"record_status is not recognized: {record_status!r}")

    if not isinstance(record.get("validation_errors"), list):
        errors.append("validation_errors must be a list")

    if (
        "ebr_export_status" in record
        and not isinstance(record["ebr_export_status"], str)
    ):
        errors.append("ebr_export_status must be a string")

    for field in _COUNT_FIELDS:
        if not _is_nonnegative_integer(record.get(field)):
            errors.append(f"{field} must be a nonnegative integer")

    for field in _COLLECTION_FIELDS:
        if not isinstance(record.get(field), list):
            errors.append(f"{field} must be a list")

    counts = record.get("reduced_ebr_classification_counts")
    if not isinstance(counts, dict):
        errors.append("reduced_ebr_classification_counts must be an object")
    else:
        for field, value in counts.items():
            if not _is_nonnegative_integer(value):
                errors.append(
                    "reduced_ebr_classification_counts."
                    f"{field} must be a nonnegative integer"
                )
    return errors


def _invalid_ingestion_record(errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "?",
        "record_status": "invalid_missing_summary",
        "reduced_table_validation_candidate_bundle_count": 0,
        "final_reduced_ebr_result_count": 0,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 0,
        "valley_irrep_records": [],
        "reduced_ebr_records": [],
        "input_excluded_ebr_records": [],
        "final_mapping_excluded_records": [],
        "reduced_ebr_classification_counts": {
            field: 0 for field in _CLASSIFICATION_FIELDS
        },
        "validation_errors": errors,
    }


def _source_provenance(
    source_input: dict[str, str] | None,
) -> dict[str, Any]:
    if source_input is None:
        return {}
    provenance: dict[str, Any] = {
        "source_input": dict(source_input),
    }
    if source_input["kind"] == "ingestion_record_file":
        provenance["source_record"] = source_input["path"]
    return provenance


def _flatten_with_source(
    record: dict[str, Any],
    *,
    run_id: str,
    source_input: dict[str, str] | None,
) -> dict[str, Any]:
    flattened = {
        **record,
        "run_id": run_id,
        **_source_provenance(source_input),
    }
    if (
        source_input is not None
        and source_input["kind"] == "analyze_output_directory"
    ):
        flattened.pop("source_record", None)
    return flattened


def build_database_index(
    records: list[dict[str, Any]],
    *,
    source_files: list[str] | None = None,
    source_inputs: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a compact cross-run index from ingestion records.

    Parameters
    ----------
    records : list[dict]
        List of parsed ``database_ingestion_record.json`` payloads.
    source_files : list[str] or None
        Compatibility list of source record file paths for provenance.
    source_inputs : list[dict] or None
        Ordered typed source inputs. When omitted, ``source_files`` entries
        are treated as ``ingestion_record_file`` inputs.

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
    typed_source_inputs = (
        _validate_source_inputs(source_inputs, record_count=len(records))
        if source_inputs is not None
        else [
            {"kind": "ingestion_record_file", "path": path}
            for path in (source_files or [])
        ]
    )
    record_source_files = [
        source_input["path"]
        for source_input in typed_source_inputs
        if source_input.get("kind") == "ingestion_record_file"
    ]

    validated_records: list[dict[str, Any]] = []
    for record in records:
        validation_errors = _validate_ingestion_record(record)
        validated_records.append(
            _invalid_ingestion_record(validation_errors)
            if validation_errors
            else record
        )

    for idx, record in enumerate(validated_records):
        run_id = f"run_{idx:04d}"
        source_input = (
            typed_source_inputs[idx]
            if idx < len(typed_source_inputs)
            else None
        )
        source_path = (
            source_input["path"] if source_input is not None else None
        )

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
        if source_input is not None:
            run_entry["source_input"] = dict(source_input)
            if source_input["kind"] == "ingestion_record_file":
                run_entry["source"] = source_input["path"]

        record_errors = record.get("validation_errors", [])
        if isinstance(record_errors, list):
            for err in record_errors:
                if source_path is not None:
                    errors.append(f"{source_path}: {err}")
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
                flat = _flatten_with_source(
                    ir,
                    run_id=run_id,
                    source_input=source_input,
                )
                all_irrep_records.append(flat)

        # Flatten reduced EBR records with run provenance.
        for rr in record.get("reduced_ebr_records", []):
            if isinstance(rr, dict):
                flat = _flatten_with_source(
                    rr,
                    run_id=run_id,
                    source_input=source_input,
                )
                all_reduced_ebr_records.append(flat)

        # Aggregate ebr_export_status counts.
        ebr_status = record.get("ebr_export_status", "not_available")
        ebr_export_status_counts[ebr_status] = ebr_export_status_counts.get(ebr_status, 0) + 1

        # Flatten input-stage exclusions with run provenance.
        for er in record.get("input_excluded_ebr_records", []):
            if isinstance(er, dict):
                flat = _flatten_with_source(
                    er,
                    run_id=run_id,
                    source_input=source_input,
                )
                all_input_excluded_ebr_records.append(flat)

        # Flatten final mapping exclusions with run provenance.
        for er in record.get("final_mapping_excluded_records", []):
            if isinstance(er, dict):
                flat = _flatten_with_source(
                    er,
                    run_id=run_id,
                    source_input=source_input,
                )
                all_final_mapping_excluded_records.append(flat)

        runs.append(run_entry)

    # Validate at least one valid record.
    if not any(
        record.get("record_status") != "invalid_missing_summary"
        for record in validated_records
    ):
        errors.append("no valid ingestion record found")

    return {
        "schema_version": _SCHEMA_VERSION,
        "record_count": len(validated_records),
        "source_files": record_source_files,
        "source_inputs": typed_source_inputs,
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
    source_inputs: list[dict[str, str]] = []
    loader_errors: list[str] = []
    seen: set[Path] = set()

    for raw_path in input_paths:
        resolved = Path(raw_path).resolve()
        if resolved in seen:
            loader_errors.append(f"duplicate resolved input: {resolved}")
            continue
        seen.add(resolved)
        source_inputs.append({
            "kind": (
                "analyze_output_directory"
                if resolved.is_dir()
                else "ingestion_record_file"
            ),
            "path": str(resolved),
        })
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
            records.append(_invalid_ingestion_record([
                f"{type(exc).__name__}: {exc}"
            ]))

    index = build_database_index(records, source_inputs=source_inputs)
    index["validation_errors"].extend(loader_errors)
    return index
