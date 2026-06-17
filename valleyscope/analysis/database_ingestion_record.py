"""High-throughput database ingestion record builder.

Builds a compact, validated ingestion record from existing public per-run
outputs.  This is an offline collector/validator path — it does not add
new default ``analyze_hsp`` output files.

Ingestion record fields are a flattened subset of public schema fields
suitable for database indexing.  No debug payloads are copied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "1.2.0"


def build_database_ingestion_record(
    *,
    valley_summary: dict[str, Any] | None = None,
    valley_ebr_export_bundle: dict[str, Any] | None = None,
    valley_reduced_ebr_mapping: dict[str, Any] | None = None,
    source_files: dict[str, str] | None = None,
    output_profile: str = "standard",
) -> dict[str, Any]:
    """Build a validated database ingestion record.

    Parameters
    ----------
    valley_summary : dict or None
        Loaded ``valley_summary.json`` payload.  Required.
    valley_ebr_export_bundle : dict or None
        Loaded ``valley_ebr_export_bundle.json`` payload.  Optional.
    valley_reduced_ebr_mapping : dict or None
        Loaded ``valley_reduced_ebr_mapping.json`` payload.  Optional.
    source_files : dict or None
        Optional map of logical name to file path for provenance.
    output_profile : str
        The output profile used for this run.

    Returns
    -------
    dict
        Ingestion record with at minimum ``schema_version``,
        ``record_status``, and ``validation_errors``.
    """
    errors: list[str] = []
    record: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "source_files": dict(source_files) if source_files else {},
        "output_profile": output_profile,
        "reduced_ebr_classification_counts": _empty_classification_counts(),
        "validation_errors": errors,
    }

    # --- valley_summary (required) ---
    if valley_summary is None:
        errors.append("valley_summary.json is missing")
        record["record_status"] = "invalid_missing_summary"
        record["summary_status"] = "missing"
        return record

    record["summary_status"] = "present"

    # Summary-level status fields.
    record["target_kpoints"] = valley_summary.get("target_kpoints", [])
    record["iband"] = valley_summary.get("iband", [])
    sym = valley_summary.get("symmetry_analysis", {}) if isinstance(valley_summary.get("symmetry_analysis"), dict) else {}
    record["space_group_international"] = sym.get("international")
    record["space_group_number"] = sym.get("spacegroup_number")
    record["spinor_convention_verified"] = valley_summary.get("input", {}).get("spinor_convention_verified", False)

    # --- valley_ebr_export_bundle (optional) ---
    ready_count = 0
    excluded_count = 0
    valley_irrep_records: list[dict[str, Any]] = []

    if valley_ebr_export_bundle is not None:
        bundles = valley_ebr_export_bundle.get("bundles", [])
        if isinstance(bundles, list):
            for bundle in bundles:
                if not isinstance(bundle, dict):
                    continue
                if bundle.get("ready_for_external_solver") is True:
                    ready_count += 1
                    _extract_irrep_records(bundle, valley_irrep_records)
                else:
                    excluded_count += 1
        excluded_count += int(valley_ebr_export_bundle.get("excluded_count", 0) or 0)
    else:
        excluded_count = 0  # No bundle file present

    record["ready_bundle_count"] = ready_count
    record["excluded_bundle_count"] = excluded_count
    record["valley_irrep_records"] = valley_irrep_records

    # --- export-bundle status and excluded EBR records ---
    if valley_ebr_export_bundle is not None:
        record["ebr_export_status"] = valley_ebr_export_bundle.get("status", "?")
        record["ebr_export_interpretation"] = valley_ebr_export_bundle.get("interpretation", "")
        excluded_instances = valley_ebr_export_bundle.get("excluded_instances", [])
        if isinstance(excluded_instances, list):
            record["excluded_ebr_records"] = _compact_excluded_records(excluded_instances)
        else:
            record["excluded_ebr_records"] = []
    else:
        record["ebr_export_status"] = "not_available"
        record["ebr_export_interpretation"] = ""
        record["excluded_ebr_records"] = []

    # --- valley_reduced_ebr_mapping (optional) ---
    if valley_reduced_ebr_mapping is not None:
        record["reduced_ebr_mapping_status"] = valley_reduced_ebr_mapping.get("status", "?")
        record["reduced_ebr_table_status"] = valley_reduced_ebr_mapping.get("table_status", "?")
        solutions = valley_reduced_ebr_mapping.get("solutions", [])
        if isinstance(solutions, list):
            atomic = sum(1 for s in solutions if isinstance(s, dict)
                         and s.get("classification") == "atomic-compatible-candidate")
            fragile = sum(1 for s in solutions if isinstance(s, dict)
                          and s.get("classification") == "fragile-topology-candidate")
            stable = sum(1 for s in solutions if isinstance(s, dict)
                         and s.get("classification") == "stable-topology-candidate")
            record["reduced_ebr_classification_counts"] = {
                "atomic_compatible": atomic,
                "fragile_topology": fragile,
                "stable_topology": stable,
            }
        # --- per-bundle reduced EBR records (compact public fields) ---
        if isinstance(solutions, list):
            reduced_ebr_records: list[dict[str, Any]] = []
            for s in solutions:
                if not isinstance(s, dict):
                    continue
                rec: dict[str, Any] = {
                    "bundle_id": s.get("bundle_id", "?"),
                    "valley": s.get("valley", "?"),
                    "subspace_group_candidate": s.get("subspace_group_candidate", "?"),
                    "status": s.get("status", "?"),
                    "classification": s.get("classification", "?"),
                    "integer_span_status": s.get("integer_span_status", "?"),
                    "nonnegative_solution_status": s.get("nonnegative_solution_status", "?"),
                    "irrep_vector": s.get("irrep_vector", []),
                }
                if "ebr_decomposition" in s:
                    rec["ebr_decomposition"] = s["ebr_decomposition"]
                if "integer_solution" in s:
                    rec["integer_solution"] = s["integer_solution"]
                if "search_status" in s:
                    rec["search_status"] = s["search_status"]
                reduced_ebr_records.append(rec)
            record["reduced_ebr_records"] = reduced_ebr_records
        else:
            record["reduced_ebr_records"] = []
    else:
        record["reduced_ebr_mapping_status"] = "not_available"
        record["reduced_ebr_table_status"] = "not_available"
        record["reduced_ebr_records"] = []

    # --- Status ---
    if ready_count > 0:
        record["record_status"] = "has_ready_ebr_bundles"
    else:
        record["record_status"] = "no_ready_ebr_bundles"

    return record


def load_database_ingestion_record_from_directory(
    run_dir: str | Path,
    *,
    output_profile: str = "standard",
) -> dict[str, Any]:
    """Load public output files from a run directory and build an ingestion record.

    Parameters
    ----------
    run_dir : str or Path
        Path to the analyze_hsp output directory.
    output_profile : str
        The output profile used for this run.

    Returns
    -------
    dict
        Ingestion record.
    """
    root = Path(run_dir)
    source_files: dict[str, str] = {}

    # valley_summary.json (required)
    summary_path = root / "valley_summary.json"
    if not summary_path.is_file():
        return build_database_ingestion_record(output_profile=output_profile)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_files["valley_summary"] = str(summary_path.resolve())

    # valley_ebr_export_bundle.json (optional)
    bundle = None
    bundle_path = root / "valley_ebr_export_bundle.json"
    if bundle_path.is_file():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        source_files["valley_ebr_export_bundle"] = str(bundle_path.resolve())

    # valley_reduced_ebr_mapping.json (optional)
    mapping = None
    mapping_path = root / "valley_reduced_ebr_mapping.json"
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        source_files["valley_reduced_ebr_mapping"] = str(mapping_path.resolve())

    return build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=bundle,
        valley_reduced_ebr_mapping=mapping,
        source_files=source_files,
        output_profile=output_profile,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _extract_irrep_records(
    bundle: dict[str, Any],
    out: list[dict[str, Any]],
) -> None:
    """Flatten trusted irrep_records_by_kpoint into a list of per-record dicts."""
    source_bundle_id = bundle.get("bundle_id", "")
    source_instance_id = bundle.get("source_instance_id", "")
    records_by_kp = bundle.get("irrep_records_by_kpoint", {})
    if not isinstance(records_by_kp, dict):
        return

    for kpoint, records in records_by_kp.items():
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            entry: dict[str, Any] = {
                "kpoint": kpoint,
                "valley": rec.get("valley", ""),
                "subspace_group_candidate": bundle.get("subspace_group_candidate", ""),
                "operation_id": rec.get("operation_id"),
                "operation_order": rec.get("operation_order"),
                "matched_irrep": rec.get("matched_irrep"),
                "eigenphases": rec.get("eigenphases", []),
                "character": rec.get("character"),
                "workflow_path": rec.get("workflow_path", ""),
                "readiness_level": rec.get("readiness_level", ""),
                "source": rec.get("source", ""),
                "source_bundle_id": source_bundle_id,
                "source_instance_id": source_instance_id,
            }
            # Preserve generic irrep provenance fields when present.
            for key in (
                "irrep_multiplicity",
                "matching_strategy",
                "subspace_space_group",
                "legacy_subspace_group_candidate",
                "valley_preserving_operation_ids",
                "source_operation_map",
            ):
                if key in rec:
                    entry[key] = rec[key]
            out.append(entry)


def _compact_excluded_records(
    excluded_instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract compact public fields from excluded_instances."""
    records: list[dict[str, Any]] = []
    for exc in excluded_instances:
        if not isinstance(exc, dict):
            continue
        records.append({
            "source_instance_id": exc.get("source_instance_id", "?"),
            "valley": exc.get("valley", "?"),
            "subspace_group_candidate": exc.get("subspace_group_candidate", "?"),
            "status": exc.get("status", "?"),
            "ready_for_ebr_decomposition": exc.get("ready_for_ebr_decomposition", False),
            "exclusion_reasons": exc.get("exclusion_reasons", []),
        })
    return records


def _empty_classification_counts() -> dict[str, int]:
    return {
        "atomic_compatible": 0,
        "fragile_topology": 0,
        "stable_topology": 0,
    }
