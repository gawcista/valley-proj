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

from valleyscope.analysis.unitary_provenance import (
    unitary_bundle_claims_time_reversal_completion,
    unitary_bundle_claims_valley_sewing_completion,
    validate_serialized_valley_sewing_completed_unitary_bundle,
    validate_unitary_bundle_provenance,
)
from valleyscope.analysis.promotion_identity import (
    build_promotion_input_identity,
    merge_table_input_provenance,
    table_input_for_bundle,
)
from valleyscope.io.wavefunction_convention import valid_sha256_identity

_SCHEMA_VERSION = "2.0.0"

_PROMOTION_REQUIRED_PASSED_CHECKS = frozenset({
    "table_provenance_check",
    "table_setting_check",
    "sg_symbol_check",
    "sg_number_check",
    "certificate_check",
    "cprime_identity_check",
    "certificate_consistency_check",
    "cert_sg_consistency_check",
    "affine_setting_check",
    "hall_setting_check",
    "spin_convention_check",
    "problem_kind_check",
    "hsp_basis_check",
    "irrep_basis_check",
    "completion_provenance_check",
})

_SOLUTION_BUNDLE_BINDING_FIELDS = (
    "problem_kind",
    "physical_object_kind",
    "valley",
    "valley_orbit",
    "subspace_group_candidate",
    "subspace_space_group",
    "expected_hsps",
    "required_source_hsp_labels",
    "covered_source_hsp_labels",
    "source_hsp_to_sampled_kpoint",
    "independent_source_hsp_to_sampled_kpoint",
    "observed_source_hsp_to_sampled_kpoint",
    "unitary_vector_construction",
    "unitary_irrep_completion_records_by_hsp",
    "unitary_valley_irreps",
    "time_reversal",
    "certificate_identity",
    "cprime_identity_by_kpoint",
)
_SOLUTION_BUNDLE_BINDING_DEFAULTS = {
    "problem_kind": "unitary_valley_reduced_ebr",
    "physical_object_kind": "unitary_valley_projected_subspace",
    "valley": "",
    "valley_orbit": [],
    "subspace_group_candidate": "",
    "subspace_space_group": {},
    "expected_hsps": [],
    "required_source_hsp_labels": [],
    "covered_source_hsp_labels": [],
    "source_hsp_to_sampled_kpoint": {},
    "independent_source_hsp_to_sampled_kpoint": {},
    "observed_source_hsp_to_sampled_kpoint": {},
    "unitary_vector_construction": {},
    "unitary_irrep_completion_records_by_hsp": {},
    "unitary_valley_irreps": {},
    "time_reversal": {},
    "certificate_identity": {},
    "cprime_identity_by_kpoint": {},
}

def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


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
        Loaded standalone ``valley_ebr_export_bundle.json`` payload.  Optional.
    valley_reduced_ebr_mapping : dict or None
        Loaded standalone ``valley_reduced_ebr_mapping.json`` payload.  Optional.
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
        "reduced_table_validation_candidate_bundle_count": 0,
        "final_reduced_ebr_result_count": 0,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 0,
        "valley_irrep_records": [],
        "reduced_ebr_records": [],
        "input_excluded_ebr_records": [],
        "final_mapping_excluded_records": [],
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
    cprime = (
        valley_summary.get("cprime", {})
        if isinstance(valley_summary.get("cprime"), dict)
        else {}
    )
    source_basis = (
        cprime.get("spinor_source_basis", {})
        if isinstance(cprime.get("spinor_source_basis"), dict)
        else {}
    )
    record["spinor_source_basis_status"] = source_basis.get(
        "status", "not_evaluated"
    )
    record["spinor_source_basis_certificate_identity"] = source_basis.get(
        "identity"
    )
    record["cprime_acceptance_matrix"] = cprime.get(
        "acceptance_matrix", []
    )

    # --- valley_ebr_export_bundle (optional) ---
    validation_candidate_count = 0
    valley_irrep_records: list[dict[str, Any]] = []

    if valley_ebr_export_bundle is not None:
        bundles = valley_ebr_export_bundle.get("bundles", [])
        if isinstance(bundles, list):
            has_tr_completed_unitary = any(
                isinstance(bundle, dict)
                and unitary_bundle_claims_time_reversal_completion(bundle)
                for bundle in bundles
            )
            for bundle in bundles:
                if not isinstance(bundle, dict):
                    continue
                is_validation_candidate = (
                    bundle.get("ready_for_reduced_table_validation") is True
                )
                if is_validation_candidate:
                    cprime_error = _validate_bundle_cprime_against_summary(
                        bundle=bundle,
                        source_basis=source_basis,
                        acceptance_matrix=record[
                            "cprime_acceptance_matrix"
                        ],
                    )
                    if cprime_error is not None:
                        errors.append(cprime_error)
                        continue
                    validation_candidate_count += 1
                    if (
                        unitary_bundle_claims_valley_sewing_completion(bundle)
                        and not _serialized_sewing_bundle_has_strict_promotion(
                            bundle, valley_reduced_ebr_mapping
                        )
                    ):
                        continue
                    # Validation candidates contain trusted valley-preserving
                    # irreps; preserve them independently of EBR readiness.
                    binding_error = _extract_irrep_records(
                        bundle,
                        valley_irrep_records,
                        allow_joint_fallback=not has_tr_completed_unitary,
                    )
                    if binding_error is not None:
                        errors.append(binding_error)
    record["reduced_table_validation_candidate_bundle_count"] = (
        validation_candidate_count
    )
    record["valley_irrep_records"] = valley_irrep_records

    # --- export-bundle status and excluded EBR records ---
    if valley_ebr_export_bundle is not None:
        record["ebr_export_status"] = valley_ebr_export_bundle.get("status", "?")
        record["ebr_export_interpretation"] = valley_ebr_export_bundle.get("interpretation", "")
        excluded_instances = valley_ebr_export_bundle.get("excluded_instances", [])
        if isinstance(excluded_instances, list):
            input_excluded = _compact_excluded_records(excluded_instances)
        else:
            input_excluded = []
        record["input_excluded_ebr_records"] = input_excluded
        record["input_excluded_instance_count"] = len(input_excluded)
    else:
        record["ebr_export_status"] = "not_available"
        record["ebr_export_interpretation"] = ""

    # --- valley_reduced_ebr_mapping (optional) ---
    if valley_reduced_ebr_mapping is not None:
        record["reduced_ebr_mapping_status"] = valley_reduced_ebr_mapping.get("status", "?")
        record["reduced_ebr_table_status"] = valley_reduced_ebr_mapping.get("table_status", "?")
        reduced_ebr_input = valley_reduced_ebr_mapping.get("reduced_ebr_input")
        if isinstance(reduced_ebr_input, dict):
            record["reduced_ebr_input"] = dict(reduced_ebr_input)
        solutions = valley_reduced_ebr_mapping.get("solutions", [])
        valid_solutions = _authoritative_mapping_solutions(
            solutions=solutions,
            valley_ebr_export_bundle=valley_ebr_export_bundle,
            reduced_ebr_input=reduced_ebr_input,
            errors=errors,
        )
        record["final_reduced_ebr_result_count"] = len(valid_solutions)
        mapping_excluded = valley_reduced_ebr_mapping.get(
            "excluded_bundles", []
        )
        if isinstance(mapping_excluded, list):
            compact_mapping_excluded = _compact_mapping_exclusions(
                mapping_excluded
            )
        else:
            compact_mapping_excluded = []
        record["final_mapping_excluded_records"] = compact_mapping_excluded
        record["final_mapping_excluded_bundle_count"] = len(
            compact_mapping_excluded
        )
        if isinstance(solutions, list):
            atomic = sum(
                1 for s in valid_solutions
                if s.get("classification") == "atomic-compatible-candidate"
            )
            no_witness = sum(
                1 for s in valid_solutions
                if s.get("classification")
                == "in_integer_span_no_nonnegative_witness"
            )
            outside = sum(
                1 for s in valid_solutions
                if s.get("classification") == "outside_integer_span"
            )
            truncated = sum(
                1 for s in valid_solutions
                if s.get("classification") == "indeterminate_truncated"
            )
            record["reduced_ebr_classification_counts"] = {
                "atomic_compatible": atomic,
                "in_integer_span_no_nonnegative_witness": no_witness,
                "outside_integer_span": outside,
                "indeterminate_truncated": truncated,
            }
        # --- per-bundle reduced EBR records (compact public fields) ---
        if isinstance(solutions, list):
            reduced_ebr_records: list[dict[str, Any]] = []
            for s in valid_solutions:
                rec: dict[str, Any] = {
                    "bundle_id": s.get("bundle_id", "?"),
                    "valley": s.get("valley", "?"),
                    "subspace_group_candidate": s.get("subspace_group_candidate", "?"),
                    "subspace_space_group": s.get("subspace_space_group", {}),
                    "status": s.get("status", "?"),
                    "classification": s.get("classification", "?"),
                    "integer_span_status": s.get("integer_span_status", "?"),
                    "nonnegative_solution_status": s.get("nonnegative_solution_status", "?"),
                    "irrep_vector": s.get("irrep_vector", []),
                }
                if "problem_kind" in s:
                    rec["problem_kind"] = s["problem_kind"]
                if "physical_object_kind" in s:
                    rec["physical_object_kind"] = s[
                        "physical_object_kind"
                    ]
                if "valley_orbit" in s:
                    rec["valley_orbit"] = s["valley_orbit"]
                if "unitary_valley_irreps" in s:
                    rec["unitary_valley_irreps"] = s["unitary_valley_irreps"]
                if "time_reversal" in s:
                    rec["time_reversal"] = s["time_reversal"]
                if "unitary_irrep_completion_records_by_hsp" in s:
                    rec["unitary_irrep_completion_records_by_hsp"] = s[
                        "unitary_irrep_completion_records_by_hsp"
                    ]
                for key in (
                    "expected_hsps",
                    "required_source_hsp_labels",
                    "covered_source_hsp_labels",
                    "source_hsp_to_sampled_kpoint",
                    "independent_source_hsp_to_sampled_kpoint",
                    "observed_source_hsp_to_sampled_kpoint",
                    "unitary_vector_construction",
                ):
                    if key in s:
                        rec[key] = s[key]
                if "ebr_decomposition" in s:
                    rec["ebr_decomposition"] = s["ebr_decomposition"]
                if "integer_solution" in s:
                    rec["integer_solution"] = s["integer_solution"]
                if "search_status" in s:
                    rec["search_status"] = s["search_status"]
                if "irrep_source_provenance_by_kpoint" in s:
                    rec["irrep_source_provenance_by_kpoint"] = (
                        s["irrep_source_provenance_by_kpoint"]
                    )
                # --- Compact table provenance (when present) ---
                tp = s.get("table_provenance")
                if isinstance(tp, dict) and tp:
                    rec["table_source"] = tp.get("source", "")
                    rec["data_source"] = tp.get("data_source", "")
                    rec["package"] = tp.get("package", "")
                    rec["package_version"] = tp.get("package_version", "")
                    rec["space_group_number"] = tp.get("space_group_number")
                    rec["spinful"] = tp.get("spinful")
                    rec["expected_hsps"] = tp.get("expected_hsps", [])
                    rec["valleyscope_reduction"] = tp.get("valleyscope_reduction", "")
                    rec["source_basis_count"] = tp.get("source_basis_count")
                    rec["reduction_basis_count"] = tp.get("reduction_basis_count")
                    rec["dropped_source_row_count"] = tp.get("dropped_source_row_count")
                    rec["dropped_source_rows"] = tp.get("dropped_source_rows", [])
                    rec["filtered_zero_vector_ebr_count"] = tp.get(
                        "filtered_zero_vector_ebr_count", 0
                    )
                    rec["filtered_zero_vector_ebrs"] = tp.get(
                        "filtered_zero_vector_ebrs", []
                    )
                    rec["independent_setting_identity"] = tp.get(
                        "independent_setting_identity"
                    )
                    rec["unitary_space_group_number"] = tp.get(
                        "unitary_space_group_number"
                    )
                    rec["time_reversal_grey_bns_number"] = tp.get(
                        "time_reversal_grey_bns_number"
                    )
                    rec["time_reversal_source"] = tp.get(
                        "time_reversal_source"
                    )
                ts = s.get("table_status")
                if ts is not None:
                    rec["table_status"] = ts
                reduced_ebr_records.append(rec)
            record["reduced_ebr_records"] = reduced_ebr_records
        else:
            record["reduced_ebr_records"] = []
    else:
        record["reduced_ebr_mapping_status"] = "not_available"
        record["reduced_ebr_table_status"] = "not_available"
        record["reduced_ebr_records"] = []

    # --- Stage-owned status ---
    if record["final_reduced_ebr_result_count"] > 0:
        record["record_status"] = "has_final_reduced_ebr_results"
    elif record["reduced_table_validation_candidate_bundle_count"] > 0:
        record["record_status"] = (
            "has_reduced_table_validation_candidates"
        )
    else:
        record["record_status"] = "no_reduced_ebr_input"

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
    summary = _load_json_object(summary_path)
    source_files["valley_summary"] = str(summary_path.resolve())

    # valley_ebr_export_bundle.json (optional)
    bundle = None
    bundle_path = root / "valley_ebr_export_bundle.json"
    if bundle_path.is_file():
        bundle = _load_json_object(bundle_path)
        source_files["valley_ebr_export_bundle"] = str(bundle_path.resolve())

    # valley_reduced_ebr_mapping.json (optional)
    mapping = None
    mapping_path = root / "valley_reduced_ebr_mapping.json"
    if mapping_path.is_file():
        mapping = _load_json_object(mapping_path)
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


def _validate_bundle_cprime_against_summary(
    *,
    bundle: dict[str, Any],
    source_basis: dict[str, Any],
    acceptance_matrix: object,
) -> str | None:
    source_identity = source_basis.get("identity")
    if (
        source_basis.get("status") != "passed"
        or not valid_sha256_identity(source_identity)
    ):
        return "summary C-prime source-basis evidence is not passed"
    if not isinstance(acceptance_matrix, list):
        return "summary C-prime acceptance matrix is malformed"
    valley = str(bundle.get("valley", ""))
    by_scope: dict[tuple[str, str], dict[str, Any]] = {}
    for row in acceptance_matrix:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("kpoint", "")), str(row.get("valley", "")))
        if key in by_scope:
            return "summary C-prime acceptance matrix has duplicate scope"
        by_scope[key] = row
    bundle_links = bundle.get("cprime_identity_by_kpoint")
    irreps = bundle.get("irreps_by_kpoint")
    expected_scopes = set(irreps) if isinstance(irreps, dict) else set()
    scope_metadata = {}
    if unitary_bundle_claims_valley_sewing_completion(bundle):
        scope_metadata = bundle.get("cprime_scope_metadata", {})
        expected_scopes = (
            set(scope_metadata)
            if isinstance(scope_metadata, dict) else set()
        )
    if (
        not isinstance(bundle_links, dict)
        or not isinstance(irreps, dict)
        or set(bundle_links) != expected_scopes
    ):
        return "bundle C-prime identity inventory is incomplete"
    for scope_key, identity in bundle_links.items():
        scope = scope_metadata.get(scope_key, {})
        kpoint = scope.get("sampled_kpoint", scope_key)
        scope_valley = scope.get("evidence_valley", valley)
        row = by_scope.get((str(kpoint), str(scope_valley)))
        if not isinstance(row, dict) or not isinstance(identity, dict):
            return f"summary C-prime scope missing for {kpoint}/{scope_valley}"
        expected = {
            "spinor_source_basis_certificate_identity": source_identity,
            "double_space_group_lift_certificate_identity": row.get(
                "double_space_group_lift_identity"
            ),
            "scoped_representation_evidence_identity": row.get(
                "scoped_representation_evidence_identity"
            ),
        }
        if (
            row.get("double_space_group_lift_status") != "passed"
            or row.get("scoped_representation_status") != "passed"
            or identity != expected
            or not all(
                valid_sha256_identity(value)
                for value in expected.values()
            )
        ):
            return f"C-prime identity mismatch for {kpoint}/{valley}"
    return None


def _authoritative_mapping_solutions(
    *,
    solutions: object,
    valley_ebr_export_bundle: dict[str, Any] | None,
    reduced_ebr_input: dict[str, Any] | None,
    errors: list[str],
) -> list[dict[str, Any]]:
    """Keep only solutions tied to the current promoted export bundle."""
    if not isinstance(solutions, list):
        return []
    bundles = (
        valley_ebr_export_bundle.get("bundles", [])
        if isinstance(valley_ebr_export_bundle, dict)
        else []
    )
    ready_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    if isinstance(bundles, list):
        for bundle in bundles:
            if (
                not isinstance(bundle, dict)
                or bundle.get("ready_for_reduced_table_validation") is not True
            ):
                continue
            bundle_id = bundle.get("bundle_id")
            if not isinstance(bundle_id, str) or not bundle_id:
                continue
            if bundle_id in ready_by_id:
                duplicate_ids.add(bundle_id)
            ready_by_id[bundle_id] = bundle
    for duplicate_id in duplicate_ids:
        ready_by_id.pop(duplicate_id, None)

    solution_counts: dict[str, int] = {}
    for solution in solutions:
        if not isinstance(solution, dict):
            continue
        bundle_id = solution.get("bundle_id")
        if isinstance(bundle_id, str) and bundle_id:
            solution_counts[bundle_id] = solution_counts.get(bundle_id, 0) + 1
    duplicate_solution_ids = {
        bundle_id
        for bundle_id, count in solution_counts.items()
        if count > 1
    }
    for bundle_id in sorted(duplicate_solution_ids):
        errors.append(
            f"mapping solution {bundle_id}: duplicate final solution ID"
        )

    authoritative: list[dict[str, Any]] = []
    for solution in solutions:
        if not isinstance(solution, dict):
            continue
        bundle_id = solution.get("bundle_id")
        display_id = bundle_id if isinstance(bundle_id, str) else "<unknown>"
        if bundle_id in duplicate_solution_ids:
            continue
        bundle = ready_by_id.get(bundle_id) if isinstance(bundle_id, str) else None
        if bundle is None:
            errors.append(
                f"mapping solution {display_id}: no matching ready export bundle"
            )
            continue
        if not _has_passed_promotion_provenance(solution):
            errors.append(
                f"mapping solution {display_id}: "
                "missing passed promotion provenance"
            )
            continue
        validation_error = _solution_validation_error(
            solution=solution,
            bundle=bundle,
            reduced_ebr_input=reduced_ebr_input,
        )
        if validation_error is not None:
            errors.append(f"mapping solution {display_id}: {validation_error}")
            continue
        authoritative.append(solution)
    return authoritative


def _has_passed_promotion_provenance(solution: dict[str, Any]) -> bool:
    promotion = solution.get("promotion_provenance")
    if (
        not isinstance(promotion, dict)
        or promotion.get("source") != "promote_bundle_for_solve"
    ):
        return False
    report = promotion.get("validation_report")
    if (
        not isinstance(report, dict)
        or solution.get("validation_report") != report
        or any(
            report.get(check) != "passed"
            for check in _PROMOTION_REQUIRED_PASSED_CHECKS
        )
    ):
        return False
    certificate = promotion.get("certificate_identity")
    if (
        not isinstance(certificate, dict)
        or not certificate
        or solution.get("certificate_identity") != certificate
    ):
        return False
    promoted_table = promotion.get("table_provenance")
    solution_table = solution.get("table_provenance")
    if (
        not isinstance(promoted_table, dict)
        or not promoted_table
        or not isinstance(solution_table, dict)
    ):
        return False
    return True


def _serialized_sewing_bundle_has_strict_promotion(bundle, mapping):
    solutions = mapping.get("solutions") if isinstance(mapping, dict) else None
    matches = [
        row for row in solutions or []
        if isinstance(row, dict) and row.get("bundle_id") == bundle.get("bundle_id")
    ]
    return bool(
        len(matches) == 1
        and _has_passed_promotion_provenance(matches[0])
        and matches[0]["promotion_provenance"].get("promotion_input_identity")
        == build_promotion_input_identity(bundle)
    )


def _solution_validation_error(
    *,
    solution: dict[str, Any],
    bundle: dict[str, Any],
    reduced_ebr_input: dict[str, Any] | None,
) -> str | None:
    promotion = solution["promotion_provenance"]
    if promotion.get("promotion_input_identity") != (
        build_promotion_input_identity(bundle)
    ):
        return "promotion input identity does not match current export bundle"
    if any(
        solution.get(field) != bundle.get(
            field,
            _SOLUTION_BUNDLE_BINDING_DEFAULTS[field],
        )
        for field in _SOLUTION_BUNDLE_BINDING_FIELDS
    ):
        return "promotion provenance does not match current export bundle"
    if promotion.get("irrep_vector") != solution.get("irrep_vector"):
        return "promoted irrep vector does not match solution"
    promoted_table = promotion["table_provenance"]
    current_input = table_input_for_bundle(
        reduced_ebr_input,
        str(solution.get("bundle_id", "")),
    )
    if solution.get("table_provenance") != merge_table_input_provenance(
        promoted_table,
        current_input,
    ):
        return "table input provenance does not match current mapping input"
    if bundle.get("problem_kind") == "unitary_valley_reduced_ebr":
        if unitary_bundle_claims_valley_sewing_completion(bundle):
            valid = validate_serialized_valley_sewing_completed_unitary_bundle(
                bundle
            )
        else:
            valid = validate_unitary_bundle_provenance(bundle)
        if not valid:
            return "current unitary provenance is invalid"
        return None
    if bundle.get("problem_kind") == "valley_orbit_reduced_ebr":
        from valleyscope.analysis.reduced_ebr_mapping import (
            validate_joint_grey_bundle_provenance,
        )
        if not validate_joint_grey_bundle_provenance(
            bundle,
            promoted_table,
        ):
            return "current joint grey provenance is invalid"
        return None
    return "current reduced EBR problem kind is invalid"


def _extract_irrep_records(
    bundle: dict[str, Any],
    out: list[dict[str, Any]],
    *,
    allow_joint_fallback: bool = True,
) -> str | None:
    """Flatten trusted irrep_records_by_kpoint into a list of per-record dicts."""
    source_bundle_id = bundle.get("bundle_id", "")
    source_instance_id = bundle.get("source_instance_id", "")
    tr_completed = unitary_bundle_claims_time_reversal_completion(bundle)
    sewing_completed = unitary_bundle_claims_valley_sewing_completion(bundle)
    if tr_completed or sewing_completed:
        valid = (
            validate_serialized_valley_sewing_completed_unitary_bundle(bundle)
            if sewing_completed else validate_unitary_bundle_provenance(bundle)
        )
        if not valid:
            kind = "unitary valley-sewing completion" if sewing_completed else "TR-completed unitary"
            return f"bundle {source_bundle_id or '<unknown>'}: invalid {kind} provenance"
        _append_tr_completed_unitary_records(
            bundle=bundle,
            out=out,
            source_bundle_id=source_bundle_id,
            source_instance_id=source_instance_id,
        )
        return None

    if (
        bundle.get("problem_kind") == "unitary_valley_reduced_ebr"
        and bundle.get("physical_object_kind")
        == "unitary_valley_projected_subspace"
        and not validate_unitary_bundle_provenance(bundle)
    ):
        return (
            f"bundle {source_bundle_id or '<unknown>'}: invalid direct "
            "unitary construction provenance"
        )

    records_by_kp = bundle.get("irrep_records_by_kpoint", {})
    if not isinstance(records_by_kp, dict):
        return None
    output_start = len(out)

    for kpoint, records in records_by_kp.items():
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            # Canonical subgroup identity: subspace_space_group is the
            # primary physical object preserved from per-record provenance
            # (see generic-field preservation loop below); flat
            # subgroup_candidate is the derived scalar key for compact
            # indexing.
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
                "certificate_identity": bundle.get("certificate_identity", {}),
            }
            # Preserve generic irrep provenance fields when present.
            for key in (
                "irrep_multiplicity",
                "matching_strategy",
                "subspace_space_group",
                "valley_preserving_operation_ids",
                "source_operation_map",
                "irrep_source_provenance",
            ):
                if key in rec:
                    entry[key] = rec[key]
            out.append(entry)

    if len(out) != output_start:
        return None
    if (
        bundle.get("problem_kind") != "valley_orbit_reduced_ebr"
        or not allow_joint_fallback
    ):
        return None
    unitary_irreps = bundle.get("unitary_valley_irreps", {})
    source_to_sampled = bundle.get("source_hsp_to_sampled_kpoint", {})
    time_reversal = bundle.get("time_reversal", {})
    source_to_sampled_by_valley = (
        time_reversal.get("source_hsp_to_sampled_kpoint_by_valley", {})
        if isinstance(time_reversal, dict) else {}
    )
    representative_valley = (
        time_reversal.get("representative_valley")
        if isinstance(time_reversal, dict) else None
    )
    binding_error = _validate_valley_orbit_source_hsp_bindings(
        unitary_irreps=unitary_irreps,
        source_to_sampled=source_to_sampled,
        source_to_sampled_by_valley=source_to_sampled_by_valley,
        representative_valley=representative_valley,
    )
    if binding_error is not None:
        bundle_id = source_bundle_id or "<unknown>"
        return (
            f"bundle {bundle_id}: invalid valley-orbit "
            f"source-HSP/sample binding ({binding_error})"
        )
    for valley in sorted(unitary_irreps):
        by_source_hsp = unitary_irreps.get(valley)
        if not isinstance(by_source_hsp, dict):
            continue
        for source_hsp in sorted(by_source_hsp):
            multiplicities = by_source_hsp.get(source_hsp)
            if not isinstance(multiplicities, dict):
                continue
            for irrep in sorted(multiplicities):
                multiplicity = multiplicities.get(irrep)
                if (
                    not isinstance(multiplicity, int)
                    or isinstance(multiplicity, bool)
                    or multiplicity <= 0
                ):
                    continue
                out.append({
                    "kpoint": source_to_sampled_by_valley[valley][source_hsp],
                    "source_hsp_label": source_hsp,
                    "valley": valley,
                    "subspace_group_candidate": bundle.get(
                        "subspace_group_candidate", ""
                    ),
                    "matched_irrep": irrep,
                    "irrep_multiplicity": multiplicity,
                    "workflow_path": bundle.get("workflow_path", ""),
                    "readiness_level": bundle.get("readiness_level", ""),
                    "source": "unitary_valley_irreps",
                    "source_bundle_id": source_bundle_id,
                    "source_instance_id": source_instance_id,
                    "certificate_identity": bundle.get(
                        "certificate_identity", {}
                    ),
                })
    return None


def _append_tr_completed_unitary_records(
    *,
    bundle: dict[str, Any],
    out: list[dict[str, Any]],
    source_bundle_id: object,
    source_instance_id: object,
) -> None:
    valley = str(bundle["valley"])
    records_by_hsp = bundle["unitary_irrep_completion_records_by_hsp"]
    if unitary_bundle_claims_valley_sewing_completion(bundle):
        for source_hsp, records in bundle.get(
            "irrep_records_by_kpoint", {}
        ).items():
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                out.append({
                    "kpoint": record.get("sampled_kpoint"),
                    "source_hsp_label": source_hsp,
                    "valley": valley,
                    "subspace_group_candidate": bundle.get(
                        "subspace_group_candidate", ""
                    ),
                    "matched_irrep": record.get("matched_irrep"),
                    "irrep_multiplicity": record.get(
                        "irrep_multiplicity"
                    ),
                    "completion_kind": "observed_at_sampled_kpoint",
                    "workflow_path": record.get("workflow_path", ""),
                    "readiness_level": record.get("readiness_level", ""),
                    "source": record.get("source", ""),
                    "source_bundle_id": source_bundle_id,
                    "source_instance_id": source_instance_id,
                    "certificate_identity": bundle.get(
                        "certificate_identity", {}
                    ),
                    "irrep_source_provenance": record.get(
                        "irrep_source_provenance", {}
                    ),
                })
    source_hsps = (
        records_by_hsp
        if unitary_bundle_claims_valley_sewing_completion(bundle)
        else bundle["expected_hsps"]
    )
    for source_hsp in source_hsps:
        for record in records_by_hsp[source_hsp]:
            kind = record["completion_kind"]
            entry: dict[str, Any] = {
                "source_hsp_label": source_hsp,
                "valley": valley,
                "subspace_group_candidate": bundle.get(
                    "subspace_group_candidate", ""
                ),
                "matched_irrep": record["irrep"],
                "irrep_multiplicity": record["multiplicity"],
                "completion_kind": kind,
                "workflow_path": bundle.get("workflow_path", ""),
                "readiness_level": bundle.get("readiness_level", ""),
                "source": "unitary_irrep_completion_records",
                "source_bundle_id": source_bundle_id,
                "source_instance_id": source_instance_id,
                "certificate_identity": bundle.get(
                    "certificate_identity", {}
                ),
                "source_candidate_identity": record[
                    "source_candidate_identity"
                ],
                "source_candidate_provenance": record[
                    "source_candidate_provenance"
                ],
                "structural_status": record["structural_status"],
                "readiness_status": record["readiness_status"],
            }
            if kind == "observed_at_sampled_kpoint":
                entry["kpoint"] = record["sampled_kpoint"]
            else:
                for key in (
                    "evidence_valley",
                    "evidence_source_hsp_label",
                    "evidence_sampled_kpoint",
                    "reviewed_time_reversal_relation",
                    "evidence_irrep_vector",
                ):
                    if key in record:
                        entry[key] = record[key]
                certificates = record.get(
                    "unitary_valley_sewing_certificates"
                )
                if isinstance(certificates, list):
                    entry[
                        "unitary_valley_sewing_certificate_identities"
                    ] = [
                        certificate.get("certificate_identity")
                        for certificate in certificates
                        if isinstance(certificate, dict)
                    ]
            out.append(entry)


def _validate_valley_orbit_source_hsp_bindings(
    *,
    unitary_irreps: object,
    source_to_sampled: object,
    source_to_sampled_by_valley: object,
    representative_valley: object,
) -> str | None:
    if not isinstance(unitary_irreps, dict) or not unitary_irreps:
        return "unitary valley irreps are missing or malformed"
    if (
        not isinstance(representative_valley, str)
        or not representative_valley
        or representative_valley not in unitary_irreps
    ):
        return "representative valley is missing or malformed"
    if not _valid_source_hsp_to_sampled_map(source_to_sampled):
        return "representative flat map is missing or malformed"
    if (
        not isinstance(source_to_sampled_by_valley, dict)
        or set(source_to_sampled_by_valley) != set(unitary_irreps)
    ):
        return "valley-resolved component coverage is incomplete"
    if source_to_sampled_by_valley.get(representative_valley) != source_to_sampled:
        return "representative flat map conflicts with its component map"

    expected_source_hsps = set(source_to_sampled)
    for valley, raw_irreps in unitary_irreps.items():
        if not isinstance(valley, str) or not valley:
            return "unitary valley label is malformed"
        if not isinstance(raw_irreps, dict) or not raw_irreps:
            return f"unitary source-HSP basis is missing for valley {valley}"
        by_source = source_to_sampled_by_valley.get(valley)
        if (
            not _valid_source_hsp_to_sampled_map(by_source)
            or set(by_source) != expected_source_hsps
        ):
            return f"source-HSP coverage is incomplete for valley {valley}"
        if not set(raw_irreps).issubset(by_source):
            return f"emitted source-HSP binding is missing for valley {valley}"
    return None


def _valid_source_hsp_to_sampled_map(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(source_hsp, str)
            and bool(source_hsp)
            and isinstance(sampled, str)
            and bool(sampled)
            for source_hsp, sampled in value.items()
        )
        and len(set(value.values())) == len(value)
    )


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
            "subspace_space_group": exc.get("subspace_space_group", {}),
            "status": exc.get("status", "?"),
            "canonical_hsp_vector_complete": exc.get(
                "canonical_hsp_vector_complete", False
            ),
            "canonical_hsp_vector_ready": exc.get(
                "canonical_hsp_vector_ready", False
            ),
            "exclusion_reasons": exc.get("exclusion_reasons", []),
        })
    return records


def _compact_mapping_exclusions(
    excluded_bundles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve final-mapping blockers without mixing input exclusions."""
    records: list[dict[str, Any]] = []
    for excluded in excluded_bundles:
        if not isinstance(excluded, dict):
            continue
        record = {
            "bundle_id": excluded.get("bundle_id", "?"),
            "subspace_group_candidate": excluded.get(
                "subspace_group_candidate", "?"
            ),
            "subspace_space_group": excluded.get(
                "subspace_space_group", {}
            ),
            "reason": excluded.get("reason", ""),
            "blocker_reasons": excluded.get("blocker_reasons", []),
        }
        for key in (
            "problem_kind",
            "valley",
            "valley_orbit",
            "validation_report",
            "certificate_identity",
            "table_provenance",
        ):
            if key in excluded:
                record[key] = excluded[key]
        records.append(record)
    return records


def _empty_classification_counts() -> dict[str, int]:
    return {
        "atomic_compatible": 0,
        "in_integer_span_no_nonnegative_witness": 0,
        "outside_integer_span": 0,
        "indeterminate_truncated": 0,
    }
