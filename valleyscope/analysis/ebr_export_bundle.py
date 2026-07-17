"""Export canonical HSP-vector problems for reviewed reduced-table validation."""

from __future__ import annotations


_SCHEMA_VERSION = "1.6.0"


def build_ebr_export_bundle(
    *,
    ebr_problem_instances: dict[str, object] | None,
) -> dict[str, object]:
    """Build reviewed-table validation inputs from canonical HSP vectors."""
    if ebr_problem_instances is None:
        return _empty("no problem instances available")

    raw = ebr_problem_instances.get("instances")
    if not isinstance(raw, list):
        return _empty("no problem instances list")

    bundles: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []

    for inst in raw:
        if not isinstance(inst, dict):
            continue

        canonical_complete = (
            inst.get("canonical_hsp_vector_complete") is True
        )
        canonical_ready = inst.get("canonical_hsp_vector_ready") is True

        if canonical_ready:
            bundles.append({
                "bundle_id": f"bundle_{inst.get('instance_id', '?')}",
                "source_instance_id": inst.get("instance_id", ""),
                "problem_kind": inst.get(
                    "problem_kind", "unitary_valley_reduced_ebr"
                ),
                "valley": inst.get("valley", ""),
                "valley_orbit": inst.get("valley_orbit", []),
                "subspace_group_candidate": inst.get("subspace_group_candidate", ""),
                "subspace_sg_number": inst.get("subspace_sg_number"),
                "subspace_space_group": inst.get("subspace_space_group", {}),
                "spinor": inst.get("spinor"),
                "certificate_identity": inst.get("certificate_identity", {}),
                "workflow_path": inst.get("workflow_path", ""),
                "readiness_level": inst.get("readiness_level", ""),
                "irreps_by_kpoint": inst.get("irreps_by_kpoint", {}),
                "operations_by_kpoint": inst.get("operations_by_kpoint", {}),
                "irrep_records_by_kpoint": inst.get("irrep_records_by_kpoint", {}),
                "unitary_valley_irreps": inst.get(
                    "unitary_valley_irreps", {}
                ),
                "time_reversal": inst.get("time_reversal", {}),
                "expected_hsps": inst.get("expected_hsps", []),
                "optional_hsps": inst.get("optional_hsps", []),
                "missing_optional_hsps": inst.get("missing_optional_hsps", []),
                "required_source_hsp_labels": inst.get(
                    "required_source_hsp_labels", []
                ),
                "covered_source_hsp_labels": inst.get(
                    "covered_source_hsp_labels", []
                ),
                "missing_source_hsp_labels": inst.get(
                    "missing_source_hsp_labels", []
                ),
                "trusted_matched_source_hsp_labels": inst.get(
                    "trusted_matched_source_hsp_labels", []
                ),
                "source_hsp_to_sampled_kpoint": inst.get(
                    "source_hsp_to_sampled_kpoint", {}
                ),
                "source_hsp_coverage_complete": inst.get(
                    "source_hsp_coverage_complete", False
                ),
                "source_hsp_coverage_provenance": inst.get(
                    "source_hsp_coverage_provenance", {}
                ),
                "canonical_hsp_vector_complete": True,
                "canonical_hsp_vector_ready": True,
                "ready_for_reduced_table_validation": True,
            })
            continue

        excluded.append({
            "source_instance_id": inst.get("instance_id", ""),
            "problem_kind": inst.get(
                "problem_kind", "unitary_valley_reduced_ebr"
            ),
            "valley": inst.get("valley", ""),
            "valley_orbit": inst.get("valley_orbit", []),
            "subspace_group_candidate": inst.get("subspace_group_candidate", ""),
            "subspace_space_group": inst.get("subspace_space_group", {}),
            "status": inst.get("status", ""),
            "required_source_hsp_labels": inst.get(
                "required_source_hsp_labels", []
            ),
            "covered_source_hsp_labels": inst.get(
                "covered_source_hsp_labels", []
            ),
            "missing_source_hsp_labels": inst.get(
                "missing_source_hsp_labels", []
            ),
            "canonical_hsp_vector_complete": canonical_complete,
            "canonical_hsp_vector_ready": canonical_ready,
            "exclusion_reasons": list(inst.get("blocked_by", [])),
        })

    overall = _classify_status(bundles, excluded)

    return {
        "status": overall,
        "bundle_count": len(bundles),
        "excluded_count": len(excluded),
        "schema_version": _SCHEMA_VERSION,
        "interpretation": (
            "Complete canonical source-HSP vectors packaged for reviewed "
            "reduced-table validation. Exact mapping outcomes are reported "
            "only by the final reduced-EBR mapping."
        ),
        "bundles": bundles,
        "excluded_instances": excluded,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _classify_status(
    bundles: list[dict[str, object]],
    excluded: list[dict[str, object]],
) -> str:
    if not bundles:
        return "no_bundles"
    if excluded:
        return "partial_export"
    return "ready_for_reduced_table_validation"


def _empty(reason: str) -> dict[str, object]:
    return {
        "status": "no_bundles",
        "bundle_count": 0,
        "excluded_count": 0,
        "schema_version": _SCHEMA_VERSION,
        "interpretation": reason,
        "bundles": [],
        "excluded_instances": [],
    }
