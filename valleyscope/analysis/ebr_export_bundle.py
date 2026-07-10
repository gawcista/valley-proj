"""EBR export bundle for downstream reduced-EBR tools.

Packages only problem instances that are ready for reduced-table validation
into a conservative export schema.  Does NOT implement reduced EBR
decomposition, EBR table matching, compatibility relations, or new physics.

State model (see ebr_problem_instances.py):
- State 1 (sampled_basis): ready_for_reduced_table_validation=true → exported
  with ready_for_external_solver=false.  A downstream reduced-table validator
  may promote the bundle after HSP/irrep basis confirmation.
- State 3+ (validated_basis): ready_for_ebr_decomposition=true → exported
  with ready_for_external_solver=true.
"""

from __future__ import annotations


_SCHEMA_VERSION = "1.0.0"


def build_ebr_export_bundle(
    *,
    ebr_problem_instances: dict[str, object] | None,
) -> dict[str, object]:
    """Build export bundle from problem instances.

    State-1 instances (sampled_basis, ready_for_reduced_table_validation=true)
    are exported with ready_for_external_solver=false so the downstream
    solver cannot treat them as validated final-ready problems.

    Only state-3+ instances (validated_basis, ready_for_ebr_decomposition=true)
    are exported with ready_for_external_solver=true.
    """
    if ebr_problem_instances is None:
        return _empty("no problem instances available")

    instances: list[dict[str, object]] = []
    raw = ebr_problem_instances.get("instances")
    if not isinstance(raw, list):
        return _empty("no problem instances list")

    bundles: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []

    for inst in raw:
        if not isinstance(inst, dict):
            continue

        ready_for_validation = bool(
            inst.get("ready_for_reduced_table_validation", False)
        )
        ready_for_decomp = bool(inst.get("ready_for_ebr_decomposition", False))
        hsp_status = str(inst.get("hsp_basis_status", ""))

        if ready_for_validation or ready_for_decomp:
            # State 1 (sampled_basis): export for validation, not for solve.
            # State 3+ (validated_basis): export for external solver.
            ready_for_solver = ready_for_decomp and hsp_status not in (
                "sampled_basis", "no_data",
            )
            bundles.append({
                "bundle_id": f"bundle_{inst.get('instance_id', '?')}",
                "source_instance_id": inst.get("instance_id", ""),
                "valley": inst.get("valley", ""),
                "subspace_group_candidate": inst.get("subspace_group_candidate", ""),
                "subspace_space_group": inst.get("subspace_space_group", {}),
                "certificate_identity": inst.get("certificate_identity", {}),
                "workflow_path": inst.get("workflow_path", ""),
                "readiness_level": inst.get("readiness_level", ""),
                "irreps_by_kpoint": inst.get("irreps_by_kpoint", {}),
                "operations_by_kpoint": inst.get("operations_by_kpoint", {}),
                "irrep_records_by_kpoint": inst.get("irrep_records_by_kpoint", {}),
                "expected_hsps": inst.get("expected_hsps", []),
                "optional_hsps": inst.get("optional_hsps", []),
                "missing_optional_hsps": inst.get("missing_optional_hsps", []),
                "hsp_basis_status": hsp_status,
                "ready_for_reduced_table_validation": ready_for_validation,
                "ready_for_external_solver": ready_for_solver,
            })
            continue

        reasons: list[str] = []
        if not ready_for_validation:
            reasons.append("ready_for_reduced_table_validation=false")
        if not ready_for_decomp:
            reasons.append("ready_for_ebr_decomposition=false")
        excluded.append({
            "source_instance_id": inst.get("instance_id", ""),
            "valley": inst.get("valley", ""),
            "subspace_group_candidate": inst.get("subspace_group_candidate", ""),
            "subspace_space_group": inst.get("subspace_space_group", {}),
            "hsp_basis_status": inst.get("hsp_basis_status", ""),
            "ready_for_reduced_table_validation": ready_for_validation,
            "ready_for_ebr_decomposition": ready_for_decomp,
            "exclusion_reasons": reasons,
        })

    overall = _classify_status(bundles, excluded)

    return {
        "status": overall,
        "bundle_count": len(bundles),
        "excluded_count": len(excluded),
        "schema_version": _SCHEMA_VERSION,
        "reduced_ebr_decomposition_status": "not_implemented",
        "interpretation": (
            "EBR problem instances packaged for downstream reduced-table "
            "validation and/or reduced-EBR tools.  State-1 (sampled_basis) "
            "bundles carry ready_for_external_solver=false; only state-3+ "
            "(validated_basis) bundles are solver-ready."
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
    if not excluded:
        return "ready_for_external_solver"
    return "partial_export"


def _empty(reason: str) -> dict[str, object]:
    return {
        "status": "no_bundles",
        "bundle_count": 0,
        "excluded_count": 0,
        "schema_version": _SCHEMA_VERSION,
        "reduced_ebr_decomposition_status": "not_implemented",
        "interpretation": reason,
        "bundles": [],
        "excluded_instances": [],
    }
