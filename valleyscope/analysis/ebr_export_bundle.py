"""EBR export bundle for downstream reduced-EBR tools.

Packages only complete, ready problem instances into a conservative export
schema.  Does NOT implement reduced EBR decomposition, EBR table matching,
compatibility relations, or new physics.
"""

from __future__ import annotations


_SCHEMA_VERSION = "1.0.0"


def build_ebr_export_bundle(
    *,
    ebr_problem_instances: dict[str, object] | None,
) -> dict[str, object]:
    """Build export bundle from ready problem instances.

    Only instances with ready_for_ebr_decomposition=True and status=="complete"
    are included.  Excluded instances are listed with reasons.
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

        ready = bool(inst.get("ready_for_ebr_decomposition", False))
        status = str(inst.get("status", ""))

        if ready and status == "complete":
            bundles.append({
                "bundle_id": f"bundle_{inst.get('instance_id', '?')}",
                "source_instance_id": inst.get("instance_id", ""),
                "valley": inst.get("valley", ""),
                "subspace_group_candidate": inst.get("subspace_group_candidate", ""),
                "workflow_path": inst.get("workflow_path", ""),
                "readiness_level": inst.get("readiness_level", ""),
                "irreps_by_kpoint": inst.get("irreps_by_kpoint", {}),
                "operations_by_kpoint": inst.get("operations_by_kpoint", {}),
                "expected_hsps": inst.get("expected_hsps", []),
                "optional_hsps": inst.get("optional_hsps", []),
                "missing_optional_hsps": inst.get("missing_optional_hsps", []),
                "ready_for_external_solver": True,
            })
            continue

        reasons: list[str] = []
        if not ready:
            reasons.append(
                f"ready_for_ebr_decomposition={ready}"
            )
        if status != "complete":
            reasons.append(f"status={status}")
            blocked = inst.get("blocked_by", [])
            if blocked:
                reasons.append(f"blocked_by={blocked}")
        excluded.append({
            "source_instance_id": inst.get("instance_id", ""),
            "valley": inst.get("valley", ""),
            "subspace_group_candidate": inst.get("subspace_group_candidate", ""),
            "status": status,
            "ready_for_ebr_decomposition": ready,
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
            "Complete, ready EBR problem instances packaged for downstream "
            "reduced-EBR tools.  Reduced EBR decomposition is not implemented "
            "here; this is a pure export/schema layer.  Excluded instances "
            "are listed with explicit reasons."
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
