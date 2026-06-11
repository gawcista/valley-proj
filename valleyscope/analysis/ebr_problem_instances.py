"""EBR problem-instance collector from trusted EBR input candidates.

Groups trusted candidate irreps into per-valley/per-subspace-group EBR
problem instances and reports HSP completeness.  Does NOT implement
reduced EBR decomposition, EBR table matching, or new physics.
"""

from __future__ import annotations

from typing import Any


# Expected HSP labels for known subspace groups.  This is a policy table,
# not a physics solver.  Missing HSPs are reported as blocked/optional.
_EXPECTED_HSP: dict[str, dict[str, object]] = {
    "P3": {
        "expected_hsps": ["GammaM", "KM"],
        "optional_hsps": ["MM"],
        "note": (
            "P3/C3_like single-valley subspace groups expect C3 valley-preserving "
            "irreps at GammaM and KM. MM is optional: current inputs may not "
            "provide nontrivial trusted local irreps at MM HSP."
        ),
    },
    "C3_like": {
        "expected_hsps": ["GammaM", "KM"],
        "optional_hsps": ["MM"],
        "note": "C3_like follows same expected HSP policy as P3.",
    },
    "P2": {
        "expected_hsps": [],
        "optional_hsps": [],
        "note": (
            "P2/C2_like completeness requires trusted non-identity C2 data "
            "at the relevant HSPs. No expected HSPs are declared until "
            "trusted C2 valley-preserving irreps are available for this "
            "subspace group candidate. Without trusted C2 irreps the "
            "instance status is no_instances."
        ),
    },
    "C2_like": {
        "expected_hsps": [],
        "optional_hsps": [],
        "note": "C2_like follows same expected HSP policy as P2.",
    },
}


def build_ebr_problem_instances(
    *,
    ebr_input_candidates: dict[str, object] | None,
) -> dict[str, object]:
    """Build EBR problem instances from trusted input candidates.

    Returns a dict with instances grouped by subspace group and valley.
    """
    if ebr_input_candidates is None:
        return _empty_report("no EBR input candidates available")

    candidates: list[dict[str, object]] = []
    raw = ebr_input_candidates.get("candidates")
    if isinstance(raw, list):
        for c in raw:
            if isinstance(c, dict) and c.get("ready_for_ebr_input") is True:
                candidates.append(c)

    if not candidates:
        return _empty_report("no trusted EBR input candidates")

    # Group by the full workflow identity.  Mixing direct q-cut and
    # symmetry-adapted candidates would make downstream EBR provenance unclear.
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for c in candidates:
        sg = str(c.get("subspace_group_candidate", ""))
        valley = str(c.get("valley", ""))
        workflow_path = str(c.get("workflow_path", ""))
        readiness_level = str(c.get("readiness_level", ""))
        groups.setdefault((sg, valley, workflow_path, readiness_level), []).append(c)

    instances: list[dict[str, object]] = []
    instance_counter = 0

    for (sg, valley, workflow_path, readiness_level), cands in groups.items():
        instance_counter += 1
        instance_id = f"ebr_instance_{instance_counter:03d}"

        irreps_by_kpoint: dict[str, list[str]] = {}
        operations_by_kpoint: dict[str, list[object]] = {}
        irrep_records_by_kpoint: dict[str, list[dict[str, object]]] = {}
        for c in cands:
            kp = str(c.get("kpoint", ""))
            irrep = c.get("matched_irrep")
            op_id = c.get("operation_id")
            if irrep:
                irreps_by_kpoint.setdefault(kp, []).append(str(irrep))
            if op_id is not None:
                operations_by_kpoint.setdefault(kp, []).append(op_id)
            # Provenance record for trusted candidates only.
            irrep_records_by_kpoint.setdefault(kp, []).append({
                "valley": valley,
                "operation_id": c.get("operation_id"),
                "operation_order": c.get("operation_order"),
                "matched_irrep": c.get("matched_irrep"),
                "character": c.get("character"),
                "eigenphases": c.get("eigenphases", []),
                "workflow_path": workflow_path,
                "readiness_level": readiness_level,
                "source": c.get("source", ""),
            })

        # Completeness: check expected HSPs
        policy = _EXPECTED_HSP.get(sg, {})
        expected = list(policy.get("expected_hsps", []))
        optional = list(policy.get("optional_hsps", []))

        actual_hsps = set(irreps_by_kpoint.keys())
        required = set(expected)
        missing_required = [h for h in required if h not in actual_hsps]
        missing_optional = [h for h in optional if h not in actual_hsps]

        blocked_by: list[str] = []
        if not expected and not optional:
            status = "no_policy"
            ready = False
            blocked_by.append(f"no expected-HSP policy for {sg}")
        elif missing_required:
            status = "partial"
            ready = False
            blocked_by.append(
                f"missing required HSPs: {missing_required}"
            )
        else:
            status = "complete"
            ready = True

        instances.append({
            "instance_id": instance_id,
            "valley": valley,
            "subspace_group_candidate": sg,
            "workflow_path": workflow_path,
            "readiness_level": readiness_level,
            "irreps_by_kpoint": {k: v for k, v in sorted(irreps_by_kpoint.items())},
            "operations_by_kpoint": {
                k: sorted(v, key=_sort_key)
                for k, v in sorted(operations_by_kpoint.items())
            },
            "irrep_records_by_kpoint": {
                k: sorted(v, key=lambda r: (_sort_key(r.get("operation_id"))))
                for k, v in sorted(irrep_records_by_kpoint.items())
            },
            "candidate_count": len(cands),
            "status": status,
            "ready_for_ebr_decomposition": ready,
            "blocked_by": blocked_by,
            "expected_hsps": expected,
            "optional_hsps": optional,
            "actual_hsps": sorted(actual_hsps),
            "missing_optional_hsps": missing_optional,
        })

    overall_status = "has_instances" if instances else "no_instances"
    return {
        "status": overall_status,
        "instance_count": len(instances),
        "reduced_ebr_decomposition_status": "not_implemented",
        "expected_hsp_policy": {
            k: v.get("note", "") for k, v in _EXPECTED_HSP.items()
        },
        "interpretation": (
            "Per-valley/per-subspace-group EBR problem instances grouped from "
            "trusted input candidates. Completeness is assessed against expected "
            "HSP labels defined in a policy table, not computed from symmetry. "
            "Missing HSP irreps are reported as blockers; no missing data is "
            "filled by assumptions, symmetry, TRS, or table inference."
        ),
        "instances": instances,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _sort_key(op_id: object) -> tuple[int, object]:
    try:
        return (0, int(str(op_id)))
    except (TypeError, ValueError):
        return (1, str(op_id))


def _empty_report(reason: str) -> dict[str, object]:
    return {
        "status": "no_instances",
        "instance_count": 0,
        "reduced_ebr_decomposition_status": "not_implemented",
        "expected_hsp_policy": {
            k: v.get("note", "") for k, v in _EXPECTED_HSP.items()
        },
        "interpretation": reason,
        "instances": [],
    }
