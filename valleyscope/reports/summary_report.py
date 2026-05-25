from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from valleyscope.io.config import AppConfig
from valleyscope.reports.json_report import _json_default


def build_summary_payload(
    *,
    config: AppConfig,
    qcut: float,
    subspace_payload: dict[str, Any],
    symmetry_payload: dict[str, Any],
    symmetry_rows: list[dict[str, Any]] | None = None,
    output_paths: dict[str, Path],
    symmetry_eigenvalue_summary: dict[str, Any] | None = None,
    projector_symmetry_report: dict[str, Any] | None = None,
    symmetry_adapted_valley_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eigen_rows = [] if symmetry_rows is None else symmetry_rows
    warnings = _collect_warnings(subspace_payload, symmetry_payload, eigen_rows)
    payload: dict[str, Any] = {
        "input": {
            "wavefunction_h5": str(config.input.wavefunction_h5),
            "operation_structure_file": None
            if config.symmetry.operations.structure_file is None
            else str(config.symmetry.operations.structure_file),
            "operation_detection_backend": config.symmetry.operations.backend,
            "spinor_convention": config.spinor.convention,
            "spinor_convention_verified": config.spinor.convention_verified,
            "spinor_benchmark": config.spinor.benchmark,
        },
        "target_kpoints": list(config.analysis.kpoints),
        "iband": list(config.analysis.iband),
        "valley_subspaces": [
            {"label": sector.name, "centers": list(sector.centers)}
            for sector in config.valley_subspaces
        ],
        "qcut": {
            "mode": config.projection.qcut_mode,
            "value_Ainv": float(qcut),
            "scan": list(config.projection.qcut_scan),
        },
        "valley_projection_summary": _projection_rows(subspace_payload),
        "valley_subspace_analysis": _subspace_rows(subspace_payload),
        "symmetry_analysis": _symmetry_analysis(symmetry_payload, config.analysis.kpoints),
        "symmetry_eigenvalues": eigen_rows,
        "symmetry_characters": _symmetry_character_rows(eigen_rows),
        "rotation_readiness_thresholds": _rotation_readiness_thresholds(config),
        "warnings": warnings,
        "output_files": {name: str(path) for name, path in output_paths.items()},
        "legend": {
            "W_val": "valley-subspace weight",
            "P_v": "valley purity",
            "eta": "signed valley polarization for a two-valley diagnostic (legacy)",
            "valley_weights_adapted": "valley weights in valley-adapted basis",
            "valley_concentration": "max weight fraction in assigned valley (general multi-valley)",
            "W_overlap": "projector-window overlap weight",
            "W_res": "residual weight",
            "topology_input_ready": (
                "HSP symmetry eigenvalue is suitable as input to later symmetry-based topology analysis; "
                "it does not validate full-mBZ valley-resolved topology"
            ),
            "topology_ready": "backward-compatible alias of topology_input_ready",
            "epsilon_seed": (
                "||D_g P_a^0 D_g^dag - P_{pi_g(a)}^0||_F / max(||P_a^0||_F, small); "
                "epsilon_seed is the seed projector symmetry error"
            ),
        },
    }
    if symmetry_eigenvalue_summary:
        payload["symmetry_eigenvalue_summary"] = symmetry_eigenvalue_summary
    if projector_symmetry_report is not None:
        payload["projector_symmetry"] = _compact_projector_symmetry(projector_symmetry_report)
    if symmetry_adapted_valley_report is not None:
        payload["symmetry_adapted_valley_analysis"] = symmetry_adapted_valley_report
    return payload


def render_summary_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    _section(lines, "Input")
    input_summary = summary["input"]
    lines.append(f"wavefunction_h5: {input_summary['wavefunction_h5']}")
    lines.append(f"operation structure: {input_summary['operation_structure_file']}")
    lines.append(f"operation-detection backend: {input_summary['operation_detection_backend']}")
    lines.append(
        "spinor convention: "
        f"{input_summary['spinor_convention']} "
        f"(verified={input_summary['spinor_convention_verified']}, "
        f"benchmark={input_summary['spinor_benchmark']})"
    )
    lines.append(f"target k-points: {', '.join(summary['target_kpoints'])}")
    lines.append(f"iband (VASP): {', '.join(str(v) for v in summary['iband'])}")
    qcut = summary["qcut"]
    lines.append(f"qcut mode: {qcut['mode']}")
    lines.append(f"qcut value: {_fmt(qcut['value_Ainv'])} A^-1")
    lines.append("")

    _section(lines, "Valley subspaces")
    lines.extend(
        _table(
            ["label", "centers"],
            [
                [row["label"], ", ".join(row["centers"])]
                for row in summary["valley_subspaces"]
            ],
        )
    )
    lines.append("")

    _section(lines, "Valley projection summary")
    lines.append("W_val:      valley-subspace weight")
    lines.append("P_v:        valley purity")
    lines.append("W_overlap: projector-window overlap weight")
    lines.append("W_res:      residual weight")
    projection_rows = summary["valley_projection_summary"]
    lines.extend(
        _table(
            ["kpoint", "band", "W_val", "P_v", "W_overlap", "W_res", "status"],
            [
                [
                    row["kpoint"],
                    row["band_vasp"],
                    _fmt(row.get("W_val")),
                    _fmt(row.get("P_v")),
                    _fmt(row.get("W_overlap")),
                    _fmt(row.get("W_res")),
                    row.get("status", ""),
                ]
                for row in projection_rows
            ],
        )
    )
    lines.append("")

    _section(lines, "Valley subspace analysis")
    lines.append("S = sum_a P_a^sub: target-valley-subspace projector in the selected target bands")
    lines.append("S_min:      minimum target-valley-subspace weight")
    lines.append("S_max:      maximum target-valley-subspace weight")
    lines.append("min_concentration: minimum valley concentration in valley-adapted basis")
    lines.append("assigned_valleys:  valley assignment per adapted state")
    lines.append("eta_adapted: signed valley polarization (two-valley only)")
    lines.extend(
        _table(
            ["kpoint", "S_min", "S_max", "min_concentration", "assigned_valleys", "eta_adapted", "basis", "status"],
            [
                [
                    row["kpoint"],
                    _fmt(row.get("s_min")),
                    _fmt(row.get("s_max")),
                    _fmt(row.get("min_valley_concentration")),
                    _short_list(row.get("assigned_valleys")),
                    _short_list(row.get("eta_adapted")),
                    _subspace_basis_label(row),
                    row.get("status", ""),
                ]
                for row in summary["valley_subspace_analysis"]
            ],
        )
    )
    lines.append("")

    _section(lines, "Symmetry analysis")
    sym = summary["symmetry_analysis"]
    lines.append(f"status: {sym['status']}")
    lines.append(f"space group: {sym.get('international')} ({sym.get('spacegroup_number')})")
    subgroup_report = sym.get("valley_preserving_subgroup_report", {})
    if isinstance(subgroup_report, dict):
        valley_preserving_subgroups = subgroup_report.get("valley_preserving_subgroups", {})
        if isinstance(valley_preserving_subgroups, dict) and valley_preserving_subgroups:
            for vname, subgroup in valley_preserving_subgroups.items():
                if isinstance(subgroup, dict):
                    ops = ", ".join(str(v) for v in subgroup.get("operation_ids", [])) or "none"
                    lines.append(f"valley-preserving subgroup({vname}): [{ops}]")
        # All-valley intersection (debug)
        all_valley = subgroup_report.get("all_valley_intersection", {})
        if isinstance(all_valley, dict) and all_valley.get("allowed_operation_ids"):
            lines.append(
                "all-valley intersection (debug): "
                f"{all_valley.get('allowed_operation_ids')}"
            )
        # Legacy standard group match
        standard_match = subgroup_report.get("standard_group_match")
        if isinstance(standard_match, dict):
            lines.append(
                "valley-preserving subgroup: "
                f"{standard_match.get('international_short')} ({standard_match.get('number')})"
            )
        elif subgroup_report.get("standard_group_match_status"):
            lines.append(
                "valley-preserving subgroup: "
                f"{subgroup_report.get('standard_group_match_status')}"
            )
        # Valley orbits
        orbits = subgroup_report.get("valley_orbits", [])
        if isinstance(orbits, list) and orbits:
            for orbit in orbits:
                if isinstance(orbit, dict):
                    vals = orbit.get("valleys", [])
                    permuting = orbit.get("valley_permuting_operation_ids",
                                          orbit.get("coset_representative_operation_ids", []))
                    if vals:
                        lines.append(f"valley orbit: {vals} (permuting ops: {permuting})")
        irrep_matching = subgroup_report.get("irrep_matching")
        if isinstance(irrep_matching, dict):
            label_matching = irrep_matching.get("label_matching", irrep_matching.get("status", ""))
            if label_matching:
                lines.append(f"irrep matching: {label_matching}")
            irrep_results = irrep_matching.get("irrep_results_by_kpoint", {})
            if isinstance(irrep_results, dict) and irrep_results:
                for kpoint, kp_result in irrep_results.items():
                    if not isinstance(kp_result, dict):
                        continue
                    # Check if per-valley format: {valley_name: result}
                    sample_val = next(iter(kp_result.values()), None) if kp_result else None
                    if isinstance(sample_val, dict) and "irrep_multiplicities" in sample_val:
                        # Per-valley format
                        for valley_name, result in kp_result.items():
                            if not isinstance(result, dict):
                                continue
                            multiplicities = result.get("irrep_multiplicities", {})
                            if not multiplicities:
                                lines.append(f"{kpoint}/{valley_name}: {result.get('status', 'none')}")
                                continue
                            terms = ", ".join(
                                f"{label} x {multiplicity}"
                                for label, multiplicity in multiplicities.items()
                            )
                            lines.append(f"{kpoint}/{valley_name}: {terms}")
                            state_results = result.get("state_irrep_results", [])
                            if (
                                result.get("state_irrep_assignment_status") == "matched"
                                and isinstance(state_results, list)
                            ):
                                state_terms = [
                                    f"state {s.get('state_index')} -> {s.get('irrep_label')}"
                                    for s in state_results
                                    if isinstance(s, dict)
                                    and s.get("status") == "matched"
                                    and s.get("irrep_label")
                                ]
                                if state_terms:
                                    lines.append(f"{kpoint}/{valley_name} state irreps: {', '.join(state_terms)}")
                    else:
                        # Legacy flat format
                        multiplicities = kp_result.get("irrep_multiplicities", {})
                        if not multiplicities:
                            lines.append(f"{kpoint}: {kp_result.get('status', 'none')}")
                            continue
                        terms = ", ".join(
                            f"{label} x {multiplicity}"
                            for label, multiplicity in multiplicities.items()
                        )
                        lines.append(f"{kpoint}: {terms}")
                        state_results = kp_result.get("state_irrep_results", [])
                        if (
                            kp_result.get("state_irrep_assignment_status") == "matched"
                            and isinstance(state_results, list)
                        ):
                            state_terms = [
                                f"state {s.get('state_index')} -> {s.get('irrep_label')}"
                                for s in state_results
                                if isinstance(s, dict)
                                and s.get("status") == "matched"
                                and s.get("irrep_label")
                            ]
                            if state_terms:
                                lines.append(f"{kpoint} state irreps: {', '.join(state_terms)}")
    lines.append(f"requested operation order: {sym.get('requested_rotation_order')}")
    lines.append(f"selected proper-rotation order: {sym.get('resolved_rotation_order')}")
    lines.append("")
    lines.append("Detected operations:")
    lines.extend(
        _table(
            ["id", "kind", "order", "det", "W", "w"],
            [
                [
                    row["operation_id"],
                    row.get("kind", ""),
                    row.get("order", ""),
                    _fmt(row.get("det")),
                    row.get("rotation_frac", ""),
                    row.get("translation_frac", ""),
                ]
                for row in sym.get("detected_operations", [])
            ],
        )
    )
    if sym.get("by_kpoint"):
        lines.append("")
        lines.append("HSP little group and valley preservation:")
        for kpoint, payload in sym["by_kpoint"].items():
            if not isinstance(payload, dict):
                continue
            # Per-valley data is nested under "per_valley" key
            per_valley = payload.get("per_valley", {})
            if isinstance(per_valley, dict) and per_valley:
                valley_names = list(per_valley.keys())
                for vname in valley_names:
                    vp = per_valley.get(vname, {})
                    if isinstance(vp, dict):
                        allowed = ", ".join(str(v) for v in vp.get("allowed_operation_ids", [])) or "none"
                        changing = ", ".join(str(v) for v in vp.get("valley_changing_operation_ids", [])) or "none"
                        lines.append(
                            f"{kpoint}/{vname}: valley-preserving [{allowed}], valley-changing [{changing}]"
                        )
            else:
                # Legacy flat format
                little_ops = ", ".join(str(v) for v in payload.get("little_group_operations", [])) or "none"
                preserving_ops = ", ".join(str(v) for v in payload.get("valley_preserving_operations", [])) or "none"
                lines.append(f"{kpoint}: HSP little group [{little_ops}], valley-preserving [{preserving_ops}]")
    if sym.get("rejected_operations"):
        lines.append("")
        lines.append("rejected operations:")
        lines.extend(
            _table(
                ["kpoint", "valley", "operation", "order", "reason"],
                [
                    [
                        row["kpoint"],
                        row.get("target_valley", ""),
                        row["operation_id"],
                        row["order"],
                        row["reason"],
                    ]
                    for row in sym["rejected_operations"]
                ],
            )
        )
    lines.append("")

    _section(lines, "Symmetry eigenvalues")
    lines.append(
        "Note: topology_input_ready only means the HSP symmetry eigenvalue is suitable as input "
        "to later symmetry-based topology analysis; it does not validate full-mBZ valley-resolved topology."
    )
    lines.extend(
        _table(
            [
                "kpoint",
                "valley",
                "operation",
                "order",
                "state",
                "phase",
                "root",
                "root_dev",
                "ready",
                "input_ready",
                "diagnostic",
                "offdiag",
                "block_leak",
                "reason",
            ],
            [
                [
                    row["kpoint"],
                    row.get("target_valley", ""),
                    row["operation_id"],
                    row["order"],
                    row["state_index"],
                    _fmt(row["phase_2pi"]),
                    _format_root_label(row["nearest_root_of_unity"]),
                    _fmt(row["root_deviation"]),
                    row.get("rotation_ready", ""),
                    row.get("topology_input_ready", row.get("topology_ready", "")),
                    row.get("diagnostic_only", ""),
                    _fmt(row.get("D_valley_offdiag_norm")),
                    _fmt(row.get("D_block_leakage_norm")),
                    row.get("reason", ""),
                ]
                for row in summary["symmetry_eigenvalues"]
            ],
        )
    )
    lines.append("")

    projector_symmetry = summary.get("projector_symmetry", {})
    if projector_symmetry:
        _section(lines, "Projector symmetry-consistency")
        lines.append(f"status: {projector_symmetry.get('status', 'no_data')}")
        lines.append(
            f"tolerances: warn={projector_symmetry.get('warn_tol')}, fail={projector_symmetry.get('fail_tol')}"
        )
        for kpoint, kp_data in projector_symmetry.get("by_kpoint", {}).items():
            total = kp_data.get("total_checks", 0)
            failed = kp_data.get("failed_count", 0)
            warned = kp_data.get("warn_count", 0)
            lines.append(
                f"{kpoint}: {total} checks, {failed} failed, {warned} warned"
            )
            for item in kp_data.get("failed", []):
                lines.append(
                    f"  FAILED op={item.get('operation_id')} "
                    f"{item.get('source_valley')}->{item.get('mapped_valley')} "
                    f"epsilon={_fmt(item.get('epsilon_seed'))}"
                )
            for item in kp_data.get("warned", []):
                lines.append(
                    f"  WARN op={item.get('operation_id')} "
                    f"{item.get('source_valley')}->{item.get('mapped_valley')} "
                    f"epsilon={_fmt(item.get('epsilon_seed'))}"
                )
        lines.append("")
        if any(kp_data.get("failed_count", 0) > 0
               for kp_data in projector_symmetry.get("by_kpoint", {}).values()):
            lines.append(
                "Seed projector symmetry-consistency failures detected: "
                "valley-preserving irrep/eigenvalue labels based on the q-cut "
                "seed basis are diagnostic-only for affected operations."
            )
        lines.append("")

    symmetry_adapted = summary.get("symmetry_adapted_valley_analysis")
    if isinstance(symmetry_adapted, dict):
        _render_symmetry_adapted_valley_analysis(lines, symmetry_adapted)

    _section(lines, "Warnings")
    if summary["warnings"]:
        lines.extend(f"- {item}" for item in summary["warnings"])
    else:
        lines.append("None")
    lines.append("")

    _section(lines, "Output files")
    for name, path in summary["output_files"].items():
        lines.append(f"{_output_file_label(name)}: {path}")
    return "\n".join(lines).rstrip() + "\n"


def write_summary_text(path: str | Path, text: str) -> Path:
    out = Path(path)
    out.write_text(text, encoding="utf-8")
    return out


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> Path:
    out = Path(path)
    out.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return out


def _projection_rows(subspace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kpoint, payload in subspace_payload.get("kpoints", {}).items():
        for weight in payload.get("weights", []):
            rows.append(
                {
                    "kpoint": kpoint,
                    "band_vasp": weight.get("band_vasp"),
                    "W_val": weight.get("W_val"),
                    "P_v": weight.get("P_v"),
                    "eta": weight.get("eta"),
                    "W_overlap": weight.get("W_overlap"),
                    "W_res": weight.get("W_res"),
                    "status": _short_valley_status(weight.get("valley_status", "")),
                }
            )
    return rows


def _subspace_rows(subspace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kpoint, payload in subspace_payload.get("kpoints", {}).items():
        diagnostic = payload.get("valley_adapted_subspace", {})
        p_v_min = _subspace_purity_min(diagnostic.get("eta"))
        rows.append(
            {
                "kpoint": kpoint,
                "basis_status": diagnostic.get("status", ""),
                "n_valleys": diagnostic.get("n_valleys", 0),
                "energy_span_meV": diagnostic.get("energy_span_meV"),
                "subspace_energy_tol_meV": subspace_payload.get("degeneracy_tol_meV"),
                "s_eigenvalues": diagnostic.get("s_eigenvalues"),
                "s_min": diagnostic.get("s_min"),
                "s_max": diagnostic.get("s_max"),
                "P_v_min": p_v_min,
                "eta_adapted": diagnostic.get("eta"),
                "min_valley_concentration": diagnostic.get("min_valley_concentration"),
                "assigned_valleys": diagnostic.get("assigned_valleys"),
                "valley_concentration": diagnostic.get("valley_concentration"),
                "valid_valley_subspace": diagnostic.get("valid_valley_subspace"),
                "stably_separable": diagnostic.get("stably_separable"),
                "status": _short_valley_status(payload.get("subspace_valley_status", "")),
            }
        )
    return rows


def _symmetry_analysis(symmetry_payload: dict[str, Any], target_kpoints: list[str]) -> dict[str, Any]:
    rejected: list[dict[str, Any]] = []
    kind_counts: dict[str, int] = {}
    operations: list[dict[str, Any]] = []
    by_kpoint: dict[str, dict[str, list[int]]] = {}
    subgroup_report = symmetry_payload.get("valley_preserving_subgroup_report", {})
    per_valley_by_kpoint = subgroup_report.get("by_kpoint", {}) if isinstance(subgroup_report, dict) else {}
    per_valley_inventory = symmetry_payload.get("per_valley_preserving_operation_inventory", {})

    for operation in symmetry_payload.get("detected_operations", []):
        kind = str(operation.get("kind", "unknown"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        operations.append(
            {
                "operation_id": operation.get("operation_id"),
                "kind": operation.get("kind"),
                "order": operation.get("order"),
                "det": operation.get("det"),
                "rotation_frac": _short_matrix(operation.get("rotation_frac")),
                "translation_frac": _short_list(operation.get("translation_frac")),
            }
        )
        for kpoint, reason in operation.get("rejection_reason_by_kpoint", {}).items():
            if kpoint not in by_kpoint:
                by_kpoint[kpoint] = {
                    "little_group_operations": [],
                    "valley_preserving_operations": [],
                }
            if operation.get("little_group_by_kpoint", {}).get(kpoint):
                by_kpoint[kpoint]["little_group_operations"].append(operation.get("operation_id"))
                if not reason:
                    by_kpoint[kpoint]["valley_preserving_operations"].append(operation.get("operation_id"))
    order = {name: idx for idx, name in enumerate(target_kpoints)}
    ordered_by_kpoint = {
        kpoint: by_kpoint[kpoint]
        for kpoint in target_kpoints
        if kpoint in by_kpoint
    }
    for kpoint in by_kpoint:
        if kpoint not in ordered_by_kpoint:
            ordered_by_kpoint[kpoint] = by_kpoint[kpoint]
    if isinstance(per_valley_inventory, dict) and per_valley_inventory:
        ordered_inventory_kpoints = [
            kpoint for kpoint in target_kpoints if kpoint in per_valley_inventory
        ]
        ordered_inventory_kpoints.extend(
            kpoint for kpoint in per_valley_inventory if kpoint not in ordered_inventory_kpoints
        )
        for kpoint in ordered_inventory_kpoints:
            valley_payload = per_valley_inventory.get(kpoint, {})
            if not isinstance(valley_payload, dict):
                continue
            for valley_name, rows in valley_payload.items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    reason = row.get("reason", "")
                    if not reason:
                        continue
                    rejected.append(
                        {
                            "kpoint": kpoint,
                            "target_valley": valley_name,
                            "operation_id": row.get("operation_id"),
                            "order": row.get("order"),
                            "kind": row.get("kind"),
                            "reason": reason,
                        }
                    )
    else:
        for operation in symmetry_payload.get("detected_operations", []):
            for kpoint, reason in operation.get("rejection_reason_by_kpoint", {}).items():
                if reason:
                    rejected.append(
                        {
                            "kpoint": kpoint,
                            "target_valley": "",
                            "operation_id": operation.get("operation_id"),
                            "order": operation.get("order"),
                            "kind": operation.get("kind"),
                            "reason": reason,
                        }
                    )
    rejected.sort(
        key=lambda row: (
            order.get(str(row["kpoint"]), len(order)),
            str(row.get("target_valley", "")),
            int(row["operation_id"]) if row.get("operation_id") is not None else -1,
        )
    )

    # Merge per-valley by_kpoint data from subgroup report
    merged_by_kpoint: dict[str, Any] = {}
    for kpoint in ordered_by_kpoint:
        merged_by_kpoint[kpoint] = dict(ordered_by_kpoint[kpoint])
        if kpoint in per_valley_by_kpoint and isinstance(per_valley_by_kpoint[kpoint], dict):
            merged_by_kpoint[kpoint]["per_valley"] = per_valley_by_kpoint[kpoint]

    return {
        "status": symmetry_payload.get("status"),
        "operation_detection_backend": symmetry_payload.get("operation_detection_backend"),
        "structure_file": symmetry_payload.get("structure_file"),
        "spacegroup_number": symmetry_payload.get("spacegroup_number"),
        "international": symmetry_payload.get("international"),
        "requested_rotation_order": symmetry_payload.get("requested_rotation_order"),
        "resolved_rotation_order": symmetry_payload.get("resolved_rotation_order"),
        "symmetry_eigenvalue_enabled": symmetry_payload.get("symmetry_eigenvalue_enabled"),
        "detected_operation_count": symmetry_payload.get("detected_operation_count", 0),
        "candidate_rotations": symmetry_payload.get("candidate_rotations", []),
        "symprec_scan_summary": symmetry_payload.get("symprec_scan_summary", []),
        "hsp_little_group_inventory": symmetry_payload.get("hsp_little_group_inventory", {}),
        "per_valley_preserving_operation_inventory": per_valley_inventory,
        "valley_preserving_subgroup_report": subgroup_report,
        "kind_counts": kind_counts,
        "detected_operations": operations,
        "by_kpoint": merged_by_kpoint,
        "little_group_check": symmetry_payload.get("little_group_check", {}),
        "valley_preservation_check": symmetry_payload.get("valley_preservation_check", {}),
        "rejected_operations": rejected,
    }


def _symmetry_character_rows(symmetry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, Any, str], dict[str, Any]] = {}
    for row in symmetry_rows:
        if not bool(row.get("little_group_passed", False)):
            continue
        if not bool(row.get("valley_preserving", False)):
            continue
        key = (str(row.get("kpoint", "")), row.get("operation_id"), str(row.get("target_valley", "")))
        if key not in grouped:
            grouped[key] = {
                "kpoint": row.get("kpoint", ""),
                "target_valley": row.get("target_valley", ""),
                "operation_id": row.get("operation_id"),
                "kind": row.get("kind", ""),
                "order": row.get("order"),
                "basis": row.get("basis", ""),
                "character_raw": "",
                "character_valley": "",
                "topology_input_ready": True,
                "diagnostic_only": False,
                "accepted_for_valley_preserving_representation": True,
            }
        item = grouped[key]
        if row.get("character_raw"):
            item["character_raw"] = row.get("character_raw")
        if row.get("character_valley"):
            item["character_valley"] = row.get("character_valley")
        item["topology_input_ready"] = bool(item["topology_input_ready"]) and bool(row.get("topology_input_ready", False))
        item["diagnostic_only"] = bool(item["diagnostic_only"]) or bool(row.get("diagnostic_only", False))
    return list(grouped.values())


def _rotation_readiness_thresholds(config: AppConfig) -> dict[str, Any]:
    return {
        "readiness_preset": config.rotation.readiness_preset,
        "unitarity_tol": config.rotation.unitarity_tol,
        "root_deviation_tol": config.rotation.root_deviation_tol,
        "D_valley_offdiag_tol": config.rotation.D_valley_offdiag_tol,
        "irrep_weight_tol": config.rotation.irrep_weight_tol,
        "interpretation": (
            "These are numerical readiness thresholds, not universal physical constants."
        ),
        "recommended_action": (
            "Check qcut stability, valley purity, spinor benchmark, plane-wave mapping, "
            "and representation quality; do not loosen thresholds only to obtain "
            "topology_input_ready=True or an irrep label."
        ),
    }


def _collect_warnings(
    subspace_payload: dict[str, Any],
    symmetry_payload: dict[str, Any],
    symmetry_rows: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if symmetry_payload.get("status") == "skipped":
        warnings.append(str(symmetry_payload.get("reason", "symmetry-operation detection skipped")))
    for kpoint, payload in subspace_payload.get("kpoints", {}).items():
        for warning in payload.get("warnings", []):
            warnings.append(f"{kpoint}: {warning}")
        diagnostic = payload.get("valley_adapted_subspace", {})
        sv_status = payload.get("subspace_valley_status", "")
        if sv_status == "valley_mixed_subspace":
            warnings.append(f"{kpoint}: valley subspace is mixed; symmetry eigenvalues are diagnostic-only")
        if sv_status == "valley_approximately_separable_subspace":
            pass
        if sv_status == "not_valley_derived":
            warnings.append(f"{kpoint}: target subspace has insufficient valley weight (S_min below threshold)")
        if sv_status == "projector_unreliable":
            warnings.append(f"{kpoint}: valley projectors unreliable; check commutator norms and idempotency")
        for row_weight in payload.get("weights", []):
            w_v_status = row_weight.get("valley_status", "")
            if w_v_status == "projector_unreliable":
                warnings.append(
                    f"{kpoint} band {row_weight.get('band_vasp')}: "
                    "projector windows are unreliable; check W_overlap and W_res"
                )
    if any(row.get("basis") != "valley_adapted" for row in symmetry_rows):
        warnings.append("Some symmetry eigenvalues are not valley-adapted and are diagnostic-only")
    if any(
        bool(row.get("spinor_rotation_applied", False)) and not bool(row.get("spinor_convention_verified", False))
        for row in symmetry_rows
    ):
        warnings.append(
            "Spinor rotation is applied, but the VASP spinor convention is not benchmark-verified; "
            "spinful symmetry eigenvalues are diagnostic-only"
        )
    if any(bool(row.get("diagnostic_only", False)) for row in symmetry_rows):
        warnings.append(
            "Some symmetry eigenvalues are diagnostic-only and are not topology_input_ready"
        )
    if any(str(row.get("seed_projector_symmetry_status", "")) == "failed" for row in symmetry_rows):
        warnings.append(
            "Valley-preserving irrep labels based on the q-cut seed basis are "
            "diagnostic-only when the seed projector symmetry-consistency check fails"
        )
    for operation in symmetry_payload.get("detected_operations", []):
        for kpoint, quality in operation.get("representation_quality", {}).items():
            if isinstance(quality, dict) and quality.get("skipped_reason"):
                warnings.append(f"{kpoint}: operation {operation.get('operation_id')} skipped: {quality['skipped_reason']}")
    return warnings


def _section(lines: list[str], title: str) -> None:
    lines.append(title)
    lines.append("-" * len(title))


def _render_symmetry_adapted_valley_analysis(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "Symmetry-adapted valley analysis (experimental)")
    lines.append("trusted_irrep_label: false")
    lines.append(
        "local_irrep_ready reports internal consistency of this experimental layer; "
        "irrep_matching_input_ready remains the gate for future table matching."
    )
    by_kpoint = report.get("by_kpoint", {})
    if not isinstance(by_kpoint, dict) or not by_kpoint:
        lines.append("(none)")
        lines.append("")
        return

    kpoint_rows: list[list[Any]] = []
    orbit_rows: list[list[Any]] = []
    subspace_rows: list[list[Any]] = []
    for kpoint, kp_data in by_kpoint.items():
        if not isinstance(kp_data, dict):
            continue
        kpoint_rows.append([
            kpoint,
            kp_data.get("status", ""),
            kp_data.get("local_irrep_ready", ""),
            kp_data.get("irrep_matching_input_ready", ""),
            kp_data.get("reason", ""),
        ])
        for item in kp_data.get("orbits", []):
            if isinstance(item, dict):
                orbit_rows.append(_symmetry_adapted_orbit_row(kpoint, item))
        for item in kp_data.get("valley_preserving_subspaces", []):
            if isinstance(item, dict):
                subspace_rows.append(_symmetry_adapted_subspace_row(kpoint, item))

    lines.append("kpoint summary:")
    lines.extend(
        _table(
            ["kpoint", "status", "local_ready", "irrep_input", "reason"],
            kpoint_rows,
        )
    )
    lines.append("")
    if orbit_rows:
        lines.append("full valley-orbit reports:")
        lines.extend(
            _table(
                ["kpoint", "orbit", "status", "rank", "proj", "max_sym", "local_ready", "irrep_input", "reason"],
                orbit_rows,
            )
        )
        lines.append("")
    if subspace_rows:
        lines.append("valley-preserving subspaces:")
        lines.extend(
            _table(
                ["kpoint", "valley", "ops", "rank", "proj", "seed_overlap", "phases", "local_ready", "irrep_input", "reason"],
                subspace_rows,
            )
        )
        lines.append("")


def _symmetry_adapted_orbit_row(kpoint: Any, item: dict[str, Any]) -> list[Any]:
    projectors = item.get("symmetry_adapted_projectors", {})
    if not isinstance(projectors, dict):
        projectors = {}
    reason = _symmetry_adapted_reason(item)
    return [
        kpoint,
        _format_orbit(item.get("orbit")),
        item.get("status", ""),
        projectors.get("selected_rank", ""),
        projectors.get("status", ""),
        _fmt(projectors.get("max_projector_symmetry_error")),
        item.get("local_irrep_ready", ""),
        item.get("irrep_matching_input_ready", ""),
        reason,
    ]


def _symmetry_adapted_subspace_row(kpoint: Any, item: dict[str, Any]) -> list[Any]:
    projectors = item.get("symmetry_adapted_projectors", {})
    if not isinstance(projectors, dict):
        projectors = {}
    reason = _symmetry_adapted_reason(item)
    return [
        kpoint,
        _format_orbit(item.get("orbit")),
        _short_list(item.get("hsp_preserving_operation_ids")),
        projectors.get("selected_rank", ""),
        projectors.get("status", ""),
        _format_seed_overlap(projectors.get("seed_overlap")),
        _format_character_phases(item.get("valley_preserving_character_diagnostics")),
        item.get("local_irrep_ready", ""),
        item.get("irrep_matching_input_ready", ""),
        reason,
    ]


def _symmetry_adapted_reason(item: dict[str, Any]) -> str:
    reason = str(item.get("reason", "") or "")
    irrep_ready = bool(item.get("irrep_matching_input_ready", False))
    irrep_reason = str(item.get("irrep_matching_input_reason", "") or "")
    if not reason or reason == "all stages passed":
        return irrep_reason or reason
    if not irrep_ready and irrep_reason and irrep_reason not in reason:
        return f"{reason}; irrep_input: {irrep_reason}"
    return reason


def _format_orbit(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    if value is None:
        return ""
    return str(value)


def _format_seed_overlap(value: Any) -> str:
    if not isinstance(value, dict):
        return _fmt(value)
    return ", ".join(f"{key}={_fmt(val)}" for key, val in value.items())


def _format_character_phases(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    per_valley = value.get("per_valley", {})
    if not isinstance(per_valley, dict):
        return ""
    terms: list[str] = []
    for rows in per_valley.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            op_id = row.get("operation_id")
            if str(op_id) == "0":
                continue
            phases = row.get("eigenphases")
            if phases is None:
                continue
            terms.append(f"op {op_id}: {_short_list(phases)}")
    return "; ".join(terms)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["(none)"]
    string_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in string_rows))
        for idx, header in enumerate(headers)
    ]
    result = ["  ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers))]
    result.append("  ".join("-" * width for width in widths))
    for row in string_rows:
        result.append("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    return result


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _short_list(value: Any) -> str:
    if value is None:
        return ""
    array = np.asarray(value)
    if array.ndim == 0:
        return _fmt(array.item())
    return "[" + ", ".join(_fmt(item) for item in array.tolist()) + "]"


def _short_matrix(value: Any) -> str:
    if value is None:
        return ""
    array = np.asarray(value)
    if array.ndim != 2:
        return _short_list(value)
    rows = ["[" + ",".join(_fmt(item) for item in row.tolist()) + "]" for row in array]
    return "[" + ";".join(rows) + "]"


def _short_valley_status(status: Any) -> str:
    value = str(status)
    if value in {"single_band", "requires_two_valley_sectors", "no_valley_sectors"}:
        return "n/a"
    if value == "not_degenerate":
        return "not_degenerate"
    if value == "not_evaluated":
        return "not_evaluated"
    if value == "not_valley_derived" or value == "poor_valley_manifold":
        return "not_derived"
    if value == "projector_unreliable":
        return "unreliable"
    if value.endswith("clean") or value == "valley_separable_subspace" or value == "valley_separable":
        return "clean"
    if "approx" in value:
        return "approx"
    if not value:
        return "n/a"
    return "mixed"


def _subspace_basis_label(row: dict[str, Any]) -> str:
    status = str(row.get("basis_status", ""))
    if status == "not_degenerate":
        span = _fmt(row.get("energy_span_meV"))
        tol = _fmt(row.get("subspace_energy_tol_meV"))
        return f"not_degenerate (span={span} meV > tol={tol} meV)"
    if status == "single_band":
        return "single_band"
    if status in ("not_evaluated", ""):
        return "n/a"
    return status


def _format_root_label(value: Any) -> str:
    text = str(value)
    match = re.fullmatch(r"exp\(2pii\*(\d+)/(\d+)\)", text)
    if not match:
        return text
    index = int(match.group(1))
    order = int(match.group(2))
    if order == 0:
        return text
    frac = Fraction(2 * index, order)
    frac -= 2 * (frac > 1)
    if frac == 0:
        return "1"
    if frac == 1:
        return "-1"
    if frac == Fraction(1, 2):
        return "i"
    if frac == Fraction(-1, 2):
        return "-i"
    sign = "-" if frac < 0 else ""
    frac = abs(frac)
    if frac == 1:
        return f"exp({sign}i*pi)"
    if frac.numerator == 1:
        return f"exp({sign}i*pi/{frac.denominator})"
    return f"exp({sign}i*{frac.numerator}pi/{frac.denominator})"


def _subspace_purity_min(eta: Any) -> float | None:
    if eta is None:
        return None
    array = np.asarray(eta, dtype=float)
    if array.size == 0:
        return None
    return float((1.0 + np.min(np.abs(array))) / 2.0)


def _infer_valley_names_from_by_kpoint(by_kpoint: dict[str, Any]) -> list[str]:
    for payload in by_kpoint.values():
        if isinstance(payload, dict):
            for key, val in payload.items():
                if isinstance(val, dict) and "allowed_operation_ids" in val:
                    return list(payload.keys())
    return []


def _output_file_label(name: str) -> str:
    labels = {
        "valley_summary_txt": "Human-readable summary",
        "valley_summary_json": "Machine-readable summary",
        "valley_weights_csv": "Valley weights",
        "valley_subspace_json": "Valley subspace analysis",
        "valley_basis_transform_h5": "Valley basis transform",
        "symmetry_report_json": "Symmetry analysis",
        "symmetry_eigenvalues_csv": "Symmetry eigenvalues",
        "diagnostics_h5": "Projector, qcut, and symmetry matrices",
        "projector_symmetry_report_json": "Projector symmetry report",
        "symmetry_adapted_valley_analysis_json": "Symmetry-adapted valley analysis (experimental)",
    }
    return labels.get(name, name.replace("_", " ").title())


def _compact_projector_symmetry(report: dict[str, Any]) -> dict[str, Any]:
    """Extract compact summary from full projector symmetry report."""
    summary: dict[str, Any] = {
        "status": report.get("status", "no_data"),
        "warn_tol": report.get("warn_tol"),
        "fail_tol": report.get("fail_tol"),
    }
    by_kpoint: dict[str, Any] = {}
    for kpoint, kp_data in report.get("by_kpoint", {}).items():
        if not isinstance(kp_data, dict):
            continue
        rows = kp_data.get("seed_projector_symmetry", [])
        if not isinstance(rows, list):
            continue
        failed = []
        warned = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            status = row.get("status", "")
            if status == "failed":
                failed.append({
                    "operation_id": row.get("operation_id"),
                    "source_valley": row.get("source_valley"),
                    "mapped_valley": row.get("mapped_valley"),
                    "epsilon_seed": row.get("epsilon_seed"),
                    "seed_projector_symmetry_error": row.get("seed_projector_symmetry_error"),
                })
            elif status == "warn":
                warned.append({
                    "operation_id": row.get("operation_id"),
                    "source_valley": row.get("source_valley"),
                    "mapped_valley": row.get("mapped_valley"),
                    "epsilon_seed": row.get("epsilon_seed"),
                    "seed_projector_symmetry_error": row.get("seed_projector_symmetry_error"),
                })
        by_kpoint[kpoint] = {
            "total_checks": len(rows),
            "failed_count": len(failed),
            "warn_count": len(warned),
            "failed": failed,
            "warned": warned,
        }
    summary["by_kpoint"] = by_kpoint
    return summary
