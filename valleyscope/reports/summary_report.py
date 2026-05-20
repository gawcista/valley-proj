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
        },
    }
    if symmetry_eigenvalue_summary:
        payload["symmetry_eigenvalue_summary"] = symmetry_eigenvalue_summary
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
            ["kpoint", "S_min", "S_max", "min_concentration", "assigned_valleys", "eta_adapted", "status"],
            [
                [
                    row["kpoint"],
                    _fmt(row.get("s_min")),
                    _fmt(row.get("s_max")),
                    _fmt(row.get("min_valley_concentration")),
                    _short_list(row.get("assigned_valleys")),
                    _short_list(row.get("eta_adapted")),
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
        irrep_matching = subgroup_report.get("irrep_matching")
        if isinstance(irrep_matching, dict):
            label_matching = irrep_matching.get("label_matching", irrep_matching.get("status", ""))
            if label_matching:
                lines.append(f"irrep matching: {label_matching}")
            irrep_results = irrep_matching.get("irrep_results_by_kpoint", {})
            if isinstance(irrep_results, dict) and irrep_results:
                for kpoint, result in irrep_results.items():
                    if not isinstance(result, dict):
                        continue
                    multiplicities = result.get("irrep_multiplicities", {})
                    if not multiplicities:
                        lines.append(f"{kpoint}: {result.get('status', 'none')}")
                        continue
                    terms = ", ".join(
                        f"{label} x {multiplicity}"
                        for label, multiplicity in multiplicities.items()
                    )
                    lines.append(f"{kpoint}: {terms}")
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
        lines.append("Little group and valley preservation:")
        for kpoint, payload in sym["by_kpoint"].items():
            little_ops = ", ".join(str(v) for v in payload.get("little_group_operations", [])) or "none"
            preserving_ops = ", ".join(str(v) for v in payload.get("valley_preserving_operations", [])) or "none"
            lines.append(f"{kpoint}: little group [{little_ops}], valley-preserving [{preserving_ops}]")
    if sym.get("rejected_operations"):
        lines.append("")
        lines.append("rejected operations:")
        lines.extend(
            _table(
                ["kpoint", "operation", "order", "reason"],
                [
                    [row["kpoint"], row["operation_id"], row["order"], row["reason"]]
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
                "reason",
            ],
            [
                [
                    row["kpoint"],
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
                    row.get("reason", ""),
                ]
                for row in summary["symmetry_eigenvalues"]
            ],
        )
    )
    lines.append("")

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
            if reason:
                rejected.append(
                    {
                        "kpoint": kpoint,
                        "operation_id": operation.get("operation_id"),
                        "order": operation.get("order"),
                        "kind": operation.get("kind"),
                        "reason": reason,
                    }
                )
    order = {name: idx for idx, name in enumerate(target_kpoints)}
    ordered_by_kpoint = {
        kpoint: by_kpoint[kpoint]
        for kpoint in target_kpoints
        if kpoint in by_kpoint
    }
    for kpoint in by_kpoint:
        if kpoint not in ordered_by_kpoint:
            ordered_by_kpoint[kpoint] = by_kpoint[kpoint]
    rejected.sort(
        key=lambda row: (
            order.get(str(row["kpoint"]), len(order)),
            int(row["operation_id"]) if row.get("operation_id") is not None else -1,
        )
    )

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
        "valley_little_group_inventory": symmetry_payload.get("valley_little_group_inventory", {}),
        "valley_preserving_subgroup_report": symmetry_payload.get("valley_preserving_subgroup_report", {}),
        "kind_counts": kind_counts,
        "detected_operations": operations,
        "by_kpoint": ordered_by_kpoint,
        "little_group_check": symmetry_payload.get("little_group_check", {}),
        "valley_preservation_check": symmetry_payload.get("valley_preservation_check", {}),
        "rejected_operations": rejected,
    }


def _symmetry_character_rows(symmetry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, Any], dict[str, Any]] = {}
    for row in symmetry_rows:
        if not bool(row.get("little_group_passed", False)):
            continue
        if not bool(row.get("valley_preserving", False)):
            continue
        key = (str(row.get("kpoint", "")), row.get("operation_id"))
        if key not in grouped:
            grouped[key] = {
                "kpoint": row.get("kpoint", ""),
                "operation_id": row.get("operation_id"),
                "kind": row.get("kind", ""),
                "order": row.get("order"),
                "basis": row.get("basis", ""),
                "character_raw": "",
                "character_valley": "",
                "topology_input_ready": True,
                "diagnostic_only": False,
                "accepted_for_single_valley_representation": True,
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
        "interpretation": (
            "These are numerical readiness thresholds, not universal physical constants."
        ),
        "recommended_action": (
            "Check qcut stability, valley purity, spinor benchmark, plane-wave mapping, "
            "and representation quality; do not loosen thresholds only to obtain "
            "topology_input_ready=True."
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
    for operation in symmetry_payload.get("detected_operations", []):
        for kpoint, quality in operation.get("representation_quality", {}).items():
            if isinstance(quality, dict) and quality.get("skipped_reason"):
                warnings.append(f"{kpoint}: operation {operation.get('operation_id')} skipped: {quality['skipped_reason']}")
    return warnings


def _section(lines: list[str], title: str) -> None:
    lines.append(title)
    lines.append("-" * len(title))


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
    if value in {"single_band", "not_degenerate", "requires_two_valley_sectors", "not_evaluated",
                 "no_valley_sectors"}:
        return "n/a"
    if value == "not_valley_derived" or value == "poor_valley_manifold":
        return "not_derived"
    if value == "projector_unreliable":
        return "unreliable"
    if value.endswith("clean") or value == "valley_separable_subspace" or value == "valley_separable":
        return "valley_separable"
    if "approx" in value:
        return "valley_approx"
    if not value:
        return "n/a"
    return "mixed"


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
    }
    return labels.get(name, name.replace("_", " ").title())
