from __future__ import annotations

import json
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
        "valley_manifolds": [
            {"label": sector.name, "centers": list(sector.centers)}
            for sector in config.valley_manifolds
        ],
        "qcut": {
            "mode": config.projection.qcut_mode,
            "value_Ainv": float(qcut),
            "scan": list(config.projection.qcut_scan),
        },
        "valley_projection_summary": _projection_rows(subspace_payload),
        "two_valley_subspace": _subspace_rows(subspace_payload),
        "symmetry_analysis": _symmetry_analysis(symmetry_payload),
        "symmetry_eigenvalues": eigen_rows,
        "warnings": warnings,
        "output_files": {name: str(path) for name, path in output_paths.items()},
        "legend": {
            "W_val": "valley-subspace weight",
            "P_v": "valley purity",
            "eta": "signed valley polarization for a two-valley diagnostic",
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

    _section(lines, "Valley manifolds")
    lines.extend(
        _table(
            ["label", "centers"],
            [
                [row["label"], ", ".join(row["centers"])]
                for row in summary["valley_manifolds"]
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

    _section(lines, "Two-valley subspace")
    lines.append("S=P_K+P_Kp checks valley-subspace weight; V=P_K-P_Kp fixes the valley-adapted basis.")
    lines.extend(
        _table(
            ["kpoint", "basis", "S_min", "S_max", "P_v_min", "eta_adapted", "status"],
            [
                [
                    row["kpoint"],
                    row.get("basis_status", ""),
                    _fmt(row.get("s_min")),
                    _fmt(row.get("s_max")),
                    _fmt(row.get("P_v_min")),
                    _short_list(row.get("eta_adapted")),
                    row.get("status", ""),
                ]
                for row in summary["two_valley_subspace"]
            ],
        )
    )
    lines.append("")

    _section(lines, "Symmetry analysis")
    sym = summary["symmetry_analysis"]
    lines.append(f"status: {sym['status']}")
    lines.append(f"space group: {sym.get('international')} ({sym.get('spacegroup_number')})")
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
                "basis",
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
                    row["basis"],
                    row["state_index"],
                    _fmt(row["phase_2pi"]),
                    row["nearest_root_of_unity"],
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
                    "analysis_level": "raw_state",
                    "derived_score": weight.get("derived_score", weight.get("W_val")),
                    "polarization_score": weight.get("polarization_score"),
                    "W_val": weight.get("W_val"),
                    "P_v": weight.get("P_v"),
                    "eta": weight.get("eta"),
                    "W_overlap": weight.get("W_overlap"),
                    "W_res": weight.get("W_res"),
                    "status": _short_valley_status(weight.get("valley_status", "")),
                    "valley_status": weight.get("valley_status", "not_valley_derived"),
                    "symmetry_status": payload.get("symmetry_status", "not_requested"),
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
                "analysis_level": "adapted_subspace",
                "basis_status": diagnostic.get("status", ""),
                "derived_score": diagnostic.get("s_min"),
                "s_eigenvalues": diagnostic.get("s_eigenvalues"),
                "s_min": diagnostic.get("s_min"),
                "s_max": diagnostic.get("s_max"),
                "P_v_min": p_v_min,
                "eta_adapted": diagnostic.get("eta"),
                "valid_valley_subspace": diagnostic.get("valid_valley_subspace"),
                "polarization_score": payload.get("polarization_score", diagnostic.get("max_abs_eta")),
                "status": _short_valley_status(payload.get("subspace_valley_status", "")),
                "valley_status": payload.get("subspace_valley_status", ""),
                "symmetry_status": payload.get("symmetry_status", "not_requested"),
            }
        )
    return rows


def _symmetry_analysis(symmetry_payload: dict[str, Any]) -> dict[str, Any]:
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
        "kind_counts": kind_counts,
        "detected_operations": operations,
        "by_kpoint": by_kpoint,
        "little_group_check": symmetry_payload.get("little_group_check", {}),
        "valley_preservation_check": symmetry_payload.get("valley_preservation_check", {}),
        "rejected_operations": rejected,
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
            warnings.append(f"{kpoint}: valley_mixed_subspace; symmetry eigenvalues are diagnostic-only")
        if sv_status == "valley_approximately_separable_subspace":
            pass
        if sv_status == "not_valley_derived":
            warnings.append(f"{kpoint}: subspace not_valley_derived; target subspace is not valley-derived")
        if sv_status == "projector_unreliable":
            warnings.append(f"{kpoint}: projector_unreliable; check W_overlap and W_res")
        for row_weight in payload.get("weights", []):
            w_v_status = row_weight.get("valley_status", "")
            if w_v_status == "projector_unreliable":
                warnings.append(f"{kpoint} band {row_weight.get('band_vasp')}: projector_unreliable; check W_overlap and W_res")
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
    if value.endswith("clean") or value == "valley_separable_subspace":
        return "clean"
    if "approx" in value:
        return "approx"
    if not value:
        return "not_evaluated"
    return "mixed"


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
        "valley_subspace_json": "Two-valley subspace data",
        "valley_basis_transform_h5": "Valley basis transform",
        "symmetry_report_json": "Symmetry analysis",
        "symmetry_eigenvalues_csv": "Symmetry eigenvalues",
        "diagnostics_h5": "Projector, qcut, and symmetry matrices",
    }
    return labels.get(name, name.replace("_", " ").title())
