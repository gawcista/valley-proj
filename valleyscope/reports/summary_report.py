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
    rotation_rows: list[dict[str, Any]],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    warnings = _collect_warnings(subspace_payload, symmetry_payload, rotation_rows)
    return {
        "input": {
            "wavefunction_h5": str(config.input.wavefunction_h5),
            "operation_structure_file": None
            if config.symmetry.operations.structure_file is None
            else str(config.symmetry.operations.structure_file),
            "operation_detection_backend": config.symmetry.operations.backend,
        },
        "target_kpoints": list(config.analysis.kpoints),
        "target_bands_vasp": list(config.analysis.target_bands_vasp),
        "valley_manifolds": [
            {"label": sector.name, "centers": list(sector.centers)}
            for sector in config.valley_sectors
        ],
        "qcut": {
            "mode": config.projection.qcut_mode,
            "value_Ainv": float(qcut),
            "scan": list(config.projection.qcut_scan),
        },
        "valley_projection_summary": _projection_rows(subspace_payload),
        "valley_adapted_subspace": _subspace_rows(subspace_payload),
        "symmetry_diagnostics": _symmetry_summary(symmetry_payload),
        "allowed_valley_preserving_rotations": _accepted_rotations(symmetry_payload),
        "rotation_eigenvalues": rotation_rows,
        "warnings": warnings,
        "output_files": {name: str(path) for name, path in output_paths.items()},
        "legend": {
            "W_val": "target-valley-subspace weight",
            "P_v": "valley purity inside the selected valley subspace",
            "eta": "signed valley polarization for a two-valley diagnostic",
            "W_overlap": "cross-sector projector-window overlap weight",
            "W_res": "out-of-valley residual weight",
            "topology_input_ready": (
                "HSP rotation eigenvalue is suitable as input to later symmetry-based topology analysis; "
                "it does not validate full-mBZ valley-resolved topology"
            ),
            "topology_ready": "backward-compatible alias of topology_input_ready",
        },
    }


def render_summary_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    _section(lines, "Input")
    input_summary = summary["input"]
    lines.append(f"wavefunction_h5: {input_summary['wavefunction_h5']}")
    lines.append(f"operation structure: {input_summary['operation_structure_file']}")
    lines.append(f"operation-detection backend: {input_summary['operation_detection_backend']}")
    lines.append(f"target k-points: {', '.join(summary['target_kpoints'])}")
    lines.append(f"target bands (VASP): {', '.join(str(v) for v in summary['target_bands_vasp'])}")
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
    lines.append(
        "Legend: W_val target-valley-subspace weight; P_v valley purity; "
        "W_overlap projector-window overlap; W_res out-of-valley residual."
    )
    projection_rows = summary["valley_projection_summary"]
    lines.extend(
        _table(
            ["kpoint", "band", "W_val", "P_v", "W_overlap", "W_res", "status"],
            [
                [
                    row["kpoint"],
                    row["band_vasp"],
                    _fmt(row["W_val"]),
                    _fmt(row["P_v"]),
                    _fmt(row["W_overlap"]),
                    _fmt(row["W_res"]),
                    row["status"],
                ]
                for row in projection_rows
            ],
        )
    )
    lines.append("")

    _section(lines, "Valley-adapted subspace")
    lines.extend(
        _table(
            ["kpoint", "status", "eta", "s_min", "s_max", "s_eigenvalues"],
            [
                [
                    row["kpoint"],
                    row["status"],
                    _short_list(row.get("eta")),
                    _fmt(row.get("s_min")),
                    _fmt(row.get("s_max")),
                    _short_list(row.get("s_eigenvalues")),
                ]
                for row in summary["valley_adapted_subspace"]
            ],
        )
    )
    lines.append("")

    _section(lines, "Symmetry diagnostics")
    sym = summary["symmetry_diagnostics"]
    lines.append(f"status: {sym['status']}")
    lines.append(f"detected candidate operations: {sym['detected_operation_count']}")
    lines.append(f"operation types: {sym['kind_counts']}")
    lines.append(f"candidate proper rotations: {len(sym['candidate_rotations'])}")
    lines.append(f"little-group check: {sym['little_group_check']['status']}")
    lines.append(f"valley-preservation check: {sym['valley_preservation_check']['status']}")
    if sym.get("rejected_operations"):
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

    _section(lines, "Allowed valley-preserving rotations")
    lines.extend(
        _table(
            ["kpoint", "operation", "order", "kind"],
            [
                [row["kpoint"], row["operation_id"], row["order"], row["kind"]]
                for row in summary["allowed_valley_preserving_rotations"]
            ],
        )
    )
    lines.append("")

    _section(lines, "Rotation eigenvalues")
    lines.append(
        "Note: topology_input_ready only means the HSP rotation eigenvalue is suitable as input "
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
                "rotation_ready",
                "topology_input_ready",
                "diagnostic_only",
                "D_valley_offdiag_norm",
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
                ]
                for row in summary["rotation_eigenvalues"]
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
        lines.append(f"{name}: {path}")
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
            classification = weight.get("classification", {})
            if not classification.get("valley_derived", False):
                status = "not-valley-derived"
            else:
                clean = classification.get("valley_clean", "mixed")
                status = "valley-clean" if clean == "clean" else clean
            rows.append(
                {
                    "kpoint": kpoint,
                    "band_vasp": weight.get("band_vasp"),
                    "W_val": weight.get("W_val"),
                    "P_v": weight.get("P_v"),
                    "eta": weight.get("eta"),
                    "W_overlap": weight.get("W_overlap"),
                    "W_res": weight.get("W_res"),
                    "status": status,
                }
            )
    return rows


def _subspace_rows(subspace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kpoint, payload in subspace_payload.get("kpoints", {}).items():
        diagnostic = payload.get("valley_adapted_subspace", {})
        rows.append(
            {
                "kpoint": kpoint,
                "status": diagnostic.get("status"),
                "eta": diagnostic.get("eta"),
                "s_eigenvalues": diagnostic.get("s_eigenvalues"),
                "s_min": diagnostic.get("s_min"),
                "s_max": diagnostic.get("s_max"),
                "valid_valley_subspace": diagnostic.get("valid_valley_subspace"),
            }
        )
    return rows


def _symmetry_summary(symmetry_payload: dict[str, Any]) -> dict[str, Any]:
    rejected: list[dict[str, Any]] = []
    kind_counts: dict[str, int] = {}
    for operation in symmetry_payload.get("detected_operations", []):
        kind = str(operation.get("kind", "unknown"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        for kpoint, reason in operation.get("rejection_reason_by_kpoint", {}).items():
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
        "detected_operation_count": symmetry_payload.get("detected_operation_count", 0),
        "candidate_rotations": symmetry_payload.get("candidate_rotations", []),
        "symprec_scan_summary": symmetry_payload.get("symprec_scan_summary", []),
        "kind_counts": kind_counts,
        "little_group_check": symmetry_payload.get("little_group_check", {}),
        "valley_preservation_check": symmetry_payload.get("valley_preservation_check", {}),
        "rejected_operations": rejected,
    }


def _accepted_rotations(symmetry_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for operation in symmetry_payload.get("detected_operations", []):
        if not operation.get("candidate_rotation", False):
            continue
        for kpoint, little in operation.get("little_group_by_kpoint", {}).items():
            reason = operation.get("rejection_reason_by_kpoint", {}).get(kpoint, "")
            if little and not reason:
                rows.append(
                    {
                        "kpoint": kpoint,
                        "operation_id": operation.get("operation_id"),
                        "order": operation.get("order"),
                        "kind": operation.get("kind"),
                    }
                )
    return rows


def _collect_warnings(
    subspace_payload: dict[str, Any],
    symmetry_payload: dict[str, Any],
    rotation_rows: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if symmetry_payload.get("status") == "skipped":
        warnings.append(str(symmetry_payload.get("reason", "symmetry-operation detection skipped")))
    for kpoint, payload in subspace_payload.get("kpoints", {}).items():
        for warning in payload.get("warnings", []):
            warnings.append(f"{kpoint}: {warning}")
        diagnostic = payload.get("valley_adapted_subspace", {})
        if diagnostic.get("status") == "poor_valley_manifold":
            warnings.append(f"{kpoint}: poor_valley_manifold; rotation eigenvalues are diagnostic-only")
    if any(row.get("basis") != "valley_adapted" for row in rotation_rows):
        warnings.append("Some rotation eigenvalues are not valley-adapted and are diagnostic-only")
    if any(
        bool(row.get("spinor_rotation_applied", False)) and not bool(row.get("spinor_convention_verified", False))
        for row in rotation_rows
    ):
        warnings.append(
            "Spinor rotation is applied, but the VASP spinor convention is not benchmark-verified; "
            "spinful rotation eigenvalues are diagnostic-only"
        )
    if any(bool(row.get("diagnostic_only", False)) for row in rotation_rows):
        warnings.append(
            "Some rotation eigenvalues are diagnostic-only and are not topology_input_ready"
        )
    for operation in symmetry_payload.get("detected_operations", []):
        for kpoint, quality in operation.get("representation_quality", {}).items():
            if isinstance(quality, dict) and quality.get("skipped_reason"):
                warnings.append(f"{kpoint}: rotation {operation.get('operation_id')} skipped: {quality['skipped_reason']}")
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
