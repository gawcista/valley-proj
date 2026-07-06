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
    target_subspace_closure_report: dict[str, Any] | None = None,
    hsp_star_conjugation_report: dict[str, Any] | None = None,
    hsp_star_derived_characters: dict[str, Any] | None = None,
    irrep_workflow_decisions: dict[str, Any] | None = None,
    valley_irrep_matching: dict[str, Any] | None = None,
    ebr_input_candidates: dict[str, Any] | None = None,
    ebr_problem_instances: dict[str, Any] | None = None,
    ebr_export_bundle: dict[str, Any] | None = None,
    reduced_ebr_mapping: dict[str, Any] | None = None,
    valley_projected_representation: dict[str, Any] | None = None,
    folded_center_payload: dict[str, Any] | None = None,
    sampled_k_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eigen_rows = [] if symmetry_rows is None else symmetry_rows
    warnings = _collect_warnings(subspace_payload, symmetry_payload, eigen_rows)
    qcut_payload: dict[str, Any] = {
        "projector_mode": config.projection.projector_mode,
        "mode": config.projection.qcut_mode,
        "value_Ainv": float(qcut),
        "scan": list(config.projection.qcut_scan),
    }
    if config.projection.qcut_mode == "relative_min_valley_distance":
        qcut_payload["fraction"] = float(config.projection.qcut_fraction)
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
        "qcut": qcut_payload,
        "valley_projection_summary": _projection_rows(subspace_payload),
        "valley_subspace_analysis": _subspace_rows(subspace_payload),
        "valley_projector_quality": _projector_quality_rows(subspace_payload),
        "symmetry_analysis": _symmetry_analysis(symmetry_payload, config.analysis.kpoints),
        "symmetry_eigenvalues": eigen_rows,
        "symmetry_characters": _symmetry_character_rows(eigen_rows),
        "rotation_readiness_thresholds": _rotation_readiness_thresholds(config),
        "warnings": warnings,
        "output_profile": config.output.profile,
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
    if target_subspace_closure_report is not None:
        payload["target_subspace_closure"] = target_subspace_closure_report
    if hsp_star_conjugation_report is not None:
        payload["hsp_star_conjugation"] = hsp_star_conjugation_report
    if hsp_star_derived_characters is not None:
        payload["hsp_star_derived_characters"] = hsp_star_derived_characters
    if irrep_workflow_decisions is not None:
        payload["irrep_workflow_decisions"] = irrep_workflow_decisions
    if valley_irrep_matching is not None:
        payload["valley_resolved_irreps"] = _build_valley_resolved_irreps(
            valley_irrep_matching,
        )
        # Full valley_irrep_matching is a debug diagnostic; not exposed
        # in standard profile to avoid duplicating the valley_resolved_irreps
        # compact irrep surface.
        if config.output.profile == "debug":
            payload["valley_irrep_matching"] = valley_irrep_matching
    if ebr_input_candidates is not None:
        payload["valley_ebr_input_candidates"] = ebr_input_candidates
    if ebr_problem_instances is not None:
        payload["valley_ebr_problem_instances"] = ebr_problem_instances
    if ebr_export_bundle is not None:
        payload["valley_ebr_export_bundle"] = ebr_export_bundle
    if reduced_ebr_mapping is not None:
        payload["valley_reduced_ebr_mapping"] = reduced_ebr_mapping
    if valley_projected_representation is not None:
        payload["valley_projected_representations"] = valley_projected_representation
    if folded_center_payload is not None:
        payload["folded_center_report"] = folded_center_payload
    if sampled_k_coverage is not None:
        payload["sampled_k_coverage"] = sampled_k_coverage
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
    lines.append(f"projector mode: {qcut.get('projector_mode', 'fixed_center')}")
    lines.append(f"qcut mode: {qcut['mode']}")
    lines.append(f"qcut value: {_fmt(qcut['value_Ainv'])} A^-1")
    if qcut.get("fraction") is not None:
        lines.append(f"qcut fraction: {_fmt(qcut['fraction'])}")
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

    quality_rows = summary.get("valley_projector_quality", [])
    if quality_rows:
        _section(lines, "Projected q-cut seed projector quality")
        lines.append("rank_estimates: estimated ranks of P_a^sub above the report threshold")
        lines.append("rank_gaps:      lambda_r - lambda_{r+1} using expected rank r")
        lines.append("S-I:            Frobenius norm ||sum_a P_a^sub - I||")
        lines.extend(
            _table(
                [
                    "kpoint", "expected_rank", "rank_estimates", "rank_gaps",
                    "S-I", "max_idemp", "max_overlap", "max_comm",
                ],
                [
                    [
                        row.get("kpoint", ""),
                        row.get("expected_rank", ""),
                        row.get("rank_estimates", ""),
                        row.get("rank_gaps", ""),
                        _fmt(row.get("sum_identity_deviation_fro")),
                        _fmt(row.get("max_idempotency_deviation")),
                        _fmt(row.get("max_trace_overlap")),
                        _fmt(row.get("max_commutator_norm")),
                    ]
                    for row in quality_rows
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
    hsp_star_report = sym.get("hsp_star_report", {})
    if isinstance(hsp_star_report, dict) and hsp_star_report.get("by_kpoint"):
        lines.append("")
        lines.append("HSP-star coverage:")
        lines.append(
            "Note: symmetry-derivable representatives do NOT require additional DFT. "
            "They can be obtained via space-group conjugation from explicit HSP data."
        )
        rows = []
        for kpoint, payload in hsp_star_report.get("by_kpoint", {}).items():
            if not isinstance(payload, dict):
                continue
            rows.append(
                [
                    kpoint,
                    payload.get("status", ""),
                    payload.get("star_size", ""),
                    payload.get("explicit_count", ""),
                    payload.get("symmetry_derivable_count", 0),
                    str(payload.get("requires_additional_dft", False)),
                    _format_hsp_star_representatives(
                        payload.get("symmetry_derivable_representatives", [])
                    ),
                ]
            )
        lines.extend(
            _table(
                [
                    "kpoint", "status", "star", "explicit",
                    "derivable", "extra_dft", "symmetry-derived reps",
                ],
                rows,
            )
        )
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

    target_closure = summary.get("target_subspace_closure")
    if isinstance(target_closure, dict):
        _render_target_subspace_closure(lines, target_closure)

    hsp_star_conj = summary.get("hsp_star_conjugation")
    if isinstance(hsp_star_conj, dict):
        _render_hsp_star_conjugation(lines, hsp_star_conj)

    hsp_star_derived = summary.get("hsp_star_derived_characters")
    if isinstance(hsp_star_derived, dict):
        _render_hsp_star_derived_characters(lines, hsp_star_derived)

    irrep_decisions = summary.get("irrep_workflow_decisions")
    if isinstance(irrep_decisions, dict):
        _render_irrep_workflow_decisions(lines, irrep_decisions)

    irrep_matching = summary.get("valley_irrep_matching")
    if isinstance(irrep_matching, dict):
        _render_valley_irrep_matching(lines, irrep_matching)

    resolved_irreps = summary.get("valley_resolved_irreps")
    if isinstance(resolved_irreps, dict):
        _render_valley_resolved_irreps(lines, resolved_irreps)

    projected_reps = summary.get("valley_projected_representations")
    if isinstance(projected_reps, dict):
        _render_valley_projected_representations(lines, projected_reps)

    ebr_candidates = summary.get("valley_ebr_input_candidates")
    if isinstance(ebr_candidates, dict):
        _render_ebr_input_candidates(lines, ebr_candidates)

    ebr_instances = summary.get("valley_ebr_problem_instances")
    if isinstance(ebr_instances, dict):
        _render_ebr_problem_instances(lines, ebr_instances)

    ebr_bundle = summary.get("valley_ebr_export_bundle")
    if isinstance(ebr_bundle, dict):
        _render_ebr_export_bundle(lines, ebr_bundle)

    ebr_solve = summary.get("valley_reduced_ebr_mapping")
    if isinstance(ebr_solve, dict):
        _render_reduced_ebr_mapping(lines, ebr_solve)

    _section(lines, "Warnings")
    if summary["warnings"]:
        lines.extend(f"- {item}" for item in summary["warnings"])
    else:
        lines.append("None")
    lines.append("")

    _section(lines, "Output files")
    for name, path in summary["output_files"].items():
        lines.append(f"{_output_file_label(name)}: {path}")
    profile = summary.get("output_profile", "standard")
    if profile == "standard":
        lines.append("")
        lines.append(
            "Debug/detail outputs suppressed (output.profile: standard). "
            "Set output.profile: debug in config to enable full diagnostics: "
            "diagnostics.h5, valley_basis_transform.h5, valley_subspace.json, "
            "symmetry_report.json, symmetry_eigenvalues.csv, "
            "projector_symmetry_report.json, symmetry_adapted_valley_analysis.json, "
            "target_subspace_closure.json, hsp_star_conjugation.json, "
            "hsp_star_derived_characters.json, subspace_representation_quality.json, "
            "irrep_workflow_decisions.json, valley_irrep_matching.json, "
            "valley_ebr_input_candidates.json, valley_ebr_problem_instances.json, "
            "folded_center_report.json, sampled_k_coverage.json."
        )
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


def _projector_quality_rows(subspace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kpoint, payload in subspace_payload.get("kpoints", {}).items():
        diagnostic = payload.get("valley_adapted_subspace", {})
        if not isinstance(diagnostic, dict):
            continue
        quality = diagnostic.get("projector_quality", {})
        if not isinstance(quality, dict) or not quality.get("per_valley"):
            continue
        per_valley = quality.get("per_valley", {})
        rank_estimates: list[str] = []
        rank_gaps: list[str] = []
        if isinstance(per_valley, dict):
            for valley, item in per_valley.items():
                if not isinstance(item, dict):
                    continue
                rank_estimates.append(f"{valley}={item.get('rank_estimate', '')}")
                rank_gap = item.get("rank_gap")
                if rank_gap is not None:
                    rank_gaps.append(f"{valley}={_fmt(rank_gap)}")
        sum_projector = quality.get("sum_projector", {})
        if not isinstance(sum_projector, dict):
            sum_projector = {}
        rows.append(
            {
                "kpoint": kpoint,
                "expected_rank": quality.get("expected_rank"),
                "rank_threshold": quality.get("rank_threshold"),
                "rank_estimates": ", ".join(rank_estimates),
                "rank_gaps": ", ".join(rank_gaps),
                "sum_identity_deviation_fro": sum_projector.get("identity_deviation_fro"),
                "sum_idempotency_deviation_fro": sum_projector.get("idempotency_deviation_fro"),
                "max_idempotency_deviation": quality.get("max_idempotency_deviation"),
                "max_trace_overlap": quality.get("max_trace_overlap"),
                "max_commutator_norm": quality.get("max_commutator_norm"),
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
        "hsp_star_report": symmetry_payload.get("hsp_star_report", {}),
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
    # Collect fixed_center_not_captured per-kpoint warnings (deduplicated).
    fc_warned_kpoints: set[str] = set()
    for kpoint, payload in subspace_payload.get("kpoints", {}).items():
        for weight in payload.get("weights", []):
            if weight.get("valley_status") == "fixed_center_not_captured":
                if kpoint not in fc_warned_kpoints:
                    fc_warned_kpoints.add(kpoint)
                    note = weight.get("valley_status_note", "")
                    warnings.append(
                        f"{kpoint}: fixed_center W_val near zero — "
                        f"k/center mismatch, not necessarily non-parent-valley. "
                        f"Consider k_resolved_parent_valley projector_mode."
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
    _section(lines, "Symmetry-adapted valley analysis")
    lines.append("trusted_irrep_label: false")
    lines.append(
        "local_irrep_ready reports internal consistency of this analysis layer; "
        "irrep_matching_input_ready gates the canonical Bilbao/irreptables restricted-character matcher."
    )
    by_kpoint = report.get("by_kpoint", {})
    if not isinstance(by_kpoint, dict) or not by_kpoint:
        lines.append("(none)")
        lines.append("")
        return

    space_group_orbits = report.get("space_group_valley_orbits", [])
    if space_group_orbits:
        lines.append(
            "space-group valley orbits: "
            + "; ".join(_format_orbit(orbit) for orbit in space_group_orbits)
        )

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
        lines.append("HSP-local valley-orbit reports:")
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
                [
                    "kpoint", "valley", "space_group", "sg_ops", "hsp_ops",
                    "rank", "proj", "seed_overlap", "phases", "local_group",
                    "ebr_ready", "ebr_blockers",
                    "local_ready", "irrep_input", "reason",
                ],
                subspace_rows,
            )
        )
        lines.append("")

    # Subspace representation quality (diagnostic-only)
    quality_rows: list[list[Any]] = []
    for kpoint, kp_data in by_kpoint.items():
        if not isinstance(kp_data, dict):
            continue
        for item in kp_data.get("valley_preserving_subspaces", []):
            if not isinstance(item, dict):
                continue
            quality = item.get("subspace_representation_quality")
            if not isinstance(quality, dict):
                continue
            for row in quality.get("rows", []):
                if not isinstance(row, dict):
                    continue
                if not row.get("is_valley_preserving"):
                    continue
                if row.get("operation_id", 0) == 0:
                    continue
                quality_rows.append([
                    kpoint,
                    row.get("valley", ""),
                    row.get("operation_id"),
                    row.get("operation_order", ""),
                    _fmt(row.get("basis_orthonormality_error")),
                    _fmt(row.get("D_raw_unitarity_error")),
                    _fmt(row.get("projector_invariance_error")),
                    _fmt(row.get("local_representation_unitarity_error")),
                    _fmt(row.get("local_group_relation_error")),
                    _fmt(row.get("eigenvalue_modulus_deviation")),
                    row.get("diagnosis", ""),
                ])
    if quality_rows:
        lines.append("subspace representation quality (diagnostic-only):")
        lines.extend(
            _table(
                [
                    "kpoint", "valley", "op", "order",
                    "basis_ortho", "D_raw_unit", "proj_inv",
                    "local_unit", "group_rel", "eval_mod_dev",
                    "diagnosis",
                ],
                quality_rows,
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
    subspace_group = item.get("subspace_group", {})
    if not isinstance(subspace_group, dict):
        subspace_group = {}
    ebr_input = item.get("ebr_mapping_input", {})
    if not isinstance(ebr_input, dict):
        ebr_input = {}
    subspace_space_group = item.get("subspace_space_group", {})
    if not isinstance(subspace_space_group, dict):
        subspace_space_group = {}
    return [
        kpoint,
        _format_orbit(item.get("orbit")),
        subspace_space_group.get("candidate_space_group_symbol", ""),
        _short_list(subspace_space_group.get("valley_preserving_operation_ids")),
        _short_list(item.get("hsp_preserving_operation_ids")),
        projectors.get("selected_rank", ""),
        projectors.get("status", ""),
        _format_seed_overlap(projectors.get("seed_overlap")),
        _format_character_phases(item.get("valley_preserving_character_diagnostics")),
        subspace_group.get("subspace_group_candidate", "") or subspace_group.get("effective_point_group", ""),
        ebr_input.get("ready", ""),
        _short_list(ebr_input.get("blocked_by")),
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


def _format_hsp_star_representatives(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    terms: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        frac = item.get("canonical_frac")
        ops = item.get("generated_by_operation_ids", [])
        terms.append(f"{_short_list(frac)} via ops {_short_list(ops)}")
    return "; ".join(terms)


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


def _short_irrep_multiplicities(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    return ", ".join(f"{label}:{mult}" for label, mult in sorted(value.items()))


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
    if value == "fixed_center_not_captured":
        return "fixed_center_not_captured"
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


def _render_target_subspace_closure(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "Target-subspace symmetry closure")
    lines.append(f"status: {report.get('status', 'no_data')}")
    lines.append(
        f"tolerances: unitarity={report.get('unitarity_tol')}, "
        f"group_relation={report.get('group_relation_tol')}"
    )
    failed_count = 0
    warn_count = 0
    detail_rows: list[list[Any]] = []
    for kpoint, rows in report.get("by_kpoint", {}).items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            status = str(row.get("status", ""))
            if status == "failed":
                failed_count += 1
            elif status == "warn":
                warn_count += 1
            # Build detail row for non-ok entries
            if status in ("failed", "warn") and row.get("is_valley_preserving", True):
                detail_rows.append([
                    kpoint,
                    row.get("operation_id"),
                    row.get("closure_quality", ""),
                    row.get("classification", ""),
                    _fmt(row.get("raw_unitarity_error")),
                    _fmt(row.get("max_closure_residual")),
                    _fmt(row.get("target_wavefunction_gram_error")),
                    _fmt(row.get("mapping_miss_count")),
                    row.get("worst_source_state", "") if row.get("worst_source_state") is not None else "",
                    "N/A" if row.get("closure_residual_by_source_state") is None
                    else _short_list(row.get("closure_residual_by_source_state")),
                ])
    if detail_rows:
        lines.extend(
            _table(
                ["kpoint", "op", "quality", "classification", "unit_err",
                 "max_residual", "gram_err", "miss_count", "worst_state",
                 "residuals"],
                detail_rows,
            )
        )
    if failed_count == 0 and warn_count == 0:
        lines.append("all operations: target subspace closed")
    elif failed_count > 0:
        lines.append(
            f"target_subspace_closure_failed: {failed_count} operation(s) "
            f"have non-unitary or non-closed D_raw in the target subspace"
        )
    lines.append("")


def _render_hsp_star_conjugation(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "HSP-star conjugation")
    lines.append(f"status: {report.get('status', 'not_evaluated')}")
    by_source = report.get("by_source_kpoint", {})
    if not isinstance(by_source, dict) or not by_source:
        lines.append("(none)")
        lines.append("")
        return
    for source_kp, entries in by_source.items():
        matched = [e for e in entries if e.get("conjugation_status") == "matched"]
        missing = [e for e in entries if e.get("conjugation_status") == "missing_operation_product"]
        antiunitary = [e for e in entries if e.get("conjugation_status") == "antiunitary_not_implemented"]
        lines.append(
            f"{source_kp}: {len(matched)} matched, {len(missing)} missing, "
            f"{len(antiunitary)} antiunitary-not-implemented"
        )
        for e in matched:
            lines.append(
                f"  {e.get('source_valley')} -> {e.get('target_valley')} "
                f"@ {e.get('target_kpoint_label')}: "
                f"g={e.get('source_preserving_operation_id')} "
                f"-> h={e.get('derived_target_operation_id')} "
                f"via r={e.get('mapping_operation_id')}"
            )
    lines.append("")


def _render_hsp_star_derived_characters(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "HSP-star derived characters")
    lines.append(f"status: {report.get('status', 'not_evaluated')}")
    lines.append(f"derivation_type: {report.get('derivation_type', '')}")
    lines.append(f"antiunitary_status: {report.get('antiunitary_status', '')}")
    derived = [e for e in report.get("entries", []) if e.get("status") == "derived"]
    diag = [e for e in report.get("entries", []) if e.get("status") == "diagnostic_only"]
    blocked = [e for e in report.get("entries", []) if "blocked" in str(e.get("status", ""))]
    not_impl = [e for e in report.get("entries", []) if e.get("status") == "not_implemented"]
    missing = [e for e in report.get("entries", []) if "missing" in str(e.get("status", ""))]

    lines.append(
        f"derived: {len(derived)}, diagnostic_only: {len(diag)}, "
        f"blocked: {len(blocked)}, not_implemented: {len(not_impl)}, "
        f"missing: {len(missing)}"
    )
    for e in derived:
        char = e.get("character", {})
        char_str = ""
        if isinstance(char, dict):
            char_str = f"{char.get('real', 0)}+{char.get('imag', 0)}i"
        lines.append(
            f"  {e.get('target_kpoint_label')}/{e.get('target_valley')} "
            f"op={e.get('derived_target_operation_id')}: "
            f"chi={char_str}, trusted={e.get('trusted_for_ebr_input')}"
        )
    if not derived and not diag and not blocked:
        lines.append("(no derived characters available)")
    blocked_sources = report.get("blocked_sources", [])
    if blocked_sources:
        lines.append("blocked sources:")
        for bs in blocked_sources:
            lines.append(f"  {bs.get('source_kpoint')}/{bs.get('source_valley')}: {bs.get('reason')}")
    lines.append("")


def _render_irrep_workflow_decisions(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "Irrep workflow decisions")
    paths = report.get("workflow_paths", [])
    levels = report.get("readiness_levels", [])
    lines.append(f"paths: {', '.join(paths)}")
    lines.append(f"levels: {', '.join(levels)}")
    by_kpoint = report.get("by_kpoint", {})
    if not isinstance(by_kpoint, dict) or not by_kpoint:
        lines.append("(none)")
        lines.append("")
        return
    rows: list[list[Any]] = []
    for kp_name, valleys in by_kpoint.items():
        if not isinstance(valleys, dict):
            continue
        for v_name, d in valleys.items():
            if not isinstance(d, dict):
                continue
            rows.append([
                kp_name, v_name,
                d.get("workflow_path", ""),
                d.get("readiness_level", ""),
                d.get("uses_symmetry_adapted_projector", ""),
                d.get("direct_qcut_allowed", ""),
                d.get("reason", "")[:80],
            ])
    if rows:
        lines.extend(
            _table(
                ["kpoint", "valley", "path", "readiness",
                 "uses_sym_adapt", "qcut_allowed", "reason"],
                rows,
            )
        )
    lines.append("")


def _build_valley_resolved_irreps(
    matching: dict[str, Any],
) -> dict[str, Any]:
    """Build compact valley-resolved irrep summary from generic matching data.

    Extracts generic restricted-character matches and summarizes per-(kpoint, valley)
    rows with subspace space group, HSP little group, valley-preserving subgroup,
    matching strategy/status, and irrep multiplicities.
    """
    generic_by_kp = matching.get("generic_matches_by_kpoint", {})
    if not isinstance(generic_by_kp, dict) or not generic_by_kp:
        return {
            "status": "no_generic_irrep_data",
            "matched_count": 0,
            "blocked_count": 0,
            "diagnostic_count": 0,
            "rows": [],
        }

    rows: list[dict[str, Any]] = []
    matched = 0
    blocked = 0
    diagnostic = 0

    for kp_name in sorted(generic_by_kp):
        valleys = generic_by_kp.get(kp_name, {})
        if not isinstance(valleys, dict):
            continue
        for v_name in sorted(valleys):
            gm = valleys.get(v_name, {})
            if not isinstance(gm, dict):
                continue
            status = str(gm.get("matching_status", ""))
            irrep_mults = gm.get("irrep_multiplicities", {})
            ssg = gm.get("subspace_space_group", {})
            vp_ids = gm.get("valley_preserving_operation_ids", [])
            subspace_hsp_ids = vp_ids

            row: dict[str, Any] = {
                "kpoint": kp_name,
                "valley": v_name,
                "subspace_space_group": ssg.get("candidate_space_group_symbol") if isinstance(ssg, dict) else None,
                "subspace_hsp_little_group_operation_ids": list(subspace_hsp_ids) if isinstance(subspace_hsp_ids, list) else [],
                "hsp_little_group_operation_ids": list(subspace_hsp_ids) if isinstance(subspace_hsp_ids, list) else [],
                "valley_preserving_operation_ids": list(vp_ids) if isinstance(vp_ids, list) else [],
                "matching_strategy": gm.get("matching_strategy"),
                "matching_status": status,
                "irrep_multiplicities": dict(irrep_mults) if isinstance(irrep_mults, dict) else {},
                "local_representation_dimension": gm.get("local_representation_dimension"),
                "readiness_level": gm.get("readiness_level"),
                "workflow_path": gm.get("workflow_path"),
                "diagnostic_only": bool(gm.get("diagnostic_only", False)),
                "reason": str(gm.get("reason", ""))[:120] if gm.get("reason") else "",
            }
            rows.append(row)
            if status == "matched":
                matched += 1
            elif status == "blocked":
                blocked += 1
            else:
                diagnostic += 1

    return {
        "status": "ok",
        "matching_mode": matching.get("matching_mode", "not_evaluated"),
        "matched_count": matched,
        "blocked_count": blocked,
        "diagnostic_count": diagnostic,
        "rows": rows,
    }


def _render_valley_resolved_irreps(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "Valley-resolved irreps")
    lines.append(f"status: {report.get('status', 'no_data')}")
    lines.append(f"matched: {report.get('matched_count', 0)}")
    lines.append(f"blocked: {report.get('blocked_count', 0)}")
    lines.append(f"diagnostic-only: {report.get('diagnostic_count', 0)}")
    rows = report.get("rows", [])
    if not rows:
        lines.append("(no generic irrep data)")
        lines.append("")
        return
    table_rows: list[list[Any]] = []
    for row in rows:
        table_rows.append([
            row.get("kpoint", ""),
            row.get("valley", ""),
            row.get("subspace_space_group") or "?",
            _short_list(row.get("valley_preserving_operation_ids", [])),
            row.get("matching_strategy", ""),
            row.get("matching_status", ""),
            _short_irrep_multiplicities(row.get("irrep_multiplicities")),
            row.get("readiness_level", ""),
            row.get("workflow_path", ""),
            row.get("reason", "")[:80],
        ])
    lines.extend(
        _table(
            ["kpoint", "valley", "subspace_sg", "vp_ops", "strategy",
             "status", "irreps", "readiness", "path", "reason"],
            table_rows,
        )
    )
    lines.append("")


def _render_valley_irrep_matching(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "Valley irrep matching")
    lines.append(f"status: {report.get('status', 'not_evaluated')}")
    lines.append(f"mode: {report.get('matching_mode', 'not_evaluated')}")

    generic_by_kpoint = report.get("generic_matches_by_kpoint", {})
    generic_rows: list[list[Any]] = []
    if isinstance(generic_by_kpoint, dict):
        for kp_name, valleys in generic_by_kpoint.items():
            if not isinstance(valleys, dict):
                continue
            for v_name, match in valleys.items():
                if not isinstance(match, dict):
                    continue
                subspace_sg = match.get("subspace_space_group", {})
                if not isinstance(subspace_sg, dict):
                    subspace_sg = {}
                generic_rows.append([
                    kp_name,
                    v_name,
                    match.get("matching_strategy", ""),
                    match.get("matching_status", ""),
                    _short_irrep_multiplicities(match.get("irrep_multiplicities")),
                    subspace_sg.get("candidate_space_group_symbol", ""),
                    _short_list(match.get("valley_preserving_operation_ids")),
                    str(match.get("reason", ""))[:80],
                    match.get("readiness_level", ""),
                ])
    if generic_rows:
        lines.append("generic restricted-character matches:")
        lines.extend(
            _table(
                ["kpoint", "valley", "strategy", "status", "irreps",
                 "subspace_sg", "vp_ops", "reason", "readiness"],
                generic_rows,
            )
        )

    if not generic_rows:
        lines.append("(none)")
        lines.append("")
        return
    lines.append("")


def _render_valley_projected_representations(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "Valley-projected representations")
    lines.append(
        "primary object: valley-projected subspace space group and HSP little-group representation"
    )
    lines.append(f"trusted entries: {report.get('trusted_representation_count', 0)}")
    lines.append(f"blocked entries: {report.get('blocked_representation_count', 0)}")
    lines.append(f"diagnostic-only entries: {report.get('diagnostic_only_count', 0)}")
    lines.append(f"grouped records: {report.get('grouped_record_count', 0)}")
    sg_counts = report.get("subspace_space_group_counts", {})
    if isinstance(sg_counts, dict) and sg_counts:
        parts = [f"{label}={count}" for label, count in sorted(sg_counts.items())]
        lines.append(f"subspace space groups: {', '.join(parts)}")
    rows = report.get("rows", [])
    if isinstance(rows, list) and rows:
        table_rows: list[list[Any]] = []
        for row in rows[:12]:
            if not isinstance(row, dict):
                continue
            subspace_sg = row.get("subspace_space_group", {})
            if not isinstance(subspace_sg, dict):
                subspace_sg = {}
            table_rows.append([
                row.get("kpoint", ""),
                row.get("valley", ""),
                row.get("operation_id", ""),
                subspace_sg.get("candidate_space_group_symbol", ""),
                _short_list(row.get("hsp_little_group_operation_ids")),
                _short_list(row.get("valley_preserving_operation_ids")),
                row.get("readiness_level", ""),
                row.get("workflow_path", ""),
                _short_list(row.get("blocking_reasons")),
            ])
        if table_rows:
            lines.extend(
                _table(
                    [
                        "kpoint", "valley", "op", "subspace_sg", "hsp_ops",
                        "vp_ops", "readiness", "path", "blockers",
                    ],
                    table_rows,
                )
            )
            if len(rows) > len(table_rows):
                lines.append(f"... {len(rows) - len(table_rows)} additional rows omitted")
    lines.append("")


def _render_ebr_input_candidates(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "EBR input candidates")
    lines.append(f"status: {report.get('status', 'no_data')}")
    lines.append(f"candidates: {report.get('candidate_count', 0)}")
    lines.append(f"blocked: {report.get('blocked_count', 0)}")
    lines.append(
        f"reduced EBR decomposition: "
        f"{report.get('reduced_ebr_decomposition_status', 'not_implemented')}"
    )
    cands = report.get("candidates", [])
    if isinstance(cands, list) and cands:
        rows: list[list[Any]] = []
        for c in cands:
            rows.append([
                c.get("kpoint", ""), c.get("valley", ""),
                c.get("operation_id", ""), c.get("operation_order", ""),
                c.get("matched_irrep", "") or "",
                _short_list(c.get("eigenphases", [])),
                c.get("workflow_path", ""),
                c.get("readiness_level", ""),
            ])
        lines.extend(
            _table(["kpoint", "valley", "op", "order", "irrep",
                    "phases", "path", "readiness"], rows)
        )
    blocked_rows = report.get("blocked", [])
    if isinstance(blocked_rows, list) and blocked_rows:
        lines.append("blocked/not-ready:")
        for b in blocked_rows[:10]:
            lines.append(
                f"  {b.get('kpoint','')}/{b.get('valley','')} "
                f"op={b.get('operation_id','')}: "
                f"{b.get('reason','')[:100]}"
            )
    lines.append("")


def _render_ebr_problem_instances(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "EBR problem instances")
    lines.append(f"status: {report.get('status', 'no_data')}")
    lines.append(f"instance count: {report.get('instance_count', 0)}")
    lines.append(
        f"reduced EBR decomposition: "
        f"{report.get('reduced_ebr_decomposition_status', 'not_implemented')}"
    )
    instances = report.get("instances", [])
    if isinstance(instances, list) and instances:
        rows: list[list[Any]] = []
        for inst in instances:
            rows.append([
                inst.get("instance_id", ""),
                inst.get("valley", ""),
                _canonical_sg_display(inst),
                inst.get("status", ""),
                str(inst.get("ready_for_ebr_decomposition", "")),
                _short_list(inst.get("blocked_by", [])),
                _short_list(inst.get("expected_hsps", [])),
                _short_list(inst.get("missing_optional_hsps", [])),
                _short_list(inst.get("actual_hsps", [])),
            ])
        lines.extend(
            _table(
                ["id", "valley", "group", "status", "ready",
                 "blocked_by", "expected_hsp", "missing_optional", "actual_hsp"],
                rows,
            )
        )
    lines.append("")


def _render_ebr_export_bundle(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "EBR export bundle")
    lines.append(f"status: {report.get('status', 'no_data')}")
    lines.append(f"schema version: {report.get('schema_version', '')}")
    lines.append(f"bundles: {report.get('bundle_count', 0)}")
    lines.append(f"excluded: {report.get('excluded_count', 0)}")
    lines.append(
        f"reduced EBR decomposition: "
        f"{report.get('reduced_ebr_decomposition_status', 'not_implemented')}"
    )
    bundles = report.get("bundles", [])
    if isinstance(bundles, list) and bundles:
        rows: list[list[Any]] = []
        for b in bundles:
            rows.append([
                b.get("bundle_id", ""),
                b.get("valley", ""),
                _canonical_sg_display(b),
                b.get("workflow_path", ""),
                _short_list(b.get("expected_hsps", [])),
                _short_list(b.get("optional_hsps", [])),
                _short_list(b.get("missing_optional_hsps", [])),
                b.get("ready_for_external_solver", ""),
            ])
        lines.extend(
            _table(
                ["bundle_id", "valley", "group", "path",
                 "expected_hsp", "optional_hsp", "missing_opt",
                 "ext_solver_ready"],
                rows,
            )
        )
    excluded = report.get("excluded_instances", [])
    if isinstance(excluded, list) and excluded:
        lines.append("excluded instances:")
        for e in excluded:
            lines.append(
                f"  {e.get('source_instance_id','')} "
                f"{e.get('valley','')}/{_canonical_sg_display(e)}: "
                f"{'; '.join(e.get('exclusion_reasons', []))[:120]}"
            )
    lines.append("")


def _render_reduced_ebr_mapping(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    _section(lines, "Reduced EBR mapping")
    lines.append(f"status: {report.get('status', 'no_data')}")
    lines.append(f"mapping status: {report.get('mapping_status', report.get('status', ''))}")
    lines.append(f"table: {report.get('table_status', '')}")
    lines.append(
        f"reduced EBR decomposition: "
        f"{report.get('reduced_ebr_decomposition_status', '')}"
    )

    solutions = report.get("solutions", [])
    if isinstance(solutions, list) and solutions:
        # Classification counts.
        atomic = sum(1 for s in solutions if isinstance(s, dict)
                     and s.get("classification") == "atomic-compatible-candidate")
        fragile = sum(1 for s in solutions if isinstance(s, dict)
                      and s.get("classification") == "fragile-topology-candidate")
        stable = sum(1 for s in solutions if isinstance(s, dict)
                     and s.get("classification") == "stable-topology-candidate")
        truncated = sum(1 for s in solutions if isinstance(s, dict)
                        and s.get("search_status") == "truncated_by_max_coefficient")
        lines.append(f"classifications: atomic-compatible={atomic}, "
                     f"fragile-topology={fragile}, "
                     f"stable-topology={stable}"
                     + (f", search_truncated={truncated}" if truncated else ""))

        for sol in solutions:
            if not isinstance(sol, dict):
                continue
            bid = sol.get("bundle_id", "?")
            val = sol.get("valley", "")
            label = f"  {bid} {val}"
            classification = sol.get("classification", "")

            if classification == "atomic-compatible-candidate":
                decomp = sol.get("ebr_decomposition")
                if isinstance(decomp, list) and decomp:
                    terms = [
                        f"{e.get('label', '')} x {e.get('coefficient', '')}"
                        for e in decomp if isinstance(e, dict)
                    ]
                    lines.append(f"{label}: atomic-compatible [{', '.join(terms)}]")
                else:
                    lines.append(f"{label}: atomic-compatible (no decomposition)")
            elif classification == "fragile-topology-candidate":
                witness = sol.get("integer_solution")
                if isinstance(witness, list) and witness:
                    terms = [
                        f"{e.get('label', '')}: {e.get('coefficient', '')}"
                        for e in witness if isinstance(e, dict)
                    ]
                    lines.append(f"{label}: fragile-topology [signed witness: {', '.join(terms)}]")
                else:
                    lines.append(f"{label}: fragile-topology (in integer span)")
            elif classification == "stable-topology-candidate":
                lines.append(f"{label}: stable-topology (outside integer span)")
            else:
                # Legacy rows without classification.
                decomp = sol.get("ebr_decomposition")
                if isinstance(decomp, list) and decomp:
                    terms = [
                        f"{e.get('label', '')} x {e.get('coefficient', '')}"
                        for e in decomp if isinstance(e, dict)
                    ]
                    lines.append(f"{label}: {' + '.join(terms)}")
                else:
                    lines.append(f"{label}: {sol.get('status', '?')}")

            if sol.get("search_status") == "truncated_by_max_coefficient":
                lines.append(f"         (search truncated by max_coefficient)")

    excluded = report.get("excluded_bundles", [])
    if isinstance(excluded, list) and excluded:
        lines.append("excluded bundles:")
        for e in excluded:
            lines.append(f"  {e.get('bundle_id', '')}: {e.get('reason', '')}")
    lines.append("")


OUTPUT_FILE_LABELS: dict[str, str] = {
    "valley_summary_txt": "Human-readable summary",
    "valley_summary_json": "Machine-readable summary",
    "valley_weights_csv": "Valley weights",
    "valley_subspace_json": "Valley subspace analysis",
    "valley_basis_transform_h5": "Valley basis transform",
    "symmetry_report_json": "Symmetry analysis",
    "symmetry_eigenvalues_csv": "Symmetry eigenvalues",
    "diagnostics_h5": "Projector, qcut, and symmetry matrices",
    "projector_symmetry_report_json": "Projector symmetry report",
    "symmetry_adapted_valley_analysis_json": "Symmetry-adapted valley analysis",
    "valley_ebr_input_candidates_json": "Valley EBR input candidates",
    "valley_ebr_problem_instances_json": "Valley EBR problem instances",
    "valley_ebr_export_bundle_json": "Valley EBR export bundle",
    "valley_reduced_ebr_mapping_json": "Valley reduced EBR mapping",
    "target_subspace_closure_json": "Target subspace closure",
    "hsp_star_conjugation_json": "HSP star conjugation",
    "hsp_star_derived_characters_json": "HSP star derived characters",
    "subspace_representation_quality_json": "Subspace representation quality",
    "irrep_workflow_decisions_json": "Irrep workflow decisions",
    "valley_irrep_matching_json": "Valley irrep matching",
    "folded_center_report_json": "Folded-center report",
    "sampled_k_coverage_json": "Sampled k-point coverage",
}


def _output_file_label(name: str) -> str:
    return OUTPUT_FILE_LABELS.get(name, name.replace("_", " ").title())


def _canonical_sg_display(record: dict[str, Any]) -> str:
    """Return the canonical subgroup symbol for display.

    Prefers ``subspace_space_group.candidate_space_group_symbol`` (the
    primary physical identifier); falls back to the flat derived
    ``subspace_group_candidate`` scalar key.
    """
    ssg = record.get("subspace_space_group", {})
    if isinstance(ssg, dict):
        symbol = ssg.get("candidate_space_group_symbol")
        if symbol and isinstance(symbol, str):
            return str(symbol)
    return str(record.get("subspace_group_candidate", ""))


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
