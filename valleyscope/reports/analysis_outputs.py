from __future__ import annotations

from pathlib import Path
from typing import Any

from valleyscope.io.config import AppConfig
from valleyscope.projection.sector_projectors import SectorProjectors
from valleyscope.reports.csv_report import (
    write_symmetry_eigenvalues_csv,
    write_valley_weights_csv,
)
from valleyscope.reports.h5_report import write_basis_transform_h5, write_diagnostics_h5
from valleyscope.reports.json_report import write_json
from valleyscope.reports.summary_report import (
    build_summary_payload,
    render_summary_text,
    write_summary_json,
    write_summary_text,
)


def write_analysis_outputs(
    *,
    config: AppConfig,
    qcut: float,
    weight_rows: list[dict[str, object]],
    sector_names: list[str],
    subspace_payload: dict[str, Any],
    symmetry_payload: dict[str, Any],
    symmetry_rows: list[dict[str, object]],
    projectors_by_kpoint: dict[str, SectorProjectors],
    qcut_scan_payload: dict[str, object],
    symmetry_representation_payload: dict[str, object],
    basis_transforms: dict[str, dict[str, object]],
    symmetry_eigenvalue_summary: dict[str, object] | None = None,
    projector_symmetry_report: dict[str, object] | None = None,
    symmetry_adapted_valley_report: dict[str, object] | None = None,
    target_subspace_closure_report: dict[str, object] | None = None,
    hsp_star_conjugation_report: dict[str, object] | None = None,
    hsp_star_derived_characters: dict[str, object] | None = None,
    irrep_workflow_decisions: dict[str, object] | None = None,
    valley_irrep_matching: dict[str, object] | None = None,
    ebr_input_candidates: dict[str, object] | None = None,
    ebr_problem_instances: dict[str, object] | None = None,
    ebr_export_bundle: dict[str, object] | None = None,
    reduced_ebr_mapping: dict[str, object] | None = None,
    valley_projected_representation: dict[str, object] | None = None,
    folded_center_payload: dict[str, object] | None = None,
    sampled_k_coverage: dict[str, object] | None = None,
) -> dict[str, object]:
    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, object] = {}
    is_debug = config.output.profile == "debug"

    # --- Public / always-write outputs ---
    # EBR export bundle is a public downstream entry (when payload exists).
    if ebr_export_bundle is not None:
        outputs["valley_ebr_export_bundle_json"] = write_json(
            output_dir / "valley_ebr_export_bundle.json",
            ebr_export_bundle,
        )
    # Reduced EBR mapping is public when enabled and payload exists.
    if reduced_ebr_mapping is not None:
        outputs["valley_reduced_ebr_mapping_json"] = write_json(
            output_dir / "valley_reduced_ebr_mapping.json",
            reduced_ebr_mapping,
        )
    # Valley weights CSV is a quick-scan file; included in standard profile.
    if config.output.write_csv and weight_rows:
        outputs["valley_weights_csv"] = write_valley_weights_csv(
            output_dir / "valley_weights.csv",
            weight_rows,
            sector_names,
        )

    # --- Debug / detail outputs ---
    if is_debug:
        _write_detailed_outputs(
            config=config,
            output_dir=output_dir,
            outputs=outputs,
            weight_rows=weight_rows,
            sector_names=sector_names,
            subspace_payload=subspace_payload,
            symmetry_payload=symmetry_payload,
            symmetry_rows=symmetry_rows,
            projectors_by_kpoint=projectors_by_kpoint,
            qcut_scan_payload=qcut_scan_payload,
            symmetry_representation_payload=symmetry_representation_payload,
            basis_transforms=basis_transforms,
            projector_symmetry_report=projector_symmetry_report,
            symmetry_adapted_valley_report=symmetry_adapted_valley_report,
            target_subspace_closure_report=target_subspace_closure_report,
            hsp_star_conjugation_report=hsp_star_conjugation_report,
            hsp_star_derived_characters=hsp_star_derived_characters,
            irrep_workflow_decisions=irrep_workflow_decisions,
            valley_irrep_matching=valley_irrep_matching,
            ebr_input_candidates=ebr_input_candidates,
            ebr_problem_instances=ebr_problem_instances,
            ebr_export_bundle=ebr_export_bundle,
            reduced_ebr_mapping=reduced_ebr_mapping,
        valley_projected_representation=valley_projected_representation,
            folded_center_payload=folded_center_payload,
            sampled_k_coverage=sampled_k_coverage,
        )
    # --- Summary outputs (always written) ---
    _write_summary_outputs(
        config=config,
        qcut=qcut,
        output_dir=output_dir,
        outputs=outputs,
        subspace_payload=subspace_payload,
        symmetry_payload=symmetry_payload,
        symmetry_rows=symmetry_rows,
        symmetry_eigenvalue_summary=symmetry_eigenvalue_summary,
        projector_symmetry_report=projector_symmetry_report,
        symmetry_adapted_valley_report=symmetry_adapted_valley_report,
        target_subspace_closure_report=target_subspace_closure_report,
        hsp_star_conjugation_report=hsp_star_conjugation_report,
        hsp_star_derived_characters=hsp_star_derived_characters,
        irrep_workflow_decisions=irrep_workflow_decisions,
        valley_irrep_matching=valley_irrep_matching,
        ebr_input_candidates=ebr_input_candidates,
        ebr_problem_instances=ebr_problem_instances,
        ebr_export_bundle=ebr_export_bundle,
        reduced_ebr_mapping=reduced_ebr_mapping,
        valley_projected_representation=valley_projected_representation,
        folded_center_payload=folded_center_payload,
        sampled_k_coverage=sampled_k_coverage,
    )
    return outputs


def _write_detailed_outputs(
    *,
    config: AppConfig,
    output_dir: Path,
    outputs: dict[str, object],
    weight_rows: list[dict[str, object]],
    sector_names: list[str],
    subspace_payload: dict[str, Any],
    symmetry_payload: dict[str, Any],
    symmetry_rows: list[dict[str, object]],
    projectors_by_kpoint: dict[str, SectorProjectors],
    qcut_scan_payload: dict[str, object],
    symmetry_representation_payload: dict[str, object],
    basis_transforms: dict[str, dict[str, object]],
    projector_symmetry_report: dict[str, object] | None = None,
    symmetry_adapted_valley_report: dict[str, object] | None = None,
    target_subspace_closure_report: dict[str, object] | None = None,
    hsp_star_conjugation_report: dict[str, object] | None = None,
    hsp_star_derived_characters: dict[str, object] | None = None,
    irrep_workflow_decisions: dict[str, object] | None = None,
    valley_irrep_matching: dict[str, object] | None = None,
    ebr_input_candidates: dict[str, object] | None = None,
    ebr_problem_instances: dict[str, object] | None = None,
    ebr_export_bundle: dict[str, object] | None = None,
    reduced_ebr_mapping: dict[str, object] | None = None,
    valley_projected_representation: dict[str, object] | None = None,
    folded_center_payload: dict[str, object] | None = None,
    sampled_k_coverage: dict[str, object] | None = None,
) -> None:
    if config.output.write_csv:
        if symmetry_payload.get("symmetry_eigenvalue_enabled", False):
            outputs["symmetry_eigenvalues_csv"] = write_symmetry_eigenvalues_csv(
                output_dir / "symmetry_eigenvalues.csv",
                symmetry_rows,
            )
    if config.output.write_json:
        outputs["valley_subspace_json"] = write_json(output_dir / "valley_subspace.json", subspace_payload)
        outputs["symmetry_report_json"] = write_json(output_dir / "symmetry_report.json", symmetry_payload)
        if projector_symmetry_report is not None:
            outputs["projector_symmetry_report_json"] = write_json(
                output_dir / "projector_symmetry_report.json", projector_symmetry_report
            )
        if symmetry_adapted_valley_report is not None:
            outputs["symmetry_adapted_valley_analysis_json"] = write_json(
                output_dir / "symmetry_adapted_valley_analysis.json",
                symmetry_adapted_valley_report,
            )
        if target_subspace_closure_report is not None:
            outputs["target_subspace_closure_json"] = write_json(
                output_dir / "target_subspace_closure.json",
                target_subspace_closure_report,
            )
        if hsp_star_conjugation_report is not None:
            outputs["hsp_star_conjugation_json"] = write_json(
                output_dir / "hsp_star_conjugation.json",
                hsp_star_conjugation_report,
            )
        if hsp_star_derived_characters is not None:
            outputs["hsp_star_derived_characters_json"] = write_json(
                output_dir / "hsp_star_derived_characters.json",
                hsp_star_derived_characters,
            )
        if symmetry_adapted_valley_report is not None and config.symmetry_adapted_valley.write_subspace_representation_quality:
            quality_json_path = output_dir / "subspace_representation_quality.json"
            quality_report = _extract_quality_report(symmetry_adapted_valley_report)
            if quality_report is not None:
                outputs["subspace_representation_quality_json"] = write_json(
                    quality_json_path, quality_report,
                )
        if irrep_workflow_decisions is not None:
            outputs["irrep_workflow_decisions_json"] = write_json(
                output_dir / "irrep_workflow_decisions.json",
                irrep_workflow_decisions,
            )
        if valley_irrep_matching is not None:
            outputs["valley_irrep_matching_json"] = write_json(
                output_dir / "valley_irrep_matching.json",
                valley_irrep_matching,
            )
        if ebr_input_candidates is not None:
            outputs["valley_ebr_input_candidates_json"] = write_json(
                output_dir / "valley_ebr_input_candidates.json",
                ebr_input_candidates,
            )
        if ebr_problem_instances is not None:
            outputs["valley_ebr_problem_instances_json"] = write_json(
                output_dir / "valley_ebr_problem_instances.json",
                ebr_problem_instances,
            )
        if folded_center_payload is not None:
            outputs["folded_center_report_json"] = write_json(
                output_dir / "folded_center_report.json",
                folded_center_payload,
            )
        if sampled_k_coverage is not None:
            outputs["sampled_k_coverage_json"] = write_json(
                output_dir / "sampled_k_coverage.json",
                sampled_k_coverage,
            )
    outputs["diagnostics_h5"] = write_diagnostics_h5(
        output_dir / "diagnostics.h5",
        projectors_by_kpoint,
        qcut_scan_payload,
        symmetry_representation_payload,
        symmetry_payload,
        projector_mode=config.projection.projector_mode,
        center_weight_rows=weight_rows if config.projection.projector_mode == "k_resolved_parent_valley" else None,
    )
    if config.output.write_hdf5_basis_transform:
        outputs["valley_basis_transform_h5"] = write_basis_transform_h5(
            output_dir / "valley_basis_transform.h5",
            basis_transforms,
        )


def _write_summary_outputs(
    *,
    config: AppConfig,
    qcut: float,
    output_dir: Path,
    outputs: dict[str, object],
    subspace_payload: dict[str, Any],
    symmetry_payload: dict[str, Any],
    symmetry_rows: list[dict[str, object]],
    symmetry_eigenvalue_summary: dict[str, object] | None = None,
    projector_symmetry_report: dict[str, object] | None = None,
    symmetry_adapted_valley_report: dict[str, object] | None = None,
    target_subspace_closure_report: dict[str, object] | None = None,
    hsp_star_conjugation_report: dict[str, object] | None = None,
    hsp_star_derived_characters: dict[str, object] | None = None,
    irrep_workflow_decisions: dict[str, object] | None = None,
    valley_irrep_matching: dict[str, object] | None = None,
    ebr_input_candidates: dict[str, object] | None = None,
    ebr_problem_instances: dict[str, object] | None = None,
    ebr_export_bundle: dict[str, object] | None = None,
    reduced_ebr_mapping: dict[str, object] | None = None,
    valley_projected_representation: dict[str, object] | None = None,
    folded_center_payload: dict[str, object] | None = None,
    sampled_k_coverage: dict[str, object] | None = None,
) -> None:
    # valley_summary.txt/json are the main user entry.  In standard profile
    # they are always written.  In debug profile the write_summary_* flags
    # may suppress one format, but the default for both flags is True.
    summary_path_plan: dict[str, Path] = {}
    write_txt = config.output.write_summary_txt or config.output.profile == "standard"
    write_json = config.output.write_summary_json or config.output.profile == "standard"
    if write_txt:
        summary_path_plan["valley_summary_txt"] = output_dir / "valley_summary.txt"
    if write_json:
        summary_path_plan["valley_summary_json"] = output_dir / "valley_summary.json"
    output_paths = {
        key: value
        for key, value in {**outputs, **summary_path_plan}.items()
        if isinstance(value, Path)
    }
    summary_payload = build_summary_payload(
        config=config,
        qcut=qcut,
        subspace_payload=subspace_payload,
        symmetry_payload=symmetry_payload,
        symmetry_rows=symmetry_rows,
        output_paths=output_paths,
        symmetry_eigenvalue_summary=symmetry_eigenvalue_summary,
        projector_symmetry_report=projector_symmetry_report,
        symmetry_adapted_valley_report=symmetry_adapted_valley_report,
        target_subspace_closure_report=target_subspace_closure_report,
        hsp_star_conjugation_report=hsp_star_conjugation_report,
        hsp_star_derived_characters=hsp_star_derived_characters,
        irrep_workflow_decisions=irrep_workflow_decisions,
        valley_irrep_matching=valley_irrep_matching,
        ebr_input_candidates=ebr_input_candidates,
        ebr_problem_instances=ebr_problem_instances,
        ebr_export_bundle=ebr_export_bundle,
        reduced_ebr_mapping=reduced_ebr_mapping,
        valley_projected_representation=valley_projected_representation,
        folded_center_payload=folded_center_payload,
        sampled_k_coverage=sampled_k_coverage,
    )
    summary_text = render_summary_text(summary_payload)
    if "valley_summary_txt" in summary_path_plan:
        outputs["valley_summary_txt"] = write_summary_text(summary_path_plan["valley_summary_txt"], summary_text)
    if "valley_summary_json" in summary_path_plan:
        outputs["valley_summary_json"] = write_summary_json(summary_path_plan["valley_summary_json"], summary_payload)
    outputs["summary_text"] = summary_text
    outputs["summary_stdout"] = config.output.summary_stdout


def _extract_quality_report(
    symmetry_adapted_valley_report: dict[str, object],
) -> dict[str, object] | None:
    """Extract consolidated subspace_representation_quality from the report."""
    all_rows: list[dict[str, object]] = []
    by_kpoint = symmetry_adapted_valley_report.get("by_kpoint", {})
    if not isinstance(by_kpoint, dict):
        return None
    for kpoint_name, kp_data in by_kpoint.items():
        if not isinstance(kp_data, dict):
            continue
        for subspace in kp_data.get("valley_preserving_subspaces", []):
            if not isinstance(subspace, dict):
                continue
            quality = subspace.get("subspace_representation_quality")
            if not isinstance(quality, dict):
                continue
            for row in quality.get("rows", []):
                if not isinstance(row, dict):
                    continue
                row_copy = dict(row)
                row_copy["kpoint"] = kpoint_name
                all_rows.append(row_copy)
    if not all_rows:
        return None
    return {
        "status": "quality_issues_detected"
        if any(r.get("diagnosis") not in ("ok", "not_valley_preserving", "missing_inputs")
               for r in all_rows)
        else "ok",
        "interpretation": (
            "Per-(kpoint, valley, operation) subspace representation quality. "
            "Decomposes local representation unitarity error. "
            "Diagnostic-only; does not modify readiness."
        ),
        "rows": all_rows,
    }
