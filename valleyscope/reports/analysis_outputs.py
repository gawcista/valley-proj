from __future__ import annotations

from pathlib import Path
from typing import Any

from valleyscope.io.config import AppConfig
from valleyscope.projection.sector_projectors import SectorProjectors
from valleyscope.reports.csv_report import (
    write_little_group_eigenvalues_csv,
    write_rotation_eigenvalues_csv,
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
    rotation_rows: list[dict[str, object]],
    projectors_by_kpoint: dict[str, SectorProjectors],
    qcut_scan_payload: dict[str, object],
    rotation_payload: dict[str, object],
    basis_transforms: dict[str, dict[str, object]],
    little_group_rows: list[dict[str, object]] | None = None,
    little_group_diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    output_dir = config.output.directory
    outputs: dict[str, object] = {}
    if config.output.write_detailed_files:
        _write_detailed_outputs(
            config=config,
            output_dir=output_dir,
            outputs=outputs,
            weight_rows=weight_rows,
            sector_names=sector_names,
            subspace_payload=subspace_payload,
            symmetry_payload=symmetry_payload,
            rotation_rows=rotation_rows,
            projectors_by_kpoint=projectors_by_kpoint,
            qcut_scan_payload=qcut_scan_payload,
            rotation_payload=rotation_payload,
            basis_transforms=basis_transforms,
            little_group_rows=[] if little_group_rows is None else little_group_rows,
        )
    _write_summary_outputs(
        config=config,
        qcut=qcut,
        output_dir=output_dir,
        outputs=outputs,
        subspace_payload=subspace_payload,
        symmetry_payload=symmetry_payload,
        rotation_rows=rotation_rows,
        little_group_diagnostics=little_group_diagnostics,
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
    rotation_rows: list[dict[str, object]],
    projectors_by_kpoint: dict[str, SectorProjectors],
    qcut_scan_payload: dict[str, object],
    rotation_payload: dict[str, object],
    basis_transforms: dict[str, dict[str, object]],
    little_group_rows: list[dict[str, object]],
) -> None:
    if config.output.write_csv:
        outputs["valley_weights_csv"] = write_valley_weights_csv(
            output_dir / "valley_weights.csv",
            weight_rows,
            sector_names,
        )
        outputs["rotation_eigenvalues_csv"] = write_rotation_eigenvalues_csv(
            output_dir / "rotation_eigenvalues.csv",
            rotation_rows,
        )
        if little_group_rows:
            outputs["little_group_eigenvalues_csv"] = write_little_group_eigenvalues_csv(
                output_dir / "little_group_eigenvalues.csv",
                little_group_rows,
            )
    if config.output.write_json:
        outputs["valley_subspace_json"] = write_json(output_dir / "valley_subspace.json", subspace_payload)
        outputs["symmetry_report_json"] = write_json(output_dir / "symmetry_report.json", symmetry_payload)
        if little_group_rows:
            outputs["little_group_representations_json"] = write_json(
                output_dir / "little_group_representations.json",
                {
                    "little_group_representations": little_group_rows,
                    "irrep_label_matching": "deferred",
                },
            )
    outputs["diagnostics_h5"] = write_diagnostics_h5(
        output_dir / "diagnostics.h5",
        projectors_by_kpoint,
        qcut_scan_payload,
        rotation_payload,
        symmetry_payload,
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
    rotation_rows: list[dict[str, object]],
    little_group_diagnostics: dict[str, object] | None = None,
) -> None:
    summary_path_plan: dict[str, Path] = {}
    if config.output.write_summary_txt or not config.output.write_detailed_files:
        summary_path_plan["valley_summary_txt"] = output_dir / "valley_summary.txt"
    if config.output.write_summary_json or not config.output.write_detailed_files:
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
        rotation_rows=rotation_rows,
        output_paths=output_paths,
        little_group_diagnostics=little_group_diagnostics,
    )
    summary_text = render_summary_text(summary_payload)
    if "valley_summary_txt" in summary_path_plan:
        outputs["valley_summary_txt"] = write_summary_text(summary_path_plan["valley_summary_txt"], summary_text)
    if "valley_summary_json" in summary_path_plan:
        outputs["valley_summary_json"] = write_summary_json(summary_path_plan["valley_summary_json"], summary_payload)
    outputs["summary_text"] = summary_text
    outputs["summary_stdout"] = config.output.summary_stdout
