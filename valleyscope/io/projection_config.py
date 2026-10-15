"""Projection-only configuration: geometry and path, without symmetry producers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector, centers_by_name
from valleyscope.io import resolve_config_path
from valleyscope.io.config import (
    AnalysisConfig, InputConfig, OutputConfig, ProjectionConfig,
    _moire_direct_cart, _normalize_projector_mode, _parse_analysis_config,
    _parse_centers, _parse_lattices, _parse_valley_subspaces,
    _projection_qcut_mode, _resolve_fallback_reciprocal,
)
from valleyscope.projection.qcut_scan import qcut_from_min_sector_distance, qcut_from_moire_shell


@dataclass(frozen=True)
class ProjectionTaskConfig:
    input: InputConfig
    analysis: AnalysisConfig
    valley_centers: list[ValleyCenter]
    valley_subspaces: list[ValleySector]
    projection: ProjectionConfig
    output: OutputConfig
    monolayer_lattices: dict[str, np.ndarray]
    layer_transforms: dict
    break_before: list[int] = field(default_factory=list)
    ticks: list[dict] = field(default_factory=list)
    energy_reference_eV: float = 0.

    def default_monolayer_reciprocal(self):
        return _resolve_fallback_reciprocal(self.monolayer_lattices, self.input.monolayer_poscars)


def _mapping(raw, key):
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _finite_positive(value, label):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and strictly positive")
    return value


def _path_index(value, count, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value < count:
        raise ValueError(f"{label} must be a zero-based path-occurrence index in [{minimum}, {count})")
    return value


def load_projection_config(path: str | Path) -> ProjectionTaskConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Projection configuration must be a mapping")
    base = path.parent
    input_raw = _mapping(raw, "input")
    if "wavefunction_h5" not in input_raw:
        raise ValueError("input.wavefunction_h5 is required")
    monolayer_poscars = {str(key): resolve_config_path(base, str(value))
                         for key, value in _mapping(input_raw, "monolayer_poscars").items()}
    lattices = _parse_lattices(_mapping(raw, "monolayer_lattices"))
    transforms = _mapping(raw, "layer_transforms")
    centers = _parse_centers(_mapping(raw, "valley_centers"), lattices, monolayer_poscars,
                             transforms, _moire_direct_cart(base, input_raw))
    valleys = _parse_valley_subspaces(raw)
    if not centers or not valleys:
        raise ValueError("valley_centers.centers and valley_subspaces must not be empty")
    center_map = centers_by_name(centers)
    for center in centers:
        if not isinstance(center.name, str) or not center.name:
            raise ValueError("Valley center names must be nonempty strings")
        if not np.all(np.isfinite(center.cart)):
            raise ValueError(f"Valley center {center.name} must be finite")
        basis = center.reciprocal_cart
        if not np.all(np.isfinite(basis)) or np.linalg.det(basis) == 0:
            raise ValueError(f"Parent reciprocal basis for {center.name} must be finite and nonsingular")
    names = [valley.name for valley in valleys]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Valley subspace names must be unique nonempty strings")
    for valley in valleys:
        if any(name not in center_map for name in valley.centers):
            raise ValueError(f"Valley {valley.name} references unknown centers")
    analysis = _parse_analysis_config(_mapping(raw, "analysis"))
    if not analysis.kpoints or any(not isinstance(name, str) or not name for name in analysis.kpoints):
        raise ValueError("analysis.kpoints must be a nonempty ordered list of names")
    if not analysis.iband or any(band <= 0 for band in analysis.iband):
        raise ValueError("analysis.iband must contain positive original VASP band IDs")
    projection_raw = _mapping(raw, "projection")
    allowed = set(ProjectionConfig.__dataclass_fields__)
    unknown = set(projection_raw) - allowed
    if unknown:
        raise ValueError(f"Unsupported projection keys: {sorted(unknown)}")
    if projection_raw.get("qcut_scan"):
        raise ValueError("project-bands accepts one cutoff per result; use separate projection configurations for qcut_scan comparisons")
    overlap = projection_raw.get("overlap_policy", "warn_exclude")
    if overlap not in {"warn_exclude", "error"}:
        raise ValueError("project-bands overlap_policy must be warn_exclude or error; include double-counts distinct valley windows and prevents disjoint full-state accounting")
    use_2d = projection_raw.get("use_2d_momentum_only", True)
    if not isinstance(use_2d, bool):
        raise ValueError("projection.use_2d_momentum_only must be a Boolean")
    mode = _projection_qcut_mode(projection_raw)
    if mode not in {"absolute", "moire_shell", "relative_min_valley_distance"}:
        raise ValueError(f"Unsupported qcut_mode: {mode}")
    projection = ProjectionConfig(
        use_2d_momentum_only=use_2d,
        projector_mode=_normalize_projector_mode(projection_raw.get("projector_mode", "fixed_center")),
        qcut_mode=mode,
        qcut_Ainv=_finite_positive(projection_raw.get("qcut_Ainv"), "projection.qcut_Ainv") if mode == "absolute" else None,
        qcut_shell=_finite_positive(projection_raw.get("qcut_shell", 3.), "projection.qcut_shell"),
        qcut_fraction=_finite_positive(projection_raw.get("qcut_fraction", .3), "projection.qcut_fraction"),
        overlap_policy=overlap,
    )
    path_raw = _mapping(raw, "path")
    breaks = path_raw.get("break_before", [])
    ticks = path_raw.get("ticks", [])
    if not isinstance(breaks, list) or not isinstance(ticks, list):
        raise ValueError("path.break_before and path.ticks must be lists")
    breaks = [_path_index(index, len(analysis.kpoints), "path.break_before", minimum=1) for index in breaks]
    if len(set(breaks)) != len(breaks):
        raise ValueError("path.break_before contains duplicate occurrence indices")
    parsed_ticks = []
    for tick in ticks:
        if not isinstance(tick, dict) or not isinstance(tick.get("label"), str) or not tick["label"]:
            raise ValueError("path.ticks entries require an index and a nonempty string label")
        parsed_ticks.append({"index": _path_index(tick.get("index"), len(analysis.kpoints), "path.ticks.index"), "label": tick["label"]})
    energy = float(_mapping(raw, "plot").get("energy_reference_eV", 0.))
    if not np.isfinite(energy):
        raise ValueError("plot.energy_reference_eV must be finite")
    output_raw = _mapping(raw, "output")
    return ProjectionTaskConfig(
        input=InputConfig(resolve_config_path(base, input_raw["wavefunction_h5"]), monolayer_poscars),
        analysis=analysis, valley_centers=centers, valley_subspaces=valleys,
        projection=projection,
        output=OutputConfig(resolve_config_path(base, output_raw.get("directory", "valley_projection")),
                            summary_stdout=bool(output_raw.get("summary_stdout", True))),
        monolayer_lattices=lattices, layer_transforms=transforms, break_before=breaks,
        ticks=parsed_ticks, energy_reference_eV=energy,
    )


def resolve_projection_qcut(config: ProjectionTaskConfig, moire_reciprocal: np.ndarray) -> float:
    projection = config.projection
    if projection.qcut_mode == "absolute":
        value = projection.qcut_Ainv
    elif projection.qcut_mode == "moire_shell":
        value = qcut_from_moire_shell(moire_reciprocal, projection.qcut_shell)
    else:
        value = qcut_from_min_sector_distance(config.valley_centers, config.valley_subspaces,
            projection.qcut_fraction, config.default_monolayer_reciprocal(),
            use_2d=projection.use_2d_momentum_only)
    return _finite_positive(value, "resolved projection qcut_Ainv")
