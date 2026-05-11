from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from valleyscope.geometry.lattice import read_poscar_lattice
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector


@dataclass(frozen=True)
class InputConfig:
    wavefunction_h5: Path
    poscar: Path | None = None
    monolayer_poscars: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisConfig:
    kpoints: list[str]
    target_bands_vasp: list[int]
    degeneracy_tol_meV: float = 1.0


@dataclass(frozen=True)
class ProjectionConfig:
    use_2d_momentum_only: bool = True
    qcut_mode: str = "moire_shell"
    qcut_shell: float = 3.0
    qcut_Ainv: float | None = None
    qcut_fraction: float = 0.3
    qcut_scan: list[float] = field(default_factory=list)
    ambiguous_cross_sector: str = "warn_exclude"
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SymmetryConfig:
    source: str = "spglib"
    symprec: float = 1e-3
    symprec_scan: list[float] = field(default_factory=list)
    angle_tolerance: float = -1.0
    allowed_orders: list[int] = field(default_factory=lambda: [2, 3, 4, 6])
    proper_rotations_only: bool = True
    little_group_check: bool = True
    valley_preservation_check: bool = True


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    write_json: bool = True
    write_csv: bool = True
    write_hdf5_basis_transform: bool = True


@dataclass(frozen=True)
class AppConfig:
    input: InputConfig
    analysis: AnalysisConfig
    valley_centers: list[ValleyCenter]
    valley_sectors: list[ValleySector]
    projection: ProjectionConfig
    output: OutputConfig
    symmetry: SymmetryConfig = field(default_factory=SymmetryConfig)
    monolayer_lattices: dict[str, np.ndarray] = field(default_factory=dict)
    layer_transforms: dict[str, dict[str, Any]] = field(default_factory=dict)

    def default_monolayer_reciprocal(self) -> np.ndarray:
        if "default" in self.monolayer_lattices:
            return self.monolayer_lattices["default"]
        if self.monolayer_lattices:
            return next(iter(self.monolayer_lattices.values()))
        for layer, path in self.input.monolayer_poscars.items():
            del layer
            return read_poscar_lattice(str(path)).reciprocal_cart
        raise ValueError(
            "No monolayer reciprocal lattice configured; provide monolayer_lattices or monolayer_poscars"
        )


def _path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _parse_lattices(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in raw.items():
        if "reciprocal_cart" not in value:
            raise ValueError(f"monolayer_lattices.{name} must define reciprocal_cart")
        reciprocal = np.asarray(value["reciprocal_cart"], dtype=float)
        if reciprocal.shape != (3, 3):
            raise ValueError(f"monolayer_lattices.{name}.reciprocal_cart must have shape [3,3]")
        result[name] = reciprocal
    return result


def _rotation_z_row(degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(degrees))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _fallback_reciprocal(
    lattices: dict[str, np.ndarray],
    monolayer_poscars: dict[str, Path],
) -> np.ndarray:
    if "default" in lattices:
        return lattices["default"]
    if lattices:
        return next(iter(lattices.values()))
    if monolayer_poscars:
        return read_poscar_lattice(str(next(iter(monolayer_poscars.values())))).reciprocal_cart
    raise ValueError(
        "No monolayer reciprocal lattice configured; provide monolayer_lattices or monolayer_poscars"
    )


def _layer_reciprocal(
    layer: str | None,
    lattices: dict[str, np.ndarray],
    monolayer_poscars: dict[str, Path],
    transforms: dict[str, dict[str, Any]],
) -> np.ndarray:
    if layer is not None and layer in lattices:
        reciprocal = lattices[layer]
    elif "default" in lattices:
        reciprocal = lattices["default"]
    elif layer is not None and layer in monolayer_poscars:
        reciprocal = read_poscar_lattice(str(monolayer_poscars[layer])).reciprocal_cart
    else:
        reciprocal = _fallback_reciprocal(lattices, monolayer_poscars)
    transform = transforms.get(layer or "", {})
    rotation = _rotation_z_row(float(transform.get("rotation_deg", 0.0)))
    return np.asarray(reciprocal, dtype=float) @ rotation


def _parse_centers(
    raw: dict[str, Any],
    lattices: dict[str, np.ndarray],
    monolayer_poscars: dict[str, Path],
    transforms: dict[str, dict[str, Any]],
) -> list[ValleyCenter]:
    coordinate_mode = str(raw.get("coordinate_mode", "cart"))
    centers: list[ValleyCenter] = []
    for item in raw.get("centers", []):
        layer = item.get("layer")
        reciprocal = _layer_reciprocal(layer, lattices, monolayer_poscars, transforms)
        if coordinate_mode == "cart":
            if "cart" not in item:
                raise ValueError(f"valley center {item.get('name')} must define cart")
            cart = np.asarray(item["cart"], dtype=float)
        elif coordinate_mode in {"frac", "layer_frac"}:
            if "frac" not in item:
                raise ValueError(f"valley center {item.get('name')} must define frac")
            frac = np.asarray(item["frac"], dtype=float)
            if frac.shape != (3,):
                raise ValueError(f"valley center {item.get('name')}.frac must have shape [3]")
            cart = frac @ reciprocal
        else:
            raise ValueError(f"Unsupported valley_centers.coordinate_mode: {coordinate_mode}")
        centers.append(
            ValleyCenter(
                name=item["name"],
                cart=cart,
                layer=layer,
                reciprocal_cart=reciprocal,
            )
        )
    return centers


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    base = config_path.parent
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    input_raw = raw.get("input", {})
    monolayer_poscars = {
        str(name): _path(base, str(value))
        for name, value in (input_raw.get("monolayer_poscars") or {}).items()
    }
    monolayer_lattices = _parse_lattices(raw.get("monolayer_lattices", {}))
    layer_transforms = dict(raw.get("layer_transforms", {}))
    centers = _parse_centers(
        raw.get("valley_centers", {}),
        monolayer_lattices,
        monolayer_poscars,
        layer_transforms,
    )
    sectors = [ValleySector(item["name"], list(item["centers"])) for item in raw.get("valley_sectors", [])]
    analysis_raw = raw.get("analysis", {})
    projection_raw = raw.get("projection", {})
    symmetry_raw = raw.get("symmetry", {})
    output_raw = raw.get("output", {})
    if "wavefunction_h5" not in input_raw:
        raise ValueError("input.wavefunction_h5 is required")
    if not centers:
        raise ValueError("valley_centers.centers must not be empty")
    if not sectors:
        raise ValueError("valley_sectors must not be empty")
    return AppConfig(
        input=InputConfig(
            wavefunction_h5=_path(base, input_raw["wavefunction_h5"]),
            poscar=_path(base, input_raw.get("poscar")),
            monolayer_poscars=monolayer_poscars,
        ),
        analysis=AnalysisConfig(
            kpoints=list(analysis_raw.get("kpoints", [])),
            target_bands_vasp=[int(value) for value in analysis_raw.get("target_bands_vasp", [])],
            degeneracy_tol_meV=float(analysis_raw.get("degeneracy_tol_meV", 1.0)),
        ),
        valley_centers=centers,
        valley_sectors=sectors,
        projection=ProjectionConfig(
            use_2d_momentum_only=bool(projection_raw.get("use_2d_momentum_only", True)),
            qcut_mode=str(projection_raw.get("qcut_mode", "moire_shell")),
            qcut_shell=float(projection_raw.get("qcut_shell", 3.0)),
            qcut_Ainv=projection_raw.get("qcut_Ainv"),
            qcut_fraction=float(projection_raw.get("qcut_fraction", 0.3)),
            qcut_scan=[float(value) for value in projection_raw.get("qcut_scan", [])],
            ambiguous_cross_sector=str(projection_raw.get("ambiguous_cross_sector", "warn_exclude")),
            thresholds=dict(projection_raw.get("thresholds", {})),
        ),
        symmetry=SymmetryConfig(
            source=str(symmetry_raw.get("source", "spglib")),
            symprec=float(symmetry_raw.get("symprec", 1e-3)),
            symprec_scan=[float(value) for value in symmetry_raw.get("symprec_scan", [])],
            angle_tolerance=float(symmetry_raw.get("angle_tolerance", -1.0)),
            allowed_orders=[int(value) for value in symmetry_raw.get("allowed_orders", [2, 3, 4, 6])],
            proper_rotations_only=bool(symmetry_raw.get("proper_rotations_only", True)),
            little_group_check=bool(symmetry_raw.get("little_group_check", True)),
            valley_preservation_check=bool(symmetry_raw.get("valley_preservation_check", True)),
        ),
        output=OutputConfig(
            directory=_path(base, output_raw.get("directory", "valley_analysis")),
            write_json=bool(output_raw.get("write_json", True)),
            write_csv=bool(output_raw.get("write_csv", True)),
            write_hdf5_basis_transform=bool(output_raw.get("write_hdf5_basis_transform", True)),
        ),
        monolayer_lattices=monolayer_lattices,
        layer_transforms=layer_transforms,
    )
