from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import h5py

from valleyscope.geometry.lattice import read_poscar_lattice, reciprocal_from_direct
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.symmetry.rotation_selection import parse_rotation_order


@dataclass(frozen=True)
class InputConfig:
    wavefunction_h5: Path
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
    overlap_cross_sector: str = "warn_exclude"
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SymmetryOperationsConfig:
    mode: str = "auto"
    structure_file: Path | None = None
    backend: str = "spglib"


@dataclass(frozen=True)
class SymmetryToleranceConfig:
    symprec: float = 1e-3
    angle_tolerance: float = -1.0
    symprec_scan: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class SymmetryFilterConfig:
    proper_rotations_only: bool = True
    allowed_orders: list[int] = field(default_factory=lambda: [2, 3, 4, 6])
    rotation_order: int | str | None = "auto"


@dataclass(frozen=True)
class SymmetryConfig:
    operations: SymmetryOperationsConfig = field(default_factory=SymmetryOperationsConfig)
    tolerance: SymmetryToleranceConfig = field(default_factory=SymmetryToleranceConfig)
    filters: SymmetryFilterConfig = field(default_factory=SymmetryFilterConfig)
    little_group_check: bool = True
    valley_preservation_check: bool = True


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    write_json: bool = True
    write_csv: bool = True
    write_hdf5_basis_transform: bool = True
    summary_stdout: bool = True
    write_summary_txt: bool = True
    write_summary_json: bool = True
    write_detailed_files: bool = True


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


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    section = raw.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"symmetry.{key} must be a mapping")
    return section


def _parse_symmetry_config(
    base: Path,
    input_raw: dict[str, Any],
    symmetry_raw: dict[str, Any],
) -> SymmetryConfig:
    if not isinstance(symmetry_raw, dict):
        raise ValueError("symmetry must be a mapping")

    for hard_check in ("little_group_check", "valley_preservation_check"):
        if symmetry_raw.get(hard_check) is False:
            raise ValueError(f"{hard_check} is a V1 hard check and cannot be disabled")

    new_sections = {name for name in ("operations", "tolerance", "filters") if name in symmetry_raw}
    legacy_keys = {
        key
        for key in (
            "source",
            "symprec",
            "symprec_scan",
            "angle_tolerance",
            "allowed_orders",
            "proper_rotations_only",
            "rotation_order",
            "little_group_check",
            "valley_preservation_check",
        )
        if key in symmetry_raw
    }
    has_legacy_poscar = input_raw.get("poscar") is not None
    if has_legacy_poscar:
        legacy_keys.add("input.poscar")

    if legacy_keys:
        message = (
            "input.poscar and symmetry.source are deprecated for symmetry-operation detection; "
            "use symmetry.operations.structure_file and symmetry.operations.backend."
        )
        if new_sections:
            message += " Legacy symmetry-operation fields are ignored because the new schema is present."
        warnings.warn(message, DeprecationWarning, stacklevel=2)

    if new_sections:
        operations_raw = _section(symmetry_raw, "operations")
        tolerance_raw = _section(symmetry_raw, "tolerance")
        filters_raw = _section(symmetry_raw, "filters")
        structure_file = _path(base, operations_raw.get("structure_file"))
        backend = str(operations_raw.get("backend", "spglib"))
        mode = str(operations_raw.get("mode", "auto"))
        symprec = float(tolerance_raw.get("symprec", 1e-3))
        angle_tolerance = float(tolerance_raw.get("angle_tolerance", -1.0))
        symprec_scan = [float(value) for value in tolerance_raw.get("symprec_scan", [])]
        proper_rotations_only = bool(filters_raw.get("proper_rotations_only", True))
        allowed_orders = [int(value) for value in filters_raw.get("allowed_orders", [2, 3, 4, 6])]
        rotation_order = parse_rotation_order(
            filters_raw.get("rotation_order", symmetry_raw.get("rotation_order", "auto"))
        )
    else:
        structure_file = _path(base, input_raw.get("poscar"))
        backend = str(symmetry_raw.get("source", "spglib"))
        mode = "auto"
        symprec = float(symmetry_raw.get("symprec", 1e-3))
        angle_tolerance = float(symmetry_raw.get("angle_tolerance", -1.0))
        symprec_scan = [float(value) for value in symmetry_raw.get("symprec_scan", [])]
        proper_rotations_only = bool(symmetry_raw.get("proper_rotations_only", True))
        allowed_orders = [int(value) for value in symmetry_raw.get("allowed_orders", [2, 3, 4, 6])]
        rotation_order = parse_rotation_order(symmetry_raw.get("rotation_order", "auto"))

    if mode != "auto":
        raise ValueError("symmetry.operations.mode currently supports only 'auto'")
    if backend != "spglib":
        raise ValueError("symmetry.operations.backend currently supports only 'spglib'")

    return SymmetryConfig(
        operations=SymmetryOperationsConfig(
            mode=mode,
            structure_file=structure_file,
            backend=backend,
        ),
        tolerance=SymmetryToleranceConfig(
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            symprec_scan=symprec_scan,
        ),
        filters=SymmetryFilterConfig(
            proper_rotations_only=proper_rotations_only,
            allowed_orders=allowed_orders,
            rotation_order=rotation_order,
        ),
        little_group_check=True,
        valley_preservation_check=True,
    )


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


def _read_h5_direct_cart(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        if "metadata/lattice/direct_cart" not in h5:
            return None
        direct = np.asarray(h5["metadata/lattice/direct_cart"][()], dtype=float)
    if direct.shape != (3, 3):
        raise ValueError("metadata/lattice/direct_cart must have shape [3,3]")
    return direct


def _moire_direct_cart(base: Path, input_raw: dict[str, Any]) -> np.ndarray | None:
    if "wavefunction_h5" in input_raw:
        direct = _read_h5_direct_cart(_path(base, input_raw["wavefunction_h5"]))
        if direct is not None:
            return direct
    if input_raw.get("poscar") is not None:
        poscar = _path(base, input_raw.get("poscar"))
        if poscar is not None and poscar.exists():
            return read_poscar_lattice(str(poscar)).direct_cart
    return None


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
    moire_direct_cart: np.ndarray | None,
) -> np.ndarray:
    transform = transforms.get(layer or "", {})
    if "supercell_matrix" in transform:
        if "rotation_deg" in transform:
            raise ValueError(
                f"layer_transforms.{layer}.supercell_matrix and rotation_deg are mutually exclusive"
            )
        if moire_direct_cart is None:
            raise ValueError(
                f"layer_transforms.{layer}.supercell_matrix requires input.wavefunction_h5 "
                "with a moire direct lattice, or legacy input.poscar"
            )
        supercell = _supercell_matrix(transform["supercell_matrix"])
        layer_direct = np.linalg.inv(supercell).T @ np.asarray(moire_direct_cart, dtype=float)
        return reciprocal_from_direct(layer_direct)

    if layer is not None and layer in lattices:
        reciprocal = lattices[layer]
    elif "default" in lattices:
        reciprocal = lattices["default"]
    elif layer is not None and layer in monolayer_poscars:
        reciprocal = read_poscar_lattice(str(monolayer_poscars[layer])).reciprocal_cart
    else:
        reciprocal = _fallback_reciprocal(lattices, monolayer_poscars)
    rotation = _rotation_z_row(float(transform.get("rotation_deg", 0.0)))
    return np.asarray(reciprocal, dtype=float) @ rotation


def _supercell_matrix(raw: Any) -> np.ndarray:
    matrix = np.asarray(raw, dtype=float)
    if matrix.shape == (2, 2):
        full = np.eye(3)
        full[:2, :2] = matrix
        matrix = full
    if matrix.shape != (3, 3):
        raise ValueError("supercell_matrix must have shape [2,2] or [3,3]")
    if abs(float(np.linalg.det(matrix))) < 1e-14:
        raise ValueError("supercell_matrix must be nonsingular")
    return matrix


def _parse_centers(
    raw: dict[str, Any],
    lattices: dict[str, np.ndarray],
    monolayer_poscars: dict[str, Path],
    transforms: dict[str, dict[str, Any]],
    moire_direct_cart: np.ndarray | None,
) -> list[ValleyCenter]:
    coordinate_mode = str(raw.get("coordinate_mode", "cart"))
    centers: list[ValleyCenter] = []
    for item in raw.get("centers", []):
        layer = item.get("layer")
        reciprocal = _layer_reciprocal(layer, lattices, monolayer_poscars, transforms, moire_direct_cart)
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
    moire_direct = _moire_direct_cart(base, input_raw)
    centers = _parse_centers(
        raw.get("valley_centers", {}),
        monolayer_lattices,
        monolayer_poscars,
        layer_transforms,
        moire_direct,
    )
    sectors = [ValleySector(item["name"], list(item["centers"])) for item in raw.get("valley_sectors", [])]
    analysis_raw = raw.get("analysis", {})
    projection_raw = raw.get("projection", {})
    allowed_projection_keys = {
        "use_2d_momentum_only",
        "qcut_mode",
        "qcut_shell",
        "qcut_Ainv",
        "qcut_fraction",
        "qcut_scan",
        "overlap_cross_sector",
        "thresholds",
    }
    unknown_projection_keys = sorted(set(projection_raw) - allowed_projection_keys)
    if unknown_projection_keys:
        raise ValueError(f"Unsupported projection keys: {unknown_projection_keys}")
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
            overlap_cross_sector=str(projection_raw.get("overlap_cross_sector", "warn_exclude")),
            thresholds=dict(projection_raw.get("thresholds", {})),
        ),
        symmetry=_parse_symmetry_config(base, input_raw, symmetry_raw),
        output=OutputConfig(
            directory=_path(base, output_raw.get("directory", "valley_analysis")),
            write_json=bool(output_raw.get("write_json", True)),
            write_csv=bool(output_raw.get("write_csv", True)),
            write_hdf5_basis_transform=bool(output_raw.get("write_hdf5_basis_transform", True)),
            summary_stdout=bool(output_raw.get("summary_stdout", True)),
            write_summary_txt=bool(output_raw.get("write_summary_txt", True)),
            write_summary_json=bool(output_raw.get("write_summary_json", True)),
            write_detailed_files=bool(output_raw.get("write_detailed_files", True)),
        ),
        monolayer_lattices=monolayer_lattices,
        layer_transforms=layer_transforms,
    )
