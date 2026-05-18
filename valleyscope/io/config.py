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
from valleyscope.io import resolve_config_path
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
class SpinorConfig:
    convention: str = "vasp_up_down_saxis_z"
    convention_verified: bool = False
    benchmark: str | None = None


@dataclass(frozen=True)
class RotationConfig:
    unitarity_tol: float = 1.0e-4
    root_deviation_tol: float = 1.0e-6
    D_valley_offdiag_tol: float = 1.0e-6


@dataclass(frozen=True)
class AppConfig:
    input: InputConfig
    analysis: AnalysisConfig
    valley_centers: list[ValleyCenter]
    valley_sectors: list[ValleySector]
    projection: ProjectionConfig
    output: OutputConfig
    symmetry: SymmetryConfig = field(default_factory=SymmetryConfig)
    spinor: SpinorConfig = field(default_factory=SpinorConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    monolayer_lattices: dict[str, np.ndarray] = field(default_factory=dict)
    layer_transforms: dict[str, dict[str, Any]] = field(default_factory=dict)

    def default_monolayer_reciprocal(self) -> np.ndarray:
        return _resolve_fallback_reciprocal(self.monolayer_lattices, self.input.monolayer_poscars)



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
        fields = dict(
            structure_file=resolve_config_path(base, operations_raw.get("structure_file")),
            backend=str(operations_raw.get("backend", "spglib")),
            mode=str(operations_raw.get("mode", "auto")),
            symprec=float(tolerance_raw.get("symprec", 1e-3)),
            angle_tolerance=float(tolerance_raw.get("angle_tolerance", -1.0)),
            symprec_scan=[float(v) for v in tolerance_raw.get("symprec_scan", [])],
            proper_rotations_only=bool(filters_raw.get("proper_rotations_only", True)),
            allowed_orders=[int(v) for v in filters_raw.get("allowed_orders", [2, 3, 4, 6])],
            rotation_order=parse_rotation_order(
                filters_raw.get("rotation_order", symmetry_raw.get("rotation_order", "auto"))
            ),
        )
    else:
        fields = dict(
            structure_file=resolve_config_path(base, input_raw.get("poscar")),
            backend=str(symmetry_raw.get("source", "spglib")),
            mode="auto",
            symprec=float(symmetry_raw.get("symprec", 1e-3)),
            angle_tolerance=float(symmetry_raw.get("angle_tolerance", -1.0)),
            symprec_scan=[float(v) for v in symmetry_raw.get("symprec_scan", [])],
            proper_rotations_only=bool(symmetry_raw.get("proper_rotations_only", True)),
            allowed_orders=[int(v) for v in symmetry_raw.get("allowed_orders", [2, 3, 4, 6])],
            rotation_order=parse_rotation_order(symmetry_raw.get("rotation_order", "auto")),
        )

    if fields["mode"] != "auto":
        raise ValueError("symmetry.operations.mode currently supports only 'auto'")
    if fields["backend"] != "spglib":
        raise ValueError("symmetry.operations.backend currently supports only 'spglib'")

    return SymmetryConfig(
        operations=SymmetryOperationsConfig(
            mode=fields["mode"],
            structure_file=fields["structure_file"],
            backend=fields["backend"],
        ),
        tolerance=SymmetryToleranceConfig(
            symprec=fields["symprec"],
            angle_tolerance=fields["angle_tolerance"],
            symprec_scan=fields["symprec_scan"],
        ),
        filters=SymmetryFilterConfig(
            proper_rotations_only=fields["proper_rotations_only"],
            allowed_orders=fields["allowed_orders"],
            rotation_order=fields["rotation_order"],
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
    try:
        h5 = h5py.File(path, "r")
    except (FileNotFoundError, OSError):
        return None
    with h5:
        if "metadata/lattice/direct_cart" not in h5:
            return None
        direct = np.asarray(h5["metadata/lattice/direct_cart"][()], dtype=float)
    if direct.shape != (3, 3):
        raise ValueError("metadata/lattice/direct_cart must have shape [3,3]")
    return direct


def _moire_direct_cart(base: Path, input_raw: dict[str, Any]) -> np.ndarray | None:
    if "wavefunction_h5" in input_raw:
        direct = _read_h5_direct_cart(resolve_config_path(base, input_raw["wavefunction_h5"]))
        if direct is not None:
            return direct
    if input_raw.get("poscar") is not None:
        poscar = resolve_config_path(base, input_raw.get("poscar"))
        if poscar is not None:
            try:
                return read_poscar_lattice(str(poscar)).direct_cart
            except (FileNotFoundError, OSError, ValueError):
                pass
    return None


def _rotation_z_row(degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(degrees))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _resolve_fallback_reciprocal(
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
        reciprocal = _resolve_fallback_reciprocal(lattices, monolayer_poscars)
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


def _parse_analysis_config(raw: dict[str, Any]) -> AnalysisConfig:
    if "iband" in raw:
        if "target_bands_vasp" in raw and list(raw["target_bands_vasp"]) != list(raw["iband"]):
            warnings.warn(
                "analysis.target_bands_vasp is ignored because analysis.iband is present.",
                DeprecationWarning,
                stacklevel=3,
            )
        target_bands = raw["iband"]
    else:
        target_bands = raw.get("target_bands_vasp", [])

    if "subspace_energy_tol_meV" in raw:
        if "degeneracy_tol_meV" in raw and float(raw["degeneracy_tol_meV"]) != float(raw["subspace_energy_tol_meV"]):
            warnings.warn(
                "analysis.degeneracy_tol_meV is ignored because analysis.subspace_energy_tol_meV is present.",
                DeprecationWarning,
                stacklevel=3,
            )
        degeneracy_tol = raw["subspace_energy_tol_meV"]
    else:
        degeneracy_tol = raw.get("degeneracy_tol_meV", 1.0)

    return AnalysisConfig(
        kpoints=list(raw.get("kpoints", [])),
        target_bands_vasp=[int(value) for value in target_bands],
        degeneracy_tol_meV=float(degeneracy_tol),
    )


def _parse_valley_sectors(raw: dict[str, Any]) -> list[ValleySector]:
    if "valley_manifolds" in raw:
        if "valley_sectors" in raw and raw["valley_sectors"] != raw["valley_manifolds"]:
            warnings.warn(
                "valley_sectors is ignored because valley_manifolds is present.",
                DeprecationWarning,
                stacklevel=3,
            )
        sectors_raw = raw.get("valley_manifolds", [])
    else:
        sectors_raw = raw.get("valley_sectors", [])
    return [ValleySector(item["name"], list(item["centers"])) for item in sectors_raw]


def _projection_qcut_mode(raw: dict[str, Any]) -> str:
    if "qcut_mode" in raw:
        return str(raw["qcut_mode"])
    if "qcut_Ainv" in raw:
        return "absolute"
    if "qcut_fraction" in raw:
        return "relative_min_sector_distance"
    return "moire_shell"


def _parse_spinor_config(raw: dict[str, Any]) -> SpinorConfig:
    convention = str(raw.get("convention", "vasp_up_down_saxis_z"))
    if convention != "vasp_up_down_saxis_z":
        raise ValueError("spinor.convention currently supports only 'vasp_up_down_saxis_z'")
    convention_verified = bool(raw.get("convention_verified", raw.get("verified", False)))
    benchmark = raw.get("benchmark")
    benchmark = None if benchmark is None else str(benchmark)
    if convention_verified and not benchmark:
        raise ValueError("spinor.convention_verified=true requires spinor.benchmark")
    return SpinorConfig(
        convention=convention,
        convention_verified=convention_verified,
        benchmark=benchmark,
    )


def _parse_rotation_config(raw: dict[str, Any]) -> RotationConfig:
    return RotationConfig(
        unitarity_tol=float(raw.get("unitarity_tol", 1.0e-4)),
        root_deviation_tol=float(raw.get("root_deviation_tol", 1.0e-6)),
        D_valley_offdiag_tol=float(raw.get("D_valley_offdiag_tol", 1.0e-6)),
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    base = config_path.parent
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    input_raw = raw.get("input", {})
    monolayer_poscars = {
        str(name): resolve_config_path(base, str(value))
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
    sectors = _parse_valley_sectors(raw)
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
            wavefunction_h5=resolve_config_path(base, input_raw["wavefunction_h5"]),
            monolayer_poscars=monolayer_poscars,
        ),
        analysis=_parse_analysis_config(analysis_raw),
        valley_centers=centers,
        valley_sectors=sectors,
        projection=ProjectionConfig(
            use_2d_momentum_only=bool(projection_raw.get("use_2d_momentum_only", True)),
            qcut_mode=_projection_qcut_mode(projection_raw),
            qcut_shell=float(projection_raw.get("qcut_shell", 3.0)),
            qcut_Ainv=projection_raw.get("qcut_Ainv"),
            qcut_fraction=float(projection_raw.get("qcut_fraction", 0.3)),
            qcut_scan=[float(value) for value in projection_raw.get("qcut_scan", [])],
            overlap_cross_sector=str(projection_raw.get("overlap_cross_sector", "warn_exclude")),
            thresholds=dict(projection_raw.get("thresholds", {})),
        ),
        symmetry=_parse_symmetry_config(base, input_raw, symmetry_raw),
        spinor=_parse_spinor_config(raw.get("spinor", {})),
        rotation=_parse_rotation_config(raw.get("rotation", {})),
        output=OutputConfig(
            directory=resolve_config_path(base, output_raw.get("directory", "valley_analysis")),
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
