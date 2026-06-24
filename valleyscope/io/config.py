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
    iband: list[int]
    degeneracy_tol_meV: float = 1.0


# Canonical projector_mode values and their deprecated aliases.
_PROJECTOR_MODE_ALIASES: dict[str, str] = {
    "fixed_center": "fixed_center",
    "k_resolved_parent_valley": "k_resolved_parent_valley",
    # Deprecated aliases — normalized internally.
    "fixed_point": "fixed_center",
    "folded_family": "k_resolved_parent_valley",
}


def _normalize_projector_mode(raw: str) -> str:
    """Normalize projector_mode, raising on invalid values."""
    canonical = _PROJECTOR_MODE_ALIASES.get(str(raw))
    if canonical is None:
        raise ValueError(
            f"projection.projector_mode must be 'fixed_center' or 'k_resolved_parent_valley'; "
            f"got {raw!r}"
        )
    return canonical


@dataclass(frozen=True)
class ProjectionConfig:
    use_2d_momentum_only: bool = True
    projector_mode: str = "fixed_center"
    qcut_mode: str = "moire_shell"
    qcut_shell: float = 3.0
    qcut_Ainv: float | None = None
    qcut_fraction: float = 0.3
    qcut_scan: list[float] = field(default_factory=list)
    overlap_policy: str = "warn_exclude"
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


_VALID_OUTPUT_PROFILES = frozenset({"standard", "debug"})


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    profile: str = "standard"
    write_json: bool = True
    write_csv: bool = True
    write_hdf5_basis_transform: bool = True
    summary_stdout: bool = True
    write_summary_txt: bool = True
    write_summary_json: bool = True
    write_detailed_files: bool = True  # deprecated — mapped to profile


@dataclass(frozen=True)
class SpinorConfig:
    convention: str = "vasp_up_down_saxis_z"
    convention_verified: bool = False
    benchmark: str | None = None


@dataclass(frozen=True)
class RotationConfig:
    readiness_preset: str = "strict"
    unitarity_tol: float = 1.0e-4
    root_deviation_tol: float = 1.0e-6
    D_valley_offdiag_tol: float = 1.0e-6
    irrep_weight_tol: float = 1.0e-5


@dataclass(frozen=True)
class ReducedEbrConfig:
    enabled: bool = False
    table_file: Path | None = None
    max_coefficient: int = 6


@dataclass(frozen=True)
class GenericIrrepSourceConfig:
    """Generic Bilbao/irreptables irrep source matching config.

    Enables the generic restricted-character irrep path.  The source table
    must be the valley-projected subspace space group, not necessarily the
    full parent moire space group.  For a one-valley projected subspace
    with P3 symmetry inside a P321 moire, use ``spacegroup_number: 143``
    (P3) rather than 150 (P321).
    """
    enabled: bool = False
    spacegroup_number: int | None = None
    """Subspace-space-group number for the generic irrep source table."""
    spinor: bool | None = None
    operation_match_tol: float = 5e-5
    source_hsp_labels: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class SymmetryAdaptedValleyConfig:
    enabled: bool = True
    seed_overlap_warn_tol: float = 0.8
    seed_overlap_fail_tol: float = 0.5
    projector_symmetry_warn_tol: float = 1e-2
    projector_symmetry_fail_tol: float = 1e-1
    representation_unitarity_warn_tol: float = 1e-3
    representation_unitarity_fail_tol: float = 1e-2
    ebr_seed_overlap_min: float = 0.8
    ebr_unitarity_max: float = 1e-3
    write_subspace_representation_quality: bool = False


@dataclass(frozen=True)
class AppConfig:
    input: InputConfig
    analysis: AnalysisConfig
    valley_centers: list[ValleyCenter]
    valley_subspaces: list[ValleySector]
    projection: ProjectionConfig
    output: OutputConfig
    symmetry: SymmetryConfig = field(default_factory=SymmetryConfig)
    spinor: SpinorConfig = field(default_factory=SpinorConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    symmetry_adapted_valley: SymmetryAdaptedValleyConfig = field(default_factory=SymmetryAdaptedValleyConfig)
    reduced_ebr: ReducedEbrConfig = field(default_factory=ReducedEbrConfig)
    generic_irrep_source: GenericIrrepSourceConfig = field(default_factory=GenericIrrepSourceConfig)
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
    if "target_bands_vasp" in raw:
        raise ValueError("analysis.target_bands_vasp has been removed; use analysis.iband")
    if "iband" not in raw:
        raise ValueError("analysis.iband is required")
    iband = raw["iband"]

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
        iband=[int(value) for value in iband],
        degeneracy_tol_meV=float(degeneracy_tol),
    )


def _parse_valley_subspaces(raw: dict[str, Any]) -> list[ValleySector]:
    if "valley_sectors" in raw:
        raise ValueError("valley_sectors has been removed; use valley_subspaces")
    if "valley_manifolds" in raw:
        raise ValueError("valley_manifolds has been removed; use valley_subspaces")
    sectors_raw = raw.get("valley_subspaces", [])
    return [ValleySector(item["name"], list(item["centers"])) for item in sectors_raw]


def _projection_qcut_mode(raw: dict[str, Any]) -> str:
    if "qcut_mode" in raw:
        return str(raw["qcut_mode"])
    if "qcut_Ainv" in raw:
        return "absolute"
    if "qcut_fraction" in raw:
        return "relative_min_valley_distance"
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


ROTATION_READINESS_PRESETS: dict[str, dict[str, float]] = {
    "strict": {
        "unitarity_tol": 1.0e-4,
        "root_deviation_tol": 1.0e-6,
        "D_valley_offdiag_tol": 1.0e-6,
        "irrep_weight_tol": 1.0e-5,
    },
    "normal": {
        "unitarity_tol": 1.0e-4,
        "root_deviation_tol": 1.0e-5,
        "D_valley_offdiag_tol": 1.0e-3,
        "irrep_weight_tol": 5.0e-5,
    },
    "loose": {
        "unitarity_tol": 1.0e-4,
        "root_deviation_tol": 1.0e-4,
        "D_valley_offdiag_tol": 1.0e-2,
        "irrep_weight_tol": 1.0e-4,
    },
}


def _parse_rotation_config(raw: dict[str, Any]) -> RotationConfig:
    preset = str(raw.get("readiness_preset", "strict")).lower()
    if preset not in ROTATION_READINESS_PRESETS:
        allowed = ", ".join(sorted(ROTATION_READINESS_PRESETS))
        raise ValueError(f"rotation.readiness_preset must be one of: {allowed}")
    values = dict(ROTATION_READINESS_PRESETS[preset])
    return RotationConfig(
        readiness_preset=preset,
        unitarity_tol=float(raw.get("unitarity_tol", values["unitarity_tol"])),
        root_deviation_tol=float(raw.get("root_deviation_tol", values["root_deviation_tol"])),
        D_valley_offdiag_tol=float(raw.get("D_valley_offdiag_tol", values["D_valley_offdiag_tol"])),
        irrep_weight_tol=float(raw.get("irrep_weight_tol", values["irrep_weight_tol"])),
    )


def _resolve_output_profile(raw: dict[str, Any]) -> str:
    """Resolve output.profile, mapping legacy write_detailed_files when needed.

    Priority: explicit profile > explicit write_detailed_files > default "standard".
    Emits a DeprecationWarning when write_detailed_files is used without profile.
    """
    has_profile = "profile" in raw
    has_wdf = "write_detailed_files" in raw
    if has_profile:
        profile = str(raw["profile"]).lower()
        if profile not in _VALID_OUTPUT_PROFILES:
            raise ValueError(
                f"output.profile must be one of {sorted(_VALID_OUTPUT_PROFILES)}; got {profile!r}"
            )
        if has_wdf:
            warnings.warn(
                "output.write_detailed_files is deprecated; output.profile takes precedence.",
                DeprecationWarning,
                stacklevel=3,
            )
        return profile
    if has_wdf:
        warnings.warn(
            "output.write_detailed_files is deprecated; use output.profile instead. "
            "Mapping write_detailed_files=false → profile='standard', "
            "write_detailed_files=true → profile='debug'.",
            DeprecationWarning,
            stacklevel=3,
        )
        return "debug" if bool(raw["write_detailed_files"]) else "standard"
    return "standard"


def _parse_reduced_ebr_config(base: Path, raw: dict[str, Any]) -> ReducedEbrConfig:
    if not isinstance(raw, dict):
        return ReducedEbrConfig()
    max_coefficient = int(raw.get("max_coefficient", 6))
    if max_coefficient < 0:
        raise ValueError("analysis.reduced_ebr.max_coefficient must be nonnegative")
    table_file_raw = raw.get("table_file")
    if raw.get("table_name") is not None:
        raise ValueError(
            "analysis.reduced_ebr.table_name was removed. "
            "Use analysis.reduced_ebr.table_file for external tables, "
            "or the irreptables runtime reducer for Bilbao-derived reduced tables."
        )
    return ReducedEbrConfig(
        enabled=bool(raw.get("enabled", False)),
        table_file=resolve_config_path(base, table_file_raw),
        max_coefficient=max_coefficient,
    )




def _parse_generic_irrep_source_config(raw: dict[str, Any]) -> GenericIrrepSourceConfig:
    if not isinstance(raw, dict) or not raw:
        return GenericIrrepSourceConfig()
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return GenericIrrepSourceConfig()
    sg = raw.get("spacegroup_number")
    if not (isinstance(sg, int) and not isinstance(sg, bool)):
        raise ValueError(
            "analysis.generic_irrep_source.spacegroup_number must be an integer "
            "when enabled"
        )
    spinor = raw.get("spinor")
    if not isinstance(spinor, bool):
        raise ValueError(
            "analysis.generic_irrep_source.spinor must be a boolean "
            "when enabled"
        )
    if not isinstance(raw.get("source_hsp_labels"), dict) or not raw.get("source_hsp_labels"):
        raise ValueError(
            "analysis.generic_irrep_source.source_hsp_labels must be a "
            "non-empty mapping when enabled"
        )
    op_tol = float(raw.get("operation_match_tol", 5e-5))
    if op_tol <= 0:
        raise ValueError(
            "analysis.generic_irrep_source.operation_match_tol must be positive"
        )
    source_hsp_labels: dict[str, dict[str, str]] = {}
    raw_hsp = raw.get("source_hsp_labels")
    if raw_hsp is not None:
        if not isinstance(raw_hsp, dict):
            raise ValueError(
                "analysis.generic_irrep_source.source_hsp_labels must be a mapping"
            )
        for kp_label, valley_map in raw_hsp.items():
            if not isinstance(kp_label, str) or not kp_label:
                raise ValueError(
                    "analysis.generic_irrep_source.source_hsp_labels keys "
                    "must be non-empty strings (kpoint labels)"
                )
            if not isinstance(valley_map, dict):
                raise ValueError(
                    f"analysis.generic_irrep_source.source_hsp_labels[{kp_label!r}] "
                    "must be a mapping (valley -> source HSP label)"
                )
            inner: dict[str, str] = {}
            for v_label, src_hsp in valley_map.items():
                if not isinstance(v_label, str) or not v_label:
                    raise ValueError(
                        f"analysis.generic_irrep_source.source_hsp_labels"
                        f"[{kp_label!r}] keys must be non-empty strings"
                    )
                if not isinstance(src_hsp, str) or not src_hsp:
                    raise ValueError(
                        f"analysis.generic_irrep_source.source_hsp_labels"
                        f"[{kp_label!r}][{v_label!r}] must be a non-empty string"
                    )
                inner[v_label] = src_hsp
            source_hsp_labels[kp_label] = inner
    return GenericIrrepSourceConfig(
        enabled=enabled,
        spacegroup_number=int(sg) if sg is not None else None,
        spinor=bool(spinor) if spinor is not None else None,
        operation_match_tol=op_tol,
        source_hsp_labels=source_hsp_labels,
    )

def _parse_symmetry_adapted_valley_config(raw: dict[str, Any]) -> SymmetryAdaptedValleyConfig:
    if not isinstance(raw, dict):
        return SymmetryAdaptedValleyConfig()
    return SymmetryAdaptedValleyConfig(
        enabled=bool(raw.get("enabled", True)),
        seed_overlap_warn_tol=float(raw.get("seed_overlap_warn_tol", 0.8)),
        seed_overlap_fail_tol=float(raw.get("seed_overlap_fail_tol", 0.5)),
        projector_symmetry_warn_tol=float(raw.get("projector_symmetry_warn_tol", 1e-2)),
        projector_symmetry_fail_tol=float(raw.get("projector_symmetry_fail_tol", 1e-1)),
        representation_unitarity_warn_tol=float(raw.get("representation_unitarity_warn_tol", 1e-3)),
        representation_unitarity_fail_tol=float(raw.get("representation_unitarity_fail_tol", 1e-2)),
        ebr_seed_overlap_min=float(raw.get("ebr_seed_overlap_min", 0.8)),
        ebr_unitarity_max=float(raw.get("ebr_unitarity_max", 1e-3)),
        write_subspace_representation_quality=bool(raw.get("write_subspace_representation_quality", False)),
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
    valley_subspaces = _parse_valley_subspaces(raw)
    analysis_raw = raw.get("analysis", {})
    projection_raw = raw.get("projection", {})
    allowed_projection_keys = {
        "use_2d_momentum_only",
        "projector_mode",
        "qcut_mode",
        "qcut_shell",
        "qcut_Ainv",
        "qcut_fraction",
        "qcut_scan",
        "overlap_policy",
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
    if not valley_subspaces:
        raise ValueError("valley_subspaces must not be empty")
    return AppConfig(
        input=InputConfig(
            wavefunction_h5=resolve_config_path(base, input_raw["wavefunction_h5"]),
            monolayer_poscars=monolayer_poscars,
        ),
        analysis=_parse_analysis_config(analysis_raw),
        valley_centers=centers,
        valley_subspaces=valley_subspaces,
        projection=ProjectionConfig(
            use_2d_momentum_only=bool(projection_raw.get("use_2d_momentum_only", True)),
            projector_mode=_normalize_projector_mode(projection_raw.get("projector_mode", "fixed_center")),
            qcut_mode=_projection_qcut_mode(projection_raw),
            qcut_shell=float(projection_raw.get("qcut_shell", 3.0)),
            qcut_Ainv=projection_raw.get("qcut_Ainv"),
            qcut_fraction=float(projection_raw.get("qcut_fraction", 0.3)),
            qcut_scan=[float(value) for value in projection_raw.get("qcut_scan", [])],
            overlap_policy=str(projection_raw.get("overlap_policy", "warn_exclude")),
            thresholds=dict(projection_raw.get("thresholds", {})),
        ),
        symmetry=_parse_symmetry_config(base, input_raw, symmetry_raw),
        spinor=_parse_spinor_config(raw.get("spinor", {})),
        rotation=_parse_rotation_config(raw.get("rotation", {})),
        symmetry_adapted_valley=_parse_symmetry_adapted_valley_config(
            analysis_raw.get("symmetry_adapted_valley", {})
        ),
        reduced_ebr=_parse_reduced_ebr_config(base, analysis_raw.get("reduced_ebr", {})),
        generic_irrep_source=_parse_generic_irrep_source_config(
            analysis_raw.get("generic_irrep_source", {}),
        ),
        output=OutputConfig(
            directory=resolve_config_path(base, output_raw.get("directory", "valley_analysis")),
            profile=_resolve_output_profile(output_raw),
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
