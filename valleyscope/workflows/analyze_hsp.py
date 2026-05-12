from __future__ import annotations

from pathlib import Path

import numpy as np

from valleyscope.geometry.lattice import (
    cart_rotation_from_fractional,
    cart_translation_from_fractional,
    read_poscar_cell,
)
from valleyscope.io.config import AppConfig, load_config
from valleyscope.io.h5_reader import read_wavefunction_h5
from valleyscope.projection.qcut_scan import (
    qcut_from_min_sector_distance,
    qcut_from_moire_shell,
    scan_qcut,
)
from valleyscope.projection.sector_projectors import SectorProjectors, build_sector_projectors
from valleyscope.projection.weights import classify_valley_weights, compute_valley_weights
from valleyscope.reports.csv_report import weight_row, write_valley_weights_csv
from valleyscope.reports.h5_report import write_basis_transform_h5, write_diagnostics_h5
from valleyscope.reports.json_report import write_json
from valleyscope.subspace.valley_basis import build_two_valley_adapted_basis
from valleyscope.symmetry.little_group import is_little_group_operation
from valleyscope.symmetry.operation_classifier import classify_operation
from valleyscope.symmetry.plane_wave_action import build_plane_wave_representation
from valleyscope.symmetry.rotation_eigenvalues import extract_rotation_eigenvalues
from valleyscope.symmetry.spglib_finder import find_symmetry_operations
from valleyscope.symmetry.valley_preservation import map_valley_sectors


def _resolve_qcut(
    config: AppConfig,
    moire_reciprocal_cart: np.ndarray,
    monolayer_reciprocal_cart: np.ndarray,
) -> float:
    projection = config.projection
    if projection.qcut_mode == "absolute":
        if projection.qcut_Ainv is None:
            raise ValueError("projection.qcut_Ainv is required when qcut_mode is absolute")
        return float(projection.qcut_Ainv)
    if projection.qcut_mode == "moire_shell":
        return qcut_from_moire_shell(moire_reciprocal_cart, projection.qcut_shell)
    if projection.qcut_mode == "relative_min_sector_distance":
        return qcut_from_min_sector_distance(
            config.valley_centers,
            config.valley_sectors,
            projection.qcut_fraction,
            monolayer_reciprocal_cart,
            use_2d=projection.use_2d_momentum_only,
        )
    raise ValueError(f"Unsupported qcut_mode: {projection.qcut_mode}")


def _target_band_positions(available_bands: np.ndarray, target_bands: list[int]) -> list[int]:
    positions: list[int] = []
    for band in target_bands:
        matches = np.where(available_bands == band)[0]
        if len(matches) == 0:
            raise ValueError(f"HDF5 is missing target VASP band index: {band}")
        positions.append(int(matches[0]))
    return positions


def analyze_hsp(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    wavefunctions = read_wavefunction_h5(config.input.wavefunction_h5)
    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    monolayer_recip = config.default_monolayer_reciprocal()
    qcut = _resolve_qcut(config, wavefunctions.metadata.lattice.reciprocal_cart, monolayer_recip)

    rows: list[dict[str, object]] = []
    rotation_rows: list[str] = [
        "kpoint,operation_id,order,eigenvalue_real,eigenvalue_imag,phase_2pi,modulus_deviation,unitarity_deviation,spinor_convention_verified\n"
    ]
    subspace_payload: dict[str, object] = {
        "degeneracy_tol_meV": config.analysis.degeneracy_tol_meV,
        "kpoints": {},
    }
    projectors_by_kpoint: dict[str, SectorProjectors] = {}
    qcut_scan_payload: dict[str, object] = {}
    basis_transforms: dict[str, dict[str, np.ndarray]] = {}
    symmetry_payload: dict[str, object] = _prepare_symmetry_payload(config, monolayer_recip)

    for kpoint_name in config.analysis.kpoints:
        kpoint = wavefunctions.find_kpoint(kpoint_name)
        positions = _target_band_positions(kpoint.band_indices_vasp, config.analysis.target_bands_vasp)
        coefficients = kpoint.coefficients[positions]
        q_cart = kpoint.cart.reshape(1, 3) + kpoint.g_vectors_cart
        projectors = build_sector_projectors(
            q_cart,
            config.valley_centers,
            config.valley_sectors,
            monolayer_recip,
            qcut,
            use_2d=config.projection.use_2d_momentum_only,
            ambiguous_policy=config.projection.ambiguous_cross_sector,
        )
        projectors_by_kpoint[kpoint_name] = projectors
        weights = compute_valley_weights(coefficients, projectors)
        sector_names = projectors.sector_names
        for local_pos, result in enumerate(weights):
            source_pos = positions[local_pos]
            rows.append(
                weight_row(
                    kpoint=kpoint_name,
                    band_vasp=int(kpoint.band_indices_vasp[source_pos]),
                    energy_eV=float(kpoint.energies_eV[source_pos]),
                    result=result,
                    sector_names=sector_names,
                )
            )
        kpoint_subspace = {
            "qcut": qcut,
            "warnings": projectors.warnings,
            "weights": [
                {
                    "band_vasp": int(kpoint.band_indices_vasp[positions[idx]]),
                    "classification": classify_valley_weights(
                        w_val=result.w_val,
                        purity=result.purity,
                        thresholds=config.projection.thresholds,
                    ),
                    "sector_weights": result.sector_weights,
                    "W_val": result.w_val,
                    "P_v": result.purity,
                    "eta": result.eta,
                    "W_res": result.residual_weight,
                    "ambiguous_weight": result.ambiguous_weight,
                }
                for idx, result in enumerate(weights)
            ],
        }
        _add_valley_subspace_diagnostic(
            kpoint_subspace,
            basis_transforms,
            kpoint_name,
            kpoint.band_indices_vasp[positions],
            kpoint.energies_eV[positions],
            coefficients,
            projectors,
            config.analysis.degeneracy_tol_meV,
        )
        subspace_payload["kpoints"][kpoint_name] = kpoint_subspace
        if config.projection.qcut_scan:
            scan_qcuts = config.projection.qcut_scan
            if config.projection.qcut_mode == "relative_min_sector_distance":
                min_qcut = qcut_from_min_sector_distance(
                    config.valley_centers,
                    config.valley_sectors,
                    1.0,
                    monolayer_recip,
                    use_2d=config.projection.use_2d_momentum_only,
                )
                scan_qcuts = [fraction * min_qcut for fraction in config.projection.qcut_scan]
            scan = scan_qcut(
                q_cart,
                coefficients,
                config.valley_centers,
                config.valley_sectors,
                monolayer_recip,
                scan_qcuts,
                use_2d=config.projection.use_2d_momentum_only,
                ambiguous_policy=config.projection.ambiguous_cross_sector,
            )
            qcut_scan_payload[kpoint_name] = {
                "has_plateau": scan.has_plateau,
                "qcuts": [entry.qcut for entry in scan.entries],
                "ambiguous_count": [entry.ambiguous_count for entry in scan.entries],
                "band_indices_vasp": kpoint.band_indices_vasp[positions],
                "sector_names": np.asarray(projectors.sector_names, dtype="S"),
                "w_val": [
                    [result.w_val for result in entry.weights]
                    for entry in scan.entries
                ],
                "purity": [
                    [result.purity for result in entry.weights]
                    for entry in scan.entries
                ],
                "eta": [
                    [np.nan if result.eta is None else result.eta for result in entry.weights]
                    for entry in scan.entries
                ],
                "ambiguous_weight": [
                    [result.ambiguous_weight for result in entry.weights]
                    for entry in scan.entries
                ],
            }
        if symmetry_payload["status"] == "ok":
            rotation_rows.extend(
                _rotation_rows_for_kpoint(
                    config,
                    kpoint_name,
                    kpoint.frac,
                    q_cart,
                    coefficients,
                    symmetry_payload,
                )
            )

    sector_names = list(projectors_by_kpoint[next(iter(projectors_by_kpoint))].sector_masks)
    outputs = {
        "valley_weights_csv": write_valley_weights_csv(output_dir / "valley_weights.csv", rows, sector_names),
        "valley_subspace_json": write_json(output_dir / "valley_subspace.json", subspace_payload),
        "symmetry_report_json": write_json(
            output_dir / "symmetry_report.json",
            symmetry_payload,
        ),
        "rotation_eigenvalues_csv": output_dir / "rotation_eigenvalues.csv",
        "diagnostics_h5": write_diagnostics_h5(output_dir / "diagnostics.h5", projectors_by_kpoint, qcut_scan_payload),
    }
    outputs["rotation_eigenvalues_csv"].write_text("".join(rotation_rows), encoding="utf-8")
    if config.output.write_hdf5_basis_transform:
        outputs["valley_basis_transform_h5"] = write_basis_transform_h5(
            output_dir / "valley_basis_transform.h5",
            basis_transforms,
        )
    return outputs


def _add_valley_subspace_diagnostic(
    payload: dict[str, object],
    basis_transforms: dict[str, dict[str, np.ndarray]],
    kpoint_name: str,
    band_indices_vasp: np.ndarray,
    energies_eV: np.ndarray,
    coefficients: np.ndarray,
    projectors: SectorProjectors,
    degeneracy_tol_meV: float,
) -> None:
    sector_names = projectors.sector_names
    energy_span_meV = float((np.max(energies_eV) - np.min(energies_eV)) * 1000.0)
    diagnostic: dict[str, object] = {
        "band_indices_vasp": np.asarray(band_indices_vasp, dtype=int),
        "energy_span_meV": energy_span_meV,
        "status": "not_evaluated",
    }
    if len(sector_names) != 2:
        diagnostic["status"] = "requires_two_valley_sectors"
    elif coefficients.shape[0] < 2:
        diagnostic["status"] = "single_band"
    elif energy_span_meV > degeneracy_tol_meV:
        diagnostic["status"] = "not_degenerate"
    else:
        result = build_two_valley_adapted_basis(
            coefficients,
            projectors.sector_masks,
            sector_names[0],
            sector_names[1],
        )
        diagnostic.update(
            {
                "status": "two_valley_adapted",
                "sectors": sector_names,
                "eta": result.eta,
                "s_eigenvalues": np.linalg.eigvalsh(result.s_matrix),
                "v_eigenvalues": np.linalg.eigvalsh(result.v_matrix),
                "transform_h5_group": kpoint_name,
            }
        )
        basis_transforms[kpoint_name] = {
            "transform": result.transform,
            "eta": result.eta,
            "s_matrix": result.s_matrix,
            "v_matrix": result.v_matrix,
            "band_indices_vasp": np.asarray(band_indices_vasp, dtype=int),
            "sectors": np.asarray(sector_names, dtype="S"),
        }
    payload["valley_adapted_subspace"] = diagnostic


def _prepare_symmetry_payload(config: AppConfig, monolayer_recip: np.ndarray) -> dict[str, object]:
    poscar = config.input.poscar
    if poscar is None or not poscar.exists():
        return {
            "source": config.symmetry.source,
            "status": "skipped",
            "reason": (
                "input.poscar is missing or does not exist. "
                "Symmetry analysis requires the moire/bilayer POSCAR or CONTCAR; "
                "input.monolayer_poscars are used for valley-center geometry, not for spglib symmetry."
            ),
        }
    cell = read_poscar_cell(str(poscar))
    dataset = find_symmetry_operations(cell, config.symmetry.symprec, config.symmetry.angle_tolerance)
    lattice = np.asarray(cell[0], dtype=float)
    operations = []
    for op_id, (rotation, translation) in enumerate(zip(dataset.rotations, dataset.translations)):
        info = classify_operation(rotation, translation, allowed_orders=config.symmetry.allowed_orders)
        rotation_cart = cart_rotation_from_fractional(rotation, lattice)
        translation_cart = cart_translation_from_fractional(translation, lattice)
        valley_mapping = map_valley_sectors(
            rotation,
            rotation_cart,
            config.valley_centers,
            config.valley_sectors,
            monolayer_recip,
            tolerance=1e-6,
        )
        operations.append(
            {
                "operation_id": op_id,
                "rotation_frac": rotation,
                "translation_frac": translation,
                "rotation_cart": rotation_cart,
                "translation_cart": translation_cart,
                "det": info.det,
                "order": info.order,
                "kind": info.kind,
                "candidate_rotation": info.allowed_for_rotation_workflow,
                "sector_mapping": valley_mapping.sector_mapping,
                "preserved": valley_mapping.preserved,
                "center_mapping": valley_mapping.center_mapping,
            }
        )
    return {
        "source": config.symmetry.source,
        "status": "ok",
        "spacegroup_number": dataset.spacegroup_number,
        "international": dataset.international,
        "symprec": config.symmetry.symprec,
        "operations": operations,
    }


def _rotation_rows_for_kpoint(
    config: AppConfig,
    kpoint_name: str,
    k_frac: np.ndarray,
    q_cart: np.ndarray,
    coefficients: np.ndarray,
    symmetry_payload: dict[str, object],
) -> list[str]:
    rows: list[str] = []
    for operation in symmetry_payload["operations"]:
        if not operation["candidate_rotation"]:
            continue
        little = is_little_group_operation(np.asarray(operation["rotation_frac"]), k_frac)
        preserves_all = all(bool(value) for value in operation["preserved"].values())
        operation["little_group_by_kpoint"] = {
            **operation.get("little_group_by_kpoint", {}),
            kpoint_name: little,
        }
        operation["allowed_for_single_valley_rotation"] = bool(little and preserves_all)
        if not little or not preserves_all:
            continue
        representation = build_plane_wave_representation(
            coefficients,
            q_cart,
            np.asarray(operation["rotation_cart"]),
            np.asarray(operation["translation_cart"]),
        )
        eigen = extract_rotation_eigenvalues(representation.matrix, spinor_convention_verified=False)
        operation.setdefault("representation_quality", {})[kpoint_name] = {
            "mapping_miss_count": representation.mapping_miss_count,
            "unitarity_deviation": eigen.unitarity_deviation,
            "max_modulus_deviation": float(np.max(eigen.modulus_deviation)) if len(eigen.modulus_deviation) else 0.0,
            "spinor_convention_verified": False,
        }
        for value, phase, modulus_deviation in zip(eigen.eigenvalues, eigen.phases_2pi, eigen.modulus_deviation):
            rows.append(
                f"{kpoint_name},{operation['operation_id']},{operation['order']},"
                f"{value.real},{value.imag},{phase},{modulus_deviation},{eigen.unitarity_deviation},False\n"
            )
    return rows
