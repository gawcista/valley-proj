from __future__ import annotations

from pathlib import Path

import numpy as np

from valley_proj.geometry.lattice import (
    cart_rotation_from_fractional,
    cart_translation_from_fractional,
    read_poscar_cell,
)
from valley_proj.io.config import AppConfig, load_config
from valley_proj.io.h5_reader import read_wavefunction_h5
from valley_proj.projection.qcut_scan import (
    qcut_from_min_sector_distance,
    qcut_from_moire_shell,
    scan_qcut,
)
from valley_proj.projection.sector_projectors import SectorProjectors, build_sector_projectors
from valley_proj.projection.weights import classify_valley_weights, compute_valley_weights
from valley_proj.reports.csv_report import weight_row, write_valley_weights_csv
from valley_proj.reports.h5_report import write_basis_transform_h5, write_diagnostics_h5
from valley_proj.reports.json_report import write_json
from valley_proj.symmetry.little_group import is_little_group_operation
from valley_proj.symmetry.operation_classifier import classify_operation
from valley_proj.symmetry.plane_wave_action import build_plane_wave_representation
from valley_proj.symmetry.rotation_eigenvalues import extract_rotation_eigenvalues
from valley_proj.symmetry.spglib_finder import find_symmetry_operations
from valley_proj.symmetry.valley_preservation import map_valley_sectors


def _resolve_qcut(config: AppConfig, moire_reciprocal_cart: np.ndarray) -> float:
    projection = config.projection
    if projection.qcut_mode == "absolute":
        if projection.qcut_Ainv is None:
            raise ValueError("projection.qcut_Ainv is required when qcut_mode is absolute")
        return float(projection.qcut_Ainv)
    if projection.qcut_mode == "moire_shell":
        return qcut_from_moire_shell(moire_reciprocal_cart, projection.qcut_shell)
    if projection.qcut_mode == "relative_min_sector_distance":
        return qcut_from_min_sector_distance(config.valley_centers, config.valley_sectors, projection.qcut_fraction)
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
    qcut = _resolve_qcut(config, wavefunctions.metadata.lattice.reciprocal_cart)

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
        subspace_payload["kpoints"][kpoint_name] = {
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
                    "leakage": result.leakage,
                    "ambiguous_weight": result.ambiguous_weight,
                }
                for idx, result in enumerate(weights)
            ],
        }
        if config.projection.qcut_scan:
            scan_qcuts = config.projection.qcut_scan
            if config.projection.qcut_mode == "relative_min_sector_distance":
                min_qcut = qcut_from_min_sector_distance(config.valley_centers, config.valley_sectors, 1.0)
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
        outputs["valley_basis_transform_h5"] = write_basis_transform_h5(output_dir / "valley_basis_transform.h5", {})
    return outputs


def _prepare_symmetry_payload(config: AppConfig, monolayer_recip: np.ndarray) -> dict[str, object]:
    poscar = config.input.poscar
    if poscar is None or not poscar.exists():
        return {
            "source": config.symmetry.source,
            "status": "skipped",
            "reason": "POSCAR/CONTCAR path is missing or does not exist",
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
