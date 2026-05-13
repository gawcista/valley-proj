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
from valleyscope.reports.csv_report import weight_row, write_rotation_eigenvalues_csv, write_valley_weights_csv
from valleyscope.reports.h5_report import write_basis_transform_h5, write_diagnostics_h5
from valleyscope.reports.json_report import write_json
from valleyscope.reports.summary_report import (
    build_summary_payload,
    render_summary_text,
    write_summary_json,
    write_summary_text,
)
from valleyscope.subspace.valley_basis import build_two_valley_adapted_basis
from valleyscope.symmetry.little_group import is_little_group_operation
from valleyscope.symmetry.operation_classifier import classify_operation, rotation_axis_angle
from valleyscope.symmetry.plane_wave_action import build_plane_wave_representation, spin_rotation_matrix
from valleyscope.symmetry.rotation_eigenvalues import extract_rotation_eigenvalues, nearest_root_of_unity
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


def analyze_hsp(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    wavefunctions = read_wavefunction_h5(config.input.wavefunction_h5)
    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    monolayer_recip = config.default_monolayer_reciprocal()
    qcut = _resolve_qcut(config, wavefunctions.metadata.lattice.reciprocal_cart, monolayer_recip)

    rows: list[dict[str, object]] = []
    rotation_rows: list[dict[str, object]] = []
    subspace_payload: dict[str, object] = {
        "degeneracy_tol_meV": config.analysis.degeneracy_tol_meV,
        "kpoints": {},
    }
    projectors_by_kpoint: dict[str, SectorProjectors] = {}
    qcut_scan_payload: dict[str, object] = {}
    basis_transforms: dict[str, dict[str, np.ndarray]] = {}
    rotation_payload: dict[str, object] = {}
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
            overlap_policy=config.projection.overlap_cross_sector,
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
                    "W_overlap": result.overlap_weight,
                    "W_res": result.residual_weight,
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
            config.projection.thresholds.get("W_val_min", 0.8),
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
                overlap_policy=config.projection.overlap_cross_sector,
            )
            qcut_scan_payload[kpoint_name] = {
                "has_plateau": scan.has_plateau,
                "qcuts": [entry.qcut for entry in scan.entries],
                "overlap_count": [entry.overlap_count for entry in scan.entries],
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
                "W_overlap": [
                    [result.overlap_weight for result in entry.weights]
                    for entry in scan.entries
                ],
                "W_res": [
                    [result.residual_weight for result in entry.weights]
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
                    basis_transforms.get(kpoint_name),
                    rotation_payload,
                )
            )

    sector_names = list(projectors_by_kpoint[next(iter(projectors_by_kpoint))].sector_masks)
    outputs: dict[str, object] = {}
    if config.output.write_detailed_files:
        if config.output.write_csv:
            outputs["valley_weights_csv"] = write_valley_weights_csv(
                output_dir / "valley_weights.csv",
                rows,
                sector_names,
            )
            outputs["rotation_eigenvalues_csv"] = write_rotation_eigenvalues_csv(
                output_dir / "rotation_eigenvalues.csv",
                rotation_rows,
            )
        if config.output.write_json:
            outputs["valley_subspace_json"] = write_json(output_dir / "valley_subspace.json", subspace_payload)
            outputs["symmetry_report_json"] = write_json(
                output_dir / "symmetry_report.json",
                symmetry_payload,
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
    )
    summary_text = render_summary_text(summary_payload)
    if "valley_summary_txt" in summary_path_plan:
        outputs["valley_summary_txt"] = write_summary_text(summary_path_plan["valley_summary_txt"], summary_text)
    if "valley_summary_json" in summary_path_plan:
        outputs["valley_summary_json"] = write_summary_json(summary_path_plan["valley_summary_json"], summary_payload)
    outputs["summary_text"] = summary_text
    outputs["summary_stdout"] = config.output.summary_stdout
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
    w_val_min: float,
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
        s_eigenvalues = np.linalg.eigvalsh(result.s_matrix)
        s_min = float(np.min(s_eigenvalues)) if len(s_eigenvalues) else 0.0
        s_max = float(np.max(s_eigenvalues)) if len(s_eigenvalues) else 0.0
        valid_valley_subspace = bool(s_min >= w_val_min)
        status = "two_valley_adapted" if valid_valley_subspace else "poor_valley_manifold"
        diagnostic.update(
            {
                "status": status,
                "sectors": sector_names,
                "eta": result.eta,
                "s_eigenvalues": s_eigenvalues,
                "s_min": s_min,
                "s_max": s_max,
                "valid_valley_subspace": valid_valley_subspace,
                "v_eigenvalues": np.linalg.eigvalsh(result.v_matrix),
                "transform_h5_group": kpoint_name,
            }
        )
        if valid_valley_subspace:
            basis_transforms[kpoint_name] = {
                "transform": result.transform,
                "eta": result.eta,
                "s_matrix": result.s_matrix,
                "v_matrix": result.v_matrix,
                "band_indices_vasp": np.asarray(band_indices_vasp, dtype=int),
                "sectors": np.asarray(sector_names, dtype="S"),
                "valid_valley_subspace": np.asarray(valid_valley_subspace),
                "s_eigenvalues": s_eigenvalues,
            }
    payload["valley_adapted_subspace"] = diagnostic


def _prepare_symmetry_payload(config: AppConfig, monolayer_recip: np.ndarray) -> dict[str, object]:
    symmetry = config.symmetry
    structure_file = symmetry.operations.structure_file
    base_payload = {
        "operation_detection_backend": symmetry.operations.backend,
        "structure_file": None if structure_file is None else str(structure_file),
        "symprec": symmetry.tolerance.symprec,
        "angle_tolerance": symmetry.tolerance.angle_tolerance,
        "symprec_scan_summary": [],
        "detected_operations": [],
        "candidate_rotations": [],
        "filters": {
            "proper_rotations_only": symmetry.filters.proper_rotations_only,
            "allowed_orders": symmetry.filters.allowed_orders,
        },
        "little_group_check": {"required": True, "status": "not_run"},
        "valley_preservation_check": {"required": True, "status": "not_run"},
    }
    if structure_file is None or not structure_file.exists():
        return {
            **base_payload,
            "status": "skipped",
            "reason": (
                "symmetry.operations.structure_file is missing or does not exist. "
                "Symmetry-operation detection requires the moire/bilayer POSCAR or CONTCAR; "
                "input.monolayer_poscars are used for monolayer reciprocal geometry and valley centers."
            ),
        }
    cell = read_poscar_cell(str(structure_file))
    dataset = find_symmetry_operations(cell, symmetry.tolerance.symprec, symmetry.tolerance.angle_tolerance)
    lattice = np.asarray(cell[0], dtype=float)
    operations = []
    for op_id, (rotation, translation) in enumerate(zip(dataset.rotations, dataset.translations)):
        info = classify_operation(rotation, translation, allowed_orders=symmetry.filters.allowed_orders)
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
    symprec_scan_summary = _symprec_scan_summary(config, cell)
    candidate_rotations = [
        operation["operation_id"]
        for operation in operations
        if operation["candidate_rotation"]
    ]
    return {
        **base_payload,
        "status": "ok",
        "structure_file": str(structure_file),
        "spacegroup_number": dataset.spacegroup_number,
        "international": dataset.international,
        "symprec_scan_summary": symprec_scan_summary,
        "detected_operation_count": len(operations),
        "candidate_rotations": candidate_rotations,
        "detected_operations": operations,
        "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
        "valley_preservation_check": {"required": True, "status": "completed"},
    }


def _symprec_scan_summary(config: AppConfig, cell: tuple) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for symprec in config.symmetry.tolerance.symprec_scan:
        try:
            dataset = find_symmetry_operations(
                cell,
                symprec,
                config.symmetry.tolerance.angle_tolerance,
            )
            candidate_count = 0
            order_counts: dict[str, int] = {}
            for rotation, translation in zip(dataset.rotations, dataset.translations):
                info = classify_operation(
                    rotation,
                    translation,
                    allowed_orders=config.symmetry.filters.allowed_orders,
                )
                order_key = "none" if info.order is None else str(info.order)
                order_counts[order_key] = order_counts.get(order_key, 0) + 1
                if info.allowed_for_rotation_workflow:
                    candidate_count += 1
            summary.append(
                {
                    "symprec": float(symprec),
                    "status": "ok",
                    "spacegroup_number": dataset.spacegroup_number,
                    "international": dataset.international,
                    "n_operations": len(dataset.rotations),
                    "n_candidate_rotations": candidate_count,
                    "order_counts": order_counts,
                    "detected_operation_count": len(dataset.rotations),
                    "candidate_rotation_count": candidate_count,
                }
            )
        except Exception as exc:
            summary.append(
                {
                    "symprec": float(symprec),
                    "status": "error",
                    "reason": str(exc),
                }
            )
    return summary


def _rotation_rows_for_kpoint(
    config: AppConfig,
    kpoint_name: str,
    k_frac: np.ndarray,
    q_cart: np.ndarray,
    coefficients: np.ndarray,
    symmetry_payload: dict[str, object],
    basis_payload: dict[str, np.ndarray] | None,
    rotation_payload: dict[str, object],
) -> list[dict[str, object]]:
    del config
    rows: list[dict[str, object]] = []
    for operation in symmetry_payload["detected_operations"]:
        if not operation["candidate_rotation"]:
            continue
        little = is_little_group_operation(np.asarray(operation["rotation_frac"]), k_frac)
        preserves_all = all(bool(value) for value in operation["preserved"].values())
        operation["little_group_by_kpoint"] = {
            **operation.get("little_group_by_kpoint", {}),
            kpoint_name: little,
        }
        operation["allowed_for_single_valley_rotation"] = bool(little and preserves_all)
        operation["allowed_for_single_valley_rotation_by_kpoint"] = {
            **operation.get("allowed_for_single_valley_rotation_by_kpoint", {}),
            kpoint_name: bool(little and preserves_all),
        }
        rejection_reason = ""
        if not little:
            rejection_reason = "not in little group"
        elif not preserves_all:
            rejection_reason = "not valley preserving"
        operation["rejection_reason_by_kpoint"] = {
            **operation.get("rejection_reason_by_kpoint", {}),
            kpoint_name: rejection_reason,
        }
        if rejection_reason:
            continue
        spin_rotation = None
        spinor_convention_verified = coefficients.shape[1] == 1
        if coefficients.shape[1] == 2:
            try:
                axis, angle = rotation_axis_angle(np.asarray(operation["rotation_cart"]))
                spin_rotation = spin_rotation_matrix(axis, angle)
            except ValueError as exc:
                operation.setdefault("representation_quality", {})[kpoint_name] = {
                    "skipped_reason": f"spinor rotation skipped: {exc}",
                    "spinor_convention_verified": False,
                }
                continue
        elif coefficients.shape[1] != 1:
            operation.setdefault("representation_quality", {})[kpoint_name] = {
                "skipped_reason": f"unsupported nspinor={coefficients.shape[1]}",
                "spinor_convention_verified": False,
            }
            continue
        representation = build_plane_wave_representation(
            coefficients,
            q_cart,
            np.asarray(operation["rotation_cart"]),
            np.asarray(operation["translation_cart"]),
            spin_rotation=spin_rotation,
        )
        basis = "raw_diagnostic"
        matrix_for_eigen = representation.matrix
        d_valley = None
        valley_eta = None
        reason = "not valley-adapted"
        topology_ready = False
        if basis_payload is not None and bool(np.asarray(basis_payload.get("valid_valley_subspace", False))):
            transform = np.asarray(basis_payload["transform"], dtype=np.complex128)
            d_valley = transform.conj().T @ representation.matrix @ transform
            matrix_for_eigen = d_valley
            basis = "valley_adapted"
            valley_eta = np.asarray(basis_payload.get("eta", []), dtype=float)
            reason = ""
            topology_ready = bool(coefficients.shape[1] == 1 and representation.mapping_miss_count == 0)
        eigen = extract_rotation_eigenvalues(matrix_for_eigen, spinor_convention_verified=spinor_convention_verified)
        operation.setdefault("representation_quality", {})[kpoint_name] = {
            "mapping_miss_count": representation.mapping_miss_count,
            "unitarity_deviation": eigen.unitarity_deviation,
            "max_modulus_deviation": float(np.max(eigen.modulus_deviation)) if len(eigen.modulus_deviation) else 0.0,
            "spinor_convention_verified": spinor_convention_verified,
            "basis": basis,
        }
        op_key = f"operation_{operation['operation_id']}"
        op_payload: dict[str, object] = {
            "D_raw": representation.matrix,
            "eigenvalues": eigen.eigenvalues,
            "mapping_miss_count": representation.mapping_miss_count,
            "unitarity_deviation": eigen.unitarity_deviation,
            "operation_order": int(operation["order"]),
        }
        if d_valley is not None:
            op_payload["D_valley"] = d_valley
        rotation_payload.setdefault(kpoint_name, {})[op_key] = op_payload
        order = int(operation["order"])
        for state_index, (value, phase, modulus_deviation) in enumerate(
            zip(eigen.eigenvalues, eigen.phases_2pi, eigen.modulus_deviation)
        ):
            root_index, _root, root_deviation = nearest_root_of_unity(value, order=order)
            rows.append(
                {
                    "kpoint": kpoint_name,
                    "operation_id": operation["operation_id"],
                    "order": order,
                    "basis": basis,
                    "state_index": state_index,
                    "eigenvalue_real": float(value.real),
                    "eigenvalue_imag": float(value.imag),
                    "phase_2pi": float(phase),
                    "modulus_deviation": float(modulus_deviation),
                    "unitarity_deviation": float(eigen.unitarity_deviation),
                    "spinor_convention_verified": spinor_convention_verified,
                    "nearest_root_of_unity": f"exp(2pii*{root_index}/{order})",
                    "root_deviation": root_deviation,
                    "topology_ready": topology_ready,
                    "reason": reason,
                    "valley_eta": "" if valley_eta is None or state_index >= len(valley_eta) else float(valley_eta[state_index]),
                }
            )
    return rows
