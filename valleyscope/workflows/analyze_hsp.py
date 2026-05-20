from __future__ import annotations

from pathlib import Path

import numpy as np

from valleyscope.analysis.decision_tree import (
    derive_derived_score,
    derive_polarization_score,
    derive_symmetry_status,
    derive_valley_status,
)
from valleyscope.analysis.symmetry_eigenvalue_diagnostic import symmetry_eigenvalue_diagnostics_for_kpoint
from valleyscope.analysis.valley_little_group import (
    add_valley_irrep_results,
    build_valley_preserving_subgroup_report,
    update_valley_little_group_inventory,
)
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
from valleyscope.projection.weights import compute_valley_weights
from valleyscope.reports.analysis_outputs import write_analysis_outputs
from valleyscope.reports.csv_report import weight_row
from valleyscope.subspace.valley_basis import (
    build_two_valley_adapted_basis,
    build_valley_adapted_basis,
    diagnose_valley_separability,
)
from valleyscope.symmetry.operation_classifier import classify_operation
from valleyscope.symmetry.rotation_selection import mark_rotation_generators, resolve_rotation_order
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
    if projection.qcut_mode == "relative_min_valley_distance":
        return qcut_from_min_sector_distance(
            config.valley_centers,
            config.valley_subspaces,
            projection.qcut_fraction,
            monolayer_reciprocal_cart,
            use_2d=projection.use_2d_momentum_only,
        )
    raise ValueError(f"Unsupported qcut_mode: {projection.qcut_mode}")


def _target_band_positions(available_bands: np.ndarray, target_bands: list[int]) -> list[int]:
    band_to_pos = {int(b): i for i, b in enumerate(available_bands)}
    positions: list[int] = []
    for band in target_bands:
        pos = band_to_pos.get(band)
        if pos is None:
            raise ValueError(f"HDF5 is missing target VASP band index: {band}")
        positions.append(pos)
    return positions


def analyze_hsp(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    wavefunctions = read_wavefunction_h5(config.input.wavefunction_h5)
    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    monolayer_recip = config.default_monolayer_reciprocal()
    qcut = _resolve_qcut(config, wavefunctions.metadata.lattice.reciprocal_cart, monolayer_recip)

    rows: list[dict[str, object]] = []
    symmetry_rows: list[dict[str, object]] = []
    subspace_payload: dict[str, object] = {
        "degeneracy_tol_meV": config.analysis.degeneracy_tol_meV,
        "kpoints": {},
    }
    projectors_by_kpoint: dict[str, SectorProjectors] = {}
    qcut_scan_payload: dict[str, object] = {}
    basis_transforms: dict[str, dict[str, np.ndarray]] = {}
    symmetry_representation_payload: dict[str, object] = {}
    symmetry_payload: dict[str, object] = _prepare_symmetry_payload(config, monolayer_recip)
    symmetry_payload["spinor_wavefunction"] = bool(wavefunctions.metadata.spinor)

    for kpoint_name in config.analysis.kpoints:
        kpoint = wavefunctions.find_kpoint(kpoint_name)
        positions = _target_band_positions(kpoint.band_indices_vasp, config.analysis.iband)
        coefficients = kpoint.coefficients[positions]
        q_cart = kpoint.cart.reshape(1, 3) + kpoint.g_vectors_cart
        projectors = build_sector_projectors(
            q_cart,
            config.valley_centers,
            config.valley_subspaces,
            monolayer_recip,
            qcut,
            use_2d=config.projection.use_2d_momentum_only,
            overlap_policy=config.projection.overlap_policy,
            emit_warnings=False,
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
            "symmetry_status": "not_requested",
            "weights": [
                _build_weight_entry(
                    band_vasp=int(kpoint.band_indices_vasp[positions[idx]]),
                    result=result,
                    thresholds=config.projection.thresholds,
                )
                for idx, result in enumerate(weights)
            ],
        }
        max_w_overlap = max((w.overlap_weight for w in weights), default=0.0)
        max_w_res = max((w.residual_weight for w in weights), default=0.0)
        _add_valley_subspace_diagnostic(
            kpoint_subspace,
            basis_transforms,
            kpoint_name,
            kpoint.band_indices_vasp[positions],
            kpoint.energies_eV[positions],
            coefficients,
            projectors,
            config.analysis.degeneracy_tol_meV,
            thresholds=config.projection.thresholds if config.projection.thresholds else None,
            max_w_overlap=max_w_overlap,
            max_w_res=max_w_res,
        )
        subspace_payload["kpoints"][kpoint_name] = kpoint_subspace
        if config.projection.qcut_scan:
            scan_qcuts = config.projection.qcut_scan
            if config.projection.qcut_mode == "relative_min_valley_distance":
                min_qcut = qcut_from_min_sector_distance(
                    config.valley_centers,
                    config.valley_subspaces,
                    1.0,
                    monolayer_recip,
                    use_2d=config.projection.use_2d_momentum_only,
                )
                scan_qcuts = [fraction * min_qcut for fraction in config.projection.qcut_scan]
            scan = scan_qcut(
                q_cart,
                coefficients,
                config.valley_centers,
                config.valley_subspaces,
                monolayer_recip,
                scan_qcuts,
                use_2d=config.projection.use_2d_momentum_only,
                overlap_policy=config.projection.overlap_policy,
                emit_warnings=False,
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
            update_valley_little_group_inventory(
                symmetry_payload=symmetry_payload,
                kpoint_name=kpoint_name,
                k_frac=kpoint.frac,
            )
        if symmetry_payload["status"] == "ok" and symmetry_payload.get("symmetry_eigenvalue_enabled", True):
            symmetry_rows.extend(
                symmetry_eigenvalue_diagnostics_for_kpoint(
                    kpoint_name=kpoint_name,
                    k_frac=kpoint.frac,
                    q_cart=q_cart,
                    coefficients=coefficients,
                    symmetry_payload=symmetry_payload,
                    basis_payload=basis_transforms.get(kpoint_name),
                    representation_payload=symmetry_representation_payload,
                    spinor_convention_verified=config.spinor.convention_verified,
                    spinor_convention=config.spinor.convention,
                    spinor_benchmark=config.spinor.benchmark,
                    unitarity_tol=config.rotation.unitarity_tol,
                    root_deviation_tol=config.rotation.root_deviation_tol,
                    d_valley_offdiag_tol=config.rotation.D_valley_offdiag_tol,
                    generators_only=False,
                )
            )
        kpoint_subspace["symmetry_status"] = _resolve_symmetry_status(
            symmetry_payload, symmetry_rows, kpoint_name,
        )

    if symmetry_payload["status"] == "ok":
        build_valley_preserving_subgroup_report(
            symmetry_payload=symmetry_payload,
            target_kpoints=config.analysis.kpoints,
        )
        add_valley_irrep_results(
            symmetry_payload=symmetry_payload,
            symmetry_rows=symmetry_rows,
        )

    sector_names = list(projectors_by_kpoint[next(iter(projectors_by_kpoint))].sector_masks)
    symmetry_eigenvalue_summary = _build_symmetry_eigenvalue_summary(symmetry_payload, symmetry_rows)
    outputs = write_analysis_outputs(
        config=config,
        qcut=qcut,
        weight_rows=rows,
        sector_names=sector_names,
        subspace_payload=subspace_payload,
        symmetry_payload=symmetry_payload,
        symmetry_rows=symmetry_rows,
        projectors_by_kpoint=projectors_by_kpoint,
        qcut_scan_payload=qcut_scan_payload,
        symmetry_representation_payload=symmetry_representation_payload,
        basis_transforms=basis_transforms,
        symmetry_eigenvalue_summary=symmetry_eigenvalue_summary,
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
    thresholds: dict[str, float] | None = None,
    max_w_overlap: float = 0.0,
    max_w_res: float = 0.0,
) -> None:
    w_val_min = float(thresholds.get("W_val_min", 0.8)) if thresholds else 0.8
    concentration_threshold = float(thresholds.get("concentration_threshold", 0.95)) if thresholds else 0.95
    sector_names = projectors.sector_names
    n_valleys = len(sector_names)
    energy_span_meV = float((np.max(energies_eV) - np.min(energies_eV)) * 1000.0)
    diagnostic: dict[str, object] = {
        "band_indices_vasp": np.asarray(band_indices_vasp, dtype=int),
        "energy_span_meV": energy_span_meV,
        "status": "not_evaluated",
        "n_valleys": n_valleys,
    }
    if coefficients.shape[0] < 2:
        diagnostic["status"] = "single_band"
    elif energy_span_meV > degeneracy_tol_meV:
        diagnostic["status"] = "not_degenerate"
    elif n_valleys < 1:
        diagnostic["status"] = "no_valley_sectors"
    else:
        # General multi-valley adapted basis
        result = build_valley_adapted_basis(
            coefficients,
            projectors.sector_masks,
        )
        diagnosed = diagnose_valley_separability(
            result,
            w_val_min=w_val_min,
            concentration_threshold=concentration_threshold,
        )
        s_eigenvalues = np.linalg.eigvalsh(diagnosed.s_matrix)
        s_min = float(np.min(s_eigenvalues)) if len(s_eigenvalues) else 0.0
        s_max = float(np.max(s_eigenvalues)) if len(s_eigenvalues) else 0.0
        valid_valley_subspace = bool(diagnosed.stably_separable)

        # Map separability to status string
        if diagnosed.stably_separable:
            status = "valley_separable"
        elif "insufficient_valley_derived" in diagnosed.reason:
            status = "poor_valley_manifold"
        elif diagnosed.min_valley_concentration >= concentration_threshold * 0.9:
            status = "valley_approximately_separable"
        elif "concentration" in diagnosed.reason:
            status = "valley_mixed"
        elif "commut" in diagnosed.reason or "idempotency" in diagnosed.reason:
            status = "projector_unreliable"
        else:
            status = "valley_mixed"

        subspace_derived = derive_derived_score(analysis_level="adapted_subspace", s_min=s_min)
        # Use concentration for polarization score in multi-valley, eta for two-valley
        subspace_polarization = derive_polarization_score(
            analysis_level="adapted_subspace",
            eta_adapted=diagnosed.eta_adapted,
            purity=diagnosed.min_valley_concentration,
        )
        diagnostic.update(
            {
                "status": status,
                "valleys": sector_names,
                "n_valleys": n_valleys,
                "s_eigenvalues": s_eigenvalues,
                "s_min": s_min,
                "s_max": s_max,
                "valid_valley_subspace": valid_valley_subspace,
                "valley_weights_adapted": diagnosed.valley_weights_adapted,
                "assigned_valleys": diagnosed.assigned_valleys,
                "valley_concentration": diagnosed.valley_concentration,
                "min_valley_concentration": diagnosed.min_valley_concentration,
                "stably_separable": diagnosed.stably_separable,
                "reason": diagnosed.reason,
                "commutator_norm_max": diagnosed.commutator_norm_max,
                "idempotency_deviation_max": diagnosed.idempotency_deviation_max,
                "transform_h5_group": kpoint_name,
            }
        )
        if n_valleys == 2:
            diagnostic["eta"] = diagnosed.eta_adapted
            diagnostic["max_abs_eta"] = (
                float(max(abs(v) for v in diagnosed.eta_adapted))
                if diagnosed.eta_adapted is not None and len(diagnosed.eta_adapted) > 0
                else 0.0
            )
            # Also provide legacy v_matrix for two-valley backward compat
            diagnostic["v_eigenvalues"] = np.linalg.eigvalsh(
                diagnosed.valley_matrices[sector_names[0]]
                - diagnosed.valley_matrices[sector_names[1]]
            )
        payload["derived_score"] = subspace_derived
        payload["polarization_score"] = subspace_polarization
        payload["subspace_valley_status"] = derive_valley_status(
            analysis_level="adapted_subspace",
            derived_score=subspace_derived,
            polarization_score=subspace_polarization,
            w_overlap=max_w_overlap,
            w_res=max_w_res,
            thresholds=thresholds,
            two_sector=(n_valleys == 2),
        )
        if valid_valley_subspace:
            transform_entry: dict[str, object] = {
                "transform": diagnosed.transform,
                "s_matrix": diagnosed.s_matrix,
                "band_indices_vasp": np.asarray(band_indices_vasp, dtype=int),
                "sectors": np.asarray(sector_names, dtype="S"),
                "valleys": np.asarray(sector_names, dtype="S"),
                "valid_valley_subspace": np.asarray(valid_valley_subspace),
                "s_eigenvalues": s_eigenvalues,
                "valley_weights_adapted": diagnosed.valley_weights_adapted,
                "assigned_valleys": np.asarray(diagnosed.assigned_valleys, dtype="S"),
                "valley_concentration": diagnosed.valley_concentration,
                "label_operator": diagnosed.label_operator,
            }
            if diagnosed.eta_adapted is not None:
                transform_entry["eta"] = diagnosed.eta_adapted
            if n_valleys == 2:
                transform_entry["v_matrix"] = (
                    diagnosed.valley_matrices[sector_names[0]]
                    - diagnosed.valley_matrices[sector_names[1]]
                )
            basis_transforms[kpoint_name] = transform_entry
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
            "rotation_order": symmetry.filters.rotation_order,
        },
        "symmetry_eigenvalue_enabled": False,
        "requested_rotation_order": symmetry.filters.rotation_order,
        "resolved_rotation_order": None,
        "little_group_check": {"required": True, "status": "not_run"},
        "valley_preservation_check": {"required": True, "status": "not_run"},
    }
    if structure_file is None:
        return {
            **base_payload,
            "status": "skipped",
            "reason": (
                "symmetry.operations.structure_file is missing. "
                "Symmetry-operation detection requires the moire/bilayer POSCAR or CONTCAR; "
                "input.monolayer_poscars are used for monolayer reciprocal geometry and valley centers."
            ),
        }
    try:
        cell = read_poscar_cell(str(structure_file))
    except (FileNotFoundError, OSError, ValueError) as exc:
        return {
            **base_payload,
            "status": "skipped",
            "reason": (
                f"symmetry.operations.structure_file could not be read: {exc}. "
                "Symmetry-operation detection requires the moire/bilayer POSCAR or CONTCAR."
            ),
        }
    dataset = find_symmetry_operations(cell, symmetry.tolerance.symprec, symmetry.tolerance.angle_tolerance)
    lattice = np.asarray(cell[0], dtype=float)
    candidate_orders = _candidate_rotation_orders(dataset.rotations)
    resolved_rotation_order = resolve_rotation_order(
        symmetry.filters.rotation_order,
        international=dataset.international,
        candidate_orders=candidate_orders,
    )
    effective_allowed_orders = [] if resolved_rotation_order is None else [resolved_rotation_order]
    inv_direct_T = np.linalg.inv(lattice.T)
    operations = []
    for op_id, (rotation, translation) in enumerate(zip(dataset.rotations, dataset.translations)):
        info = classify_operation(rotation, translation, allowed_orders=effective_allowed_orders)
        rotation_cart = cart_rotation_from_fractional(rotation, lattice, inv_direct_T=inv_direct_T)
        translation_cart = cart_translation_from_fractional(translation, lattice)
        valley_mapping = map_valley_sectors(
            rotation,
            rotation_cart,
            config.valley_centers,
            config.valley_subspaces,
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
    mark_rotation_generators(operations)
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
        "symmetry_eigenvalue_enabled": resolved_rotation_order is not None,
        "requested_rotation_order": symmetry.filters.rotation_order,
        "resolved_rotation_order": resolved_rotation_order,
        "symprec_scan_summary": symprec_scan_summary,
        "lattice_direct_cart": lattice,
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
            detected_orders = _candidate_rotation_orders(dataset.rotations)
            resolved_rotation_order = resolve_rotation_order(
                config.symmetry.filters.rotation_order,
                international=dataset.international,
                candidate_orders=detected_orders,
            )
            effective_allowed_orders = [] if resolved_rotation_order is None else [resolved_rotation_order]
            for rotation, translation in zip(dataset.rotations, dataset.translations):
                info = classify_operation(
                    rotation,
                    translation,
                    allowed_orders=effective_allowed_orders,
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
                    "requested_rotation_order": config.symmetry.filters.rotation_order,
                    "resolved_rotation_order": resolved_rotation_order,
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


def _candidate_rotation_orders(rotations: list[np.ndarray]) -> list[int]:
    orders: list[int] = []
    for rotation in rotations:
        info = classify_operation(rotation, np.zeros(3), allowed_orders=[2, 3, 4, 6])
        if info.det == 1 and info.order in {2, 3, 4, 6}:
            orders.append(int(info.order))
    return orders


def _build_weight_entry(
    *,
    band_vasp: int,
    result,
    thresholds: dict[str, float],
) -> dict[str, object]:
    sector_count = len(result.sector_weights)
    two_sector = sector_count == 2
    eta_raw = result.eta if two_sector else None
    derived = derive_derived_score(analysis_level="raw_state", w_val=result.w_val)
    polarization = derive_polarization_score(
        analysis_level="raw_state",
        eta_raw=eta_raw if two_sector else None,
        purity=None if two_sector else result.purity,
    )
    status = derive_valley_status(
        analysis_level="raw_state",
        derived_score=derived,
        polarization_score=polarization,
        w_overlap=result.overlap_weight,
        w_res=result.residual_weight,
        thresholds=thresholds,
        two_sector=two_sector,
    )
    return {
        "band_vasp": band_vasp,
        "analysis_level": "raw_state",
        "derived_score": derived,
        "polarization_score": polarization,
        "valley_status": status,
        "sector_weights": result.sector_weights,
        "W_val": result.w_val,
        "P_v": result.purity,
        "eta": eta_raw,
        "W_overlap": result.overlap_weight,
        "W_res": result.residual_weight,
    }


def _resolve_symmetry_status(
    symmetry_payload: dict[str, object],
    symmetry_rows: list[dict[str, object]],
    kpoint_name: str,
) -> str:
    symmetry_skipped = symmetry_payload.get("status") == "skipped"
    if symmetry_payload.get("symmetry_eigenvalue_enabled") is False:
        return "not_requested"
    has_topology_ready = any(
        row.get("topology_input_ready") for row in symmetry_rows
        if row.get("kpoint") == kpoint_name
    )
    has_diagnostic = any(
        row.get("kpoint") == kpoint_name for row in symmetry_rows
    )

    little_group_passed: bool | None = None
    valley_preserving: bool | None = None
    for op in symmetry_payload.get("detected_operations", []):
        if not op.get("candidate_rotation", False):
            continue
        lg = op.get("little_group_by_kpoint", {}).get(kpoint_name)
        if lg is False:
            little_group_passed = False
            continue
        if lg is True:
            little_group_passed = True
            allowed = op.get("allowed_for_single_valley_representation_by_kpoint", {}).get(kpoint_name, False)
            if not allowed:
                preserved = op.get("preserved", {})
                if any(not bool(v) for v in preserved.values()):
                    valley_preserving = False
                else:
                    valley_preserving = True

    return derive_symmetry_status(
        symmetry_skipped=symmetry_skipped,
        little_group_passed=little_group_passed,
        valley_preserving=valley_preserving,
        topology_input_ready=has_topology_ready if has_topology_ready else (False if has_diagnostic and not has_topology_ready else None),
    )


def _build_symmetry_eigenvalue_summary(
    symmetry_payload: dict[str, object],
    symmetry_rows: list[dict[str, object]],
) -> dict[str, object]:
    ops = symmetry_payload.get("detected_operations", [])
    total = len(ops)
    by_kpoint: dict[str, dict[str, object]] = {}
    for row in symmetry_rows:
        kp = str(row.get("kpoint", ""))
        if kp not in by_kpoint:
            by_kpoint[kp] = {"operations": [], "_by_operation": {}}
        by_operation = by_kpoint[kp]["_by_operation"]
        key = str(row.get("operation_id"))
        if key not in by_operation:
            accepted = bool(row.get("little_group_passed", False)) and bool(row.get("valley_preserving", False))
            op_info = {
                "operation_id": row.get("operation_id"),
                "kind": row.get("kind"),
                "order": row.get("order"),
                "accepted": accepted,
                "reason": "" if accepted else row.get("reason"),
                "diagnostic_reasons": [],
                "topology_input_ready": [],
                "character_valley": row.get("character_valley"),
                "character_raw": row.get("character_raw"),
                "eigenvalues": [],
            }
            by_operation[key] = op_info
            by_kpoint[kp]["operations"].append(op_info)
        by_operation[key]["eigenvalues"].append(row.get("phase_2pi"))
        by_operation[key]["topology_input_ready"].append(bool(row.get("topology_input_ready", False)))
        if row.get("reason") and row.get("little_group_passed", False) and row.get("valley_preserving", False):
            reasons = by_operation[key]["diagnostic_reasons"]
            if row.get("reason") not in reasons:
                reasons.append(row.get("reason"))
        if row.get("character_valley"):
            by_operation[key]["character_valley"] = row.get("character_valley")
        if row.get("character_raw"):
            by_operation[key]["character_raw"] = row.get("character_raw")
    for kp_info in by_kpoint.values():
        kp_info.pop("_by_operation", None)

    little_count = 0
    vp_count = 0
    computed = sum(len(kp_info.get("operations", [])) for kp_info in by_kpoint.values())
    for op in ops:
        has_lg = any(
            bool(v) for v in op.get("little_group_by_kpoint", {}).values()
        )
        has_vp = all(
            bool(v) for v in op.get("preserved", {}).values()
        )
        if has_lg:
            little_count += 1
        if has_lg and has_vp:
            vp_count += 1

    return {
        "total_operations": total,
        "little_group_count": little_count,
        "valley_preserving_count": vp_count,
        "computed_count": computed,
        "irrep_label_matching": "deferred",
        "by_kpoint": by_kpoint,
    }
