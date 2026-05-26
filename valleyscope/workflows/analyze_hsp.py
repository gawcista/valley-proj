from __future__ import annotations

from pathlib import Path

import numpy as np

from valleyscope.analysis.projector_symmetry import (
    apply_projector_symmetry_gate,
    build_projector_symmetry_report,
)
from valleyscope.analysis.hsp_star import build_hsp_star_report
from valleyscope.analysis.symmetry_adapted_valley_report import (
    build_symmetry_adapted_valley_report,
    summarize_symmetry_adapted_valley_report,
)
from valleyscope.analysis.decision_tree import (
    _resolve_concentration_thresholds,
    derive_derived_score,
    derive_polarization_score,
    derive_symmetry_status,
    derive_valley_status,
)
from valleyscope.analysis.symmetry_eigenvalue_diagnostic import (
    build_raw_representations_for_kpoint,
    symmetry_eigenvalue_diagnostics_for_kpoint,
)
from valleyscope.analysis.valley_little_group import (
    add_valley_irrep_results,
    build_valley_preserving_subgroup_report,
    update_valley_preserving_operation_inventory,
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
    summarize_valley_projector_quality,
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
    raw_representations_by_kpoint: dict[str, dict[object, dict[str, object]]] = {}
    valley_matrices_by_kpoint: dict[str, dict[str, np.ndarray]] = {}
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
        seed_matrices = _add_valley_subspace_diagnostic(
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
        if seed_matrices is not None:
            valley_matrices_by_kpoint[kpoint_name] = seed_matrices
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
        valley_names = list(projectors.sector_names)
        if symmetry_payload["status"] == "ok":
            update_valley_preserving_operation_inventory(
                symmetry_payload=symmetry_payload,
                kpoint_name=kpoint_name,
                k_frac=kpoint.frac,
                valley_names=valley_names,
            )
        if symmetry_payload["status"] == "ok" and symmetry_payload.get("symmetry_eigenvalue_enabled", True):
            # Build D_raw for ALL proper little-group ops before per-valley gate,
            # so valley-permuting operations (e.g. C3 cycling M1/M2/M3) are included
            # in the projector symmetry-consistency diagnostic.
            if seed_matrices is not None:
                raw_representations_by_kpoint[kpoint_name] = (
                    build_raw_representations_for_kpoint(
                        kpoint_name=kpoint_name,
                        k_frac=kpoint.frac,
                        q_cart=q_cart,
                        coefficients=coefficients,
                        symmetry_payload=symmetry_payload,
                        spinor_convention_verified=config.spinor.convention_verified,
                    )
                )
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
                    valley_names=valley_names,
                )
            )
        kpoint_subspace["symmetry_status"] = _resolve_symmetry_status(
            symmetry_payload, symmetry_rows, kpoint_name,
        )

    # --- Projector symmetry-consistency diagnostic ---
    valley_names = list(
        projectors_by_kpoint[next(iter(projectors_by_kpoint))].sector_names
    ) if projectors_by_kpoint else []
    projector_symmetry_report: dict[str, object] | None = None
    if symmetry_payload["status"] == "ok" and symmetry_payload.get("symmetry_eigenvalue_enabled", True):
        projector_symmetry_report = build_projector_symmetry_report(
            valley_matrices_by_kpoint=valley_matrices_by_kpoint,
            raw_representations_by_kpoint=raw_representations_by_kpoint,
            valley_names=valley_names,
        )
        apply_projector_symmetry_gate(
            symmetry_rows=symmetry_rows,
            projector_symmetry_report=projector_symmetry_report,
        )

    if symmetry_payload["status"] == "ok":
        symmetry_payload["hsp_star_report"] = build_hsp_star_report(
            kpoint_frac_by_name=symmetry_payload.get("kpoint_frac_by_name", {}),
            operations=symmetry_payload.get("detected_operations", []),
        )

    # --- Formal symmetry-adapted valley report ---
    symmetry_adapted_valley_report: dict[str, object] | None = None
    if config.symmetry_adapted_valley.enabled:
        symmetry_adapted_valley_report = _build_symmetry_adapted_valley_report(
            valley_matrices_by_kpoint=valley_matrices_by_kpoint,
            raw_representations_by_kpoint=raw_representations_by_kpoint,
            symmetry_payload=symmetry_payload,
            config=config,
        )

    if symmetry_payload["status"] == "ok":
        build_valley_preserving_subgroup_report(
            symmetry_payload=symmetry_payload,
            target_kpoints=config.analysis.kpoints,
        )
        add_valley_irrep_results(
            symmetry_payload=symmetry_payload,
            symmetry_rows=symmetry_rows,
            representation_payload=symmetry_representation_payload,
            tolerance=config.rotation.irrep_weight_tol,
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
        projector_symmetry_report=projector_symmetry_report,
        symmetry_adapted_valley_report=symmetry_adapted_valley_report,
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
) -> dict[str, np.ndarray] | None:
    """Returns seed projected valley matrices {valley_name: P_a^0} or None."""
    w_val_min = float(thresholds.get("W_val_min", 0.8)) if thresholds else 0.8
    # Resolve via new naming with legacy fallback
    concentration_clean, _ = _resolve_concentration_thresholds(thresholds)
    concentration_threshold = concentration_clean
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
        payload["valley_adapted_subspace"] = diagnostic
        return None
    if energy_span_meV > degeneracy_tol_meV:
        diagnostic["status"] = "not_degenerate"
        payload["valley_adapted_subspace"] = diagnostic
        return None
    if n_valleys < 1:
        diagnostic["status"] = "no_valley_sectors"
        payload["valley_adapted_subspace"] = diagnostic
        return None

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

    subspace_derived = derive_derived_score(analysis_level="adapted_subspace", s_min=s_min)
    subspace_polarization = derive_polarization_score(
        analysis_level="adapted_subspace",
        eta_adapted=diagnosed.eta_adapted,
        purity=diagnosed.min_valley_concentration,
    )
    subspace_valley_status = derive_valley_status(
        analysis_level="adapted_subspace",
        derived_score=subspace_derived,
        polarization_score=subspace_polarization,
        w_overlap=max_w_overlap,
        w_res=max_w_res,
        thresholds=thresholds,
        two_sector=(n_valleys == 2),
    )
    valid_valley_subspace = subspace_valley_status == "valley_separable_subspace"
    write_valley_basis = subspace_valley_status in {
        "valley_separable_subspace",
        "valley_approximately_separable_subspace",
    }

    if subspace_valley_status == "valley_separable_subspace":
        status = "valley_separable"
    elif subspace_valley_status == "valley_approximately_separable_subspace":
        status = "valley_approximately_separable"
    elif subspace_valley_status == "not_valley_derived":
        status = "poor_valley_manifold"
    elif subspace_valley_status == "projector_unreliable":
        status = "projector_unreliable"
    elif "concentration" in diagnosed.reason:
        status = "valley_mixed"
    else:
        status = "valley_mixed"

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
            "projector_quality": summarize_valley_projector_quality(
                diagnosed.valley_matrices,
                expected_rank=(
                    coefficients.shape[0] // n_valleys
                    if n_valleys > 0 and coefficients.shape[0] % n_valleys == 0
                    else None
                ),
            ),
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
        diagnostic["v_eigenvalues"] = np.linalg.eigvalsh(
            diagnosed.valley_matrices[sector_names[0]]
            - diagnosed.valley_matrices[sector_names[1]]
        )
    payload["derived_score"] = subspace_derived
    payload["polarization_score"] = subspace_polarization
    payload["subspace_valley_status"] = subspace_valley_status
    if write_valley_basis:
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
    return dict(diagnosed.valley_matrices)


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
        "valley_weights": result.sector_weights,
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
        lg = op.get("little_group_by_kpoint", {}).get(kpoint_name)
        if lg is False:
            little_group_passed = False
            continue
        if lg is True:
            little_group_passed = True
            sector_mapping = op.get("sector_mapping", {})
            preserves_any = any(
                v is not None and str(v) == str(src)
                for src, v in sector_mapping.items()
            )
            if preserves_any:
                valley_preserving = True
            else:
                valley_preserving = False

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
        op_id = str(row.get("operation_id"))
        target_valley = str(row.get("target_valley", ""))
        key = f"{op_id}_{target_valley}"
        if key not in by_operation:
            accepted = bool(row.get("little_group_passed", False)) and bool(row.get("valley_preserving", False))
            op_info = {
                "operation_id": row.get("operation_id"),
                "target_valley": row.get("target_valley"),
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
    vp_count = sum(
        1 for op in ops
        if any(
            v is not None and str(v) == str(src)
            for src, v in op.get("sector_mapping", {}).items()
            if any(
                bool(op.get("little_group_by_kpoint", {}).get(kp))
                for kp in (symmetry_payload.get("kpoint_frac_by_name") or {}).keys()
            )
        )
    ) if ops else 0
    computed = sum(len(kp_info.get("operations", [])) for kp_info in by_kpoint.values())
    for op in ops:
        has_lg = any(
            bool(v) for v in op.get("little_group_by_kpoint", {}).values()
        )
        if has_lg:
            little_count += 1

    return {
        "total_operations": total,
        "little_group_count": little_count,
        "valley_preserving_count": vp_count,
        "computed_count": computed,
        "irrep_label_matching": "deferred",
        "by_kpoint": by_kpoint,
    }


def _build_symmetry_adapted_valley_report(
    *,
    valley_matrices_by_kpoint: dict[str, dict[str, np.ndarray]],
    raw_representations_by_kpoint: dict[str, dict[object, dict[str, object]]],
    symmetry_payload: dict[str, object],
    config: object,
) -> dict[str, object]:
    """Build per-kpoint symmetry-adapted valley analysis.

    Returns a ``by_kpoint`` dict keyed by kpoint label. Each entry is a
    ``not_evaluated`` stub or contains orbit-level reports when inputs are
    sufficient for the toy pipeline.
    """
    by_kpoint: dict[str, object] = {}
    valley_names = list(
        symmetry_payload.get("valley_names", [])
        or [sector.name for sector in getattr(config, "valley_subspaces", [])]
    )
    target_kpoints = list(getattr(config.analysis, "kpoints", []))
    if not target_kpoints:
        target_kpoints = sorted(
            set(valley_matrices_by_kpoint) | set(raw_representations_by_kpoint)
        )
    (
        space_group_valley_mappings,
        space_group_operation_orders,
    ) = _space_group_valley_mapping_payload(
        symmetry_payload=symmetry_payload,
        valley_names=valley_names,
    )

    for kpoint_name in target_kpoints:
        valley_matrices = valley_matrices_by_kpoint.get(kpoint_name, {})
        raw_reps = raw_representations_by_kpoint.get(kpoint_name, {})
        if not valley_names or not raw_reps or not valley_matrices:
            by_kpoint[kpoint_name] = _not_evaluated_symmetry_adapted_kpoint(
                "missing seed projectors, D_raw, or valley_names"
            )
            continue

        # Build representations dict from raw_reps
        d_g_dict: dict[object, np.ndarray] = {}
        valley_mappings_dict: dict[object, dict[str, str]] = {}
        operation_orders_by_id: dict[object, int] = {}
        for op_id, op_data in raw_reps.items():
            if not isinstance(op_data, dict):
                continue
            d_raw = op_data.get("D_raw")
            vm = op_data.get("sector_mapping", {})
            if d_raw is not None and vm:
                d_g_dict[op_id] = np.asarray(d_raw, dtype=np.complex128)
                valley_mappings_dict[op_id] = {str(k): str(v) for k, v in vm.items()}
                try:
                    operation_orders_by_id[op_id] = int(op_data.get("order", 0))
                except (TypeError, ValueError):
                    pass
        fallback_dim = None
        if valley_matrices:
            first_matrix = next(iter(valley_matrices.values()))
            fallback_dim = int(np.asarray(first_matrix).shape[0])
        _add_identity_representation_if_missing(
            d_g_dict=d_g_dict,
            valley_mappings_dict=valley_mappings_dict,
            valley_names=valley_names,
            symmetry_payload=symmetry_payload,
            fallback_dim=fallback_dim,
        )
        for op_id, mapping in valley_mappings_dict.items():
            if op_id not in operation_orders_by_id and all(
                str(mapping.get(valley)) == str(valley) for valley in valley_names
            ):
                d_g = np.asarray(d_g_dict.get(op_id))
                if d_g.shape == (fallback_dim, fallback_dim) and np.allclose(
                    d_g, np.eye(fallback_dim, dtype=np.complex128), atol=1e-10,
                ):
                    operation_orders_by_id[op_id] = 1

        if not d_g_dict:
            by_kpoint[kpoint_name] = _not_evaluated_symmetry_adapted_kpoint(
                "no valid D_raw with valley_mapping"
            )
            continue

        orbits = _partition_valley_orbits(
            valley_names=valley_names,
            valley_mappings=valley_mappings_dict,
        )
        orbit_reports: list[dict[str, object]] = []
        for orbit in orbits:
            seed_projectors = {
                valley: valley_matrices[valley]
                for valley in orbit
                if valley in valley_matrices
            }
            if len(seed_projectors) != len(orbit):
                missing = [valley for valley in orbit if valley not in seed_projectors]
                orbit_reports.append(
                    {
                        "status": "not_evaluated",
                        "reason": f"missing seed projectors for orbit valleys: {missing}",
                        "diagnostic_only": True,
                        "local_irrep_ready": False,
                        "feature_status": "formal",
                        "workflow_integration_status": "integrated",
                        "trusted_irrep_label": False,
                        "irrep_matching_input_ready": False,
                        "irrep_matching_input_status": "not_evaluated",
                        "irrep_matching_input_reason": (
                            f"missing seed projectors for orbit valleys: {missing}"
                        ),
                        "orbit": orbit,
                        "reference_valley": orbit[0] if orbit else "",
                    }
                )
                continue

            report = build_symmetry_adapted_valley_report(
                seed_projectors=seed_projectors,
                representations=d_g_dict,
                valley_mappings=valley_mappings_dict,
                orbit=orbit,
                reference_valley=orbit[0],
                rank=None,
                rank_method="gap",
                unitarity_tol=float(config.symmetry_adapted_valley.representation_unitarity_fail_tol),
                modulus_tol=float(config.rotation.root_deviation_tol),
                spinor_wavefunction=bool(symmetry_payload.get("spinor_wavefunction", False)),
                spinor_convention_verified=bool(config.spinor.convention_verified),
                operation_orders=operation_orders_by_id,
                seed_overlap_warn_tol=float(config.symmetry_adapted_valley.seed_overlap_warn_tol),
                seed_overlap_fail_tol=float(config.symmetry_adapted_valley.seed_overlap_fail_tol),
                projector_symmetry_warn_tol=float(config.symmetry_adapted_valley.projector_symmetry_warn_tol),
                projector_symmetry_fail_tol=float(config.symmetry_adapted_valley.projector_symmetry_fail_tol),
                ebr_seed_overlap_min=float(config.symmetry_adapted_valley.ebr_seed_overlap_min),
                ebr_unitarity_max=float(config.symmetry_adapted_valley.ebr_unitarity_max),
            )
            orbit_reports.append(summarize_symmetry_adapted_valley_report(report))

        valley_preserving_subspaces = _build_valley_preserving_subspace_reports(
            valley_matrices=valley_matrices,
            d_g_dict=d_g_dict,
            valley_mappings_dict=valley_mappings_dict,
            valley_names=valley_names,
            unitarity_tol=float(config.symmetry_adapted_valley.representation_unitarity_fail_tol),
            modulus_tol=float(config.rotation.root_deviation_tol),
            spinor_wavefunction=bool(symmetry_payload.get("spinor_wavefunction", False)),
            spinor_convention_verified=bool(config.spinor.convention_verified),
            operation_orders_by_id=operation_orders_by_id,
            seed_overlap_warn_tol=float(config.symmetry_adapted_valley.seed_overlap_warn_tol),
            seed_overlap_fail_tol=float(config.symmetry_adapted_valley.seed_overlap_fail_tol),
            projector_symmetry_warn_tol=float(config.symmetry_adapted_valley.projector_symmetry_warn_tol),
            projector_symmetry_fail_tol=float(config.symmetry_adapted_valley.projector_symmetry_fail_tol),
            ebr_seed_overlap_min=float(config.symmetry_adapted_valley.ebr_seed_overlap_min),
            ebr_unitarity_max=float(config.symmetry_adapted_valley.ebr_unitarity_max),
            space_group_valley_mappings=space_group_valley_mappings,
            space_group_operation_orders=space_group_operation_orders,
        )

        by_kpoint[kpoint_name] = _aggregate_symmetry_adapted_kpoint(
            orbit_reports,
            valley_preserving_subspaces=valley_preserving_subspaces,
        )

    return {"by_kpoint": by_kpoint}


def _build_valley_preserving_subspace_reports(
    *,
    valley_matrices: dict[str, np.ndarray],
    d_g_dict: dict[object, np.ndarray],
    valley_mappings_dict: dict[object, dict[str, str]],
    valley_names: list[str],
    unitarity_tol: float,
    modulus_tol: float,
    spinor_wavefunction: bool,
    spinor_convention_verified: bool,
    operation_orders_by_id: dict[object, int] | None = None,
    seed_overlap_warn_tol: float = 0.8,
    seed_overlap_fail_tol: float = 0.5,
    projector_symmetry_warn_tol: float = 1e-2,
    projector_symmetry_fail_tol: float = 1e-1,
    ebr_seed_overlap_min: float = 0.8,
    ebr_unitarity_max: float = 1e-3,
    space_group_valley_mappings: dict[object, dict[str, str]] | None = None,
    space_group_operation_orders: dict[object, int] | None = None,
) -> list[dict[str, object]]:
    """Build singleton reports for per-valley preserving-subgroup analysis.

    These reports answer the local question: for each valley a, what does the
    subgroup G_k^(a) do inside one symmetry-adapted valley subspace?  They are
    intentionally separate from the full valley-orbit report, which also tracks
    valley-changing operations and sewing data.
    """
    inferred_rank, rank_source = _infer_uniform_local_valley_rank(
        valley_matrices=valley_matrices,
        valley_names=valley_names,
    )
    reports: list[dict[str, object]] = []
    for valley in valley_names:
        if valley not in valley_matrices:
            reports.append(
                _not_evaluated_valley_preserving_subspace(
                    valley=valley,
                    reason=f"missing seed projector for valley: {valley}",
                    rank_source=rank_source,
                )
            )
            continue

        preserving_ops = [
            op_id for op_id, mapping in valley_mappings_dict.items()
            if op_id in d_g_dict and str(mapping.get(valley)) == str(valley)
        ]
        preserving_ops = sorted(preserving_ops, key=_operation_sort_key)
        if not preserving_ops:
            reports.append(
                _not_evaluated_valley_preserving_subspace(
                    valley=valley,
                    reason=f"no valley-preserving operation found for {valley}",
                    rank_source=rank_source,
                )
            )
            continue

        local_representations = {
            op_id: np.asarray(d_g_dict[op_id], dtype=np.complex128)
            for op_id in preserving_ops
        }
        local_mappings = {
            op_id: {str(valley): str(valley)}
            for op_id in preserving_ops
        }
        report = build_symmetry_adapted_valley_report(
            seed_projectors={str(valley): valley_matrices[valley]},
            representations=local_representations,
            valley_mappings=local_mappings,
            orbit=[str(valley)],
            reference_valley=str(valley),
            rank=inferred_rank,
            rank_method="gap",
            unitarity_tol=unitarity_tol,
            modulus_tol=modulus_tol,
            spinor_wavefunction=spinor_wavefunction,
            spinor_convention_verified=spinor_convention_verified,
            operation_orders=operation_orders_by_id,
            seed_overlap_warn_tol=seed_overlap_warn_tol,
            seed_overlap_fail_tol=seed_overlap_fail_tol,
            projector_symmetry_warn_tol=projector_symmetry_warn_tol,
            projector_symmetry_fail_tol=projector_symmetry_fail_tol,
            ebr_seed_overlap_min=ebr_seed_overlap_min,
            ebr_unitarity_max=ebr_unitarity_max,
        )
        summary = summarize_symmetry_adapted_valley_report(report)
        summary["analysis_scope"] = "valley_preserving_subspace"
        summary["local_rank_source"] = rank_source
        summary["hsp_preserving_operation_ids"] = list(preserving_ops)
        summary["subspace_space_group"] = _build_subspace_space_group_for_valley(
            valley=str(valley),
            valley_mappings=space_group_valley_mappings or valley_mappings_dict,
            operation_orders=space_group_operation_orders or operation_orders_by_id or {},
        )
        ebr_mapping = summary.get("ebr_mapping_input")
        if isinstance(ebr_mapping, dict):
            _refine_ebr_mapping_with_subspace_space_group(
                ebr_mapping=ebr_mapping,
                subspace_space_group=summary["subspace_space_group"],
            )
        reports.append(summary)
    return reports


def _infer_uniform_local_valley_rank(
    *,
    valley_matrices: dict[str, np.ndarray],
    valley_names: list[str],
) -> tuple[int | None, str]:
    """Infer per-valley rank from target-subspace dimension when unambiguous."""
    if not valley_names or not valley_matrices:
        return None, "not_available"
    available = [str(v) for v in valley_names if str(v) in valley_matrices]
    if not available:
        return None, "not_available"
    first = np.asarray(valley_matrices[available[0]])
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        return None, "not_available"
    dim = int(first.shape[0])
    n_valleys = len(valley_names)
    if n_valleys > 0 and dim % n_valleys == 0:
        rank = dim // n_valleys
        if rank > 0:
            return rank, "target_dim_div_valley_count"
    return None, "auto_gap"


def _not_evaluated_valley_preserving_subspace(
    *,
    valley: str,
    reason: str,
    rank_source: str,
) -> dict[str, object]:
    return {
        "status": "not_evaluated",
        "reason": reason,
        "feature_status": "formal",
        "workflow_integration_status": "integrated",
        "trusted_irrep_label": False,
        "local_irrep_ready": False,
        "diagnostic_only": True,
        "irrep_matching_input_ready": False,
        "irrep_matching_input_status": "not_evaluated",
        "irrep_matching_input_reason": reason,
        "orbit": [str(valley)],
        "reference_valley": str(valley),
        "analysis_scope": "valley_preserving_subspace",
        "local_rank_source": rank_source,
        "hsp_preserving_operation_ids": [],
        "subspace_space_group": _empty_subspace_space_group(reason),
    }


def _operation_sort_key(op_id: object) -> tuple[int, object]:
    if isinstance(op_id, int):
        return (0, op_id)
    try:
        return (0, int(op_id))
    except (TypeError, ValueError):
        return (1, str(op_id))


def _space_group_valley_mapping_payload(
    *,
    symmetry_payload: dict[str, object],
    valley_names: list[str],
) -> tuple[dict[object, dict[str, str]], dict[object, int]]:
    """Collect full-space-group valley mappings, not restricted to one HSP.

    This drives the subspace space-group candidate.  It deliberately differs
    from the HSP little-group operation set: a C2 can preserve a monolayer
    valley label while mapping one moire HSP to another point in the HSP star.
    """
    valley_set = {str(valley) for valley in valley_names}
    mappings: dict[object, dict[str, str]] = {}
    orders: dict[object, int] = {}
    for operation in symmetry_payload.get("detected_operations", []):
        if not isinstance(operation, dict):
            continue
        op_id = operation.get("operation_id")
        if op_id is None:
            continue
        raw_mapping = operation.get("sector_mapping", {})
        if not isinstance(raw_mapping, dict):
            continue
        mapping: dict[str, str] = {}
        for valley in valley_set:
            mapped = raw_mapping.get(valley)
            if mapped is not None:
                mapping[valley] = str(mapped)
        if not mapping:
            continue
        mappings[op_id] = mapping
        try:
            orders[op_id] = int(operation.get("order", 0))
        except (TypeError, ValueError):
            pass
    return mappings, orders


def _build_subspace_space_group_for_valley(
    *,
    valley: str,
    valley_mappings: dict[object, dict[str, str]],
    operation_orders: dict[object, int],
) -> dict[str, object]:
    """Build the valley subspace space-group candidate from full operations."""
    preserving_ops = sorted(
        [
            op_id for op_id, mapping in valley_mappings.items()
            if str(mapping.get(valley)) == str(valley)
        ],
        key=_operation_sort_key,
    )
    changing_ops = sorted(
        [
            op_id for op_id, mapping in valley_mappings.items()
            if mapping.get(valley) is not None and str(mapping.get(valley)) != str(valley)
        ],
        key=_operation_sort_key,
    )
    non_identity_orders = [
        int(operation_orders[op_id])
        for op_id in preserving_ops
        if op_id in operation_orders and int(operation_orders[op_id]) > 1
    ]
    effective_order = max(non_identity_orders) if non_identity_orders else 1
    point_group = f"C{effective_order}" if effective_order > 1 else "C1"
    space_group = f"P{effective_order}" if effective_order > 1 else "P1"
    status = "candidate" if effective_order > 1 else "trivial"
    reason = (
        f"{space_group} candidate from valley-preserving operation order {effective_order}"
        if effective_order > 1 else
        "only identity preserves this valley in the detected space group"
    )
    return {
        "status": status,
        "candidate_space_group_symbol": space_group,
        "candidate_point_group": point_group,
        "valley_preserving_operation_ids": preserving_ops,
        "valley_changing_operation_ids": changing_ops,
        "operation_orders": {
            str(op_id): int(operation_orders[op_id])
            for op_id in preserving_ops + changing_ops
            if op_id in operation_orders
        },
        "source": "full_space_group_valley_mapping",
        "reason": reason,
    }


def _refine_ebr_mapping_with_subspace_space_group(
    *,
    ebr_mapping: dict[str, object],
    subspace_space_group: dict[str, object],
) -> None:
    """Attach full-space-group candidate without pretending local characters exist."""
    candidate = subspace_space_group.get("candidate_space_group_symbol")
    ebr_mapping["subspace_space_group_candidate"] = candidate
    blockers = ebr_mapping.get("blocked_by")
    if not isinstance(blockers, list):
        return
    if candidate in (None, "", "P1"):
        return
    refined_blockers = [
        (
            "hsp_local_preserving_character_missing"
            if blocker == "subspace_group_candidate_missing"
            else blocker
        )
        for blocker in blockers
    ]
    ebr_mapping["blocked_by"] = refined_blockers
    if "hsp_local_preserving_character_missing" in refined_blockers:
        notes = str(ebr_mapping.get("notes", "") or "")
        addition = (
            " Subspace space-group candidate is present, but this HSP does not "
            "contain a non-identity valley-preserving operation for this valley; "
            "use the corresponding HSP-star member before assigning local "
            "valley-preserving irreps."
        )
        if addition.strip() not in notes:
            ebr_mapping["notes"] = notes + addition


def _empty_subspace_space_group(reason: str) -> dict[str, object]:
    return {
        "status": "not_evaluated",
        "candidate_space_group_symbol": None,
        "candidate_point_group": None,
        "valley_preserving_operation_ids": [],
        "valley_changing_operation_ids": [],
        "operation_orders": {},
        "source": "full_space_group_valley_mapping",
        "reason": reason,
    }


def _not_evaluated_symmetry_adapted_kpoint(reason: str) -> dict[str, object]:
    return {
        "status": "not_evaluated",
        "reason": reason,
        "diagnostic_only": True,
        "local_irrep_ready": False,
        "feature_status": "formal",
        "workflow_integration_status": "integrated",
        "trusted_irrep_label": False,
        "irrep_matching_input_ready": False,
        "irrep_matching_input_status": "not_evaluated",
        "irrep_matching_input_reason": reason,
        "orbits": [],
        "valley_preserving_subspaces": [],
    }


def _aggregate_symmetry_adapted_kpoint(
    orbit_reports: list[dict[str, object]],
    *,
    valley_preserving_subspaces: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    valley_preserving_subspaces = list(valley_preserving_subspaces or [])
    if not orbit_reports:
        result = _not_evaluated_symmetry_adapted_kpoint("no valley orbits inferred")
        result["valley_preserving_subspaces"] = valley_preserving_subspaces
        return result
    diagnostic_only = any(bool(report.get("diagnostic_only", True)) for report in orbit_reports)
    local_irrep_ready = all(bool(report.get("local_irrep_ready", False)) for report in orbit_reports)
    irrep_matching_input_ready = all(
        bool(report.get("irrep_matching_input_ready", False))
        for report in orbit_reports
    )
    statuses = {str(report.get("status", "")) for report in orbit_reports}
    if diagnostic_only:
        status = "diagnostic_only"
    elif "warn" in statuses:
        status = "warn"
    else:
        status = "ok"
    reasons = [
        str(report.get("reason", ""))
        for report in orbit_reports
        if str(report.get("reason", "")) and str(report.get("reason", "")) != "all stages passed"
    ]
    irrep_reasons = [
        str(report.get("irrep_matching_input_reason", ""))
        for report in orbit_reports
        if not bool(report.get("irrep_matching_input_ready", False))
        and str(report.get("irrep_matching_input_reason", ""))
    ]
    return {
        "status": status,
        "reason": "; ".join(reasons) if reasons else "all orbits evaluated",
        "diagnostic_only": diagnostic_only,
        "local_irrep_ready": local_irrep_ready,
        "feature_status": "formal",
        "workflow_integration_status": "integrated",
        "trusted_irrep_label": False,
        "irrep_matching_input_ready": irrep_matching_input_ready,
        "irrep_matching_input_status": "ready" if irrep_matching_input_ready else "blocked",
        "irrep_matching_input_reason": (
            "all orbit reports are ready for irrep matching input"
            if irrep_matching_input_ready else "; ".join(irrep_reasons)
        ),
        "orbits": orbit_reports,
        "valley_preserving_subspaces": valley_preserving_subspaces,
    }


def _partition_valley_orbits(
    *,
    valley_names: list[str],
    valley_mappings: dict[object, dict[str, str]],
) -> list[list[str]]:
    """Infer valley orbit partitions from operation-induced valley mappings."""
    valley_set = {str(valley) for valley in valley_names}
    adjacency: dict[str, set[str]] = {str(valley): {str(valley)} for valley in valley_names}
    for mapping in valley_mappings.values():
        for src, tgt in mapping.items():
            src = str(src)
            tgt = str(tgt)
            if src in valley_set and tgt in valley_set:
                adjacency[src].add(tgt)
                adjacency[tgt].add(src)

    seen: set[str] = set()
    orbits: list[list[str]] = []
    for valley in [str(v) for v in valley_names]:
        if valley in seen:
            continue
        stack = [valley]
        component: list[str] = []
        seen.add(valley)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        ordered = [str(v) for v in valley_names if str(v) in set(component)]
        orbits.append(ordered)
    return orbits


def _add_identity_representation_if_missing(
    *,
    d_g_dict: dict[object, np.ndarray],
    valley_mappings_dict: dict[object, dict[str, str]],
    valley_names: list[str],
    symmetry_payload: dict[str, object],
    fallback_dim: int | None,
) -> bool:
    """Add identity D_g for symmetry-adapted projector analysis when D_raw omits it.

    `build_raw_representations_for_kpoint` is tuned for rotation-eigenvalue
    diagnostics and may skip order-1 operations.  Symmetry-adapted projector
    construction still needs the identity element because every
    valley-preserving subgroup contains it.
    """
    if not valley_names or fallback_dim is None:
        return False
    identity_mapping = {str(valley): str(valley) for valley in valley_names}
    for op_id, mapping in valley_mappings_dict.items():
        if all(str(mapping.get(valley)) == str(valley) for valley in valley_names):
            d_g = np.asarray(d_g_dict.get(op_id))
            if d_g.shape == (fallback_dim, fallback_dim) and np.allclose(
                d_g, np.eye(fallback_dim, dtype=np.complex128), atol=1e-10,
            ):
                return False

    op_id = _detected_identity_operation_id(symmetry_payload, valley_names)
    if op_id in d_g_dict:
        return False
    d_g_dict[op_id] = np.eye(fallback_dim, dtype=np.complex128)
    valley_mappings_dict[op_id] = identity_mapping
    return True


def _detected_identity_operation_id(
    symmetry_payload: dict[str, object],
    valley_names: list[str],
) -> object:
    identity_mapping = {str(valley): str(valley) for valley in valley_names}
    for operation in symmetry_payload.get("detected_operations", []):
        if not isinstance(operation, dict):
            continue
        try:
            order = int(operation.get("order", -1))
        except (TypeError, ValueError):
            continue
        mapping = {
            str(k): str(v)
            for k, v in dict(operation.get("sector_mapping", {})).items()
        }
        if order == 1 and all(mapping.get(valley) == target for valley, target in identity_mapping.items()):
            return operation.get("operation_id", "__identity__")
    return "__identity__"
