from __future__ import annotations

from pathlib import Path

import numpy as np

from valleyscope.analysis.projector_symmetry import (
    apply_projector_symmetry_gate,
    build_projector_symmetry_report,
)
from valleyscope.analysis.hsp_star import build_hsp_star_report
from valleyscope.analysis.hsp_star_conjugation import (
    build_hsp_star_conjugation_report,
    compute_target_kpoint_key,
)
from valleyscope.analysis.subspace_representation_quality import (
    build_subspace_representation_quality_report,
)
from valleyscope.analysis.irrep_workflow_decision import (
    build_irrep_workflow_decisions,
)
from valleyscope.analysis.valley_irrep_matching import (
    build_valley_irrep_matching_report,
)
from valleyscope.analysis.ebr_input_candidates import (
    build_ebr_input_candidates,
)
from valleyscope.analysis.ebr_problem_instances import (
    build_ebr_problem_instances,
)
from valleyscope.analysis.ebr_export_bundle import (
    build_ebr_export_bundle,
)
from valleyscope.analysis.projected_hsp_coverage import (
    build_projected_hsp_coverage_report,
    classify_projected_subspace_kpoint,
    derive_projected_subspace_source_hsp_basis,
)
from valleyscope.analysis.time_reversal_orbits import (
    build_time_reversal_valley_orbit_report,
    derive_time_reversal_valley_mapping,
)
from valleyscope.analysis.time_reversal_sewing import (
    build_time_reversal_sewing_report,
)
from valleyscope.irreps.tables import load_standard_irrep_table
from valleyscope.irreps.ebr_data_adapter import load_ebr_source_data
from valleyscope.irreps.time_reversal_ebr import (
    validate_grey_group_time_reversal_source,
)
from valleyscope.irreps.time_reversal_source import (
    derive_time_reversal_source_irrep_orbits,
)
from valleyscope.irreps.source_payload import (
    build_source_payload_for_projected_hsp_matching,
)
from valleyscope.analysis.valley_projected_representation import (
    build_valley_projected_representation_report,
)
from valleyscope.analysis.reduced_ebr_mapping import (
    build_reduced_ebr_mapping,
    load_reduced_ebr_table,
)
from valleyscope.analysis.hsp_star_derived_characters import (
    build_hsp_star_derived_characters,
    collect_derived_characters_by_target,
)
from valleyscope.analysis.target_subspace_closure import (
    build_target_subspace_closure_report,
    check_target_subspace_closure_blocked,
    check_target_subspace_closure_blocked_for_operation,
)
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
from valleyscope.projection.sector_projectors import (
    SectorProjectors,
    adjust_centers_for_parent_valley,
    build_sector_projectors,
)
from valleyscope.projection.weights import compute_valley_weights
from valleyscope.projection.folded_center import (
    build_folded_center_report,
    folded_center_report_to_dict,
)
from valleyscope.reports.analysis_outputs import (
    prepare_analysis_output_directory,
    write_analysis_outputs,
)
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


def _generic_irrep_override_blocker(
    *,
    computed_sg: int | None,
    wavefunction_spinor: bool,
    override_sg: int,
    override_spinor: bool | None,
) -> str | None:
    if computed_sg is not None and override_sg != computed_sg:
        return (
            f"generic_irrep_source override sg={override_sg} disagrees with "
            f"computed subgroup sg={computed_sg}"
        )
    if (
        override_spinor is not None
        and bool(override_spinor) != wavefunction_spinor
    ):
        return (
            f"generic_irrep_source override spinor={bool(override_spinor)} "
            f"disagrees with wavefunction spinor={wavefunction_spinor}"
        )
    return None


def _resolve_generic_irrep_hsp_label(
    *,
    table,
    k_frac: np.ndarray | None,
    override_label: str | None,
    standard_match: dict[str, object] | None = None,
) -> tuple[str | None, str | None]:
    label, blocker, _ = _resolve_generic_irrep_hsp_label_with_provenance(
        table=table,
        k_frac=k_frac,
        override_label=override_label,
        standard_match=standard_match,
    )
    return label, blocker


def _resolve_generic_irrep_hsp_label_with_provenance(
    *,
    table,
    k_frac: np.ndarray | None,
    override_label: str | None,
    standard_match: dict[str, object] | None = None,
    lattice_direct_cart: np.ndarray | None = None,
    detected_operations: list[dict[str, object]] | None = None,
    parent_to_standard_direct_transform: np.ndarray | None = None,
    origin_shift_fractional: np.ndarray | None = None,
    transform_provenance: str | None = None,
) -> tuple[str | None, str | None, dict[str, object]]:
    """Resolve a standard-setting Bilbao HSP label.

    First tries direct coordinate match; if that fails, attempts
    standard-setting HSP k-coordinate mapping using crystallographic
    setting provenance from the per-valley standard group match.
    """
    from valleyscope.analysis.standard_setting_kmap import (
        resolve_standard_setting_hsp_label,
    )
    if k_frac is not None:
        label, blocker, prov = resolve_standard_setting_hsp_label(
            k_frac=np.asarray(k_frac, dtype=float),
            table=table,
            standard_match=standard_match,
            lattice_direct_cart=lattice_direct_cart,
            detected_operations=detected_operations,
            parent_to_standard_direct_transform=(
                parent_to_standard_direct_transform
            ),
            origin_shift_fractional=origin_shift_fractional,
            transform_provenance=transform_provenance,
        )
    else:
        label, blocker, prov = None, (
            "no_source_hsp_label: k_frac is None"
        ), {}
    if override_label is not None:
        if label is None:
            # Override cannot bypass an unresolved standard-setting mapping.
            return None, (
                f"generic_irrep_source HSP override {override_label!r} "
                f"cannot be applied: "
                f"standard_setting_hsp_mapping_unresolved; "
                f"no standard-setting HSP label was resolved for this "
                f"k-point - {blocker or 'mapping failed'}"
            ), prov
        if override_label != label:
            return None, (
                f"generic_irrep_source HSP override {override_label!r} "
                f"disagrees with resolved HSP {label!r}"
            ), prov
        # Override confirms the resolved label; pass through.
    if label is None:
        return None, blocker or (
            "no_source_hsp_label: could not determine Bilbao HSP label "
            "for this kpoint"
        ), prov
    return label, None, prov


def analyze_hsp(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    wavefunctions = read_wavefunction_h5(config.input.wavefunction_h5)
    output_dir = config.output.directory
    prepare_analysis_output_directory(config)
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
    coefficients_by_kpoint: dict[str, np.ndarray] = {}
    g_vectors_frac_by_kpoint: dict[str, np.ndarray] = {}
    band_indices_by_kpoint: dict[str, np.ndarray] = {}
    valley_matrices_by_kpoint: dict[str, dict[str, np.ndarray]] = {}
    kpoint_frac_by_name: dict[str, np.ndarray] = {}
    symmetry_payload: dict[str, object] = _prepare_symmetry_payload(config, monolayer_recip)
    symmetry_payload["spinor_wavefunction"] = bool(wavefunctions.metadata.spinor)

    for kpoint_name in config.analysis.kpoints:
        kpoint = wavefunctions.find_kpoint(kpoint_name)
        positions = _target_band_positions(kpoint.band_indices_vasp, config.analysis.iband)
        coefficients = kpoint.coefficients[positions]
        coefficients_by_kpoint[kpoint_name] = coefficients
        g_vectors_frac_by_kpoint[kpoint_name] = np.asarray(
            kpoint.g_vectors_frac
        )
        band_indices_by_kpoint[kpoint_name] = np.asarray(
            kpoint.band_indices_vasp[positions]
        )
        kpoint_frac_by_name[kpoint_name] = np.asarray(kpoint.frac, dtype=float)
        q_cart = kpoint.cart.reshape(1, 3) + kpoint.g_vectors_cart
        # --- Reporting projectors (may use k-dependent centers) ---
        reporting_centers = config.valley_centers
        if config.projection.projector_mode == "k_resolved_parent_valley":
            reporting_centers = adjust_centers_for_parent_valley(
                config.valley_centers,
                kpoint.cart,
                wavefunctions.metadata.lattice.reciprocal_cart,
                use_2d=config.projection.use_2d_momentum_only,
            )
        reporting_projectors = build_sector_projectors(
            q_cart,
            reporting_centers,
            config.valley_subspaces,
            monolayer_recip,
            qcut,
            use_2d=config.projection.use_2d_momentum_only,
            overlap_policy=config.projection.overlap_policy,
            emit_warnings=False,
        )
        projectors_by_kpoint[kpoint_name] = reporting_projectors
        weights = compute_valley_weights(coefficients, reporting_projectors)

        # --- Readiness seed projectors (always fixed_center) ---
        # Seed matrices, projector symmetry-consistency, symmetry-adapted
        # diagnostics, irrep workflow decisions, and EBR pipeline always use
        # fixed-center projectors so that k_resolved_parent_valley is a
        # weight/report-only diagnostic and does not affect readiness gates.
        seed_projectors = reporting_projectors
        if config.projection.projector_mode == "k_resolved_parent_valley":
            seed_projectors = build_sector_projectors(
                q_cart,
                config.valley_centers,
                config.valley_subspaces,
                monolayer_recip,
                qcut,
                use_2d=config.projection.use_2d_momentum_only,
                overlap_policy=config.projection.overlap_policy,
                emit_warnings=False,
            )
        sector_names = reporting_projectors.sector_names
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
            "warnings": reporting_projectors.warnings,
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
            seed_projectors,
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
                reporting_centers,
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
                "sector_names": np.asarray(reporting_projectors.sector_names, dtype="S"),
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
        valley_names = list(reporting_projectors.sector_names)
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
    target_subspace_closure_report: dict[str, object] | None = None
    target_subspace_closure_blockers: list[str] = []
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
        # Build target-subspace symmetry-closure diagnostic (independent of projector symmetry)
        effective_orders: dict[object, int] = {}
        for op in symmetry_payload.get("detected_operations", []):
            op_id = op.get("operation_id")
            order = op.get("order")
            if op_id is not None and order is not None:
                effective_orders[op_id] = int(order)
        target_subspace_closure_report = build_target_subspace_closure_report(
            raw_representations_by_kpoint=raw_representations_by_kpoint,
            operation_orders=effective_orders,
            spinor_wavefunction=bool(symmetry_payload.get("spinor_wavefunction", False)),
            unitarity_tol=float(config.symmetry_adapted_valley.representation_unitarity_fail_tol),
            coefficients_by_kpoint=_coefficients_lookup(coefficients_by_kpoint, raw_representations_by_kpoint),
        )
        target_subspace_closure_blockers = check_target_subspace_closure_blocked(
            target_subspace_closure_report,
        )

    if symmetry_payload["status"] == "ok":
        symmetry_payload["hsp_star_report"] = build_hsp_star_report(
            kpoint_frac_by_name=symmetry_payload.get("kpoint_frac_by_name", {}),
            operations=symmetry_payload.get("detected_operations", []),
        )

    # --- Valley-preserving subgroup report (before symmetry-adapted) ---
    # Build before the symmetry-adapted valley report so that
    # per_valley_standard_matches is available for subspace_space_group
    # resolution during initial construction.
    if symmetry_payload["status"] == "ok":
        build_valley_preserving_subgroup_report(
            symmetry_payload=symmetry_payload,
            target_kpoints=config.analysis.kpoints,
        )
        # The canonical downstream Bilbao/irreptables restricted-character
        # matcher is build_valley_irrep_matching_report().

    # --- Formal symmetry-adapted valley report ---
    symmetry_adapted_valley_report: dict[str, object] | None = None
    hsp_star_conjugation_report: dict[str, object] | None = None
    hsp_star_derived_characters: dict[str, object] | None = None
    if config.symmetry_adapted_valley.enabled:
        symmetry_adapted_valley_report = _build_symmetry_adapted_valley_report(
            valley_matrices_by_kpoint=valley_matrices_by_kpoint,
            raw_representations_by_kpoint=raw_representations_by_kpoint,
            symmetry_payload=symmetry_payload,
            config=config,
            target_subspace_closure_report=target_subspace_closure_report,
            target_subspace_closure_blockers=target_subspace_closure_blockers,
        )
        # Build HSP-star conjugation and derived characters
        hsp_star_conjugation_report, hsp_star_derived_characters = (
            _build_hsp_star_derived_character_layer(
                symmetry_payload=symmetry_payload,
                symmetry_adapted_valley_report=symmetry_adapted_valley_report,
                target_subspace_closure_report=target_subspace_closure_report,
                valley_names=valley_names,
            )
        )
        # Apply derived character gate as post-processing
        if hsp_star_derived_characters is not None:
            _apply_derived_characters_to_report(
                symmetry_adapted_valley_report=symmetry_adapted_valley_report,
                hsp_star_derived_characters=hsp_star_derived_characters,
                kpoint_frac_by_name=symmetry_payload.get("kpoint_frac_by_name", {}),
                operations=symmetry_payload.get("detected_operations", []),
            )

    sector_names = list(projectors_by_kpoint[next(iter(projectors_by_kpoint))].sector_masks)
    symmetry_eigenvalue_summary = _build_symmetry_eigenvalue_summary(symmetry_payload, symmetry_rows)

    # --- Irrep workflow decision layer ---
    irrep_workflow_decisions: dict[str, object] | None = None
    valley_irrep_matching: dict[str, object] | None = None
    ebr_input_candidates: dict[str, object] | None = None
    ebr_problem_instances: dict[str, object] | None = None
    ebr_export_bundle: dict[str, object] | None = None
    reduced_ebr_mapping: dict[str, object] | None = None
    projected_hsp_coverage: dict[str, object] | None = None
    time_reversal_orbit_report: dict[str, object] | None = None
    if config.symmetry_adapted_valley.enabled:
        irrep_workflow_decisions = build_irrep_workflow_decisions(
            projector_symmetry_report=projector_symmetry_report,
            target_subspace_closure_report=target_subspace_closure_report,
            symmetry_adapted_valley_report=symmetry_adapted_valley_report,
            symmetry_rows=symmetry_rows,
            valley_names=valley_names,
            spinor_convention_verified=config.spinor.convention_verified,
            spinor_wavefunction=bool(symmetry_payload.get("spinor_wavefunction", False)),
        )
    # --- Canonical per-valley irrep matching preflight ---
    # Builds one source-payload context per (kpoint, valley) from the
    # computed per-valley standard subgroup identity.  The explicit
    # generic_irrep_source config block is an optional override; when
    # present it must agree with the computed subgroup or the result is
    # diagnostic-only with explicit provenance.
    generic_source_payloads: dict[str, Any] | None = None
    generic_source_blocked_rows: list[dict[str, Any]] = []
    generic_source_classification_rows: list[dict[str, Any]] = []
    projected_hsp_classifications: list[dict[str, Any]] = []
    source_hsp_basis_by_valley: dict[str, dict[str, object]] = {}
    source_table_by_valley: dict[str, object] = {}
    source_ebr_data_by_valley: dict[str, dict[str, object]] = {}
    source_certificate_by_valley: dict[str, dict[str, object]] = {}
    ebr_source_basis_cache: dict[tuple[int, bool], dict[str, Any]] = {}

    # --- Canonical subgroup identity from per-valley standard matches ---
    subgroup_report = symmetry_payload.get(
        "valley_preserving_subgroup_report", {},
    ) if isinstance(symmetry_payload, dict) else {}
    per_valley_matches = subgroup_report.get(
        "per_valley_standard_matches", {},
    ) if isinstance(subgroup_report, dict) else {}
    # Table spinfulness follows the wavefunction, not convention_verified.
    # convention_verified is a readiness gate, not a table selection criterion.
    spinor_wf = bool(symmetry_payload.get("spinor_wavefunction", False))
    kpoint_frac = symmetry_payload.get("kpoint_frac_by_name", {})

    # --- Explicit override validation ---
    override_sg: int | None = None
    override_spinor: bool | None = None
    override_hsp: dict[str, dict[str, str]] | None = None
    if config.generic_irrep_source.enabled:
        gis_cfg = config.generic_irrep_source
        override_sg = gis_cfg.spacegroup_number
        override_spinor = gis_cfg.spinor
        override_hsp = (
            dict(gis_cfg.source_hsp_labels)
            if gis_cfg.source_hsp_labels else None
        )

    src_chars: dict[str, dict[str, dict[str, dict[int, complex]]]] = {}
    src_op_maps: dict[str, dict[str, dict[int, int]]] = {}
    src_provenance: dict[str, dict[str, dict[str, Any]]] = {}
    by_kp = symmetry_adapted_valley_report.get("by_kpoint", {}) if isinstance(symmetry_adapted_valley_report, dict) else {}

    if isinstance(by_kp, dict):
        for kp_name, kp_data in by_kp.items():
            if not isinstance(kp_data, dict):
                continue
            vp_subspaces = kp_data.get("valley_preserving_subspaces", [])
            if not isinstance(vp_subspaces, list):
                continue
            for vs in vp_subspaces:
                if not isinstance(vs, dict):
                    continue
                orbit = vs.get("orbit", [])
                if not orbit:
                    continue
                v_name = str(orbit[0])

                # --- Per-valley canonical subgroup identity ---
                sg_number: int
                spinor_flag: bool
                # Always read the computed subgroup for agreement checking.
                match_info = per_valley_matches.get(v_name, {})
                standard_match = match_info.get("standard_group_match")
                computed_sg: int | None = None
                if isinstance(standard_match, dict):
                    num = standard_match.get("number")
                    if isinstance(num, int) and not isinstance(num, bool) and num > 0:
                        computed_sg = int(num)

                if override_sg is not None:
                    override_blocker = _generic_irrep_override_blocker(
                        computed_sg=computed_sg,
                        wavefunction_spinor=spinor_wf,
                        override_sg=override_sg,
                        override_spinor=override_spinor,
                    )
                    if override_blocker is not None:
                        generic_source_blocked_rows.append({
                            "kpoint": kp_name, "valley": v_name,
                            "reason": f"{override_blocker}; diagnostic-only",
                        })
                        continue
                    sg_number = override_sg
                    spinor_flag = spinor_wf
                elif computed_sg is not None:
                    sg_number = computed_sg
                    spinor_flag = spinor_wf
                else:
                    generic_source_blocked_rows.append({
                        "kpoint": kp_name, "valley": v_name,
                        "reason": "no per-valley standard subgroup match",
                    })
                    continue

                # --- Load source irrep table ---
                try:
                    table = load_standard_irrep_table(sg_number, spinor=spinor_flag)
                except Exception as exc:
                    generic_source_blocked_rows.append({
                        "kpoint": kp_name, "valley": v_name,
                        "reason": f"load_standard_irrep_table sg={sg_number} "
                                  f"spinor={spinor_flag}: {exc}",
                    })
                    continue

                # --- Standard-setting HSP correspondence (conservative) ---
                # --- G_k^(a) operation IDs (before HSP labelling) ---
                ssg = vs.get("subspace_space_group", {})
                full_vp_ids = ssg.get("valley_preserving_operation_ids", []) if isinstance(ssg, dict) else []
                hsp_lg_ids = vs.get("hsp_preserving_operation_ids", [])
                if isinstance(hsp_lg_ids, list) and hsp_lg_ids:
                    vp_ids = [op for op in full_vp_ids if op in hsp_lg_ids]
                else:
                    vp_ids = list(full_vp_ids)

                if not vp_ids:
                    # Empty G_k^(a): inconsistent input, blocked.
                    generic_source_blocked_rows.append({
                        "kpoint": kp_name, "valley": v_name,
                        "reason": (
                            "empty_valley_preserving_hsp_subgroup: "
                            "no valley-preserving operation in the HSP "
                            "little group"
                        ),
                    })
                    continue
                k_frac_raw = kpoint_frac.get(kp_name)
                override_label = (
                    override_hsp.get(kp_name, {}).get(v_name)
                    if override_hsp is not None
                    else None
                )
                ss_cfg = config.standard_setting
                _src_hsp, hsp_blocker, hsp_provenance = (
                    _resolve_generic_irrep_hsp_label_with_provenance(
                        table=table,
                        k_frac=(
                            np.asarray(k_frac_raw, dtype=float)
                            if k_frac_raw is not None else None
                        ),
                        override_label=override_label,
                        standard_match=(
                            standard_match
                            if isinstance(standard_match, dict)
                            else None
                        ),
                        lattice_direct_cart=(
                            np.asarray(symmetry_payload.get("lattice_direct_cart"), dtype=float)
                            if symmetry_payload.get("lattice_direct_cart") is not None
                            else None
                        ),
                        detected_operations=(
                            symmetry_payload.get("detected_operations")
                            if isinstance(symmetry_payload.get("detected_operations"), list)
                            else None
                        ),
                        parent_to_standard_direct_transform=(
                            np.asarray(ss_cfg.parent_to_standard_direct_transform, dtype=float)
                            if ss_cfg.parent_to_standard_direct_transform is not None
                            else None
                        ),
                        origin_shift_fractional=(
                            np.asarray(ss_cfg.origin_shift_fractional, dtype=float)
                            if ss_cfg.origin_shift_fractional is not None
                            else None
                        ),
                        transform_provenance=ss_cfg.transform_provenance,
                    )
                )
                # --- Projected 2D source basis and sampled-k classification ---
                # The legacy resolver is retained as the certificate producer.
                # A direct-label miss is not itself a blocker: the new layer
                # still checks star membership before classifying the point as
                # generic.
                ssg_context = _resolved_subspace_group_context(
                    standard_match=(
                        standard_match
                        if isinstance(standard_match, dict)
                        else {}
                    ),
                    local_gka_operation_ids=list(vp_ids),
                )
                certificate = (
                    hsp_provenance.get("standard_setting_certificate", {})
                    if isinstance(hsp_provenance, dict) else {}
                )
                if not isinstance(certificate, dict):
                    certificate = {}

                source_key = (sg_number, spinor_flag)
                if source_key not in ebr_source_basis_cache:
                    try:
                        ebr_source_basis_cache[source_key] = (
                            load_ebr_source_data(sg_number, spinor_flag)
                        )
                    except Exception as exc:
                        generic_source_blocked_rows.append({
                            "kpoint": kp_name,
                            "valley": v_name,
                            "reason": (
                                "irreptables_ebr_source_basis_unavailable: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            "subspace_space_group": ssg_context,
                            "valley_preserving_operation_ids": list(vp_ids),
                            "hsp_little_group_operation_ids": list(vp_ids),
                            "standard_setting_hsp_mapping": dict(
                                hsp_provenance
                            ) if isinstance(hsp_provenance, dict) else {},
                        })
                        continue
                ebr_source_data = ebr_source_basis_cache[source_key]
                source_basis = derive_projected_subspace_source_hsp_basis(
                    table=table,
                    ebr_source_basis_labels=ebr_source_data.get(
                        "source_basis_labels", []
                    ),
                    standard_setting_certificate=certificate,
                    use_2d_momentum_only=(
                        config.projection.use_2d_momentum_only
                    ),
                )
                source_basis_provenance = source_basis.get("provenance", {})
                if isinstance(source_basis_provenance, dict):
                    source_basis_provenance.update({
                        "ebr_data_source": ebr_source_data.get("data_source"),
                        "ebr_source_basis_count": ebr_source_data.get(
                            "source_basis_count"
                        ),
                    })
                existing_basis = source_hsp_basis_by_valley.get(v_name)
                if existing_basis is None:
                    source_hsp_basis_by_valley[v_name] = source_basis
                    source_table_by_valley[v_name] = table
                    source_ebr_data_by_valley[v_name] = ebr_source_data
                    source_certificate_by_valley[v_name] = certificate
                elif (
                    existing_basis.get("required_source_hsp_labels")
                    != source_basis.get("required_source_hsp_labels")
                    or existing_basis.get("standard_plane_basis")
                    != source_basis.get("standard_plane_basis")
                    or existing_basis.get(
                        "projected_subspace_space_group"
                    ) != source_basis.get(
                        "projected_subspace_space_group"
                    )
                ):
                    generic_source_blocked_rows.append({
                        "kpoint": kp_name,
                        "valley": v_name,
                        "reason": (
                            "per_valley_source_hsp_basis_inconsistent_across_"
                            "sampled_kpoints"
                        ),
                        "subspace_space_group": ssg_context,
                        "valley_preserving_operation_ids": list(vp_ids),
                        "hsp_little_group_operation_ids": list(vp_ids),
                    })
                    continue

                classification = classify_projected_subspace_kpoint(
                    parent_k_frac=k_frac_raw,
                    table=table,
                    source_hsp_basis=source_basis,
                    standard_setting_certificate=certificate,
                    override_source_hsp_label=override_label,
                    kpoint=kp_name,
                    valley=v_name,
                )

                if classification.get("classification") == "generic":
                    projected_hsp_classifications.append(classification)
                    generic_source_classification_rows.append({
                        "kpoint": kp_name,
                        "valley": v_name,
                        "subspace_space_group": ssg_context,
                        "valley_preserving_operation_ids": list(vp_ids),
                        "hsp_little_group_operation_ids": list(vp_ids),
                        "projected_hsp_classification": classification,
                    })
                    continue
                if classification.get("classification") == "unresolved":
                    projected_hsp_classifications.append(classification)
                    generic_source_blocked_rows.append({
                        "kpoint": kp_name,
                        "valley": v_name,
                        "reason": classification.get("blocker")
                        or hsp_blocker
                        or "projected-subspace HSP classification unresolved",
                        "subspace_space_group": ssg_context,
                        "valley_preserving_operation_ids": list(vp_ids),
                        "hsp_little_group_operation_ids": list(vp_ids),
                        "projected_hsp_classification": classification,
                        "standard_setting_hsp_mapping": dict(
                            hsp_provenance
                        ) if isinstance(hsp_provenance, dict) else {},
                    })
                    continue

                # --- Build representative/star-aware source payload ---
                tol = float(
                    config.generic_irrep_source.operation_match_tol
                    if config.generic_irrep_source.enabled else 5e-5
                )
                payload = build_source_payload_for_projected_hsp_matching(
                    table=table,
                    projected_hsp_classification=classification,
                    detected_operations=symmetry_payload.get("detected_operations", []),
                    valley_preserving_operation_ids=list(vp_ids),
                    source_hsp_basis=source_basis,
                    tol=tol,
                )
                if payload.get("operation_mapping_evaluated") is True:
                    classification = classify_projected_subspace_kpoint(
                        parent_k_frac=k_frac_raw,
                        table=table,
                        source_hsp_basis=source_basis,
                        standard_setting_certificate=certificate,
                        mapped_standard_little_group_operation_ids=list(
                            payload["source_operation_map"].values()
                        ),
                        override_source_hsp_label=override_label,
                        kpoint=kp_name,
                        valley=v_name,
                    )
                projected_hsp_classifications.append(classification)
                payload_provenance = dict(
                    payload.get("provenance", {})
                    if isinstance(payload.get("provenance", {}), dict)
                    else {}
                )
                if isinstance(hsp_provenance, dict) and hsp_provenance:
                    payload_provenance["standard_setting_hsp_mapping"] = (
                        dict(hsp_provenance)
                    )
                payload_provenance["projected_hsp_classification"] = dict(
                    classification
                )
                if (
                    payload["status"] == "ok"
                    and classification.get("classification")
                    in ("representative", "star_equivalent")
                    and classification.get("validation_status") == "validated"
                    and classification.get("representation_transport_status")
                    == "validated"
                ):
                    src_chars.setdefault(kp_name, {})[v_name] = (
                        payload["source_irrep_characters"]
                    )
                    src_op_maps.setdefault(kp_name, {})[v_name] = (
                        payload["source_operation_map"]
                    )
                    src_provenance.setdefault(kp_name, {})[v_name] = (
                        payload_provenance
                    )
                else:
                    generic_source_blocked_rows.append({
                        "kpoint": kp_name,
                        "valley": v_name,
                        "source_hsp_label": classification.get(
                            "source_hsp_label"
                        ),
                        "table_sg_number": table.number,
                        "table_name": table.name,
                        "table_spinor": table.spinor,
                        "valley_preserving_operation_ids": list(vp_ids),
                        "hsp_little_group_operation_ids": (
                            list(hsp_lg_ids) if isinstance(hsp_lg_ids, list)
                            else list(vp_ids)
                        ),
                        "provenance": payload_provenance,
                        "projected_hsp_classification": classification,
                        "blocker_reasons": (
                            payload["blocker_reasons"]
                            if payload["status"] != "ok"
                            else [classification.get("blocker")
                                  or classification.get("matching_blocker")
                                  or "source-HSP validation blocked"]
                        ),
                    })

    # --- Build resolved canonical subgroup contexts ---
    resolved_subspace_groups: dict[str, dict[str, dict[str, object]]] = {}
    for v_name, match_info in per_valley_matches.items():
        if not isinstance(match_info, dict):
            continue
        standard_match = match_info.get("standard_group_match")
        if not isinstance(standard_match, dict):
            continue
        num = standard_match.get("number")
        symbol = standard_match.get("international_short")
        op_ids = standard_match.get("operation_ids", [])
        if not (isinstance(num, int) and not isinstance(num, bool) and num > 0):
            continue
        resolved_ssg: dict[str, object] = {
            "status": "resolved",
            "candidate_space_group_number": int(num),
            "candidate_space_group_symbol": (
                str(symbol) if isinstance(symbol, str) and symbol else ""
            ),
            "valley_preserving_operation_ids": (
                list(op_ids) if isinstance(op_ids, list) else []
            ),
            "source": (
                "symmetry_analysis.valley_preserving_subgroup_report"
                ".per_valley_standard_matches"
            ),
        }
        # Resolved subgroup applies to all kpoints for this valley.
        for kp_name in (by_kp if isinstance(by_kp, dict) else {}):
            resolved_subspace_groups.setdefault(kp_name, {})[v_name] = resolved_ssg

    if src_chars and src_op_maps:
        generic_source_payloads = {
            "source_irrep_characters": src_chars,
            "source_operation_maps": src_op_maps,
            "source_payload_provenance": src_provenance,
        }

    valley_irrep_matching = build_valley_irrep_matching_report(
        irrep_workflow_decisions=irrep_workflow_decisions,
        symmetry_adapted_valley_report=symmetry_adapted_valley_report,
        source_irrep_characters_flattened=(
            generic_source_payloads.get("source_irrep_characters", {})
            if generic_source_payloads else None
        ),
        source_operation_maps=(
            generic_source_payloads.get("source_operation_maps", {})
            if generic_source_payloads else None
        ),
        source_payload_provenance=(
            generic_source_payloads.get("source_payload_provenance", {})
            if generic_source_payloads else None
        ),
        source_payload_blocked_rows=generic_source_blocked_rows,
        source_payload_classification_rows=generic_source_classification_rows,
        resolved_subspace_groups=(
            resolved_subspace_groups if resolved_subspace_groups else None
        ),
    )


    projected_hsp_coverage = build_projected_hsp_coverage_report(
        source_hsp_basis_by_valley=source_hsp_basis_by_valley,
        classifications=projected_hsp_classifications,
        matching_by_kpoint=(
            valley_irrep_matching.get("generic_matches_by_kpoint", {})
            if isinstance(valley_irrep_matching, dict) else {}
        ),
        workflow_decisions_by_kpoint=(
            irrep_workflow_decisions.get("by_kpoint", {})
            if isinstance(irrep_workflow_decisions, dict) else {}
        ),
    )

    ebr_input_candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=irrep_workflow_decisions,
        valley_irrep_matching=valley_irrep_matching,
        symmetry_adapted_valley_report=symmetry_adapted_valley_report,
    )
    valley_mapping_report = derive_time_reversal_valley_mapping(
        enabled=config.time_reversal.enabled,
        centers=config.valley_centers,
        valley_subspaces=config.valley_subspaces,
        spinor=spinor_wf,
    )
    if config.time_reversal.enabled:
        antiunitary_sewing_report = build_time_reversal_sewing_report(
            kpoint_frac_by_name=kpoint_frac_by_name,
            g_vectors_frac_by_kpoint=g_vectors_frac_by_kpoint,
            coefficients_by_kpoint=coefficients_by_kpoint,
            band_indices_by_kpoint=band_indices_by_kpoint,
            valley_projectors_by_kpoint=valley_matrices_by_kpoint,
            time_reversal_valley_mapping=valley_mapping_report.get(
                "time_reversal_valley_mapping", {}
            ),
            spinor=spinor_wf,
            spinor_convention_verified=config.spinor.convention_verified,
        )
        source_orbits_by_valley: dict[str, dict[str, object]] = {}
        grey_source_by_valley: dict[str, dict[str, object]] = {}
        for valley, basis in source_hsp_basis_by_valley.items():
            reviewed_by_label = basis.get(
                "_reviewed_source_irreps_by_label", {}
            )
            if not isinstance(reviewed_by_label, dict):
                continue
            reviewed_rows = list(reviewed_by_label.values())
            certificate = source_certificate_by_valley.get(valley, {})
            centering_vectors = certificate.get(
                "normalized_centering_vectors",
                certificate.get("centering_vectors", []),
            )
            if not isinstance(centering_vectors, list):
                centering_vectors = []
            full_source_orbits = derive_time_reversal_source_irrep_orbits(
                reviewed_rows=reviewed_rows,
                centering_vectors=centering_vectors,
            )
            required_hsps = set(
                basis.get("required_source_hsp_labels", [])
            )
            in_plane_rows = [
                row for row in reviewed_rows
                if getattr(row, "kpoint_label", None) in required_hsps
            ]
            source_orbits_by_valley[valley] = (
                derive_time_reversal_source_irrep_orbits(
                    reviewed_rows=in_plane_rows,
                    centering_vectors=centering_vectors,
                )
            )
            table = source_table_by_valley.get(valley)
            source_data = source_ebr_data_by_valley.get(valley)
            if table is not None and isinstance(source_data, dict):
                grey_source_by_valley[valley] = (
                    validate_grey_group_time_reversal_source(
                        unitary_table=table,
                        reviewed_rows=reviewed_rows,
                        unitary_source_data=source_data,
                        irrep_partner_by_label=full_source_orbits.get(
                            "irrep_partner_by_label", {}
                        ),
                        centering_vectors=centering_vectors,
                    )
                )
        time_reversal_orbit_report = (
            build_time_reversal_valley_orbit_report(
                valley_mapping_report=valley_mapping_report,
                source_irrep_orbits_by_valley=source_orbits_by_valley,
                grey_source_by_valley=grey_source_by_valley,
                ebr_input_candidates=ebr_input_candidates,
                antiunitary_sewing_report=antiunitary_sewing_report,
            )
        )
        projected_hsp_coverage["time_reversal"] = (
            time_reversal_orbit_report
        )
    ebr_problem_instances = build_ebr_problem_instances(
        ebr_input_candidates=ebr_input_candidates,
        projected_hsp_coverage=projected_hsp_coverage,
        time_reversal_orbit_report=time_reversal_orbit_report,
    )
    ebr_export_bundle = build_ebr_export_bundle(
        ebr_problem_instances=ebr_problem_instances,
    )
    if config.reduced_ebr.enabled:
        table = None
        reduced_ebr_input: dict[str, object] = {"source": "not_provided"}
        reduced_ebr_mapping = None
        if config.reduced_ebr.table_file:
            table = load_reduced_ebr_table(config.reduced_ebr.table_file)
            reduced_ebr_input = {
                "source": "table_file",
                "table_file_stem": config.reduced_ebr.table_file.stem,
            }
        elif config.reduced_ebr.spec_file:
            from valleyscope.analysis.irreptables_runtime_table_builder import (
                build_reduced_table_from_spec_file,
            )
            table = build_reduced_table_from_spec_file(
                str(config.reduced_ebr.spec_file),
            )
            provenance = table.get("provenance", {}) if isinstance(table.get("provenance"), dict) else {}
            reduced_ebr_input = {
                "source": "spec_file",
                "spec_file_stem": config.reduced_ebr.spec_file.stem,
                "subspace_group_candidate": table.get("subspace_group_candidate", ""),
                "data_source": provenance.get("data_source", ""),
            }
        else:
            # --- Auto-canonical path: per-group auto tables + merged result ---
            reduced_ebr_mapping = _build_auto_canonical_mapping(
                ebr_export_bundle=ebr_export_bundle,
                spinor_wf=spinor_wf,
                max_coefficient=config.reduced_ebr.max_coefficient,
            )
        if reduced_ebr_mapping is None:
            reduced_ebr_mapping = build_reduced_ebr_mapping(
                ebr_export_bundle=ebr_export_bundle,
                table=table,
                max_coefficient=config.reduced_ebr.max_coefficient,
                reduced_ebr_input=reduced_ebr_input,
            )


# --- Valley-projected representation report ---
    valley_projected_representation = build_valley_projected_representation_report(
        kpoint_names=config.analysis.kpoints,
        valley_names=sector_names,
        symmetry_eigenvalue_rows=symmetry_rows if isinstance(symmetry_rows, list) else None,
        symmetry_adapted_valley_report=symmetry_adapted_valley_report,
        irrep_workflow_decisions=irrep_workflow_decisions,
        valley_irrep_matching=valley_irrep_matching,
        symmetry_analysis=symmetry_payload,
    )

    # --- Folded-center report ---
    folded_center_report = build_folded_center_report(
        centers=config.valley_centers,
        moire_reciprocal_cart=wavefunctions.metadata.lattice.reciprocal_cart,
        sampled_k_frac=kpoint_frac_by_name,
        use_2d=config.projection.use_2d_momentum_only,
    )
    folded_center_payload = folded_center_report_to_dict(
        folded_center_report,
        kpoint_names=config.analysis.kpoints,
    )
    # --- Sampled-k coverage diagnostic ---
    sampled_k_coverage = _build_sampled_k_coverage(
        folded_center_report=folded_center_report,
        kpoint_names=config.analysis.kpoints,
        kpoint_frac_by_name=kpoint_frac_by_name,
    )
    if projected_hsp_coverage is not None:
        sampled_k_coverage["projected_subspace_hsp_coverage"] = (
            projected_hsp_coverage
        )
    # --- Warn when fixed_center projector has large k-center mismatch ---
    if config.projection.projector_mode == "fixed_center":
        _warn_fixed_center_distance(
            folded_center_report=folded_center_report,
            kpoint_names=config.analysis.kpoints,
            weight_rows=rows,
            subspace_payload=subspace_payload,
        )

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
        target_subspace_closure_report=target_subspace_closure_report,
        hsp_star_conjugation_report=hsp_star_conjugation_report,
        hsp_star_derived_characters=hsp_star_derived_characters,
        irrep_workflow_decisions=irrep_workflow_decisions,
        valley_irrep_matching=valley_irrep_matching,
        ebr_input_candidates=ebr_input_candidates,
        ebr_problem_instances=ebr_problem_instances,
        ebr_export_bundle=ebr_export_bundle,
        reduced_ebr_mapping=reduced_ebr_mapping,
        valley_projected_representation=valley_projected_representation,
        folded_center_payload=folded_center_payload,
        sampled_k_coverage=sampled_k_coverage,
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
        "center_weights": result.center_weights,
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
    target_subspace_closure_report: dict[str, object] | None = None,
    target_subspace_closure_blockers: list[str] | None = None,
) -> dict[str, object]:
    """Build per-kpoint symmetry-adapted valley analysis.

    Returns a ``by_kpoint`` dict keyed by kpoint label. Each entry is a
    ``not_evaluated`` stub or contains orbit-level reports when inputs are
    sufficient for the toy pipeline.
    """
    closure_blockers = list(target_subspace_closure_blockers or [])
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
    space_group_valley_orbits = _partition_valley_orbits(
        valley_names=valley_names,
        valley_mappings=space_group_valley_mappings,
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

            orbit_inferred_rank, orbit_rank_source = _infer_orbit_rank(
                seed_projectors=seed_projectors,
                orbit=orbit,
            )
            report = build_symmetry_adapted_valley_report(
                seed_projectors=seed_projectors,
                representations=d_g_dict,
                valley_mappings=valley_mappings_dict,
                orbit=orbit,
                reference_valley=orbit[0],
                rank=orbit_inferred_rank,
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
            kpoint_name=kpoint_name,
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
            target_subspace_closure_blockers=closure_blockers,
            target_subspace_closure_report=target_subspace_closure_report,
            per_valley_standard_matches=_extract_per_valley_matches(symmetry_payload),
        )

        by_kpoint[kpoint_name] = _aggregate_symmetry_adapted_kpoint(
            orbit_reports,
            valley_preserving_subspaces=valley_preserving_subspaces,
        )

    return {
        "space_group_valley_orbits": space_group_valley_orbits,
        "by_kpoint": by_kpoint,
    }


def _build_valley_preserving_subspace_reports(
    *,
    kpoint_name: str = "",
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
    target_subspace_closure_blockers: list[str] | None = None,
    target_subspace_closure_report: dict[str, object] | None = None,
    per_valley_standard_matches: dict[str, Any] | None = None,
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
        # Build subspace representation quality diagnostics from raw matrices
        # before they are stripped by summarization.
        raw_eigenvectors = report.get("_internal_raw_eigenvectors", {})
        raw_projectors = report.get("_internal_raw_projectors", {})
        if not isinstance(raw_eigenvectors, dict):
            raw_eigenvectors = {}
        if not isinstance(raw_projectors, dict):
            raw_projectors = {}
        quality_report = build_subspace_representation_quality_report(
            valley_bases=raw_eigenvectors,
            projectors=raw_projectors,
            representations=local_representations,
            valley_mappings=local_mappings,
            operation_orders=operation_orders_by_id,
            spinor_wavefunction=spinor_wavefunction,
            target_valleys=[str(valley)],
        )

        summary = summarize_symmetry_adapted_valley_report(report)
        summary["subspace_representation_quality"] = quality_report
        summary["analysis_scope"] = "valley_preserving_subspace"
        summary["local_rank_source"] = rank_source
        summary["hsp_preserving_operation_ids"] = list(preserving_ops)
        summary["subspace_space_group"] = _build_subspace_space_group_for_valley(
            valley=str(valley),
            valley_mappings=space_group_valley_mappings or valley_mappings_dict,
            operation_orders=space_group_operation_orders or operation_orders_by_id or {},
            per_valley_standard_matches=per_valley_standard_matches,
        )
        ebr_mapping = summary.get("ebr_mapping_input")
        if isinstance(ebr_mapping, dict):
            _refine_ebr_mapping_with_subspace_space_group(
                ebr_mapping=ebr_mapping,
                subspace_space_group=summary["subspace_space_group"],
                local_gka_operation_ids=summary.get(
                    "hsp_preserving_operation_ids", []
                ),
            )
            _apply_target_subspace_closure_gate(
                ebr_mapping=ebr_mapping,
                closure_blockers=target_subspace_closure_blockers,
                target_subspace_closure_report=target_subspace_closure_report,
                kpoint_name=kpoint_name,
                valley_preserving_ops=summary.get("hsp_preserving_operation_ids", []),
            )
        reports.append(summary)
    return reports


def _apply_derived_characters_to_report(
    *,
    symmetry_adapted_valley_report: dict[str, object],
    hsp_star_derived_characters: dict[str, object],
    kpoint_frac_by_name: dict[str, list[float]],
    operations: list[dict[str, object]] | None = None,
) -> None:
    """Post-process: apply HSP-star derived characters to EBR readiness.

    For each subspace, computes the target HSP-star key for each
    full-space-group preserving operation, then checks whether a trusted
    derived character exists for exactly (target_kpoint_key, valley, op).
    """
    by_kpoint = symmetry_adapted_valley_report.get("by_kpoint", {})
    if not isinstance(by_kpoint, dict):
        return
    for kpoint_name, kpoint_data in by_kpoint.items():
        if not isinstance(kpoint_data, dict):
            continue
        k_frac = kpoint_frac_by_name.get(kpoint_name)
        for subspace in kpoint_data.get("valley_preserving_subspaces", []):
            if not isinstance(subspace, dict):
                continue
            ebr_mapping = subspace.get("ebr_mapping_input")
            if not isinstance(ebr_mapping, dict):
                continue
            valley = str(subspace.get("orbit", [""])[0] if subspace.get("orbit") else "")
            subspace_sg = subspace.get("subspace_space_group", {})
            if not isinstance(subspace_sg, dict):
                subspace_sg = {}
            full_preserving_ops = subspace_sg.get("valley_preserving_operation_ids", [])
            _apply_hsp_star_derived_character_gate(
                ebr_mapping=ebr_mapping,
                valley=valley,
                hsp_star_derived_characters=hsp_star_derived_characters,
                full_preserving_ops=full_preserving_ops,
                kpoint_name=kpoint_name,
                k_frac=k_frac,
                kpoint_frac_by_name=kpoint_frac_by_name,
                operations=operations,
            )


def _build_hsp_star_derived_character_layer(
    *,
    symmetry_payload: dict[str, object],
    symmetry_adapted_valley_report: dict[str, object] | None,
    target_subspace_closure_report: dict[str, object] | None,
    valley_names: list[str],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Build HSP-star conjugation and derived character reports.

    Returns (conjugation_report, derived_characters) or (None, None).
    """
    if symmetry_adapted_valley_report is None:
        return None, None
    kpoint_frac_by_name = symmetry_payload.get("kpoint_frac_by_name", {})
    operations = symmetry_payload.get("detected_operations", [])
    if not kpoint_frac_by_name or not operations or not valley_names:
        return None, None

    conjugation_report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name=kpoint_frac_by_name,
        operations=operations,
        valley_names=valley_names,
    )

    # Merge ALL singleton subspace character diagnostics for each kpoint,
    # preserving per-valley readiness so one valley's diagnostic_only
    # status does not pollute another valley at the same kpoint.
    source_char_diagnostics: dict[str, dict[str, object]] = {}
    by_kpoint = symmetry_adapted_valley_report.get("by_kpoint", {})
    if isinstance(by_kpoint, dict):
        for kpoint_name, kp_data in by_kpoint.items():
            if not isinstance(kp_data, dict):
                continue
            merged_per_valley: dict[str, list[dict[str, object]]] = {}
            per_valley_status: dict[str, str] = {}
            per_valley_diag_only: dict[str, bool] = {}
            per_valley_ready: dict[str, bool] = {}
            for subspace in kp_data.get("valley_preserving_subspaces", []):
                if not isinstance(subspace, dict):
                    continue
                char_diag = subspace.get("valley_preserving_character_diagnostics")
                if not isinstance(char_diag, dict):
                    continue
                subspace_valley = str(
                    subspace.get("orbit", [""])[0] if subspace.get("orbit") else ""
                )
                pv = char_diag.get("per_valley", {})
                if not isinstance(pv, dict):
                    continue
                for valley, items in pv.items():
                    if not isinstance(items, list):
                        continue
                    merged_per_valley.setdefault(str(valley), []).extend(items)
                # Per-valley trust: only set for the valley this subspace
                # actually represents.
                if subspace_valley:
                    per_valley_status[subspace_valley] = str(
                        char_diag.get("status", "ok")
                    )
                    per_valley_diag_only[subspace_valley] = bool(
                        char_diag.get("diagnostic_only", True)
                    )
                    per_valley_ready[subspace_valley] = bool(
                        char_diag.get("local_irrep_ready", False)
                    )
            if merged_per_valley:
                source_char_diagnostics[kpoint_name] = {
                    "status": "ok",
                    "local_irrep_ready": True,
                    "diagnostic_only": False,
                    "per_valley": merged_per_valley,
                    "per_valley_status": per_valley_status,
                    "per_valley_diagnostic_only": per_valley_diag_only,
                    "per_valley_ready": per_valley_ready,
                }

    derived_chars = build_hsp_star_derived_characters(
        conjugation_report=conjugation_report,
        source_character_diagnostics=source_char_diagnostics,
        target_subspace_closure_report=target_subspace_closure_report,
    )

    return conjugation_report, derived_chars


def _coefficients_lookup(
    coefficients_by_kpoint: dict[str, np.ndarray],
    raw_representations_by_kpoint: dict[str, dict[object, dict[str, object]]],
) -> dict[str, np.ndarray]:
    """Build coefficient lookup matching raw_representations_by_kpoint keys."""
    result: dict[str, np.ndarray] = {}
    for kp_name in raw_representations_by_kpoint:
        if kp_name in coefficients_by_kpoint:
            result[kp_name] = coefficients_by_kpoint[kp_name]
    return result


def _infer_orbit_rank(
    *,
    seed_projectors: dict[str, np.ndarray],
    orbit: list[str],
) -> tuple[int | None, str]:
    """Infer per-valley rank for a full orbit when target dim is divisible by orbit size.

    Returns (rank, rank_source).  If not divisible or no data, returns (None, "auto_gap").
    """
    if not orbit or not seed_projectors:
        return None, "auto_gap"
    available = [v for v in orbit if v in seed_projectors]
    if not available:
        return None, "auto_gap"
    first = np.asarray(seed_projectors[available[0]])
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        return None, "auto_gap"
    dim = int(first.shape[0])
    n_orbit = len(orbit)
    if n_orbit > 0 and dim % n_orbit == 0:
        rank = dim // n_orbit
        if rank > 0:
            return rank, "target_dim_div_orbit_size"
    return None, "auto_gap"


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


def _extract_per_valley_matches(
    symmetry_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract per_valley_standard_matches from symmetry_payload."""
    if not isinstance(symmetry_payload, dict):
        return None
    report = symmetry_payload.get("valley_preserving_subgroup_report", {})
    if not isinstance(report, dict):
        return None
    return report.get("per_valley_standard_matches", None)


def _build_subspace_space_group_for_valley(
    *,
    valley: str,
    valley_mappings: dict[object, dict[str, str]],
    operation_orders: dict[object, int],
    per_valley_standard_matches: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Build the valley subspace space-group candidate from full operations.

    When ``per_valley_standard_matches`` provides a resolved standard group
    identification (via spglib), propagate it to the subspace space group.
    """
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

    # Check per_valley_standard_matches for a resolved standard group.
    standard_match = None
    if isinstance(per_valley_standard_matches, dict):
        vm = per_valley_standard_matches.get(valley, {})
        if isinstance(vm, dict) and vm.get("standard_group_match_status") == "matched":
            standard_match = vm.get("standard_group_match")

    if standard_match and isinstance(standard_match, dict):
        sg_number = standard_match.get("number")
        sg_symbol = standard_match.get("international_short")
        status = "resolved"
        reason = (
            f"resolved from per_valley_standard_matches via spglib: "
            f"{sg_symbol} (No. {sg_number})"
        )
        source = (
            "symmetry_analysis.valley_preserving_subgroup_report"
            ".per_valley_standard_matches"
        )
    elif effective_order > 1:
        status = "unresolved"
        reason = (
            "subspace-space-group identity is unresolved: no reviewed or generic "
            "identification source is available for the valley-preserving operation set"
        )
        sg_number = None
        sg_symbol = None
        source = "full_space_group_valley_mapping"
    else:
        status = "trivial"
        reason = "only identity preserves this valley in the detected space group"
        sg_number = None
        sg_symbol = None
        source = "full_space_group_valley_mapping"

    return {
        "status": status,
        "candidate_space_group_number": sg_number,
        "candidate_space_group_symbol": sg_symbol,
        "candidate_point_group": None,
        "valley_preserving_operation_ids": preserving_ops,
        "valley_changing_operation_ids": changing_ops,
        "operation_orders": {
            str(op_id): int(operation_orders[op_id])
            for op_id in preserving_ops + changing_ops
            if op_id in operation_orders
        },
        "source": source,
        "reason": reason,
    }


def _apply_hsp_star_derived_character_gate(
    *,
    ebr_mapping: dict[str, object],
    valley: str,
    hsp_star_derived_characters: dict[str, object] | None,
    full_preserving_ops: list[object] | None = None,
    kpoint_name: str = "",
    k_frac: list[float] | None = None,
    kpoint_frac_by_name: dict[str, list[float]] | None = None,
    operations: list[dict[str, object]] | None = None,
) -> None:
    """Apply HSP-star derived character status to EBR readiness.

    For each non-identity full-space-group valley-preserving operation,
    computes the target HSP-star key that this operation maps the current
    kpoint to, then checks for an exact (target_key, valley, op_id) match
    in the trusted derived character set.

    Success is recorded in resolved_by and character_source, not blocked_by.
    """
    if hsp_star_derived_characters is None:
        return
    blockers: list[str] = list(ebr_mapping.get("blocked_by", []) or [])

    relevant_blockers = {"hsp_local_preserving_character_missing",
                         "hsp_star_derivation_not_available"}
    if not any(b in blockers for b in relevant_blockers):
        return

    preserving_ops = list(full_preserving_ops or [])
    kpfbn = dict(kpoint_frac_by_name or {})
    ops = list(operations or [])
    derived_by_target = collect_derived_characters_by_target(hsp_star_derived_characters)

    # Build operation rotation lookup
    op_rotation: dict[object, np.ndarray] = {}
    for op in ops:
        op_id = op.get("operation_id")
        rot = op.get("rotation_frac")
        if op_id is not None and rot is not None:
            op_rotation[op_id] = np.asarray(rot, dtype=float)

    # For each non-identity preserving op, compute the target HSP-star key
    # and check for exact (target_key, valley, op_id) match.
    has_exact_derived_match = False
    matched_targets: list[str] = []
    source_frac = np.asarray(k_frac, dtype=float) if k_frac is not None else None

    for op_id in preserving_ops:
        if op_id == 0 or op_id == "__identity__":
            continue
        # Compute target key: where does this op map the current kpoint?
        if source_frac is not None and op_id in op_rotation:
            target_key = compute_target_kpoint_key(
                source_frac=source_frac,
                operation_rotation=op_rotation[op_id],
                kpoint_frac_by_name=kpfbn,
            )
        else:
            # Fallback: check all targets (non-precise).
            target_key = None

        # Check exact match in derived characters
        if target_key is not None:
            kp_data = derived_by_target.get(target_key, {})
            valley_data = kp_data.get(valley, {})
            if op_id in valley_data:
                has_exact_derived_match = True
                matched_targets.append(f"{target_key}:{valley}:{op_id}")
        else:
            # Fallback: check any target (kept for backward compat when
            # k_frac or operation info is unavailable).
            for tk, kp_data in derived_by_target.items():
                valley_data = kp_data.get(valley, {})
                if op_id in valley_data:
                    has_exact_derived_match = True
                    matched_targets.append(f"{tk}:{valley}:{op_id}")
                    break

    resolved_by: list[str] = list(ebr_mapping.get("resolved_by", []) or [])

    if has_exact_derived_match:
        new_blockers = [b for b in blockers if b not in relevant_blockers]
        ebr_mapping["blocked_by"] = new_blockers
        ebr_mapping["character_source"] = "hsp_star_derived"
        if "hsp_star_character_derived" not in resolved_by:
            resolved_by.append("hsp_star_character_derived")
        ebr_mapping["resolved_by"] = resolved_by
        if matched_targets:
            ebr_mapping["derived_character_targets"] = matched_targets
        notes = str(ebr_mapping.get("notes", "") or "")
        if "HSP-star derived character available" not in notes:
            ebr_mapping["notes"] = notes + (
                " HSP-star derived character available; "
                "local character was sourced from symmetry conjugation."
            )
        if not new_blockers:
            ebr_mapping["ready"] = True
    else:
        new_blockers = [
            "hsp_star_derivation_not_available"
            if b in relevant_blockers else b
            for b in blockers
        ]
        deduped: list[str] = []
        for b in new_blockers:
            if b not in deduped:
                deduped.append(b)
        ebr_mapping["blocked_by"] = deduped


def _apply_target_subspace_closure_gate(
    *,
    ebr_mapping: dict[str, object],
    closure_blockers: list[str] | None,
    target_subspace_closure_report: dict[str, object] | None = None,
    kpoint_name: str = "",
    valley_preserving_ops: list[object] | None = None,
) -> None:
    """Apply target-subspace closure blockers to EBR readiness.

    Only blocks when a valley-preserving operation has a closure failure,
    since valley-changing operations are expected to have non-unitary D_raw
    in the target subspace.
    """
    if not closure_blockers:
        return
    preserving_ops = list(valley_preserving_ops or [])

    # Check if any closure failure is for a valley-preserving operation
    has_vp_closure_failure = False
    if target_subspace_closure_report is not None and kpoint_name and preserving_ops:
        for op_id in preserving_ops:
            if check_target_subspace_closure_blocked_for_operation(
                target_subspace_closure_report, kpoint_name, op_id,
            ):
                has_vp_closure_failure = True
                break

    filtered_blockers: list[str] = []
    for blocker in closure_blockers:
        if blocker == "target_subspace_closure_failed":
            if has_vp_closure_failure:
                filtered_blockers.append(blocker)
        elif blocker == "target_subspace_closure_not_evaluated":
            if has_vp_closure_failure:
                filtered_blockers.append(blocker)
        else:
            filtered_blockers.append(blocker)

    if not filtered_blockers:
        return
    if ebr_mapping.get("ready") is True and filtered_blockers:
        ebr_mapping["ready"] = False
    blockers: list[str] = list(ebr_mapping.get("blocked_by", []) or [])
    for blocker in filtered_blockers:
        if blocker not in blockers:
            blockers.append(blocker)
    ebr_mapping["blocked_by"] = blockers


def _refine_ebr_mapping_with_subspace_space_group(
    *,
    ebr_mapping: dict[str, object],
    subspace_space_group: dict[str, object],
    local_gka_operation_ids: list[object] | None = None,
) -> None:
    """Attach subspace SG identity without inventing local character blockers."""
    candidate = subspace_space_group.get("candidate_space_group_symbol")
    ebr_mapping["subspace_space_group_candidate"] = candidate
    blockers = ebr_mapping.get("blocked_by")
    if not isinstance(blockers, list):
        return
    if candidate in (None, "", "P1"):
        return
    local_ops = (
        list(local_gka_operation_ids)
        if isinstance(local_gka_operation_ids, list)
        else list(
            subspace_space_group.get("hsp_little_group_gka_operation_ids", [])
            or []
        )
    )
    has_nonidentity_local_op = any(
        op not in (0, "0", "__identity__")
        for op in local_ops
    )
    refined_blockers = []
    for blocker in blockers:
        if blocker == "subspace_group_candidate_missing":
            if has_nonidentity_local_op:
                continue
            refined_blockers.append("hsp_local_preserving_character_missing")
        else:
            refined_blockers.append(blocker)
    refined_blockers = [
        blocker for blocker in refined_blockers
        if isinstance(blocker, str) and blocker
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


def _resolved_subspace_group_context(
    *,
    standard_match: dict[str, object],
    local_gka_operation_ids: list[object],
) -> dict[str, object]:
    """Resolved subspace-space-group context from per-valley standard match.

    Carries the full spglib subgroup identification including Hall/setting
    provenance, global valley-preserving operation IDs, and the local
    HSP-little-group G_k^(a) operation IDs separately.
    """
    sg_number = standard_match.get("number")
    global_vp_ids = standard_match.get("operation_ids", [])
    ssg: dict[str, object] = {
        "status": "resolved",
        "candidate_space_group_number": (
            int(sg_number)
            if isinstance(sg_number, int) and not isinstance(sg_number, bool)
            else None
        ),
        "candidate_space_group_symbol": str(
            standard_match.get("international_short", "")
        ),
        "valley_preserving_operation_ids": list(
            global_vp_ids
        ) if isinstance(global_vp_ids, (list, tuple)) else [],
        "source": (
            "symmetry_analysis.valley_preserving_subgroup_report"
            ".per_valley_standard_matches"
        ),
    }
    # Hall/setting provenance when available.
    hall_number = standard_match.get("hall_number")
    if isinstance(hall_number, int) and not isinstance(hall_number, bool):
        ssg["hall_number"] = int(hall_number)
    hall_symbol = standard_match.get("hall_symbol")
    if isinstance(hall_symbol, str) and hall_symbol:
        ssg["hall_symbol"] = str(hall_symbol)
    # Local G_k^(a) — may differ from global VP ops when some VP
    # operations map the HSP to another star member.
    if list(local_gka_operation_ids) != ssg.get("valley_preserving_operation_ids"):
        ssg["hsp_little_group_gka_operation_ids"] = list(local_gka_operation_ids)
    return ssg


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


def _build_sampled_k_coverage(
    *,
    folded_center_report,
    kpoint_names: list[str],
    kpoint_frac_by_name: dict[str, np.ndarray],
) -> dict[str, object]:
    """Build a sampled-k branch coverage diagnostic.

    Reports nearest sampled k for each folded center and flags when
    sampled k-points appear to cover only one side/branch of available
    k-space locations.  Purely diagnostic — does not block readiness.
    """
    import numpy as np

    coverage_per_center: dict[str, object] = {}
    kp_names = list(kpoint_names)
    for entry in folded_center_report.entries:
        dists = folded_center_report.kpoint_distances.get(entry.center_name, [])
        if not dists:
            continue
        nearest_idx = int(np.argmin(dists))
        nearest_k = kp_names[nearest_idx] if nearest_idx < len(kp_names) else "?"
        coverage_per_center[entry.center_name] = {
            "folded_frac": entry.folded_frac.tolist(),
            "nearest_sampled_k": nearest_k,
            "nearest_distance_Ainv": float(dists[nearest_idx]),
            "all_distances": {
                kp: float(d) for kp, d in zip(kp_names, dists)
            },
        }

    # Simple branch coverage heuristic: for each center, check if all sampled
    # k-points lie on one side of the folded center in frac space.
    one_sided_warnings: list[dict[str, object]] = []
    for entry in folded_center_report.entries:
        folded_frac = np.asarray(entry.folded_frac, dtype=float)
        k_fracs = []
        for kp in kp_names:
            kf = np.asarray(kpoint_frac_by_name.get(kp, np.zeros(3)), dtype=float)
            delta = kf[:2] - folded_frac[:2]
            delta -= np.rint(delta)
            k_fracs.append(delta)
        if len(k_fracs) < 2:
            continue
        k_arr = np.array(k_fracs, dtype=float)
        if np.allclose(k_arr, 0.0, atol=1e-14):
            continue
        # Use uncentered SVD: projects points from origin (folded center).
        # If all projections have the same sign, all sampled k lie on one
        # side of the folded center.
        u, s, vt = np.linalg.svd(k_arr, full_matrices=False)
        proj = k_arr @ vt[0]
        if len(proj) >= 2 and (np.all(proj >= -1e-10) or np.all(proj <= 1e-10)):
            one_sided_warnings.append({
                "center_name": entry.center_name,
                "message": (
                    f"Sampled k-points appear to cover only one side of "
                    f"folded center {entry.center_name} (one-sided projection). "
                    "This is a sampling diagnostic, not a code blocker."
                ),
            })

    return {
        "per_center": coverage_per_center,
        "one_sided_branch_warnings": one_sided_warnings,
    }


def _warn_fixed_center_distance(
    *,
    folded_center_report,
    kpoint_names: list[str],
    weight_rows: list[dict[str, object]],
    subspace_payload: dict[str, object],
) -> None:
    """Inject mode-aware qualification into subspace_payload weight entries.

    For each (kpoint, row) with W_val near zero, check whether the row's
    center-resolved weights are all zero AND the corresponding folded
    centers are far from that specific kpoint.  Only then reclassify as
    fixed_center_not_captured.

    No stderr warnings — summary warnings are collected downstream by
    _collect_warnings() in summary_report.
    """
    kp_names = list(kpoint_names)
    large_distance_Ainv = 0.1

    # Build per-center distance lookup: center_name -> {kpoint_name: distance}
    center_dist: dict[str, dict[str, float]] = {}
    for entry in folded_center_report.entries:
        dists = folded_center_report.kpoint_distances.get(entry.center_name, [])
        if len(dists) == len(kp_names):
            center_dist[entry.center_name] = dict(zip(kp_names, dists))

    for kp_name, kp_data in subspace_payload.get("kpoints", {}).items():
        if not isinstance(kp_data, dict):
            continue
        for w in kp_data.get("weights", []):
            if not isinstance(w, dict):
                continue
            if w.get("valley_status") not in ("not_derived", "not_valley_derived"):
                continue
            if w.get("W_val", 1.0) >= 0.01:
                continue
            # Check per-center distances for THIS kpoint only.
            center_weights = w.get("center_weights", {})
            all_far = True
            any_center = False
            for cname, cw in center_weights.items():
                any_center = True
                cdist = center_dist.get(cname, {}).get(kp_name, 0.0)
                if cw == 0.0 and cdist > large_distance_Ainv:
                    continue  # this center is far, weight is zero
                all_far = False
                break
            if any_center and all_far:
                w["valley_status"] = "fixed_center_not_captured"
                w["valley_status_note"] = (
                    "fixed_center projector: all center weights are zero "
                    "and every folded center is > 0.1 A^-1 from this k-point. "
                    "This is a k/center mismatch, not necessarily non-parent-valley. "
                    "Consider k_resolved_parent_valley projector_mode."
                )


# ---------------------------------------------------------------------------
# Auto-canonical reduced EBR mapping (per-bundle processing)
# ---------------------------------------------------------------------------

def _build_auto_canonical_mapping(
    *,
    ebr_export_bundle: dict[str, object] | None,
    spinor_wf: bool,
    max_coefficient: int = 6,
) -> dict[str, object] | None:
    """Build auto-canonical tables and solve reduced EBR per ready bundle.

    Each ready bundle is processed independently: its exact Bilbao→ValleyScope
    HSP mapping is derived, a reduced table built, and the solver run.
    Every input bundle must appear exactly once in the combined results
    (solution or excluded).  Global ``solved_exact`` requires every ready
    bundle to have an exact nonnegative witness.
    """
    if ebr_export_bundle is None:
        return None

    bundles = ebr_export_bundle.get("bundles", [])
    if not isinstance(bundles, list) or not bundles:
        return None

    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
        build_auto_time_reversal_reduced_ebr_table,
    )
    from valleyscope.analysis.reduced_ebr_mapping import (
        build_reduced_ebr_mapping as _solve_mapping,
    )

    all_solutions: list[dict[str, object]] = []
    all_excluded: list[dict[str, object]] = []
    per_bundle_statuses: list[dict[str, object]] = []
    ready_count = 0
    time_reversal_ready_count = 0

    for b in bundles:
        if not isinstance(b, dict):
            continue

        bundle_id = str(b.get("bundle_id", "?"))
        is_ready = bool(b.get("ready_for_external_solver"))
        is_validation_candidate = bool(
            b.get("ready_for_reduced_table_validation")
        )

        if not is_ready and not is_validation_candidate:
            all_excluded.append({
                "bundle_id": bundle_id,
                "subspace_group_candidate": b.get("subspace_group_candidate", ""),
                "subspace_space_group": b.get("subspace_space_group", {}),
                "reason": "not ready for external solver",
            })
            continue

        ready_count += 1

        # Validate required fields for auto-canonical processing.
        ssg = b.get("subspace_space_group", {})
        if not isinstance(ssg, dict) or not ssg:
            all_excluded.append({
                "bundle_id": bundle_id,
                "subspace_group_candidate": b.get("subspace_group_candidate", ""),
                "subspace_space_group": b.get("subspace_space_group", {}),
                "reason": "missing subspace_space_group in ready bundle",
            })
            per_bundle_statuses.append({
                "bundle_id": bundle_id,
                "status": "blocked",
                "reason": "missing subspace_space_group",
            })
            continue

        sg_num = ssg.get("candidate_space_group_number")
        if not isinstance(sg_num, int) or isinstance(sg_num, bool) or sg_num <= 0:
            all_excluded.append({
                "bundle_id": bundle_id,
                "subspace_group_candidate": b.get("subspace_group_candidate", ""),
                "subspace_space_group": ssg,
                "reason": "missing or invalid candidate_space_group_number",
            })
            per_bundle_statuses.append({
                "bundle_id": bundle_id,
                "status": "blocked",
                "reason": "invalid candidate_space_group_number",
            })
            continue

        irreps_by_kp = b.get("irreps_by_kpoint", {})
        if not isinstance(irreps_by_kp, dict) or not irreps_by_kp:
            all_excluded.append({
                "bundle_id": bundle_id,
                "subspace_group_candidate": b.get("subspace_group_candidate", ""),
                "subspace_space_group": ssg,
                "reason": "missing irreps_by_kpoint",
            })
            per_bundle_statuses.append({
                "bundle_id": bundle_id,
                "status": "blocked",
                "reason": "missing irreps_by_kpoint",
            })
            continue

        expected_hsps = b.get("expected_hsps", [])
        if not isinstance(expected_hsps, list) or not expected_hsps:
            all_excluded.append({
                "bundle_id": bundle_id,
                "subspace_group_candidate": b.get("subspace_group_candidate", ""),
                "subspace_space_group": ssg,
                "reason": "missing expected_hsps",
            })
            per_bundle_statuses.append({
                "bundle_id": bundle_id,
                "status": "blocked",
                "reason": "missing expected_hsps",
            })
            continue

        sg_num = int(sg_num)
        sg_candidate = str(b.get("subspace_group_candidate", ""))

        # Build auto table and solve for this single bundle.  The table's
        # standard-setting evidence is derived independently by the promotion
        # validator (spglib); it is never copied from the bundle certificate.
        try:
            if b.get("problem_kind") == "valley_orbit_reduced_ebr":
                time_reversal_ready_count += 1
                time_reversal = b.get("time_reversal", {})
                grey_bns_number = (
                    time_reversal.get("grey_bns_number")
                    if isinstance(time_reversal, dict) else None
                )
                if not isinstance(grey_bns_number, str) or not grey_bns_number:
                    raise ValueError(
                        "valley-orbit bundle has no reviewed grey BNS number"
                    )
                table = build_auto_time_reversal_reduced_ebr_table(
                    unitary_space_group_number=sg_num,
                    grey_bns_number=grey_bns_number,
                    spinor=spinor_wf,
                    bundle_irreps_by_kpoint=irreps_by_kp,
                    expected_hsps=expected_hsps,
                    subspace_group_candidate=sg_candidate,
                    subspace_space_group=(
                        ssg if isinstance(ssg, dict) else None
                    ),
                )
            else:
                table = build_auto_canonical_reduced_ebr_table(
                    subspace_sg_number=sg_num,
                    spinor=spinor_wf,
                    bundle_irreps_by_kpoint=irreps_by_kp,
                    expected_hsps=expected_hsps,
                    subspace_group_candidate=sg_candidate,
                    subspace_space_group=ssg if isinstance(ssg, dict) else None,
                )
        except Exception as exc:
            all_excluded.append({
                "bundle_id": bundle_id,
                "subspace_group_candidate": sg_candidate,
                "subspace_space_group": ssg,
                "reason": (
                    f"auto-canonical table build failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            })
            per_bundle_statuses.append({
                "bundle_id": bundle_id,
                "sg_number": sg_num,
                "expected_hsps": expected_hsps,
                "status": "blocked",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        prov = table.get("provenance", {}) if isinstance(table.get("provenance"), dict) else {}
        table_irreps = table.get("irreps", [])
        num_irreps = len(table_irreps) if isinstance(table_irreps, list) else 0
        # source_basis_count = total irreptables EBR basis labels before HSP
        # reduction; reduction_basis_count = after sampled-HSP k-vector filtering.
        source_basis = int(prov.get("source_basis_count", num_irreps))
        reduction_basis = int(prov.get("reduction_basis_count", num_irreps))
        is_time_reversal = (
            b.get("problem_kind") == "valley_orbit_reduced_ebr"
        )
        bundle_input: dict[str, object] = {
            "source": (
                "auto_time_reversal_grey"
                if is_time_reversal else "auto_canonical"
            ),
            "auto_canonical": not is_time_reversal,
            "auto_time_reversal": is_time_reversal,
            "subspace_group_candidate": table.get("subspace_group_candidate", ""),
            "space_group_number": prov.get("space_group_number", sg_num),
            "spinful": prov.get("spinful", spinor_wf),
            "data_source": prov.get("data_source", ""),
            "package": prov.get("package", ""),
            "package_version": prov.get("package_version", ""),
            "expected_hsps": list(expected_hsps) if isinstance(expected_hsps, list) else [],
            "valleyscope_reduction": prov.get("valleyscope_reduction", ""),
            "source_basis_count": source_basis,
            "reduction_basis_count": reduction_basis,
        }
        if is_time_reversal:
            bundle_input["time_reversal_grey_bns_number"] = prov.get(
                "time_reversal_grey_bns_number"
            )
            bundle_input["time_reversal_source"] = prov.get(
                "time_reversal_source"
            )
        zero_ebrs = prov.get("filtered_zero_vector_ebrs")
        zero_count = prov.get("filtered_zero_vector_ebr_count")
        if isinstance(zero_ebrs, list) and zero_ebrs:
            bundle_input["filtered_zero_vector_ebrs"] = list(zero_ebrs)
        elif isinstance(zero_count, int) and not isinstance(zero_count, bool) and zero_count > 0:
            bundle_input["filtered_zero_vector_ebr_count"] = int(zero_count)
        if prov.get("dropped_source_rows"):
            bundle_input["dropped_source_rows"] = prov["dropped_source_rows"]
            bundle_input["dropped_source_row_count"] = prov.get("dropped_source_row_count", len(prov["dropped_source_rows"]))

        # Solve this single bundle against its auto table.
        single_export = {"bundles": [b]}
        bundle_result = _solve_mapping(
            ebr_export_bundle=single_export,
            table=table,
            max_coefficient=max_coefficient,
            reduced_ebr_input=bundle_input,
        )

        # Inject table-level provenance into each solution so the
        # reduced EBR output is self-auditing.
        for sol in bundle_result.get("solutions", []):
            if isinstance(sol, dict):
                sol["table_provenance"] = dict(bundle_input)
                sol["table_status"] = "loaded"
        all_solutions.extend(bundle_result.get("solutions", []))
        all_excluded.extend(bundle_result.get("excluded_bundles", []))
        per_bundle_statuses.append({
            "bundle_id": bundle_id,
            "sg_number": sg_num,
            "expected_hsps": list(expected_hsps) if isinstance(expected_hsps, list) else [],
            "status": bundle_result.get("mapping_status", "unknown"),
            "table_status": "loaded",
        })

    # --- Aggregate top-level status ---
    # Zero ready bundles → not_evaluated.
    if ready_count == 0:
        return {
            "status": "not_evaluated",
            "mapping_status": "not_evaluated",
            "reduced_ebr_decomposition_status": "not_evaluated",
            "table_status": "not_provided",
            "solutions": [],
            "excluded_bundles": all_excluded,
            "solver": "smith_normal_form_plus_bounded_nonnegative_search",
            "max_coefficient": max_coefficient,
            "interpretation": "no ready bundles available for auto-canonical evaluation",
            "reduced_ebr_input": {
                "source": "auto_canonical_blocked",
                "ready_bundle_count": 0,
            },
            "auto_canonical_bundles": per_bundle_statuses,
        }

    all_ready_solved = (
        len(all_solutions) == ready_count
        and all(s.get("status") == "solved_exact" for s in all_solutions)
    )
    any_blocked = any(
        s.get("status") == "blocked" for s in per_bundle_statuses
    )
    any_indeterminate = any(
        s.get("status") == "indeterminate_truncated" for s in per_bundle_statuses
    ) or any(
        s.get("status") == "indeterminate_truncated"
        for s in all_solutions
    )

    any_solved = len(all_solutions) > 0
    if all_ready_solved:
        global_status = "solved_exact"
    elif any_blocked and any_solved:
        global_status = "partial"
    elif any_blocked:
        global_status = "blocked"
    elif any_indeterminate:
        global_status = "indeterminate_truncated"
    else:
        global_status = "no_exact_solution"

    table_status = "loaded" if any_solved else "not_provided"

    result: dict[str, object] = {
        "status": global_status,
        "mapping_status": global_status,
        "reduced_ebr_decomposition_status": global_status,
        "table_status": table_status,
        "solutions": all_solutions,
        "excluded_bundles": all_excluded,
        "solver": "smith_normal_form_plus_bounded_nonnegative_search",
        "max_coefficient": max_coefficient,
        "interpretation": (
            "Exact integer linear combination of EBR vectors matching the "
            "bundle irrep count vector.  No heuristic fit; only exact matches "
            "are reported.  Auto-canonical tables were built per ready bundle "
            "from irreptables source data."
        ),
        "reduced_ebr_input": {
            "source": (
                "auto_time_reversal_grey"
                if time_reversal_ready_count == ready_count
                else "auto_canonical"
            ),
            "auto_canonical": time_reversal_ready_count == 0,
            "auto_time_reversal": time_reversal_ready_count > 0,
            "spinful": spinor_wf,
            "ready_bundle_count": ready_count,
            "solved_count": len(all_solutions),
            "blocked_count": sum(
                1 for s in per_bundle_statuses
                if s.get("status") in ("blocked", "unknown")
            ),
            "excluded_count": len(all_excluded),
        },
        "auto_canonical_bundles": per_bundle_statuses,
    }
    return result
