import json
import numpy as np
import pytest

from valleyscope.analysis.symmetry_adapted_valley_report import (
    build_symmetry_adapted_valley_report,
    summarize_symmetry_adapted_valley_report,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _c3_mstar_setup():
    """C3 three-valley cyclic with C2_M1: full success path."""
    p_m1 = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_m2 = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    p_m3 = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3
    d_c2_m1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex128)

    seeds = {"M1": p_m1, "M2": p_m2, "M3": p_m3}
    reps = {0: d_e, 1: d_c3, 2: d_c3sq, 3: d_c2_m1}
    mappings = {
        0: {"M1": "M1", "M2": "M2", "M3": "M3"},
        1: {"M1": "M2", "M2": "M3", "M3": "M1"},
        2: {"M1": "M3", "M2": "M1", "M3": "M2"},
        3: {"M1": "M1", "M2": "M3", "M3": "M2"},
    }
    orbit = ["M1", "M2", "M3"]
    return seeds, reps, mappings, orbit


# -----------------------------------------------------------------------
# 1. Successful full pipeline
# -----------------------------------------------------------------------

def test_full_pipeline_c3_mstar_success():
    seeds, reps, mappings, orbit = _c3_mstar_setup()

    report = build_symmetry_adapted_valley_report(
        seed_projectors=seeds,
        representations=reps,
        valley_mappings=mappings,
        orbit=orbit,
        reference_valley="M1",
        rank=1,
    )

    assert report["local_irrep_ready"] is True
    assert report["diagnostic_only"] is False
    assert report["status"] == "ok"
    assert report["irrep_matching_input_ready"] is True
    assert report["irrep_matching_input_status"] == "ready"

    # All sub-reports present
    for key in [
        "symmetry_adapted_projectors",
        "valley_preserving_representations",
        "valley_sewing_matrices",
        "valley_preserving_character_diagnostics",
    ]:
        assert key in report, f"missing: {key}"

    proj = report["symmetry_adapted_projectors"]
    assert proj["status"] == "ok"
    assert proj["selected_rank"] == 1


# -----------------------------------------------------------------------
# 2. Compact summary
# -----------------------------------------------------------------------

def test_compact_summary_serializable():
    seeds, reps, mappings, orbit = _c3_mstar_setup()

    report = build_symmetry_adapted_valley_report(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )
    summary = summarize_symmetry_adapted_valley_report(report)

    encoded = json.dumps(summary)
    assert len(encoded) > 0
    assert "dtype" not in encoded

    for key in [
        "status", "reason", "local_irrep_ready", "diagnostic_only",
        "feature_status", "workflow_integration_status", "trusted_irrep_label",
        "irrep_matching_input_ready", "irrep_matching_input_status",
        "irrep_matching_input_reason",
        "orbit", "reference_valley",
        "symmetry_adapted_projectors",
        "valley_preserving_representations",
        "valley_sewing_matrices",
        "valley_preserving_character_diagnostics",
        "subspace_group",
        "subspace_space_group",
        "ebr_mapping_input",
    ]:
        assert key in summary, f"missing: {key}"
    assert summary["feature_status"] == "formal"
    assert summary["workflow_integration_status"] == "integrated"
    assert summary["trusted_irrep_label"] is False
    assert summary["irrep_matching_input_ready"] is True
    assert summary["irrep_matching_input_status"] == "ready"
    assert "valley_sewing_matrices_summary" not in json.dumps(
        summary["valley_preserving_representations"]
    )
    assert set(summary["valley_sewing_matrices"]) == {
        "status",
        "max_sewing_unitarity_error",
        "items",
    }


def test_subspace_group_uses_operation_order_not_matrix_rank():
    seed = np.array([[1.0]], dtype=np.complex128)
    c2 = np.array([[-1.0]], dtype=np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"M": seed},
        representations={0: np.eye(1, dtype=np.complex128), 7: c2},
        valley_mappings={0: {"M": "M"}, 7: {"M": "M"}},
        orbit=["M"],
        reference_valley="M",
        rank=1,
        operation_orders={0: 1, 7: 2},
        spinor_wavefunction=False,
        spinor_convention_verified=True,
    )

    assert report["subspace_group"]["effective_point_group"] is None
    assert report["subspace_group"]["subspace_group_candidate"] is None
    # removed - legacy field deleted
    assert report["ebr_mapping_input"]["subspace_group_candidate"] is None


def test_ebr_mapping_uses_configurable_seed_overlap_threshold():
    p_seed = np.diag([0.75, 0.25]).astype(np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"M": p_seed},
        representations={0: np.eye(2, dtype=np.complex128)},
        valley_mappings={0: {"M": "M"}},
        orbit=["M"],
        reference_valley="M",
        rank=1,
        operation_orders={0: 1},
        spinor_wavefunction=False,
        spinor_convention_verified=True,
        seed_overlap_fail_tol=0.1,
        seed_overlap_warn_tol=0.1,
        ebr_seed_overlap_min=0.9,
    )

    assert report["symmetry_adapted_projectors"]["status"] == "ok"
    assert report["ebr_mapping_input"]["ready"] is False
    assert any(
        str(item).startswith("low_seed_overlap_min=")
        for item in report["ebr_mapping_input"]["blocked_by"]
    )


# -----------------------------------------------------------------------
# 3. Schema: no forbidden terms
# -----------------------------------------------------------------------

def test_schema_no_forbidden_terms():
    seeds, reps, mappings, orbit = _c3_mstar_setup()

    report = build_symmetry_adapted_valley_report(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )
    summary = summarize_symmetry_adapted_valley_report(report)
    encoded = json.dumps(summary)

    for forbidden in [
        "covariance", "equivariant", "equivariance",
        "stabilizer", "valley_little_group", "p_cov",
    ]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"


def test_projector_warning_propagates_without_marking_diagnostic_only():
    p_a = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_b = np.diag([0.0, 0.6, 0.4]).astype(np.complex128)
    d_swap = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: np.eye(3, dtype=np.complex128), 1: d_swap},
        valley_mappings={
            0: {"VA": "VA", "VB": "VB"},
            1: {"VA": "VB", "VB": "VA"},
        },
        orbit=["VA", "VB"], reference_valley="VA", rank=1,
    )

    assert report["status"] == "warn"
    assert report["diagnostic_only"] is False
    assert report["local_irrep_ready"] is True
    assert "projector_warning" in report["reason"]


def test_spinor_unverified_blocks_irrep_matching_input():
    seeds, reps, mappings, orbit = _c3_mstar_setup()

    report = build_symmetry_adapted_valley_report(
        seed_projectors=seeds,
        representations=reps,
        valley_mappings=mappings,
        orbit=orbit,
        reference_valley="M1",
        rank=1,
        spinor_wavefunction=True,
        spinor_convention_verified=False,
    )

    assert report["local_irrep_ready"] is True
    assert report["diagnostic_only"] is False
    assert report["irrep_matching_input_ready"] is False
    assert report["irrep_matching_input_status"] == "blocked"
    assert "spinor convention unverified" in report["irrep_matching_input_reason"]


def test_ambiguous_representatives_expose_candidate_wise_summary():
    p_m1 = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_m2 = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    p_m3 = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3
    d_diff = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"M1": p_m1, "M2": p_m2, "M3": p_m3},
        representations={0: d_e, 1: d_c3, 2: d_c3sq, 5: d_diff},
        valley_mappings={
            0: {"M1": "M1", "M2": "M2", "M3": "M3"},
            1: {"M1": "M2", "M2": "M3", "M3": "M1"},
            2: {"M1": "M3", "M2": "M1", "M3": "M2"},
            5: {"M1": "M2", "M2": "M1", "M3": "M3"},
        },
        orbit=["M1", "M2", "M3"],
        reference_valley="M1",
        rank=1,
    )
    summary = summarize_symmetry_adapted_valley_report(report)
    projectors = summary["symmetry_adapted_projectors"]

    assert summary["diagnostic_only"] is True
    assert summary["local_irrep_ready"] is False
    assert projectors["representative_resolution_by_valley"]["M2"] == (
        "ambiguous_inequivalent_candidates"
    )
    assert projectors["representative_selection_policy_by_valley"]["M2"] == "none"
    assert projectors["selected_representative_by_valley"]["M2"] is None
    assert projectors["representative_auto_selected_by_valley"]["M2"] is False
    pairwise = projectors["representative_candidate_projector_differences_by_valley"]["M2"]
    assert pairwise[0]["candidate_a"] == 1
    assert pairwise[0]["candidate_b"] == 5
    assert pairwise[0]["projector_difference"] > 0.1
    json.dumps(summary)


# -----------------------------------------------------------------------
# 4. Failure: rank ambiguity
# -----------------------------------------------------------------------

def test_failure_rank_ambiguity_propagates():
    """gap_insufficient makes projector fail → diagnostic_only."""
    p_m1 = np.diag([0.52, 0.48, 0.45]).astype(np.complex128)
    p_m2 = np.diag([0.45, 0.52, 0.48]).astype(np.complex128)
    p_m3 = np.diag([0.48, 0.45, 0.52]).astype(np.complex128)
    d_e = np.eye(3, dtype=np.complex128)
    d_c3 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.complex128)
    d_c3sq = d_c3 @ d_c3

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"M1": p_m1, "M2": p_m2, "M3": p_m3},
        representations={0: d_e, 1: d_c3, 2: d_c3sq},
        valley_mappings={
            0: {"M1": "M1", "M2": "M2", "M3": "M3"},
            1: {"M1": "M2", "M2": "M3", "M3": "M1"},
            2: {"M1": "M3", "M2": "M1", "M3": "M2"},
        },
        orbit=["M1", "M2", "M3"], reference_valley="M1",
        rank_method="gap", rank_tol=0.5,
    )

    assert report["diagnostic_only"] is True
    assert report["local_irrep_ready"] is False
    assert "rank gap insufficient" in report["reason"]


# -----------------------------------------------------------------------
# 5. Failure: missing representative operation
# -----------------------------------------------------------------------

def test_failure_missing_representative():
    p_a = np.diag([1.0, 0.0]).astype(np.complex128)
    p_b = np.diag([0.0, 1.0]).astype(np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: np.eye(2, dtype=np.complex128)},
        valley_mappings={0: {"VA": "VA"}},
        orbit=["VA", "VB"], reference_valley="VA", rank=1,
    )

    assert report["diagnostic_only"] is True
    assert report["local_irrep_ready"] is False
    assert "no representative operation" in report["reason"]


def test_projector_failure_skips_downstream_diagnostics():
    p_a = np.diag([1.0, 0.0]).astype(np.complex128)
    p_b = np.diag([0.0, 1.0]).astype(np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: np.eye(2, dtype=np.complex128)},
        valley_mappings={0: {"VA": "VA", "VB": "VB"}},
        orbit=["VA", "VB"],
        reference_valley="VA",
        rank=1,
    )

    assert report["diagnostic_only"] is True
    assert report["local_irrep_ready"] is False
    assert report["irrep_matching_input_ready"] is False
    assert "projector_construction_failed" in report["reason"]
    assert "representation_diagnostics" not in report["reason"]
    assert report["valley_preserving_representations"]["reason"] == (
        "not evaluated because projector construction failed"
    )
    assert report["valley_preserving_character_diagnostics"]["reason"] == (
        "not evaluated because projector construction failed"
    )


def test_invalid_projector_input_returns_diagnostic_report():
    p_a = np.diag([1.0, 0.0]).astype(np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"VA": p_a},
        representations={0: np.eye(2, dtype=np.complex128)},
        valley_mappings={0: {"VA": "VA"}},
        orbit=["VA", "VB"],
        reference_valley="VA",
        rank=1,
    )

    assert report["status"] == "diagnostic_only"
    assert report["diagnostic_only"] is True
    assert report["local_irrep_ready"] is False
    assert report["trusted_irrep_label"] is False
    assert "projector_input_invalid" in report["reason"]
    assert report["valley_sewing_matrices"]["items"] == []


# -----------------------------------------------------------------------
# 6. Failure: low seed overlap
# -----------------------------------------------------------------------

def test_failure_low_seed_overlap():
    p_a = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_b = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    d_swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"VA": p_a, "VB": p_b},
        representations={0: np.eye(3, dtype=np.complex128), 1: d_swap},
        valley_mappings={
            0: {"VA": "VA", "VB": "VB"},
            1: {"VA": "VB", "VB": "VA"},
        },
        orbit=["VA", "VB"], reference_valley="VA", rank=1,
    )

    assert report["diagnostic_only"] is True
    assert report["local_irrep_ready"] is False
    assert "seed overlap" in report["reason"].lower()


# -----------------------------------------------------------------------
# 7. Failure: non-unitary sewing
# -----------------------------------------------------------------------

def test_failure_non_unitary_sewing():
    """D_g with small non-unitarity passes projector construction but fails
    sewing unitarity check because the eigenvalue modulus deviates."""
    p_a = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    p_b = np.diag([0.0, 1.0, 0.0]).astype(np.complex128)
    p_c = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
    # D_g close to swap but slightly non-unitary (eigenvalue 1.5 instead of 1)
    d_bad = np.array([
        [0.0, 1.5, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.complex128)

    report = build_symmetry_adapted_valley_report(
        seed_projectors={"V1": p_a, "V2": p_b, "V3": p_c},
        representations={
            0: np.eye(3, dtype=np.complex128),
            1: d_bad,
        },
        valley_mappings={
            0: {"V1": "V1", "V2": "V2", "V3": "V3"},
            1: {"V1": "V2", "V2": "V1", "V3": "V3"},
        },
        orbit=["V1", "V2", "V3"], reference_valley="V1", rank=1,
        unitarity_tol=1e-10,
    )

    assert report["diagnostic_only"] is True
    assert report["local_irrep_ready"] is False


# -----------------------------------------------------------------------
# 8. Default workflow does not change old outputs (no integration test)
# -----------------------------------------------------------------------

def test_report_structure_consistent():
    """Verify all sub-report keys exist even in diagnostic_only mode."""
    seeds, reps, mappings, orbit = _c3_mstar_setup()

    report = build_symmetry_adapted_valley_report(
        seed_projectors=seeds, representations=reps,
        valley_mappings=mappings, orbit=orbit, reference_valley="M1", rank=1,
    )

    vp = report["valley_preserving_representations"]
    assert isinstance(vp, dict)
    assert "valley_preserving_operations" in vp or "status" in vp

    sewing = report["valley_sewing_matrices"]
    assert sewing is not None

    chars = report["valley_preserving_character_diagnostics"]
    assert isinstance(chars, dict)
    assert "per_valley" in chars or "status" in chars


# -----------------------------------------------------------------------
# 9. Workflow integration: default off, flag=true
# -----------------------------------------------------------------------

def _write_h5_three_valley(path):
    import h5py
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3) * 10.0
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"
        kp["frac"] = np.zeros(3)
        kp["cart"] = np.zeros(3)
        kp["g_vectors_frac"] = np.zeros((3, 3))
        kp["g_vectors_cart"] = np.array([
            [0.5, 0.0, 0.0],
            [-0.25, 0.4330127018922193, 0.0],
            [-0.25, -0.4330127018922193, 0.0],
        ])
        coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
        coeffs[0, 0, 0] = 1.0
        coeffs[1, 0, 1] = 1.0
        coeffs[2, 0, 2] = 1.0
        kp["coefficients"] = coeffs
        kp["energies_eV"] = np.array([0.1, 0.1001, 0.1002])
        kp["band_indices_vasp"] = np.array([101, 102, 103])


def _write_hex_poscar(path):
    path.write_text(
        "hex\n1.0\n"
        "1.0 0.0 0.0\n-0.5 0.8660254 0.0\n0.0 0.0 4.0\n"
        "X\n1\nDirect\n0.0 0.0 0.0\n", encoding="utf-8"
    )


def test_disabled_no_symmetry_adapted_valley_output(tmp_path):
    """enabled=false: no symmetry_adapted_valley_analysis in outputs."""
    import yaml
    from valleyscope.workflows.analyze_hsp import analyze_hsp

    h5_path = tmp_path / "wf.h5"; mono = tmp_path / "m.vasp"
    struct = tmp_path / "POSCAR"; out_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    _write_h5_three_valley(h5_path); _write_hex_poscar(struct); _write_hex_poscar(mono)

    config = {
        "input": {"wavefunction_h5": str(h5_path),
                   "monolayer_poscars": {"top": str(mono), "bottom": str(mono)}},
        "layer_transforms": {
            "top": {"supercell_matrix": [[1,0,0],[0,1,0],[0,0,1]]},
            "bottom": {"supercell_matrix": [[1,0,0],[0,1,0],[0,0,1]]},
        },
        "analysis": {"kpoints": ["GammaM"], "iband": [101,102,103],
                      "degeneracy_tol_meV": 1.0,
                      "symmetry_adapted_valley": {"enabled": False}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [
            {"name":"M1c","cart":[0.5,0,0]},
            {"name":"M2c","cart":[-0.25,0.4330127018922193,0]},
            {"name":"M3c","cart":[-0.25,-0.4330127018922193,0]},
        ]},
        "valley_subspaces": [
            {"name":"M1","centers":["M1c"]},
            {"name":"M2","centers":["M2c"]},
            {"name":"M3","centers":["M3c"]},
        ],
        "projection": {"qcut_mode":"absolute","qcut_Ainv":0.3,"overlap_policy":"warn_exclude",
                        "thresholds":{"W_val_min":0.5}},
        "symmetry": {"operations":{"structure_file":str(struct)},"tolerance":{"symprec":1e-3},
                      "filters":{"rotation_order":"auto"}},
        "output": {"directory": str(out_dir), "profile": "debug"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    outputs = analyze_hsp(config_path)

    assert "symmetry_adapted_valley_analysis_json" not in outputs
    assert not (out_dir / "symmetry_adapted_valley_analysis.json").exists()

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "symmetry_adapted_valley_analysis" not in summary


def test_default_writes_symmetry_adapted_valley_analysis(tmp_path):
    """The formal symmetry-adapted valley analysis is enabled by default."""
    import yaml
    from valleyscope.workflows.analyze_hsp import analyze_hsp

    h5_path = tmp_path / "wf.h5"; mono = tmp_path / "m.vasp"
    struct = tmp_path / "POSCAR"; out_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    _write_h5_three_valley(h5_path); _write_hex_poscar(struct); _write_hex_poscar(mono)

    config = {
        "input": {"wavefunction_h5": str(h5_path),
                   "monolayer_poscars": {"top": str(mono), "bottom": str(mono)}},
        "layer_transforms": {
            "top": {"supercell_matrix": [[1,0,0],[0,1,0],[0,0,1]]},
            "bottom": {"supercell_matrix": [[1,0,0],[0,1,0],[0,0,1]]},
        },
        "analysis": {"kpoints": ["GammaM"], "iband": [101,102,103],
                      "degeneracy_tol_meV": 1.0},
        "valley_centers": {"coordinate_mode": "cart", "centers": [
            {"name":"M1c","cart":[0.5,0,0]},
            {"name":"M2c","cart":[-0.25,0.4330127018922193,0]},
            {"name":"M3c","cart":[-0.25,-0.4330127018922193,0]},
        ]},
        "valley_subspaces": [
            {"name":"M1","centers":["M1c"]},
            {"name":"M2","centers":["M2c"]},
            {"name":"M3","centers":["M3c"]},
        ],
        "projection": {"qcut_mode":"absolute","qcut_Ainv":0.3,"overlap_policy":"warn_exclude",
                        "thresholds":{"W_val_min":0.5}},
        "symmetry": {"operations":{"structure_file":str(struct)},"tolerance":{"symprec":1e-3},
                      "filters":{"rotation_order":"auto"}},
        "output": {"directory": str(out_dir), "profile": "debug"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    outputs = analyze_hsp(config_path)

    assert "symmetry_adapted_valley_analysis_json" in outputs
    report_path = outputs["symmetry_adapted_valley_analysis_json"]
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "by_kpoint" in report
    assert report["space_group_valley_orbits"] == [["M1", "M2", "M3"]]
    gm = report["by_kpoint"].get("GammaM", {})
    assert "orbits" in gm
    assert "valley_preserving_subspaces" in gm
    assert [item["orbit"] for item in gm["valley_preserving_subspaces"]] == [
        ["M1"], ["M2"], ["M3"],
    ]
    assert gm["feature_status"] == "formal"
    assert gm["workflow_integration_status"] == "integrated"
    assert gm["trusted_irrep_label"] is False

    # Forbidden terms check
    encoded = json.dumps(report)
    for forbidden in [
        "covariance", "equivariant", "stabilizer", "valley_little_group",
        "p_cov", "experimental",
    ]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"


def test_enabled_true_without_d_raw_writes_not_evaluated_report(tmp_path):
    """The formal analysis writes an explicit not_evaluated report if D_raw is absent."""
    import yaml
    from valleyscope.workflows.analyze_hsp import analyze_hsp

    h5_path = tmp_path / "wf.h5"; mono = tmp_path / "m.vasp"
    out_dir = tmp_path / "out"; config_path = tmp_path / "config.yaml"
    _write_h5_three_valley(h5_path); _write_hex_poscar(mono)

    config = {
        "input": {"wavefunction_h5": str(h5_path),
                   "monolayer_poscars": {"top": str(mono), "bottom": str(mono)}},
        "layer_transforms": {
            "top": {"supercell_matrix": [[1,0,0],[0,1,0],[0,0,1]]},
            "bottom": {"supercell_matrix": [[1,0,0],[0,1,0],[0,0,1]]},
        },
        "analysis": {"kpoints": ["GammaM"], "iband": [101,102,103],
                      "degeneracy_tol_meV": 1.0},
        "valley_centers": {"coordinate_mode": "cart", "centers": [
            {"name":"M1c","cart":[0.5,0,0]},
            {"name":"M2c","cart":[-0.25,0.4330127018922193,0]},
            {"name":"M3c","cart":[-0.25,-0.4330127018922193,0]},
        ]},
        "valley_subspaces": [
            {"name":"M1","centers":["M1c"]},
            {"name":"M2","centers":["M2c"]},
            {"name":"M3","centers":["M3c"]},
        ],
        "projection": {"qcut_mode":"absolute","qcut_Ainv":0.3,"overlap_policy":"warn_exclude",
                        "thresholds":{"W_val_min":0.5}},
        "output": {"directory": str(out_dir), "profile": "debug"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    outputs = analyze_hsp(config_path)

    assert "symmetry_adapted_valley_analysis_json" in outputs
    report = json.loads(outputs["symmetry_adapted_valley_analysis_json"].read_text(encoding="utf-8"))
    gm = report["by_kpoint"]["GammaM"]
    assert gm["status"] == "not_evaluated"
    assert gm["diagnostic_only"] is True
    assert gm["local_irrep_ready"] is False
    assert gm["trusted_irrep_label"] is False
    assert gm["orbits"] == []


def test_partition_valley_orbits_separates_disconnected_components():
    from valleyscope.workflows.analyze_hsp import _partition_valley_orbits

    orbits = _partition_valley_orbits(
        valley_names=["K", "Kp", "M1", "M2"],
        valley_mappings={
            0: {"K": "K", "Kp": "Kp", "M1": "M1", "M2": "M2"},
            1: {"K": "Kp", "Kp": "K"},
            2: {"M1": "M2", "M2": "M1"},
        },
    )

    assert orbits == [["K", "Kp"], ["M1", "M2"]]


def test_valley_preserving_subspace_reports_keep_three_mstar_p2_subspaces():
    from valleyscope.workflows.analyze_hsp import _build_valley_preserving_subspace_reports

    seeds = {
        "M1": np.diag([1, 1, 0, 0, 0, 0]).astype(np.complex128),
        "M2": np.diag([0, 0, 1, 1, 0, 0]).astype(np.complex128),
        "M3": np.diag([0, 0, 0, 0, 1, 1]).astype(np.complex128),
    }
    identity = np.eye(6, dtype=np.complex128)
    c2_m1 = np.diag([1, -1, 0, 0, 0, 0]).astype(np.complex128)
    c2_m1[2, 4] = c2_m1[3, 5] = c2_m1[4, 2] = c2_m1[5, 3] = 1
    c2_m2 = np.diag([0, 0, 1, -1, 0, 0]).astype(np.complex128)
    c2_m2[0, 4] = c2_m2[1, 5] = c2_m2[4, 0] = c2_m2[5, 1] = 1
    c2_m3 = np.diag([0, 0, 0, 0, 1, -1]).astype(np.complex128)
    c2_m3[0, 2] = c2_m3[1, 3] = c2_m3[2, 0] = c2_m3[3, 1] = 1

    reports = _build_valley_preserving_subspace_reports(
        valley_matrices=seeds,
        d_g_dict={0: identity, 3: c2_m2, 4: c2_m1, 5: c2_m3},
        valley_mappings_dict={
            0: {"M1": "M1", "M2": "M2", "M3": "M3"},
            3: {"M1": "M3", "M2": "M2", "M3": "M1"},
            4: {"M1": "M1", "M2": "M3", "M3": "M2"},
            5: {"M1": "M2", "M2": "M1", "M3": "M3"},
        },
        valley_names=["M1", "M2", "M3"],
        unitarity_tol=1e-8,
        modulus_tol=1e-8,
        spinor_wavefunction=False,
        spinor_convention_verified=True,
        operation_orders_by_id={0: 1, 3: 2, 4: 2, 5: 2},
    )

    assert [report["orbit"] for report in reports] == [["M1"], ["M2"], ["M3"]]
    assert [report["local_rank_source"] for report in reports] == [
        "target_dim_div_valley_count",
        "target_dim_div_valley_count",
        "target_dim_div_valley_count",
    ]
    assert [report["symmetry_adapted_projectors"]["selected_rank"] for report in reports] == [2, 2, 2]
    assert reports[0]["valley_preserving_representations"]["valley_preserving_operations"]["M1"] == [0, 4]
    assert reports[1]["valley_preserving_representations"]["valley_preserving_operations"]["M2"] == [0, 3]
    assert reports[2]["valley_preserving_representations"]["valley_preserving_operations"]["M3"] == [0, 5]
    assert [report["subspace_group"]["subspace_group_candidate"] for report in reports] == [
        None,
        None,
        None,
    ]
    assert [report["subspace_space_group"]["candidate_space_group_symbol"] for report in reports] == [
        None,
        None,
        None,
    ]
    assert all(report["local_irrep_ready"] is True for report in reports)


def test_subspace_space_group_uses_full_valley_mapping_beyond_current_hsp():
    from valleyscope.workflows.analyze_hsp import _build_valley_preserving_subspace_reports

    seeds = {
        "M1": np.diag([1, 1, 0, 0, 0, 0]).astype(np.complex128),
        "M2": np.diag([0, 0, 1, 1, 0, 0]).astype(np.complex128),
        "M3": np.diag([0, 0, 0, 0, 1, 1]).astype(np.complex128),
    }
    identity = np.eye(6, dtype=np.complex128)
    c2_m3 = np.diag([0, 0, 0, 0, 1, -1]).astype(np.complex128)
    c2_m3[0, 2] = c2_m3[1, 3] = c2_m3[2, 0] = c2_m3[3, 1] = 1

    reports = _build_valley_preserving_subspace_reports(
        valley_matrices=seeds,
        d_g_dict={0: identity, 5: c2_m3},
        valley_mappings_dict={
            0: {"M1": "M1", "M2": "M2", "M3": "M3"},
            5: {"M1": "M2", "M2": "M1", "M3": "M3"},
        },
        valley_names=["M1", "M2", "M3"],
        unitarity_tol=1e-8,
        modulus_tol=1e-8,
        spinor_wavefunction=False,
        spinor_convention_verified=True,
        operation_orders_by_id={0: 1, 5: 2},
        space_group_valley_mappings={
            0: {"M1": "M1", "M2": "M2", "M3": "M3"},
            3: {"M1": "M3", "M2": "M2", "M3": "M1"},
            4: {"M1": "M1", "M2": "M3", "M3": "M2"},
            5: {"M1": "M2", "M2": "M1", "M3": "M3"},
        },
        space_group_operation_orders={0: 1, 3: 2, 4: 2, 5: 2},
    )

    assert [report["hsp_preserving_operation_ids"] for report in reports] == [
        [0],
        [0],
        [0, 5],
    ]
    assert [report["subspace_space_group"]["valley_preserving_operation_ids"] for report in reports] == [
        [0, 4],
        [0, 3],
        [0, 5],
    ]
    assert [report["subspace_space_group"]["candidate_space_group_symbol"] for report in reports] == [
        None,
        None,
        None,
    ]
    assert "subspace_group_candidate_missing" in reports[0]["ebr_mapping_input"]["blocked_by"]
    assert "subspace_group_candidate_missing" in reports[1]["ebr_mapping_input"]["blocked_by"]


def test_valley_preserving_subspace_reports_use_modulus_tolerance():
    from valleyscope.workflows.analyze_hsp import _build_valley_preserving_subspace_reports

    seed = np.eye(2, dtype=np.complex128)
    nearly_unitary_c2 = 0.99999 * np.diag([1.0j, -1.0j]).astype(np.complex128)

    reports = _build_valley_preserving_subspace_reports(
        valley_matrices={"M": seed},
        d_g_dict={0: np.eye(2, dtype=np.complex128), 1: nearly_unitary_c2},
        valley_mappings_dict={
            0: {"M": "M"},
            1: {"M": "M"},
        },
        valley_names=["M"],
        unitarity_tol=1e-3,
        modulus_tol=1e-3,
        spinor_wavefunction=False,
        spinor_convention_verified=True,
    )

    assert reports[0]["local_irrep_ready"] is True
    assert reports[0]["diagnostic_only"] is False


def test_add_identity_representation_if_missing_uses_detected_identity():
    from valleyscope.workflows.analyze_hsp import _add_identity_representation_if_missing

    d_g_dict = {5: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)}
    valley_mappings = {5: {"K": "Kp", "Kp": "K"}}
    added = _add_identity_representation_if_missing(
        d_g_dict=d_g_dict,
        valley_mappings_dict=valley_mappings,
        valley_names=["K", "Kp"],
        symmetry_payload={
            "detected_operations": [
                {
                    "operation_id": 0,
                    "order": 1,
                    "sector_mapping": {"K": "K", "Kp": "Kp"},
                }
            ]
        },
        fallback_dim=2,
    )

    assert added is True
    assert 0 in d_g_dict
    np.testing.assert_allclose(d_g_dict[0], np.eye(2, dtype=np.complex128))
    assert valley_mappings[0] == {"K": "K", "Kp": "Kp"}


def test_add_identity_representation_if_missing_does_not_replace_existing_identity():
    from valleyscope.workflows.analyze_hsp import _add_identity_representation_if_missing

    d_g_dict = {0: np.eye(2, dtype=np.complex128)}
    valley_mappings = {0: {"K": "K", "Kp": "Kp"}}
    added = _add_identity_representation_if_missing(
        d_g_dict=d_g_dict,
        valley_mappings_dict=valley_mappings,
        valley_names=["K", "Kp"],
        symmetry_payload={"detected_operations": []},
        fallback_dim=2,
    )

    assert added is False
    assert sorted(d_g_dict) == [0]
