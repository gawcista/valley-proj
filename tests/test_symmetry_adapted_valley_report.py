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
        "experimental", "workflow_integration_status", "trusted_irrep_label",
        "orbit", "reference_valley",
        "symmetry_adapted_projectors",
        "valley_preserving_representations",
        "valley_sewing_matrices",
        "valley_preserving_character_diagnostics",
    ]:
        assert key in summary, f"missing: {key}"
    assert summary["experimental"] is True
    assert summary["workflow_integration_status"] == "not_integrated"
    assert summary["trusted_irrep_label"] is False
    assert "valley_sewing_matrices_summary" not in json.dumps(
        summary["valley_preserving_representations"]
    )
    assert set(summary["valley_sewing_matrices"]) == {
        "status",
        "max_sewing_unitarity_error",
        "items",
    }


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


def test_default_off_no_symmetry_adapted_valley_output(tmp_path):
    """Default enabled=false: no symmetry_adapted_valley_analysis in outputs."""
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
        "output": {"directory": str(out_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    outputs = analyze_hsp(config_path)

    # Default: no symmetry_adapted_valley_analysis
    assert "symmetry_adapted_valley_analysis_json" not in outputs
    assert not (out_dir / "symmetry_adapted_valley_analysis.json").exists()

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "symmetry_adapted_valley_analysis" not in summary


def test_enabled_true_writes_symmetry_adapted_valley_analysis(tmp_path):
    """enabled=true writes symmetry_adapted_valley_analysis.json."""
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
                      "symmetry_adapted_valley": {"enabled": True}},
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
        "output": {"directory": str(out_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    outputs = analyze_hsp(config_path)

    assert "symmetry_adapted_valley_analysis_json" in outputs
    report_path = outputs["symmetry_adapted_valley_analysis_json"]
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "by_kpoint" in report
    gm = report["by_kpoint"].get("GammaM", {})
    assert "orbits" in gm
    assert gm["experimental"] is True
    assert gm["trusted_irrep_label"] is False

    # Forbidden terms check
    encoded = json.dumps(report)
    for forbidden in ["covariance", "equivariant", "stabilizer", "valley_little_group", "p_cov"]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"


def test_enabled_true_without_d_raw_writes_not_evaluated_report(tmp_path):
    """enabled=true still writes an explicit not_evaluated report if D_raw is absent."""
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
                      "degeneracy_tol_meV": 1.0,
                      "symmetry_adapted_valley": {"enabled": True}},
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
        "output": {"directory": str(out_dir)},
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
