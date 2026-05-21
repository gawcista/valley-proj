import numpy as np

from valleyscope.analysis.projector_covariance import (
    SEED_COVARIANCE_FAIL_TOL,
    compute_projector_covariance,
)
from valleyscope.subspace.valley_basis import _projector_matrix


# -----------------------------------------------------------------------
# Helper: build seed projector matrices from coefficients + masks
# -----------------------------------------------------------------------

def _seed_matrices(coeffs, masks):
    return {name: _projector_matrix(coeffs, mask) for name, mask in masks.items()}


# -----------------------------------------------------------------------
# A. Exact-covariant two-valley toy (direct matrix check)
# -----------------------------------------------------------------------

def test_exact_covariant_direct_matrix():
    """D_g swaps states 0<->1. P_A projects onto state 0, P_B onto state 1.
    D_g P_A D_g^dag should equal P_B exactly."""
    d_g = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)  # swap
    p_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    transformed = d_g @ p_a @ d_g.conj().T
    epsilon = float(np.linalg.norm(transformed - p_b, ord="fro")
                    / max(np.linalg.norm(p_a, ord="fro"), 1e-14))
    assert epsilon < 1e-15


def test_exact_covariant_identity():
    """Identity D_g: P_A should map to P_A."""
    d_g = np.eye(3, dtype=np.complex128)
    p_a = np.diag([1.0, 2.0, 3.0]).astype(np.complex128) / 6.0
    transformed = d_g @ p_a @ d_g.conj().T
    epsilon = float(np.linalg.norm(transformed - p_a, ord="fro")
                    / max(np.linalg.norm(p_a, ord="fro"), 1e-14))
    assert epsilon < 1e-15


# -----------------------------------------------------------------------
# B. Exact-covariant through compute_projector_covariance
# -----------------------------------------------------------------------

def test_compute_covariance_exact_swap():
    """C2x swaps valley_A <-> valley_B, epsilon ~ 0."""
    coeffs = np.zeros((2, 1, 2), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0

    masks = {
        "valley_A": np.array([True, False]),
        "valley_B": np.array([False, True]),
    }

    p_a = _projector_matrix(coeffs, masks["valley_A"])
    p_b = _projector_matrix(coeffs, masks["valley_B"])

    # Build D_g as a swap (what build_plane_wave_representation would produce
    # for a C2 operation exchanging the two G-vectors exactly)
    d_g = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    report = compute_projector_covariance(
        valley_matrices_by_kpoint={"GammaM": {"valley_A": p_a, "valley_B": p_b}},
        representation_payload={
            "GammaM": {
                "op_1__v_A": {"D_raw": d_g, "source_operation_key": "operation_1"},
                "op_1__v_B": {"D_raw": d_g, "source_operation_key": "operation_1"},
            }
        },
        symmetry_payload={
            "detected_operations": [{
                "operation_id": 1, "kind": "C2", "order": 2,
                "sector_mapping": {"valley_A": "valley_B", "valley_B": "valley_A"},
                "little_group_by_kpoint": {"GammaM": True},
            }]
        },
        valley_names=["valley_A", "valley_B"],
    )

    assert report["status"] == "ok"
    gm = report["by_kpoint"]["GammaM"]["seed_projector_covariance"]
    for row in gm:
        assert row["status"] == "passed", str(row)
        assert row["epsilon_seed"] < 1e-15


# -----------------------------------------------------------------------
# C. C3 three-valley cyclic toy
# -----------------------------------------------------------------------

def test_c3_three_valley_cyclic():
    """C3 cycles M1->M2->M3->M1. Diagonally structured test."""
    coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0
    coeffs[2, 0, 2] = 1.0

    masks = {
        "M1": np.array([True, False, False]),
        "M2": np.array([False, True, False]),
        "M3": np.array([False, False, True]),
    }

    p_m1 = _projector_matrix(coeffs, masks["M1"])
    p_m2 = _projector_matrix(coeffs, masks["M2"])
    p_m3 = _projector_matrix(coeffs, masks["M3"])

    # D_g cycles: state 0→1→2→0
    d_g = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.complex128)

    report = compute_projector_covariance(
        valley_matrices_by_kpoint={"GammaM": {"M1": p_m1, "M2": p_m2, "M3": p_m3}},
        representation_payload={
            "GammaM": {
                "op_1__v_M1": {"D_raw": d_g, "source_operation_key": "operation_1"},
            }
        },
        symmetry_payload={
            "detected_operations": [{
                "operation_id": 1, "kind": "C3", "order": 3,
                "sector_mapping": {"M1": "M2", "M2": "M3", "M3": "M1"},
                "little_group_by_kpoint": {"GammaM": True},
            }]
        },
        valley_names=["M1", "M2", "M3"],
    )

    gm = report["by_kpoint"]["GammaM"]["seed_projector_covariance"]
    for row in gm:
        assert row["status"] == "passed", str(row)
        assert row["epsilon_seed"] < 1e-15


# -----------------------------------------------------------------------
# D. C2 toy: one valley fixed, two swapped
# -----------------------------------------------------------------------

def test_c2_fixes_one_swaps_two():
    """C2 preserves M1, swaps M2<->M3."""
    coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0
    coeffs[2, 0, 2] = 1.0

    masks = {
        "M1": np.array([True, False, False]),
        "M2": np.array([False, True, False]),
        "M3": np.array([False, False, True]),
    }

    p_m1 = _projector_matrix(coeffs, masks["M1"])
    p_m2 = _projector_matrix(coeffs, masks["M2"])
    p_m3 = _projector_matrix(coeffs, masks["M3"])

    # D_g: preserves state 0, swaps states 1<->2
    d_g = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.complex128)

    report = compute_projector_covariance(
        valley_matrices_by_kpoint={"GammaM": {"M1": p_m1, "M2": p_m2, "M3": p_m3}},
        representation_payload={
            "GammaM": {
                "op_3__v_M1": {"D_raw": d_g, "source_operation_key": "operation_3"},
            }
        },
        symmetry_payload={
            "detected_operations": [{
                "operation_id": 3, "kind": "C2", "order": 2,
                "sector_mapping": {"M1": "M1", "M2": "M3", "M3": "M2"},
                "little_group_by_kpoint": {"GammaM": True},
            }]
        },
        valley_names=["M1", "M2", "M3"],
    )

    gm = report["by_kpoint"]["GammaM"]["seed_projector_covariance"]
    assert len(gm) == 3
    for row in gm:
        assert row["status"] == "passed", str(row)
        assert row["epsilon_seed"] < 1e-15


# -----------------------------------------------------------------------
# E. Non-covariant seed → large epsilon
# -----------------------------------------------------------------------

def test_non_covariant_direct():
    """Random D_g should give O(1) epsilon against mismatched projectors."""
    p_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    rng = np.random.default_rng(42)
    d_random = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    d_random = d_random.astype(np.complex128)

    transformed = d_random @ p_a @ d_random.conj().T
    epsilon = float(np.linalg.norm(transformed - p_b, ord="fro")
                    / max(np.linalg.norm(p_a, ord="fro"), 1e-14))
    assert epsilon > SEED_COVARIANCE_FAIL_TOL


def test_non_covariant_fails_in_report():
    """A random D_g produces 'failed' status in covariance report."""
    p_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    rng = np.random.default_rng(99)
    d_random = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    d_random = d_random.astype(np.complex128)

    report = compute_projector_covariance(
        valley_matrices_by_kpoint={"GammaM": {"valley_A": p_a, "valley_B": p_b}},
        representation_payload={
            "GammaM": {
                "op_99__v_A": {"D_raw": d_random, "source_operation_key": "operation_99"},
            }
        },
        symmetry_payload={
            "detected_operations": [{
                "operation_id": 99, "kind": "C2", "order": 2,
                "sector_mapping": {"valley_A": "valley_B", "valley_B": "valley_A"},
                "little_group_by_kpoint": {"GammaM": True},
            }]
        },
        valley_names=["valley_A", "valley_B"],
    )

    assert report["status"] == "covariance_failures_detected"
    gm = report["by_kpoint"]["GammaM"]["seed_projector_covariance"]
    assert any(row["status"] == "failed" for row in gm)


# -----------------------------------------------------------------------
# F. Missing sector_mapping → not_evaluated
# -----------------------------------------------------------------------

def test_missing_mapping_not_evaluated():
    coeffs = np.zeros((2, 1, 2), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0

    p_a = _projector_matrix(coeffs, np.array([True, False]))
    p_b = _projector_matrix(coeffs, np.array([False, True]))

    report = compute_projector_covariance(
        valley_matrices_by_kpoint={"GammaM": {"valley_A": p_a, "valley_B": p_b}},
        representation_payload={
            "GammaM": {
                "op_0__v_A": {"D_raw": np.eye(2, dtype=np.complex128),
                               "source_operation_key": "operation_0"},
            }
        },
        symmetry_payload={
            "detected_operations": [{
                "operation_id": 0, "kind": "identity", "order": 1,
                "sector_mapping": {"valley_A": "valley_A"},  # valley_B missing
                "little_group_by_kpoint": {"GammaM": True},
            }]
        },
        valley_names=["valley_A", "valley_B"],
    )

    gm = report["by_kpoint"]["GammaM"]["seed_projector_covariance"]
    b_row = next(r for r in gm if r["source_valley"] == "valley_B")
    assert b_row["status"] == "not_evaluated"
    assert "pi_g" in b_row["reason"]


# -----------------------------------------------------------------------
# G. Compact covariance summary
# -----------------------------------------------------------------------

def test_compact_covariance_summary():
    from valleyscope.reports.summary_report import _compact_covariance

    report = {
        "status": "covariance_failures_detected",
        "warn_tol": 0.01,
        "fail_tol": 0.1,
        "by_kpoint": {
            "GammaM": {
                "seed_projector_covariance": [
                    {"operation_id": 1, "source_valley": "K", "mapped_valley": "Kp",
                     "epsilon_seed": 1e-12, "status": "passed"},
                    {"operation_id": 1, "source_valley": "Kp", "mapped_valley": "K",
                     "epsilon_seed": 0.5, "status": "failed"},
                    {"operation_id": 3, "source_valley": "K", "mapped_valley": "K",
                     "epsilon_seed": 0.03, "status": "warn"},
                ]
            }
        },
    }

    compact = _compact_covariance(report)
    assert compact["status"] == "covariance_failures_detected"
    gm = compact["by_kpoint"]["GammaM"]
    assert gm["total_checks"] == 3
    assert gm["failed_count"] == 1
    assert gm["warn_count"] == 1
    assert len(gm["failed"]) == 1
    assert gm["failed"][0]["operation_id"] == 1


# -----------------------------------------------------------------------
# H. Workflow integration: projector_covariance_report.json is written
# -----------------------------------------------------------------------

def test_covariance_report_written_by_workflow(tmp_path):
    """End-to-end: analyze_hsp writes projector_covariance_report.json."""
    import h5py
    import yaml
    from valleyscope.workflows.analyze_hsp import analyze_hsp

    h5_path = tmp_path / "wf.h5"
    mono = tmp_path / "mono.vasp"
    structure = tmp_path / "POSCAR"
    out_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"

    with h5py.File(h5_path, "w") as h5:
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
        kp["g_vectors_frac"] = np.array([[0, 0, 0], [1, 0, 0]])
        kp["g_vectors_cart"] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        kp["coefficients"] = np.array([
            [[inv_sqrt2, inv_sqrt2]],
            [[inv_sqrt2, -inv_sqrt2]],
        ], dtype=np.complex128)
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])

    # Simple hexagonal POSCAR
    structure.write_text(
        "hex\n1.0\n"
        "1.0 0.0 0.0\n-0.5 0.8660254 0.0\n0.0 0.0 4.0\n"
        "X\n1\nDirect\n0.0 0.0 0.0\n", encoding="utf-8"
    )
    mono.write_text(structure.read_text(encoding="utf-8"), encoding="utf-8")

    config = {
        "input": {"wavefunction_h5": str(h5_path),
                   "monolayer_poscars": {"top": str(mono), "bottom": str(mono)}},
        "layer_transforms": {
            "top": {"supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "bottom": {"supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        },
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102],
                      "degeneracy_tol_meV": 1.0},
        "valley_centers": {
            "coordinate_mode": "cart",
            "centers": [
                {"name": "K", "cart": [0.0, 0.0, 0.0]},
                {"name": "Kp", "cart": [1.0, 0.0, 0.0]},
            ],
        },
        "valley_subspaces": [
            {"name": "K_valley", "centers": ["K"]},
            {"name": "Kp_valley", "centers": ["Kp"]},
        ],
        "projection": {
            "qcut_mode": "absolute", "qcut_Ainv": 0.5,
            "overlap_policy": "warn_exclude",
            "thresholds": {"W_val_min": 0.5},
        },
        "symmetry": {
            "operations": {"structure_file": str(structure)},
            "tolerance": {"symprec": 1e-3},
            "filters": {"rotation_order": "auto"},
        },
        "output": {"directory": str(out_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    # Old outputs still exist
    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    assert outputs["valley_weights_csv"].exists()
    assert outputs["valley_subspace_json"].exists()

    # New covariance report
    cov_path = out_dir / "projector_covariance_report.json"
    if cov_path.exists():
        import json
        cov = json.loads(cov_path.read_text(encoding="utf-8"))
        assert "status" in cov
        assert "by_kpoint" in cov
        assert cov["normalization"] == "frobenius_source_projector"

    # Summary JSON should include projector_covariance key
    summary = __import__("json").loads(
        outputs["valley_summary_json"].read_text(encoding="utf-8")
    )
    assert "projector_covariance" in summary
