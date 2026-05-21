import json
import numpy as np
import yaml
import h5py

from valleyscope.analysis.projector_symmetry import (
    SEED_PROJECTOR_SYMMETRY_FAIL_TOL,
    apply_projector_symmetry_gate,
    build_projector_symmetry_report,
)
from valleyscope.subspace.valley_basis import _projector_matrix


# -----------------------------------------------------------------------
# Helper: seed matrices from coefficients + masks
# -----------------------------------------------------------------------

def _seed_dict(coeffs, masks):
    return {name: _projector_matrix(coeffs, mask) for name, mask in masks.items()}


def _raw_rep_entry(d_raw, sector_mapping, kind="C2", order=2):
    return {
        "D_raw": d_raw,
        "kind": kind,
        "order": order,
        "sector_mapping": dict(sector_mapping),
        "little_group_passed": True,
    }


# -----------------------------------------------------------------------
# A. Exact-symmetry_consistent direct matrix checks
# -----------------------------------------------------------------------

def test_exact_symmetry_consistent_direct_swap():
    d_g = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    p_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    transformed = d_g @ p_a @ d_g.conj().T
    epsilon = float(np.linalg.norm(transformed - p_b, ord="fro")
                    / max(np.linalg.norm(p_a, ord="fro"), 1e-14))
    assert epsilon < 1e-15


def test_exact_symmetry_consistent_direct_identity():
    d_g = np.eye(3, dtype=np.complex128)
    p_a = np.diag([1.0, 2.0, 3.0]).astype(np.complex128) / 6.0
    transformed = d_g @ p_a @ d_g.conj().T
    epsilon = float(np.linalg.norm(transformed - p_a, ord="fro")
                    / max(np.linalg.norm(p_a, ord="fro"), 1e-14))
    assert epsilon < 1e-15


# -----------------------------------------------------------------------
# B. build_projector_symmetry_report with raw_representations_by_kpoint
# -----------------------------------------------------------------------

def test_symmetry_consistency_exact_c2_swap():
    coeffs = np.zeros((2, 1, 2), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0

    masks = {"valley_A": np.array([True, False]),
             "valley_B": np.array([False, True])}
    d_g = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    report = build_projector_symmetry_report(
        valley_matrices_by_kpoint={"GammaM": _seed_dict(coeffs, masks)},
        raw_representations_by_kpoint={
            "GammaM": {
                1: _raw_rep_entry(d_g, {"valley_A": "valley_B", "valley_B": "valley_A"}),
            }
        },
        valley_names=["valley_A", "valley_B"],
    )

    assert report["status"] == "ok"
    gm = report["by_kpoint"]["GammaM"]["seed_projector_symmetry"]
    assert len(gm) == 2  # one per source valley, deduplicated
    for row in gm:
        assert row["status"] == "passed", str(row)
        assert row["epsilon_seed"] < 1e-15


def test_symmetry_consistency_c3_three_valley_cyclic():
    coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0
    coeffs[2, 0, 2] = 1.0

    masks = {"M1": np.array([True, False, False]),
             "M2": np.array([False, True, False]),
             "M3": np.array([False, False, True])}

    d_g = np.array([[0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]], dtype=np.complex128)

    report = build_projector_symmetry_report(
        valley_matrices_by_kpoint={"GammaM": _seed_dict(coeffs, masks)},
        raw_representations_by_kpoint={
            "GammaM": {
                1: _raw_rep_entry(d_g, {"M1": "M2", "M2": "M3", "M3": "M1"},
                                  kind="C3", order=3),
            }
        },
        valley_names=["M1", "M2", "M3"],
    )

    gm = report["by_kpoint"]["GammaM"]["seed_projector_symmetry"]
    assert len(gm) == 3
    for row in gm:
        assert row["status"] == "passed", str(row)
        assert row["epsilon_seed"] < 1e-15


# -----------------------------------------------------------------------
# C. C2 fixes one valley, swaps the other two
# -----------------------------------------------------------------------

def test_symmetry_consistency_c2_fixes_one_swaps_two():
    coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0
    coeffs[2, 0, 2] = 1.0

    masks = {"M1": np.array([True, False, False]),
             "M2": np.array([False, True, False]),
             "M3": np.array([False, False, True])}

    d_g = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0]], dtype=np.complex128)

    report = build_projector_symmetry_report(
        valley_matrices_by_kpoint={"GammaM": _seed_dict(coeffs, masks)},
        raw_representations_by_kpoint={
            "GammaM": {
                3: _raw_rep_entry(d_g, {"M1": "M1", "M2": "M3", "M3": "M2"}),
            }
        },
        valley_names=["M1", "M2", "M3"],
    )

    gm = report["by_kpoint"]["GammaM"]["seed_projector_symmetry"]
    assert len(gm) == 3
    for row in gm:
        assert row["status"] == "passed", str(row)
        assert row["epsilon_seed"] < 1e-15


# -----------------------------------------------------------------------
# D. Non-symmetry_consistent seed → large epsilon / failed status
# -----------------------------------------------------------------------

def test_non_symmetry_consistent_direct():
    p_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    rng = np.random.default_rng(42)
    d_random = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    d_random = d_random.astype(np.complex128)
    transformed = d_random @ p_a @ d_random.conj().T
    epsilon = float(np.linalg.norm(transformed - p_b, ord="fro")
                    / max(np.linalg.norm(p_a, ord="fro"), 1e-14))
    assert epsilon > SEED_PROJECTOR_SYMMETRY_FAIL_TOL


def test_non_symmetry_consistent_fails_in_report():
    p_a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p_b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    rng = np.random.default_rng(99)
    d_random = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    d_random = d_random.astype(np.complex128)

    report = build_projector_symmetry_report(
        valley_matrices_by_kpoint={"GammaM": {"valley_A": p_a, "valley_B": p_b}},
        raw_representations_by_kpoint={
            "GammaM": {
                99: _raw_rep_entry(d_random, {"valley_A": "valley_B", "valley_B": "valley_A"}),
            }
        },
        valley_names=["valley_A", "valley_B"],
    )

    assert report["status"] == "symmetry_consistency_failures_detected"
    gm = report["by_kpoint"]["GammaM"]["seed_projector_symmetry"]
    assert any(row["status"] == "failed" for row in gm)


# -----------------------------------------------------------------------
# E. Missing sector_mapping → not_evaluated
# -----------------------------------------------------------------------

def test_missing_mapping_not_evaluated():
    coeffs = np.zeros((2, 1, 2), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0

    p_a = _projector_matrix(coeffs, np.array([True, False]))
    p_b = _projector_matrix(coeffs, np.array([False, True]))

    report = build_projector_symmetry_report(
        valley_matrices_by_kpoint={"GammaM": {"valley_A": p_a, "valley_B": p_b}},
        raw_representations_by_kpoint={
            "GammaM": {
                0: _raw_rep_entry(np.eye(2, dtype=np.complex128),
                                  {"valley_A": "valley_A"}, kind="identity", order=1),
            }
        },
        valley_names=["valley_A", "valley_B"],
    )

    gm = report["by_kpoint"]["GammaM"]["seed_projector_symmetry"]
    b_row = next(r for r in gm if r["source_valley"] == "valley_B")
    assert b_row["status"] == "not_evaluated"
    assert "pi_g" in b_row["reason"]


def test_missing_raw_representation_not_evaluated():
    report = build_projector_symmetry_report(
        valley_matrices_by_kpoint={
            "GammaM": {
                "K_valley": np.eye(2, dtype=np.complex128),
            }
        },
        raw_representations_by_kpoint={
            "GammaM": {
                3: {
                    "D_raw": None,
                    "sector_mapping": {"K_valley": "K_valley"},
                    "little_group_passed": True,
                    "skipped_reason": "plane-wave mapping_miss_count=2",
                }
            }
        },
        valley_names=["K_valley"],
    )

    assert report["status"] == "partial"
    rows = report["by_kpoint"]["GammaM"]["seed_projector_symmetry"]
    assert rows == [
        {
            "operation_id": 3,
            "source_valley": "K_valley",
            "mapped_valley": "K_valley",
            "epsilon_seed": None,
            "seed_projector_symmetry_error": None,
            "little_group_passed": True,
            "seed_projector_symmetry_status": "not_evaluated",
            "status": "not_evaluated",
            "reason": "plane-wave mapping_miss_count=2",
        }
    ]


def test_symmetry_consistency_failure_demotes_symmetry_rows():
    report = {
        "status": "symmetry_consistency_failures_detected",
        "by_kpoint": {
            "GammaM": {
                "seed_projector_symmetry": [
                    {
                        "operation_id": 3,
                        "source_valley": "K_valley",
                        "mapped_valley": "K_valley",
                        "epsilon_seed": 0.5,
                        "status": "failed",
                        "reason": "",
                    }
                ]
            }
        },
    }
    rows = [
        {
            "kpoint": "GammaM",
            "operation_id": 3,
            "target_valley": "K_valley",
            "little_group_passed": True,
            "valley_preserving": True,
            "topology_input_ready": True,
            "topology_ready": True,
            "diagnostic_only": False,
            "reason": "",
        }
    ]

    apply_projector_symmetry_gate(symmetry_rows=rows, projector_symmetry_report=report)

    row = rows[0]
    assert row["projector_symmetry_status"] == "failed"
    assert row["epsilon_seed"] == 0.5
    assert row["topology_input_ready"] is False
    assert row["topology_ready"] is False
    assert row["diagnostic_only"] is True
    assert row["reason"] == "seed projector symmetry-consistency failed"


# -----------------------------------------------------------------------
# F. Deduplication: one operation_id → one set of rows, not duplicated
# -----------------------------------------------------------------------

def test_single_operation_not_duplicated_by_valley_count():
    """One operation_id with 3-valley mapping gives exactly 3 rows, not 9."""
    coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0
    coeffs[2, 0, 2] = 1.0

    masks = {"M1": np.array([True, False, False]),
             "M2": np.array([False, True, False]),
             "M3": np.array([False, False, True])}

    d_g = np.eye(3, dtype=np.complex128)

    report = build_projector_symmetry_report(
        valley_matrices_by_kpoint={"GammaM": _seed_dict(coeffs, masks)},
        raw_representations_by_kpoint={
            "GammaM": {
                0: _raw_rep_entry(d_g, {"M1": "M1", "M2": "M2", "M3": "M3"},
                                  kind="identity", order=1),
            }
        },
        valley_names=["M1", "M2", "M3"],
    )

    gm = report["by_kpoint"]["GammaM"]["seed_projector_symmetry"]
    assert len(gm) == 3, f"Expected 3 rows (one per source valley), got {len(gm)}"


# -----------------------------------------------------------------------
# G. Compact summary
# -----------------------------------------------------------------------

def test_compact_projector_symmetry_summary():
    from valleyscope.reports.summary_report import _compact_projector_symmetry

    report = {
        "status": "symmetry_consistency_failures_detected",
        "warn_tol": 0.01,
        "fail_tol": 0.1,
        "by_kpoint": {
            "GammaM": {
                "seed_projector_symmetry": [
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

    compact = _compact_projector_symmetry(report)
    assert compact["status"] == "symmetry_consistency_failures_detected"
    gm = compact["by_kpoint"]["GammaM"]
    assert gm["total_checks"] == 3
    assert gm["failed_count"] == 1
    assert gm["warn_count"] == 1
    assert len(gm["failed"]) == 1
    assert gm["failed"][0]["operation_id"] == 1


# -----------------------------------------------------------------------
# H. Workflow integration tests
# -----------------------------------------------------------------------

def _write_hex_poscar(path):
    path.write_text(
        "hex\n1.0\n"
        "1.0 0.0 0.0\n-0.5 0.8660254 0.0\n0.0 0.0 4.0\n"
        "X\n1\nDirect\n0.0 0.0 0.0\n", encoding="utf-8"
    )


def test_c3_valley_permuting_c3_appears_in_symmetry_consistency_report(tmp_path):
    """C3 cycling M1/M2/M3 must appear in projector_symmetry_report.json
    even though it is valley-permuting and does not enter valley-preserving irrep."""
    from valleyscope.workflows.analyze_hsp import analyze_hsp

    h5_path = tmp_path / "wf.h5"
    mono = tmp_path / "mono.vasp"
    structure = tmp_path / "POSCAR"
    out_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"

    _write_hex_poscar(structure)
    _write_hex_poscar(mono)

    # 3 states, each at a different G-vector simulating 3 M valleys
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.array(
            [[1.0, 0.0, 0.0], [-0.5, 0.8660254037844386, 0.0], [0.0, 0.0, 8.0]]
        )
        lattice["reciprocal_cart"] = np.linalg.inv(lattice["direct_cart"][()]).T * 2*np.pi
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"
        kp["frac"] = np.zeros(3)
        kp["cart"] = np.zeros(3)
        # 3 G-vectors at M-like positions
        q_cart = np.array([
            [0.5, 0.0, 0.0],
            [-0.25, 0.4330127018922193, 0.0],
            [-0.25, -0.4330127018922193, 0.0],
        ])
        kp["g_vectors_frac"] = np.zeros((3, 3))  # placeholder
        kp["g_vectors_cart"] = q_cart
        coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
        coeffs[0, 0, 0] = 1.0
        coeffs[1, 0, 1] = 1.0
        coeffs[2, 0, 2] = 1.0
        kp["coefficients"] = coeffs
        kp["energies_eV"] = np.array([0.1, 0.1001, 0.1002])
        kp["band_indices_vasp"] = np.array([101, 102, 103])

    config = {
        "input": {"wavefunction_h5": str(h5_path),
                   "monolayer_poscars": {"top": str(mono), "bottom": str(mono)}},
        "layer_transforms": {
            "top": {"supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "bottom": {"supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        },
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102, 103],
                      "degeneracy_tol_meV": 1.0},
        "valley_centers": {
            "coordinate_mode": "cart",
            "centers": [
                {"name": "M1c", "cart": [0.5, 0.0, 0.0]},
                {"name": "M2c", "cart": [-0.25, 0.4330127018922193, 0.0]},
                {"name": "M3c", "cart": [-0.25, -0.4330127018922193, 0.0]},
            ],
        },
        "valley_subspaces": [
            {"name": "M1_valley", "centers": ["M1c"]},
            {"name": "M2_valley", "centers": ["M2c"]},
            {"name": "M3_valley", "centers": ["M3c"]},
        ],
        "projection": {
            "qcut_mode": "absolute", "qcut_Ainv": 0.3,
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

    analyze_hsp(config_path)

    cov_path = out_dir / "projector_symmetry_report.json"
    assert cov_path.exists(), "projector_symmetry_report.json should be written"
    cov = json.loads(cov_path.read_text(encoding="utf-8"))
    gm = cov["by_kpoint"]["GammaM"]["seed_projector_symmetry"]
    op_ids = {row["operation_id"] for row in gm}
    # C3 should be present even if it is valley-permuting
    assert len(op_ids) >= 1, f"Expected at least one operation, got {op_ids}"


def test_summary_json_exposes_symmetry_consistency_failure_flag(tmp_path):
    """valley_summary.json must include 'projector_symmetry' with status info."""
    from valleyscope.workflows.analyze_hsp import analyze_hsp

    h5_path = tmp_path / "wf.h5"
    mono = tmp_path / "mono.vasp"
    structure = tmp_path / "POSCAR"
    out_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"

    _write_hex_poscar(structure)
    _write_hex_poscar(mono)

    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.array(
            [[1.0, 0.0, 0.0], [-0.5, 0.8660254037844386, 0.0], [0.0, 0.0, 8.0]]
        )
        lattice["reciprocal_cart"] = np.linalg.inv(lattice["direct_cart"][()]).T * 2*np.pi
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"
        kp["frac"] = np.zeros(3)
        kp["cart"] = np.zeros(3)
        q_cart = np.array([
            [0.5, 0.0, 0.0],
            [-0.25, 0.4330127018922193, 0.0],
            [-0.25, -0.4330127018922193, 0.0],
        ])
        kp["g_vectors_frac"] = np.zeros((3, 3))
        kp["g_vectors_cart"] = q_cart
        coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
        coeffs[0, 0, 0] = 1.0
        coeffs[1, 0, 1] = 1.0
        coeffs[2, 0, 2] = 1.0
        kp["coefficients"] = coeffs
        kp["energies_eV"] = np.array([0.1, 0.1001, 0.1002])
        kp["band_indices_vasp"] = np.array([101, 102, 103])

    config = {
        "input": {"wavefunction_h5": str(h5_path),
                   "monolayer_poscars": {"top": str(mono), "bottom": str(mono)}},
        "layer_transforms": {
            "top": {"supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "bottom": {"supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        },
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102, 103],
                      "degeneracy_tol_meV": 1.0},
        "valley_centers": {
            "coordinate_mode": "cart",
            "centers": [
                {"name": "M1c", "cart": [0.5, 0.0, 0.0]},
                {"name": "M2c", "cart": [-0.25, 0.4330127018922193, 0.0]},
                {"name": "M3c", "cart": [-0.25, -0.4330127018922193, 0.0]},
            ],
        },
        "valley_subspaces": [
            {"name": "M1_valley", "centers": ["M1c"]},
            {"name": "M2_valley", "centers": ["M2c"]},
            {"name": "M3_valley", "centers": ["M3c"]},
        ],
        "projection": {
            "qcut_mode": "absolute", "qcut_Ainv": 0.3,
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

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "projector_symmetry_report_json" in outputs
    assert "projector_covariance_report_json" not in outputs
    assert "projector_symmetry" in summary
    assert "projector_covariance" not in summary
    assert "projector_equivariance" not in summary
    encoded = json.dumps(summary)
    assert "seed_projector_covariance" not in encoded
    assert "valley_stabilizer" not in encoded
    assert "stabilizer_operations" not in encoded
    projector_symmetry = summary["projector_symmetry"]
    assert "status" in projector_symmetry
    assert "by_kpoint" in projector_symmetry
