"""Parent-valley reporting/readiness boundary tests."""

import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from valleyscope.workflows.analyze_hsp import analyze_hsp


# ---------------------------------------------------------------------------
# helpers shared within this file
# ---------------------------------------------------------------------------

def _make_toy_wf_with_far_k(path: Path):
    """Write toy HDF5 with k at (2.0,0,0) - far from centers at (0,0) and (4,0).

    Moire reciprocal differs from monolayer reciprocal so that dynamic
    centers for V0 and V0p map to distinct torus positions.
    """
    moire_recip = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = moire_recip
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"
        kp["frac"] = np.array([0.5, 0.0, 0.0])
        kp["cart"] = np.array([2.0, 0.0, 0.0])
        kp["g_vectors_frac"] = np.array([[0, 0, 0], [0, 1, 0]])
        kp["g_vectors_cart"] = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        # Two bands: one fully at G=0, one split
        kp["coefficients"] = np.array([
            [[1.0 + 0.0j, 0.0 + 0.0j]],
            [[0.0 + 0.0j, 1.0 + 0.0j]],
        ])
        kp["energies_eV"] = np.array([0.1, 0.1005])
        kp["band_indices_vasp"] = np.array([101, 102])


def _make_toy_config(path: Path, h5_path: Path, out_dir: Path, projector_mode: str):
    """Write analyze.yaml with given projector_mode."""
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0},
        "monolayer_lattices": {
            "default": {"reciprocal_cart": [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 1.0]]}
        },
        "valley_centers": {
            "coordinate_mode": "cart",
            "centers": [
                {"name": "V0", "cart": [0.0, 0.0, 0.0]},
                {"name": "V0p", "cart": [4.0, 0.0, 0.0]},
            ],
        },
        "valley_subspaces": [
            {"name": "V0_valley", "centers": ["V0"]},
            {"name": "V0p_valley", "centers": ["V0p"]},
        ],
        "projection": {
            "projector_mode": projector_mode,
            "use_2d_momentum_only": True,
            "qcut_mode": "absolute",
            "qcut_Ainv": 0.3,
            "overlap_policy": "warn_exclude",
        },
        "output": {"directory": str(out_dir), "profile": "debug"},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

def test_parent_valley_mode_changes_reporting_but_not_readiness(tmp_path):
    """k_resolved_parent_valley changes reporting weights but NOT seed matrices."""

    # --- fixed_center run ---
    h5_fc = tmp_path / "wf_fc.h5"
    cfg_fc = tmp_path / "cfg_fc.yaml"
    out_fc = tmp_path / "out_fc"
    _make_toy_wf_with_far_k(h5_fc)
    _make_toy_config(cfg_fc, h5_fc, out_fc, "fixed_center")
    analyze_hsp(cfg_fc)

    # --- k_resolved_parent_valley run ---
    h5_pv = tmp_path / "wf_pv.h5"
    cfg_pv = tmp_path / "cfg_pv.yaml"
    out_pv = tmp_path / "out_pv"
    _make_toy_wf_with_far_k(h5_pv)
    _make_toy_config(cfg_pv, h5_pv, out_pv, "k_resolved_parent_valley")
    analyze_hsp(cfg_pv)

    # Load both outputs
    fc_sub = json.loads((out_fc / "valley_subspace.json").read_text())
    pv_sub = json.loads((out_pv / "valley_subspace.json").read_text())
    fc_sum = json.loads((out_fc / "valley_summary.json").read_text())
    pv_sum = json.loads((out_pv / "valley_summary.json").read_text())

    # 1. Reporting: weights differ between modes
    fc_w = fc_sub["kpoints"]["GammaM"]["weights"]
    pv_w = pv_sub["kpoints"]["GammaM"]["weights"]
    # fixed_center: k at (2.5,0,0) far from both V0(0,0) and V0p(5,0) - all mask empty
    assert fc_w[0]["W_val"] == 0.0
    assert fc_w[1]["W_val"] == 0.0
    # k_resolved_parent_valley: dynamic centers bring V0 and V0p near k
    assert pv_w[0]["W_val"] > 0.0 or pv_w[1]["W_val"] > 0.0, (
        "k_resolved_parent_valley should recover non-zero parent-valley weight"
    )
    # Center weights should differ
    assert fc_w[0].get("center_weights", {}) != pv_w[0].get("center_weights", {})

    # 2. Readiness: seed subspace diagnostics are identical
    fc_adapted = fc_sub["kpoints"]["GammaM"].get("valley_adapted_subspace", {})
    pv_adapted = pv_sub["kpoints"]["GammaM"].get("valley_adapted_subspace", {})
    # s_eigenvalues (target-valley-subspace score) match
    fc_s = fc_adapted.get("s_eigenvalues")
    pv_s = pv_adapted.get("s_eigenvalues")
    assert fc_s is not None and pv_s is not None
    assert fc_s == pv_s, f"s_eigenvalues differ: fixed_center={fc_s}, parent_valley={pv_s}"
    # s_min matches
    assert fc_adapted.get("s_min") == pv_adapted.get("s_min")
    # valley_concentration matches
    assert fc_adapted.get("valley_concentration") == pv_adapted.get("valley_concentration")

    # 3. Summary projector_mode is correctly recorded
    assert fc_sum["qcut"]["projector_mode"] == "fixed_center"
    assert pv_sum["qcut"]["projector_mode"] == "k_resolved_parent_valley"

    # 4. fixed_center mode status should be fixed_center_not_captured
    fc_statuses = [w["valley_status"] for w in fc_sub["kpoints"]["GammaM"]["weights"]]
    assert all(s == "fixed_center_not_captured" for s in fc_statuses), (
        f"fixed_center low-W_val should be fixed_center_not_captured, got {fc_statuses}"
    )
