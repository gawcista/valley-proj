"""Shared test fixtures for io/workflow tests."""

from pathlib import Path

import h5py
import numpy as np
import yaml


def write_fixture(path: Path):
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
        kp["frac"] = np.array([0.0, 0.0, 0.0])
        kp["cart"] = np.array([0.0, 0.0, 0.0])
        kp["g_vectors_frac"] = np.array([[0, 0, 0], [1, 0, 0]])
        kp["g_vectors_cart"] = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        kp["coefficients"] = np.array([[[1.0 + 0.0j, 0.0 + 0.0j]]])
        kp["energies_eV"] = np.array([0.1])
        kp["band_indices_vasp"] = np.array([101])


def write_fixture_with_lattice(path: Path, direct_cart: np.ndarray):
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.asarray(direct_cart, dtype=float)
        lattice["reciprocal_cart"] = np.eye(3)
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1


def write_config(path: Path, h5_path: Path, out_dir: Path):
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101], "degeneracy_tol_meV": 1.0},
        "monolayer_lattices": {
            "default": {"reciprocal_cart": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]}
        },
        "valley_centers": {
            "coordinate_mode": "cart",
            "centers": [
                {"name": "K", "cart": [0.0, 0.0, 0.0]},
                {"name": "Kp", "cart": [5.0, 0.0, 0.0]},
            ],
        },
        "valley_subspaces": [
            {"name": "K_valley", "centers": ["K"]},
            {"name": "Kp_valley", "centers": ["Kp"]},
        ],
        "projection": {
            "use_2d_momentum_only": True,
            "qcut_mode": "absolute",
            "qcut_Ainv": 0.5,
            "qcut_scan": [0.5],
            "overlap_policy": "warn_exclude",
        },
        "symmetry": {
            "operations": {
                "mode": "auto",
                "structure_file": "CONTCAR",
                "backend": "spglib",
            },
            "tolerance": {
                "symprec": 1.0e-3,
                "angle_tolerance": -1.0,
                "symprec_scan": [1.0e-5, 1.0e-3],
            },
            "filters": {
                "proper_rotations_only": True,
                "allowed_orders": [2, 3, 4, 6],
            },
        },
        "output": {"directory": str(out_dir), "profile": "debug", "write_json": True, "write_csv": True, "write_hdf5_basis_transform": True},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def write_simple_poscar(path: Path):
    path.write_text(
        "simple\n"
        "1.0\n"
        "1.0 0.0 0.0\n"
        "-0.5 0.8660254037844386 0.0\n"
        "0.0 0.0 8.0\n"
        "X\n"
        "1\n"
        "Direct\n"
        "0.0 0.0 0.0\n",
        encoding="utf-8",
    )


def write_square_poscar(path: Path):
    path.write_text(
        "square\n"
        "1.0\n"
        "1.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 8.0\n"
        "X\n"
        "1\n"
        "Direct\n"
        "0.0 0.0 0.0\n",
        encoding="utf-8",
    )

# Shared E2E smoke helpers.
import json as _json

_E2E_SAMPLE_TABLE = {
    "schema_version": "1.0.0",
    "subspace_group_candidate": "C3_like",
    "expected_hsps": ["GammaM", "KM"],
    "irreps": [
        "GammaM:C3_spinor_phase_+1/2",
        "KM:C3_spinor_phase_+1/6",
        "KM:C3_spinor_phase_-1/6",
    ],
    "ebrs": [
        {"label": "EBR_A", "vector": [1, 0, 1]},
        {"label": "EBR_B", "vector": [1, 1, 0]},
    ],
}

def e2e_write_table(path, data):
    path.write_text(_json.dumps(data), encoding="utf-8")
