from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.io.h5_reader import read_wavefunction_h5
from valleyscope.workflows.analyze_hsp import analyze_hsp


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


def write_config(path: Path, h5_path: Path, out_dir: Path):
    config = {
        "input": {"wavefunction_h5": str(h5_path), "poscar": "CONTCAR"},
        "analysis": {"kpoints": ["GammaM"], "target_bands_vasp": [101], "degeneracy_tol_meV": 1.0},
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
        "valley_sectors": [
            {"name": "K_sector", "centers": ["K"]},
            {"name": "Kp_sector", "centers": ["Kp"]},
        ],
        "projection": {
            "use_2d_momentum_only": True,
            "qcut_mode": "absolute",
            "qcut_Ainv": 0.5,
            "ambiguous_cross_sector": "warn_exclude",
        },
        "output": {"directory": str(out_dir), "write_json": True, "write_csv": True, "write_hdf5_basis_transform": True},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_h5_reader_validates_group_schema(tmp_path):
    h5_path = tmp_path / "wf.h5"
    write_fixture(h5_path)

    data = read_wavefunction_h5(h5_path)

    assert data.kpoints[0].name == "GammaM"
    assert data.kpoints[0].coefficients.shape == (1, 1, 2)
    assert data.metadata.spinor is False


def test_config_loader_parses_core_schema(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    config = load_config(config_path)

    assert config.analysis.target_bands_vasp == [101]
    assert config.projection.qcut_mode == "absolute"
    assert config.valley_sectors[0].name == "K_sector"


def test_config_loader_builds_layer_rotated_fractional_valley_centers(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "target_bands_vasp": [101]},
        "monolayer_lattices": {
            "default": {"reciprocal_cart": [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]}
        },
        "layer_transforms": {
            "top": {"rotation_deg": 90.0},
            "bottom": {"rotation_deg": -90.0},
        },
        "valley_centers": {
            "coordinate_mode": "layer_frac",
            "centers": [
                {"name": "top_K", "layer": "top", "frac": [0.5, 0.0, 0.0]},
                {"name": "bottom_K", "layer": "bottom", "frac": [0.5, 0.0, 0.0]},
            ],
        },
        "valley_sectors": [{"name": "K_sector", "centers": ["top_K", "bottom_K"]}],
        "output": {"directory": str(out_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    config = load_config(config_path)
    centers = {center.name: center for center in config.valley_centers}

    assert centers["top_K"].cart == pytest.approx([0.0, 1.0, 0.0])
    assert centers["bottom_K"].cart == pytest.approx([0.0, -1.0, 0.0])
    assert centers["top_K"].reciprocal_cart[0] == pytest.approx([0.0, 2.0, 0.0])


def test_analyze_hsp_writes_csv_json_and_diagnostics_h5(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    outputs = analyze_hsp(config_path)

    assert outputs["valley_weights_csv"].exists()
    assert outputs["valley_subspace_json"].exists()
    assert outputs["diagnostics_h5"].exists()
    assert "K_sector" in outputs["valley_weights_csv"].read_text(encoding="utf-8")


def test_analyze_hsp_writes_two_valley_subspace_transform_for_degenerate_pair(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
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
        kp["frac"] = np.array([0.0, 0.0, 0.0])
        kp["cart"] = np.array([0.0, 0.0, 0.0])
        kp["g_vectors_frac"] = np.array([[0, 0, 0], [1, 0, 0]])
        kp["g_vectors_cart"] = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        kp["coefficients"] = np.array(
            [
                [[inv_sqrt2 + 0.0j, inv_sqrt2 + 0.0j]],
                [[inv_sqrt2 + 0.0j, -inv_sqrt2 + 0.0j]],
            ]
        )
        kp["energies_eV"] = np.array([0.1, 0.1002])
        kp["band_indices_vasp"] = np.array([101, 102])
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["analysis"]["target_bands_vasp"] = [101, 102]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    with h5py.File(outputs["valley_basis_transform_h5"], "r") as h5:
        assert "GammaM" in h5
        assert h5["GammaM/transform"].shape == (2, 2)
