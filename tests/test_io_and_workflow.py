import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.io.h5_reader import read_wavefunction_h5
from valleyscope.geometry.lattice import read_poscar_cell, read_poscar_lattice
from valleyscope.cli import main as cli_main
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
            "qcut_scan": [0.5],
            "overlap_cross_sector": "warn_exclude",
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
        "output": {"directory": str(out_dir), "write_json": True, "write_csv": True, "write_hdf5_basis_transform": True},
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
    assert config.projection.overlap_cross_sector == "warn_exclude"
    assert config.symmetry.operations.mode == "auto"
    assert config.symmetry.operations.structure_file == config_path.parent / "CONTCAR"
    assert config.symmetry.operations.backend == "spglib"
    assert config.symmetry.tolerance.symprec == pytest.approx(1.0e-3)
    assert config.symmetry.tolerance.angle_tolerance == pytest.approx(-1.0)
    assert config.symmetry.tolerance.symprec_scan == [1.0e-5, 1.0e-3]
    assert config.symmetry.filters.proper_rotations_only is True
    assert config.symmetry.filters.allowed_orders == [2, 3, 4, 6]
    assert config.valley_sectors[0].name == "K_sector"


def test_config_loader_accepts_legacy_symmetry_schema_with_deprecation_warning(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "legacy.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["input"]["poscar"] = "legacy-CONTCAR"
    raw["symmetry"] = {
        "source": "spglib",
        "symprec": 3.0e-4,
        "symprec_scan": [1.0e-5, 3.0e-4],
        "angle_tolerance": 0.5,
        "allowed_orders": [3],
        "proper_rotations_only": True,
        "little_group_check": True,
        "valley_preservation_check": True,
    }
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="input.poscar and symmetry.source are deprecated"):
        config = load_config(config_path)

    assert config.symmetry.operations.structure_file == config_path.parent / "legacy-CONTCAR"
    assert config.symmetry.operations.backend == "spglib"
    assert config.symmetry.tolerance.symprec == pytest.approx(3.0e-4)
    assert config.symmetry.tolerance.angle_tolerance == pytest.approx(0.5)
    assert config.symmetry.tolerance.symprec_scan == [1.0e-5, 3.0e-4]
    assert config.symmetry.filters.allowed_orders == [3]


def test_config_loader_prefers_new_symmetry_schema_over_legacy_fields(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "mixed.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["input"]["poscar"] = "legacy-CONTCAR"
    raw["symmetry"]["source"] = "spglib"
    raw["symmetry"]["symprec"] = 1.0e-1
    raw["symmetry"]["allowed_orders"] = [2]
    raw["symmetry"]["operations"]["structure_file"] = "new-CONTCAR"
    raw["symmetry"]["tolerance"]["symprec"] = 2.0e-4
    raw["symmetry"]["filters"]["allowed_orders"] = [3, 6]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="ignored"):
        config = load_config(config_path)

    assert config.symmetry.operations.structure_file == config_path.parent / "new-CONTCAR"
    assert config.symmetry.tolerance.symprec == pytest.approx(2.0e-4)
    assert config.symmetry.filters.allowed_orders == [3, 6]


def test_input_poscar_fallback_for_legacy_symmetry_structure(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "fallback.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["input"]["poscar"] = "fallback-CONTCAR"
    raw.pop("symmetry")
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="input.poscar"):
        config = load_config(config_path)

    assert config.symmetry.operations.structure_file == config_path.parent / "fallback-CONTCAR"


@pytest.mark.parametrize("check_name", ["little_group_check", "valley_preservation_check"])
def test_legacy_false_symmetry_hard_checks_are_rejected(tmp_path, check_name):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "legacy_false_check.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["symmetry"][check_name] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=f"{check_name}.*hard check"):
        load_config(config_path)


def test_config_loader_rejects_unknown_projection_keys(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["projection"]["stale_projection_key"] = "warn_exclude"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported projection keys"):
        load_config(config_path)


def test_poscar_readers_accept_blank_title_line(tmp_path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "\n"
        "1.0\n"
        "2.0 0.0 0.0\n"
        "0.0 3.0 0.0\n"
        "0.0 0.0 4.0\n"
        "Mo Te\n"
        "1 2\n"
        "Direct\n"
        "0.0 0.0 0.0\n"
        "0.5 0.0 0.0\n"
        "0.0 0.5 0.0\n",
        encoding="utf-8",
    )

    lattice = read_poscar_lattice(poscar)
    cell = read_poscar_cell(poscar)

    assert lattice.direct_cart[0].tolist() == [2.0, 0.0, 0.0]
    assert cell[0][1].tolist() == [0.0, 3.0, 0.0]
    assert cell[2].tolist() == [1, 2, 2]


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


def test_config_loader_derives_layer_reciprocal_from_supercell_matrix(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    primitive_direct = np.diag([2.0, 3.0, 5.0])
    supercell = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    moire_direct = supercell.T @ primitive_direct
    write_fixture_with_lattice(h5_path, moire_direct)
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": [], "target_bands_vasp": []},
        "layer_transforms": {
            "top": {
                "supercell_matrix": [[2, 1, 0], [0, 1, 0], [0, 0, 1]],
            },
        },
        "valley_centers": {
            "coordinate_mode": "layer_frac",
            "centers": [
                {"name": "top_K", "layer": "top", "frac": [0.5, 0.0, 0.0]},
            ],
        },
        "valley_sectors": [{"name": "K_sector", "centers": ["top_K"]}],
        "output": {"directory": str(tmp_path / "out")},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    config = load_config(config_path)
    center = config.valley_centers[0]

    np.testing.assert_allclose(
        center.reciprocal_cart,
        [[np.pi, 0.0, 0.0], [0.0, 2.0 * np.pi / 3.0, 0.0], [0.0, 0.0, 2.0 * np.pi / 5.0]],
    )
    assert center.cart == pytest.approx([0.5 * np.pi, 0.0, 0.0])


def test_analyze_hsp_writes_csv_json_and_diagnostics_h5(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    outputs = analyze_hsp(config_path)

    assert outputs["valley_weights_csv"].exists()
    assert outputs["valley_subspace_json"].exists()
    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    assert outputs["diagnostics_h5"].exists()
    report = json.loads(outputs["symmetry_report_json"].read_text(encoding="utf-8"))
    assert report["status"] == "skipped"
    assert "symmetry.operations.structure_file" in report["reason"]
    csv_text = outputs["valley_weights_csv"].read_text(encoding="utf-8")
    csv_header = csv_text.splitlines()[0].split(",")
    assert csv_header == [
        "kpoint",
        "band_vasp",
        "energy_eV",
        "K_sector",
        "Kp_sector",
        "W_val",
        "P_v",
        "eta",
        "W_overlap",
        "W_res",
    ]
    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    weight = subspace["kpoints"]["GammaM"]["weights"][0]
    assert set(weight) == {
        "band_vasp",
        "classification",
        "sector_weights",
        "W_val",
        "P_v",
        "eta",
        "W_overlap",
        "W_res",
    }
    with h5py.File(outputs["diagnostics_h5"], "r") as h5:
        projector_group = h5["projectors"]["GammaM"]
        assert "overlap_mask" in projector_group
        assert "ambiguous_mask" not in projector_group
        scan_group = h5["qcut_scan"]["GammaM"]
        assert "W_overlap" in scan_group
        assert "W_res" in scan_group
        assert "overlap_count" in scan_group
        assert "ambiguous_weight" not in scan_group
        assert "ambiguous_count" not in scan_group
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert {
        "input",
        "valley_projection_summary",
        "valley_adapted_subspace",
        "symmetry_diagnostics",
        "allowed_valley_preserving_rotations",
        "rotation_eigenvalues",
        "warnings",
        "output_files",
        "legend",
    } <= set(summary)


def test_cli_prints_human_readable_summary(tmp_path, capsys):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    assert cli_main(["analyze-hsp", str(config_path)]) == 0

    out = capsys.readouterr().out
    assert "Input" in out
    assert "Valley projection summary" in out
    assert "W_val" in out
    assert "P_v" in out
    assert "W_overlap" in out
    assert "W_res" in out
    assert "Valley manifolds" in out
    assert "K_sector" in out
    assert "qcut mode: absolute" in out
    assert "qcut value" in out


def test_analyze_hsp_writes_symmetry_operation_detection_report(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    structure = tmp_path / "CONTCAR"
    write_fixture(h5_path)
    write_simple_poscar(structure)
    write_config(config_path, h5_path, out_dir)

    outputs = analyze_hsp(config_path)
    report = json.loads(outputs["symmetry_report_json"].read_text(encoding="utf-8"))

    assert report["status"] == "ok"
    assert report["operation_detection_backend"] == "spglib"
    assert report["structure_file"] == str(structure)
    assert report["symprec"] == pytest.approx(1.0e-3)
    assert report["angle_tolerance"] == pytest.approx(-1.0)
    assert "symprec_scan_summary" in report
    assert {"symprec", "spacegroup_number", "international", "n_operations", "n_candidate_rotations", "order_counts"} <= set(
        report["symprec_scan_summary"][0]
    )
    assert "detected_operations" in report
    assert "candidate_rotations" in report
    assert "operations" not in report
    assert report["little_group_check"]["required"] is True
    assert report["valley_preservation_check"]["required"] is True


def test_readme_symmetry_example_uses_parser_schema(tmp_path):
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "symmetry:\n  operations:" in readme
    assert "input:\n  wavefunction_h5: ./wave.h5\n\n  # Moire/bilayer POSCAR" not in readme
    assert "symmetry:\n  source: spglib" not in readme
    assert "Example screen summary" in readme
    assert "Valley projection summary" in readme
    assert "qcut mode:" in readme

    match = re.search(
        r"For a `generate_hexagonal_210\(9, 5, \.\.\.\)` style cell:\n\n```yaml\n(.*?)\n```",
        readme,
        flags=re.DOTALL,
    )
    assert match is not None

    h5_path = tmp_path / "wave.h5"
    mono = tmp_path / "2dm-5370.vasp"
    structure = tmp_path / "2dm-5370-7.34.vasp"
    out_dir = tmp_path / "valley_analysis"
    write_fixture(h5_path)
    write_simple_poscar(mono)
    write_simple_poscar(structure)

    yaml_text = match.group(1)
    yaml_text = yaml_text.replace("./wave.h5", str(h5_path))
    yaml_text = yaml_text.replace("./2dm-5370.vasp", str(mono))
    yaml_text = yaml_text.replace("./2dm-5370-7.34.vasp", str(structure))
    yaml_text = yaml_text.replace("./valley_analysis", str(out_dir))
    config_path = tmp_path / "readme_example.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    config = load_config(config_path)

    assert config.symmetry.operations.structure_file == structure
    assert config.symmetry.operations.backend == "spglib"
    assert config.symmetry.tolerance.symprec == pytest.approx(1.0e-3)
    assert config.symmetry.filters.allowed_orders == [2, 3, 4, 6]


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
    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    diagnostic = subspace["kpoints"]["GammaM"]["valley_adapted_subspace"]
    assert diagnostic["status"] == "two_valley_adapted"
    assert diagnostic["valid_valley_subspace"] is True
    assert diagnostic["s_min"] == pytest.approx(1.0)
    assert diagnostic["s_max"] == pytest.approx(1.0)


def test_rotation_eigenvalues_use_valley_adapted_basis_and_write_diagnostics(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    structure = tmp_path / "CONTCAR"
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3)
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"
        kp["frac"] = np.array([0.0, 0.0, 0.0])
        kp["cart"] = np.array([0.0, 0.0, 0.0])
        q_cart = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        kp["g_vectors_frac"] = q_cart
        kp["g_vectors_cart"] = q_cart
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        k_state = np.array([inv_sqrt2, inv_sqrt2, 0.0, 0.0])
        kp_state = np.array([0.0, 0.0, inv_sqrt2, inv_sqrt2])
        kp["coefficients"] = np.array(
            [
                [[*(inv_sqrt2 * (k_state + kp_state))]],
                [[*(inv_sqrt2 * (k_state - kp_state))]],
            ],
            dtype=np.complex128,
        )
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])
    write_square_poscar(structure)

    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "target_bands_vasp": [101, 102], "degeneracy_tol_meV": 1.0},
        "monolayer_lattices": {
            "default": {"reciprocal_cart": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]}
        },
        "valley_centers": {
            "coordinate_mode": "cart",
            "centers": [
                {"name": "K_plus", "cart": [1.0, 0.0, 0.0]},
                {"name": "K_minus", "cart": [-1.0, 0.0, 0.0]},
                {"name": "Kp_plus", "cart": [0.0, 1.0, 0.0]},
                {"name": "Kp_minus", "cart": [0.0, -1.0, 0.0]},
            ],
        },
        "valley_sectors": [
            {"name": "K_sector", "centers": ["K_plus", "K_minus"]},
            {"name": "Kp_sector", "centers": ["Kp_plus", "Kp_minus"]},
        ],
        "projection": {
            "use_2d_momentum_only": True,
            "qcut_mode": "absolute",
            "qcut_Ainv": 0.25,
            "overlap_cross_sector": "warn_exclude",
            "thresholds": {"W_val_min": 0.8, "P_v_clean": 0.95, "P_v_approx": 0.85},
        },
        "symmetry": {
            "operations": {"mode": "auto", "structure_file": str(structure), "backend": "spglib"},
            "tolerance": {"symprec": 1.0e-5, "angle_tolerance": -1.0},
            "filters": {"proper_rotations_only": True, "allowed_orders": [2, 4]},
        },
        "output": {"directory": str(out_dir), "write_json": True, "write_csv": True, "write_hdf5_basis_transform": True},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    with outputs["rotation_eigenvalues_csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {
        "basis",
        "nearest_root_of_unity",
        "root_deviation",
        "rotation_ready",
        "topology_input_ready",
        "topology_ready",
        "spinor_rotation_applied",
        "spinor_convention_verified",
        "diagnostic_only",
        "D_valley_offdiag_norm",
        "reason",
        "valley_eta",
    } <= set(rows[0])
    assert any(row["basis"] == "valley_adapted" for row in rows)
    assert all(row["basis"] != "raw_vasp_final" for row in rows)

    with h5py.File(outputs["diagnostics_h5"], "r") as h5:
        assert "rotation/GammaM" in h5
        operation_groups = list(h5["rotation/GammaM"].values())
        assert operation_groups
        assert any("D_valley" in group for group in operation_groups)
        assert all("D_raw" in group for group in operation_groups)
        assert all("operation_order" in group for group in operation_groups)
        assert all("rotation_cart" in group for group in operation_groups)
        assert all("translation_cart" in group for group in operation_groups)
        assert all("basis" in group.attrs for group in operation_groups)
        assert all("spinor_rotation_applied" in group for group in operation_groups)
        assert all("spinor_convention_verified" in group for group in operation_groups)
        assert all("rotation_ready" in group for group in operation_groups)
        assert all("topology_input_ready" in group for group in operation_groups)
        assert all("diagnostic_only" in group for group in operation_groups)
        assert all("D_valley_offdiag_norm" in group for group in operation_groups)
        assert all("root_deviation" in group for group in operation_groups)
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "rejected" in summary_text
    assert "not in little group" in summary_text or "not valley preserving" in summary_text
    assert "topology_input_ready" in summary_text
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert any("topology_input_ready" in row for row in summary["rotation_eigenvalues"])


def test_write_detailed_files_false_writes_only_summary_files(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["write_detailed_files"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    assert not (out_dir / "valley_weights.csv").exists()
    assert not (out_dir / "valley_subspace.json").exists()
    assert not (out_dir / "symmetry_report.json").exists()
    assert not (out_dir / "rotation_eigenvalues.csv").exists()
    assert not (out_dir / "diagnostics.h5").exists()


def test_summary_warns_about_skipped_rotation_representation(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={
            "kpoints": {
                "GammaM": {
                    "weights": [],
                    "warnings": [],
                    "valley_adapted_subspace": {"status": "single_band"},
                }
            }
        },
        symmetry_payload={
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "CONTCAR",
            "detected_operation_count": 1,
            "candidate_rotations": [0],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
            "detected_operations": [
                {
                    "operation_id": 0,
                    "kind": "C2",
                    "candidate_rotation": True,
                    "representation_quality": {
                        "GammaM": {"skipped_reason": "unsupported nspinor=3"}
                    },
                }
            ],
        },
        rotation_rows=[],
        output_paths={},
    )

    assert any("unsupported nspinor=3" in warning for warning in summary["warnings"])
    assert "unsupported nspinor=3" in render_summary_text(summary)


def test_summary_marks_spinor_rotation_as_diagnostic_only(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={
            "kpoints": {
                "GammaM": {
                    "weights": [],
                    "warnings": [],
                    "valley_adapted_subspace": {"status": "two_valley_adapted"},
                }
            }
        },
        symmetry_payload={
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "CONTCAR",
            "detected_operation_count": 1,
            "candidate_rotations": [0],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
            "detected_operations": [],
        },
        rotation_rows=[
            {
                "kpoint": "GammaM",
                "operation_id": 0,
                "order": 2,
                "basis": "valley_adapted",
                "state_index": 0,
                "phase_2pi": 0.5,
                "nearest_root_of_unity": "exp(2pii*1/2)",
                "root_deviation": 0.0,
                "rotation_ready": True,
                "topology_input_ready": False,
                "topology_ready": False,
                "spinor_rotation_applied": True,
                "spinor_convention_verified": False,
                "diagnostic_only": True,
                "D_valley_offdiag_norm": 0.0,
            }
        ],
        output_paths={},
    )

    assert any("Spinor rotation is applied" in warning for warning in summary["warnings"])
    text = render_summary_text(summary)
    assert "topology_input_ready" in text
    assert "diagnostic_only" in text
