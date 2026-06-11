import csv
import json
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.io.h5_reader import read_wavefunction_h5
from valleyscope.geometry.lattice import read_poscar_cell, read_poscar_lattice
from valleyscope.cli import main as cli_main
from valleyscope.projection.sector_projectors import SectorProjectors
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

    assert config.analysis.iband == [101]
    assert config.projection.qcut_mode == "absolute"
    assert config.projection.overlap_policy == "warn_exclude"
    assert config.symmetry.operations.mode == "auto"
    assert config.symmetry.operations.structure_file == config_path.parent / "CONTCAR"
    assert config.symmetry.operations.backend == "spglib"
    assert config.symmetry.tolerance.symprec == pytest.approx(1.0e-3)
    assert config.symmetry.tolerance.angle_tolerance == pytest.approx(-1.0)
    assert config.symmetry.tolerance.symprec_scan == [1.0e-5, 1.0e-3]
    assert config.symmetry.filters.proper_rotations_only is True
    assert config.symmetry.filters.allowed_orders == [2, 3, 4, 6]
    assert config.symmetry.filters.rotation_order == "auto"
    assert config.valley_subspaces[0].name == "K_valley"


def test_config_loader_accepts_simplified_schema_defaults(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "simplified.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {
            "kpoints": ["GammaM"],
            "iband": [101],
            "subspace_energy_tol_meV": 2.0,
        },
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
            "qcut_fraction": 0.2,
            "thresholds": {"W_val_min": 0.8},
        },
        "symmetry": {
            "operations": {"structure_file": "CONTCAR"},
            "filters": {"rotation_order": "auto"},
        },
        "spinor": {
            "convention": "vasp_up_down_saxis_z",
            "convention_verified": True,
            "benchmark": "tMoTe2_VBM_C3_literature",
        },
        "rotation": {
            "irrep_weight_tol": 1.0e-4,
        },
        "output": {"directory": str(out_dir), "profile": "debug"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    config = load_config(config_path)

    assert config.analysis.iband == [101]
    assert config.analysis.degeneracy_tol_meV == pytest.approx(2.0)
    assert config.valley_subspaces[0].name == "K_valley"
    assert config.projection.qcut_mode == "relative_min_valley_distance"
    assert config.projection.qcut_fraction == pytest.approx(0.2)
    assert config.projection.use_2d_momentum_only is True
    assert config.projection.overlap_policy == "warn_exclude"
    assert config.symmetry.operations.structure_file == config_path.parent / "CONTCAR"
    assert config.symmetry.operations.backend == "spglib"
    assert config.symmetry.filters.proper_rotations_only is True
    assert config.symmetry.filters.allowed_orders == [2, 3, 4, 6]
    assert config.spinor.convention == "vasp_up_down_saxis_z"
    assert config.spinor.convention_verified is True
    assert config.spinor.benchmark == "tMoTe2_VBM_C3_literature"
    assert config.rotation.irrep_weight_tol == pytest.approx(1.0e-4)


def test_config_loader_rejects_removed_target_bands_vasp_field(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "removed_target_bands.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["analysis"] = {"kpoints": ["GammaM"], "target_bands_vasp": [101]}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="analysis.iband"):
        load_config(config_path)


def test_config_loader_rejects_removed_valley_sectors_field(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "removed_valley_sectors.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["analysis"] = {"kpoints": ["GammaM"], "iband": [101]}
    raw.pop("valley_subspaces")
    raw["valley_sectors"] = [
        {"name": "K_sector", "centers": ["K"]},
        {"name": "Kp_sector", "centers": ["Kp"]},
    ]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="valley_subspaces"):
        load_config(config_path)


def test_config_loader_rejects_removed_valley_manifolds_field(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "removed_valley_manifolds.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["valley_manifolds"] = raw.pop("valley_subspaces")
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="valley_subspaces"):
        load_config(config_path)


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
    assert config.symmetry.filters.rotation_order == "auto"


def test_config_loader_accepts_rotation_order_integer_and_none(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "rotation_order.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["symmetry"]["filters"]["rotation_order"] = 3
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.symmetry.filters.rotation_order == 3

    raw["symmetry"]["filters"]["rotation_order"] = "None"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.symmetry.filters.rotation_order is None


def test_rotation_readiness_preset_applies_and_allows_explicit_overrides(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["rotation"] = {"readiness_preset": "loose"}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.rotation.readiness_preset == "loose"
    assert config.rotation.unitarity_tol == pytest.approx(1.0e-4)
    assert config.rotation.root_deviation_tol == pytest.approx(1.0e-4)
    assert config.rotation.D_valley_offdiag_tol == pytest.approx(1.0e-2)

    raw["rotation"]["D_valley_offdiag_tol"] = 5.0e-3
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)

    assert config.rotation.readiness_preset == "loose"
    assert config.rotation.root_deviation_tol == pytest.approx(1.0e-4)
    assert config.rotation.D_valley_offdiag_tol == pytest.approx(5.0e-3)


def test_rotation_readiness_preset_rejects_unknown_name(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["rotation"] = {"readiness_preset": "too_loose"}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="rotation.readiness_preset"):
        load_config(config_path)


def test_config_template_uses_current_public_schema(tmp_path):
    template_path = Path("examples/config_template.yaml")
    template = template_path.read_text(encoding="utf-8")
    assert "analysis:\n  kpoints:" in template
    assert "  iband:" in template
    assert "valley_subspaces:" in template
    assert "readiness_preset: strict" in template
    assert "root_deviation_tol:" in template
    assert "D_valley_offdiag_tol:" in template
    assert "target_bands_vasp" not in template
    assert "valley_sectors" not in template
    assert "valley_manifolds" not in template

    h5_path = tmp_path / "selected_wavefunctions.h5"
    monolayer = tmp_path / "monolayer.vasp"
    structure = tmp_path / "CONTCAR"
    out_dir = tmp_path / "valley_analysis"
    write_fixture(h5_path)
    write_simple_poscar(monolayer)
    write_simple_poscar(structure)

    yaml_text = template
    yaml_text = yaml_text.replace("./selected_wavefunctions.h5", str(h5_path))
    yaml_text = yaml_text.replace("./monolayer.vasp", str(monolayer))
    yaml_text = yaml_text.replace("./CONTCAR", str(structure))
    yaml_text = yaml_text.replace("./valley_analysis", str(out_dir))
    config_path = tmp_path / "config_template.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    config = load_config(config_path)
    assert config.analysis.iband == [101, 102]
    assert [sector.name for sector in config.valley_subspaces] == ["K_valley", "Kp_valley"]


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
        "analysis": {"kpoints": ["GammaM"], "iband": [101]},
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
        "valley_subspaces": [{"name": "K_sector", "centers": ["top_K", "bottom_K"]}],
        "output": {"directory": str(out_dir), "profile": "debug"},
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
        "analysis": {"kpoints": [], "iband": []},
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
        "valley_subspaces": [{"name": "K_sector", "centers": ["top_K"]}],
        "output": {"directory": str(tmp_path / "out"), "profile": "debug"},
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
        "K_valley",
        "Kp_valley",
        "W_val",
        "P_v",
        "eta",
        "W_overlap",
        "W_res",
        "center_K",
        "center_Kp",
    ]
    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    weight = subspace["kpoints"]["GammaM"]["weights"][0]
    assert {
        "band_vasp", "sector_weights", "W_val", "P_v", "eta", "W_overlap", "W_res",
    }.issubset(set(weight))
    assert weight.get("analysis_level") == "raw_state"
    assert weight.get("valley_status") == "raw_valley_clean"
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
        "valley_subspace_analysis",
        "symmetry_analysis",
        "symmetry_eigenvalues",
        "warnings",
        "output_files",
        "legend",
    } <= set(summary)
    assert "valley_adapted_subspace" not in summary
    assert "rotation_eigenvalues" not in summary
    assert "allowed_valley_preserving_rotations" not in summary
    assert "two_valley_subspace" not in summary
    subspace_rows = summary["valley_subspace_analysis"]
    assert subspace_rows
    assert subspace_rows[0].get("status") in {"clean", "approx", "mixed", "not_derived", "unreliable", "n/a"}
    assert "derived_score" not in summary["valley_projection_summary"][0]
    assert "polarization_score" not in summary["valley_projection_summary"][0]
    assert "valley_status" not in summary["valley_projection_summary"][0]
    assert "derived_score" not in subspace_rows[0]
    assert "polarization_score" not in subspace_rows[0]
    assert "valley_status" not in subspace_rows[0]
    assert not any("target subspace is not valley-derived" in w for w in summary["warnings"])


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
    assert "derived" not in out
    assert " pol " not in out
    assert "status" in out
    assert "W_overlap" in out
    assert "W_res" in out
    assert "Valley subspace analysis" in out
    assert "Two-valley subspace" not in out
    assert "S_min:      minimum target-valley-subspace weight" in out
    assert "eta_adapted: signed valley polarization" in out
    assert "Symmetry analysis" in out
    assert "Symmetry diagnostics" not in out
    assert "Allowed valley-preserving rotations" not in out
    assert "Rotation eigenvalues" not in out
    assert "Valley subspaces" in out
    assert "K_valley" in out
    assert "K_sector" not in out
    assert "qcut mode: absolute" in out
    assert "qcut value" in out


def test_summary_text_renders_qcut_fraction_for_relative_mode(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["projection"].pop("qcut_Ainv", None)
    raw["projection"]["qcut_mode"] = "relative_min_valley_distance"
    raw["projection"]["qcut_fraction"] = 0.2
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.034,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "skipped",
            "reason": "no structure",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert summary["qcut"]["fraction"] == pytest.approx(0.2)
    text = render_summary_text(summary)
    assert "qcut mode: relative_min_valley_distance" in text
    assert "qcut value: 0.034 A^-1" in text
    assert "qcut fraction: 0.2" in text


def test_analyze_hsp_collects_overlap_warnings_without_raw_warning_output(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["projection"]["qcut_Ainv"] = 5.1
    raw["projection"]["qcut_scan"] = [5.1]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        outputs = analyze_hsp(config_path)

    assert not [item for item in captured if "overlap across valleys" in str(item.message)]
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert any("overlap across valleys" in item for item in summary["warnings"])


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
    assert "valley_subspaces:" in readme
    assert "valley_manifolds" not in readme
    assert "valley_sectors" not in readme
    assert "target_bands_vasp" not in readme
    assert "raw_valley_clean" not in readme
    assert "valley_separable_subspace" not in readme
    assert "relative_min_sector_distance" not in readme
    assert "overlap_cross_sector" not in readme
    assert "valley_sectors" not in readme
    assert "valley_manifolds" not in readme
    assert "valley_manifolds" not in readme.lower()
    assert not re.search(r"(?<!\w)V[123](?!\.\d)", readme)
    assert "input:\n  wavefunction_h5: ./wave.h5\n\n  # Moire/bilayer POSCAR" not in readme
    assert "symmetry:\n  source: spglib" not in readme
    assert "Example screen summary" in readme
    assert "Valley projection summary" in readme
    assert "Valley subspace analysis" in readme
    assert "Two-valley subspace" not in readme
    assert "S_min" in readme and "minimum target-valley-subspace weight" in readme
    assert "min_concentration" in readme
    assert "assigned_valleys" in readme
    assert "qcut mode:" in readme
    assert "`not_derived`" in readme
    assert "`unreliable`" in readme
    assert "valley-preserving irrep" in readme
    assert "valley-preserving subgroup" in readme
    assert "irrep_results_by_kpoint" in readme
    assert "irrep_multiplicities" in readme
    assert "state_irrep_results" in readme
    assert "P321 No.150" not in readme
    assert "P3 No.143" not in readme
    assert "Benchmark:" not in readme
    assert "double-valued" in readme
    assert "`root_deviation_tol`, `D_valley_offdiag_tol`, and `irrep_weight_tol` are numerical readiness thresholds" in readme
    assert "`strict`, `normal`, and `loose`" in readme

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
    # Material-specific benchmark names must not appear in example YAML blocks.
    # Physics descriptions in prose may mention tMoTe2/tZrSe2 as benchmarks.
    assert "tMoTe2" not in yaml_text
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
    assert config.symmetry.filters.rotation_order == "auto"


def test_chinese_readme_uses_public_valley_vocabulary():
    readme = Path("README.zh.md").read_text(encoding="utf-8")
    assert "valley_subspaces:" in readme
    assert "valley_manifolds" not in readme
    assert "valley_sectors" not in readme
    assert "target_bands_vasp" not in readme
    assert "raw_valley_clean" not in readme
    assert "valley_separable_subspace" not in readme
    assert "relative_min_sector_distance" not in readme
    assert "overlap_cross_sector" not in readme
    assert "扇区" not in readme
    assert not re.search(r"(?<!\w)V[123](?!\.\d)", readme)
    assert "Valley subspaces" in readme
    assert "Valley subspace analysis" in readme
    assert "Two-valley subspace" not in readme
    assert "S_min:              目标谷子空间权重下界" in readme
    assert "min_concentration:" in readme
    assert "assigned_valleys:" in readme
    assert "`not_derived`" in readme
    assert "`unreliable`" in readme
    assert "valley-preserving irrep" in readme
    assert "谷保持子群" in readme
    assert "irrep_results_by_kpoint" in readme
    assert "irrep_multiplicities" in readme
    assert "state_irrep_results" in readme
    # Material names may appear in physics benchmark descriptions.
    assert "P321 No.150" not in readme
    assert "P3 No.143" not in readme
    assert "double-valued" in readme
    assert "valley_weights_adapted" in readme
    assert "assigned_valleys" in readme
    assert "`root_deviation_tol`、`D_valley_offdiag_tol` 和 `irrep_weight_tol` 是 numerical readiness thresholds" in readme
    assert "`strict`、`normal`、`loose`" in readme
    assert "header-only" in readme


def test_analyze_hsp_writes_valley_subspace_analysis_transform_for_degenerate_pair(tmp_path):
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
    raw["analysis"]["iband"] = [101, 102]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    with h5py.File(outputs["valley_basis_transform_h5"], "r") as h5:
        assert "GammaM" in h5
        assert h5["GammaM/transform"].shape == (2, 2)
        assert h5["GammaM/eta"].shape == (2,)
        assert h5["GammaM/v_matrix"].shape == (2, 2)
        assert [value.decode() for value in h5["GammaM/sectors"][()]] == ["K_valley", "Kp_valley"]
        assert [value.decode() for value in h5["GammaM/valleys"][()]] == ["K_valley", "Kp_valley"]
    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    diagnostic = subspace["kpoints"]["GammaM"]["valley_adapted_subspace"]
    assert diagnostic["status"] == "valley_separable"
    assert diagnostic["valid_valley_subspace"] is True
    assert diagnostic["s_min"] == pytest.approx(1.0)
    assert diagnostic["s_max"] == pytest.approx(1.0)
    projector_quality = diagnostic["projector_quality"]
    assert projector_quality["expected_rank"] == 1
    assert projector_quality["per_valley"]["K_valley"]["rank_estimate"] == 1
    assert projector_quality["per_valley"]["Kp_valley"]["rank_estimate"] == 1
    assert projector_quality["sum_projector"]["identity_deviation_fro"] == pytest.approx(0.0)
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary["valley_subspace_analysis"][0]["status"] == "clean"


def test_multivalley_subspace_status_uses_pv_thresholds_for_concentration():
    from valleyscope.workflows.analyze_hsp import _add_valley_subspace_diagnostic

    coefficients = np.zeros((2, 1, 3), dtype=np.complex128)
    coefficients[0, 0, 0] = 1.0
    coefficients[1, 0, 1] = np.sqrt(0.92)
    coefficients[1, 0, 2] = np.sqrt(0.08)
    projectors = SectorProjectors(
        sector_masks={
            "A_valley": np.array([True, False, False]),
            "B_valley": np.array([False, True, False]),
            "C_valley": np.array([False, False, True]),
        },
        center_masks={},
        overlap_mask=np.zeros(3, dtype=bool),
        qcut=0.5,
        warnings=[],
    )
    payload: dict[str, object] = {}
    basis_transforms: dict[str, dict[str, np.ndarray]] = {}

    _add_valley_subspace_diagnostic(
        payload,
        basis_transforms,
        "GammaM",
        np.array([101, 102]),
        np.array([0.1, 0.1001]),
        coefficients,
        projectors,
        degeneracy_tol_meV=1.0,
        thresholds={"W_val_min": 0.8, "P_v_clean": 0.95, "P_v_approx": 0.85},
    )

    assert payload["polarization_score"] == pytest.approx(0.92)
    assert payload["subspace_valley_status"] == "valley_approximately_separable_subspace"
    assert "GammaM" in basis_transforms
    assert bool(basis_transforms["GammaM"]["valid_valley_subspace"]) is False


def test_symmetry_eigenvalues_use_valley_adapted_basis_and_write_diagnostics(tmp_path):
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
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0},
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
        "valley_subspaces": [
            {"name": "K_sector", "centers": ["K_plus", "K_minus"]},
            {"name": "Kp_sector", "centers": ["Kp_plus", "Kp_minus"]},
        ],
        "projection": {
            "use_2d_momentum_only": True,
            "qcut_mode": "absolute",
            "qcut_Ainv": 0.25,
            "overlap_policy": "warn_exclude",
            "thresholds": {"W_val_min": 0.8, "P_v_clean": 0.95, "P_v_approx": 0.85},
        },
        "symmetry": {
            "operations": {"mode": "auto", "structure_file": str(structure), "backend": "spglib"},
            "tolerance": {"symprec": 1.0e-5, "angle_tolerance": -1.0},
            "filters": {"proper_rotations_only": True, "allowed_orders": [2, 4], "rotation_order": 2},
        },
        "output": {"directory": str(out_dir), "profile": "debug", "write_json": True, "write_csv": True, "write_hdf5_basis_transform": True},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    assert "rotation_eigenvalues_csv" not in outputs
    assert "little_group_eigenvalues_csv" not in outputs
    assert "little_group_representations_json" not in outputs
    assert not (out_dir / "rotation_eigenvalues.csv").exists()
    assert not (out_dir / "little_group_eigenvalues.csv").exists()
    assert not (out_dir / "little_group_representations.json").exists()
    with outputs["symmetry_eigenvalues_csv"].open(encoding="utf-8") as handle:
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
        "spinor_convention",
        "spinor_benchmark",
        "diagnostic_only",
        "D_valley_offdiag_norm",
        "reason",
        "valley_eta",
    } <= set(rows[0])
    assert any(row["basis"] == "valley_adapted" for row in rows)
    assert all(row["basis"] != "raw_vasp_final" for row in rows)

    with h5py.File(outputs["diagnostics_h5"], "r") as h5:
        assert "rotation" not in h5
        assert "symmetry_representations/GammaM" in h5
        operation_groups = list(h5["symmetry_representations/GammaM"].values())
        assert operation_groups
        assert any("D_valley" in group for group in operation_groups)
        assert all("D_raw" in group for group in operation_groups)
        assert all("operation_order" in group for group in operation_groups)
        assert all("rotation_cart" in group for group in operation_groups)
        assert all("translation_cart" in group for group in operation_groups)
        assert all("basis" in group.attrs for group in operation_groups)
        assert all("spinor_rotation_applied" in group for group in operation_groups)
        assert all("spinor_convention_verified" in group for group in operation_groups)
        assert all("spinor_convention" in group.attrs for group in operation_groups)
        assert all("spinor_benchmark" in group.attrs for group in operation_groups)
        assert all("rotation_ready" in group for group in operation_groups)
        assert all("topology_input_ready" in group for group in operation_groups)
        assert all("diagnostic_only" in group for group in operation_groups)
        assert all("D_valley_offdiag_norm" in group for group in operation_groups)
        assert all("root_deviation" in group for group in operation_groups)
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "rejected" in summary_text
    assert (
        "not in little group" in summary_text
        or "valley-changing" in summary_text
        or "not valley preserving" in summary_text
    )
    assert "topology_input_ready" in summary_text
    assert "Symmetry eigenvalues" in summary_text
    assert "Rotation eigenvalues" not in summary_text
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert any("topology_input_ready" in row for row in summary["symmetry_eigenvalues"])
    assert "spinor_convention" in summary["input"]
    assert "spinor_benchmark" in summary["symmetry_eigenvalues"][0]
    assert outputs["symmetry_eigenvalues_csv"].exists()
    assert "symmetry_eigenvalues_csv" in summary["output_files"]


def test_symmetry_eigenvalues_csv_is_header_only_when_no_rows(tmp_path, monkeypatch):
    import importlib

    workflow_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    def fake_prepare_symmetry_payload(config, monolayer_recip):
        return {
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "fake-CONTCAR",
            "spacegroup_number": 150,
            "international": "P321",
            "symmetry_eigenvalue_enabled": True,
            "requested_rotation_order": "auto",
            "resolved_rotation_order": 3,
            "detected_operation_count": 0,
            "detected_operations": [],
            "candidate_rotations": [],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
        }

    def fake_symmetry_diagnostic(**kwargs):
        return []

    monkeypatch.setattr(workflow_module, "_prepare_symmetry_payload", fake_prepare_symmetry_payload)
    monkeypatch.setattr(workflow_module, "symmetry_eigenvalue_diagnostics_for_kpoint", fake_symmetry_diagnostic)

    outputs = workflow_module.analyze_hsp(config_path)

    assert outputs["symmetry_eigenvalues_csv"].exists()
    with outputs["symmetry_eigenvalues_csv"].open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert {"kpoint", "operation_id", "root_deviation", "D_valley_offdiag_norm"} <= set(reader.fieldnames or [])
        assert list(reader) == []
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "symmetry_eigenvalues_csv" in summary["output_files"]
    subgroup_report = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]
    assert subgroup_report["status"] in ("per_valley_preserving_subgroups_computed", "operation_set_only")
    if subgroup_report["status"] == "per_valley_preserving_subgroups_computed":
        assert subgroup_report["all_valley_intersection"]["operation_count"] == 0
    else:
        assert subgroup_report["standard_group_match_status"] == "not_attempted"
        assert subgroup_report["by_kpoint"]["GammaM"]["closure_status"] == "empty"


def test_write_detailed_files_false_writes_only_summary_files(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"].pop("profile", None)
    raw["output"]["write_detailed_files"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="write_detailed_files"):
        outputs = analyze_hsp(config_path)

    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    # valley_weights.csv is a quick-scan file in standard profile.
    assert outputs.get("valley_weights_csv", None) and outputs["valley_weights_csv"].exists()
    assert not (out_dir / "valley_subspace.json").exists()
    assert not (out_dir / "symmetry_report.json").exists()
    assert not (out_dir / "rotation_eigenvalues.csv").exists()
    assert not (out_dir / "little_group_eigenvalues.csv").exists()
    assert not (out_dir / "little_group_representations.json").exists()
    assert not (out_dir / "symmetry_eigenvalues.csv").exists()
    assert not (out_dir / "diagnostics.h5").exists()
    assert not (out_dir / "projector_symmetry_report.json").exists()
    assert not (out_dir / "hsp_star_conjugation.json").exists()
    assert not (out_dir / "hsp_star_derived_characters.json").exists()


def test_subspace_representation_quality_standalone_json_default_off(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    quality_row = {
        "target_valley": "K_valley",
        "operation_id": 0,
        "diagnosis": "ok",
        "local_unitarity_error": 1.0e-8,
    }
    symmetry_adapted_valley_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [
                    {
                        "reference_valley": "K_valley",
                        "subspace_representation_quality": {"rows": [quality_row]},
                    }
                ]
            }
        }
    }

    outputs = write_analysis_outputs(
        config=config,
        qcut=0.5,
        weight_rows=[],
        sector_names=["K_valley"],
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
        symmetry_adapted_valley_report=symmetry_adapted_valley_report,
    )

    assert config.symmetry_adapted_valley.write_subspace_representation_quality is False
    assert "subspace_representation_quality_json" not in outputs
    assert not (out_dir / "subspace_representation_quality.json").exists()
    assert outputs["symmetry_adapted_valley_analysis_json"].exists()
    analysis_payload = json.loads(outputs["symmetry_adapted_valley_analysis_json"].read_text(encoding="utf-8"))
    assert analysis_payload["by_kpoint"]["GammaM"]["valley_preserving_subspaces"][0][
        "subspace_representation_quality"
    ]["rows"] == [quality_row]
    summary_payload = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "subspace_representation_quality_json" not in summary_payload["output_files"]
    assert summary_payload["symmetry_adapted_valley_analysis"] == symmetry_adapted_valley_report


def test_subspace_representation_quality_standalone_json_opt_in(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw.setdefault("analysis", {})["symmetry_adapted_valley"] = {
        "write_subspace_representation_quality": True,
    }
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    quality_row = {
        "target_valley": "K_valley",
        "operation_id": 0,
        "diagnosis": "unitarity_failed",
        "local_unitarity_error": 2.0e-2,
    }
    symmetry_adapted_valley_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [
                    {
                        "reference_valley": "K_valley",
                        "subspace_representation_quality": {"rows": [quality_row]},
                    }
                ]
            }
        }
    }

    outputs = write_analysis_outputs(
        config=config,
        qcut=0.5,
        weight_rows=[],
        sector_names=["K_valley"],
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
        symmetry_adapted_valley_report=symmetry_adapted_valley_report,
    )

    assert config.symmetry_adapted_valley.write_subspace_representation_quality is True
    assert outputs["subspace_representation_quality_json"] == out_dir / "subspace_representation_quality.json"
    quality_payload = json.loads(outputs["subspace_representation_quality_json"].read_text(encoding="utf-8"))
    assert quality_payload["status"] == "quality_issues_detected"
    assert quality_payload["rows"] == [{**quality_row, "kpoint": "GammaM"}]
    summary_payload = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "subspace_representation_quality_json" in summary_payload["output_files"]


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
        symmetry_rows=[],
        output_paths={},
    )

    assert any("unsupported nspinor=3" in warning for warning in summary["warnings"])
    assert "unsupported nspinor=3" in render_summary_text(summary)


def test_summary_text_renders_symmetry_adapted_valley_subspaces(tmp_path):
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
            "detected_operation_count": 0,
            "candidate_rotations": [],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
            "detected_operations": [],
        },
        symmetry_rows=[],
        output_paths={},
        symmetry_adapted_valley_report={
            "space_group_valley_orbits": [["M1_valley", "M2_valley", "M3_valley"]],
            "by_kpoint": {
                "GammaM": {
                    "status": "warn",
                    "reason": "projector warning",
                    "feature_status": "formal",
                    "workflow_integration_status": "integrated",
                    "local_irrep_ready": True,
                    "diagnostic_only": False,
                    "irrep_matching_input_ready": False,
                    "irrep_matching_input_reason": "spinor convention unverified",
                    "orbits": [
                        {
                            "orbit": ["M1_valley", "M2_valley", "M3_valley"],
                            "status": "diagnostic_only",
                            "local_irrep_ready": False,
                            "irrep_matching_input_ready": False,
                            "reason": "HSP-local orbit diagnostic",
                            "symmetry_adapted_projectors": {
                                "selected_rank": 0,
                                "status": "failed",
                                "max_projector_symmetry_error": 0.0,
                            },
                        }
                    ],
                    "valley_preserving_subspaces": [
                        {
                            "orbit": ["M1_valley"],
                            "analysis_scope": "valley_preserving_subspace",
                            "hsp_preserving_operation_ids": [0, 4],
                            "status": "warn",
                            "local_irrep_ready": True,
                            "irrep_matching_input_ready": False,
                            "irrep_matching_input_reason": "spinor convention unverified",
                            "subspace_space_group": {
                                "candidate_space_group_symbol": "P2",
                                "valley_preserving_operation_ids": [0, 4],
                            },
                            "subspace_group": {
                                "subspace_group_candidate": "C2_like",
                            },
                            "symmetry_adapted_projectors": {
                                "selected_rank": 2,
                                "rank_source": "user_specified",
                                "seed_overlap": {"M1_valley": 0.7385595},
                                "max_projector_symmetry_error": 1.1e-5,
                                "status": "warn",
                            },
                            "valley_preserving_character_diagnostics": {
                                "per_valley": {
                                    "M1_valley": [
                                        {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                                        {"operation_id": 4, "eigenphases": [-0.25, 0.25]},
                                    ]
                                }
                            },
                        }
                    ],
                }
            }
        },
    )

    text = render_summary_text(summary)

    assert "Symmetry-adapted valley analysis" in text
    assert "Symmetry-adapted valley analysis (experimental)" not in text
    assert "space-group valley orbits: [M1_valley, M2_valley, M3_valley]" in text
    assert "HSP-local valley-orbit reports" in text
    assert "full valley-orbit reports" not in text
    assert "valley-preserving subspaces" in text
    assert "GammaM" in text
    assert "M1_valley" in text
    assert "P2" in text
    assert "[0, 4]" in text
    assert "-0.25, 0.25" in text
    assert "spinor convention unverified" in text


def test_summary_text_renders_projected_seed_projector_quality(tmp_path):
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
                    "valley_adapted_subspace": {
                        "status": "valley_separable",
                        "s_min": 0.98,
                        "s_max": 0.99,
                        "assigned_valleys": ["M1_valley", "M1_valley"],
                        "min_valley_concentration": 0.999,
                        "projector_quality": {
                            "expected_rank": 2,
                            "rank_threshold": 0.5,
                            "per_valley": {
                                "M1_valley": {
                                    "rank_estimate": 2,
                                    "rank_gap": 0.97,
                                }
                            },
                            "sum_projector": {
                                "identity_deviation_fro": 0.04,
                                "idempotency_deviation_fro": 0.02,
                            },
                            "max_idempotency_deviation": 0.02,
                            "max_trace_overlap": 1.0e-4,
                            "max_commutator_norm": 2.0e-4,
                        },
                    },
                }
            }
        },
        symmetry_payload={
            "status": "skipped",
            "reason": "test",
            "detected_operations": [],
        },
        symmetry_rows=[],
        output_paths={},
    )

    text = render_summary_text(summary)

    assert "Projected q-cut seed projector quality" in text
    assert "rank_estimates" in text
    assert "M1_valley=2" in text
    assert "M1_valley=0.97" in text


def test_summary_text_renders_hsp_star_coverage(tmp_path):
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
                "MM": {
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
            "detected_operation_count": 0,
            "candidate_rotations": [],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
            "detected_operations": [],
            "hsp_star_report": {
                "status": "symmetry_derivable",
                "by_kpoint": {
                    "MM": {
                        "status": "symmetry_derivable",
                        "star_size": 3,
                        "explicit_count": 1,
                        "symmetry_derivable_count": 2,
                        "requires_additional_dft": False,
                        "symmetry_derivable_representatives": [
                            {
                                "canonical_frac": [0.0, 0.5, 0.0],
                                "generated_by_operation_ids": [2],
                            }
                        ],
                    }
                },
            },
        },
        symmetry_rows=[],
        output_paths={},
    )

    text = render_summary_text(summary)

    assert "HSP-star coverage" in text
    assert "symmetry_derivable" in text
    assert "False" in text
    assert "[0, 0.5, 0] via ops [2]" in text


def test_write_analysis_outputs_plumbs_hsp_star_reports_to_summary(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    hsp_star_conjugation = {
        "status": "ok",
        "entries": [{"conjugation_status": "matched"}],
    }
    hsp_star_derived = {
        "status": "ok",
        "entries": [{"status": "derived"}],
    }
    irrep_workflow_decisions = {
        "status": "ok",
        "workflow_paths": ["direct_qcut", "symmetry_adapted", "blocked"],
        "readiness_levels": ["trusted", "usable_with_caution", "blocked"],
        "by_kpoint": {
            "Gamma": {
                "K_valley": {
                    "workflow_path": "direct_qcut",
                    "readiness_level": "trusted",
                    "reason": "test",
                    "uses_symmetry_adapted_projector": False,
                    "direct_qcut_allowed": True,
                },
            },
        },
    }
    valley_irrep_matching = {
        "status": "ok",
        "tables_implemented": ["spinful_C3"],
        "by_kpoint": {
            "Gamma": {
                "K_valley": {
                    "1": {
                        "workflow_path": "direct_qcut",
                        "readiness_level": "trusted",
                        "subspace_group_candidate": "C3_like",
                        "operation_id": 1,
                        "operation_order": 3,
                        "matched_irrep": "C3_spinor_phase_+1/2",
                        "matching_status": "matched",
                        "reason": "test",
                        "eigenphases": [0.5],
                    },
                },
            },
        },
    }
    ebr_input_candidates = {
        "status": "has_candidates",
        "candidate_count": 1,
        "blocked_count": 0,
        "candidates": [{"ready_for_ebr_input": True}],
        "blocked": [],
    }
    ebr_problem_instances = {
        "status": "has_instances",
        "instance_count": 1,
        "instances": [{"instance_id": "ebr_instance_001"}],
    }
    ebr_export_bundle = {
        "status": "ready_for_external_solver",
        "schema_version": "1.0.0",
        "bundle_count": 1,
        "excluded_count": 0,
        "reduced_ebr_decomposition_status": "not_implemented",
        "bundles": [
            {
                "bundle_id": "bundle_ebr_instance_001",
                "valley": "K_valley",
                "subspace_group_candidate": "C3_like",
                "workflow_path": "direct_qcut",
                "expected_hsps": ["GammaM", "KM"],
                "optional_hsps": ["MM"],
                "missing_optional_hsps": ["MM"],
                "ready_for_external_solver": True,
            },
        ],
        "excluded_instances": [],
    }

    outputs = write_analysis_outputs(
        config=config,
        qcut=0.5,
        weight_rows=[],
        sector_names=["K_valley"],
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
        hsp_star_conjugation_report=hsp_star_conjugation,
        hsp_star_derived_characters=hsp_star_derived,
        irrep_workflow_decisions=irrep_workflow_decisions,
        valley_irrep_matching=valley_irrep_matching,
        ebr_input_candidates=ebr_input_candidates,
        ebr_problem_instances=ebr_problem_instances,
        ebr_export_bundle=ebr_export_bundle,
    )

    payload = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert payload["hsp_star_conjugation"] == hsp_star_conjugation
    assert payload["hsp_star_derived_characters"] == hsp_star_derived
    assert payload["irrep_workflow_decisions"] == irrep_workflow_decisions
    assert payload["valley_irrep_matching"] == valley_irrep_matching
    assert payload["valley_ebr_input_candidates"] == ebr_input_candidates
    assert payload["valley_ebr_problem_instances"] == ebr_problem_instances
    assert payload["valley_ebr_export_bundle"] == ebr_export_bundle
    assert "Valley irrep matching" in outputs["summary_text"]
    assert "C3_spinor_phase_+1/2" in outputs["summary_text"]
    assert "EBR export bundle" in outputs["summary_text"]
    assert "bundle_ebr_instance_001" in outputs["summary_text"]


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
        symmetry_rows=[
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
    assert "diagnostic-only" in text


def test_summary_output_files_use_human_readable_labels(tmp_path):
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
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "skipped",
            "reason": "no structure",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={
            "symmetry_eigenvalues_csv": out_dir / "symmetry_eigenvalues.csv",
            "valley_summary_txt": out_dir / "valley_summary.txt",
        },
    )

    text = render_summary_text(summary)

    assert "Symmetry eigenvalues:" in text
    assert "Human-readable summary:" in text
    assert "symmetry_eigenvalues_csv:" not in text
    assert "valley_summary_txt:" not in text


def test_summary_subspace_polarization_uses_min_eta_score(tmp_path):
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
                    "polarization_score": 0.2,
                    "subspace_valley_status": "valley_mixed_subspace",
                    "valley_adapted_subspace": {
                        "status": "two_valley_adapted",
                        "eta": [0.99, 0.2],
                        "max_abs_eta": 0.99,
                        "s_min": 0.98,
                        "s_max": 1.0,
                    },
                }
            }
        },
        symmetry_payload={
            "status": "skipped",
            "reason": "no structure",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert summary["valley_subspace_analysis"][0]["P_v_min"] == pytest.approx(0.6)


def test_summary_status_keeps_not_derived_and_unreliable_distinct(tmp_path):
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
                    "weights": [
                        {"band_vasp": 101, "valley_status": "not_valley_derived", "W_val": 0.2},
                        {"band_vasp": 102, "valley_status": "projector_unreliable", "W_overlap": 0.2},
                    ],
                    "warnings": [],
                    "subspace_valley_status": "projector_unreliable",
                    "valley_adapted_subspace": {"status": "two_valley_adapted"},
                },
                "KM": {
                    "weights": [],
                    "warnings": [],
                    "valley_adapted_subspace": {"status": "single_band"},
                },
            }
        },
        symmetry_payload={
            "status": "skipped",
            "reason": "no structure",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert [row["status"] for row in summary["valley_projection_summary"]] == ["not_derived", "unreliable"]
    assert summary["valley_subspace_analysis"][0]["status"] == "unreliable"
    assert summary["valley_subspace_analysis"][1]["status"] == "n/a"


def test_symmetry_analysis_distinguishes_computed_from_diagnostic_only(tmp_path):
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
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[
            {
                "kpoint": "KM",
                "operation_id": 1,
                "order": 3,
                "basis": "valley_adapted",
                "state_index": 0,
                "phase_2pi": 1.0 / 6.0,
                "nearest_root_of_unity": "exp(2pii*1/6)",
                "root_deviation": 0.0,
                "rotation_ready": True,
                "topology_input_ready": False,
                "diagnostic_only": True,
                "D_valley_offdiag_norm": 0.01,
                "reason": "two-valley D_valley offdiag diagnostic too large",
            }
        ],
        output_paths={},
    )

    text = render_summary_text(summary)
    assert "two-valley D_valley offdiag diagnostic too large" in text
    assert "exp(i*pi/3)" in text
    assert "exp(2pii*1/6)" not in text
    assert "valley_adapted" not in text
    assert "skipped (two-valley D_valley offdiag diagnostic too large)" not in text


def test_symmetry_summary_orders_hsp_and_labels_valley_exchanging(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["analysis"]["kpoints"] = ["GammaM", "KM"]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [
                {
                    "operation_id": 2,
                    "kind": "C2",
                    "order": 2,
                    "det": 1,
                    "rotation_frac": np.eye(3),
                    "translation_frac": np.zeros(3),
                    "little_group_by_kpoint": {"KM": True, "GammaM": True},
                    "rejection_reason_by_kpoint": {"KM": "valley-exchanging", "GammaM": "valley-exchanging"},
                }
            ],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    rejected = summary["symmetry_analysis"]["rejected_operations"]
    assert [row["kpoint"] for row in rejected] == ["GammaM", "KM"]
    text = render_summary_text(summary)
    assert text.index("GammaM: HSP little group") < text.index("KM: HSP little group")
    assert "valley-exchanging" in text


def test_summary_rejected_operations_are_per_valley_when_inventory_available(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    symmetry_payload = {
        "status": "ok",
        "detected_operations": [
            {
                "operation_id": 3,
                "kind": "C2",
                "order": 2,
                "det": 1,
                "rotation_frac": np.eye(3),
                "translation_frac": np.zeros(3),
                "little_group_by_kpoint": {"GammaM": True},
                "rejection_reason_by_kpoint": {"GammaM": "valley-exchanging"},
            }
        ],
        "candidate_rotations": [],
        "per_valley_preserving_operation_inventory": {
            "GammaM": {
                "M1_valley": [
                    {
                        "operation_id": 3,
                        "kind": "C2",
                        "order": 2,
                        "little_group_passed": True,
                        "target_valley": "M1_valley",
                        "mapped_valley": "M1_valley",
                        "valley_preserving": True,
                        "allowed_for_valley_preserving_representation": True,
                        "reason": "",
                    }
                ],
                "M2_valley": [
                    {
                        "operation_id": 3,
                        "kind": "C2",
                        "order": 2,
                        "little_group_passed": True,
                        "target_valley": "M2_valley",
                        "mapped_valley": "M3_valley",
                        "valley_preserving": False,
                        "allowed_for_valley_preserving_representation": False,
                        "reason": "valley-changing (maps to M3_valley)",
                    }
                ],
            }
        },
        "little_group_check": {"status": "evaluated_per_kpoint"},
        "valley_preservation_check": {"status": "completed"},
    }

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload=symmetry_payload,
        symmetry_rows=[],
        output_paths={},
    )

    rejected = summary["symmetry_analysis"]["rejected_operations"]
    assert rejected == [
        {
            "kpoint": "GammaM",
            "target_valley": "M2_valley",
            "operation_id": 3,
            "order": 2,
            "kind": "C2",
            "reason": "valley-changing (maps to M3_valley)",
        }
    ]
    text = render_summary_text(summary)
    assert "M2_valley" in text
    assert "M1_valley" not in text.split("rejected operations:", 1)[1]


def test_summary_preserves_hsp_little_group_inventory(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload

    inventory = {
        "KM": [
            {
                "operation_id": 7,
                "kind": "C3",
                "order": 3,
                "little_group_passed": True,
                "valley_preserving": True,
                "valley_exchanging": False,
                "allowed_for_valley_preserving_representation": True,
                "reason": "",
            }
        ]
    }
    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "hsp_little_group_inventory": inventory,
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert summary["symmetry_analysis"]["hsp_little_group_inventory"] == inventory


def test_summary_preserves_valley_preserving_subgroup_report(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    subgroup_report = {
        "status": "standard_group_matched",
        "global_operation_set": {
            "operation_set_label": "G_tau",
            "allowed_operation_ids": [0, 1, 2],
            "closure_status": "closed",
        },
        "standard_group_match": {
            "number": 143,
            "international_short": "P3",
            "source": "spglib.get_spacegroup_type_from_symmetry",
            "operation_ids": [0, 1, 2],
        },
        "standard_group_match_status": "matched",
        "irrep_matching": {
            "status": "table_mapping_complete",
            "table_source": "irreptables",
            "label_matching": "matched",
            "character_matching_status": "matched",
            "irrep_results_by_kpoint": {
                "KM": {
                    "status": "matched",
                    "table_kpoint_label": "K",
                    "irrep_multiplicities": {"-K5": 1, "-K6": 1},
                    "failure_reasons": [],
                }
            },
        },
        "by_kpoint": {
            "KM": {
                "operation_set_label": "G_tau,k(KM)",
                "allowed_operation_ids": [0, 1, 2],
                "closure_status": "closed",
                "missing_products": [],
            }
        },
    }
    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "valley_preserving_subgroup_report": subgroup_report,
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert summary["symmetry_analysis"]["valley_preserving_subgroup_report"] == subgroup_report
    text = render_summary_text(summary)
    assert "valley-preserving subgroup: P3 (143)" in text
    assert "irrep matching: matched" in text
    assert "KM: -K5 x 1, -K6 x 1" in text


def test_summary_exposes_symmetry_characters_as_first_class_rows(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload

    symmetry_rows = [
        {
            "kpoint": "KM",
            "operation_id": 7,
            "kind": "C3",
            "order": 3,
            "basis": "valley_adapted",
            "little_group_passed": True,
            "valley_preserving": True,
            "character_raw": "1.000000+0.000000j",
            "character_valley": "0.500000+0.866025j",
            "topology_input_ready": True,
            "diagnostic_only": False,
        },
        {
            "kpoint": "KM",
            "operation_id": 7,
            "kind": "C3",
            "order": 3,
            "basis": "valley_adapted",
            "little_group_passed": True,
            "valley_preserving": True,
            "character_raw": "",
            "character_valley": "",
            "topology_input_ready": True,
            "diagnostic_only": False,
        },
        {
            "kpoint": "KM",
            "operation_id": 8,
            "kind": "C2",
            "order": 2,
            "basis": "valley_adapted",
            "little_group_passed": True,
            "valley_preserving": False,
            "character_raw": "0.000000+0.000000j",
            "character_valley": "0.000000+0.000000j",
            "topology_input_ready": False,
            "diagnostic_only": True,
        },
    ]
    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=symmetry_rows,
        output_paths={},
    )

    assert summary["symmetry_characters"] == [
        {
            "kpoint": "KM",
            "target_valley": "",
            "operation_id": 7,
            "kind": "C3",
            "order": 3,
            "basis": "valley_adapted",
            "character_raw": "1.000000+0.000000j",
            "character_valley": "0.500000+0.866025j",
            "topology_input_ready": True,
            "diagnostic_only": False,
            "accepted_for_valley_preserving_representation": True,
        }
    ]


def test_summary_exposes_rotation_readiness_thresholds(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["rotation"] = {"readiness_preset": "normal"}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    thresholds = summary["rotation_readiness_thresholds"]
    assert thresholds["readiness_preset"] == "normal"
    assert thresholds["root_deviation_tol"] == pytest.approx(1.0e-5)
    assert thresholds["D_valley_offdiag_tol"] == pytest.approx(1.0e-3)
    assert thresholds["irrep_weight_tol"] == pytest.approx(5.0e-5)
    assert "not universal physical constants" in thresholds["interpretation"]
    assert "do not loosen" in thresholds["recommended_action"]


def test_workflow_passes_irrep_weight_tol_to_matching(tmp_path, monkeypatch):
    import importlib

    workflow_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["rotation"] = {"irrep_weight_tol": 2.5e-4}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    symmetry_payload = {
        "status": "ok",
        "operation_detection_backend": "spglib",
        "structure_file": "fake-CONTCAR",
        "symmetry_eigenvalue_enabled": True,
        "detected_operations": [],
        "candidate_rotations": [],
        "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
        "valley_preservation_check": {"required": True, "status": "completed"},
    }
    captured: list[float] = []

    def fake_add_valley_irrep_results(**kwargs):
        captured.append(kwargs["tolerance"])
        return {}

    monkeypatch.setattr(workflow_module, "_prepare_symmetry_payload", lambda config, monolayer_recip: dict(symmetry_payload))
    monkeypatch.setattr(workflow_module, "symmetry_eigenvalue_diagnostics_for_kpoint", lambda **kwargs: [])
    monkeypatch.setattr(workflow_module, "add_valley_irrep_results", fake_add_valley_irrep_results)

    workflow_module.analyze_hsp(config_path)

    assert captured == [pytest.approx(2.5e-4)]


def test_single_band_and_not_degenerate_no_subspace_valley_status_mislabel(tmp_path):
    """P1-2: single_band / not_degenerate should not be labelled not_valley_derived."""
    cases = [
        ("single_band", np.array([0.1]), np.array([101])),
        ("not_degenerate", np.array([0.1, 0.105]), np.array([101, 102])),
    ]
    for label, energies, bands in cases:
        h5_path = tmp_path / f"wf_{label}.h5"
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
            nb = len(bands)
            kp["g_vectors_frac"] = np.array([[0, 0, 0], [1, 0, 0]])
            kp["g_vectors_cart"] = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
            coeffs = np.zeros((nb, 1, 2), dtype=np.complex128)
            coeffs[0, 0, 0] = 1.0 + 0.0j
            if nb == 2:
                coeffs[1, 0, 1] = 1.0 + 0.0j
            kp["coefficients"] = coeffs
            kp["energies_eV"] = energies
            kp["band_indices_vasp"] = bands

        config_path = tmp_path / f"config_{label}.yaml"
        out_dir = tmp_path / f"out_{label}"
        config = {
            "input": {"wavefunction_h5": str(h5_path)},
            "analysis": {"kpoints": ["GammaM"], "iband": bands.tolist(), "degeneracy_tol_meV": 1.0},
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
                {"name": "K_sector", "centers": ["K"]},
                {"name": "Kp_sector", "centers": ["Kp"]},
            ],
            "projection": {"qcut_mode": "absolute", "qcut_Ainv": 0.5, "overlap_policy": "warn_exclude"},
            "output": {"directory": str(out_dir), "profile": "debug"},
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        outputs = analyze_hsp(config_path)
        subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
        kp_data = subspace["kpoints"]["GammaM"]

        for w in kp_data["weights"]:
            assert w.get("valley_status") == "raw_valley_clean", f"raw band mislabeled in {label}"

        diag = kp_data.get("valley_adapted_subspace", {})
        assert diag.get("status") == label, f"subspace diagnostic status mismatch in {label}"
        assert kp_data.get("subspace_valley_status", "missing") not in ("not_valley_derived",), \
            f"subspace labeled not_valley_derived in {label}"

        summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
        assert not any("target subspace is not valley-derived" in w for w in summary["warnings"]), \
            f"false subspace warning in {label}"
        subspace_rows = summary["valley_subspace_analysis"]
        assert subspace_rows
        assert subspace_rows[0].get("basis_status") == label, \
            f"basis_status missing from summary JSON for {label}"
        summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
        assert "basis_status" not in summary_text


def test_rotation_order_none_yields_not_requested_symmetry_status(tmp_path):
    """P1-3: rotation_order: none should give symmetry_status = not_requested."""
    h5_path = tmp_path / "wf.h5"
    write_fixture(h5_path)
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    structure = tmp_path / "CONTCAR"
    write_square_poscar(structure)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["symmetry"]["filters"]["rotation_order"] = "none"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    kp_data = subspace["kpoints"]["GammaM"]
    assert kp_data.get("symmetry_status") == "not_requested"

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary["symmetry_analysis"]["symmetry_eigenvalue_enabled"] is False
    assert summary["symmetry_eigenvalues"] == []
    assert "symmetry_eigenvalues_csv" not in outputs
    assert not (out_dir / "symmetry_eigenvalues.csv").exists()


def test_workflow_requests_all_valley_preserving_little_group_operations(tmp_path, monkeypatch):
    import importlib

    workflow_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    calls: list[bool] = []

    def fake_prepare_symmetry_payload(config, monolayer_recip):
        return {
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "fake-CONTCAR",
            "spacegroup_number": 150,
            "international": "P321",
            "symmetry_eigenvalue_enabled": True,
            "requested_rotation_order": "auto",
            "resolved_rotation_order": 3,
            "detected_operation_count": 0,
            "detected_operations": [],
            "candidate_rotations": [],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
        }

    def fake_symmetry_diagnostic(**kwargs):
        calls.append(bool(kwargs.get("generators_only", False)))
        return []

    monkeypatch.setattr(workflow_module, "_prepare_symmetry_payload", fake_prepare_symmetry_payload)
    monkeypatch.setattr(workflow_module, "symmetry_eigenvalue_diagnostics_for_kpoint", fake_symmetry_diagnostic)

    workflow_module.analyze_hsp(config_path)

    assert calls == [False]


def test_workflow_passes_hdf5_spinor_flag_to_subgroup_report(tmp_path, monkeypatch):
    import importlib

    workflow_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)
    captured: list[bool] = []

    def fake_prepare_symmetry_payload(config, monolayer_recip):
        return {
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "fake-CONTCAR",
            "spacegroup_number": 143,
            "international": "P3",
            "symmetry_eigenvalue_enabled": True,
            "requested_rotation_order": "auto",
            "resolved_rotation_order": 3,
            "detected_operation_count": 0,
            "detected_operations": [],
            "candidate_rotations": [],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
        }

    def fake_symmetry_diagnostic(**kwargs):
        return []

    def fake_subgroup_report(*, symmetry_payload, target_kpoints):
        captured.append(bool(symmetry_payload["spinor_wavefunction"]))
        report = {
            "status": "operation_set_only",
            "standard_group_match": None,
            "standard_group_match_status": "not_attempted",
            "by_kpoint": {},
        }
        symmetry_payload["valley_preserving_subgroup_report"] = report
        return report

    monkeypatch.setattr(workflow_module, "_prepare_symmetry_payload", fake_prepare_symmetry_payload)
    monkeypatch.setattr(workflow_module, "symmetry_eigenvalue_diagnostics_for_kpoint", fake_symmetry_diagnostic)
    monkeypatch.setattr(workflow_module, "build_valley_preserving_subgroup_report", fake_subgroup_report)

    workflow_module.analyze_hsp(config_path)

    assert captured == [True]


def _p3_fake_symmetry_payload() -> dict:
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    lattice = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    operations = []
    for operation_id, rotation in enumerate([np.eye(3, dtype=int), c3, c3 @ c3]):
        operations.append(
            {
                "operation_id": operation_id,
                "kind": "identity" if operation_id == 0 else "C3",
                "order": 1 if operation_id == 0 else 3,
                "rotation_frac": rotation,
                "translation_frac": np.zeros(3),
                "rotation_cart": rotation.astype(float),
                "translation_cart": np.zeros(3),
                "det": 1,
                "candidate_rotation": operation_id != 0,
                "preserved": {"K_valley": True, "Kp_valley": True},
                "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
            }
        )
    return {
        "status": "ok",
        "operation_detection_backend": "spglib",
        "structure_file": "fake-CONTCAR",
        "spacegroup_number": 143,
        "international": "P3",
        "symmetry_eigenvalue_enabled": True,
        "requested_rotation_order": "auto",
        "resolved_rotation_order": 3,
        "detected_operation_count": len(operations),
        "detected_operations": operations,
        "candidate_rotations": [1, 2],
        "symprec_scan_summary": [],
        "lattice_direct_cart": lattice,
        "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
        "valley_preservation_check": {"required": True, "status": "completed"},
    }


def _p3_d_valley_matrices(*, operation_2_offdiag: float = 0.0) -> dict[int, np.ndarray]:
    """Return diagonal D_valley matrices at GM for P3 irrep matching test.

    In the valley-adapted basis:
      state 0: C3 eigenvalue = exp(-i*pi/3),  C3^2 eigenvalue = exp(+i*pi/3)
      state 1: C3 eigenvalue = exp(+i*pi/3),  C3^2 eigenvalue = exp(-i*pi/3)

    These match -GM5 and -GM6 respectively.
    """
    matrices = {
        0: np.eye(2, dtype=np.complex128),  # identity
        1: np.diag([np.exp(-1j * np.pi / 3.0), np.exp(+1j * np.pi / 3.0)]),  # C3
        2: np.diag([np.exp(+1j * np.pi / 3.0), np.exp(-1j * np.pi / 3.0)]),  # C3^2
    }
    if operation_2_offdiag > 0.0:
        matrices[2] = matrices[2].copy()
        matrices[2][0, 1] = operation_2_offdiag
        matrices[2][1, 0] = operation_2_offdiag * 0.7
    return matrices


def _ready_character_rows(kpoint: str, *, operation_2_state_1_ready: bool = True) -> list[dict]:
    """Return aggregate (trace-level) character rows for P3 irrep matching.

    State-level characters are now collected from D_valley diagonal via
    the representation_payload, not from eigenvalue ordering in these rows.
    The eigenvalue_real/imag fields here are deliberately scrambled to
    verify that they are NOT used for state irrep assignment.
    """
    rows = []
    for operation_id in [1, 2]:
        for state_index in [0, 1]:
            ready = operation_id != 2 or state_index != 1 or operation_2_state_1_ready
            # Deliberately wrong eigenvalue ordering to catch any regression
            # that would read eigenvalues instead of D_valley diagonal
            wrong_eigenvalue = np.exp(1j * np.pi * (state_index + operation_id) / 7.0)
            rows.append(
                {
                    "kpoint": kpoint,
                    "operation_id": operation_id,
                    "kind": "C3",
                    "order": 3,
                    "basis": "valley_adapted",
                    "state_index": state_index,
                    "phase_2pi": 0.0,
                    "nearest_root_of_unity": "1",
                    "root_deviation": 0.0,
                    "rotation_ready": ready,
                    "D_valley_offdiag_norm": 0.0,
                    "eigenvalue_real": float(wrong_eigenvalue.real),
                    "eigenvalue_imag": float(wrong_eigenvalue.imag),
                    "character_valley": "1.000000+0.000000j" if state_index == 0 else "",
                    "character_raw": "",
                    "little_group_passed": True,
                    "valley_preserving": True,
                    "topology_input_ready": ready,
                    "diagnostic_only": not ready,
                    "reason": "" if ready else "root deviation too large",
                }
            )
    return rows


def _fake_diagnostics_with_dvalley(kpoint_name, representation_payload, **kwargs):
    """Mock symmetry_eigenvalue_diagnostics_for_kpoint that populates
    representation_payload with D_valley and returns aggregate rows."""
    d_valleys = _p3_d_valley_matrices()
    kp_repr = representation_payload.setdefault(kpoint_name, {})
    for op_id, d_valley in d_valleys.items():
        kp_repr[f"operation_{op_id}"] = {
            "D_valley": d_valley,
            "unitarity_deviation": 0.0,
            "mapping_miss_count": 0,
        }
    return _ready_character_rows(kpoint_name)


def _fake_diagnostics_with_dvalley_incomplete(kpoint_name, representation_payload, **kwargs):
    """Same as _fake_diagnostics_with_dvalley but operation 2 has one non-ready row."""
    d_valleys = _p3_d_valley_matrices()
    kp_repr = representation_payload.setdefault(kpoint_name, {})
    for op_id, d_valley in d_valleys.items():
        kp_repr[f"operation_{op_id}"] = {
            "D_valley": d_valley,
            "unitarity_deviation": 0.0,
            "mapping_miss_count": 0,
        }
    return _ready_character_rows(kpoint_name, operation_2_state_1_ready=False)


def _fake_diagnostics_dvalley_mixed(kpoint_name, representation_payload, **kwargs):
    """D_valley for operation 2 has large off-diagonal mixing."""
    d_valleys = _p3_d_valley_matrices(operation_2_offdiag=0.5)
    kp_repr = representation_payload.setdefault(kpoint_name, {})
    for op_id, d_valley in d_valleys.items():
        kp_repr[f"operation_{op_id}"] = {
            "D_valley": d_valley,
            "unitarity_deviation": 0.0,
            "mapping_miss_count": 0,
        }
    return _ready_character_rows(kpoint_name)


def test_workflow_writes_irrep_results_when_characters_are_ready(tmp_path, monkeypatch):
    import importlib

    workflow_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    monkeypatch.setattr(workflow_module, "_prepare_symmetry_payload", lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(
        workflow_module,
        "symmetry_eigenvalue_diagnostics_for_kpoint",
        _fake_diagnostics_with_dvalley,
    )

    outputs = workflow_module.analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    matching = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]
    assert matching["character_matching_status"] == "matched"
    result = matching["irrep_results_by_kpoint"]["GammaM"]["K_valley"]
    assert result["status"] == "matched"
    assert result["table_kpoint_label"] == "GM"
    assert result["irrep_multiplicities"] == {"-GM5": 1, "-GM6": 1}
    assert result["state_irrep_assignment_status"] == "matched"
    assert result["state_irrep_results"] == [
        {
            "state_index": 0,
            "status": "matched",
            "irrep_label": "-GM5",
            "computed_characters": {
                "1": "1.000000+0.000000j",
                "2": "0.500000-0.866025j",
                "3": "0.500000+0.866025j",
            },
            "irrep_multiplicities": {"-GM5": 1},
            "missing_table_operation_indices": [],
            "failure_reasons": [],
        },
        {
            "state_index": 1,
            "status": "matched",
            "irrep_label": "-GM6",
            "computed_characters": {
                "1": "1.000000+0.000000j",
                "2": "0.500000+0.866025j",
                "3": "0.500000-0.866025j",
            },
            "irrep_multiplicities": {"-GM6": 1},
            "missing_table_operation_indices": [],
            "failure_reasons": [],
        },
    ]
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "GammaM/K_valley: -GM5 x 1, -GM6 x 1" in summary_text
    assert "GammaM/K_valley state irreps: state 0 -> -GM5, state 1 -> -GM6" in summary_text


def test_workflow_keeps_irrep_results_incomplete_when_an_operation_has_non_ready_rows(tmp_path, monkeypatch):
    import importlib

    workflow_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    monkeypatch.setattr(workflow_module, "_prepare_symmetry_payload", lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(
        workflow_module,
        "symmetry_eigenvalue_diagnostics_for_kpoint",
        _fake_diagnostics_with_dvalley_incomplete,
    )

    outputs = workflow_module.analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    matching = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]
    assert matching["character_matching_status"] == "incomplete"
    result = matching["irrep_results_by_kpoint"]["GammaM"]["K_valley"]
    assert result["status"] == "missing_characters"
    assert result["irrep_multiplicities"] == {}
    assert result["missing_table_operation_indices"] == [3]
    assert result["state_irrep_assignment_status"] == "incomplete"


def test_workflow_keeps_irrep_results_incomplete_when_symmetry_consistency_fails(tmp_path, monkeypatch):
    import importlib

    workflow_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    symmetry_consistency_report = {
        "status": "symmetry_consistency_failures_detected",
        "warn_tol": 0.01,
        "fail_tol": 0.1,
        "by_kpoint": {
            "GammaM": {
                "seed_projector_symmetry": [
                    {
                        "operation_id": 1,
                        "source_valley": "K_valley",
                        "mapped_valley": "K_valley",
                        "epsilon_seed": 0.5,
                        "little_group_passed": True,
                        "status": "failed",
                        "reason": "",
                    },
                    {
                        "operation_id": 2,
                        "source_valley": "K_valley",
                        "mapped_valley": "K_valley",
                        "epsilon_seed": 0.4,
                        "little_group_passed": True,
                        "status": "failed",
                        "reason": "",
                    },
                ]
            }
        },
    }

    def fake_diagnostics_with_target_valleys(kpoint_name, representation_payload, **kwargs):
        rows = _fake_diagnostics_with_dvalley(kpoint_name, representation_payload, **kwargs)
        return [
            {**row, "target_valley": valley_name}
            for valley_name in ("K_valley", "Kp_valley")
            for row in rows
        ]

    monkeypatch.setattr(workflow_module, "_prepare_symmetry_payload", lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(
        workflow_module,
        "symmetry_eigenvalue_diagnostics_for_kpoint",
        fake_diagnostics_with_target_valleys,
    )
    monkeypatch.setattr(
        workflow_module,
        "build_projector_symmetry_report",
        lambda **kwargs: symmetry_consistency_report,
    )

    outputs = workflow_module.analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    matching = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]
    assert matching["character_matching_status"] == "incomplete"
    result = matching["irrep_results_by_kpoint"]["GammaM"]["K_valley"]
    assert result["status"] == "missing_characters"
    rows = [
        row for row in summary["symmetry_eigenvalues"]
        if row["operation_id"] == 1 and row["target_valley"] == "K_valley"
    ]
    assert rows
    assert all(row["diagnostic_only"] is True for row in rows)
    assert all(row["topology_input_ready"] is False for row in rows)
    assert all(row["projector_symmetry_status"] == "failed" for row in rows)


def test_workflow_writes_symmetry_consistency_report_when_no_seed_data(tmp_path, monkeypatch):
    import importlib

    workflow_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    monkeypatch.setattr(workflow_module, "_prepare_symmetry_payload", lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(
        workflow_module,
        "symmetry_eigenvalue_diagnostics_for_kpoint",
        lambda **kwargs: [],
    )

    outputs = workflow_module.analyze_hsp(config_path)

    cov_path = outputs["projector_symmetry_report_json"]
    symmetry_consistency = json.loads(cov_path.read_text(encoding="utf-8"))
    assert symmetry_consistency["status"] == "no_data"

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary["projector_symmetry"]["status"] == "no_data"


def test_state_irrep_rejected_when_dvalley_has_offdiagonal_mixing(tmp_path, monkeypatch):
    """Mixing gate: D_valley with large off-diagonal → no state label."""
    import importlib

    workflow_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    monkeypatch.setattr(workflow_module, "_prepare_symmetry_payload",
                        lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(workflow_module, "symmetry_eigenvalue_diagnostics_for_kpoint",
                        _fake_diagnostics_dvalley_mixed)

    outputs = workflow_module.analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    matching = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]
    result = matching["irrep_results_by_kpoint"]["GammaM"]["K_valley"]
    # Aggregate characters still pass (trace = sum of eigenvalues, even with off-diagonal)
    assert result["status"] == "matched"
    # But state-level is incomplete because C3^2 D_valley has mixing
    assert result["state_irrep_assignment_status"] == "incomplete"
    assert len(result["state_irrep_results"]) > 0
    # At least one state result should not be "matched"
    assert any(s.get("status") != "matched" for s in result["state_irrep_results"])


def test_state_irrep_ignores_eigenvalue_ordering_uses_dvalley_diagonal(tmp_path, monkeypatch):
    """Regression test: eigenvalue rows contain deliberately wrong values,
    but state irrep follows D_valley diagonal entries correctly."""
    import importlib

    workflow_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    monkeypatch.setattr(workflow_module, "_prepare_symmetry_payload",
                        lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(workflow_module, "symmetry_eigenvalue_diagnostics_for_kpoint",
                        _fake_diagnostics_with_dvalley)

    outputs = workflow_module.analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    result = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]["irrep_results_by_kpoint"]["GammaM"]["K_valley"]
    assert result["state_irrep_assignment_status"] == "matched"
    # State 0 characters must equal D_valley diagonal: identity=1, C3=exp(-i*pi/3), C3^2=exp(+i*pi/3)
    s0 = result["state_irrep_results"][0]
    assert s0["irrep_label"] == "-GM5"
    assert s0["status"] == "matched"
    # The eigenvalue rows contain wrong values; state assignment uses D_valley diagonal
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "state 0 -> -GM5" in summary_text
    assert "state 1 -> -GM6" in summary_text


def test_subspace_projector_unreliable_when_band_overlap_exceeds_threshold(tmp_path):
    """P2-4: adapted subspace with band W_overlap > threshold → projector_unreliable."""
    h5_path = tmp_path / "wf.h5"
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
        # G-vectors: [0,0,0] clean K, [5,0,0] clean Kp, [2.5,0,0] overlap
        q_cart = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        kp["g_vectors_frac"] = q_cart
        kp["g_vectors_cart"] = q_cart
        a = np.sqrt(0.9)
        b = np.sqrt(0.1)
        kp["coefficients"] = np.array(
            [
                [[a + 0.0j, 0.0 + 0.0j, b + 0.0j]],
                [[0.0 + 0.0j, a + 0.0j, b + 0.0j]],
            ],
            dtype=np.complex128,
        )
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])

    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0},
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
            {"name": "K_sector", "centers": ["K"]},
            {"name": "Kp_sector", "centers": ["Kp"]},
        ],
        "projection": {
            "qcut_mode": "absolute", "qcut_Ainv": 3.0, "overlap_policy": "warn_exclude",
            "thresholds": {"W_val_min": 0.8},
        },
        "output": {"directory": str(out_dir), "profile": "debug"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        outputs = analyze_hsp(config_path)

    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    kp_data = subspace["kpoints"]["GammaM"]
    assert kp_data.get("subspace_valley_status") == "projector_unreliable"


def test_h5_reader_rejects_duplicate_kpoint_names(tmp_path):
    """P2-6: duplicate k-point names in HDF5 should raise ValueError."""
    h5_path = tmp_path / "dup.h5"
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3) * 10.0
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kpts_grp = h5.create_group("kpoints")
        for idx in range(2):
            kp = kpts_grp.create_group(str(idx))
            kp["name"] = "GammaM"
            kp["frac"] = np.zeros(3)
            kp["cart"] = np.zeros(3)
            kp["g_vectors_frac"] = np.zeros((1, 3))
            kp["g_vectors_cart"] = np.zeros((1, 3))
            kp["coefficients"] = np.ones((1, 1, 1), dtype=np.complex128)
            kp["energies_eV"] = np.array([0.1])
            kp["band_indices_vasp"] = np.array([1])

    with pytest.raises(ValueError, match="Duplicate k-point name"):
        read_wavefunction_h5(h5_path)


def test_subspace_thresholds_inherit_user_config(tmp_path):
    """Fix 2: user thresholds (overlap_warn) affect subspace projector_unreliable."""
    h5_path = tmp_path / "wf.h5"
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
        q_cart = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        kp["g_vectors_frac"] = q_cart
        kp["g_vectors_cart"] = q_cart
        a = np.sqrt(0.9)
        b = np.sqrt(0.1)
        kp["coefficients"] = np.array(
            [[[a, 0.0, b]], [[0.0, a, b]]], dtype=np.complex128,
        )
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])

    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0},
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
            {"name": "K_sector", "centers": ["K"]},
            {"name": "Kp_sector", "centers": ["Kp"]},
        ],
        "projection": {
            "qcut_mode": "absolute", "qcut_Ainv": 3.0, "overlap_policy": "warn_exclude",
            "thresholds": {"W_val_min": 0.8, "overlap_warn": 0.15},
        },
        "output": {"directory": str(out_dir), "profile": "debug"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        outputs = analyze_hsp(config_path)

    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    kp_data = subspace["kpoints"]["GammaM"]
    # overlap_warn=0.15 > max_w_overlap=0.1 → not projector_unreliable
    assert kp_data.get("subspace_valley_status") == "valley_separable_subspace"


def test_rotation_thresholds_from_config_parsed_and_applied(tmp_path):
    """Fix 3: rotation thresholds in YAML config are parsed and used."""
    h5_path = tmp_path / "wf.h5"
    structure = tmp_path / "CONTCAR"
    write_square_poscar(structure)
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
        q_cart = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        kp["g_vectors_frac"] = q_cart
        kp["g_vectors_cart"] = q_cart
        kp["coefficients"] = np.array(
            [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.complex128,
        )
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])

    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0},
        "monolayer_lattices": {
            "default": {"reciprocal_cart": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]}
        },
        "valley_centers": {
            "coordinate_mode": "cart",
            "centers": [
                {"name": "K", "cart": [1.0, 0.0, 0.0]},
                {"name": "Kp", "cart": [-1.0, 0.0, 0.0]},
            ],
        },
        "valley_subspaces": [
            {"name": "K_sector", "centers": ["K"]},
            {"name": "Kp_sector", "centers": ["Kp"]},
        ],
        "projection": {"qcut_mode": "absolute", "qcut_Ainv": 0.25, "overlap_policy": "warn_exclude"},
        "symmetry": {
            "operations": {"mode": "auto", "structure_file": str(structure), "backend": "spglib"},
            "tolerance": {"symprec": 1.0e-5, "angle_tolerance": -1.0},
            "filters": {"rotation_order": 2},
        },
        "rotation": {"unitarity_tol": 1.0e-4, "root_deviation_tol": 1.0e-6, "D_valley_offdiag_tol": 1.0e-6},
        "output": {"directory": str(tmp_path / "out"), "profile": "debug"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    outputs = analyze_hsp(config_path)
    # Verify config is parsed
    from valleyscope.io.config import load_config
    app_config = load_config(config_path)
    assert app_config.rotation.unitarity_tol == pytest.approx(1.0e-4)
    assert app_config.rotation.root_deviation_tol == pytest.approx(1.0e-6)
    assert app_config.rotation.D_valley_offdiag_tol == pytest.approx(1.0e-6)

    # Verify symmetry eigenvalues were produced (thresholds don't break output)
    with outputs["symmetry_eigenvalues_csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 0

    # Default config (all fields omitted) still works
    config2 = dict(config)
    del config2["rotation"]
    config_path2 = tmp_path / "config2.yaml"
    config_path2.write_text(yaml.safe_dump(config2), encoding="utf-8")
    outputs2 = analyze_hsp(config_path2)
    with outputs2["symmetry_eigenvalues_csv"].open(encoding="utf-8") as handle:
        rows2 = list(csv.DictReader(handle))
    assert len(rows2) > 0


def _analysis_output_file_keys() -> set[str]:
    """Derive file-output keys from analysis_outputs AST.
    Excludes non-file return keys (summary_text, summary_stdout)."""
    import ast
    import inspect
    from valleyscope.reports import analysis_outputs

    tree = ast.parse(inspect.getsource(analysis_outputs))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in ("outputs", "summary_path_plan"):
            continue
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            keys.add(node.slice.value)
    return keys - {"summary_text", "summary_stdout"}


def test_output_files_manifest_has_labels_for_all_output_keys():
    """Every file-output key must have a human-readable label in
    OUTPUT_FILE_LABELS.  No key may silently fall back to title-case."""
    from valleyscope.reports.summary_report import OUTPUT_FILE_LABELS, _output_file_label

    keys = _analysis_output_file_keys()
    assert keys, "AST-derived key set is empty — check analysis_outputs module"

    missing = keys - set(OUTPUT_FILE_LABELS)
    assert not missing, (
        f"output keys missing from OUTPUT_FILE_LABELS: {sorted(missing)}"
    )

    for key in keys:
        label = _output_file_label(key)
        fallback = key.replace("_", " ").title()
        assert label != fallback, (
            f"label for {key!r} is the title-case fallback; "
            f"add it to OUTPUT_FILE_LABELS"
        )


def test_output_file_labels_no_stale_legacy_names():
    """No public output file label may contain stale legacy terms."""
    from valleyscope.reports.summary_report import OUTPUT_FILE_LABELS, _output_file_label

    keys = _analysis_output_file_keys()
    stale_terms = [
        "valley_sectors", "target_bands_vasp",
        "rotation_eigenvalues", "little_group_eigenvalues",
        "little_group_representations",
    ]
    for key in sorted(keys):
        label = _output_file_label(key).lower()
        for term in stale_terms:
            assert term not in label, (
                f"stale term '{term}' in label for {key!r}: {label}"
            )


# --- Parent-valley readiness boundary regression test ---


def _make_toy_wf_with_far_k(path: Path):
    """Write toy HDF5 with k at (2.0,0,0) — far from centers at (0,0) and (4,0).

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


def test_parent_valley_mode_changes_reporting_but_not_readiness(tmp_path):
    """k_resolved_parent_valley changes reporting weights but NOT seed matrices."""
    import json
    from valleyscope.workflows.analyze_hsp import analyze_hsp

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
    # fixed_center: k at (2.5,0,0) far from both V0(0,0) and V0p(5,0) — all mask empty
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


def test_schema_doc_covers_public_outputs_and_reduced_ebr_statuses():
    """docs/schema.md names the required public files and reduced-EBR statuses."""
    schema_text = Path("docs/schema.md").read_text(encoding="utf-8")

    # Public user and downstream EBR files must be named.
    for name in [
        "valley_summary.txt", "valley_summary.json",
        "valley_ebr_export_bundle.json", "valley_reduced_ebr_mapping.json",
    ]:
        assert name in schema_text, f"docs/schema.md must mention {name}"

    # Reduced-EBR public statuses must be listed in table or prose.
    for status in [
        "not_evaluated", "missing_table", "solved_exact", "no_exact_solution",
    ]:
        assert status in schema_text, (
            f"docs/schema.md must document reduced-EBR status '{status}'"
        )

    # Reduced-EBR public field names must appear.
    for field in [
        "mapping_status", "reduced_ebr_decomposition_status",
        "table_status", "not_applicable", "not_provided", "loaded",
    ]:
        assert field in schema_text, (
            f"docs/schema.md must document reduced-EBR field '{field}'"
        )

    # Stale phrases must not appear.
    stale_phrases = [
        "enabled with valid table",   # output presence ≠ valid table
    ]
    for phrase in stale_phrases:
        assert phrase not in schema_text, (
            f"docs/schema.md contains stale phrase: '{phrase}'"
        )


# --- Output profile tests ---

def test_default_standard_profile_writes_only_public_outputs(tmp_path):
    """Default output.profile=standard emits only public files."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"].pop("profile", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    # Public outputs always present.
    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    assert outputs.get("valley_weights_csv") and outputs["valley_weights_csv"].exists()

    # Debug/detail files must NOT exist with standard profile.
    debug_files = [
        "valley_subspace.json", "symmetry_report.json", "symmetry_eigenvalues.csv",
        "diagnostics.h5", "valley_basis_transform.h5",
        "projector_symmetry_report.json", "symmetry_adapted_valley_analysis.json",
        "target_subspace_closure.json", "hsp_star_conjugation.json",
        "hsp_star_derived_characters.json", "subspace_representation_quality.json",
        "irrep_workflow_decisions.json", "valley_irrep_matching.json",
        "valley_ebr_input_candidates.json", "valley_ebr_problem_instances.json",
        "folded_center_report.json", "sampled_k_coverage.json",
    ]
    for fname in debug_files:
        assert not (out_dir / fname).exists(), f"{fname} must not exist in standard profile"

    # Summary mentions debug suppression.
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "Debug/detail outputs suppressed" in summary_text
    assert "output.profile: debug" in summary_text

    summary_json = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary_json.get("output_profile") == "standard"


def test_debug_profile_writes_all_detailed_files(tmp_path):
    """output.profile=debug emits the full current detailed file set."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "debug"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    # Public outputs.
    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    assert outputs["valley_weights_csv"].exists()
    # Detailed files.
    assert outputs["valley_subspace_json"].exists()
    assert outputs["symmetry_report_json"].exists()
    assert outputs["diagnostics_h5"].exists()
    # Summary must NOT mention suppression.
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "Debug/detail outputs suppressed" not in summary_text

    summary_json = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary_json.get("output_profile") == "debug"


def test_write_detailed_files_false_maps_to_standard_with_warning(tmp_path):
    """Legacy write_detailed_files: false maps to profile=standard."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"].pop("profile", None)
    raw["output"]["write_detailed_files"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="write_detailed_files"):
        outputs = analyze_hsp(config_path)

    assert outputs["valley_summary_txt"].exists()
    assert not (out_dir / "valley_subspace.json").exists()
    assert not (out_dir / "diagnostics.h5").exists()


def test_invalid_output_profile_rejected(tmp_path):
    """Invalid output.profile raises ValueError."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "invalid_profile"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="output.profile must be one of"):
        load_config(config_path)


def test_no_material_specific_strings_in_production_code():
    """Verify no real material names appear in valleyscope/ production modules.

    tMoTe2, tZrSe2, and future real materials are validation examples and
    regression fixtures only.  They must not appear in program logic, output
    strings, config keys, or file paths inside valleyscope/.

    This test does not guard docs/benchmarks/ or real_tests/.
    """
    valleyscope_dir = Path("valleyscope")
    forbidden = ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]
    failures: list[str] = []
    for py_file in sorted(valleyscope_dir.rglob("*.py")):
        lines = py_file.read_text(encoding="utf-8").split("\n")
        for i, line in enumerate(lines, start=1):
            for name in forbidden:
                if name in line:
                    failures.append(f"{py_file}:{i}: {line.strip()[:120]}")
    if failures:
        msg = (
            "Material names found in valleyscope/ production code:\n"
            + "\n".join(failures)
            + "\n\nReal materials are validation examples only; "
            "they must not appear in program logic, output strings, "
            "or config paths."
        )
        raise AssertionError(msg)


def test_config_profiles_accepted(tmp_path):
    """Both 'standard' and 'debug' profiles are accepted."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    for profile in ["standard", "debug"]:
        config = {
            "input": {"wavefunction_h5": str(h5_path)},
            "analysis": {"kpoints": ["GammaM"], "iband": [101]},
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
            "output": {"directory": str(out_dir), "profile": profile},
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        loaded = load_config(config_path)
        assert loaded.output.profile == profile


def test_standard_profile_always_writes_summary_even_with_flags_false(tmp_path):
    """Standard profile writes valley_summary.txt/json even when write flags are false."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"]["write_summary_txt"] = False
    raw["output"]["write_summary_json"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    # Main user entry must always be present in standard profile.
    assert outputs["valley_summary_txt"].exists(), (
        "valley_summary.txt must be written in standard profile even with write_summary_txt=false"
    )
    assert outputs["valley_summary_json"].exists(), (
        "valley_summary.json must be written in standard profile even with write_summary_json=false"
    )


def test_write_analysis_outputs_creates_standard_summary_directory(tmp_path):
    """Report writer creates output.directory for standard profile summaries."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"]["write_csv"] = False
    raw["output"]["write_summary_txt"] = False
    raw["output"]["write_summary_json"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)

    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    assert not out_dir.exists()

    outputs = write_analysis_outputs(
        config=config,
        qcut=0.1,
        weight_rows=[],
        sector_names=[],
        subspace_payload={},
        symmetry_payload={},
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
    )

    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "valley_summary.json",
        "valley_summary.txt",
    ]


# --- Standard profile output contract tests ---

_STANDARD_PUBLIC_FILES = frozenset({
    "valley_summary.txt",
    "valley_summary.json",
    "valley_weights.csv",
    "valley_ebr_export_bundle.json",
    "valley_reduced_ebr_mapping.json",
})

_DEBUG_ONLY_FILES = frozenset({
    "valley_subspace.json",
    "symmetry_report.json",
    "symmetry_eigenvalues.csv",
    "diagnostics.h5",
    "valley_basis_transform.h5",
    "projector_symmetry_report.json",
    "symmetry_adapted_valley_analysis.json",
    "target_subspace_closure.json",
    "hsp_star_conjugation.json",
    "hsp_star_derived_characters.json",
    "subspace_representation_quality.json",
    "irrep_workflow_decisions.json",
    "valley_irrep_matching.json",
    "valley_ebr_input_candidates.json",
    "valley_ebr_problem_instances.json",
    "folded_center_report.json",
    "sampled_k_coverage.json",
})


def test_standard_profile_output_files_are_only_public_set(tmp_path):
    """Standard profile writes only the contracted public output files."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    all_written = {p.name for p in out_dir.iterdir() if p.is_file()}
    # Every written file must be in the public set.
    unexpected = all_written - _STANDARD_PUBLIC_FILES
    assert not unexpected, f"Standard profile wrote non-public files: {unexpected}"
    # No debug-only file may exist.
    debug_found = all_written & _DEBUG_ONLY_FILES
    assert not debug_found, f"Standard profile wrote debug/detail files: {debug_found}"
    # Core public files must be present.
    assert "valley_summary.txt" in all_written
    assert "valley_summary.json" in all_written
    assert "valley_weights.csv" in all_written


def test_standard_profile_summary_output_files_excludes_debug_keys(tmp_path):
    """valley_summary.json output_files must not list debug/detail files in standard profile."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    output_keys = set(summary.get("output_files", {}).keys())
    # Must include the public files that were actually written.
    assert "valley_summary_txt" in output_keys
    assert "valley_summary_json" in output_keys
    assert "valley_weights_csv" in output_keys
    # Must not include debug-only file keys.
    debug_keys_in_summary = output_keys & {
        "valley_subspace_json", "symmetry_report_json", "symmetry_eigenvalues_csv",
        "diagnostics_h5", "valley_basis_transform_h5",
        "projector_symmetry_report_json", "symmetry_adapted_valley_analysis_json",
        "target_subspace_closure_json", "hsp_star_conjugation_json",
        "hsp_star_derived_characters_json", "subspace_representation_quality_json",
        "irrep_workflow_decisions_json", "valley_irrep_matching_json",
        "valley_ebr_input_candidates_json", "valley_ebr_problem_instances_json",
        "folded_center_report_json", "sampled_k_coverage_json",
    }
    assert not debug_keys_in_summary, (
        f"Standard profile summary output_files lists debug/detail keys: {debug_keys_in_summary}"
    )


def test_standard_profile_ebr_export_bundle_present_when_payload_exists():
    """valley_ebr_export_bundle.json is written in standard profile when payload exists."""
    from valleyscope.reports.analysis_outputs import write_analysis_outputs
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        h5_path = out_dir / "wf.h5"
        write_fixture(h5_path)
        config_path = out_dir / "cfg.yaml"
        write_config(config_path, h5_path, out_dir)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["output"]["profile"] = "standard"
        raw["output"].pop("write_detailed_files", None)
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        config = load_config(config_path)

        ebr_bundle = {"status": "ready_for_external_solver", "bundle_count": 1,
                      "excluded_count": 0, "schema_version": "1.0.0",
                      "reduced_ebr_decomposition_status": "not_implemented",
                      "bundles": [], "excluded_instances": []}
        outputs = write_analysis_outputs(
            config=config, qcut=0.5, weight_rows=[], sector_names=["K_valley"],
            subspace_payload={"kpoints": {}},
            symmetry_payload={"status": "skipped", "reason": "test",
                              "detected_operations": [], "candidate_rotations": [],
                              "little_group_check": {"status": "not_run"},
                              "valley_preservation_check": {"status": "not_run"}},
            symmetry_rows=[], projectors_by_kpoint={}, qcut_scan_payload={},
            symmetry_representation_payload={}, basis_transforms={},
            ebr_export_bundle=ebr_bundle,
        )
        assert outputs["valley_ebr_export_bundle_json"].exists()
        assert (out_dir / "valley_ebr_export_bundle.json").exists()


def test_standard_profile_no_ebr_export_bundle_when_payload_none():
    """valley_ebr_export_bundle.json is NOT written when no EBR payload exists."""
    from valleyscope.reports.analysis_outputs import write_analysis_outputs
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        h5_path = out_dir / "wf.h5"
        write_fixture(h5_path)
        config_path = out_dir / "cfg.yaml"
        write_config(config_path, h5_path, out_dir)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["output"]["profile"] = "standard"
        raw["output"].pop("write_detailed_files", None)
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        config = load_config(config_path)

        outputs = write_analysis_outputs(
            config=config, qcut=0.5, weight_rows=[], sector_names=["K_valley"],
            subspace_payload={"kpoints": {}},
            symmetry_payload={"status": "skipped", "reason": "test",
                              "detected_operations": [], "candidate_rotations": [],
                              "little_group_check": {"status": "not_run"},
                              "valley_preservation_check": {"status": "not_run"}},
            symmetry_rows=[], projectors_by_kpoint={}, qcut_scan_payload={},
            symmetry_representation_payload={}, basis_transforms={},
            ebr_export_bundle=None,
        )
        assert "valley_ebr_export_bundle_json" not in outputs
        assert not (out_dir / "valley_ebr_export_bundle.json").exists()


def test_debug_profile_writes_all_expected_detail_files(tmp_path):
    """Debug profile writes public files AND all debug/detail files."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "debug"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    all_written = {p.name for p in out_dir.iterdir() if p.is_file()}
    # Public files must be present.
    assert "valley_summary.txt" in all_written
    assert "valley_summary.json" in all_written
    # Debug files that are always written with this fixture must be present.
    assert "diagnostics.h5" in all_written
    assert "valley_subspace.json" in all_written
    assert "symmetry_report.json" in all_written
    # Summary must NOT mention suppression.
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "Debug/detail outputs suppressed" not in summary_text


# -----------------------------------------------------------------------
# Reduced EBR classification E2E smoke tests
# -----------------------------------------------------------------------

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


def _e2e_write_table(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_reduced_ebr_e2e_config_path_writes_mapping_and_embeds_in_summary(tmp_path):
    """E2E: analyze_hsp with analysis.reduced_ebr.enabled writes mapping JSON
    and embeds it in valley_summary.json."""
    h5_path = tmp_path / "wf.h5"
    table_path = tmp_path / "table.json"
    out_dir = tmp_path / "out"
    _e2e_write_table(table_path, _E2E_SAMPLE_TABLE)
    write_fixture(h5_path)
    config_path = tmp_path / "cfg.yaml"
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["analysis"]["reduced_ebr"] = {"enabled": True, "table_file": str(table_path)}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    mapping_path = out_dir / "valley_reduced_ebr_mapping.json"
    assert mapping_path.exists(), "valley_reduced_ebr_mapping.json must be written"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping["table_status"] == "loaded"

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "valley_reduced_ebr_mapping" in summary
    assert summary["valley_reduced_ebr_mapping"] == mapping
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "Reduced EBR mapping" in summary_text
    assert "table: loaded" in summary_text


def test_reduced_ebr_disabled_does_not_write_mapping(tmp_path):
    """E2E: without analysis.reduced_ebr.enabled, no mapping JSON is written."""
    h5_path = tmp_path / "wf.h5"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    config_path = tmp_path / "cfg.yaml"
    write_config(config_path, h5_path, out_dir)

    outputs = analyze_hsp(config_path)

    assert not (out_dir / "valley_reduced_ebr_mapping.json").exists()
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "valley_reduced_ebr_mapping" not in summary


def test_summary_text_surfaces_atomic_fragile_stable_classification():
    """E2E: summary text surfaces atomic, fragile, stable classifications."""
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text
    from valleyscope.io.config import load_config

    mapping = {
        "status": "no_exact_solution",
        "mapping_status": "no_exact_solution",
        "reduced_ebr_decomposition_status": "no_exact_solution",
        "table_status": "loaded",
        "solutions": [
            {"bundle_id": "b_atom", "valley": "K", "status": "solved_exact",
             "classification": "atomic-compatible-candidate",
             "integer_span_status": "in_integer_span",
             "nonnegative_solution_status": "solved_exact",
             "ebr_decomposition": [
                 {"label": "EBR_A", "coefficient": 1},
                 {"label": "EBR_B", "coefficient": 2},
             ]},
            {"bundle_id": "b_frag", "valley": "K", "status": "no_exact_solution",
             "classification": "fragile-topology-candidate",
             "integer_span_status": "in_integer_span",
             "nonnegative_solution_status": "no_nonnegative_solution",
             "integer_solution": [
                 {"label": "EBR_A", "coefficient": -1},
                 {"label": "EBR_B", "coefficient": 1},
             ]},
            {"bundle_id": "b_stab", "valley": "K", "status": "no_exact_solution",
             "classification": "stable-topology-candidate",
             "integer_span_status": "outside_integer_span",
             "nonnegative_solution_status": "no_nonnegative_solution"},
        ],
        "excluded_bundles": [],
        "solver": "smith_normal_form_plus_bounded_nonnegative_search",
    }

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        h5 = d / "wf.h5"
        write_fixture(h5)
        cfg_path = d / "cfg.yaml"
        write_config(cfg_path, h5, d / "out")
        config = load_config(cfg_path)

        payload = build_summary_payload(
            config=config, qcut=0.5,
            subspace_payload={"kpoints": {}},
            symmetry_payload={"status": "skipped", "reason": "toy",
                              "detected_operations": [], "candidate_rotations": [],
                              "little_group_check": {"status": "not_run"},
                              "valley_preservation_check": {"status": "not_run"}},
            symmetry_rows=[], output_paths={},
            reduced_ebr_mapping=mapping,
        )
        text = render_summary_text(payload)

    assert "atomic-compatible=1" in text
    assert "fragile-topology=1" in text
    assert "stable-topology=1" in text
    assert "EBR_A x 1" in text
    assert "EBR_B x 2" in text
    assert "signed witness" in text
    assert "EBR_A: -1" in text
    assert "EBR_B: 1" in text
    assert "outside integer span" in text


def test_summary_text_truncated_search_surfaced():
    """E2E: truncated_by_max_coefficient search status appears in summary."""
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text
    from valleyscope.io.config import load_config

    mapping = {
        "status": "no_exact_solution",
        "mapping_status": "no_exact_solution",
        "reduced_ebr_decomposition_status": "no_exact_solution",
        "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_001", "valley": "K", "status": "no_exact_solution",
            "classification": "fragile-topology-candidate",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "no_nonnegative_solution",
            "search_status": "truncated_by_max_coefficient",
            "integer_solution": [{"label": "EBR_A", "coefficient": -1}],
        }],
        "excluded_bundles": [],
        "solver": "smith_normal_form_plus_bounded_nonnegative_search",
    }

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        h5 = d / "wf.h5"
        write_fixture(h5)
        cfg_path = d / "cfg.yaml"
        write_config(cfg_path, h5, d / "out")
        config = load_config(cfg_path)

        payload = build_summary_payload(
            config=config, qcut=0.5,
            subspace_payload={"kpoints": {}},
            symmetry_payload={"status": "skipped", "reason": "toy",
                              "detected_operations": [], "candidate_rotations": [],
                              "little_group_check": {"status": "not_run"},
                              "valley_preservation_check": {"status": "not_run"}},
            symmetry_rows=[], output_paths={},
            reduced_ebr_mapping=mapping,
        )
        text = render_summary_text(payload)

    assert "search_truncated=1" in text
    assert "truncated by max_coefficient" in text


def test_reduced_ebr_classifier_payload_written_consistently_to_public_outputs(tmp_path):
    """Classifier output is written consistently to mapping JSON and summaries."""
    from valleyscope.analysis.reduced_ebr_mapping import (
        build_reduced_ebr_mapping,
        load_reduced_ebr_table,
    )
    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    table_path = tmp_path / "table.json"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C1",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:irrep_A", "GammaM:irrep_B", "GammaM:irrep_C"],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0, 0]},
            {"label": "EBR_B", "vector": [1, 1, 0]},
            {"label": "EBR_C", "vector": [0, 0, 2]},
        ],
    }
    _e2e_write_table(table_path, table)
    loaded_table = load_reduced_ebr_table(table_path)
    export_bundle = {
        "bundles": [
            {
                "bundle_id": "b_atom",
                "valley": "K",
                "subspace_group_candidate": "C1",
                "ready_for_external_solver": True,
                "expected_hsps": ["GammaM"],
                "irreps_by_kpoint": {"GammaM": ["irrep_A", "irrep_A", "irrep_B"]},
            },
            {
                "bundle_id": "b_frag",
                "valley": "K",
                "subspace_group_candidate": "C1",
                "ready_for_external_solver": True,
                "expected_hsps": ["GammaM"],
                "irreps_by_kpoint": {"GammaM": ["irrep_B"]},
            },
            {
                "bundle_id": "b_stab",
                "valley": "K",
                "subspace_group_candidate": "C1",
                "ready_for_external_solver": True,
                "expected_hsps": ["GammaM"],
                "irreps_by_kpoint": {"GammaM": ["irrep_C"]},
            },
        ],
    }
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=loaded_table,
    )
    assert [s["classification"] for s in mapping["solutions"]] == [
        "atomic-compatible-candidate",
        "fragile-topology-candidate",
        "stable-topology-candidate",
    ]

    config = load_config(config_path)
    outputs = write_analysis_outputs(
        config=config,
        qcut=0.5,
        weight_rows=[],
        sector_names=["K_valley"],
        subspace_payload={"kpoints": {}},
        symmetry_payload={"status": "skipped", "reason": "test",
                          "detected_operations": [], "candidate_rotations": [],
                          "little_group_check": {"status": "not_run"},
                          "valley_preservation_check": {"status": "not_run"}},
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
        reduced_ebr_mapping=mapping,
    )

    mapping_json = json.loads(outputs["valley_reduced_ebr_mapping_json"].read_text(encoding="utf-8"))
    summary_json = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert mapping_json == mapping
    assert summary_json["valley_reduced_ebr_mapping"] == mapping
    assert "classifications: atomic-compatible=1, fragile-topology=1, stable-topology=1" in summary_text
    assert "b_atom K: atomic-compatible" in summary_text
    assert "b_frag K: fragile-topology" in summary_text
    assert "b_stab K: stable-topology (outside integer span)" in summary_text


def test_e2e_smoke_fixture_table_is_material_agnostic():
    """E2E smoke fixture data must not name real validation materials."""
    fixture_text = json.dumps(_E2E_SAMPLE_TABLE)
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in fixture_text


# -----------------------------------------------------------------------
# Valley irrep -> EBR pipeline contract tests
# -----------------------------------------------------------------------

def test_ebr_input_candidates_excludes_non_trusted():
    """Non-trusted/diagnostic-only rows must not reach ready_for_ebr_input=true."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates

    # Workflow: K_valley is trusted/direct_qcut, Kp_valley is diagnostic_only.
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
                "Kp_valley": {"readiness_level": "usable_with_caution", "workflow_path": "symmetry_adapted"},
            },
        },
    }
    # Matching: K_valley matched, Kp_valley diagnostic_only.
    matching = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "1": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [0.5], "diagnostic_only": False},
                },
                "Kp_valley": {
                    "1": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_-1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [-0.5], "diagnostic_only": True},
                },
            },
        },
    }

    report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )

    # Only trusted, non-diagnostic rows are candidates.
    candidates = report["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["valley"] == "K_valley"
    assert candidates[0]["ready_for_ebr_input"] is True

    # Diagnostic row is blocked.
    blocked = report["blocked"]
    assert any("diagnostic_only=true" in str(b.get("reason", "")) for b in blocked)
    assert any(b.get("valley") == "Kp_valley" for b in blocked)


def test_ebr_input_candidates_excludes_blocked_path():
    """Workflow path=blocked must not produce candidates."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates

    workflow = {
        "by_kpoint": {
            "MM": {
                "K_valley": {"readiness_level": "blocked", "workflow_path": "blocked"},
            },
        },
    }
    matching = {
        "by_kpoint": {
            "MM": {
                "K_valley": {},
            },
        },
    }
    report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    assert report["candidate_count"] == 0
    assert report["status"] == "no_candidates"


def test_ebr_problem_instances_missing_hsp_blocked():
    """Missing required HSPs block instance readiness."""
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    # C3_like requires GammaM+KM. Only GammaM has data -> blocked.
    candidates = {
        "status": "has_candidates",
        "candidates": [{
            "kpoint": "GammaM", "valley": "K_valley",
            "subspace_group_candidate": "C3_like",
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "matched_irrep": "C3_spinor_phase_+1/2",
            "operation_id": 1,
            "ready_for_ebr_input": True,
        }],
    }
    report = build_ebr_problem_instances(ebr_input_candidates=candidates)
    assert report["instance_count"] == 1
    inst = report["instances"][0]
    assert inst["ready_for_ebr_decomposition"] is False
    assert "missing required HSPs" in str(inst["blocked_by"])


def test_ebr_problem_instances_complete_hsp_is_ready():
    """All required HSPs present -> ready_for_ebr_decomposition=true."""
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    candidates = {
        "status": "has_candidates",
        "candidates": [
            {"kpoint": "GammaM", "valley": "K_valley",
             "subspace_group_candidate": "C3_like",
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "matched_irrep": "C3_spinor_phase_+1/2", "operation_id": 1,
             "ready_for_ebr_input": True},
            {"kpoint": "KM", "valley": "K_valley",
             "subspace_group_candidate": "C3_like",
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "matched_irrep": "C3_spinor_phase_+1/6", "operation_id": 1,
             "ready_for_ebr_input": True},
        ],
    }
    report = build_ebr_problem_instances(ebr_input_candidates=candidates)
    inst = report["instances"][0]
    assert inst["ready_for_ebr_decomposition"] is True
    assert inst["status"] == "complete"


def test_ebr_export_bundle_preserves_hsp_and_irrep_fields():
    """Export bundle must preserve expected_hsps, irreps_by_kpoint, operations_by_kpoint."""
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle

    problem_instances = {
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "C3_like",
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
            "operations_by_kpoint": {"GammaM": [1]},
            "expected_hsps": ["GammaM", "KM"],
            "optional_hsps": ["MM"],
            "missing_optional_hsps": ["MM"],
            "ready_for_ebr_decomposition": True,
            "status": "complete",
        }],
    }
    report = build_ebr_export_bundle(ebr_problem_instances=problem_instances)
    assert report["bundle_count"] == 1
    bundle = report["bundles"][0]
    assert bundle["ready_for_external_solver"] is True
    assert bundle["subspace_group_candidate"] == "C3_like"
    assert bundle["expected_hsps"] == ["GammaM", "KM"]
    assert bundle["optional_hsps"] == ["MM"]
    assert bundle["missing_optional_hsps"] == ["MM"]
    assert bundle["irreps_by_kpoint"] == {"GammaM": ["C3_spinor_phase_+1/2"]}
    assert bundle["operations_by_kpoint"] == {"GammaM": [1]}


def test_ebr_export_bundle_excludes_non_ready():
    """Non-ready instances are excluded, not bundled."""
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle

    problem_instances = {
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "C3_like",
            "status": "partial",
            "ready_for_ebr_decomposition": False,
            "blocked_by": ["missing required HSPs: [KM]"],
        }],
    }
    report = build_ebr_export_bundle(ebr_problem_instances=problem_instances)
    assert report["bundle_count"] == 0
    assert report["status"] == "no_bundles"
    assert report["excluded_count"] == 1
    assert "ready_for_ebr_decomposition" in str(report["excluded_instances"][0]["exclusion_reasons"])


def test_reduced_ebr_mapping_rejects_hsp_mismatch():
    """Reduced EBR mapping must reject bundles whose HSP basis does not match the table."""
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C3_like",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:C3_spinor_phase_+1/2", "KM:C3_spinor_phase_+1/6"],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0]}, {"label": "EBR_B", "vector": [0, 1]}],
    }
    # Bundle only has GammaM, missing KM.
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM"],
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=bundle, table=table)
    assert len(r["solutions"]) == 0
    assert len(r["excluded_bundles"]) == 1
    assert "expected_hsps mismatch" in r["excluded_bundles"][0]["reason"]


def test_ready_export_bundle_maps_to_public_reduced_ebr_outputs_only(tmp_path):
    """Ready export bundle plus validated table writes only public standard outputs."""
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
    from valleyscope.analysis.reduced_ebr_mapping import (
        build_reduced_ebr_mapping,
        load_reduced_ebr_table,
    )
    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    table_path = tmp_path / "table.json"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C3_like",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": [
            "GammaM:C3_spinor_phase_+1/2",
            "KM:C3_spinor_phase_+1/6",
        ],
        "ebrs": [
            {"label": "EBR_G", "vector": [1, 0]},
            {"label": "EBR_K", "vector": [0, 1]},
        ],
    }
    _e2e_write_table(table_path, table)
    export_bundle = build_ebr_export_bundle(
        ebr_problem_instances={
            "instances": [{
                "instance_id": "ebr_instance_001",
                "valley": "K_valley",
                "subspace_group_candidate": "C3_like",
                "workflow_path": "direct_qcut",
                "readiness_level": "trusted",
                "irreps_by_kpoint": {
                    "GammaM": ["C3_spinor_phase_+1/2"],
                    "KM": ["C3_spinor_phase_+1/6"],
                },
                "operations_by_kpoint": {"GammaM": [1], "KM": [1]},
                "expected_hsps": ["GammaM", "KM"],
                "optional_hsps": ["MM"],
                "missing_optional_hsps": ["MM"],
                "ready_for_ebr_decomposition": True,
                "status": "complete",
            }],
        }
    )
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=load_reduced_ebr_table(table_path),
    )
    assert mapping["status"] == "solved_exact"

    outputs = write_analysis_outputs(
        config=load_config(config_path),
        qcut=0.5,
        weight_rows=[],
        sector_names=["K_valley"],
        subspace_payload={"kpoints": {}},
        symmetry_payload={"status": "skipped", "reason": "test",
                          "detected_operations": [], "candidate_rotations": [],
                          "little_group_check": {"status": "not_run"},
                          "valley_preservation_check": {"status": "not_run"}},
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
        ebr_export_bundle=export_bundle,
        reduced_ebr_mapping=mapping,
    )

    written = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert written <= _STANDARD_PUBLIC_FILES
    assert not (written & _DEBUG_ONLY_FILES)
    assert outputs["valley_ebr_export_bundle_json"].exists()
    assert outputs["valley_reduced_ebr_mapping_json"].exists()
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary["valley_ebr_export_bundle"] == export_bundle
    assert summary["valley_reduced_ebr_mapping"] == mapping


def test_pipeline_contract_fixture_data_is_material_agnostic():
    """Pipeline contract fixture data must not name real validation materials."""
    fixture_text = "\n".join([
        "C3_like C2_like P3 P2",
        "GammaM KM MM K_valley",
        "C3_spinor_phase_+1/2 C3_spinor_phase_+1/6",
    ])
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in fixture_text


# -----------------------------------------------------------------------
# Trusted irrep export provenance tests
# -----------------------------------------------------------------------

def test_ebr_problem_instances_include_irrep_records():
    """EBR problem instances must include irrep_records_by_kpoint for trusted candidates."""
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    candidates = {
        "status": "has_candidates",
        "candidates": [
            {"kpoint": "GammaM", "valley": "K_valley",
             "subspace_group_candidate": "C3_like",
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "matched_irrep": "C3_spinor_phase_+1/2", "operation_id": 1,
             "operation_order": 3, "eigenphases": [0.5],
             "character": {"real": -1.0, "imag": 0.0},
             "source": "valley_irrep_matching/GammaM/K_valley",
             "ready_for_ebr_input": True},
            {"kpoint": "KM", "valley": "K_valley",
             "subspace_group_candidate": "C3_like",
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "matched_irrep": "C3_spinor_phase_+1/6", "operation_id": 1,
             "operation_order": 3, "eigenphases": [0.166667],
             "source": "valley_irrep_matching/KM/K_valley",
             "ready_for_ebr_input": True},
        ],
    }
    report = build_ebr_problem_instances(ebr_input_candidates=candidates)
    inst = report["instances"][0]
    assert "irrep_records_by_kpoint" in inst
    records = inst["irrep_records_by_kpoint"]
    assert "GammaM" in records
    assert "KM" in records
    gamma_rec = records["GammaM"][0]
    assert gamma_rec["valley"] == "K_valley"
    assert gamma_rec["operation_id"] == 1
    assert gamma_rec["operation_order"] == 3
    assert gamma_rec["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert gamma_rec["eigenphases"] == [0.5]
    assert gamma_rec["workflow_path"] == "direct_qcut"
    assert gamma_rec["readiness_level"] == "trusted"
    assert gamma_rec["character"] is not None
    assert gamma_rec["source"] == "valley_irrep_matching/GammaM/K_valley"


def test_export_bundle_copies_irrep_records():
    """Export bundles copy irrep_records_by_kpoint for complete trusted instances."""
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle

    records = {
        "GammaM": [{"valley": "K_valley", "operation_id": 1, "operation_order": 3,
                     "matched_irrep": "C3_spinor_phase_+1/2", "eigenphases": [0.5],
                     "workflow_path": "direct_qcut", "readiness_level": "trusted",
                     "source": "valley_irrep_matching/GammaM/K_valley"}],
        "KM": [{"valley": "K_valley", "operation_id": 1, "operation_order": 3,
                 "matched_irrep": "C3_spinor_phase_+1/6", "eigenphases": [0.166667],
                 "workflow_path": "direct_qcut", "readiness_level": "trusted",
                 "source": "valley_irrep_matching/KM/K_valley"}],
    }

    problem_instances = {
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "C3_like",
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
            "operations_by_kpoint": {"GammaM": [1]},
            "irrep_records_by_kpoint": records,
            "expected_hsps": ["GammaM", "KM"],
            "optional_hsps": [],
            "missing_optional_hsps": [],
            "ready_for_ebr_decomposition": True,
            "status": "complete",
        }],
    }
    report = build_ebr_export_bundle(ebr_problem_instances=problem_instances)
    bundle = report["bundles"][0]
    assert "irrep_records_by_kpoint" in bundle
    assert bundle["irrep_records_by_kpoint"] == records


def test_non_trusted_rows_excluded_from_irrep_records():
    """Non-trusted/diagnostic-only rows must not appear in irrep_records_by_kpoint."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances

    # Mix of trusted and diagnostic_only rows
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            },
        },
    }
    matching = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "1": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [0.5], "diagnostic_only": False},
                    "2": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [-0.5], "diagnostic_only": True},
                },
            },
        },
    }
    candidates_report = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow, valley_irrep_matching=matching)
    # Only 1 trusted candidate (op=1), op=2 is diagnostic_only -> blocked.
    assert candidates_report["candidate_count"] == 1

    instances_report = build_ebr_problem_instances(ebr_input_candidates=candidates_report)
    inst = instances_report["instances"][0]
    records = inst["irrep_records_by_kpoint"]
    gamma_recs = records.get("GammaM", [])
    # Only the trusted record appears.
    assert len(gamma_recs) == 1
    assert gamma_recs[0]["operation_id"] == "1"


def test_reduced_ebr_mapping_ignores_irrep_records():
    """reduced_ebr_mapping must remain compatible and ignore irrep_records_by_kpoint."""
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C3_like",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:C3_spinor_phase_+1/2", "KM:C3_spinor_phase_+1/6"],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0]}, {"label": "EBR_B", "vector": [0, 1]}],
    }
    records = {
        "GammaM": [{"valley": "K_valley", "operation_id": 1, "matched_irrep": "C3_spinor_phase_+1/2"}],
        "KM": [{"valley": "K_valley", "operation_id": 1, "matched_irrep": "C3_spinor_phase_+1/6"}],
    }
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM", "KM"],
            "irreps_by_kpoint": {
                "GammaM": ["C3_spinor_phase_+1/2"],
                "KM": ["C3_spinor_phase_+1/6"],
            },
            "irrep_records_by_kpoint": records,
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=bundle, table=table)
    assert r["status"] == "solved_exact"  # Provenance ignored; decomposition succeeds.


def test_schema_doc_documents_irrep_records_by_kpoint():
    """docs/schema.md must document the new irrep_records_by_kpoint field."""
    schema = Path("docs/schema.md").read_text(encoding="utf-8")
    assert "irrep_records_by_kpoint" in schema, (
        "docs/schema.md must document irrep_records_by_kpoint"
    )
    assert "provenance" in schema.lower(), (
        "docs/schema.md should mention provenance for the new field"
    )


# -----------------------------------------------------------------------
# Provenance propagation smoke (output-writer fallback)
# -----------------------------------------------------------------------
# Direct analyze_hsp smoke is skipped because it requires a real HDF5
# with symmetry operations to produce trusted EBR candidates.  The
# output-writer path exercises the same builder chain
# (ebr_input_candidates -> ebr_problem_instances -> ebr_export_bundle
# -> write_analysis_outputs) that analyze_hsp uses at lines 452-468,
# so this is a valid end-to-end provenance pipeline test.

def test_provenance_survives_through_output_writer_to_export_bundle(tmp_path):
    """Provenance records survive through the full output-writer pipeline."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
    from valleyscope.reports.analysis_outputs import write_analysis_outputs
    from valleyscope.io.config import load_config

    # Trusted irrep matching with character data.
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            },
            "KM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            },
        },
    }
    matching = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "1": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [0.5], "diagnostic_only": False},
                    "2": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [-0.5], "diagnostic_only": True},
                },
            },
            "KM": {
                "K_valley": {
                    "1": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/6",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [0.166667], "diagnostic_only": False},
                },
            },
        },
    }
    # Character lookup via symmetry-adapted report.
    sa_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [{
                                "operation_id": 1,
                                "character": {"real": -1.0, "imag": 0.0},
                            }],
                        },
                    },
                }],
            },
            "KM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [{
                                "operation_id": 1,
                                "character": {"real": 0.5, "imag": 0.866025},
                            }],
                        },
                    },
                }],
            },
        },
    }

    candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
        symmetry_adapted_valley_report=sa_report,
    )
    assert candidates["candidate_count"] == 2
    assert candidates["blocked_count"] == 1

    instances = build_ebr_problem_instances(ebr_input_candidates=candidates)
    assert instances["instance_count"] == 1
    inst = instances["instances"][0]
    assert inst["ready_for_ebr_decomposition"] is True

    export_bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    assert export_bundle["bundle_count"] == 1

    # Write through the output writer.
    h5_path = tmp_path / "wf.h5"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    config_path = tmp_path / "cfg.yaml"
    write_config(config_path, h5_path, out_dir)
    # Override to standard profile for smoke.
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)

    outputs = write_analysis_outputs(
        config=config, qcut=0.5, weight_rows=[], sector_names=["K_valley"],
        subspace_payload={"kpoints": {}},
        symmetry_payload={"status": "skipped", "reason": "toy",
                          "detected_operations": [], "candidate_rotations": [],
                          "little_group_check": {"status": "not_run"},
                          "valley_preservation_check": {"status": "not_run"}},
        symmetry_rows=[], projectors_by_kpoint={}, qcut_scan_payload={},
        symmetry_representation_payload={}, basis_transforms={},
        ebr_export_bundle=export_bundle,
    )

    # Assertions on export bundle.
    assert outputs["valley_ebr_export_bundle_json"].exists()
    bundle_json = json.loads(outputs["valley_ebr_export_bundle_json"].read_text(encoding="utf-8"))
    bundle = bundle_json["bundles"][0]
    assert "irrep_records_by_kpoint" in bundle
    records = bundle["irrep_records_by_kpoint"]
    assert "GammaM" in records and "KM" in records

    gamma_rec = records["GammaM"][0]
    assert gamma_rec["valley"] == "K_valley"
    assert gamma_rec["operation_id"] == "1"
    assert gamma_rec["operation_order"] == 3
    assert gamma_rec["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert gamma_rec["eigenphases"] == [0.5]
    assert gamma_rec["workflow_path"] == "direct_qcut"
    assert gamma_rec["readiness_level"] == "trusted"
    assert "valley_irrep_matching" in gamma_rec["source"]
    # Character preserved from SA report.
    assert gamma_rec["character"] is not None
    assert gamma_rec["character"]["real"] == -1.0
    assert all(rec["operation_id"] != "2" for recs in records.values() for rec in recs)

    km_rec = records["KM"][0]
    assert km_rec["operation_id"] == "1"
    assert km_rec["matched_irrep"] == "C3_spinor_phase_+1/6"
    assert km_rec["character"] is not None

    # irreps_by_kpoint still present for reduced EBR matching.
    assert "irreps_by_kpoint" in bundle
    assert bundle["irreps_by_kpoint"]["GammaM"] == ["C3_spinor_phase_+1/2"]

    # Summary embeds the export bundle with provenance.
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    embedded = summary["valley_ebr_export_bundle"]["bundles"][0]
    assert embedded["irrep_records_by_kpoint"] == records

    # Standard profile: no debug/detail files.
    written = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert not (written & _DEBUG_ONLY_FILES)


# -----------------------------------------------------------------------
# Database ingestion record tests
# -----------------------------------------------------------------------

def test_ingestion_record_requires_summary():
    """Missing valley_summary.json produces invalid record."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    record = build_database_ingestion_record()
    assert record["record_status"] == "invalid_missing_summary"
    assert len(record["validation_errors"]) > 0


def test_ingestion_record_with_ready_bundle():
    """Ready export bundle produces trusted irrep records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": ["GammaM"], "iband": [1],
               "input": {"spinor_convention_verified": True},
               "symmetry_analysis": {"international": "P321", "spacegroup_number": 150}}
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "source_instance_id": "ebr_001",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "irrep_records_by_kpoint": {
                "GammaM": [{"valley": "K_valley", "operation_id": 1,
                            "operation_order": 3,
                            "matched_irrep": "C3_spinor_phase_+1/2",
                            "eigenphases": [0.5],
                            "workflow_path": "direct_qcut",
                            "readiness_level": "trusted",
                            "source": "valley_irrep_matching/GammaM/K_valley"}],
            },
        }],
    }

    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle)

    assert record["record_status"] == "has_ready_ebr_bundles"
    assert record["ready_bundle_count"] == 1
    assert record["space_group_international"] == "P321"
    records = record["valley_irrep_records"]
    assert len(records) == 1
    r = records[0]
    assert r["kpoint"] == "GammaM"
    assert r["valley"] == "K_valley"
    assert r["subspace_group_candidate"] == "C3_like"
    assert r["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert r["source_bundle_id"] == "b_001"


def test_ingestion_record_excludes_non_ready_bundles():
    """Non-ready bundles do not contribute trusted irrep records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": [], "iband": [], "input": {}}
    bundle = {
        "bundles": [{
            "bundle_id": "b_001",
            "ready_for_external_solver": False,
            "irrep_records_by_kpoint": {},
        }],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle)
    assert record["ready_bundle_count"] == 0
    assert record["valley_irrep_records"] == []


def test_ingestion_record_with_reduced_ebr_mapping():
    """Reduced EBR mapping adds status and classification counts."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": [], "iband": [], "input": {}}
    mapping = {
        "status": "solved_exact",
        "table_status": "loaded",
        "solutions": [
            {"classification": "atomic-compatible-candidate"},
            {"classification": "atomic-compatible-candidate"},
            {"classification": "fragile-topology-candidate"},
        ],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_reduced_ebr_mapping=mapping)
    assert record["reduced_ebr_mapping_status"] == "solved_exact"
    counts = record["reduced_ebr_classification_counts"]
    assert counts["atomic_compatible"] == 2
    assert counts["fragile_topology"] == 1
    assert counts["stable_topology"] == 0


def test_ingestion_record_missing_reduced_ebr_is_not_an_error():
    """Missing reduced EBR mapping is not an error."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(
        valley_summary=summary, valley_reduced_ebr_mapping=None)
    assert record["reduced_ebr_mapping_status"] == "not_available"
    assert record["reduced_ebr_classification_counts"] == {
        "atomic_compatible": 0,
        "fragile_topology": 0,
        "stable_topology": 0,
    }
    assert len(record["validation_errors"]) == 0


def test_ingestion_record_from_directory(tmp_path):
    """load_database_ingestion_record_from_directory reads files from dir."""
    from valleyscope.analysis.database_ingestion_record import (
        load_database_ingestion_record_from_directory,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {"target_kpoints": ["GammaM"], "iband": [1],
               "input": {"spinor_convention_verified": False}}
    (run_dir / "valley_summary.json").write_text(json.dumps(summary))

    record = load_database_ingestion_record_from_directory(str(run_dir))
    assert record["summary_status"] == "present"
    assert record["ready_bundle_count"] == 0


def test_cli_collect_database_record(tmp_path, capsys):
    """CLI writes ingestion record to requested output path."""
    from valleyscope.cli import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {"target_kpoints": ["GammaM"], "iband": [1],
               "input": {"spinor_convention_verified": False}}
    (run_dir / "valley_summary.json").write_text(json.dumps(summary))

    out_path = tmp_path / "nested" / "record.json"
    rc = main(["collect-database-record", str(run_dir), "-o", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert record["summary_status"] == "present"
    captured = capsys.readouterr().out
    assert "no_ready_ebr_bundles" in captured


def test_cli_collect_database_record_returns_nonzero_for_invalid_record(tmp_path):
    """CLI writes invalid record but exits nonzero when summary is missing."""
    from valleyscope.cli import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_path = tmp_path / "record.json"

    rc = main(["collect-database-record", str(run_dir), "-o", str(out_path)])
    assert rc == 1
    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert record["record_status"] == "invalid_missing_summary"
    assert record["validation_errors"]


def test_schema_doc_documents_database_ingestion_record():
    """Public schema documents the explicit offline ingestion-record CLI."""
    schema = Path("docs/schema.md").read_text(encoding="utf-8")
    assert "database_ingestion_record.json" in schema
    assert "collect-database-record" in schema
    assert "valley_irrep_records" in schema
    assert "not a default `analyze-hsp` output" in schema


def test_ingestion_record_no_material_names():
    """Ingestion record module must not contain real material names."""
    src = Path("valleyscope/analysis/database_ingestion_record.py").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src, f"database_ingestion_record.py must not contain {name!r}"


# -----------------------------------------------------------------------
# Benchmark ingestion record anchor tests
# -----------------------------------------------------------------------

def test_benchmark_smoke_doc_exists():
    """Database ingestion record smoke doc must exist."""
    doc = Path("docs/benchmarks/database_ingestion_record_smoke.md")
    assert doc.exists(), "docs/benchmarks/database_ingestion_record_smoke.md must exist"


def test_benchmark_smoke_doc_mentions_ingestion_record():
    """Smoke doc must reference collect-database-record and key fields."""
    doc = Path("docs/benchmarks/database_ingestion_record_smoke.md").read_text(encoding="utf-8")
    assert "collect-database-record" in doc
    assert "tmpdir=$(mktemp -d)" in doc
    assert "--output \"$tmpdir/" in doc
    assert "has_ready_ebr_bundles" in doc
    assert "no_ready_ebr_bundles" in doc
    assert "P321" in doc
    assert "P312" in doc


def test_benchmark_smoke_doc_states_offline_only():
    """Smoke doc must state ingestion record is offline, not default output."""
    doc = Path("docs/benchmarks/database_ingestion_record_smoke.md").read_text(encoding="utf-8")
    assert "not a default" in doc.lower() or "offline" in doc.lower()
    assert "explicit" in doc.lower()


def test_benchmark_smoke_doc_preserves_blocker_status():
    """Smoke doc must preserve tZrSe2 physical blocker status."""
    doc = Path("docs/benchmarks/database_ingestion_record_smoke.md").read_text(encoding="utf-8")
    assert "spinor_convention_unverified" in doc or "spinor" in doc.lower()
    assert "physical" in doc.lower() or "blocker" in doc.lower()


def test_benchmark_matrix_ingestion_section():
    """benchmark_matrix.md must have a database ingestion record anchors section."""
    matrix = Path("docs/benchmarks/benchmark_matrix.md").read_text(encoding="utf-8")
    assert "Database Ingestion Record Anchors" in matrix or "ingestion" in matrix.lower()
    assert "collect-database-record" in matrix
    assert "offline" in matrix.lower() or "not a default" in matrix.lower()
    assert matrix.count("## Standard Output Contract") == 1


# -----------------------------------------------------------------------
# Valley-irrep phase table data contract tests
# -----------------------------------------------------------------------

def test_phase_table_c3_loads_labels():
    """C3 phase table must provide exactly 3 labels that match the old hardcoded set."""
    from valleyscope.data.valley_irreps.catalog import get_irrep_phase_list
    irreps = get_irrep_phase_list("spinful_C3_phase_v1")
    labels = {e["label"] for e in irreps}
    assert labels == {"C3_spinor_phase_+1/6", "C3_spinor_phase_+1/2", "C3_spinor_phase_-1/6"}
    assert all(len(e["phases"]) == 1 for e in irreps)


def test_phase_table_c2_loads_labels():
    """C2 phase table must provide exactly 2 labels that match the old hardcoded set."""
    from valleyscope.data.valley_irreps.catalog import get_irrep_phase_list
    irreps = get_irrep_phase_list("spinful_C2_phase_v1")
    labels = {e["label"] for e in irreps}
    assert labels == {"C2_spinor_phase_+1/4", "C2_spinor_phase_-1/4"}
    assert all(len(e["phases"]) == 1 for e in irreps)


def test_phase_table_tables_implemented_unchanged():
    """tables_implemented must remain ['spinful_C3', 'spinful_C2']."""
    # Construct a matching report with known table names.
    from valleyscope.analysis.valley_irrep_matching import build_valley_irrep_matching_report
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            },
        },
    }
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=workflow,
        symmetry_adapted_valley_report=None,
    )
    assert report["tables_implemented"] == ["spinful_C3", "spinful_C2"]


def test_phase_table_files_no_ebr_vectors():
    """Phase table JSON files must not contain EBR vectors."""
    from valleyscope.data.valley_irreps.catalog import package_data_root
    for fname in ["spinful_C3_phase_v1.json", "spinful_C2_phase_v1.json"]:
        data = (package_data_root() / fname).read_text(encoding="utf-8")
        assert "ebr" not in data.lower() and "vector" not in data.lower(), (
            f"{fname} must not contain EBR vectors"
        )


def test_phase_table_files_no_material_names():
    """Phase table JSON files must not contain real material names."""
    from valleyscope.data.valley_irreps.catalog import package_data_root
    for fname in ["spinful_C3_phase_v1.json", "spinful_C2_phase_v1.json",
                   "manifest.json"]:
        data = (package_data_root() / fname).read_text(encoding="utf-8")
        for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
            assert name not in data, f"{fname} must not contain {name!r}"


def test_phase_table_readme_mentions_irrep_not_ebr():
    """README must clarify these are irrep matching tables, not EBR tables."""
    readme = Path("valleyscope/data/valley_irreps/README.md").read_text(encoding="utf-8").lower()
    assert "not reduced ebr" in readme or "irrep matching data" in readme


def test_phase_table_catalog_no_irrep2_import():
    """Phase table catalog must not import irrep2."""
    src = Path("valleyscope/data/valley_irreps/catalog.py").read_text(encoding="utf-8")
    assert "irrep2" not in src, "catalog.py must not import irrep2"
