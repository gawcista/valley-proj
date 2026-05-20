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
        "output": {"directory": str(out_dir)},
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
        "K_valley",
        "Kp_valley",
        "W_val",
        "P_v",
        "eta",
        "W_overlap",
        "W_res",
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
    assert "sector" not in readme.lower()
    assert "manifold" not in readme.lower()
    assert not re.search(r"\bV[123]\b", readme)
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
    assert "single-valley irrep" in readme
    assert "valley-preserving subgroup" in readme
    assert "irrep_results_by_kpoint" in readme
    assert "irrep_multiplicities" in readme
    assert "state_irrep_results" in readme
    assert "tMoTe2" not in readme
    assert "P321 No.150" not in readme
    assert "P3 No.143" not in readme
    assert "Benchmark:" not in readme
    assert "double-valued" in readme
    assert "`root_deviation_tol` and `D_valley_offdiag_tol` are numerical readiness thresholds" in readme
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
    assert not re.search(r"\bV[123]\b", readme)
    assert "Valley subspaces" in readme
    assert "Valley subspace analysis" in readme
    assert "Two-valley subspace" not in readme
    assert "S_min:              目标谷子空间权重下界" in readme
    assert "min_concentration:" in readme
    assert "assigned_valleys:" in readme
    assert "`not_derived`" in readme
    assert "`unreliable`" in readme
    assert "single-valley irrep" in readme
    assert "谷保持子群" in readme
    assert "irrep_results_by_kpoint" in readme
    assert "irrep_multiplicities" in readme
    assert "state_irrep_results" in readme
    assert "tMoTe2" not in readme
    assert "P321 No.150" not in readme
    assert "P3 No.143" not in readme
    assert "double-valued" in readme
    assert "valley_weights_adapted" in readme
    assert "assigned_valleys" in readme
    assert "`root_deviation_tol` 和 `D_valley_offdiag_tol` 是 numerical readiness thresholds" in readme
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
        "output": {"directory": str(out_dir), "write_json": True, "write_csv": True, "write_hdf5_basis_transform": True},
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
        or "valley-exchanging" in summary_text
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
    assert subgroup_report["status"] == "operation_set_only"
    assert subgroup_report["standard_group_match_status"] == "not_attempted"
    assert subgroup_report["by_kpoint"]["GammaM"]["closure_status"] == "empty"


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
    assert not (out_dir / "little_group_eigenvalues.csv").exists()
    assert not (out_dir / "little_group_representations.json").exists()
    assert not (out_dir / "symmetry_eigenvalues.csv").exists()
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
        symmetry_rows=[],
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
    assert text.index("GammaM: little group") < text.index("KM: little group")
    assert "valley-exchanging" in text


def test_summary_preserves_valley_little_group_inventory(tmp_path):
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
                "allowed_for_single_valley_representation": True,
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
            "valley_little_group_inventory": inventory,
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert summary["symmetry_analysis"]["valley_little_group_inventory"] == inventory


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
            "operation_id": 7,
            "kind": "C3",
            "order": 3,
            "basis": "valley_adapted",
            "character_raw": "1.000000+0.000000j",
            "character_valley": "0.500000+0.866025j",
            "topology_input_ready": True,
            "diagnostic_only": False,
            "accepted_for_single_valley_representation": True,
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
    assert "not universal physical constants" in thresholds["interpretation"]
    assert "do not loosen" in thresholds["recommended_action"]


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
            "output": {"directory": str(out_dir)},
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
        calls.append(bool(kwargs["generators_only"]))
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
    result = matching["irrep_results_by_kpoint"]["GammaM"]
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
    assert "GammaM: -GM5 x 1, -GM6 x 1" in summary_text
    assert "GammaM state irreps: state 0 -> -GM5, state 1 -> -GM6" in summary_text


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
    result = matching["irrep_results_by_kpoint"]["GammaM"]
    assert result["status"] == "missing_characters"
    assert result["irrep_multiplicities"] == {}
    assert result["missing_table_operation_indices"] == [3]
    assert result["state_irrep_assignment_status"] == "incomplete"


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
    result = matching["irrep_results_by_kpoint"]["GammaM"]
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
    result = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]["irrep_results_by_kpoint"]["GammaM"]
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
        "output": {"directory": str(out_dir)},
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
        "output": {"directory": str(out_dir)},
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
        "output": {"directory": str(tmp_path / "out")},
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
