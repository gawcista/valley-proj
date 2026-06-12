import csv
import json
import pathlib
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

from tests.helpers_io_workflow import (
    write_fixture,
    write_fixture_with_lattice,
    write_config,
    write_simple_poscar,
    write_square_poscar,
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


