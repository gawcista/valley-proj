"""Generated numerical Cn checks; not a globally certified valley Chern test."""

import json

import h5py
import numpy as np
import yaml

from valleyscope.workflows.analyze_hsp import analyze_hsp


# Reciprocal C6 acts on column coordinates by [[0,-1],[1,1]]. These stars
# are specified independently of the production symmetry/matching helpers.
_C6_STARS = {
    "GM": ([0, 0, 0], [[1, 1, 0], [-1, 2, 0], [-2, 1, 0],
                      [-1, -1, 0], [1, -2, 0], [2, -1, 0]]),
    "K": ([1 / 3, 1 / 3, 0], [[1, 1, 0], [-3, 1, 0], [1, -3, 0]]),
    "M": ([0.5, 0, 0], [[1, 1, 0], [-2, -1, 0]]),
}
_C6_DIRECT_ROTATION = np.array([[1, -1], [1, 0]], dtype=int)
_C2_STARS = {
    "GM": ([0, 0, 0], [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]),
    "X": ([0.5, 0, 0], [[1, 1, 0], [-2, 1, 0], [1, -1, 0], [-2, -1, 0]]),
    "Y": ([0, 0.5, 0], [[1, 1, 0], [-1, 1, 0], [1, -2, 0], [-1, -2, 0]]),
    "M": ([0.5, 0.5, 0], [[1, 1, 0], [-2, 1, 0], [1, -2, 0], [-2, -2, 0]]),
}


def _write_c6_fixture(root, *, broken_coefficients=False):
    root.mkdir(parents=True, exist_ok=True)
    direct = np.array([[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 8]])
    reciprocal = 2 * np.pi * np.linalg.inv(direct).T
    orbit_sites = []
    for x, y, z in ((0.173, 0.287, 0.119), (0.329, 0.071, 0.437)):
        point = np.array([x, y])
        for _ in range(6):
            orbit_sites.append(f"{point[0] % 1:.12f} {point[1] % 1:.12f} {z:.12f}")
            point = _C6_DIRECT_ROTATION @ point
    structure = root / "POSCAR"
    structure.write_text(
        "Generated P6 numerical fixture\n1.0\n1 0 0\n"
        "-0.5 0.8660254037844386 0\n0 0 8\nH He\n6 6\nDirect\n"
        + "\n".join(orbit_sites) + "\n", encoding="utf-8",
    )
    wavefunction = root / "wavefunctions.h5"
    with h5py.File(wavefunction, "w") as h5:
        metadata = h5.create_group("metadata")
        lattice = metadata.create_group("lattice")
        lattice["direct_cart"] = direct
        lattice["reciprocal_cart"] = reciprocal
        metadata["spinor"] = True
        metadata["source"] = "generated_c6_numerical_unit_fixture"
        metadata["wavecar_rtag"] = 45210
        metadata["vasp_band_index_base"] = 1
        kpoints = h5.create_group("kpoints")
        for index, (name, (frac, grid)) in enumerate(_C6_STARS.items()):
            kpoint = kpoints.create_group(str(index))
            kpoint["name"] = name
            kpoint["frac"] = frac
            kpoint["cart"] = np.asarray(frac) @ reciprocal
            kpoint["g_vectors_frac"] = np.asarray(grid, dtype=int)
            kpoint["g_vectors_cart"] = np.asarray(grid) @ reciprocal
            orbital = np.full(len(grid), 1 / np.sqrt(len(grid)))
            if broken_coefficients and name == "GM":
                orbital = np.zeros(len(grid))
                orbital[0] = 1
            coefficients = np.zeros((2, 2, len(grid)), dtype=complex)
            coefficients[0, 0] = orbital
            coefficients[1, 1] = orbital
            kpoint["coefficients"] = coefficients
            kpoint["energies_eV"] = np.zeros(2)
            kpoint["band_indices_vasp"] = [1, 2]
    config = {
        "input": {"wavefunction_h5": str(wavefunction)},
        "analysis": {"kpoints": list(_C6_STARS), "iband": [1, 2],
                     "reduced_ebr": {"enabled": False}},
        "monolayer_lattices": {"default": {"reciprocal_cart":
                                              (3 * reciprocal).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [
            {"name": "center", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "center", "centers": ["center"]}],
        "projection": {"use_2d_momentum_only": True,
                       "qcut_mode": "absolute", "qcut_Ainv": 20.0},
        "symmetry": {"operations": {"structure_file": str(structure)}},
        "output": {"directory": str(root / "out"), "profile": "debug",
                   "summary_stdout": False},
    }
    config_path = root / "analyze.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _run(root, *, broken_coefficients=False):
    outputs = analyze_hsp(_write_c6_fixture(
        root, broken_coefficients=broken_coefficients,
    ))
    reports = {name: json.loads(path.read_text(encoding="utf-8"))
               for name, path in outputs.items()
               if getattr(path, "suffix", None) == ".json"}
    return outputs, reports


def _write_c2_fixture(root, *, broken_coefficients=False, left_handed_plane=False):
    root.mkdir(parents=True, exist_ok=True)
    # Reverse both a and c: the full 3D cell remains right-handed, while
    # the (a,b) plane is left-handed relative to fixed Cartesian +z.
    direct = (np.diag([-1.0, 1.3, -8.0]) if left_handed_plane
              else np.diag([1.0, 1.3, 8.0]))
    reciprocal = 2 * np.pi * np.linalg.inv(direct).T
    sites = []
    for x, y, z in ((0.173, 0.287, 0.119), (0.329, 0.071, 0.437)):
        for a, b, c in ((x, y, z), (x, -y, -z),
                        (-x, y, -z), (-x, -y, z)):
            sites.append(f"{a % 1:.12f} {b % 1:.12f} {c % 1:.12f}")
    structure = root / "POSCAR"
    axes = "\n".join(" ".join(f"{value:.16g}" for value in axis)
                     for axis in direct)
    structure.write_text(
        "Generated P222 numerical fixture\n1.0\n" + axes + "\n"
        "H He\n4 4\nDirect\n" + "\n".join(sites) + "\n",
        encoding="utf-8",
    )
    wavefunction = root / "wavefunctions.h5"
    with h5py.File(wavefunction, "w") as h5:
        metadata = h5.create_group("metadata")
        lattice = metadata.create_group("lattice")
        lattice["direct_cart"] = direct
        lattice["reciprocal_cart"] = reciprocal
        metadata["spinor"] = True
        metadata["source"] = "generated_c2_numerical_unit_fixture"
        metadata["wavecar_rtag"] = 45210
        metadata["vasp_band_index_base"] = 1
        kpoints = h5.create_group("kpoints")
        for index, (name, (frac, grid)) in enumerate(_C2_STARS.items()):
            kpoint = kpoints.create_group(str(index))
            kpoint["name"] = name
            kpoint["frac"] = frac
            kpoint["cart"] = np.asarray(frac) @ reciprocal
            kpoint["g_vectors_frac"] = np.asarray(grid, dtype=int)
            kpoint["g_vectors_cart"] = np.asarray(grid) @ reciprocal
            orbital = np.full(len(grid), 1 / np.sqrt(len(grid)))
            if broken_coefficients and name == "GM":
                orbital = np.array([1.0, 0.0, 0.0, 0.0])
            coefficients = np.zeros((2, 2, len(grid)), dtype=complex)
            coefficients[0, 0] = orbital
            coefficients[1, 1] = orbital
            kpoint["coefficients"] = coefficients
            kpoint["energies_eV"] = np.zeros(2)
            kpoint["band_indices_vasp"] = [1, 2]
    config = {
        "input": {"wavefunction_h5": str(wavefunction)},
        "analysis": {"kpoints": list(_C2_STARS), "iband": [1, 2],
                     "reduced_ebr": {"enabled": False}},
        "monolayer_lattices": {"default": {"reciprocal_cart":
                                              (3 * reciprocal).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [
            {"name": "center", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "center", "centers": ["center"]}],
        "projection": {"use_2d_momentum_only": True,
                       "qcut_mode": "absolute", "qcut_Ainv": 20.0},
        "symmetry": {"operations": {"structure_file": str(structure)}},
        "output": {"directory": str(root / "out"), "profile": "debug",
                   "summary_stdout": False},
    }
    config_path = root / "analyze.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_generated_c6_spin_pair_reports_conditional_zero_residue(tmp_path):
    outputs, reports = _run(tmp_path)
    assert reports["symmetry_report_json"]["spacegroup_number"] == 168
    assert reports["symmetry_report_json"]["detected_operation_count"] == 6
    matches = reports["valley_irrep_matching_json"]["generic_matches_by_kpoint"]
    assert set(matches) == set(_C6_STARS)
    assert all(matches[k]["center"]["matching_status"] == "matched"
               for k in _C6_STARS), matches
    with h5py.File(outputs["diagnostics_h5"], "r") as h5:
        evidence = json.loads(h5["cprime/scoped_representation_evidence"][()])
    assert all(evidence[k]["center"]["status"] == "passed"
               for k in _C6_STARS), evidence
    report = reports["valley_summary_json"]["valley_chern_mod"]
    assert report["global_valley_subspace_status"] == "not_evaluated"
    assert [(row["valley"], row["modulus"], row["residue"], row["status"])
            for row in report["rows"]] == [("center", 6, 0, "conditional")]
    assert len(report["rows"][0]["eigenvalue_evidence"]) == 3
    assert "valley_reduced_ebr_mapping_json" not in outputs


def test_generated_c6_broken_star_blocks_residue(tmp_path):
    _, reports = _run(tmp_path, broken_coefficients=True)
    closure = reports["target_subspace_closure_json"]["by_kpoint"]["GM"]
    assert any(row["classification"] == "target_subspace_not_closed"
               for row in closure)
    row = reports["valley_summary_json"]["valley_chern_mod"]["rows"][0]
    assert row["valley"] == "center"
    assert row["modulus"] == 6
    assert row["status"] == "blocked"
    assert row["residue"] is None
    assert reports["valley_irrep_matching_json"]["generic_matches_by_kpoint"]["GM"]["center"]["matching_status"] == "blocked"


def test_generated_c2_spin_pair_reports_conditional_zero_residue(tmp_path):
    outputs = analyze_hsp(_write_c2_fixture(tmp_path))
    summary = json.loads(outputs["valley_summary_json"].read_text())
    symmetry = json.loads(outputs["symmetry_report_json"].read_text())
    matches = json.loads(outputs["valley_irrep_matching_json"].read_text())[
        "generic_matches_by_kpoint"
    ]
    assert symmetry["spacegroup_number"] == 16
    assert symmetry["detected_operation_count"] == 4
    assert set(matches) == set(_C2_STARS)
    assert all(matches[k]["center"]["matching_status"] == "matched"
               for k in _C2_STARS), matches
    with h5py.File(outputs["diagnostics_h5"], "r") as h5:
        evidence = json.loads(h5["cprime/scoped_representation_evidence"][()])
    assert all(evidence[k]["center"]["status"] == "passed"
               for k in _C2_STARS), evidence
    report = summary["valley_chern_mod"]
    assert report["global_valley_subspace_status"] == "not_evaluated"
    assert [(row["valley"], row["modulus"], row["residue"], row["status"])
            for row in report["rows"]] == [("center", 2, 0, "conditional")]
    assert len(report["rows"][0]["eigenvalue_evidence"]) == 4
    assert "valley_reduced_ebr_mapping_json" not in outputs


def test_generated_c2_broken_star_blocks_residue(tmp_path):
    outputs = analyze_hsp(_write_c2_fixture(tmp_path, broken_coefficients=True))
    summary = json.loads(outputs["valley_summary_json"].read_text())
    matches = json.loads(outputs["valley_irrep_matching_json"].read_text())[
        "generic_matches_by_kpoint"
    ]
    closure = json.loads(outputs["target_subspace_closure_json"].read_text())[
        "by_kpoint"
    ]["GM"]
    assert any(row["classification"] == "target_subspace_not_closed"
               for row in closure)
    assert matches["GM"]["center"]["matching_status"] == "blocked"
    row = summary["valley_chern_mod"]["rows"][0]
    assert row["valley"] == "center"
    assert row["modulus"] == 2
    assert row["status"] == "blocked"
    assert row["residue"] is None


def test_left_handed_in_plane_basis_keeps_c2_residue(tmp_path):
    outputs = analyze_hsp(_write_c2_fixture(tmp_path, left_handed_plane=True))
    summary = json.loads(outputs["valley_summary_json"].read_text())
    row = summary["valley_chern_mod"]["rows"][0]
    assert (row["valley"], row["modulus"], row["residue"], row["status"]) == (
        "center", 2, 0, "conditional",
    )
