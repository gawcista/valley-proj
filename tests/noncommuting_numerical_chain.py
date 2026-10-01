"""Generated P4mm spinful acceptance; no injected representation/trust data.

The orbital is an equal-amplitude reciprocal star times a spin pair. Its
spin-half C4v representation is irreducible: C4 has distinct eigenvalues,
while a vertical mirror exchanges the eigenvectors. This is test data, not
a DFT/material model. Expected characters and labels are independently
checked against the reviewed SG99 source, never inferred from test output.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml

from valleyscope.analysis.database_ingestion_record import (
    load_database_ingestion_record_from_directory,
)
from valleyscope.workflows.analyze_hsp import analyze_hsp
from tests.rotation_chern_acceptance import assert_rotation_chern_summary


# Hand-specified stars, closed under the local little group and time reversal.
# At X/M the momenta are k+G, not G alone. No production symmetry helper is used.
STARS = {
    "GM": ([0, 0, 0], [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]),
    "X": ([0, 0.5, 0], [[1, 0, 0], [-1, -1, 0], [1, -1, 0], [-1, 0, 0]]),
    "M": ([0.5, 0.5, 0], [[0, 0, 0], [-1, -1, 0], [0, -1, 0], [-1, 0, 0]]),
}
EXPECTED_IRREPS = {"GM": {"-GM7": 1}, "X": {"-X5": 1}, "M": {"-M7": 1}}
SCOPE_SIZES = {"GM": 8, "X": 4, "M": 8}


def write_noncommuting_fixture(
    root: Path, *, broken_coefficients: bool = False, profile: str = "debug",
) -> Path:
    """Generate physical HDF5/POSCAR inputs and ordinary public configuration."""
    root.mkdir(parents=True, exist_ok=True)
    direct = np.diag([1.0, 1.0, 8.0])
    reciprocal = 2 * np.pi * np.linalg.inv(direct).T
    # Two chemically distinct generic C4v orbits at different z remove
    # accidental inversion/horizontal mirrors and fractional translations.
    sites = []
    for x, y, z in ((0.173, 0.287, 0.119), (0.329, 0.071, 0.437)):
        for a, b in ((x, y), (y, x)):
            for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                sites.append(f"{(sx * a) % 1:.12f} {(sy * b) % 1:.12f} {z:.12f}")
    structure = root / "POSCAR"
    structure.write_text(
        "Generated P4mm symmetry fixture\n1.0\n1 0 0\n0 1 0\n0 0 8\n"
        "H He\n8 8\nDirect\n" + "\n".join(sites) + "\n", encoding="utf-8",
    )
    payload = root / "wavefunctions.h5"
    with h5py.File(payload, "w") as h5:
        metadata = h5.create_group("metadata")
        lattice = metadata.create_group("lattice")
        lattice["direct_cart"] = direct
        lattice["reciprocal_cart"] = reciprocal
        metadata["spinor"] = True
        metadata["source"] = "generated_noncommuting_numerical_acceptance"
        metadata["wavecar_rtag"] = 45210
        metadata["vasp_band_index_base"] = 1
        kpoints = h5.create_group("kpoints")
        for index, (label, (frac, grid)) in enumerate(STARS.items()):
            kpoint = kpoints.create_group(str(index))
            kpoint["name"] = label
            kpoint["frac"] = frac
            kpoint["cart"] = np.array(frac) @ reciprocal
            kpoint["g_vectors_frac"] = np.array(grid, dtype=int)
            kpoint["g_vectors_cart"] = np.array(grid) @ reciprocal
            # Each adjacent pair consists of opposite momenta. Unequal pair
            # weights preserve TR and normalization but break spatial closure.
            orbital = np.array([2, 2, 1, 1]) / np.sqrt(10) if broken_coefficients else np.full(4, 0.5)
            coefficients = np.zeros((2, 2, 4), dtype=complex)
            coefficients[0, 0] = orbital
            coefficients[1, 1] = orbital
            kpoint["coefficients"] = coefficients
            kpoint["energies_eV"] = np.zeros(2)
            kpoint["band_indices_vasp"] = [1, 2]
    config = {
        "input": {"wavefunction_h5": str(payload)},
        "analysis": {"kpoints": list(STARS), "iband": [1, 2], "reduced_ebr": {"enabled": True}},
        "monolayer_lattices": {"default": {"reciprocal_cart": (4 * reciprocal).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "center", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "center", "centers": ["center"]}],
        "projection": {"use_2d_momentum_only": True, "qcut_mode": "absolute", "qcut_Ainv": 8.0},
        "symmetry": {"operations": {"structure_file": str(structure)}},
        "output": {"directory": str(root / "out"), "profile": profile, "summary_stdout": False},
    }
    path = root / "analyze.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def run_noncommuting_workflow(root: Path, **kwargs) -> dict[str, object]:
    """Run the unpatched public workflow and read its serialized output."""
    outputs = analyze_hsp(write_noncommuting_fixture(root, **kwargs))
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in outputs.items()
        if isinstance(path, Path) and path.suffix == ".json"
    }
    assert reports["valley_summary_json"]["output_profile"] == kwargs.get("profile", "debug")
    return {
        "outputs": outputs,
        "reports": reports,
        "ingestion": load_database_ingestion_record_from_directory(
            root / "out", output_profile=kwargs.get("profile", "debug"),
        ),
    }


def _assert_numerical_matrices(result: dict[str, object]) -> None:
    """Use actual pre-promotion matrices, not characters copied from a table."""
    with h5py.File(result["outputs"]["diagnostics_h5"], "r") as h5:
        source = json.loads(h5["cprime/spinor_source_basis_certificate"][()])
        assert source["status"] == "passed"
        assert source["extractor_provenance"] is None
        lifts = json.loads(h5["cprime/double_space_group_lift_certificates"][()])
        evidence = json.loads(h5["cprime/scoped_representation_evidence"][()])
        assert set(evidence) == set(STARS)
        for hsp, size in SCOPE_SIZES.items():
            assert set(evidence[hsp]) == {"center"}
            row = evidence[hsp]["center"]
            assert row["status"] == "passed", row
            required = row["scope"]["required_operation_ids"]
            assert len(required) == size
            lift = lifts[hsp]["center"]
            assert lift["status"] == "passed", lift
            # The double lift covers the full subspace group, even at X,
            # whose local C-prime scope contains only four operations.
            assert len(lift["operation_ids"]) == 8
            assert set(required) <= set(lift["operation_ids"])
            assert len(lift["pairwise_products"]) == 64
            mapping = row["plane_wave_mapping"]
            assert len(mapping["operation_rows"]) == size
            for operation in mapping["operation_rows"]:
                assert operation["reciprocal_grid_permutation_passed"]
                assert sorted(operation["source_to_target_map"]) == list(range(4))
            assert len(mapping["composition_rows"]) == size * size
            assert all(op["passed"] for op in mapping["composition_rows"])
            maps = {op["operation_id"]: op["source_to_target_map"] for op in mapping["operation_rows"]}
            groups = h5[f"symmetry_representations/{hsp}"]
            assert len(groups) == size
            matrices, rotations, operations = [], [], set()
            for group in groups.values():
                assert group.attrs["target_valley"] == "center"
                operations.add(group.attrs["source_operation_key"])
                matrix, rotation = group["D_valley"][()], group["rotation_frac"][()]
                frac, grid = STARS[hsp]
                transformed_grid = (np.array(grid) + frac) @ rotation.T - frac
                np.testing.assert_allclose(transformed_grid, np.rint(transformed_grid), atol=1e-12)
                lookup = {tuple(g): index for index, g in enumerate(grid)}
                expected_map = [lookup[tuple(g)] for g in np.rint(transformed_grid).astype(int)]
                operation_id = int(group.attrs["source_operation_key"].removeprefix("operation_"))
                assert maps[operation_id] == expected_map
                assert matrix.shape == (2, 2)
                np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2), atol=1e-12)
                # Trivial orbital times canonical spin half: trace 2 at E,
                # sqrt(2) at either C4, zero at C2 and every vertical mirror.
                character = 2 if np.array_equal(rotation, np.eye(3)) else (
                    np.sqrt(2) if np.linalg.det(rotation) > 0 and np.trace(rotation) == 1 else 0
                )
                np.testing.assert_allclose(np.trace(matrix), character, atol=1e-12)
                matrices.append(matrix)
                rotations.append(rotation)
            assert operations == {f"operation_{op}" for op in required}
            c4_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            mirror_rotation = np.diag([1, -1, 1])
            powers = (0, 1, 2, 3) if hsp != "X" else (0, 2)
            expected_rotations = {
                tuple((np.linalg.matrix_power(c4_rotation, power) @ mirror).ravel())
                for power in powers for mirror in (np.eye(3, dtype=int), mirror_rotation)
            }
            assert {tuple(rotation.ravel()) for rotation in rotations} == expected_rotations
            # Schur's criterion on the *emitted* matrices: commutant dimension
            # one proves irreducibility; shape (2,2) alone does not.
            commutant = np.vstack([np.kron(np.eye(2), d) - np.kron(d.T, np.eye(2)) for d in matrices])
            singular_values = np.linalg.svd(commutant, compute_uv=False)
            assert np.count_nonzero(singular_values > 1e-10) == 3
            if hsp in ("GM", "M"):
                c4 = next(d for r, d in zip(rotations, matrices) if np.array_equal(r, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
                mirror = next(d for r, d in zip(rotations, matrices) if np.array_equal(r, np.diag([1, -1, 1])))
                np.testing.assert_allclose(c4 @ c4 @ c4 @ c4, -np.eye(2), atol=1e-12)
                np.testing.assert_allclose(mirror @ mirror, -np.eye(2), atol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(c4 @ mirror - mirror @ c4), 2, atol=1e-12)


def assert_noncommuting_positive(result: dict[str, object]) -> dict[str, object]:
    """Require reviewed 2D irreps and exact downstream EBR/ingestion."""
    reports = result["reports"]
    chern = assert_rotation_chern_summary(
        reports["valley_summary_json"], valleys={"center"}, modulus=4,
        expected_hsp_powers={"GM": 1, "M": 1, "X": 2},
    )
    rows = reports["valley_summary_json"]["valley_resolved_irreps"]["rows"]
    assert len(rows) == 3
    assert {row["kpoint"] for row in rows} == set(STARS)
    for row in rows:
        hsp = row["kpoint"]
        assert row["valley"] == "center"
        assert row["subspace_space_group_number"] == 99
        assert row["matching_status"] == "matched", row
        assert row["irrep_multiplicities"] == EXPECTED_IRREPS[hsp], row
        assert row["readiness_level"] == "trusted"
        assert row["workflow_path"] == "direct_qcut"
        assert row["diagnostic_only"] is False
        assert len(row["hsp_little_group_operation_ids"]) == SCOPE_SIZES[hsp]
        assert row["source_hsp_label"] == hsp
    export = reports["valley_ebr_export_bundle_json"]
    assert export["bundle_count"] == 1, export
    bundle = export["bundles"][0]
    assert set(bundle["expected_hsps"]) == set(STARS)
    assert bundle["unitary_vector_construction"]["kind"] == "direct_observed_unitary_rows"
    solutions = reports["valley_reduced_ebr_mapping_json"]["solutions"]
    assert len(solutions) == 1, solutions
    solution = solutions[0]
    assert solution["status"] == "solved_exact", solution
    assert solution["classification"] == "atomic-compatible-candidate"
    assert solution["ebr_decomposition"] == [{"label": "-E1↑G(2) @ 1a(4mm,4mm)", "coefficient": 1}]
    ingestion = result["ingestion"]
    assert ingestion["validation_errors"] == [], ingestion
    assert ingestion["final_reduced_ebr_result_count"] == 1, ingestion
    if reports["valley_summary_json"]["output_profile"] == "debug":
        assert "diagnostics_h5" in result["outputs"]
        assert "symmetry_report_json" in reports
        symmetry = reports["symmetry_report_json"]
        assert symmetry["spacegroup_number"] == 99
        assert symmetry["detected_operation_count"] == 8
        _assert_numerical_matrices(result)
    return {
        "source_space_group": 99, "spinor": True,
        "observed_hsps": ["GM", "X", "M"],
        "required_operations_by_hsp": SCOPE_SIZES,
        "plane_waves_per_kpoint": 4, "bands": 2,
        "final_reduced_ebr_result_count": 1, "validation_errors": [],
        "rotation_chern_positive": chern,
    }


def assert_noncommuting_negative(result: dict[str, object]) -> dict[str, object]:
    """Good purity, normalization and TR cannot replace spatial closure."""
    reports = result["reports"]
    chern = assert_rotation_chern_summary(
        reports["valley_summary_json"], valleys={"center"}, modulus=4,
        expected_hsp_powers={"GM": 1, "M": 1, "X": 2}, blocked=True,
    )
    with result["outputs"]["valley_weights_csv"].open(encoding="utf-8") as handle:
        weights = list(csv.DictReader(handle))
    assert len(weights) == 6
    for field in ("W_val", "P_v"):
        np.testing.assert_allclose([float(row[field]) for row in weights], 1, atol=1e-12)
    rows = reports["valley_summary_json"]["valley_resolved_irreps"]["rows"]
    assert len(rows) == 3
    assert {row["kpoint"] for row in rows} == set(STARS)
    assert all(row["matching_status"] == "blocked" for row in rows), rows
    assert all(row["diagnostic_only"] for row in rows), rows
    assert all(row["irrep_multiplicities"] == {} for row in rows), rows
    if reports["valley_summary_json"]["output_profile"] == "debug":
        assert "target_subspace_closure_json" in reports
        assert "irrep_workflow_decisions_json" in reports
        closure = reports["target_subspace_closure_json"]["by_kpoint"]
        for hsp in STARS:
            local = [row for row in closure[hsp] if row["little_group_passed"]]
            assert len(local) == SCOPE_SIZES[hsp]
            assert all(row["target_frame_status"] == "passed" for row in local)
            failed = [row for row in local if row["classification"] == "target_subspace_not_closed"]
            assert failed, closure[hsp]
            assert all(row["closure_quality"] == "blocked" for row in failed)
        decisions = reports["irrep_workflow_decisions_json"]["by_kpoint"]
        assert all(decisions[hsp]["center"]["readiness_level"] == "blocked" for hsp in STARS)
    assert result["ingestion"]["validation_errors"] == []
    assert result["ingestion"]["final_reduced_ebr_result_count"] == 0
    # No ready bundle means these optional public files must not be emitted.
    assert "valley_ebr_export_bundle_json" not in reports
    assert "valley_reduced_ebr_mapping_json" not in reports
    summary = reports["valley_summary_json"]
    if summary["output_profile"] == "standard":
        reduced = summary["reduced_ebr_summary"]
        assert reduced["trusted_bundle_count"] == 0
        assert reduced["result_count"] == 0
    else:
        assert summary["valley_ebr_export_bundle"]["bundles"] == []
        assert summary["valley_reduced_ebr_mapping"]["solutions"] == []
    return chern


def run_installed_noncommuting_acceptance(workdir: Path | None = None) -> dict[str, object]:
    """Run both real numerical cases against the installed artifact."""
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="valleyscope_noncommuting_") as tmp:
            return run_installed_noncommuting_acceptance(Path(tmp))
    summary = assert_noncommuting_positive(run_noncommuting_workflow(workdir / "positive"))
    summary["rotation_chern_broken_coefficients"] = assert_noncommuting_negative(
        run_noncommuting_workflow(workdir / "negative", broken_coefficients=True)
    )
    summary["broken_coefficients_final_result_count"] = 0
    return summary
