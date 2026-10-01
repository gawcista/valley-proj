"""Rotation residues from genuine generated wavefunction-to-irrep chains."""

import json
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import yaml


@pytest.mark.parametrize("profile", ["standard", "debug"])
def test_p3_numerical_chain_reports_conditional_rank_two_residues(tmp_path, profile):
    from tests.portable_numerical_chain import run_numerical_workflow

    result = run_numerical_workflow(tmp_path, profile=profile)
    report = result["reports"]["valley_summary_json"]["valley_chern_mod"]
    assert report["global_valley_subspace_status"] == "not_evaluated"
    assert {(r["valley"], r["modulus"], r["residue"], r["status"])
            for r in report["rows"]} == {
        ("plus", 3, 0, "conditional"), ("minus", 3, 0, "conditional"),
    }
    for row in report["rows"]:
        assert row["subspace_rank"] == 2
        assert len(row["eigenvalue_evidence"]) == 3
        assert {e["completion_kind"] for e in row["eigenvalue_evidence"]} == {
            "observed_at_sampled_kpoint",
        }


def test_noncommuting_numerical_chain_reports_strongest_rotation_residue(tmp_path):
    from tests.noncommuting_numerical_chain import run_noncommuting_workflow

    result = run_noncommuting_workflow(tmp_path, profile="standard")
    report = result["reports"]["valley_summary_json"]["valley_chern_mod"]
    assert {(r["modulus"], r["residue"], r["status"])
            for r in report["rows"]} == {
        (4, 0, "conditional"),
    }


def test_missing_rotation_hsp_blocks_without_requiring_ebr_mapping(tmp_path):
    from tests.portable_numerical_chain import write_numerical_fixture
    from valleyscope.workflows.analyze_hsp import analyze_hsp

    path = write_numerical_fixture(tmp_path, profile="standard")
    config = yaml.safe_load(path.read_text())
    config["analysis"]["kpoints"] = ["GM", "K"]
    config["analysis"]["reduced_ebr"]["enabled"] = False
    path.write_text(yaml.safe_dump(config))
    outputs = analyze_hsp(path)
    report = json.loads(outputs["valley_summary_json"].read_text())["valley_chern_mod"]
    assert report["rows"]
    for row in report["rows"]:
        assert row["status"] == "blocked"
        assert row["residue"] is None
        assert any("missing_rotation_hsp" in x for x in row["blocking_reasons"])


def test_high_purity_closure_failure_cannot_produce_rotation_residue(tmp_path):
    from tests.portable_numerical_chain import run_numerical_workflow

    result = run_numerical_workflow(tmp_path, broken_coefficients=True)
    report = result["reports"]["valley_summary_json"]["valley_chern_mod"]
    failed = next(r for r in report["rows"] if r["valley"] == "plus")
    assert failed["status"] == "blocked"
    assert failed["residue"] is None


def test_valley_changing_c3_and_inplane_c2_are_not_cnz():
    from valleyscope.analysis.valley_chern import find_valley_rotations

    angle = 2 * np.pi / 3
    c3 = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    operations = [
        {"operation_id": 4, "rotation_cart": c3,
         "rotation_frac": [[0, -1, 0], [1, -1, 0], [0, 0, 1]],
         "translation_frac": [0, 0, 0], "sector_mapping": {"a": "b"}},
        {"operation_id": 8, "rotation_cart": np.diag([1, -1, -1]),
         "rotation_frac": np.diag([1, -1, -1]),
         "translation_frac": [0, 0, 0], "sector_mapping": {"a": "a"}},
    ]
    assert find_valley_rotations(operations, "a") == []


def test_positive_rotation_direction_is_selected_without_using_operation_ids():
    from valleyscope.analysis.valley_chern import find_valley_rotations

    operations = []
    for op_id, sign in ((29, -1), (71, 1)):
        angle = sign * np.pi / 2
        matrix = np.array([[round(np.cos(angle)), -round(np.sin(angle)), 0],
                           [round(np.sin(angle)), round(np.cos(angle)), 0], [0, 0, 1]])
        operations.append({"operation_id": op_id, "rotation_cart": matrix,
                           "rotation_frac": matrix, "translation_frac": [0, 0, 0],
                           "sector_mapping": {"a": "a"}})
    rotations = find_valley_rotations(operations, "a")
    assert [(n, op["operation_id"]) for n, op in rotations] == [(4, 71)]


def _capture_runtime_inputs(path):
    from valleyscope.analysis.valley_chern import build_valley_chern_report
    from valleyscope.workflows.analyze_hsp import analyze_hsp

    captured = {}

    def observe(**kwargs):
        captured.update(deepcopy(kwargs))
        return build_valley_chern_report(**kwargs)

    # Observe genuine producer outputs, without replacing physical evidence.
    with patch("valleyscope.analysis.valley_chern.build_valley_chern_report", side_effect=observe):
        analyze_hsp(path)
    return captured


@pytest.fixture(scope="module")
def runtime_inputs(tmp_path_factory):
    from tests.portable_numerical_chain import write_numerical_fixture

    path = write_numerical_fixture(tmp_path_factory.mktemp("chern_runtime"), profile="standard")
    config = yaml.safe_load(path.read_text())
    config["analysis"]["reduced_ebr"]["enabled"] = False
    path.write_text(yaml.safe_dump(config))
    return _capture_runtime_inputs(path)


@pytest.fixture(scope="module")
def tr_runtime_inputs(tmp_path_factory):
    from tests.portable_numerical_chain import write_numerical_fixture

    path = write_numerical_fixture(tmp_path_factory.mktemp("chern_tr_runtime"), profile="standard")
    config = yaml.safe_load(path.read_text())
    config["analysis"]["kpoints"] = ["GM", "K", "M"]
    config["analysis"]["time_reversal"] = {"enabled": True}
    config["analysis"]["reduced_ebr"]["enabled"] = False
    path.write_text(yaml.safe_dump(config))
    return _capture_runtime_inputs(path)


def test_validated_tr_completion_supplies_unobserved_corner(tr_runtime_inputs):
    from valleyscope.analysis.valley_chern import build_valley_chern_report

    report = build_valley_chern_report(**deepcopy(tr_runtime_inputs))
    assert report["status"] == "conditional", report
    for row in report["rows"]:
        assert row["residue"] == 0
        inferred = [e for e in row["eigenvalue_evidence"]
                    if e["completion_kind"] == "inferred_by_time_reversal"]
        assert len(inferred) == 1
        assert "sampled_kpoint" not in inferred[0]
        assert inferred[0]["evidence_sampled_kpoint"] == "K"


def test_rotation_hsp_coordinates_must_bind_to_actual_wavefunctions(runtime_inputs):
    from valleyscope.analysis.valley_chern import build_valley_chern_report

    inputs = deepcopy(runtime_inputs)
    coordinates = inputs["kpoint_frac_by_name"]
    coordinates["GM"], coordinates["K"] = coordinates["K"], coordinates["GM"]
    report = build_valley_chern_report(**inputs)
    assert all(row["status"] == "blocked" and row["residue"] is None
               for row in report["rows"]), report


def test_missing_raw_context_does_not_discard_other_valley(runtime_inputs):
    from valleyscope.analysis.valley_chern import build_valley_chern_report

    inputs = deepcopy(runtime_inputs)
    contexts = inputs["cprime_validation_context"]
    for key, context in list(contexts.items()):
        if context["record"]["scope"]["source_valleys"] == ["plus"]:
            del contexts[key]
    report = build_valley_chern_report(**inputs)
    states = {row["valley"]: row["status"] for row in report["rows"]}
    assert states == {"plus": "blocked", "minus": "conditional"}


def test_tampered_raw_rotation_matrix_cannot_use_serialized_passed_status(runtime_inputs):
    from valleyscope.analysis.valley_chern import build_valley_chern_report

    inputs = deepcopy(runtime_inputs)
    context = next(c for c in inputs["cprime_validation_context"].values()
                   if c["record"]["scope"]["kpoint_label"] == "GM"
                   and c["record"]["scope"]["source_valleys"] == ["plus"])
    raw = context["raw_inputs"]
    op_id = next(i for i in raw["representations"] if i != 0)
    raw["representations"][op_id] *= -1
    report = build_valley_chern_report(**inputs)
    row = next(r for r in report["rows"] if r["valley"] == "plus")
    assert row["status"] == "blocked"
    assert row["residue"] is None


def test_tr_completion_certificate_is_revalidated(tr_runtime_inputs):
    from valleyscope.analysis.valley_chern import build_valley_chern_report

    inputs = deepcopy(tr_runtime_inputs)
    for orbit in inputs["time_reversal_orbit_report"]["valley_orbits"]:
        for by_hsp in orbit.get("unitary_valley_irrep_completion_records", {}).values():
            for records in by_hsp.values():
                for record in records:
                    if record["completion_kind"] == "inferred_by_time_reversal":
                        record["tr_irrep_completion_certificate"]["certificate_identity"] = "sha256:forged"
    report = build_valley_chern_report(**inputs)
    assert all(row["status"] == "blocked" and row["residue"] is None
               for row in report["rows"]), report


def test_tr_completion_cannot_be_relocated_to_other_target_valley(tr_runtime_inputs):
    from valleyscope.analysis.valley_chern import build_valley_chern_report

    inputs = deepcopy(tr_runtime_inputs)
    orbit = inputs["time_reversal_orbit_report"]["valley_orbits"][0]
    records = orbit["unitary_valley_irrep_completion_records"]
    records["plus"]["KA"] = deepcopy(records["minus"]["KA"])
    report = build_valley_chern_report(**inputs)
    row = next(r for r in report["rows"] if r["valley"] == "plus")
    assert row["status"] == "blocked", row
    assert row["residue"] is None


def test_screw_axis_does_not_hide_available_finite_order_rotation():
    from valleyscope.analysis.valley_chern import find_valley_rotations

    rotation = np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]])
    operations = []
    for power in range(6):
        angle = power * np.pi / 3
        cart = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        operations.append({"operation_id": power, "rotation_cart": cart,
                           "rotation_frac": np.linalg.matrix_power(rotation, power),
                           "translation_frac": [0, 0, (power * 0.5) % 1],
                           "sector_mapping": {"a": "a"}})
    assert [(n, op["operation_id"]) for n, op in find_valley_rotations(operations, "a")] == [(3, 2)]


@pytest.mark.parametrize("malformation", ["operation", "mapping", "match", "context", "symmetry"])
def test_malformed_evidence_blocks_optional_residue_without_crashing(runtime_inputs, malformation):
    from valleyscope.analysis.valley_chern import build_valley_chern_report

    inputs = deepcopy(runtime_inputs)
    if malformation == "operation":
        inputs["symmetry_payload"]["detected_operations"] = [None]
    elif malformation == "mapping":
        inputs["symmetry_payload"]["detected_operations"][0]["sector_mapping"] = None
    elif malformation == "match":
        inputs["valley_irrep_matching"]["generic_matches_by_kpoint"]["GM"] = None
    elif malformation == "context":
        for key in inputs["cprime_validation_context"]:
            inputs["cprime_validation_context"][key] = None
    else:
        inputs["symmetry_payload"] = None
    report = build_valley_chern_report(**inputs)
    assert all(row["status"] == "blocked" and row["residue"] is None
               for row in report["rows"]), report


def test_spinful_formula_must_bind_to_validated_source_layout(runtime_inputs):
    from valleyscope.analysis.valley_chern import build_valley_chern_report

    inputs = deepcopy(runtime_inputs)
    inputs["spinful"] = False
    report = build_valley_chern_report(**inputs)
    assert all(row["status"] == "blocked" and row["residue"] is None
               for row in report["rows"]), report


def test_seitz_power_uses_translation_phase_for_shifted_rotation_center():
    """An isolated matrix unit test, not a substitute for C-prime validation."""
    from valleyscope.analysis.valley_chern import _physical_determinant

    generator = {"rotation_frac": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                 "rotation_cart": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                 "translation_frac": [0.1, 0.7, 0]}
    half_turn = {"operation_id": 41, "rotation_frac": np.diag([-1, -1, 1]),
                 "rotation_cart": np.diag([-1, -1, 1]), "translation_frac": [0.4, 0.8, 0]}
    raw = {"lift_validation_inputs": {"expected_operations": [half_turn]},
           "required_operation_ids": [41], "valley_mappings": {41: {"a": "a"}},
           "valley_bases": {"a": np.eye(1)}, "representations": {41: np.array([[1j]])},
           "kpoint_frac": np.array([0.5, 0.5, 0])}
    det, rank, _ = _physical_determinant(raw, "a", generator, 2, spinful=True)
    assert rank == 1
    assert det == pytest.approx(-1j)


def test_seitz_power_uses_central_sign_for_odd_rank():
    from valleyscope.analysis.valley_chern import _physical_determinant

    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    generator = {"rotation_frac": rotation, "rotation_cart": rotation, "translation_frac": [0, 0, 0]}
    inverse = {"operation_id": 53, "rotation_frac": rotation.T,
               "rotation_cart": rotation.T, "translation_frac": [0, 0, 0]}
    raw = {"lift_validation_inputs": {"expected_operations": [inverse]},
           "required_operation_ids": [53], "valley_mappings": {53: {"a": "a"}},
           "valley_bases": {"a": np.eye(1)},
           "representations": {53: np.array([[np.exp(1j * np.pi / 4)]])},
           "kpoint_frac": np.zeros(3)}
    det, rank, _ = _physical_determinant(raw, "a", generator, 3, spinful=True)
    assert rank == 1
    assert det == pytest.approx(np.exp(-3j * np.pi / 4))
