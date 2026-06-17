import numpy as np
import pytest

from valleyscope.irreps.tables import load_standard_irrep_table, match_table_operations
from valleyscope.irreps.matching import (
    decompose_characters_into_irreps,
    match_single_state_irrep,
)


def test_load_sg143_spinor_table_exposes_double_valued_k_irreps():
    table = load_standard_irrep_table(143, spinor=True)

    assert table.number == 143
    assert table.name == "P3"
    assert table.spinor is True
    assert [operation.table_index for operation in table.operations] == [1, 2, 3]

    c3 = table.operation_by_index(2)
    np.testing.assert_array_equal(
        c3.rotation_frac,
        np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int),
    )
    np.testing.assert_allclose(c3.translation_frac, np.zeros(3), atol=1e-12)

    k_irreps = {irrep.label: irrep for irrep in table.irreps_by_kpoint("K")}
    assert {"-K4", "-K5", "-K6"} <= set(k_irreps)
    assert k_irreps["-K5"].dimension == 1
    assert k_irreps["-K6"].dimension == 1
    assert k_irreps["-K5"].characters[2] == pytest.approx(
        np.exp(-1j * np.pi / 3),
        abs=1e-4,
    )
    assert k_irreps["-K6"].characters[2] == pytest.approx(
        np.exp(+1j * np.pi / 3),
        abs=1e-4,
    )
    assert table.operation_indices_for_kpoint("K") == [1, 2, 3]
    assert table.operation_indices_for_kpoint("M") == [1]


def test_match_table_operations_maps_detected_p3_operations_to_irreptables_indices():
    table = load_standard_irrep_table(143, spinor=True)
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    detected_operations = [
        {"operation_id": 10, "rotation_frac": np.eye(3, dtype=int), "translation_frac": np.zeros(3)},
        {"operation_id": 11, "rotation_frac": c3, "translation_frac": np.zeros(3)},
        {"operation_id": 12, "rotation_frac": c3 @ c3, "translation_frac": np.zeros(3)},
    ]

    report = match_table_operations(detected_operations, table)

    assert report.status == "complete"
    assert report.mapping_by_operation_id == {10: 1, 11: 2, 12: 3}
    assert report.unmatched_operation_ids == []
    assert report.unused_table_operation_indices == []


def test_match_table_operations_reports_unmatched_extra_operation():
    table = load_standard_irrep_table(143, spinor=True)
    detected_operations = [
        {"operation_id": 0, "rotation_frac": np.eye(3, dtype=int), "translation_frac": np.zeros(3)},
        {"operation_id": 9, "rotation_frac": np.diag([-1, -1, 1]), "translation_frac": np.zeros(3)},
    ]

    report = match_table_operations(detected_operations, table)

    assert report.status == "incomplete"
    assert report.mapping_by_operation_id == {0: 1}
    assert report.unmatched_operation_ids == [9]
    assert report.unused_table_operation_indices == [2, 3]


def test_match_table_operations_maps_conjugate_twofold_subgroup_to_c2_table():
    table = load_standard_irrep_table(5, spinor=True)
    detected_operations = [
        {"operation_id": 0, "rotation_frac": np.eye(3, dtype=int), "translation_frac": np.zeros(3)},
        {
            "operation_id": 4,
            "rotation_frac": np.array([[-1, 1, 0], [0, 1, 0], [0, 0, -1]], dtype=int),
            "translation_frac": np.zeros(3),
        },
    ]

    report = match_table_operations(detected_operations, table)

    assert report.status == "complete"
    assert report.mapping_by_operation_id == {0: 1, 4: 2}
    assert report.unmatched_operation_ids == []
    assert report.unused_table_operation_indices == []


def test_decompose_characters_into_sg143_spinor_k_irrep_multiplicities():
    table = load_standard_irrep_table(143, spinor=True)

    result = decompose_characters_into_irreps(
        table=table,
        table_kpoint_label="K",
        computed_characters={1: 2.0 + 0.0j, 2: 1.0 + 0.0j, 3: 1.0 + 0.0j},
    )

    assert result.status == "matched"
    assert result.irrep_multiplicities == {"-K5": 1, "-K6": 1}
    assert result.irrep_weights["-K4"] == pytest.approx(0.0, abs=1e-8)
    assert result.irrep_weights["-K5"] == pytest.approx(1.0, abs=1e-5)
    assert result.irrep_weights["-K6"] == pytest.approx(1.0, abs=1e-5)
    assert result.missing_table_operation_indices == []
    assert result.failure_reasons == []


def test_decompose_characters_reports_missing_table_operations():
    table = load_standard_irrep_table(143, spinor=True)

    result = decompose_characters_into_irreps(
        table=table,
        table_kpoint_label="K",
        computed_characters={1: 2.0 + 0.0j, 2: 1.0 + 0.0j},
    )

    assert result.status == "missing_characters"
    assert result.irrep_multiplicities == {}
    assert result.missing_table_operation_indices == [3]
    assert result.failure_reasons == ["Missing computed characters for table operations: [3]"]


# -------------------------------------------------------------------------
# Single-state irrep matching tests
# -------------------------------------------------------------------------

def test_match_single_state_irrep_identifies_clean_one_dimensional_result():
    table = load_standard_irrep_table(143, spinor=True)

    result = match_single_state_irrep(
        table=table,
        table_kpoint_label="K",
        state_index=0,
        computed_characters={
            1: 1.0 + 0.0j,
            2: np.exp(-1j * np.pi / 3.0),
            3: np.exp(+1j * np.pi / 3.0),
        },
    )

    assert result.status == "matched"
    assert result.irrep_label == "-K5"
    assert result.state_index == 0
    assert result.irrep_multiplicities == {"-K5": 1}
    assert result.failure_reasons == []


def test_match_single_state_irrep_returns_missing_characters_when_operation_missing():
    table = load_standard_irrep_table(143, spinor=True)

    result = match_single_state_irrep(
        table=table,
        table_kpoint_label="K",
        state_index=0,
        computed_characters={1: 1.0 + 0.0j, 2: np.exp(-1j * np.pi / 3.0)},
    )

    assert result.status == "missing_characters"
    assert result.irrep_label is None


def test_match_single_state_irrep_rejects_ambiguous_decomposition():
    table = load_standard_irrep_table(143, spinor=True)

    result = match_single_state_irrep(
        table=table,
        table_kpoint_label="K",
        state_index=0,
        computed_characters={
            1: 2.0 + 0.0j,  # two-dimensional aggregate → ambiguous per state
            2: 1.0 + 0.0j,
            3: 1.0 + 0.0j,
        },
    )

    assert result.status == "ambiguous_irrep_label"
    assert result.irrep_label is None


def test_match_single_state_irrep_rejects_non_one_dimensional_irrep():
    """Characters that decompose to >1 irrep or multiplicity >1 are ambiguous per state."""
    table = load_standard_irrep_table(143, spinor=True)

    # Characters decomposing to -K4 (1) + -K5 (1) → ambiguous
    k4 = table.irreps_by_kpoint("K")[0]  # -K4
    k5 = table.irreps_by_kpoint("K")[1]  # -K5
    chars = {
        op: k4.characters[op] + k5.characters[op]
        for op in [1, 2, 3]
    }

    result = match_single_state_irrep(
        table=table,
        table_kpoint_label="K",
        state_index=0,
        computed_characters=chars,
    )

    assert result.status == "ambiguous_irrep_label"
    assert result.irrep_label is None


# -----------------------------------------------------------------------
# Source payload adapter for generic irrep matching
# -----------------------------------------------------------------------

from valleyscope.irreps.source_payload import build_source_payload_for_generic_matching


def test_adapter_sg143_spinor_c3_like_payload():
    """Adapter builds valid payload for SG143 spinor C3-like operations."""
    table = load_standard_irrep_table(143, spinor=True)
    detected = [
        {"operation_id": 1, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 2, "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 3, "rotation_frac": np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=int),
         "translation_frac": np.zeros(3)},
    ]
    payload = build_source_payload_for_generic_matching(
        table=table, source_hsp_label="K",
        detected_operations=detected,
        valley_preserving_operation_ids=[1, 2, 3],
    )
    assert payload["status"] == "ok"
    assert payload["provenance"]["source_hsp_label"] == "K"
    chars = payload["source_irrep_characters"]
    assert "-K5" in chars
    assert chars["-K5"][2] == pytest.approx(np.exp(-1j * np.pi / 3), abs=1e-4)


def test_adapter_blocked_missing_source_hsp():
    """Adapter blocks when source HSP has no irreps."""
    table = load_standard_irrep_table(143, spinor=True)
    detected = [{"operation_id": 1, "rotation_frac": np.eye(3, dtype=int),
                  "translation_frac": np.zeros(3)}]
    payload = build_source_payload_for_generic_matching(
        table=table, source_hsp_label="NONEXISTENT",
        detected_operations=detected,
        valley_preserving_operation_ids=[1],
    )
    assert payload["status"] == "blocked"


def test_adapter_blocked_incomplete_ops():
    """Adapter blocks when VP op not in detected_operations."""
    table = load_standard_irrep_table(143, spinor=True)
    detected = [{"operation_id": 1, "rotation_frac": np.eye(3, dtype=int),
                  "translation_frac": np.zeros(3)}]
    payload = build_source_payload_for_generic_matching(
        table=table, source_hsp_label="K",
        detected_operations=detected,
        valley_preserving_operation_ids=[1, 2],
    )
    assert payload["status"] == "blocked"
