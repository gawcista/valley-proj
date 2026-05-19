import numpy as np
import pytest

from valleyscope.irreps.tables import load_standard_irrep_table, match_table_operations


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
