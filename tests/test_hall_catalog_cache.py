"""Static Hall metadata reuse must never reuse source-setting trust."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import spglib

from valleyscope.analysis.standard_setting_kmap import (
    derive_irreptables_standard_setting_identity as derive_identity,
)
from valleyscope.irreps.tables import load_standard_irrep_table


def _count_type_queries(monkeypatch):
    query = spglib.get_spacegroup_type
    calls = []

    def counted(hall_number):
        calls.append(hall_number)
        return query(hall_number)

    monkeypatch.setattr(spglib, "get_spacegroup_type", counted)
    return calls


def test_successful_catalog_is_reused_but_affine_queries_remain_live(monkeypatch):
    table = load_standard_irrep_table(5, spinor=False)
    calls = _count_type_queries(monkeypatch)
    first = derive_identity(table, 5)
    cold_count = len(calls)
    assert calls[:530] == list(range(1, 531))
    calls.clear()
    second = derive_identity(table, 5)

    assert first == second
    assert second["status"] == "unique_match"
    assert second["candidate_hall_numbers"] == list(range(9, 18))
    assert second["affine_matching_hall_numbers"] == [9]
    assert cold_count - len(calls) == 530
    assert calls and all(hall in range(9, 18) for hall in calls)


def test_warm_catalog_revalidates_source_operations_and_centering(monkeypatch):
    table = load_standard_irrep_table(5, spinor=False)
    calls = _count_type_queries(monkeypatch)
    assert derive_identity(table, 5)["hall_number"] == 9
    calls.clear()

    ambiguous = derive_identity(replace(table, name=""), 5)
    assert ambiguous["status"] == "ambiguous"
    assert ambiguous["affine_matching_hall_numbers"] == [9, 10, 11]
    bad_operation = replace(
        table.operations[1], translation_frac=np.array([0.123, 0.0, 0.0]),
    )
    invalid = derive_identity(
        replace(table, operations=(table.operations[0], bad_operation)), 5,
    )
    assert invalid["status"] == "no_match"
    assert invalid["affine_matching_hall_numbers"] == []
    assert derive_identity(table, 5)["hall_number"] == 9
    assert all(hall in range(9, 18) for hall in calls)


def test_returned_lists_cannot_pollute_warm_catalog(monkeypatch):
    table = load_standard_irrep_table(5, spinor=False)
    _count_type_queries(monkeypatch)
    first = derive_identity(table, 5)
    expected = deepcopy(first)
    first["candidate_hall_numbers"].clear()
    first["affine_matching_hall_numbers"].append(17)
    first["centering_cosets"][0][0] = 0.25
    assert derive_identity(table, 5) == expected


def test_cached_metadata_does_not_alias_provider_objects(monkeypatch):
    table = load_standard_irrep_table(5, spinor=False)
    query = spglib.get_spacegroup_type
    returned_rows = []

    def mutable_row(hall_number):
        row = query(hall_number)
        result = SimpleNamespace(
            number=row.number, hall_number=row.hall_number,
            hall_symbol=row.hall_symbol, international_short=row.international_short,
        )
        returned_rows.append(result)
        return result

    monkeypatch.setattr(spglib, "get_spacegroup_type", mutable_row)
    expected = derive_identity(table, 5)
    for row in returned_rows:
        row.number = 1
        row.hall_symbol = "P 1"
    assert derive_identity(table, 5) == expected


@pytest.mark.parametrize("hall_number", [10, 530])
@pytest.mark.parametrize("defect", ["none", "bool_number", "wrong_hall", "missing_symbol"])
def test_incomplete_catalog_is_blocked_and_retried(monkeypatch, hall_number, defect):
    table = load_standard_irrep_table(5, spinor=False)
    query = spglib.get_spacegroup_type
    failing = True
    calls = []

    def transient(hall):
        calls.append(hall)
        row = query(hall)
        if failing and hall == hall_number:
            if defect == "none":
                return None
            changes = {
                "bool_number": {"number": True},
                "wrong_hall": {"hall_number": hall + 1},
                "missing_symbol": {"hall_symbol": None},
            }
            return replace(row, **changes[defect])
        return row

    monkeypatch.setattr(spglib, "get_spacegroup_type", transient)
    failed = derive_identity(table, 5)
    assert failed["status"] == "unresolved"
    assert failed["reason"] == "spglib_hall_catalog_unavailable"
    assert failed["candidate_hall_numbers"] == []
    assert "hall_number" not in failed

    failing = False
    calls.clear()
    recovered = derive_identity(table, 5)
    assert calls[:530] == list(range(1, 531))
    assert recovered["status"] == "unique_match"
    assert recovered["candidate_hall_numbers"] == list(range(9, 18))
    assert recovered["hall_number"] == 9


def test_provider_exception_propagates_without_caching_failure(monkeypatch):
    table = load_standard_irrep_table(5, spinor=False)
    query = spglib.get_spacegroup_type
    failing = True

    def transient(hall):
        if failing and hall == 10:
            raise RuntimeError("Hall query failed")
        return query(hall)

    monkeypatch.setattr(spglib, "get_spacegroup_type", transient)
    with pytest.raises(RuntimeError, match="Hall query failed"):
        derive_identity(table, 5)
    failing = False
    assert derive_identity(table, 5)["hall_number"] == 9


def test_replaced_provider_cannot_reuse_previous_catalog(monkeypatch):
    table = load_standard_irrep_table(5, spinor=False)
    _count_type_queries(monkeypatch)
    assert derive_identity(table, 5)["hall_number"] == 9
    monkeypatch.setattr(spglib, "get_spacegroup_type", lambda hall: None)
    failed = derive_identity(table, 5)
    assert failed["status"] == "unresolved"
    assert failed["reason"] == "spglib_hall_catalog_unavailable"


def test_warm_catalog_does_not_hide_operation_database_failure(monkeypatch):
    table = load_standard_irrep_table(5, spinor=False)
    _count_type_queries(monkeypatch)
    assert derive_identity(table, 5)["hall_number"] == 9
    monkeypatch.setattr(spglib, "get_symmetry_from_database", lambda hall: None)
    failed = derive_identity(table, 5)
    assert failed["status"] == "no_match"
    assert failed["affine_matching_hall_numbers"] == []
    assert "hall_number" not in failed
