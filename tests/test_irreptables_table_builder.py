"""Tests for the public irreptables -> ValleyScope reduced table builder."""

import builtins
import json
from pathlib import Path

import pytest

from valleyscope.analysis.irreptables_runtime_table_builder import (
    _load_ebr_data_from_irreptables,
    build_reduced_table_from_irreptables,
)
from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table


_SAMPLE_EBR_DATA = {
    "basis": {
        "irrep_labels": ["-GM5", "-K5", "-K6", "-A5"],
        "degeneracies": [1, 1, 1, 1],
    },
    "ebrs": [
        {"ebr_name": "EBR_A", "wyckoff_position": "1a", "vector": [1, 0, 1, 1]},
        {"ebr_name": "EBR_B", "wyckoff_position": "1b", "vector": [1, 1, 0, 0]},
        {"ebr_name": "EBR_GHOST", "wyckoff_position": "2c", "vector": [0, 0, 0, 1]},
    ],
}

_SOURCE_HSP_BY_IRREP = {
    "-GM5": "GammaM",
    "-K5": "KM",
    "-K6": "KM",
    "-A5": "A",
}

_VALLEYSCOPE_KEY_BY_SOURCE_IRREP = {
    "-GM5": "GammaM:C3_spinor_phase_+1/2",
    "-K5": "KM:C3_spinor_phase_+1/6",
    "-K6": "KM:C3_spinor_phase_-1/6",
    "-A5": "A:C1_spinor",
}

_EXPECTED_HSPS = ["GammaM", "KM"]
_ALLOWED_KEYS = [
    "GammaM:C3_spinor_phase_+1/2",
    "KM:C3_spinor_phase_+1/6",
    "KM:C3_spinor_phase_-1/6",
]


def _fake_loader(calls):
    def loader(space_group_number, spinful):
        calls.append((space_group_number, spinful))
        return _SAMPLE_EBR_DATA

    return loader


def _build_with_fake_loader(**overrides):
    calls = []
    kwargs = {
        "space_group_number": 150,
        "spinful": True,
        "source_loader": _fake_loader(calls),
        "source_hsp_by_irrep": _SOURCE_HSP_BY_IRREP,
        "valleyscope_key_by_source_irrep": _VALLEYSCOPE_KEY_BY_SOURCE_IRREP,
        "expected_hsps": _EXPECTED_HSPS,
        "allowed_irrep_keys": _ALLOWED_KEYS,
        "subspace_group_candidate": "C3_like",
    }
    kwargs.update(overrides)
    return build_reduced_table_from_irreptables(**kwargs), calls


def test_builder_fake_loader_produces_loadable_reduced_table(tmp_path):
    table, calls = _build_with_fake_loader()
    assert calls == [(150, True)]
    assert table["subspace_group_candidate"] == "C3_like"
    assert table["expected_hsps"] == _EXPECTED_HSPS
    assert table["irreps"] == _ALLOWED_KEYS
    assert [ebr["label"] for ebr in table["ebrs"]] == ["EBR_A", "EBR_B"]
    assert table["provenance"]["filtered_zero_vector_ebr_count"] == 1
    assert table["provenance"]["filtered_zero_vector_ebrs"] == ["EBR_GHOST"]

    path = tmp_path / "reduced_table.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    loaded = load_reduced_ebr_table(path)
    assert loaded["irreps"] == _ALLOWED_KEYS


def test_builder_provenance_marks_public_source_and_reduction_contract():
    table, _calls = _build_with_fake_loader(
        provenance={"review_status": "fixture-only"},
    )
    provenance = table["provenance"]
    assert provenance["data_source"] == "irreptables"
    assert provenance["package"] == "irreptables"
    assert "package_version" in provenance
    assert provenance["space_group_number"] == 150
    assert provenance["spinful"] is True
    assert provenance["expected_hsps"] == _EXPECTED_HSPS
    assert provenance["subspace_group_candidate"] == "C3_like"
    assert provenance["valleyscope_reduction"] == "sampled_hsp_valley_preserving"
    assert provenance["review_status"] == "fixture-only"


def test_builder_accepts_legacy_aliases_for_initial_callers():
    table, calls = _build_with_fake_loader(
        space_group_number=None,
        spinful=None,
        sg_number="150",
        spinor=False,
    )
    assert calls == [("150", False)]
    assert table["provenance"]["space_group_number"] == "150"
    assert table["provenance"]["spinful"] is False


def test_builder_rejects_conflicting_aliases():
    with pytest.raises(ValueError, match="space_group_number or sg_number"):
        _build_with_fake_loader(sg_number=150)
    with pytest.raises(ValueError, match="spinful or spinor"):
        _build_with_fake_loader(spinor=True)


def test_builder_propagates_explicit_mapping_errors():
    with pytest.raises(ValueError, match="source_hsp_by_irrep"):
        _build_with_fake_loader(source_hsp_by_irrep={})
    with pytest.raises(ValueError, match="valleyscope_key_by_source_irrep"):
        _build_with_fake_loader(valleyscope_key_by_source_irrep={})


def test_builder_propagates_noninteger_source_vectors():
    bad_data = {
        "basis": _SAMPLE_EBR_DATA["basis"],
        "ebrs": [{"ebr_name": "bad", "vector": [1, 0.5, 0, 0]}],
    }

    def bad_loader(space_group_number, spinful):
        return bad_data

    with pytest.raises(ValueError, match="not an integer"):
        _build_with_fake_loader(source_loader=bad_loader)


def test_default_loader_missing_import_reports_clear_runtimeerror(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "irreptables.ebrs":
            raise ImportError("forced missing irreptables")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="cannot import public irreptables"):
        _load_ebr_data_from_irreptables(150, True)


def test_builder_source_has_no_forbidden_dependencies_or_material_names():
    src = Path("valleyscope/analysis/irreptables_runtime_table_builder.py").read_text(
        encoding="utf-8"
    )
    for forbidden in [
        "import irrep2",
        "from irrep2",
        "from irrep.ebrs",
        "import irrep.ebrs",
        "compute_ebr_decomposition(",
        "tMoTe2",
        "tZrSe2",
        "MoTe2",
        "ZrSe2",
    ]:
        assert forbidden not in src
