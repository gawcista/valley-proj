"""Tests for the public irreptables -> ValleyScope reduced table builder."""

import builtins
import json
from pathlib import Path

import pytest

from valleyscope.analysis.irreptables_runtime_table_builder import (
    _load_ebr_data_from_irreptables,
    build_reduced_table_from_irreptables,
    build_reduced_table_from_spec_file,
)
from valleyscope.analysis.reduced_ebr_mapping import (
    build_reduced_ebr_mapping,
    load_reduced_ebr_table,
)


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


def _canonical_spec(**overrides):
    spec = {
        "schema_version": "1.0.0",
        "data_source": "irreptables",
        "space_group_number": 150,
        "spinful": True,
        "source_hsp_by_irrep": _SOURCE_HSP_BY_IRREP,
        "valleyscope_key_by_source_irrep": _VALLEYSCOPE_KEY_BY_SOURCE_IRREP,
        "expected_hsps": _EXPECTED_HSPS,
        "allowed_irrep_keys": _ALLOWED_KEYS,
        "subspace_group_candidate": "C3_like",
        "provenance": {"review_status": "fixture-only"},
    }
    spec.update(overrides)
    return spec


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


def test_spec_file_helper_uses_canonical_spec_with_fake_loader(tmp_path):
    calls = []
    spec_path = tmp_path / "mapping_spec.json"
    spec_path.write_text(json.dumps(_canonical_spec()), encoding="utf-8")

    table = build_reduced_table_from_spec_file(
        spec_path,
        source_loader=_fake_loader(calls),
    )

    assert calls == [(150, True)]
    assert table["irreps"] == _ALLOWED_KEYS
    assert table["provenance"]["data_source"] == "irreptables"
    assert table["provenance"]["review_status"] == "fixture-only"
    out_path = tmp_path / "table.json"
    out_path.write_text(json.dumps(table), encoding="utf-8")
    assert load_reduced_ebr_table(out_path)["irreps"] == _ALLOWED_KEYS


def test_spec_file_table_feeds_reduced_ebr_mapping_e2e(tmp_path):
    calls = []
    spec_path = tmp_path / "mapping_spec.json"
    spec_path.write_text(json.dumps(_canonical_spec()), encoding="utf-8")

    table = build_reduced_table_from_spec_file(
        spec_path,
        source_loader=_fake_loader(calls),
    )
    table_path = tmp_path / "reduced_table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    validated = load_reduced_ebr_table(table_path)

    bundle = {
        "bundles": [
            {
                "bundle_id": "synthetic_bundle",
                "valley": "K",
                "subspace_group_candidate": "C3_like",
                "ready_for_external_solver": True,
                "expected_hsps": _EXPECTED_HSPS,
                "irreps_by_kpoint": {
                    "GammaM": [
                        "C3_spinor_phase_+1/2",
                        "C3_spinor_phase_+1/2",
                    ],
                    "KM": [
                        "C3_spinor_phase_+1/6",
                        "C3_spinor_phase_-1/6",
                    ],
                },
            }
        ],
    }
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle,
        table=validated,
    )

    assert calls == [(150, True)]
    provenance = validated["provenance"]
    assert provenance["data_source"] == "irreptables"
    assert provenance["valleyscope_reduction"] == "sampled_hsp_valley_preserving"
    assert result["status"] == "solved_exact"
    solution = result["solutions"][0]
    assert solution["classification"] == "atomic-compatible-candidate"
    assert solution["ebr_decomposition"] == [
        {"label": "EBR_A", "coefficient": 1},
        {"label": "EBR_B", "coefficient": 1},
    ]


def test_spec_file_helper_requires_canonical_keys(tmp_path):
    legacy_spec = {
        "sg_number": 150,
        "spinor": True,
        "source_hsp_by_irrep": _SOURCE_HSP_BY_IRREP,
        "valleyscope_key_by_source_irrep": _VALLEYSCOPE_KEY_BY_SOURCE_IRREP,
        "expected_hsps": _EXPECTED_HSPS,
        "allowed_irrep_keys": _ALLOWED_KEYS,
        "subspace_group_candidate": "C3_like",
    }
    spec_path = tmp_path / "legacy_spec.json"
    spec_path.write_text(json.dumps(legacy_spec), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        build_reduced_table_from_spec_file(spec_path, source_loader=_fake_loader([]))


def test_spec_file_helper_rejects_wrong_data_source(tmp_path):
    spec_path = tmp_path / "mapping_spec.json"
    spec_path.write_text(
        json.dumps(_canonical_spec(data_source="irrep")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data_source"):
        build_reduced_table_from_spec_file(spec_path, source_loader=_fake_loader([]))


def test_spec_file_helper_requires_bool_spinful(tmp_path):
    spec_path = tmp_path / "mapping_spec.json"
    spec_path.write_text(
        json.dumps(_canonical_spec(spinful="false")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="spinful"):
        build_reduced_table_from_spec_file(spec_path, source_loader=_fake_loader([]))


def test_cli_build_reduced_ebr_table_writes_validated_table_with_fake_builder(
    tmp_path, capsys, monkeypatch
):
    from valleyscope.analysis import irreptables_runtime_table_builder as builder
    from valleyscope.cli import main

    def fake_build_from_spec_file(spec_path):
        assert Path(spec_path).name == "mapping_spec.json"
        table, _calls = _build_with_fake_loader()
        return table

    monkeypatch.setattr(
        builder,
        "build_reduced_table_from_spec_file",
        fake_build_from_spec_file,
    )
    spec_path = tmp_path / "mapping_spec.json"
    spec_path.write_text(json.dumps(_canonical_spec()), encoding="utf-8")
    out_path = tmp_path / "table.json"

    rc = main(["build-reduced-ebr-table", str(spec_path), "-o", str(out_path)])

    assert rc == 0
    loaded = load_reduced_ebr_table(out_path)
    assert loaded["irreps"] == _ALLOWED_KEYS
    captured = capsys.readouterr().out
    assert "space group number: 150" in captured
    assert "spinful:            True" in captured
    assert "filtered zero EBRs: 1" in captured


def test_build_reduced_ebr_table_spec_doc_is_linked_and_material_free():
    doc = Path("docs/build_reduced_ebr_table_spec.md")
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for key in [
        "schema_version",
        "data_source",
        "space_group_number",
        "spinful",
        "source_hsp_by_irrep",
        "valleyscope_key_by_source_irrep",
        "expected_hsps",
        "allowed_irrep_keys",
        "subspace_group_candidate",
    ]:
        assert key in text
    for forbidden in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert forbidden not in text

    schema = Path("docs/reduced_ebr_table_schema.md").read_text(encoding="utf-8")
    assert "build_reduced_ebr_table_spec.md" in schema


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
