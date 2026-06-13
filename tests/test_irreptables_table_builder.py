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


# -----------------------------------------------------------------------
# Source-basis inspector (fake loader, no real SG 150 dependency)
# -----------------------------------------------------------------------

_FAKE_EBR_DATA_INSPECT = {
    "basis": {"irrep_labels": ["-GM5", "-K5", "-K6", "-A5"], "degeneracies": [1, 1, 1, 1]},
    "ebrs": [
        {"ebr_name": "EBR_A", "wyckoff_position": "1a",
         "vector": [1.0, 0.0, 1.0, 1.0]},
        {"ebr_name": "EBR_B", "wyckoff_position": "1a",
         "vector": [1.0, 1.0, 0.0, 0.0]},
    ],
}


def _fake_inspect_loader(sg, spinor):
    return dict(_FAKE_EBR_DATA_INSPECT)


def test_inspector_returns_canonical_payload_with_fake_loader():
    """Inspector returns canonical payload when using fake source_loader."""
    from valleyscope.analysis.reduced_ebr_source_basis_inspector import (
        inspect_irreptables_source_basis,
    )
    info = inspect_irreptables_source_basis(
        150, spinful=True, source_loader=_fake_inspect_loader,
    )
    assert info["schema_version"] == "1.0.0"
    assert info["data_source"] == "irreptables"
    assert info["space_group_number"] == 150
    assert info["spinful"] is True
    assert info["source_basis_count"] == 4
    assert info["source_ebr_count"] == 2
    assert info["provenance"]["package"] == "irreptables"
    basis = info["source_basis"]
    assert len(basis) == 4
    assert basis[0] == {"source_label": "-GM5", "degeneracy": 1}


def test_inspector_mismatched_degeneracies_raises():
    """Mismatched label/degeneracy lengths raise ValueError."""
    from valleyscope.analysis.reduced_ebr_source_basis_inspector import (
        inspect_irreptables_source_basis,
    )
    bad = dict(_FAKE_EBR_DATA_INSPECT)
    bad["basis"] = {"irrep_labels": ["-GM5"], "degeneracies": []}

    def bad_loader(sg, spinor):
        return bad

    with pytest.raises(ValueError, match="degeneracies"):
        inspect_irreptables_source_basis(150, source_loader=bad_loader)


def test_inspector_missing_source_basis_raises():
    from valleyscope.analysis.reduced_ebr_source_basis_inspector import (
        inspect_irreptables_source_basis,
    )

    def bad_loader(sg, spinor):
        return {"basis": {"irrep_labels": [], "degeneracies": []}, "ebrs": []}

    with pytest.raises(ValueError, match="source basis"):
        inspect_irreptables_source_basis(150, source_loader=bad_loader)


def test_inspector_noninteger_degeneracy_raises():
    from valleyscope.analysis.reduced_ebr_source_basis_inspector import (
        inspect_irreptables_source_basis,
    )
    bad = dict(_FAKE_EBR_DATA_INSPECT)
    bad["basis"] = {"irrep_labels": ["-GM5"], "degeneracies": [1.5]}

    def bad_loader(sg, spinor):
        return bad

    with pytest.raises(ValueError, match="positive integer"):
        inspect_irreptables_source_basis(150, source_loader=bad_loader)


def test_cli_inspect_writes_json(tmp_path, capsys, monkeypatch):
    """CLI writes canonical JSON to --output with fake loader."""
    from valleyscope.cli import main
    from valleyscope.analysis.reduced_ebr_source_basis_inspector import (
        _default_irreptables_loader,
    )

    monkeypatch.setattr(
        "valleyscope.analysis.reduced_ebr_source_basis_inspector._default_irreptables_loader",
        staticmethod(lambda sg, spinor: dict(_FAKE_EBR_DATA_INSPECT)),
    )
    out_path = tmp_path / "inspect.json"
    rc = main([
        "inspect-ebr-source", "--space-group-number", "150",
        "-o", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source_basis_count"] == 4
    captured = capsys.readouterr().out
    assert "source basis count:" in captured


def test_inspector_module_no_material_names():
    """Inspector module must not contain material names."""
    src = Path(
        "valleyscope/analysis/reduced_ebr_source_basis_inspector.py"
    ).read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src, f"inspector must not contain {name!r}"


def test_inspector_no_forbidden_imports():
    """Inspector must not import irrep2, OR-Tools, or irrep.ebrs raw decomposition."""
    src = Path(
        "valleyscope/analysis/reduced_ebr_source_basis_inspector.py"
    ).read_text(encoding="utf-8")
    for forbidden in [
        "import irrep2", "from irrep2",
        "import ortools", "from ortools",
        "from irrep.ebrs", "import irrep.ebrs",
        "compute_ebr_decomposition(",
    ]:
        assert forbidden not in src, f"must not import {forbidden!r}"


# -----------------------------------------------------------------------
# Spec template / preflight validator
# -----------------------------------------------------------------------

_SOURCE_BASIS_PAYLOAD = {
    "schema_version": "1.0.0",
    "data_source": "irreptables",
    "space_group_number": 150,
    "spinful": True,
    "source_basis": [
        {"source_label": "-GM5", "degeneracy": 1},
        {"source_label": "-K5", "degeneracy": 1},
        {"source_label": "-K6", "degeneracy": 1},
        {"source_label": "-A5", "degeneracy": 1},
    ],
    "source_basis_count": 4,
    "source_ebr_count": 2,
    "provenance": {"package": "irreptables", "package_version": "fake"},
}

_VALID_SPEC = {
    "schema_version": "1.0.0",
    "data_source": "irreptables",
    "space_group_number": 150,
    "spinful": True,
    "source_hsp_by_irrep": {
        "-GM5": "GammaM", "-K5": "KM", "-K6": "KM", "-A5": "A",
    },
    "valleyscope_key_by_source_irrep": {
        "-GM5": "GammaM:C3_spinor_phase_+1/2",
        "-K5": "KM:C3_spinor_phase_+1/6",
        "-K6": "KM:C3_spinor_phase_-1/6",
        "-A5": "A:C1_spinor",
    },
    "expected_hsps": ["GammaM", "KM", "A"],
    "allowed_irrep_keys": [
        "GammaM:C3_spinor_phase_+1/2",
        "KM:C3_spinor_phase_+1/6",
        "KM:C3_spinor_phase_-1/6",
        "A:C1_spinor",
    ],
    "subspace_group_candidate": "C3_like",
}


def test_template_contains_all_source_labels():
    """Template has every source label with REQUIRED_ placeholder values."""
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        build_mapping_spec_template,
    )
    tmpl = build_mapping_spec_template(_SOURCE_BASIS_PAYLOAD)
    assert tmpl["schema_version"] == "1.0.0"
    assert tmpl["data_source"] == "irreptables"
    assert set(tmpl["source_hsp_by_irrep"].keys()) == {"-GM5", "-K5", "-K6", "-A5"}
    assert set(tmpl["valleyscope_key_by_source_irrep"].keys()) == {"-GM5", "-K5", "-K6", "-A5"}
    assert tmpl["subspace_group_candidate"] == "REQUIRED_FILL_BY_HUMAN"


def test_template_is_not_buildable(tmp_path):
    """Template contains placeholder values that will fail build."""
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        build_mapping_spec_template,
    )
    tmpl = build_mapping_spec_template(_SOURCE_BASIS_PAYLOAD)
    # Placeholder HSP mappings are not valid HSP labels.
    for v in tmpl["source_hsp_by_irrep"].values():
        assert v == "REQUIRED_FILL_BY_HUMAN"
    spec_path = tmp_path / "template.json"
    spec_path.write_text(json.dumps(tmpl), encoding="utf-8")
    with pytest.raises(ValueError, match="REQUIRED_FILL_BY_HUMAN|placeholder"):
        build_reduced_table_from_spec_file(spec_path, source_loader=_fake_loader([]))


def test_validator_accepts_completed_spec():
    """Validator accepts a fully filled spec."""
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        validate_mapping_spec_against_source_basis,
    )
    result = validate_mapping_spec_against_source_basis(
        _VALID_SPEC, _SOURCE_BASIS_PAYLOAD,
    )
    assert result["valid"] is True
    assert result["errors"] == []


def test_validator_rejects_placeholder():
    """Validator rejects specs with placeholder values."""
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        build_mapping_spec_template, validate_mapping_spec_against_source_basis,
    )
    tmpl = build_mapping_spec_template(_SOURCE_BASIS_PAYLOAD)
    result = validate_mapping_spec_against_source_basis(
        tmpl, _SOURCE_BASIS_PAYLOAD,
    )
    assert result["valid"] is False
    assert any("REQUIRED_FILL_BY_HUMAN" in e or "placeholder" in e for e in result["errors"])


def test_validator_rejects_missing_source_label():
    """Validator rejects spec missing a source label."""
    import copy
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        validate_mapping_spec_against_source_basis,
    )
    bad = copy.deepcopy(_VALID_SPEC)
    bad["source_hsp_by_irrep"] = {"-GM5": "GammaM"}
    bad["valleyscope_key_by_source_irrep"] = {"-GM5": "GammaM:C3_spinor_phase_+1/2"}
    result = validate_mapping_spec_against_source_basis(bad, _SOURCE_BASIS_PAYLOAD)
    assert result["valid"] is False
    assert any("missing" in e.lower() for e in result["errors"])


def test_validator_rejects_extra_label():
    """Validator rejects spec with extra label not in source basis."""
    import copy
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        validate_mapping_spec_against_source_basis,
    )
    bad = copy.deepcopy(_VALID_SPEC)
    bad["source_hsp_by_irrep"]["-Z99"] = "Z"
    bad["valleyscope_key_by_source_irrep"]["-Z99"] = "Z:irrep"
    result = validate_mapping_spec_against_source_basis(bad, _SOURCE_BASIS_PAYLOAD)
    assert result["valid"] is False
    assert any("extra" in e.lower() for e in result["errors"])


def test_template_rejects_nonbool_source_spinful():
    import copy
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        build_mapping_spec_template,
    )
    bad = copy.deepcopy(_SOURCE_BASIS_PAYLOAD)
    bad["spinful"] = "false"
    with pytest.raises(ValueError, match="spinful"):
        build_mapping_spec_template(bad)


def test_template_rejects_duplicate_source_labels():
    import copy
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        build_mapping_spec_template,
    )
    bad = copy.deepcopy(_SOURCE_BASIS_PAYLOAD)
    bad["source_basis"].append({"source_label": "-GM5", "degeneracy": 1})
    bad["source_basis_count"] = 5
    with pytest.raises(ValueError, match="duplicate"):
        build_mapping_spec_template(bad)


def test_validator_rejects_hsp_mapping_not_in_expected_hsps():
    import copy
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        validate_mapping_spec_against_source_basis,
    )
    bad = copy.deepcopy(_VALID_SPEC)
    bad["source_hsp_by_irrep"]["-GM5"] = "GhostHSP"
    result = validate_mapping_spec_against_source_basis(bad, _SOURCE_BASIS_PAYLOAD)
    assert result["valid"] is False
    assert any("expected_hsps" in e for e in result["errors"])


def test_validator_rejects_irrep_key_not_in_allowed_keys():
    import copy
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        validate_mapping_spec_against_source_basis,
    )
    bad = copy.deepcopy(_VALID_SPEC)
    bad["valleyscope_key_by_source_irrep"]["-GM5"] = "GammaM:ghost"
    result = validate_mapping_spec_against_source_basis(bad, _SOURCE_BASIS_PAYLOAD)
    assert result["valid"] is False
    assert any("allowed_irrep_keys" in e for e in result["errors"])


def test_cli_scaffold_writes_template(tmp_path, capsys):
    """CLI scaffolds a template from source basis JSON."""
    from valleyscope.cli import main
    src = tmp_path / "source.json"
    src.write_text(json.dumps(_SOURCE_BASIS_PAYLOAD), encoding="utf-8")
    out = tmp_path / "template.json"
    rc = main(["scaffold-spec", str(src), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    tmpl = json.loads(out.read_text(encoding="utf-8"))
    assert "REQUIRED_FILL_BY_HUMAN" in json.dumps(tmpl)


def test_cli_validate_spec_rejects_placeholder(tmp_path, capsys):
    """CLI validates and returns nonzero on invalid spec."""
    from valleyscope.cli import main
    src = tmp_path / "source.json"
    src.write_text(json.dumps(_SOURCE_BASIS_PAYLOAD), encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema_version": "1.0.0"}), encoding="utf-8")
    rc = main(["validate-spec", str(spec), str(src)])
    assert rc != 0


def test_cli_validate_spec_accepts_valid(tmp_path):
    """CLI returns zero on valid spec."""
    from valleyscope.cli import main
    src = tmp_path / "source.json"
    src.write_text(json.dumps(_SOURCE_BASIS_PAYLOAD), encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(_VALID_SPEC), encoding="utf-8")
    rc = main(["validate-spec", str(spec), str(src)])
    assert rc == 0


def test_validator_module_no_material_names():
    """Template/validator module must not contain material names."""
    src = Path(
        "valleyscope/analysis/reduced_ebr_spec_template_validator.py"
    ).read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src


def test_validator_module_no_forbidden_imports():
    """Template/validator must not import irrep2, OR-Tools, or irrep.ebrs."""
    src = Path(
        "valleyscope/analysis/reduced_ebr_spec_template_validator.py"
    ).read_text(encoding="utf-8")
    for forbidden in [
        "import irrep2", "from irrep2",
        "import ortools", "from ortools",
        "from irrep.ebrs", "import irrep.ebrs",
    ]:
        assert forbidden not in src


# -----------------------------------------------------------------------
# build-reduced-ebr-table --source-basis preflight E2E
# -----------------------------------------------------------------------

# Minimal EBR data matching _VALID_SPEC labels.
_PREFLIGHT_EBR_DATA = {
    "basis": {"irrep_labels": ["-GM5", "-K5", "-K6", "-A5"], "degeneracies": [1, 1, 1, 1]},
    "ebrs": [
        {"ebr_name": "EBR_A", "wyckoff_position": "1a", "vector": [1, 0, 1, 1]},
        {"ebr_name": "EBR_B", "wyckoff_position": "1a", "vector": [1, 1, 0, 0]},
    ],
}

_PREFLIGHT_SOURCE_BASIS = {
    "schema_version": "1.0.0", "data_source": "irreptables",
    "space_group_number": 150, "spinful": True,
    "source_basis": [
        {"source_label": "-GM5", "degeneracy": 1},
        {"source_label": "-K5", "degeneracy": 1},
        {"source_label": "-K6", "degeneracy": 1},
        {"source_label": "-A5", "degeneracy": 1},
    ],
    "source_basis_count": 4, "source_ebr_count": 2,
    "provenance": {"package": "irreptables", "package_version": "fake"},
}

_PREFLIGHT_VALID_SPEC = {
    "schema_version": "1.0.0", "data_source": "irreptables",
    "space_group_number": 150, "spinful": True,
    "source_hsp_by_irrep": {"-GM5": "GammaM", "-K5": "KM", "-K6": "KM", "-A5": "A"},
    "valleyscope_key_by_source_irrep": {
        "-GM5": "GammaM:C3_spinor_phase_+1/2", "-K5": "KM:C3_spinor_phase_+1/6",
        "-K6": "KM:C3_spinor_phase_-1/6", "-A5": "A:C1_spinor",
    },
    "expected_hsps": ["GammaM", "KM", "A"],
    "allowed_irrep_keys": [
        "GammaM:C3_spinor_phase_+1/2", "KM:C3_spinor_phase_+1/6",
        "KM:C3_spinor_phase_-1/6", "A:C1_spinor",
    ],
    "subspace_group_candidate": "C3_like",
}


def test_build_with_source_basis_preflight_passes_for_valid_spec(
    tmp_path, capsys, monkeypatch,
):
    """--source-basis validates spec before building; valid spec passes."""
    from valleyscope.cli import main

    source = tmp_path / "source.json"
    source.write_text(json.dumps(_PREFLIGHT_SOURCE_BASIS), encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_PREFLIGHT_VALID_SPEC), encoding="utf-8")
    out = tmp_path / "table.json"

    monkeypatch.setattr(
        "valleyscope.analysis.irreptables_runtime_table_builder._load_ebr_data_from_irreptables",
        lambda sg, spinor: dict(_PREFLIGHT_EBR_DATA),
    )

    rc = main([
        "build-reduced-ebr-table", str(spec_path),
        "--source-basis", str(source), "-o", str(out),
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "preflight validation passed" in captured
    assert out.exists()


def test_build_with_source_basis_fails_on_placeholder_template(
    tmp_path, capsys, monkeypatch,
):
    """--source-basis fails preflight on a template with placeholders."""
    from valleyscope.cli import main
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        build_mapping_spec_template,
    )

    source = tmp_path / "source.json"
    source.write_text(json.dumps(_PREFLIGHT_SOURCE_BASIS), encoding="utf-8")
    tmpl = build_mapping_spec_template(_PREFLIGHT_SOURCE_BASIS)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(tmpl), encoding="utf-8")
    out = tmp_path / "table.json"
    source_loader_calls = []

    def _source_loader_should_not_run(sg, spinor):
        source_loader_calls.append((sg, spinor))
        raise AssertionError("source loader should not run after preflight failure")

    monkeypatch.setattr(
        "valleyscope.analysis.irreptables_runtime_table_builder._load_ebr_data_from_irreptables",
        _source_loader_should_not_run,
    )

    rc = main([
        "build-reduced-ebr-table", str(spec_path),
        "--source-basis", str(source), "-o", str(out),
    ])
    assert rc != 0, "should fail preflight on placeholder template"
    captured = capsys.readouterr()
    assert "preflight validation failed" in captured.err
    assert captured.out == ""
    assert source_loader_calls == []
    assert not out.exists()


def test_build_without_source_basis_skips_preflight(
    tmp_path, capsys, monkeypatch,
):
    """Omitting --source-basis preserves existing behavior."""
    from valleyscope.cli import main

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_PREFLIGHT_VALID_SPEC), encoding="utf-8")
    out = tmp_path / "table.json"

    monkeypatch.setattr(
        "valleyscope.analysis.irreptables_runtime_table_builder._load_ebr_data_from_irreptables",
        lambda sg, spinor: dict(_PREFLIGHT_EBR_DATA),
    )

    rc = main([
        "build-reduced-ebr-table", str(spec_path), "-o", str(out),
    ])
    assert rc == 0
    assert out.exists()
    captured = capsys.readouterr().out
    assert "preflight" not in captured


def test_build_preflight_no_material_names():
    """Preflight fixture data must not contain material names."""
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in json.dumps(_PREFLIGHT_SOURCE_BASIS)
        assert name not in json.dumps(_PREFLIGHT_VALID_SPEC)
        assert name not in json.dumps(_PREFLIGHT_EBR_DATA)


# -----------------------------------------------------------------------
# C3 reduced EBR authoring audit doc contract
# -----------------------------------------------------------------------

def test_c3_audit_doc_exists_and_covers_physical_objects():
    """C3 audit doc must exist and cover HSP little group, valley mapping, etc."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md")
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for term in [
        "HSP little group", "valley mapping", "valley-preserving subgroup",
        "valley-preserving operation", "valley-changing operation",
        "valley sewing matrix", "source 3D irrep labels",
        "reduced EBR vector basis", "C3_spinor_phase",
        "inspect-ebr-source", "150",
    ]:
        assert term.lower() in text.lower(), f"missing '{term}'"
    assert "source_basis_count" in text or "22 source" in text


def test_c3_audit_doc_no_material_names_and_no_builtin_tables():
    """C3 audit doc must not contain material names or claim built-in tables."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in doc
    assert "no built-in" in doc.lower() or "not hardcoded" in doc.lower()


def test_c3_audit_doc_does_not_put_c2_in_reduced_c3_basis():
    """C3 audit doc must keep C2 sewing data out of the reduced C3 irrep basis."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "{E, C3, C3^2, C2" not in doc
    assert "MM identity-only" in doc


def test_c3_audit_doc_keeps_degenerate_k6_out_of_1d_phase_basis():
    """C3 audit doc must not map degenerate K6 source labels to 1D phases."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "`-K6` | KM | 1 | `KM:C3_spinor_phase_-1/6`" not in doc
    assert "`-K6` | 2" in doc


def test_c3_audit_doc_blocks_public_api_phase_mapping_without_evidence():
    """Opaque public source labels need human review, not review-ready status.
    The doc may say "no label is review_ready" as a meta-statement, but no
    individual label row may carry `review_ready` as its status value."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    # "No source irrep label is `review_ready`" is the correct meta-statement.
    assert "No source irrep label is `review_ready`" in doc
    assert "needs_human_review" in doc
    assert "blocked_by_missing_restriction_data" in doc


def test_c3_audit_doc_names_public_ebr_api_boundary_precisely():
    """C3 audit doc must name the public irreptables EBR loader boundary."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "irreptables.ebrs.load_ebr_data" in doc
    assert "irreptables.load_ebr_data" not in doc


def test_c3_convention_packet_uses_2pi_phase_character_formula():
    """C3 convention packet must use phase labels in units of 2*pi."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "exp(2*pi*i*phase_i)" in doc
    assert "exp(4*pi*i*phase_i)" in doc
    assert "exp(+i*pi*phase_i)" not in doc
    assert "(conjugate)" not in doc


def test_c3_audit_has_readiness_summary_table():
    """C3 audit doc must have a concise readiness summary table at the top."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "C3 Convention Readiness Summary" in doc
    assert "No source irrep label is `review_ready`" in doc
    # Summary table must list both candidate and blocked labels.
    for label in ["-GM5", "-K5", "-GM4", "-GM6", "-K4", "-K6"]:
        assert label in doc


def test_c3_audit_has_machine_checked_evidence_section():
    """C3 audit doc must separate machine-checked from external evidence."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "Machine-Checked vs. External Evidence" in doc
    assert "Evidence Already Machine-Checked By Existing Tests" in doc
    assert "Evidence Requiring External / Manual Review" in doc
    assert "test_phase_tables.py" in doc
    assert "test_irreptables_table_builder.py" in doc
    assert "test_reduced_ebr_smoke.py" in doc


def test_c3_audit_has_explicit_human_decisions_section():
    """C3 audit doc must list exactly what human decisions are still required."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "Human Decisions Still Required" in doc
    assert "Confirm the 1D labels with Bilbao character evidence" in doc
    assert "`-GM4`, `-GM5`" in doc
    assert "`-K4`, and `-K5`" in doc
    assert "Confirm `-GM6` and `-K6`" in doc
    assert "Decide which labels enter the first reduced basis" in doc
    assert "Sign off on provenance record" in doc
    assert "no C3-like reduced EBR table" in doc
    assert "claimed as reviewed" in doc


def test_c3_audit_has_feasibility_assessment_section():
    """C3 audit doc must have a feasibility assessment section with the
    full physical mapping chain, per-label evidence table, and API audit."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "C3 Character Evidence Feasibility Assessment" in doc
    # Physical mapping chain
    for term in [
        "source SG 150 spinful irrep label",
        "sampled moire HSP",
        "HSP little group",
        "valley mapping",
        "valley-preserving subgroup",
        "ValleyScope spinful C3 irrep phase key",
    ]:
        assert term.lower() in doc.lower(), f"missing '{term}'"
    # Per-label evidence table
    assert "Per-Label Evidence Requirements" in doc
    # API audit entries
    assert "Path 1:" in doc or "irreptables.ebrs.load_ebr_data(150, True)" in doc
    assert "Path 2:" in doc or 'IrrepTable("150", True)' in doc
    assert "Path 3:" in doc or "irrep.spacegroup_irreps.SpaceGroupIrreps" in doc
    assert "INSUFFICIENT" in doc
    assert "AVAILABLE" in doc
    assert "BILBAO" in doc
    # No evidence source claims review_ready
    assert "`review_ready`" not in doc.split("Feasibility Assessment")[1] if "Feasibility Assessment" in doc else True


def test_c3_feasibility_has_checklist():
    """Feasibility section must have a human-reviewer checklist with at
    least items 1-8 covering evidence source, per-label confirmation,
    restriction decomposition, convention verification, and sign-off."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "Human Decision Checklist" in doc
    for item in [
        "Select evidence source",
        "Confirm the 1D C3 eigenphases",
        "Confirm `-K6`",
        "Confirm `-GM6`",
        "Decide the first C3 reduced-basis source labels",
        "Verify phase convention",
        "Sign off",
    ]:
        assert item in doc, f"missing checklist item: '{item}'"


def test_c3_feasibility_distinguishes_facts():
    """Feasibility section must separate machine-checkable from human-required facts."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "Machine-Checkable Facts" in doc
    assert "External / Human-Required Facts" in doc
    # Machine-checkable facts cite test files
    assert "test_irreptables_table_builder.py" in doc
    assert "test_phase_tables.py" in doc
    # Human-required facts cite external references
    assert "Bradley" in doc or "Bilbao" in doc or "literature" in doc.lower()


def test_c3_feasibility_conclusion_is_conservative():
    """Feasibility conclusion must state C3 character evidence is now
    machine-checkable via irreptables Bilbao data, while still requiring
    human review before shipping any reviewed table."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "Feasibility Conclusion" in doc
    assert "C3 character evidence is now machine-checkable" in doc
    assert 'IrrepTable("150", True)' in doc
    assert "no C3-like reduced EBR table" in doc
    assert "may be shipped" in doc


def test_c3_audit_has_no_stale_irreptables_character_claims():
    """C3 audit doc must not retain pre-IrrepTable correction claims."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    stale_claims = [
        "`-K5` | KM | 1 | `KM:C3_spinor_phase_+1/6`",
        "Independent run evidence suggests +1/6",
        "irreptables has no SG 150 character data",
        "No — `irreptables` has no phase data",
        "No public API exposes C3 character",
        "Confirm `-K5` → `C3_spinor_phase_+1/6`",
        "Expected: `exp(+i*pi/3)` = phase +1/6",
        "`IrrepTable(150, True)`\n   fails because `irreptables/data/` lacks SG 150 files",
        "decomposes to {+1/6, -1/6}",
        "and decompose as `{+1/6, -1/6}`",
    ]
    for stale in stale_claims:
        assert stale not in doc, f"stale C3 audit claim remains: {stale}"

    current_claims = [
        'IrrepTable("150", True)',
        "`-K5` | 1 | op2=-1, op3=-1 | +1/2 from op2 | `KM:C3_spinor_phase_+1/2`",
        "`-GM4` | 1 | op2=-1, op3=-1 | +1/2 from op2 | `GammaM:C3_spinor_phase_+1/2`",
        "`-K4` | 1 | op2=-1, op3=-1 | +1/2 from op2 | `KM:C3_spinor_phase_+1/2`",
        "candidate multiplicity `{+1/6: 1, -1/6: 1}`",
        "pending human provenance sign-off",
    ]
    for current in current_claims:
        assert current in doc, f"missing corrected C3 audit claim: {current}"


# -----------------------------------------------------------------------
# Irreptables Bilbao irrep data verification
# -----------------------------------------------------------------------

def _require_irreptables_irreps():
    """Skip test if irreptables.irreps cannot be imported."""
    try:
        import irreptables.irreps  # noqa: F401
    except ImportError:
        pytest.skip("irreptables.irreps not available")


def test_irreptables_irrep_table_loads_sg150_spinful():
    """IrrepTable('150', True) loads SG 150 spinful irreps from Bilbao data."""
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    tbl = ir.IrrepTable("150", True)
    assert tbl.number_str == "150"
    assert tbl.spinor is True
    assert tbl.nsym == 6
    assert len(tbl.irreps) == 16


def test_irreptables_sg150_contains_all_6_target_source_labels():
    """All six in-scope source labels appear in the table."""
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    tbl = ir.IrrepTable("150", True)
    names = {irrep.name for irrep in tbl.irreps}
    for label in ["-GM4", "-GM5", "-GM6", "-K4", "-K5", "-K6"]:
        assert label in names, f"missing source label {label}"


def test_irreptables_sg150_1d_labels_have_c3_character_minus_one():
    """All four 1D labels at GM and K have op2 C3 character = -1.

    This gives a candidate ValleyScope C3_spinor_phase_+1/2 after human
    convention review.  The -4/-5 distinction is from C2 characters, not C3.
    """
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    tbl = ir.IrrepTable("150", True)
    for irrep in tbl.irreps:
        if irrep.name in ("-GM4", "-GM5", "-K4", "-K5"):
            assert irrep.dim == 1, f"{irrep.name} should be 1D"
            c3_char = irrep.characters.get(2)
            c32_char = irrep.characters.get(3)
            assert c3_char is not None, f"{irrep.name} missing C3 character"
            assert abs(c3_char.real + 1) < 1e-10, (
                f"{irrep.name} C3 char should be -1, got {c3_char}"
            )
            assert abs(c3_char.imag) < 1e-10
            assert abs(c32_char.real + 1) < 1e-10


def test_irreptables_sg150_2d_labels_need_explicit_c3_restriction():
    """-GM6 and -K6 have op2/op3 characters +1/+1 and need provenance review."""
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    tbl = ir.IrrepTable("150", True)
    for irrep in tbl.irreps:
        if irrep.name in ("-GM6", "-K6"):
            assert irrep.dim == 2, f"{irrep.name} should be 2D"
            c3_char = irrep.characters.get(2)
            c32_char = irrep.characters.get(3)
            assert c3_char is not None, f"{irrep.name} missing C3 character"
            assert c32_char is not None, f"{irrep.name} missing C3^2 character"
            assert abs(c3_char.real - 1) < 1e-10
            assert abs(c3_char.imag) < 1e-10
            assert abs(c32_char.real - 1) < 1e-10
            assert abs(c32_char.imag) < 1e-10


def test_irreptables_sg150_c3_operations_are_indices_2_and_3():
    """Operations 2 and 3 (1-indexed Bilbao convention) are C3 and C3^2."""
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    import numpy as np
    tbl = ir.IrrepTable("150", True)
    # Operation 2 -> C3 (order 3, trace 0)
    R2 = tbl.symmetries[1].R
    assert np.allclose(np.linalg.matrix_power(R2, 3), np.eye(3))
    assert not np.allclose(R2, np.eye(3))
    # Operation 3 -> C3^2 (order 3, trace 0)
    R3 = tbl.symmetries[2].R
    assert np.allclose(np.linalg.matrix_power(R3, 3), np.eye(3))
    assert not np.allclose(R3, np.eye(3))
    # Operations 4,5,6 -> C2 (order 2, trace -1)
    for i in [3, 4, 5]:
        R = tbl.symmetries[i].R
        assert np.allclose(R @ R, np.eye(3))


def test_irreptables_sg150_ops_4_5_6_are_c2_valley_changing():
    """Operations 4,5,6 are C2 (order 2, trace -1) — valley-changing at K/K'."""
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    import numpy as np
    tbl = ir.IrrepTable("150", True)
    for i in [3, 4, 5]:
        R = tbl.symmetries[i].R
        assert int(np.trace(R)) == -1
        assert int(np.linalg.det(R)) == 1


def test_irreptables_sg150_no_kA_labels():
    """KA source labels should not exist — A labels belong to A HSP, not KA."""
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    tbl = ir.IrrepTable("150", True)
    names = {irrep.name for irrep in tbl.irreps}
    assert "-KA4" not in names
    assert "-KA5" not in names
    assert "-KA6" not in names
    # But -A4, -A5, -A6 do exist (at A HSP, not KA)
    for label in ["-A4", "-A5", "-A6"]:
        assert label in names, f"missing {label}"


# -----------------------------------------------------------------------
# C3 double-group lift convention audit
# -----------------------------------------------------------------------

def test_double_group_s2_cubed_is_minus_identity():
    """S2³ = -I confirms the Bilbao convention satisfies
    the spinful double-group relation g³ = -E for the C3 generator."""
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    import numpy as np
    tbl = ir.IrrepTable("150", True)
    S2 = tbl.symmetries[1].S  # op2 = C3 generator
    S2_cubed = S2 @ S2 @ S2
    assert np.allclose(S2_cubed, -np.eye(2), atol=1e-4), (
        f"S2³ should be -I, got {S2_cubed}"
    )


def test_double_group_op3_is_central_negative_op2_squared():
    """S2 @ S2 = -S3 within data precision.

    The Bilbao op3 lift is the central-negative representative of the
    group-theoretic square of the op2 C3 generator.
    """
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    import numpy as np
    tbl = ir.IrrepTable("150", True)
    S2 = tbl.symmetries[1].S
    S3 = tbl.symmetries[2].S
    S22 = S2 @ S2
    assert not np.allclose(S3, S22, atol=1e-6), (
        "S3 should not equal S2 @ S2"
    )
    assert np.allclose(S22, -S3, atol=1e-4), (
        "S2 @ S2 should equal -S3 within irreptables data precision"
    )


def test_double_group_r3_is_r2_squared():
    """R3 = R2 @ R2: spatial rotations satisfy C3^2 = C3²."""
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    import numpy as np
    tbl = ir.IrrepTable("150", True)
    R2 = tbl.symmetries[1].R
    R3 = tbl.symmetries[2].R
    assert np.allclose(R3, R2 @ R2)


def test_double_group_chi_op3_is_central_negative_chi_op2_squared():
    """For all four 1D irreps at GM/K, -chi(op3) = chi(op2)^2.

    ValleyScope may compute chi(C3^2) from op2 alone or by flipping the
    central-negative Bilbao op3 character.
    """
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    tbl = ir.IrrepTable("150", True)
    for irrep in tbl.irreps:
        if irrep.name in ("-GM4", "-GM5", "-K4", "-K5") and irrep.dim == 1:
            chi2 = irrep.characters.get(2)
            chi3 = irrep.characters.get(3)
            chi2_sq = chi2 * chi2
            assert abs((-chi3) - chi2_sq) < 1e-10, (
                f"{irrep.name}: -chi(op3)={-chi3} should equal "
                f"chi(op2)^2={chi2_sq}"
            )


def test_double_group_chi_op2_squared_is_positive_unity():
    """chi(op2)² = +1 for all four 1D labels at GM/K.
    This is the correct group-theoretic chi(C3²) for the +1/2 phase."""
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    tbl = ir.IrrepTable("150", True)
    for irrep in tbl.irreps:
        if irrep.name in ("-GM4", "-GM5", "-K4", "-K5") and irrep.dim == 1:
            chi2 = irrep.characters.get(2)
            chi2_sq = chi2 * chi2
            assert abs(chi2_sq - 1.0) < 1e-10, (
                f"{irrep.name}: chi(op2)² should be +1, got {chi2_sq}"
            )


def test_double_group_2d_chi_op3_not_minus_one():
    """For -GM6/-K6 (2D), -chi(op3) gives the group-theoretic C3^2 trace."""
    _require_irreptables_irreps()
    import irreptables.irreps as ir
    tbl = ir.IrrepTable("150", True)
    for irrep in tbl.irreps:
        if irrep.name in ("-GM6", "-K6") and irrep.dim == 2:
            chi3 = irrep.characters.get(3)
            assert abs(chi3 - 1.0) < 1e-10, (
                f"{irrep.name}: Bilbao chi(op3) is {chi3}, not the "
                f"group-theoretic -1"
            )
            assert abs((-chi3) - (-1.0)) < 1e-10


def test_double_group_audit_doc_covers_lift_convention():
    """C3 audit doc must cover the double-group lift convention finding."""
    doc = Path("docs/reduced_ebr_c3_authoring_audit.md").read_text(encoding="utf-8")
    assert "C3 Double-Group Lift Convention Audit" in doc
    assert "S2 @ S2 = -S3" in doc
    assert "central-negative" in doc
    assert "chi(C3²) = -chi(op3)" in doc or "chi(C3^2) = -chi(op3)" in doc
    assert "diag(exp(+2iπ/3), exp(-2iπ/3))" in doc or "diag(exp(+2i" in doc


# -----------------------------------------------------------------------
# C3 mapping signoff packet doc-contract tests
# -----------------------------------------------------------------------

def test_signoff_packet_exists():
    """docs/reduced_ebr_c3_mapping_signoff_packet.md must exist."""
    path = Path("docs/reduced_ebr_c3_mapping_signoff_packet.md")
    assert path.exists(), "signoff packet file missing"


def test_signoff_packet_contains_all_six_source_labels():
    """Signoff packet must list all six in-scope source labels with
    their accepted ValleyScope keys."""
    doc = Path("docs/reduced_ebr_c3_mapping_signoff_packet.md").read_text(encoding="utf-8")
    assert "-GM4" in doc
    assert "-GM5" in doc
    assert "-GM6" in doc
    assert "-K4" in doc
    assert "-K5" in doc
    assert "-K6" in doc
    # 1D labels -> +1/2
    for label in ["-GM4", "-GM5", "-K4", "-K5"]:
        assert "C3_spinor_phase_+1/2" in doc
    # degenerate labels -> multiplicity
    assert "{+1/6: 1, -1/6: 1}" in doc


def test_signoff_packet_states_central_sign_convention():
    """Signoff packet must state chi(C3)=chi(op2) and chi(C3^2)=-chi(op3)."""
    doc = Path("docs/reduced_ebr_c3_mapping_signoff_packet.md").read_text(encoding="utf-8")
    assert "chi(C3) = chi(op2)" in doc or "chi(C3)=chi(op2)" in doc
    assert "chi(C3²) = -chi(op3)" in doc or "chi(C3^2) = -chi(op3)" in doc


def test_signoff_packet_excludes_c2_from_c3_basis():
    """Signoff packet must state C2/op4-6 are valley sewing data and
    must not enter the C3 reduced EBR vector basis."""
    doc = Path("docs/reduced_ebr_c3_mapping_signoff_packet.md").read_text(encoding="utf-8")
    assert "valley sewing data" in doc.lower()
    assert "must not enter" in doc.lower() and "c3 reduced ebr vector basis" in doc.lower()


def test_signoff_packet_states_no_builtin_table_shipped():
    """Signoff packet must state it is not a reduced EBR table, not a
    decomposition report, and no JSON is shipped."""
    doc = Path("docs/reduced_ebr_c3_mapping_signoff_packet.md").read_text(encoding="utf-8")
    assert "not a reduced ebr table" in doc.lower()
    assert "not a reduced ebr decomposition report" in doc.lower()
    assert "does not ship any" in doc.lower() and "reduced ebr table" in doc.lower()
    assert "does not ship any json" in doc.lower()


def test_signoff_packet_has_checklist():
    """Signoff packet must include a signoff checklist with at least 8 items."""
    doc = Path("docs/reduced_ebr_c3_mapping_signoff_packet.md").read_text(encoding="utf-8")
    assert "Signoff Checklist" in doc
    assert "Source data accepted" in doc
    assert "HSP set confirmed" in doc
    assert "Valley-preserving subgroup confirmed" in doc
    assert "Central-sign convention confirmed" in doc
    assert "Six source labels mapped" in doc
    assert "C2 valley sewing data excluded" in doc
    assert "Reviewer signoff" in doc


def test_signoff_packet_records_provenance_fields():
    """Signoff packet must list required provenance fields for a future
    mapping spec/table."""
    doc = Path("docs/reduced_ebr_c3_mapping_signoff_packet.md").read_text(encoding="utf-8")
    for field in ["data_source", "space_group_number", "spinful",
                  "subspace_group_candidate", "expected_hsps",
                  "valleyscope_reduction", "review_status",
                  "reviewer", "review_date", "review_method",
                  "source_reference", "central_sign_convention"]:
        assert field in doc, f"missing provenance field: {field}"


def test_signoff_packet_distinguishes_catalog_enforced_provenance():
    """Signoff packet must not claim every reviewer-required field is
    currently enforced by the package-data catalog gate."""
    doc = Path("docs/reduced_ebr_c3_mapping_signoff_packet.md").read_text(encoding="utf-8")
    assert "All provenance fields are enforced" not in doc
    assert "Catalog-enforced provenance fields" in doc
    assert "Reviewer-required signoff fields" in doc


def test_signoff_packet_no_material_names():
    """Signoff packet must not contain real material names."""
    doc = Path("docs/reduced_ebr_c3_mapping_signoff_packet.md").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in doc, f"signoff packet contains {name!r}"


# -----------------------------------------------------------------------
# C3 external mapping spec draft tests
# -----------------------------------------------------------------------

def _load_spec_draft():
    path = Path("docs/reduced_ebr_c3_external_mapping_spec_draft.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_spec_draft_exists_and_parses():
    """Draft JSON must exist and parse as valid JSON."""
    spec = _load_spec_draft()
    assert isinstance(spec, dict)


def test_spec_draft_status_is_draft_not_builder_compatible():
    """Status must be draft_not_builder_compatible."""
    spec = _load_spec_draft()
    assert spec["status"] == "draft_not_builder_compatible"


def test_spec_draft_has_all_six_source_labels():
    """source_hsp_by_irrep must contain exactly the six in-scope labels."""
    spec = _load_spec_draft()
    assert spec["source_hsp_by_irrep"] == {
        "-GM4": "GammaM",
        "-GM5": "GammaM",
        "-GM6": "GammaM",
        "-K4": "KM",
        "-K5": "KM",
        "-K6": "KM",
    }


def test_spec_draft_allowed_irrep_keys_has_all_six_c3_keys():
    """allowed_irrep_keys must contain exactly the six C3 phase keys."""
    spec = _load_spec_draft()
    assert set(spec["allowed_irrep_keys"]) == {
        "GammaM:C3_spinor_phase_+1/6",
        "GammaM:C3_spinor_phase_+1/2",
        "GammaM:C3_spinor_phase_-1/6",
        "KM:C3_spinor_phase_+1/6",
        "KM:C3_spinor_phase_+1/2",
        "KM:C3_spinor_phase_-1/6"
    }


def test_spec_draft_multiplicity_maps_1d_labels_to_plus_half():
    """GM4/GM5/K4/K5 must each map to +1/2 with multiplicity 1."""
    spec = _load_spec_draft()
    m = spec["valleyscope_irrep_multiplicity_by_source_irrep"]
    assert set(m) == {"-GM4", "-GM5", "-GM6", "-K4", "-K5", "-K6"}
    assert m["-GM4"] == {"GammaM:C3_spinor_phase_+1/2": 1}
    assert m["-GM5"] == {"GammaM:C3_spinor_phase_+1/2": 1}
    assert m["-K4"] == {"KM:C3_spinor_phase_+1/2": 1}
    assert m["-K5"] == {"KM:C3_spinor_phase_+1/2": 1}


def test_spec_draft_multiplicity_decomposes_gm6_k6():
    """GM6 must decompose to GammaM +1/6 and -1/6; K6 to KM +1/6 and -1/6."""
    spec = _load_spec_draft()
    m = spec["valleyscope_irrep_multiplicity_by_source_irrep"]
    assert m["-GM6"] == {
        "GammaM:C3_spinor_phase_+1/6": 1,
        "GammaM:C3_spinor_phase_-1/6": 1,
    }
    assert m["-K6"] == {
        "KM:C3_spinor_phase_+1/6": 1,
        "KM:C3_spinor_phase_-1/6": 1,
    }


def test_spec_draft_builder_compatibility_is_false_with_reason():
    """builder_compatibility must be false and reason must mention both
    many-to-one aggregation and one-to-many decomposition."""
    spec = _load_spec_draft()
    bc = spec["builder_compatibility"]
    assert bc["compatible_with_build_reduced_table_from_spec_file"] is False
    reason = bc["reason"].lower()
    assert "many-to-one" in reason
    assert "aggregation" in reason
    assert "one-to-many" in reason
    assert "decomposition" in reason


def test_spec_draft_not_under_package_data():
    """Draft must be under docs/, not under valleyscope/data/reduced_ebr/."""
    path = Path("docs/reduced_ebr_c3_external_mapping_spec_draft.json")
    assert path.exists()
    assert "valleyscope/data/reduced_ebr" not in str(path.resolve())


def test_spec_draft_no_material_names():
    """Draft spec JSON must not contain real material names."""
    spec = _load_spec_draft()
    text = json.dumps(spec)
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in text, f"spec draft contains {name!r}"


def test_spec_draft_provenance_has_required_fields():
    """Provenance must include central_sign_convention and valleyscope_reduction."""
    spec = _load_spec_draft()
    p = spec["provenance"]
    assert p["valleyscope_reduction"] == "sampled_hsp_valley_preserving"
    assert "central_sign_convention" in p
    assert "chi(C3)=chi(op2)" in p["central_sign_convention"]


# -----------------------------------------------------------------------
# Multiplicity-aware reducer / builder tests
# -----------------------------------------------------------------------

# C3-like test data: 6 source labels, fake EBR vectors of all ones.
_C3_SOURCE_LABELS = ["-GM4", "-GM5", "-GM6", "-K4", "-K5", "-K6"]
_C3_HSPS = {
    "-GM4": "GammaM", "-GM5": "GammaM", "-GM6": "GammaM",
    "-K4": "KM", "-K5": "KM", "-K6": "KM",
}
_C3_MULTIPLICITIES = {
    "-GM4": {"GammaM:C3_spinor_phase_+1/2": 1},
    "-GM5": {"GammaM:C3_spinor_phase_+1/2": 1},
    "-GM6": {
        "GammaM:C3_spinor_phase_+1/6": 1,
        "GammaM:C3_spinor_phase_-1/6": 1,
    },
    "-K4": {"KM:C3_spinor_phase_+1/2": 1},
    "-K5": {"KM:C3_spinor_phase_+1/2": 1},
    "-K6": {
        "KM:C3_spinor_phase_+1/6": 1,
        "KM:C3_spinor_phase_-1/6": 1,
    },
}
_C3_EXPECTED_HSPS = ["GammaM", "KM"]
_C3_ALLOWED_KEYS = [
    "GammaM:C3_spinor_phase_+1/6",
    "GammaM:C3_spinor_phase_+1/2",
    "GammaM:C3_spinor_phase_-1/6",
    "KM:C3_spinor_phase_+1/6",
    "KM:C3_spinor_phase_+1/2",
    "KM:C3_spinor_phase_-1/6",
]

_C3_FAKE_EBR_DATA = {
    "basis": {
        "irrep_labels": _C3_SOURCE_LABELS,
        "degeneracies": [1, 1, 2, 1, 1, 2],
    },
    "ebrs": [
        {"ebr_name": "EBR_A", "vector": [1, 1, 1, 1, 1, 1]},
    ],
}


def test_multiplicity_aware_normalizer_expands_degenerate_labels():
    """Normalizer produces multiple basis entries for degenerate source labels."""
    from valleyscope.analysis.irrep_data_normalizer import (
        build_runtime_source_payload_from_ebr_data,
    )
    payload = build_runtime_source_payload_from_ebr_data(
        ebr_data=_C3_FAKE_EBR_DATA,
        source_hsp_by_irrep=_C3_HSPS,
        valleyscope_irrep_multiplicity_by_source_irrep=_C3_MULTIPLICITIES,
    )
    basis = payload["basis"]
    # -GM6 contributes 2 entries (+1/6 and -1/6)
    gm6_entries = [e for e in basis if e["source_label"] == "-GM6"]
    assert len(gm6_entries) == 2
    gm6_keys = {e["valleyscope_irrep_key"] for e in gm6_entries}
    assert gm6_keys == {
        "GammaM:C3_spinor_phase_+1/6",
        "GammaM:C3_spinor_phase_-1/6",
    }
    # All source_index values point to the -GM6 index (2)
    for e in gm6_entries:
        assert e["source_index"] == 2


def test_multiplicity_aware_reducer_produces_correct_vector():
    """C3-like all-ones source vector reduced with multiplicities gives
    [1, 2, 1, 1, 2, 1] in allowed_irrep_keys order."""
    from valleyscope.analysis.irrep_data_normalizer import (
        build_runtime_source_payload_from_ebr_data,
    )
    from valleyscope.analysis.irrep_runtime_reducer import (
        build_reduced_table_from_runtime_source,
    )
    payload = build_runtime_source_payload_from_ebr_data(
        ebr_data=_C3_FAKE_EBR_DATA,
        source_hsp_by_irrep=_C3_HSPS,
        valleyscope_irrep_multiplicity_by_source_irrep=_C3_MULTIPLICITIES,
    )
    table = build_reduced_table_from_runtime_source(
        source_payload=payload,
        expected_hsps=_C3_EXPECTED_HSPS,
        allowed_irrep_keys=_C3_ALLOWED_KEYS,
        subspace_group_candidate="C3_like",
    )
    assert table["irreps"] == _C3_ALLOWED_KEYS
    # Source vector: all ones over [-GM4,-GM5,-GM6,-K4,-K5,-K6]
    # Reduced:
    #   GM +1/6: -GM6 x mul=1    => 1
    #   GM +1/2: -GM4 x mul=1 + -GM5 x mul=1  => 2
    #   GM -1/6: -GM6 x mul=1    => 1
    #   KM +1/6: -K6 x mul=1     => 1
    #   KM +1/2: -K4 x mul=1 + -K5 x mul=1   => 2
    #   KM -1/6: -K6 x mul=1     => 1
    assert table["ebrs"][0]["vector"] == [1, 2, 1, 1, 2, 1]


def test_multiplicity_aware_builder_v1_1_spec(tmp_path):
    """build_reduced_table_from_spec_file accepts v1.1 multiplicity spec."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_reduced_table_from_spec_file,
    )
    spec = {
        "schema_version": "1.1.0",
        "data_source": "irreptables",
        "space_group_number": 150,
        "spinful": True,
        "source_hsp_by_irrep": _C3_HSPS,
        "valleyscope_irrep_multiplicity_by_source_irrep": _C3_MULTIPLICITIES,
        "expected_hsps": _C3_EXPECTED_HSPS,
        "allowed_irrep_keys": _C3_ALLOWED_KEYS,
        "subspace_group_candidate": "C3_like",
    }
    spec_path = tmp_path / "spec_v11.json"
    spec_path.write_text(json.dumps(spec))

    def fake_loader(sg, spinor):
        return dict(_C3_FAKE_EBR_DATA)

    table = build_reduced_table_from_spec_file(
        spec_path, source_loader=fake_loader,
    )
    assert table["irreps"] == _C3_ALLOWED_KEYS
    assert table["ebrs"][0]["vector"] == [1, 2, 1, 1, 2, 1]


def test_multiplicity_aware_builder_filters_unmapped_nonsampled_source_labels(tmp_path):
    """v1.1 specs need multiplicities only for sampled-HSP source labels."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_reduced_table_from_spec_file,
    )
    labels = [*_C3_SOURCE_LABELS, "-A5"]
    hsp_by_irrep = {**_C3_HSPS, "-A5": "A"}
    ebr_data = {
        "basis": {
            "irrep_labels": labels,
            "degeneracies": [1, 1, 2, 1, 1, 2, 1],
        },
        "ebrs": [
            {"ebr_name": "EBR_A", "vector": [1, 1, 1, 1, 1, 1, 1]},
        ],
    }
    spec = {
        "schema_version": "1.1.0",
        "data_source": "irreptables",
        "space_group_number": 150,
        "spinful": True,
        "source_hsp_by_irrep": hsp_by_irrep,
        "valleyscope_irrep_multiplicity_by_source_irrep": _C3_MULTIPLICITIES,
        "expected_hsps": _C3_EXPECTED_HSPS,
        "allowed_irrep_keys": _C3_ALLOWED_KEYS,
        "subspace_group_candidate": "C3_like",
    }
    spec_path = tmp_path / "spec_v11_partial.json"
    spec_path.write_text(json.dumps(spec))

    table = build_reduced_table_from_spec_file(
        spec_path, source_loader=lambda sg, spinor: ebr_data,
    )

    assert table["ebrs"][0]["vector"] == [1, 2, 1, 1, 2, 1]
    assert table["provenance"]["source_basis_count"] == 7


def test_multiplicity_aware_builder_still_accepts_v1_0_spec(tmp_path):
    """build_reduced_table_from_spec_file still accepts v1.0 one-to-one spec."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_reduced_table_from_spec_file,
    )
    spec = _canonical_spec()
    spec_path = tmp_path / "spec_v10.json"
    spec_path.write_text(json.dumps(spec))
    calls = []
    table = build_reduced_table_from_spec_file(
        spec_path, source_loader=_fake_loader(calls),
    )
    assert calls == [(150, True)]
    assert table["irreps"] == _ALLOWED_KEYS


def test_multiplicity_aware_normalizer_rejects_both_maps():
    """Providing both legacy and multiplicity maps raises ValueError."""
    from valleyscope.analysis.irrep_data_normalizer import (
        build_runtime_source_payload_from_ebr_data,
    )
    with pytest.raises(ValueError, match="only one"):
        build_runtime_source_payload_from_ebr_data(
            ebr_data=_C3_FAKE_EBR_DATA,
            source_hsp_by_irrep=_C3_HSPS,
            valleyscope_key_by_source_irrep={"-GM4": "GammaM:C3_spinor_phase_+1/2"},
            valleyscope_irrep_multiplicity_by_source_irrep={"-GM4": {"GammaM:C3_spinor_phase_+1/2": 1}},
        )


def test_multiplicity_aware_normalizer_rejects_neither_map():
    """Providing neither legacy nor multiplicity map raises ValueError."""
    from valleyscope.analysis.irrep_data_normalizer import (
        build_runtime_source_payload_from_ebr_data,
    )
    with pytest.raises(ValueError, match="either"):
        build_runtime_source_payload_from_ebr_data(
            ebr_data=_C3_FAKE_EBR_DATA,
            source_hsp_by_irrep=_C3_HSPS,
        )


def test_multiplicity_aware_normalizer_rejects_non_integer_mult():
    """Non-integer multiplicity raises ValueError."""
    from valleyscope.analysis.irrep_data_normalizer import (
        build_runtime_source_payload_from_ebr_data,
    )
    bad_mult = dict(_C3_MULTIPLICITIES)
    bad_mult["-GM4"] = {"GammaM:C3_spinor_phase_+1/2": 1.5}
    with pytest.raises(ValueError, match="integer"):
        build_runtime_source_payload_from_ebr_data(
            ebr_data=_C3_FAKE_EBR_DATA,
            source_hsp_by_irrep=_C3_HSPS,
            valleyscope_irrep_multiplicity_by_source_irrep=bad_mult,
        )


def test_multiplicity_aware_normalizer_rejects_zero_mult():
    """Zero or negative multiplicity raises ValueError."""
    from valleyscope.analysis.irrep_data_normalizer import (
        build_runtime_source_payload_from_ebr_data,
    )
    bad_mult = dict(_C3_MULTIPLICITIES)
    bad_mult["-GM4"] = {"GammaM:C3_spinor_phase_+1/2": 0}
    with pytest.raises(ValueError, match="positive"):
        build_runtime_source_payload_from_ebr_data(
            ebr_data=_C3_FAKE_EBR_DATA,
            source_hsp_by_irrep=_C3_HSPS,
            valleyscope_irrep_multiplicity_by_source_irrep=bad_mult,
        )


def test_multiplicity_aware_normalizer_rejects_missing_source_label():
    """Missing source label in multiplicity map raises ValueError."""
    from valleyscope.analysis.irrep_data_normalizer import (
        build_runtime_source_payload_from_ebr_data,
    )
    bad_mult = dict(_C3_MULTIPLICITIES)
    bad_mult.pop("-GM6")
    with pytest.raises(ValueError, match="missing"):
        build_runtime_source_payload_from_ebr_data(
            ebr_data=_C3_FAKE_EBR_DATA,
            source_hsp_by_irrep=_C3_HSPS,
            valleyscope_irrep_multiplicity_by_source_irrep=bad_mult,
        )


def test_no_forbidden_imports_in_production_files():
    """No production file may import irrep2, OR-Tools, or irrep.ebrs."""
    for fname in [
        "valleyscope/analysis/irrep_data_normalizer.py",
        "valleyscope/analysis/irrep_runtime_reducer.py",
        "valleyscope/analysis/irreptables_runtime_table_builder.py",
    ]:
        src = Path(fname).read_text(encoding="utf-8")
        for forbidden in [
            "import irrep2", "from irrep2",
            "import ortools", "from ortools",
            "from irrep.ebrs", "import irrep.ebrs",
        ]:
            assert forbidden not in src, f"{fname} must not import {forbidden!r}"


def test_no_material_names_in_production_files():
    """No material names in production code."""
    for fname in [
        "valleyscope/analysis/irrep_data_normalizer.py",
        "valleyscope/analysis/irrep_runtime_reducer.py",
        "valleyscope/analysis/irreptables_runtime_table_builder.py",
    ]:
        src = Path(fname).read_text(encoding="utf-8")
        for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
            assert name not in src, f"{fname} contains {name!r}"
