"""Tests for package-style 3D EBR data normalization and availability probing."""

import json
import subprocess
from pathlib import Path

import pytest

from valleyscope.analysis.irrep_availability_probe import (
    probe_irrep_availability,
    probe_irrep_is_importable,
    probe_irrep_runtime_sources,
)
from valleyscope.analysis.irrep_data_normalizer import (
    build_runtime_source_payload_from_ebr_data,
    normalize_irrep_ebr_data_to_source_payload,
)
from valleyscope.analysis.irrep_runtime_reducer import (
    build_reduced_table_from_runtime_source,
)
from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table


_SAMPLE_EBR_DATA = {
    "basis": {
        "irrep_labels": ["-GM5", "-K5", "-K6", "-A5"],
        "degeneracies": [1, 1, 1, 1],
    },
    "ebrs": [
        {"ebr_name": "EBR_A", "wyckoff_position": "1a", "vector": [1.0, 0.0, 1.0, 1.0]},
        {"ebr_name": "EBR_B", "wyckoff_position": "1a", "vector": [1.0, 1.0, 0.0, 0.0]},
    ],
    "source": {"package": "irreptables", "version": "fake-test"},
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

_HSP = ["GammaM", "KM"]
_KEYS = [
    "GammaM:C3_spinor_phase_+1/2",
    "KM:C3_spinor_phase_+1/6",
    "KM:C3_spinor_phase_-1/6",
]


def _payload(ebr_data=_SAMPLE_EBR_DATA):
    return build_runtime_source_payload_from_ebr_data(
        ebr_data=ebr_data,
        source_hsp_by_irrep=_SOURCE_HSP_BY_IRREP,
        valleyscope_key_by_source_irrep=_VALLEYSCOPE_KEY_BY_SOURCE_IRREP,
        source={"package": "irreptables", "version": "fake-test"},
    )


def test_package_ebr_data_normalizes_to_runtime_source_payload():
    payload = _payload()
    # Legacy one-to-one normalizer output now includes source_index and
    # multiplicity=1 for reducer backward compat.
    for i, entry in enumerate(payload["basis"]):
        assert entry["source_index"] == i
        assert entry["multiplicity"] == 1
    # Verify the essential fields excluding auto-generated ones.
    core = [{k: v for k, v in e.items() if k not in ("source_index", "multiplicity")}
            for e in payload["basis"]]
    assert core == [
        {
            "source_label": "-GM5",
            "hsp": "GammaM",
            "valleyscope_irrep_key": "GammaM:C3_spinor_phase_+1/2",
        },
        {
            "source_label": "-K5",
            "hsp": "KM",
            "valleyscope_irrep_key": "KM:C3_spinor_phase_+1/6",
        },
        {
            "source_label": "-K6",
            "hsp": "KM",
            "valleyscope_irrep_key": "KM:C3_spinor_phase_-1/6",
        },
        {"source_label": "-A5", "hsp": "A", "valleyscope_irrep_key": "A:C1_spinor"},
    ]
    assert payload["ebrs"][0]["label"] == "EBR_A"
    assert payload["ebrs"][0]["wyckoff_position"] == "1a"
    assert payload["source"]["package"] == "irreptables"


def test_integer_valued_floats_become_exact_ints():
    payload = _payload()
    assert payload["ebrs"][0]["vector"] == [1, 0, 1, 1]
    assert all(type(v) is int for v in payload["ebrs"][0]["vector"])


def test_non_integer_vector_entry_rejected():
    bad = dict(_SAMPLE_EBR_DATA)
    bad["ebrs"] = [{"ebr_name": "bad", "vector": [1.0, 0.5, 0.0, 0.0]}]
    with pytest.raises(ValueError, match="not an integer"):
        _payload(bad)


def test_vector_length_mismatch_rejected():
    bad = dict(_SAMPLE_EBR_DATA)
    bad["ebrs"] = [{"ebr_name": "bad", "vector": [1.0, 0.0]}]
    with pytest.raises(ValueError, match="vector length"):
        _payload(bad)


def test_duplicate_ebr_names_are_qualified_by_wyckoff_position():
    data = {
        "basis": _SAMPLE_EBR_DATA["basis"],
        "ebrs": [
            {"ebr_name": "EBR_DUP", "wyckoff_position": "1a", "vector": [1, 0, 1, 0]},
            {"ebr_name": "EBR_DUP", "wyckoff_position": "1b", "vector": [1, 1, 0, 0]},
        ],
    }
    payload = _payload(data)
    assert [ebr["label"] for ebr in payload["ebrs"]] == [
        "EBR_DUP @ 1a",
        "EBR_DUP @ 1b",
    ]


def test_duplicate_ebr_name_same_wyckoff_position_rejected():
    data = {
        "basis": _SAMPLE_EBR_DATA["basis"],
        "ebrs": [
            {"ebr_name": "EBR_DUP", "wyckoff_position": "1a", "vector": [1, 0, 1, 0]},
            {"ebr_name": "EBR_DUP", "wyckoff_position": "1a", "vector": [1, 1, 0, 0]},
        ],
    }
    with pytest.raises(ValueError, match="duplicate EBR label"):
        _payload(data)


def test_missing_explicit_hsp_mapping_rejected():
    hsp_map = dict(_SOURCE_HSP_BY_IRREP)
    hsp_map.pop("-K6")
    with pytest.raises(ValueError, match="missing source_hsp_by_irrep"):
        build_runtime_source_payload_from_ebr_data(
            ebr_data=_SAMPLE_EBR_DATA,
            source_hsp_by_irrep=hsp_map,
            valleyscope_key_by_source_irrep=_VALLEYSCOPE_KEY_BY_SOURCE_IRREP,
        )


def test_missing_explicit_valleyscope_key_mapping_rejected():
    key_map = dict(_VALLEYSCOPE_KEY_BY_SOURCE_IRREP)
    key_map.pop("-K6")
    with pytest.raises(ValueError, match="missing valleyscope_key_by_source_irrep"):
        build_runtime_source_payload_from_ebr_data(
            ebr_data=_SAMPLE_EBR_DATA,
            source_hsp_by_irrep=_SOURCE_HSP_BY_IRREP,
            valleyscope_key_by_source_irrep=key_map,
        )


def test_compatibility_wrapper_still_requires_explicit_maps():
    with pytest.raises(ValueError, match="hsp_name_map is required"):
        normalize_irrep_ebr_data_to_source_payload(_SAMPLE_EBR_DATA)


def test_normalizer_output_feeds_reducer_and_table_loader(tmp_path):
    source_payload = _payload()
    table = build_reduced_table_from_runtime_source(
        source_payload=source_payload,
        expected_hsps=_HSP,
        allowed_irrep_keys=_KEYS,
        subspace_group_candidate="P3",
    )
    assert table["provenance"]["package"] == "irreptables"
    path = tmp_path / "reduced_table.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    loaded = load_reduced_ebr_table(path)
    assert loaded["irreps"] == _KEYS


def test_probe_runtime_sources_returns_structured_status_without_raising():
    info = probe_irrep_runtime_sources()
    assert set(info) >= {"irrep", "irreptables", "submodules", "errors"}
    assert info["unsafe_native_probe_enabled"] is False
    assert "irrep.spacegroup_irreps" in info["submodules"]
    assert "irrep.ebrs" in info["submodules"]
    assert "irreptables.ebrs" in info["submodules"]
    assert info["submodules"]["irrep.ebrs"]["available"] is False
    assert "probe_skipped_reason" in info["submodules"]["irrep.ebrs"]
    assert "load_ebr_data_available" in info["irreptables"]


def test_probe_skips_irrep_ebrs_native_import_by_default(monkeypatch):
    from valleyscope.analysis import irrep_availability_probe as probe

    calls = []

    def fake_run(args, **kwargs):
        script = args[2]
        calls.append(script)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps({"public_names": ["load_ebr_data"]}),
            stderr="",
        )

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    info = probe.probe_irrep_runtime_sources(probe_unsafe_native=False)

    assert not any("irrep.ebrs" in script for script in calls)
    assert any("irreptables.ebrs" in script for script in calls)
    irrep_ebrs = info["submodules"]["irrep.ebrs"]
    assert irrep_ebrs["available"] is False
    assert irrep_ebrs["probe_skipped_reason"].startswith("unsafe optional native")


def test_probe_can_opt_into_irrep_ebrs_native_import(monkeypatch):
    from valleyscope.analysis import irrep_availability_probe as probe

    calls = []

    def fake_run(args, **kwargs):
        script = args[2]
        calls.append(script)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps({"public_names": ["compute_ebr_decomposition"]}),
            stderr="",
        )

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    info = probe.probe_irrep_runtime_sources(probe_unsafe_native=True)

    assert info["unsafe_native_probe_enabled"] is True
    assert any("irrep.ebrs" in script for script in calls)
    assert info["submodules"]["irrep.ebrs"]["available"] is True
    assert info["submodules"]["irrep.ebrs"]["probe_skipped_reason"] is None


def test_probe_compatibility_wrappers_return_expected_shapes():
    info = probe_irrep_availability()
    for key in [
        "irrep_available",
        "irrep_version",
        "spacegroup_irreps_available",
        "ebrs_available",
        "errors",
        "runtime_sources",
    ]:
        assert key in info
    assert isinstance(probe_irrep_is_importable(), bool)


def test_adapter_sources_do_not_import_irrep2_or_call_raw_decomposition():
    for path in [
        Path("valleyscope/analysis/irrep_data_normalizer.py"),
        Path("valleyscope/analysis/irrep_availability_probe.py"),
    ]:
        src = path.read_text(encoding="utf-8")
        for forbidden in [
            "import irrep2",
            "from irrep2",
            "compute_ebr_decomposition(",
        ]:
            assert forbidden not in src


def test_adapter_sources_have_no_material_names():
    for path in [
        Path("valleyscope/analysis/irrep_data_normalizer.py"),
        Path("valleyscope/analysis/irrep_availability_probe.py"),
    ]:
        src = path.read_text(encoding="utf-8")
        for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
            assert name not in src
