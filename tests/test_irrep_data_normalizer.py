"""Tests for irrep ebr_data normalizer -> reducer chain and availability probe."""

import json
import pytest
from pathlib import Path

from valleyscope.analysis.irrep_data_normalizer import (
    normalize_irrep_ebr_data_to_source_payload,
)
from valleyscope.analysis.irrep_runtime_reducer import (
    build_reduced_table_from_runtime_source,
)
from valleyscope.analysis.irrep_availability_probe import (
    probe_irrep_availability,
    probe_irrep_is_importable,
)
from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table

# ---- Sample irrep package-style ebr_data ----
_SAMPLE_EBR_DATA = {
    "basis": {
        "irrep_labels": ["GM:K5", "K:K6", "K:K6'", "A:C1"],
        "degeneracies": [1, 1, 1, 1],
    },
    "ebrs": [
        {"ebr_name": "EBR_A", "wyckoff_position": "1a", "vector": [1, 0, 1, 1]},
        {"ebr_name": "EBR_B", "wyckoff_position": "1a", "vector": [1, 1, 0, 0]},
    ],
}

_HSP_MAP = {"GM": "GammaM", "K": "KM", "A": "A"}
_IRREP_KEY_MAP = {
    "GM:K5": "GammaM:C3_spinor_phase_+1/2",
    "K:K6": "KM:C3_spinor_phase_+1/6",
    "K:K6'": "KM:C3_spinor_phase_-1/6",
    "A:C1": "A:C1_spinor",
}
_HSP = ["GammaM", "KM"]
_KEYS = ["GammaM:C3_spinor_phase_+1/2", "KM:C3_spinor_phase_+1/6", "KM:C3_spinor_phase_-1/6"]


# -----------------------------------------------------------------------
# 1. Normalizer -> reducer chain
# -----------------------------------------------------------------------

def test_full_chain_3d_ebr_data_to_external_table(tmp_path):
    """irrep-style ebr_data -> normalizer -> reducer -> load_reduced_ebr_table."""
    payload = normalize_irrep_ebr_data_to_source_payload(
        _SAMPLE_EBR_DATA, hsp_name_map=_HSP_MAP, irrep_key_map=_IRREP_KEY_MAP)
    assert len(payload["basis"]) == 4
    assert payload["ebrs"][0]["label"] == "EBR_A"

    table = build_reduced_table_from_runtime_source(
        source_payload=payload,
        expected_hsps=_HSP,
        allowed_irrep_keys=_KEYS,
        subspace_group_candidate="C3_like",
    )
    assert table["irreps"] == _KEYS
    assert table["subspace_group_candidate"] == "C3_like"

    path = tmp_path / "table.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    loaded = load_reduced_ebr_table(path)
    assert loaded["subspace_group_candidate"] == "C3_like"


def test_normalizer_preserves_wyckoff_positions():
    """Wyckoff positions are preserved through normalization."""
    payload = normalize_irrep_ebr_data_to_source_payload(
        _SAMPLE_EBR_DATA, hsp_name_map=_HSP_MAP, irrep_key_map=_IRREP_KEY_MAP)
    assert payload["ebrs"][0].get("wyckoff_position") == "1a"


def test_normalizer_no_hsp_map_uses_raw_labels():
    """Without hsp_name_map, HSP labels are used as-is."""
    payload = normalize_irrep_ebr_data_to_source_payload(_SAMPLE_EBR_DATA)
    assert payload["basis"][0]["hsp"] == "GM"
    assert payload["basis"][0]["valleyscope_irrep_key"] == "GM:K5"


def test_normalizer_missing_irrep_key_map_raises():
    """Missing irrep_key_map entry raises ValueError."""
    incomplete = {"GM:K5": "GammaM:C3_spinor_phase_+1/2"}  # missing K:K6
    with pytest.raises(ValueError, match="no irrep_key_map entry"):
        normalize_irrep_ebr_data_to_source_payload(
            _SAMPLE_EBR_DATA, hsp_name_map=_HSP_MAP, irrep_key_map=incomplete)


def test_normalizer_no_colon_in_label_raises():
    """irrep labels without ':' separator raise ValueError."""
    bad = dict(_SAMPLE_EBR_DATA)
    bad["basis"] = {"irrep_labels": ["K5"], "degeneracies": [1]}
    with pytest.raises(ValueError, match=":"):
        normalize_irrep_ebr_data_to_source_payload(bad)


def test_normalizer_degeneracies_length_mismatch_raises():
    """Mismatched irreps/degeneracies lengths raise ValueError."""
    bad = dict(_SAMPLE_EBR_DATA)
    bad["basis"] = {"irrep_labels": ["GM:K5"], "degeneracies": []}
    with pytest.raises(ValueError, match="degeneracies"):
        normalize_irrep_ebr_data_to_source_payload(bad)


# -----------------------------------------------------------------------
# 2. Availability probe
# -----------------------------------------------------------------------

def test_probe_returns_dict():
    """probe_irrep_availability returns a dict with expected keys."""
    info = probe_irrep_availability()
    for key in ["irrep_available", "irrep_version", "spacegroup_irreps_available",
                "ebrs_available", "errors"]:
        assert key in info, f"missing key '{key}'"


def test_probe_is_importable_returns_bool():
    """probe_irrep_is_importable returns a bool."""
    assert isinstance(probe_irrep_is_importable(), bool)


def test_probe_does_not_import_irrep2():
    """Probe must not import irrep2."""
    src = Path("valleyscope/analysis/irrep_availability_probe.py").read_text(encoding="utf-8")
    for forbidden in ["import irrep2", "from irrep2"]:
        assert forbidden not in src, "probe must not import irrep2"


def test_normalizer_does_not_import_irrep2():
    """Normalizer must not import irrep2."""
    src = Path("valleyscope/analysis/irrep_data_normalizer.py").read_text(encoding="utf-8")
    for forbidden in ["import irrep2", "from irrep2"]:
        assert forbidden not in src, "normalizer must not import irrep2"


# -----------------------------------------------------------------------
# 3. No material names
# -----------------------------------------------------------------------

def test_normalizer_no_material_names():
    src = Path("valleyscope/analysis/irrep_data_normalizer.py").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src, f"normalizer must not contain {name!r}"


def test_probe_no_material_names():
    src = Path("valleyscope/analysis/irrep_availability_probe.py").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src, f"probe must not contain {name!r}"
