import json
import pytest
from pathlib import Path
import yaml

from valleyscope.analysis.reduced_ebr_mapping import (
    load_reduced_ebr_table,
    build_reduced_ebr_mapping,
)
from valleyscope.io.config import load_config


def _write_table(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


_SAMPLE_TABLE = {
    "schema_version": "1.0.0",
    "subspace_group_candidate": "C3_like",
    "expected_hsps": ["GammaM", "KM"],
    "irreps": [
        "GammaM:C3_spinor_phase_+1/2",
        "KM:C3_spinor_phase_+1/6",
        "KM:C3_spinor_phase_-1/6",
    ],
    "ebrs": [
        {"label": "EBR_A", "vector": [1, 0, 1]},
        {"label": "EBR_B", "vector": [1, 1, 0]},
    ],
}


def _bundle():
    return {
        "bundles": [{
            "bundle_id": "bundle_ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {
                "GammaM": ["C3_spinor_phase_+1/2"],
                "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
            },
        }],
    }


# -----------------------------------------------------------------------
# 1. Table loading
# -----------------------------------------------------------------------

def test_load_valid_table(tmp_path):
    _write_table(tmp_path / "t.json", _SAMPLE_TABLE)
    t = load_reduced_ebr_table(tmp_path / "t.json")
    assert t["subspace_group_candidate"] == "C3_like"


def test_load_missing_keys_raises(tmp_path):
    _write_table(tmp_path / "t.json", {"irreps": [], "ebrs": []})
    with pytest.raises(ValueError, match="missing keys"):
        load_reduced_ebr_table(tmp_path / "t.json")


def test_load_mismatched_vector_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    bad["ebrs"] = [{"label": "X", "vector": [1, 2]}]
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="vector length"):
        load_reduced_ebr_table(tmp_path / "t.json")


def test_load_negative_vector_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    bad["ebrs"] = [{"label": "X", "vector": [1, -1, 0]}]
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="nonnegative"):
        load_reduced_ebr_table(tmp_path / "t.json")


def test_load_duplicate_irreps_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    bad["irreps"] = [
        "GammaM:C3_spinor_phase_+1/2",
        "GammaM:C3_spinor_phase_+1/2",
    ]
    bad["ebrs"] = [{"label": "X", "vector": [1, 0]}]
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="unique"):
        load_reduced_ebr_table(tmp_path / "t.json")


# -----------------------------------------------------------------------
# 2. Exact solution
# -----------------------------------------------------------------------

def test_exact_solution_found():
    # Bundle: 1x GammaM:+1/2, 3x KM:+1/6, 1x KM:-1/6
    # EBR_A = [1,0,1], EBR_B = [1,1,0]
    # 1*A + 2*B = [3, 2, 1] ≠ [1, 3, 1]. Not exact match.
    # Actually let me construct an exact match:
    # Bundle: GammaM:+1/2=2, KM:+1/6=1, KM:-1/6=1
    # 1*A + 1*B = [2, 1, 1] ✓
    bundle_vec = {"GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
                  "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"]}
    b = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": bundle_vec,
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
    assert r["status"] == "solved_exact"
    s = r["solutions"][0]
    assert s["status"] == "solved_exact"
    labels = {e["label"] for e in s["ebr_decomposition"]}
    assert labels == {"EBR_A", "EBR_B"}


# -----------------------------------------------------------------------
# 3. No exact solution
# -----------------------------------------------------------------------

def test_no_exact_solution():
    bundle_vec = {"GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
                  "KM": []}
    b = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": bundle_vec,
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
    assert r["status"] == "no_exact_solution"


# -----------------------------------------------------------------------
# 4. Missing table
# -----------------------------------------------------------------------

def test_missing_table():
    r = build_reduced_ebr_mapping(ebr_export_bundle=_bundle(), table=None)
    assert r["status"] == "missing_table"
    assert r["table_status"] == "not_provided"


# -----------------------------------------------------------------------
# 5. Null input
# -----------------------------------------------------------------------

def test_null_bundle():
    r = build_reduced_ebr_mapping(ebr_export_bundle=None)
    assert r["status"] == "not_evaluated"
    assert r["mapping_status"] == "not_evaluated"
    assert r["solutions"] == []


# -----------------------------------------------------------------------
# 6. Mismatched group
# -----------------------------------------------------------------------

def test_group_mismatch_excluded():
    b = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "M1",
            "subspace_group_candidate": "C2_like",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {"GammaM": ["C2_spinor_phase_+1/4"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
    ex = r["excluded_bundles"]
    assert len(ex) == 1
    assert "table group" in ex[0]["reason"]


# -----------------------------------------------------------------------
# 7. Schema
# -----------------------------------------------------------------------

def test_schema_fields():
    r = build_reduced_ebr_mapping(ebr_export_bundle=_bundle(), table=_SAMPLE_TABLE)
    for k in ["status", "mapping_status", "reduced_ebr_decomposition_status", "table_status",
              "solutions", "excluded_bundles", "solver"]:
        assert k in r, f"missing: {k}"


def test_json_serializable():
    r = build_reduced_ebr_mapping(ebr_export_bundle=_bundle(), table=_SAMPLE_TABLE)
    encoded = json.dumps(r)
    assert len(encoded) > 0


def test_no_forbidden_terms():
    r = build_reduced_ebr_mapping(ebr_export_bundle=_bundle(), table=_SAMPLE_TABLE)
    encoded = json.dumps(r)
    for forbidden in ["covariance", "equivariance", "stabilizer",
                      "valley_little_group"]:
        assert forbidden not in encoded.lower()


# -----------------------------------------------------------------------
# 8. Not ready excluded
# -----------------------------------------------------------------------

def test_not_ready_excluded():
    b = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": False,
            "irreps_by_kpoint": {},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
    ex = r["excluded_bundles"]
    assert len(ex) == 1
    assert "not ready" in ex[0]["reason"]


def test_unknown_irrep_label_is_not_matched_by_hsp_only():
    table = dict(_SAMPLE_TABLE)
    table["irreps"] = ["GammaM:wrong_label"]
    table["ebrs"] = [{"label": "X", "vector": [1]}]
    b = {
        "bundles": [{
            "bundle_id": "b_001",
            "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=table)
    assert r["status"] == "not_evaluated"
    assert "could not resolve" in r["excluded_bundles"][0]["reason"]


def test_unique_operation_suffix_fallback_does_not_double_count():
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C3_like",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:C3_spinor_phase_+1/2:op1"],
        "ebrs": [{"label": "EBR_A", "vector": [1]}],
    }
    b = {
        "bundles": [{
            "bundle_id": "b_001",
            "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=table)
    assert r["status"] == "solved_exact"
    assert r["solutions"][0]["irrep_vector"] == [1]


def test_ambiguous_operation_suffix_fallback_is_excluded():
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C3_like",
        "expected_hsps": ["GammaM"],
        "irreps": [
            "GammaM:C3_spinor_phase_+1/2:op1",
            "GammaM:C3_spinor_phase_+1/2:op2",
        ],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0]}],
    }
    b = {
        "bundles": [{
            "bundle_id": "b_001",
            "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=table)
    assert r["status"] == "not_evaluated"
    assert "could not resolve" in r["excluded_bundles"][0]["reason"]


def test_negative_max_coefficient_raises():
    with pytest.raises(ValueError, match="max_coefficient"):
        build_reduced_ebr_mapping(
            ebr_export_bundle=_bundle(),
            table=_SAMPLE_TABLE,
            max_coefficient=-1,
        )


def test_reduced_ebr_table_file_resolves_relative_to_config(tmp_path):
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    table_path = table_dir / "p3.json"
    _write_table(table_path, _SAMPLE_TABLE)
    config = {
        "input": {"wavefunction_h5": "wave.h5"},
        "analysis": {
            "kpoints": ["GammaM"],
            "iband": [1],
            "reduced_ebr": {
                "enabled": True,
                "table_file": "tables/p3.json",
                "max_coefficient": 2,
            },
        },
        "monolayer_lattices": {
            "default": {
                "reciprocal_cart": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        },
        "valley_centers": {
            "coordinate_mode": "frac",
            "centers": [{"name": "K", "frac": [0.0, 0.0, 0.0]}],
        },
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": "out"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded.reduced_ebr.enabled is True
    assert loaded.reduced_ebr.max_coefficient == 2
    assert loaded.reduced_ebr.table_file == table_path
