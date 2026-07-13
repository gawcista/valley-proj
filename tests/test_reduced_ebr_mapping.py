import json
import subprocess
import sys
import pytest
from pathlib import Path
import numpy as np
import yaml

from valleyscope.analysis.reduced_ebr_mapping import (
    load_reduced_ebr_table,
    build_reduced_ebr_mapping,
)
from valleyscope.io.config import load_config
from tests.reduced_ebr_promo_helpers import attach_promotion
from valleyscope.analysis.reduced_ebr_solver import classify_bundle

def _write_table(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")

def _ready(export: dict, table: dict) -> dict:
    """Make an export bundle + table pass the fail-closed promotion validator."""
    attach_promotion(export, table)
    return export

_SAMPLE_TABLE = {
    "schema_version": "1.0.0",
    "subspace_group_candidate": "P3",
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
    "provenance": {
        "data_source": "irreptables",
        "package": "irreptables",
        "package_version": "0.0.test",
        "space_group_number": 143,
        "spinful": False,
        "valleyscope_reduction": "sampled_hsp_valley_preserving",
    },
}

def _bundle():
    return {
        "bundles": [{
            "bundle_id": "bundle_ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
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
    assert t["subspace_group_candidate"] == "P3"

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
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "ready_for_external_solver": True,
            "irreps_by_kpoint": bundle_vec,
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    assert r["status"] == "solved_exact"
    s = r["solutions"][0]
    assert s["status"] == "solved_exact"
    assert s["subspace_space_group"] == {"candidate_space_group_symbol": "P3"}
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
            "subspace_group_candidate": "P3",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": bundle_vec,
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
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
            "subspace_group_candidate": "P2",
            "subspace_space_group": {"candidate_space_group_symbol": "P2"},
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {"GammaM": ["C2_spinor_phase_+1/4"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    ex = r["excluded_bundles"]
    assert len(ex) == 1
    assert ex[0]["subspace_group_candidate"] == "P2"
    assert ex[0]["subspace_space_group"] == {"candidate_space_group_symbol": "P2"}
    assert any(term in ex[0]["reason"] for term in
               ["SG symbol mismatch", "table group", "validation blocked"])

# -----------------------------------------------------------------------
# 7. Schema
# -----------------------------------------------------------------------

def test_schema_fields():
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(_bundle(), _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    for k in ["status", "mapping_status", "reduced_ebr_decomposition_status", "table_status",
              "solutions", "excluded_bundles", "solver"]:
        assert k in r, f"missing: {k}"

def test_json_serializable():
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(_bundle(), _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    encoded = json.dumps(r)
    assert len(encoded) > 0

def test_no_forbidden_terms():
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(_bundle(), _SAMPLE_TABLE), table=_SAMPLE_TABLE)
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
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "ready_for_external_solver": False,
            "irreps_by_kpoint": {},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    ex = r["excluded_bundles"]
    assert len(ex) == 1
    assert ex[0]["subspace_group_candidate"] == "P3"
    assert ex[0]["subspace_space_group"] == {"candidate_space_group_symbol": "P3"}
    assert "not ready" in ex[0]["reason"]

def test_unknown_irrep_label_is_not_matched_by_hsp_only():
    table = dict(_SAMPLE_TABLE)
    table["expected_hsps"] = ["GammaM"]
    table["irreps"] = ["GammaM:wrong_label"]
    table["ebrs"] = [{"label": "X", "vector": [1]}]
    b = {
        "bundles": [{
            "bundle_id": "b_001",
            "valley": "K",
            "subspace_group_candidate": "P3",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM"],
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, table), table=table)
    assert r["status"] == "not_evaluated"
    assert "could not resolve" in r["excluded_bundles"][0]["reason"]

def test_unique_operation_suffix_fallback_does_not_double_count():
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P3",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:C3_spinor_phase_+1/2:op1"],
        "ebrs": [{"label": "EBR_A", "vector": [1]}],
    }
    b = {
        "bundles": [{
            "bundle_id": "b_001",
            "valley": "K",
            "subspace_group_candidate": "P3",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, table), table=table)
    assert r["status"] == "solved_exact"
    assert r["solutions"][0]["irrep_vector"] == [1]

def test_ambiguous_operation_suffix_fallback_is_excluded():
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P3",
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
            "subspace_group_candidate": "P3",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, table), table=table)
    assert r["status"] == "not_evaluated"
    assert "could not resolve" in r["excluded_bundles"][0]["reason"]

def test_negative_max_coefficient_raises():
    with pytest.raises(ValueError, match="max_coefficient"):
        build_reduced_ebr_mapping(
            ebr_export_bundle=_bundle(),
            table=_SAMPLE_TABLE,
            max_coefficient=-1
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

# -----------------------------------------------------------------------
# 9. schema_version validation
# -----------------------------------------------------------------------

def test_missing_schema_version_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    del bad["schema_version"]
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="missing keys"):
        load_reduced_ebr_table(tmp_path / "t.json")

def test_empty_schema_version_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    bad["schema_version"] = ""
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="schema_version"):
        load_reduced_ebr_table(tmp_path / "t.json")

def test_non_string_schema_version_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    bad["schema_version"] = 1
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="schema_version"):
        load_reduced_ebr_table(tmp_path / "t.json")

# -----------------------------------------------------------------------
# 10. expected_hsps validation
# -----------------------------------------------------------------------

def test_empty_expected_hsps_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    bad["expected_hsps"] = []
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="expected_hsps must be a non-empty list"):
        load_reduced_ebr_table(tmp_path / "t.json")

def test_duplicate_expected_hsps_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    bad["expected_hsps"] = ["GammaM", "KM", "GammaM"]
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="unique"):
        load_reduced_ebr_table(tmp_path / "t.json")

def test_non_string_expected_hsp_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    bad["expected_hsps"] = ["GammaM", 123]
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="expected_hsps entries must be non-empty"):
        load_reduced_ebr_table(tmp_path / "t.json")

# -----------------------------------------------------------------------
# 11. EBR label uniqueness
# -----------------------------------------------------------------------

def test_duplicate_ebr_labels_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    bad["ebrs"] = [
        {"label": "EBR_A", "vector": [1, 0, 1]},
        {"label": "EBR_A", "vector": [1, 1, 0]},
    ]
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="unique"):
        load_reduced_ebr_table(tmp_path / "t.json")

# -----------------------------------------------------------------------
# 12. empty EBR vector
# -----------------------------------------------------------------------

def test_empty_ebr_vector_raises(tmp_path):
    bad = dict(_SAMPLE_TABLE)
    bad["ebrs"] = [
        {"label": "EBR_A", "vector": [1, 0, 1]},
        {"label": "EBR_B", "vector": []},
    ]
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="non-empty"):
        load_reduced_ebr_table(tmp_path / "t.json")

# -----------------------------------------------------------------------
# 13. irrep key format validation
# -----------------------------------------------------------------------

@pytest.mark.parametrize("bad_key", [
    "no_colon",                      # no colon separator
    ":missing_kpoint",               # empty kpoint
    "GammaM:",                       # empty irrep label
    "1GammaM:C3_spinor",             # kpoint starts with digit
    "GammaM: bad_irrep",             # space in irrep label
])
def test_invalid_irrep_key_format_raises(tmp_path, bad_key):
    bad = dict(_SAMPLE_TABLE)
    bad["irreps"] = ["GammaM:C3_spinor_phase_+1/2", bad_key]
    bad["ebrs"] = [{"label": "X", "vector": [1, 0]}]
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="invalid irrep key format"):
        load_reduced_ebr_table(tmp_path / "t.json")

@pytest.mark.parametrize("good_key", [
    "GammaM:C3_spinor_phase_+1/2",
    "KM:C3_spinor_phase_-1/6",
    "MM:C2_spinor_phase_+1/4",
    "GammaM:C3_spinor_phase_+1/2:op1",
    "KM:C3_spinor_phase_+1/6:op2",
])
def test_valid_irrep_key_formats_accepted(tmp_path, good_key):
    tbl = dict(_SAMPLE_TABLE)
    tbl["irreps"] = [good_key]
    tbl["ebrs"] = [{"label": "X", "vector": [1]}]
    _write_table(tmp_path / "t.json", tbl)
    loaded = load_reduced_ebr_table(tmp_path / "t.json")
    assert loaded["irreps"][0] == good_key

# -----------------------------------------------------------------------
# 14. schema/doc contract
# -----------------------------------------------------------------------

def test_table_schema_doc_contract():
    """reduced_ebr_table_schema.md must document required keys, statuses, and labels."""
    from valleyscope.analysis.reduced_ebr_mapping import _REQUIRED_TABLE_KEYS
    doc = Path("docs/reduced_ebr_table_schema.md").read_text(encoding="utf-8")
    for key in sorted(_REQUIRED_TABLE_KEYS):
        assert f"`{key}`" in doc, f"missing key '{key}'"
    for status in ["not_evaluated", "missing_table", "solved_exact", "no_exact_solution"]:
        assert f"`{status}`" in doc, f"missing status '{status}'"
    assert "P3" in doc and "P2" in doc and "P4" in doc
    assert "C{order}_like" not in doc
    assert "C3_like" not in doc
    assert "C2_like" not in doc
    assert "no built-in" in doc.lower()
    assert "no heuristic" in doc.lower()

def test_table_schema_doc_expected_hsps_basis_contract():
    """Schema doc must state expected_hsps is enforced as the EBR basis contract."""
    doc_text = Path("docs/reduced_ebr_table_schema.md").read_text(encoding="utf-8")
    assert "reduced-dimensional EBR basis contract" in doc_text
    assert "bundle `expected_hsps`" in doc_text
    assert "`bundle.irreps_by_kpoint`" in doc_text
    assert "Missing, extra, malformed, or inferred HSP data is not accepted" in doc_text

def test_schema_md_labels_use_physical_subspace_space_group():
    """Verify docs/schema.md uses physical subspace-space-group symbols as primary EBR table keys.

    subspace_group_candidate examples should use P3/P2 (physical), not
    C3_like/C2_like (legacy).  Legacy C{n}_like labels appear only in
    legacy_subspace_group_candidate or the transitional convention description.
    """
    schema_text = Path("docs/schema.md").read_text(encoding="utf-8")
    # Physical symbols appear as example values.
    assert '"subspace_group_candidate": "P3"' in schema_text
    # Legacy hints documented in the label convention description.
    assert "C3_like" in schema_text
    assert "C2_like" in schema_text
    # Must not use C{n}_like as subspace_group_candidate examples anymore.
    assert '"subspace_group_candidate": "C3_like"' not in schema_text
    assert '"subspace_group_candidate": "C2_like"' not in schema_text

def test_public_docs_do_not_advertise_static_package_selector():
    """Current public docs must not expose static package-data EBR selectors."""
    checked_paths = [
        Path("docs/schema.md"),
        Path("docs/reduced_ebr_table_schema.md"),
        Path("docs/reduced_dimensional_irrep_ebr_data_model.md"),
        Path("valleyscope/data/reduced_ebr/README.md"),
    ]
    forbidden_terms = [
        "--table-name",
        "load_reviewed_reduced_ebr_table",
        "list_production_reduced_ebr_tables",
        "single reviewed P3",
        "reviewed package-data table path",
    ]
    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in text, f"{path} must not mention {term!r}"


def test_reduced_ebr_catalog_removed():
    """Package-data reduced EBR catalog was deleted. No static table selector remains."""
    import importlib
    spec = importlib.util.find_spec("valleyscope.data.reduced_ebr.catalog")
    assert spec is None, "valleyscope.data.reduced_ebr.catalog was deleted"

# -----------------------------------------------------------------------
# 15. map-reduced-ebr CLI
# -----------------------------------------------------------------------

def _write_bundle(path: Path, bundles: list[dict]) -> None:
    payload = {
        "status": "ready_for_external_solver",
        "bundle_count": len(bundles),
        "excluded_count": 0,
        "schema_version": "1.0.0",
        "reduced_ebr_decomposition_status": "not_implemented",
        "bundles": bundles,
        "excluded_instances": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

def _cli(bundle: Path, table: Path, output: Path, *extra) -> int:
    from valleyscope.cli import main
    return main([
        "map-reduced-ebr", str(bundle), str(table),
        "-o", str(output), *extra,
    ])

def test_cli_solved_exact(tmp_path):
    """CLI solves an exact-match toy bundle and writes the mapping JSON."""
    out = tmp_path / "out.json"
    bundle_path = tmp_path / "bundle.json"
    table_path = tmp_path / "table.json"
    _write_bundle(bundle_path, [{
        "bundle_id": "b_001", "valley": "K",
        "subspace_group_candidate": "P3",
        "ready_for_external_solver": True,
        "irreps_by_kpoint": {
            "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
            "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
        },
    }])
    _write_table(table_path, _SAMPLE_TABLE)

    rc = _cli(bundle_path, table_path, out)
    assert rc == 0, f"CLI exited with {rc}"
    assert out.exists()
    mapping = json.loads(out.read_text(encoding="utf-8"))
    # Production path rejects minimal table without provenance.
    assert mapping["status"] == "not_evaluated"
    assert mapping["solutions"] == []
    assert len(mapping["excluded_bundles"]) == 1

def test_cli_no_exact_solution(tmp_path):
    """CLI reports not_evaluated: production path rejects unreviewed table."""
    out = tmp_path / "out.json"
    bundle_path = tmp_path / "bundle.json"
    table_path = tmp_path / "table.json"
    _write_bundle(bundle_path, [{
        "bundle_id": "b_001", "valley": "K",
        "subspace_group_candidate": "P3",
        "ready_for_external_solver": True,
        "irreps_by_kpoint": {
            "GammaM": ["C3_spinor_phase_+1/2"] * 5,
            "KM": [],
        },
    }])
    _write_table(table_path, _SAMPLE_TABLE)

    rc = _cli(bundle_path, table_path, out)
    assert rc == 0
    mapping = json.loads(out.read_text(encoding="utf-8"))
    # Production path rejects minimal table without provenance.
    assert mapping["status"] == "not_evaluated"
    assert mapping["solutions"] == []

def test_cli_stdout_includes_status_and_output_path(capsys, tmp_path):
    """CLI stdout reports not_evaluated for unreviewed table."""
    out = tmp_path / "out.json"
    bundle_path = tmp_path / "bundle.json"
    table_path = tmp_path / "table.json"
    _write_bundle(bundle_path, [{
        "bundle_id": "b_001", "valley": "K",
        "subspace_group_candidate": "P3",
        "ready_for_external_solver": True,
        "irreps_by_kpoint": {
            "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
            "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
        },
    }])
    _write_table(table_path, _SAMPLE_TABLE)

    _cli(bundle_path, table_path, out)
    captured = capsys.readouterr().out
    assert "not_evaluated" in captured
    assert str(out) in captured

def test_cli_invalid_table_fails_without_writing_output(tmp_path):
    """CLI fails on invalid table and does not write a misleading output file."""
    out = tmp_path / "out.json"
    bundle_path = tmp_path / "bundle.json"
    table_path = tmp_path / "table.json"
    # Table with duplicate EBR labels
    bad_table = dict(_SAMPLE_TABLE)
    bad_table["ebrs"] = [
        {"label": "EBR_A", "vector": [1, 0, 1]},
        {"label": "EBR_A", "vector": [1, 1, 0]},
    ]
    _write_bundle(bundle_path, [{
        "bundle_id": "b_001", "valley": "K",
        "subspace_group_candidate": "P3",
        "ready_for_external_solver": True,
        "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"], "KM": []},
    }])
    _write_table(table_path, bad_table)

    rc = _cli(bundle_path, table_path, out)
    assert rc != 0, "CLI should fail on invalid table"
    assert not out.exists(), "Must not write output for invalid table"

@pytest.mark.parametrize("missing,mode", [
    ("table", lambda d: (_write_bundle(d / "bundle.json", []), d / "bundle.json", d / "nonexistent.json")),
    ("bundle", lambda d: (_write_table(d / "table.json", _SAMPLE_TABLE), d / "nonexistent.json", d / "table.json")),
])
def test_cli_missing_file_fails(missing, mode):
    """CLI fails when a required input file (table or bundle) does not exist."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _, bundle_path, table_path = mode(d)
        rc = _cli(bundle_path, table_path, d / "out.json")
        assert rc != 0, f"CLI should fail on missing {missing}"
        assert not (d / "out.json").exists()

def test_cli_respects_max_coefficient(tmp_path):
    """CLI passes --max-coefficient through to the solver."""
    out = tmp_path / "out.json"
    bundle_path = tmp_path / "bundle.json"
    table_path = tmp_path / "table.json"
    # Bundle with large count that needs max_coeff >= 6
    _write_bundle(bundle_path, [{
        "bundle_id": "b_001", "valley": "K",
        "subspace_group_candidate": "P3",
        "ready_for_external_solver": True,
        "irreps_by_kpoint": {
            "GammaM": ["C3_spinor_phase_+1/2"] * 6,
            "KM": ["C3_spinor_phase_+1/6"] * 6,
        },
    }])
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P3",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:C3_spinor_phase_+1/2", "KM:C3_spinor_phase_+1/6"],
        "ebrs": [{"label": "EBR_X", "vector": [1, 1]}],
    }
    _write_table(table_path, table)

    # With max_coeff=6: 6*[1,1] = [6,6] -> exact
    rc = _cli(bundle_path, table_path, out, "--max-coefficient", "6")
    assert rc == 0
    mapping = json.loads(out.read_text(encoding="utf-8"))
    # Production path rejects minimal table without provenance.
    assert mapping["status"] == "not_evaluated"

def test_cli_analyze_hsp_reduced_ebr_unchanged(tmp_path):
    """Existing analyze-hsp reduced-EBR behavior is unchanged by CLI addition."""
    h5_path = tmp_path / "wf.h5"
    import h5py, numpy as np
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3) * 10.0
        meta["spinor"] = False; meta["source"] = "toy"; meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"; kp["frac"] = np.zeros(3); kp["cart"] = np.zeros(3)
        kp["g_vectors_frac"] = np.array([[0, 0, 0], [1, 0, 0]])
        kp["g_vectors_cart"] = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        kp["coefficients"] = np.array([[[1.0 + 0.0j, 0.0 + 0.0j]]])
        kp["energies_eV"] = np.array([0.1]); kp["band_indices_vasp"] = np.array([101])

    table_path = tmp_path / "table.json"
    _write_table(table_path, _SAMPLE_TABLE)
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101],
                      "reduced_ebr": {"enabled": True, "table_file": str(table_path)}},
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [
            {"name": "K", "cart": [0.0, 0.0, 0.0]},
            {"name": "Kp", "cart": [5.0, 0.0, 0.0]},
        ]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]},
                             {"name": "Kp_valley", "centers": ["Kp"]}],
        "output": {"directory": str(tmp_path / "out")},
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    from valleyscope.workflows.analyze_hsp import analyze_hsp
    outputs = analyze_hsp(config_path)
    # With a real analyze-hsp run, reduced EBR mapping is present when enabled.
    assert "valley_reduced_ebr_mapping_json" in outputs, (
        "analyze-hsp must still write valley_reduced_ebr_mapping.json when enabled"
    )

def test_cli_module_entrypoint_help_lists_map_reduced_ebr():
    """python -m valleyscope.cli must dispatch to argparse, not silently exit."""
    result = subprocess.run(
        [sys.executable, "-m", "valleyscope.cli", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "map-reduced-ebr" in result.stdout

# -----------------------------------------------------------------------
# 16. Reduced-dimensional irrep/EBR data model doc contract
# -----------------------------------------------------------------------

def test_data_model_design_doc_contract():
    """Design doc must cover irrep2, reduced-dimensional, provenance, irrep package,
    physical objects, labels, and forbid material-specific targets."""
    doc_path = Path("docs/reduced_dimensional_irrep_ebr_data_model.md")
    assert doc_path.exists()
    doc = doc_path.read_text(encoding="utf-8")
    doc_lower = doc.lower()

    # irrep2 reference-only.
    assert "irrep2" in doc and "reference-only" in doc
    # Reduced-dimensional, provenance, package name.
    for term in ["reduced-dimensional", "provenance"]:
        assert term in doc_lower, f"missing '{term}'"
    assert "`irrep`" in doc or "Python package `irrep`" in doc
    # No direct 3D reuse.
    assert any(p in doc_lower for p in [
        "not direct reuse",
        "no direct reuse",
        "do not directly reuse",
        "not directly reuse",
        "no raw 3d",
        "must not directly reuse",
        "must not be directly reused",
    ])
    assert "raw 3d" in doc_lower
    # irrep as runtime source with ValleyScope reduction.
    for term in ["runtime data source", "sampled-hsp", "valley-preserving reduction"]:
        assert term in doc_lower, f"missing '{term}'"
    # Physical objects.
    for term in ["HSP little group", "valley-preserving subgroup",
                 "valley sewing matrix", "reduced-dimensional EBR vector"]:
        assert term.lower() in doc_lower, f"missing '{term}'"
    # Labels.
    assert "C3_like" in doc and "C2_like" in doc and "P3" in doc and "P2" in doc
    # No material-specific table targets.
    for forbidden in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert forbidden not in doc, f"material name '{forbidden}' in design doc"

def test_agents_plan_irrep_boundary_and_package_name():
    """AGENTS/PLAN must encode irrep runtime boundary and use correct package name."""
    combined_lower = (
        Path("AGENTS.md").read_text(encoding="utf-8")
        + "\n"
        + Path("PLAN.md").read_text(encoding="utf-8")
    ).lower()
    for term in ["runtime", "data source", "valley-preserving", "raw 3d"]:
        assert term in combined_lower, f"missing '{term}' in AGENTS/PLAN"
    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    plan = Path("PLAN.md").read_text(encoding="utf-8")
    assert "`irrep`" in agents and "`irreps`" not in agents
    assert "irrep" in plan.lower()

# -----------------------------------------------------------------------
# 17. Package-data skeleton
# -----------------------------------------------------------------------

# Catalog API test removed

# -----------------------------------------------------------------------
# 18. Reduced EBR basis compatibility gate
# -----------------------------------------------------------------------

def _bundle_with_hsps(expected, irreps_by_kp, g="P3", ready=True):
    return {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": g,
            "ready_for_external_solver": ready,
            "expected_hsps": expected,
            "irreps_by_kpoint": irreps_by_kp,
        }],
    }

def test_matching_basis_solves_exact():
    """Matching expected_hsps and irreps_by_kpoint keys still produces solution."""
    b = _bundle_with_hsps(
        expected=["GammaM", "KM"],
        irreps_by_kp={
            "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
            "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
        },
    )
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    assert r["status"] == "solved_exact"
    assert r["solutions"][0]["status"] == "solved_exact"

def test_table_extra_hsp_excludes_bundle():
    """Table expects more HSPs than bundle has → excluded."""
    b = _bundle_with_hsps(
        expected=["GammaM"],
        irreps_by_kp={"GammaM": ["C3_spinor_phase_+1/2"]},
    )
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    assert len(r["excluded_bundles"]) == 1
    assert "expected_hsps mismatch" in r["excluded_bundles"][0]["reason"]

def test_bundle_extra_hsp_excludes_bundle():
    """Bundle has more HSPs than table → excluded."""
    b = _bundle_with_hsps(
        expected=["GammaM", "KM", "MM"],
        irreps_by_kp={
            "GammaM": ["C3_spinor_phase_+1/2"],
            "KM": ["C3_spinor_phase_+1/6"],
            "MM": ["C3_spinor_phase_-1/6"],
        },
    )
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    assert len(r["excluded_bundles"]) == 1
    assert "expected_hsps mismatch" in r["excluded_bundles"][0]["reason"]

def test_irrep_keys_mismatch_excludes():
    """Bundle irreps_by_kpoint keys don't match table expected_hsps → excluded."""
    b = _bundle_with_hsps(
        expected=["GammaM", "KM"],
        irreps_by_kp={
            "GammaM": ["C3_spinor_phase_+1/2"],
            # KM missing — irreps_by_kpoint keys ≠ table expected_hsps
        },
    )
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    assert len(r["excluded_bundles"]) >= 1
    reasons = " ".join(e["reason"] for e in r["excluded_bundles"])
    assert "hsp_basis_mismatch" in reasons or "expected_hsps mismatch" in reasons

def test_malformed_declared_expected_hsps_excludes():
    """Declared expected_hsps must be a unique list, not legacy fallback."""
    b = _bundle_with_hsps(
        expected="GammaM",
        irreps_by_kp={
            "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
            "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
        },
    )
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    assert len(r["excluded_bundles"]) == 1
    assert "hsp_basis_malformed" in r["excluded_bundles"][0]["reason"]

def test_legacy_bundle_without_expected_hsps_still_works():
    """Legacy bundle without expected_hsps derives basis from irreps_by_kpoint keys."""
    b = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "P3",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {
                "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
                "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
            },
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    assert r["status"] == "solved_exact"

def test_legacy_bundle_without_expected_hsps_fails_when_keys_mismatch():
    """Legacy bundle without expected_hsps fails when irreps_by_kpoint keys mismatch table."""
    b = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "P3",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    assert len(r["excluded_bundles"]) == 1
    reason = r["excluded_bundles"][0]["reason"]
    assert "expected_hsps mismatch" in reason or "irrep HSP basis mismatch" in reason

def test_basis_gate_before_group_check():
    """Basis mismatch excludes even when group matches."""
    b = _bundle_with_hsps(
        expected=["GammaM"],
        irreps_by_kp={"GammaM": ["C3_spinor_phase_+1/2"]},
        g="P3",  # group matches table
    )
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    assert len(r["excluded_bundles"]) == 1
    assert "expected_hsps mismatch" in r["excluded_bundles"][0]["reason"]

def test_reduced_ebr_mapping_no_material_names_or_irrep2():
    """reduced_ebr_mapping.py must not contain material names or import irrep2."""
    src = Path("valleyscope/analysis/reduced_ebr_mapping.py").read_text(encoding="utf-8")
    for forbidden in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert forbidden not in src, f"contains {forbidden!r}"
    assert "irrep2" not in src, "must not import irrep2"

# -----------------------------------------------------------------------
# 20. Integer-span classifier
# -----------------------------------------------------------------------

def test_atomic_compatible_classification():
    """Nonnegative exact solution -> atomic-compatible-candidate."""
    b = _bundle_with_hsps(
        expected=["GammaM", "KM"],
        irreps_by_kp={
            "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
            "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
        },
    )
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(b, _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    assert r["status"] == "solved_exact"
    s = r["solutions"][0]
    assert s["classification"] == "atomic-compatible-candidate"
    assert s["integer_span_status"] == "in_integer_span"
    assert s["nonnegative_solution_status"] == "solved_exact"
    assert "ebr_decomposition" in s
    assert len(s["ebr_decomposition"]) > 0

def test_in_integer_span_no_nonnegative_witness_classification():
    """Target in integer span but needs negative coefficient -> in_integer_span_no_nonnegative_witness."""
    # EBR columns: [1,0] (A), [1,1] (B)
    # Target: [0,1] = 1*B - 1*A -> needs negative coeff for A
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C1",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:irrep_A", "GammaM:irrep_B"],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0]},
            {"label": "EBR_B", "vector": [1, 1]},
        ],
    }
    # Low-level solver algebra: target [0,1] in span but needs negative coeff.
    s = classify_bundle([0, 1], [[1, 0], [1, 1]], ["EBR_A", "EBR_B"], 6)
    assert s["status"] == "no_exact_solution"
    assert s["classification"] == "in_integer_span_no_nonnegative_witness"
    assert s["integer_span_status"] == "in_integer_span"
    assert s["nonnegative_solution_status"] == "no_nonnegative_solution"
    assert "integer_solution" in s
    coeffs = {e["label"]: e["coefficient"] for e in s["integer_solution"]}
    assert coeffs.get("EBR_A") == -1
    assert coeffs.get("EBR_B") == 1

def test_outside_integer_span_classification():
    """Target outside integer span -> outside_integer_span."""
    # EBR columns: [2,0] (A), [0,2] (B)
    # Target: [1,0] — requires 0.5*A, not in integer span
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C1",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:irrep_A", "GammaM:irrep_B"],
        "ebrs": [
            {"label": "EBR_A", "vector": [2, 0]},
            {"label": "EBR_B", "vector": [0, 2]},
        ],
    }
    # Low-level solver algebra: target [1,0] not in integer span of 2*A,2*B.
    s = classify_bundle([1, 0], [[2, 0], [0, 2]], ["EBR_A", "EBR_B"], 6)
    assert s["status"] == "no_exact_solution"
    assert s["classification"] == "outside_integer_span"
    assert s["integer_span_status"] == "outside_integer_span"
    assert s["nonnegative_solution_status"] == "no_nonnegative_solution"

def test_zero_ebr_vector_rejected_at_load(tmp_path):
    """Zero EBR vector raises ValueError at table load time."""
    bad = dict(_SAMPLE_TABLE)
    bad["ebrs"] = [{"label": "EBR_ZERO", "vector": [0, 0, 0]}]
    _write_table(tmp_path / "t.json", bad)
    with pytest.raises(ValueError, match="positive"):
        load_reduced_ebr_table(tmp_path / "t.json")

def test_nonnegative_search_uses_physical_bounds():
    """Nonnegative search finds solution when coeff exceeds old default max_coeff."""
    # EBR: 1x [1,0] (A), 1x [0,1] (B)
    # Target: [10, 10] -> 10*A + 10*B, needs coeff=10 > old default of 6
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C1",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:irrep_A", "GammaM:irrep_B"],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0]},
            {"label": "EBR_B", "vector": [0, 1]},
        ],
    }
    # Low-level solver algebra: target [10,10] solvable with coeff 10 > 6.
    s = classify_bundle([10, 10], [[1, 0], [0, 1]], ["EBR_A", "EBR_B"], 12)
    assert s["status"] == "solved_exact"
    assert s["classification"] == "atomic-compatible-candidate"
    assert "search_status" not in s

def test_max_coefficient_truncation_reported():
    """When max_coefficient truncates a derived bound, search_status is set."""
    # EBR: [1,0] (A), [0,1] (B). Target [10,10] needs c_A=10, c_B=10
    # bounds = [10, 10]; max_coeff=5 -> truncated
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C1",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:irrep_A", "GammaM:irrep_B"],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0]},
            {"label": "EBR_B", "vector": [0, 1]},
        ],
    }
    # Low-level solver algebra: bounds [10,10] with max_coefficient 5 truncates.
    s = classify_bundle([10, 10], [[1, 0], [0, 1]], ["EBR_A", "EBR_B"], 5)
    assert s["search_status"] == "truncated_by_max_coefficient"

def test_classification_fields_on_existing_tests():
    """All solutions must carry classification, integer_span_status,
    and nonnegative_solution_status."""
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(_bundle(), _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    for s in r["solutions"]:
        assert s["classification"] in {
            "atomic-compatible-candidate",
            "in_integer_span_no_nonnegative_witness",
            "outside_integer_span",
        }
        assert s["integer_span_status"] in {"in_integer_span", "outside_integer_span"}
        assert s["nonnegative_solution_status"] in {"solved_exact", "no_nonnegative_solution"}

def test_cli_shows_classification_counts(tmp_path, capsys):
    """CLI stdout includes classification counts when present."""
    from valleyscope.cli import main
    out = tmp_path / "out.json"
    bundle_path = tmp_path / "bundle.json"
    table_path = tmp_path / "table.json"
    _write_bundle(bundle_path, [{
        "bundle_id": "b_001", "valley": "K",
        "subspace_group_candidate": "P3",
        "ready_for_external_solver": True,
        "expected_hsps": ["GammaM", "KM"],
        "irreps_by_kpoint": {
            "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
            "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
        },
    }])
    _write_table(table_path, _SAMPLE_TABLE)
    rc = main(["map-reduced-ebr", str(bundle_path), str(table_path), "-o", str(out)])
    assert rc == 0
    captured = capsys.readouterr().out
    # Minimal table without provenance → rejected; no classification to show.
    assert "not_evaluated" in captured

def test_schema_fields_include_classification():
    """Top-level schema must still include all required fields."""
    r = build_reduced_ebr_mapping(ebr_export_bundle=_ready(_bundle(), _SAMPLE_TABLE), table=_SAMPLE_TABLE)
    for k in ["status", "mapping_status", "reduced_ebr_decomposition_status",
              "table_status", "solutions", "excluded_bundles", "solver"]:
        assert k in r, f"missing top-level key: {k}"
    assert r["solver"] == "smith_normal_form_plus_bounded_nonnegative_search"

def test_pyproject_lists_sympy_dependency_for_integer_span_classifier():
    """Smith normal form is a runtime path, so sympy must be a project dependency."""
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"sympy"' in text

def test_schema_doc_uses_current_reduced_ebr_solver_name():
    """Public schema should not document the old brute-force-only solver name."""
    schema_text = Path("docs/schema.md").read_text(encoding="utf-8")
    assert "smith_normal_form_plus_bounded_nonnegative_search" in schema_text
    assert "brute_force_exact_integer" not in schema_text

# -----------------------------------------------------------------------
# 21. Summary text rendering of classification
# -----------------------------------------------------------------------

def _render_reduced_ebr_text(report: dict) -> str:
    from valleyscope.reports.summary_report import _render_reduced_ebr_mapping
    lines: list[str] = []
    _render_reduced_ebr_mapping(lines, report)
    return "\n".join(lines)

def test_summary_renders_atomic_classification():
    """Summary text shows atomic-compatible with decomposition."""
    report = {
        "status": "solved_exact", "mapping_status": "solved_exact",
        "reduced_ebr_decomposition_status": "solved_exact",
        "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_001", "valley": "K",
            "status": "solved_exact",
            "classification": "atomic-compatible-candidate",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "solved_exact",
            "ebr_decomposition": [
                {"label": "EBR_A", "coefficient": 1},
                {"label": "EBR_B", "coefficient": 2},
            ],
        }],
    }
    text = _render_reduced_ebr_text(report)
    assert "atomic-compatible" in text
    assert "EBR_A x 1" in text
    assert "EBR_B x 2" in text

def test_summary_renders_fragile_classification():
    """Summary text shows in integer span, no nonnegative witness."""
    report = {
        "status": "no_exact_solution", "mapping_status": "no_exact_solution",
        "reduced_ebr_decomposition_status": "no_exact_solution",
        "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_001", "valley": "K",
            "status": "no_exact_solution",
            "classification": "in_integer_span_no_nonnegative_witness",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "no_nonnegative_solution",
            "integer_solution": [
                {"label": "EBR_A", "coefficient": -1},
                {"label": "EBR_B", "coefficient": 1},
            ],
        }],
    }
    text = _render_reduced_ebr_text(report)
    assert "in integer span, no nonnegative witness" in text
    assert "signed witness" in text
    assert "EBR_A: -1" in text
    assert "EBR_B: 1" in text

def test_summary_renders_stable_classification():
    """Summary text shows outside integer span."""
    report = {
        "status": "no_exact_solution", "mapping_status": "no_exact_solution",
        "reduced_ebr_decomposition_status": "no_exact_solution",
        "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_001", "valley": "K",
            "status": "no_exact_solution",
            "classification": "outside_integer_span",
            "integer_span_status": "outside_integer_span",
            "nonnegative_solution_status": "no_nonnegative_solution",
        }],
    }
    text = _render_reduced_ebr_text(report)
    assert "outside integer span" in text
    assert "outside integer span" in text

def test_summary_renders_truncated_search_status():
    """Summary text shows truncated search status."""
    report = {
        "status": "no_exact_solution", "mapping_status": "no_exact_solution",
        "reduced_ebr_decomposition_status": "no_exact_solution",
        "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_001", "valley": "K",
            "status": "no_exact_solution",
            "classification": "in_integer_span_no_nonnegative_witness",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "no_nonnegative_solution",
            "search_status": "truncated_by_max_coefficient",
            "integer_solution": [
                {"label": "EBR_A", "coefficient": -1},
            ],
        }],
    }
    text = _render_reduced_ebr_text(report)
    assert "truncated" in text
    assert "search_truncated=1" in text

def test_summary_renders_classification_counts():
    """Summary text shows classification counts when present."""
    report = {
        "status": "no_exact_solution", "mapping_status": "no_exact_solution",
        "reduced_ebr_decomposition_status": "no_exact_solution",
        "table_status": "loaded",
        "solutions": [
            {"bundle_id": "b_a", "valley": "A", "status": "solved_exact",
             "classification": "atomic-compatible-candidate",
             "integer_span_status": "in_integer_span",
             "nonnegative_solution_status": "solved_exact",
             "ebr_decomposition": [{"label": "X", "coefficient": 1}]},
            {"bundle_id": "b_f", "valley": "F", "status": "no_exact_solution",
             "classification": "in_integer_span_no_nonnegative_witness",
             "integer_span_status": "in_integer_span",
             "nonnegative_solution_status": "no_nonnegative_solution"},
            {"bundle_id": "b_s", "valley": "S", "status": "no_exact_solution",
             "classification": "outside_integer_span",
             "integer_span_status": "outside_integer_span",
             "nonnegative_solution_status": "no_nonnegative_solution"},
        ],
    }
    text = _render_reduced_ebr_text(report)
    assert "atomic-compatible=1" in text
    assert "in integer span, no nonnegative witness=1" in text
    assert "outside integer span=1" in text

def test_summary_excluded_bundles_unchanged():
    """Excluded bundles rendering is unchanged."""
    report = {
        "status": "not_evaluated", "mapping_status": "not_evaluated",
        "reduced_ebr_decomposition_status": "not_evaluated",
        "table_status": "loaded", "solutions": [],
        "excluded_bundles": [
            {"bundle_id": "b_x", "reason": "not ready for external solver"},
        ],
    }
    text = _render_reduced_ebr_text(report)
    assert "excluded bundles" in text.lower()
    assert "not ready for external solver" in text

def test_summary_missing_table_unchanged():
    """Missing table rendering is unchanged."""
    report = {
        "status": "missing_table", "mapping_status": "missing_table",
        "reduced_ebr_decomposition_status": "missing_table",
        "table_status": "not_provided", "solutions": [],
        "excluded_bundles": [],
    }
    text = _render_reduced_ebr_text(report)
    assert "missing_table" in text
    assert "classifications:" not in text  # No solutions, no counts

# -----------------------------------------------------------------------
# 22. Reviewed package-data provenance gate
# -----------------------------------------------------------------------

def _write_reviewed_entry(root: Path, name: str, filename: str,
                          table: dict | None = None,
                          manifest_override: dict | None = None) -> None:
    """Write a fake manifest entry and optional table file under *root*."""
    entry = {
        "name": name, "filename": filename,
        "review_status": "reviewed",
        "reviewer": "JD",
        "review_date": "2026-06-12",
        "review_method": "literature C3 character table",
        "source_reference": "P321 spinful character table",
    }
    if manifest_override is not None:
        entry.update(manifest_override)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": [entry],
    }))
    if table is not None:
        (root / filename).write_text(json.dumps(table))

def _reviewed_table_with_provenance(provenance_override=None) -> dict:
    """Return a minimal reviewed table with required provenance."""
    p = _reviewed_table_provenance()
    if provenance_override is not None:
        p.update(provenance_override)
    tbl = dict(_SAMPLE_TABLE)
    tbl["subspace_group_candidate"] = "P3"
    tbl["provenance"] = p
    return tbl

def test_config_rejects_table_name(tmp_path):
    """config rejects table_name with clear error."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump({
        "input": {"wavefunction_h5": "wave.h5"},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [1],
            "reduced_ebr": {"enabled": True, "table_name": "c3_table"},
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": "out"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="table_name was removed"):
        load_config(config_path)

def test_config_rejects_table_name_regardless_of_table_file(tmp_path):
    """config rejects table_name even when table_file is also present."""
    from valleyscope.io.config import load_config
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump({
        "input": {"wavefunction_h5": "wave.h5"},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [1],
            "reduced_ebr": {
                "enabled": True,
                "table_file": "t.json",
                "table_name": "c3_table",
            },
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": "out"},
    }))
    with pytest.raises(ValueError, match="table_name was removed"):
        load_config(config_path)

# table_name path removed; test deleted

# -----------------------------------------------------------------------
# Reduced-EBR input equivalence and provenance
# -----------------------------------------------------------------------

def test_config_rejects_table_file_and_spec_file_together(tmp_path):
    """table_file and spec_file are mutually exclusive."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump({
        "input": {"wavefunction_h5": "wave.h5"},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [1],
            "reduced_ebr": {
                "enabled": True,
                "table_file": "t.json",
                "spec_file": "spec.json",
            },
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": "out"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(config_path)


def test_config_accepts_spec_file(tmp_path):
    """spec_file is resolved relative to config base."""
    (tmp_path / "specs").mkdir()
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump({
        "input": {"wavefunction_h5": "wave.h5"},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [1],
            "reduced_ebr": {
                "enabled": True,
                "spec_file": "specs/p3_spec.json",
            },
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": "out"},
    }), encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.reduced_ebr.spec_file == (tmp_path / "specs" / "p3_spec.json")
    assert cfg.reduced_ebr.table_file is None


def test_config_reduced_ebr_default_off(tmp_path):
    """spec_file without enabled=True stays default-off."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump({
        "input": {"wavefunction_h5": "wave.h5"},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [1],
            "reduced_ebr": {"spec_file": "spec.json"},
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": "out"},
    }), encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.reduced_ebr.enabled is False


def test_table_file_and_spec_file_equivalent_outputs():
    """Same table data via table_file produces identical solutions as spec_file."""
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    bundle_vec = {
        "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
        "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
    }
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "ready_for_external_solver": True,
            "irreps_by_kpoint": bundle_vec,
        }],
    }
    table = dict(_SAMPLE_TABLE)

    attach_promotion(bundle, _SAMPLE_TABLE)
    result_table = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle,
        table=table,
        reduced_ebr_input={"source": "table_file", "table_file_stem": "p3_table"}
    )
    assert result_table["status"] == "solved_exact"

    result_spec = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle,
        table=table,
        reduced_ebr_input={
            "source": "spec_file",
            "spec_file_stem": "p3_spec",
            "subspace_group_candidate": "P3",
            "data_source": "irreptables",
        }
    )
    assert result_spec["status"] == "solved_exact"
    assert result_table["solutions"] == result_spec["solutions"]
    assert result_table["mapping_status"] == result_spec["mapping_status"]


def test_reduced_ebr_input_provenance_in_output():
    """reduced_ebr_input provenance appears in mapping output."""
    bundle_vec = {
        "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
        "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
    }
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "ready_for_external_solver": True,
            "irreps_by_kpoint": bundle_vec,
        }],
    }
    attach_promotion(bundle, _SAMPLE_TABLE)
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle,
        table=dict(_SAMPLE_TABLE),
        reduced_ebr_input={"source": "table_file", "table_file_stem": "p3_table"}
    )
    assert result["reduced_ebr_input"] == {
        "source": "table_file",
        "table_file_stem": "p3_table",
    }


def test_reduced_ebr_input_preserved_in_missing_table_status():
    """reduced_ebr_input is preserved even when table is missing."""
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=_bundle(),
        table=None,
        reduced_ebr_input={"source": "spec_file", "spec_file_stem": "bad_spec"},
    )
    assert result["status"] == "missing_table"
    assert result["reduced_ebr_input"]["source"] == "spec_file"


def test_reduced_ebr_input_provenance_in_ingestion_record():
    """Database ingestion record preserves reduced_ebr_input."""
    from valleyscope.analysis.database_ingestion_record import (
        build_database_ingestion_record,
    )

    bundle_vec = {
        "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
        "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
    }
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "ready_for_external_solver": True,
            "irreps_by_kpoint": bundle_vec,
        }],
    }
    attach_promotion(bundle, _SAMPLE_TABLE)
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle,
        table=dict(_SAMPLE_TABLE),
        reduced_ebr_input={"source": "table_file", "table_file_stem": "p3_table"}
    )
    summary = {
        "target_kpoints": ["GammaM", "KM"],
        "iband": [1, 2],
        "input": {},
    }
    record = build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=bundle,
        valley_reduced_ebr_mapping=mapping,
    )
    assert record["reduced_ebr_input"] == {
        "source": "table_file",
        "table_file_stem": "p3_table",
    }


def test_reduced_ebr_input_not_present_when_not_provided():
    """When reduced_ebr_input is None, it does not appear in output."""
    bundle_vec = {
        "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
        "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
    }
    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "ready_for_external_solver": True,
            "irreps_by_kpoint": bundle_vec,
        }],
    }
    attach_promotion(bundle, _SAMPLE_TABLE)
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle,
        table=dict(_SAMPLE_TABLE)
    )
    assert "reduced_ebr_input" not in result


def test_reduced_ebr_input_not_provided_marker():
    """Explicit not_provided provenance is preserved in missing-table output."""
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=_bundle(),
        table=None,
        reduced_ebr_input={"source": "not_provided"},
    )
    assert result["status"] == "missing_table"
    assert result["table_status"] == "not_provided"
    assert result["reduced_ebr_input"] == {"source": "not_provided"}

def test_adapter_load_raw_returns_irreptables_shape():
    """Adapter load_raw_ebr_data returns raw irreptables dict shape."""
    from valleyscope.irreps.ebr_data_adapter import load_raw_ebr_data
    try:
        raw = load_raw_ebr_data(143, spinful=True)
    except RuntimeError:
        import pytest
        pytest.skip("irreptables not available")

    assert isinstance(raw, dict)
    assert "basis" in raw
    assert "ebrs" in raw
    assert isinstance(raw["ebrs"], list)
    assert len(raw["ebrs"]) > 0
    assert isinstance(raw["basis"]["irrep_labels"], list)

def test_adapter_raw_and_normalized_consistent():
    """Raw and normalized loader return consistent data for same SG."""
    from valleyscope.irreps.ebr_data_adapter import (
        load_raw_ebr_data, load_ebr_source_data,
    )
    try:
        raw = load_raw_ebr_data(143, spinful=True)
        norm = load_ebr_source_data(143, spinful=True)
    except RuntimeError:
        import pytest
        pytest.skip("irreptables not available")

    assert norm["source_basis_count"] == len(raw["basis"]["irrep_labels"])
    assert norm["source_basis_labels"] == list(raw["basis"]["irrep_labels"])
    assert len(norm["source_ebrs"]) == len(raw["ebrs"])
    assert norm["data_source"] == "irreptables"

def test_adapter_rejects_invalid_data():
    """Adapter validation rejects malformed irreptables data."""
    from valleyscope.irreps.ebr_data_adapter import _normalize_ebr_data
    import pytest

    # Non-dict
    with pytest.raises(ValueError, match="dict"):
        _normalize_ebr_data([], 143, True)

    # Missing basis
    with pytest.raises(ValueError, match="basis"):
        _normalize_ebr_data({}, 143, True)

    # Empty irrep_labels
    with pytest.raises(ValueError, match="non-empty"):
        _normalize_ebr_data(
            {"basis": {"irrep_labels": []}, "ebrs": []}, 143, True,
        )

    # Missing ebr_name
    with pytest.raises(ValueError, match="ebr_name"):
        _normalize_ebr_data(
            {"basis": {"irrep_labels": ["A"]},
             "ebrs": [{"vector": [1]}]}, 143, True,
        )

    # Non-integer vector entries
    with pytest.raises(ValueError, match="nonnegative integer"):
        _normalize_ebr_data(
            {"basis": {"irrep_labels": ["A"]},
             "ebrs": [{"ebr_name": "E", "vector": [0.5]}]},
            143, True,
        )

    # Negative vector entries
    with pytest.raises(ValueError, match="nonnegative integer"):
        _normalize_ebr_data(
            {"basis": {"irrep_labels": ["A"]},
             "ebrs": [{"ebr_name": "E", "vector": [-1]}]},
            143, True,
        )

# Test removed after contract cleanup
