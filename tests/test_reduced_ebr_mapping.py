import json
import subprocess
import sys
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
    table["expected_hsps"] = ["GammaM"]
    table["irreps"] = ["GammaM:wrong_label"]
    table["ebrs"] = [{"label": "X", "vector": [1]}]
    b = {
        "bundles": [{
            "bundle_id": "b_001",
            "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM"],
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

def test_table_schema_doc_required_keys():
    """Verify docs/reduced_ebr_table_schema.md documents all required table keys."""
    from valleyscope.analysis.reduced_ebr_mapping import _REQUIRED_TABLE_KEYS
    doc_path = Path("docs/reduced_ebr_table_schema.md")
    assert doc_path.exists(), "docs/reduced_ebr_table_schema.md must exist"
    doc_text = doc_path.read_text(encoding="utf-8")
    for key in sorted(_REQUIRED_TABLE_KEYS):
        assert f"`{key}`" in doc_text, (
            f"docs/reduced_ebr_table_schema.md must document key '{key}'"
        )
    # Must state no built-in tables.
    assert "No built-in EBR tables" in doc_text or "no built-in" in doc_text.lower()
    # Must state no heuristic fits.
    assert "heuristic" in doc_text.lower() and "no" in doc_text.lower()


def test_table_schema_doc_status_values():
    """Verify docs/reduced_ebr_table_schema.md documents allowed status values."""
    doc_text = Path("docs/reduced_ebr_table_schema.md").read_text(encoding="utf-8")
    for status in ["not_evaluated", "missing_table", "solved_exact", "no_exact_solution"]:
        assert f"`{status}`" in doc_text, (
            f"docs/reduced_ebr_table_schema.md must document status '{status}'"
        )


def test_table_schema_doc_labels_use_clike_form():
    """Verify docs/reduced_ebr_table_schema.md uses C{order}_like for subspace_group_candidate."""
    doc_text = Path("docs/reduced_ebr_table_schema.md").read_text(encoding="utf-8")
    assert "C3_like" in doc_text
    assert "C2_like" in doc_text
    assert 'C{order}_like' in doc_text


def test_table_schema_doc_expected_hsps_basis_contract():
    """Schema doc must state expected_hsps is enforced as the EBR basis contract."""
    doc_text = Path("docs/reduced_ebr_table_schema.md").read_text(encoding="utf-8")
    assert "reduced-dimensional EBR basis contract" in doc_text
    assert "bundle `expected_hsps`" in doc_text
    assert "`bundle.irreps_by_kpoint`" in doc_text
    assert "Missing, extra, malformed, or inferred HSP data is not accepted" in doc_text


def test_schema_md_labels_use_clike_form():
    """Verify docs/schema.md uses C{order}_like for subspace_group_candidate examples."""
    schema_text = Path("docs/schema.md").read_text(encoding="utf-8")
    assert "C3_like" in schema_text
    assert "C2_like" in schema_text
    # Must not use P3/P2 as subspace_group_candidate example values.
    assert '"subspace_group_candidate": "P3"' not in schema_text
    assert '"subspace_group_candidate": "P2"' not in schema_text


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
        "subspace_group_candidate": "C3_like",
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
    assert mapping["status"] == "solved_exact"
    s = mapping["solutions"][0]
    assert s["status"] == "solved_exact"
    labels = {e["label"] for e in s["ebr_decomposition"]}
    assert labels == {"EBR_A", "EBR_B"}


def test_cli_no_exact_solution(tmp_path):
    """CLI reports no_exact_solution for an unsolvable toy bundle."""
    out = tmp_path / "out.json"
    bundle_path = tmp_path / "bundle.json"
    table_path = tmp_path / "table.json"
    _write_bundle(bundle_path, [{
        "bundle_id": "b_001", "valley": "K",
        "subspace_group_candidate": "C3_like",
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
    assert mapping["status"] == "no_exact_solution"
    assert mapping["solutions"][0]["status"] == "no_exact_solution"


def test_cli_stdout_includes_status_and_output_path(capsys, tmp_path):
    """CLI stdout includes solved_exact and the output path."""
    out = tmp_path / "out.json"
    bundle_path = tmp_path / "bundle.json"
    table_path = tmp_path / "table.json"
    _write_bundle(bundle_path, [{
        "bundle_id": "b_001", "valley": "K",
        "subspace_group_candidate": "C3_like",
        "ready_for_external_solver": True,
        "irreps_by_kpoint": {
            "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
            "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
        },
    }])
    _write_table(table_path, _SAMPLE_TABLE)

    _cli(bundle_path, table_path, out)
    captured = capsys.readouterr().out
    assert "solved_exact" in captured
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
        "subspace_group_candidate": "C3_like",
        "ready_for_external_solver": True,
        "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"], "KM": []},
    }])
    _write_table(table_path, bad_table)

    rc = _cli(bundle_path, table_path, out)
    assert rc != 0, "CLI should fail on invalid table"
    assert not out.exists(), "Must not write output for invalid table"


def test_cli_missing_table_file_fails():
    """CLI fails when table file does not exist."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_bundle(d / "bundle.json", [])
        rc = _cli(d / "bundle.json", d / "nonexistent.json", d / "out.json")
        assert rc != 0
        assert not (d / "out.json").exists()


def test_cli_missing_bundle_file_fails():
    """CLI fails when bundle file does not exist."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_table(d / "table.json", _SAMPLE_TABLE)
        rc = _cli(d / "nonexistent.json", d / "table.json", d / "out.json")
        assert rc != 0
        assert not (d / "out.json").exists()


def test_cli_respects_max_coefficient(tmp_path):
    """CLI passes --max-coefficient through to the solver."""
    out = tmp_path / "out.json"
    bundle_path = tmp_path / "bundle.json"
    table_path = tmp_path / "table.json"
    # Bundle with large count that needs max_coeff >= 6
    _write_bundle(bundle_path, [{
        "bundle_id": "b_001", "valley": "K",
        "subspace_group_candidate": "C3_like",
        "ready_for_external_solver": True,
        "irreps_by_kpoint": {
            "GammaM": ["C3_spinor_phase_+1/2"] * 6,
            "KM": ["C3_spinor_phase_+1/6"] * 6,
        },
    }])
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C3_like",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:C3_spinor_phase_+1/2", "KM:C3_spinor_phase_+1/6"],
        "ebrs": [{"label": "EBR_X", "vector": [1, 1]}],
    }
    _write_table(table_path, table)

    # With max_coeff=6: 6*[1,1] = [6,6] -> exact
    rc = _cli(bundle_path, table_path, out, "--max-coefficient", "6")
    assert rc == 0
    mapping = json.loads(out.read_text(encoding="utf-8"))
    assert mapping["status"] == "solved_exact"


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

def test_data_model_design_doc_exists():
    """Design doc must exist at the expected path."""
    doc_path = Path("docs/reduced_dimensional_irrep_ebr_data_model.md")
    assert doc_path.exists(), (
        "docs/reduced_dimensional_irrep_ebr_data_model.md must exist"
    )


def test_data_model_doc_mentions_irrep2():
    """Design doc must reference irrep2 as the reduced-dimensional inspiration."""
    doc = Path("docs/reduced_dimensional_irrep_ebr_data_model.md").read_text(encoding="utf-8")
    assert "irrep2" in doc, "design doc must mention irrep2"
    assert "reference-only" in doc, "irrep2 must be documented as reference-only"
    assert "No `irrep2` runtime dependency" in doc, (
        "design doc must forbid irrep2 as a runtime dependency"
    )


def test_data_model_doc_mentions_reduced_dimensional():
    """Design doc must use the 'reduced-dimensional' terminology."""
    doc = Path("docs/reduced_dimensional_irrep_ebr_data_model.md").read_text(encoding="utf-8")
    assert "reduced-dimensional" in doc.lower(), (
        "design doc must mention reduced-dimensional"
    )


def test_data_model_doc_mentions_provenance():
    """Design doc must discuss provenance tracking."""
    doc = Path("docs/reduced_dimensional_irrep_ebr_data_model.md").read_text(encoding="utf-8")
    assert "provenance" in doc.lower(), "design doc must mention provenance"


def test_data_model_doc_mentions_package_name_irrep():
    """Design doc must use the correct Python package name 'irrep' (singular)."""
    doc = Path("docs/reduced_dimensional_irrep_ebr_data_model.md").read_text(encoding="utf-8")
    assert "`irrep`" in doc or "Python package `irrep`" in doc, (
        "design doc must reference Python package 'irrep' (singular)"
    )


def test_data_model_doc_no_direct_3d_ebr_reuse():
    """Design doc must state that 3D irrep EBR tables must not be directly reused."""
    doc = Path("docs/reduced_dimensional_irrep_ebr_data_model.md").read_text(encoding="utf-8")
    doc_lower = doc.lower()
    assert any(phrase in doc_lower for phrase in [
        "not direct reuse", "no direct reuse", "do not directly reuse",
        "not directly reuse", "no 3d", "ebr table reuse as final",
    ]), "design doc must state no direct 3D irrep EBR table reuse"


def test_data_model_doc_physical_objects():
    """Design doc must name the key physical objects."""
    doc = Path("docs/reduced_dimensional_irrep_ebr_data_model.md").read_text(encoding="utf-8")
    required = [
        "HSP little group",
        "valley-preserving subgroup",
        "valley sewing matrix",
        "reduced-dimensional EBR vector",
    ]
    for term in required:
        assert term.lower() in doc.lower(), (
            f"design doc must mention '{term}'"
        )


def test_data_model_doc_label_conventions():
    """Design doc must document C3_like/C2_like vs crystallographic notation."""
    doc = Path("docs/reduced_dimensional_irrep_ebr_data_model.md").read_text(encoding="utf-8")
    assert "C3_like" in doc and "C2_like" in doc, (
        "design doc must document C3_like and C2_like labels"
    )
    assert "P3" in doc and "P2" in doc, (
        "design doc must document crystallographic P3/P2 notation"
    )


def test_data_model_doc_no_material_specific_table_targets():
    """Design doc must keep package-data targets symmetry-based, not material-based."""
    doc = Path("docs/reduced_dimensional_irrep_ebr_data_model.md").read_text(encoding="utf-8")
    for forbidden in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert forbidden not in doc, (
            f"design doc must not use real material '{forbidden}' as a package-data target"
        )


def test_agents_md_uses_correct_package_name_irrep():
    """AGENTS.md must use the correct Python package name 'irrep' (singular)."""
    text = Path("AGENTS.md").read_text(encoding="utf-8")
    assert "`irrep`" in text, "AGENTS.md must reference Python package 'irrep'"
    assert "`irreps`" not in text, "AGENTS.md must not reference 'irreps' (plural)"


def test_plan_md_uses_correct_package_name_irrep():
    """PLAN.md must use the correct Python package name 'irrep' (singular)."""
    text = Path("PLAN.md").read_text(encoding="utf-8")
    assert "`irrep`" in text, "PLAN.md must reference Python package 'irrep'"
    assert "`irreps`" not in text, "PLAN.md must not reference 'irreps' (plural)"


# -----------------------------------------------------------------------
# 17. Package-data skeleton
# -----------------------------------------------------------------------

def test_package_data_root_exists():
    """Package-data directory structure must exist."""
    from valleyscope.data.reduced_ebr.catalog import package_data_root
    root = package_data_root()
    assert root.is_dir()
    assert (root / "__init__.py").exists()
    assert (root / "manifest.json").exists()
    assert (root / "README.md").exists()
    assert (root / "catalog.py").exists()


def test_manifest_has_schema_version_and_empty_tables():
    """Manifest must have schema_version and an empty tables list."""
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    m = load_reduced_ebr_manifest()
    assert isinstance(m.get("schema_version"), str) and m["schema_version"], (
        "manifest must have non-empty schema_version"
    )
    assert m.get("tables") == [], "manifest tables must be empty"


def test_list_tables_returns_empty():
    """list_reviewed_reduced_ebr_tables() must return an empty list."""
    from valleyscope.data.reduced_ebr.catalog import list_reviewed_reduced_ebr_tables
    tables = list_reviewed_reduced_ebr_tables()
    assert tables == [], "no reviewed tables should exist yet"
    assert isinstance(tables, list)


def test_load_nonexistent_table_raises_value_error():
    """load_reviewed_reduced_ebr_table for any name must raise ValueError."""
    from valleyscope.data.reduced_ebr.catalog import load_reviewed_reduced_ebr_table
    with pytest.raises(ValueError, match="no reviewed reduced EBR package table"):
        load_reviewed_reduced_ebr_table("nonexistent_table")


def test_no_table_json_files_except_manifest():
    """Only manifest.json should exist; no other .json table files."""
    from valleyscope.data.reduced_ebr.catalog import package_data_root
    root = package_data_root()
    json_files = sorted(root.glob("*.json"))
    json_names = {f.name for f in json_files}
    assert json_names == {"manifest.json"}, (
        f"Only manifest.json should exist; found: {json_names}"
    )


def test_package_data_readme_states_no_tables():
    """Package-data README must state that no reviewed tables are shipped."""
    from valleyscope.data.reduced_ebr.catalog import package_data_root
    readme = (package_data_root() / "README.md").read_text(encoding="utf-8")
    assert "no reviewed tables" in readme.lower() or "currently empty" in readme.lower(), (
        "README must state no reviewed tables are currently shipped"
    )
    assert "currently" in readme.lower()


def test_catalog_does_not_import_irrep2():
    """Package-data catalog must not import the private irrep2 repository."""
    from valleyscope.data.reduced_ebr.catalog import package_data_root
    src = (package_data_root() / "catalog.py").read_text(encoding="utf-8")
    for forbidden in ["import irrep2", "from irrep2"]:
        assert forbidden not in src, (
            f"catalog.py must not import {forbidden}"
        )


def test_package_data_no_material_names():
    """Package-data files must not contain real material names."""
    import os
    from valleyscope.data.reduced_ebr.catalog import package_data_root
    forbidden = ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]
    for dirpath, _dirnames, filenames in os.walk(str(package_data_root())):
        for fname in filenames:
            if fname.endswith(".pyc"):
                continue
            fpath = Path(dirpath) / fname
            text = fpath.read_text(encoding="utf-8") if not fname.endswith(".json") else ""
            if fname.endswith(".json"):
                text = fpath.read_text(encoding="utf-8")
            for name in forbidden:
                assert name not in text, (
                    f"{fpath} contains forbidden material name {name!r}"
                )
                assert name not in fname, (
                    f"filename {fname!r} contains forbidden material name {name!r}"
                )


def test_data_init_has_no_forbidden_imports():
    """valleyscope/data/__init__.py must not eagerly import solver packages."""
    src = Path("valleyscope/data/__init__.py").read_text(encoding="utf-8")
    for forbidden in ["import irrep", "from irrep", "import irrep2",
                       "from irrep2", "import ortools", "from ortools"]:
        assert forbidden not in src, (
            f"valleyscope/data/__init__.py must not import {forbidden}"
        )


# -----------------------------------------------------------------------
# 18. Loader integration — catalog manifest validation
# -----------------------------------------------------------------------

def _make_fake_catalog_root(tmp_path: Path) -> Path:
    """Create a fake package-data root with minimal manifest."""
    root = tmp_path / "fake_reduced_ebr"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "description": "test catalog",
        "tables": [],
    }))
    return root


def _set_fake_root(monkeypatch, root: Path):
    """Monkeypatch package_data_root to return a fake directory."""
    from valleyscope.data.reduced_ebr import catalog
    monkeypatch.setattr(catalog, "package_data_root", lambda: root)


def test_empty_real_manifest_lists_empty(monkeypatch):
    """Real repo manifest must still list []."""
    from valleyscope.data.reduced_ebr.catalog import list_reviewed_reduced_ebr_tables
    tables = list_reviewed_reduced_ebr_tables()
    assert tables == []


def test_valid_fake_manifest_lists_entries(monkeypatch, tmp_path):
    """A valid fake manifest with entries returns them."""
    root = _make_fake_catalog_root(tmp_path)
    tbl = dict(_SAMPLE_TABLE)
    (root / "toy.json").write_text(json.dumps(tbl))
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": [{"name": "toy", "filename": "toy.json"}],
    }))
    _set_fake_root(monkeypatch, root)

    from valleyscope.data.reduced_ebr.catalog import (
        list_reviewed_reduced_ebr_tables,
        load_reviewed_reduced_ebr_table,
    )
    entries = list_reviewed_reduced_ebr_tables()
    assert len(entries) == 1
    assert entries[0]["name"] == "toy"

    loaded = load_reviewed_reduced_ebr_table("toy")
    assert loaded["subspace_group_candidate"] == "C3_like"


def test_fake_table_fails_same_validation_as_external(monkeypatch, tmp_path):
    """Invalid fake table raises the same validation error as external path."""
    root = _make_fake_catalog_root(tmp_path)
    bad_table = dict(_SAMPLE_TABLE)
    bad_table["ebrs"] = [{"label": "X", "vector": [1, 2]}]  # wrong length
    (root / "bad.json").write_text(json.dumps(bad_table))
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": [{"name": "bad", "filename": "bad.json"}],
    }))
    _set_fake_root(monkeypatch, root)

    from valleyscope.data.reduced_ebr.catalog import load_reviewed_reduced_ebr_table
    with pytest.raises(ValueError, match="vector length"):
        load_reviewed_reduced_ebr_table("bad")


def test_manifest_not_dict_raises(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text("[]")
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    with pytest.raises(ValueError, match="JSON object"):
        load_reduced_ebr_manifest()


def test_manifest_missing_schema_version_raises(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text(json.dumps({"tables": []}))
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    with pytest.raises(ValueError, match="schema_version"):
        load_reduced_ebr_manifest()


def test_manifest_non_list_tables_raises(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0", "tables": "not_a_list",
    }))
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    with pytest.raises(ValueError, match="'tables' must be a list"):
        load_reduced_ebr_manifest()


def test_manifest_duplicate_names_raises(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": [
            {"name": "dup", "filename": "a.json"},
            {"name": "dup", "filename": "b.json"},
        ],
    }))
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    with pytest.raises(ValueError, match="duplicate"):
        load_reduced_ebr_manifest()


def test_manifest_entry_not_dict_raises(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": ["not_a_dict"],
    }))
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    with pytest.raises(ValueError, match="JSON object"):
        load_reduced_ebr_manifest()


def test_manifest_entry_missing_name_raises(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": [{"filename": "x.json"}],
    }))
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    with pytest.raises(ValueError, match="'name'"):
        load_reduced_ebr_manifest()


def test_manifest_entry_missing_filename_raises(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": [{"name": "no_file"}],
    }))
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    with pytest.raises(ValueError, match="'filename'"):
        load_reduced_ebr_manifest()


def test_manifest_absolute_filename_rejected(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": [{"name": "abs", "filename": "/etc/passwd"}],
    }))
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    with pytest.raises(ValueError, match="relative"):
        load_reduced_ebr_manifest()


def test_manifest_dotdot_filename_rejected(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": [{"name": "escape", "filename": "../secret.json"}],
    }))
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    with pytest.raises(ValueError, match="outside"):
        load_reduced_ebr_manifest()


def test_manifest_sibling_dotdot_rejected(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": [{"name": "escape", "filename": "sub/../../secret.json"}],
    }))
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reduced_ebr_manifest
    with pytest.raises(ValueError, match="'..'"):
        load_reduced_ebr_manifest()


def test_missing_table_file_raises(monkeypatch, tmp_path):
    root = _make_fake_catalog_root(tmp_path)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "tables": [{"name": "ghost", "filename": "ghost.json"}],
    }))
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import load_reviewed_reduced_ebr_table
    with pytest.raises(FileNotFoundError):
        load_reviewed_reduced_ebr_table("ghost")


def test_load_nonexistent_name_raises(monkeypatch):
    from valleyscope.data.reduced_ebr.catalog import load_reviewed_reduced_ebr_table
    with pytest.raises(ValueError, match="no reviewed reduced EBR package table"):
        load_reviewed_reduced_ebr_table("nonexistent")


def test_load_empty_name_raises(monkeypatch):
    from valleyscope.data.reduced_ebr.catalog import load_reviewed_reduced_ebr_table
    with pytest.raises(ValueError, match="non-empty"):
        load_reviewed_reduced_ebr_table("")


def test_catalog_loader_integration_does_not_import_irrep2():
    """catalog.py must not import irrep2 (private reference-only repo)."""
    from valleyscope.data.reduced_ebr.catalog import package_data_root
    src = (package_data_root() / "catalog.py").read_text(encoding="utf-8")
    for forbidden in ["import irrep2", "from irrep2"]:
        assert forbidden not in src, f"catalog.py must not import {forbidden}"


def test_empty_manifest_still_lists_empty(monkeypatch, tmp_path):
    """Even a fake root with empty manifest returns []."""
    root = _make_fake_catalog_root(tmp_path)
    _set_fake_root(monkeypatch, root)
    from valleyscope.data.reduced_ebr.catalog import list_reviewed_reduced_ebr_tables
    assert list_reviewed_reduced_ebr_tables() == []


# -----------------------------------------------------------------------
# 19. Reduced EBR basis compatibility gate
# -----------------------------------------------------------------------

def _bundle_with_hsps(expected, irreps_by_kp, g="C3_like", ready=True):
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
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
    assert r["status"] == "solved_exact"
    assert r["solutions"][0]["status"] == "solved_exact"


def test_table_extra_hsp_excludes_bundle():
    """Table expects more HSPs than bundle has → excluded."""
    b = _bundle_with_hsps(
        expected=["GammaM"],
        irreps_by_kp={"GammaM": ["C3_spinor_phase_+1/2"]},
    )
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
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
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
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
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
    assert len(r["excluded_bundles"]) >= 1
    reasons = " ".join(e["reason"] for e in r["excluded_bundles"])
    assert "irrep HSP basis mismatch" in reasons


def test_malformed_declared_expected_hsps_excludes():
    """Declared expected_hsps must be a unique list, not legacy fallback."""
    b = _bundle_with_hsps(
        expected="GammaM",
        irreps_by_kp={
            "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
            "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
        },
    )
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
    assert len(r["excluded_bundles"]) == 1
    assert "malformed expected_hsps" in r["excluded_bundles"][0]["reason"]


def test_legacy_bundle_without_expected_hsps_still_works():
    """Legacy bundle without expected_hsps derives basis from irreps_by_kpoint keys."""
    b = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {
                "GammaM": ["C3_spinor_phase_+1/2", "C3_spinor_phase_+1/2"],
                "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"],
            },
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
    assert r["status"] == "solved_exact"


def test_legacy_bundle_without_expected_hsps_fails_when_keys_mismatch():
    """Legacy bundle without expected_hsps fails when irreps_by_kpoint keys mismatch table."""
    b = {
        "bundles": [{
            "bundle_id": "b_001", "valley": "K",
            "subspace_group_candidate": "C3_like",
            "ready_for_external_solver": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
        }],
    }
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
    assert len(r["excluded_bundles"]) == 1
    reason = r["excluded_bundles"][0]["reason"]
    assert "expected_hsps mismatch" in reason or "irrep HSP basis mismatch" in reason


def test_basis_gate_before_group_check():
    """Basis mismatch excludes even when group matches."""
    b = _bundle_with_hsps(
        expected=["GammaM"],
        irreps_by_kp={"GammaM": ["C3_spinor_phase_+1/2"]},
        g="C3_like",  # group matches table
    )
    r = build_reduced_ebr_mapping(ebr_export_bundle=b, table=_SAMPLE_TABLE)
    assert len(r["excluded_bundles"]) == 1
    assert "expected_hsps mismatch" in r["excluded_bundles"][0]["reason"]


def test_no_material_names_in_basis_gate_code():
    """Basis compatibility gate must not use material-specific logic."""
    src = Path("valleyscope/analysis/reduced_ebr_mapping.py").read_text(encoding="utf-8")
    for forbidden in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert forbidden not in src, (
            f"reduced_ebr_mapping.py must not contain {forbidden!r}"
        )


def test_no_irrep2_import_in_basis_gate():
    """Basis compatibility gate must not import irrep2."""
    src = Path("valleyscope/analysis/reduced_ebr_mapping.py").read_text(encoding="utf-8")
    assert "irrep2" not in src, "reduced_ebr_mapping.py must not import irrep2"
