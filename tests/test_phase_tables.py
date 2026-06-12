import json
from pathlib import Path

import pytest

from valleyscope.data.valley_irreps.catalog import get_irrep_phase_list
from valleyscope.analysis.valley_irrep_matching import build_valley_irrep_matching_report

# Valley-irrep phase table data contract tests
# -----------------------------------------------------------------------

def test_phase_table_c3_loads_labels():
    """C3 phase table must provide exactly 3 labels that match the old hardcoded set."""
    from valleyscope.data.valley_irreps.catalog import get_irrep_phase_list
    irreps = get_irrep_phase_list("spinful_C3_phase_v1")
    labels = {e["label"] for e in irreps}
    assert labels == {"C3_spinor_phase_+1/6", "C3_spinor_phase_+1/2", "C3_spinor_phase_-1/6"}
    assert all(len(e["phases"]) == 1 for e in irreps)


def test_phase_table_c2_loads_labels():
    """C2 phase table must provide exactly 2 labels that match the old hardcoded set."""
    from valleyscope.data.valley_irreps.catalog import get_irrep_phase_list
    irreps = get_irrep_phase_list("spinful_C2_phase_v1")
    labels = {e["label"] for e in irreps}
    assert labels == {"C2_spinor_phase_+1/4", "C2_spinor_phase_-1/4"}
    assert all(len(e["phases"]) == 1 for e in irreps)


def test_phase_table_tables_implemented_unchanged():
    """tables_implemented must remain ['spinful_C3', 'spinful_C2']."""
    # Construct a matching report with known table names.
    from valleyscope.analysis.valley_irrep_matching import build_valley_irrep_matching_report
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            },
        },
    }
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=workflow,
        symmetry_adapted_valley_report=None,
    )
    assert report["tables_implemented"] == ["spinful_C3", "spinful_C2"]


def test_phase_table_files_no_ebr_vectors():
    """Phase table JSON files must not contain EBR vectors."""
    from valleyscope.data.valley_irreps.catalog import package_data_root
    for fname in ["spinful_C3_phase_v1.json", "spinful_C2_phase_v1.json"]:
        data = (package_data_root() / fname).read_text(encoding="utf-8")
        assert "ebr" not in data.lower() and "vector" not in data.lower(), (
            f"{fname} must not contain EBR vectors"
        )


def test_phase_table_files_no_material_names():
    """Phase table JSON files must not contain real material names."""
    from valleyscope.data.valley_irreps.catalog import package_data_root
    for fname in ["spinful_C3_phase_v1.json", "spinful_C2_phase_v1.json",
                   "manifest.json"]:
        data = (package_data_root() / fname).read_text(encoding="utf-8")
        for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
            assert name not in data, f"{fname} must not contain {name!r}"


def test_phase_table_readme_mentions_irrep_not_ebr():
    """README must clarify these are irrep matching tables, not EBR tables."""
    readme = Path("valleyscope/data/valley_irreps/README.md").read_text(encoding="utf-8").lower()
    assert "not reduced ebr" in readme or "irrep matching data" in readme


def test_phase_table_validator_rejects_noncanonical_phase():
    """Package phase tables must store phases in the documented convention."""
    from valleyscope.data.valley_irreps.catalog import _validate_phase_table

    raw = {
        "schema_version": "1.0.0",
        "name": "bad_table",
        "spinful": True,
        "operation_order": 3,
        "subspace_group_candidates": ["C3_like", "P3"],
        "phase_convention": "eigenvalue = exp(2*pi*i*phase), phase in (-0.5, 0.5]",
        "irreps": [{"label": "bad", "phases": [1.25]}],
    }
    with pytest.raises(ValueError, match="canonical range"):
        _validate_phase_table(raw, "bad_table")


def test_phase_table_validator_rejects_ebr_vector_payload():
    """Valley-irrep phase data must not carry EBR vectors."""
    from valleyscope.data.valley_irreps.catalog import _validate_phase_table

    raw = {
        "schema_version": "1.0.0",
        "name": "bad_table",
        "spinful": True,
        "operation_order": 2,
        "subspace_group_candidates": ["C2_like", "P2"],
        "phase_convention": "eigenvalue = exp(2*pi*i*phase), phase in (-0.5, 0.5]",
        "irreps": [{"label": "bad", "phases": [0.25], "vector": [1]}],
    }
    with pytest.raises(ValueError, match="forbidden EBR"):
        _validate_phase_table(raw, "bad_table")


def test_phase_table_design_doc_updated():
    """Design doc must reflect that minimal phase tables are package data."""
    doc = Path("docs/reduced_dimensional_irrep_ebr_data_model.md").read_text(encoding="utf-8")
    assert "valleyscope/data/valley_irreps" in doc
    assert "versioned package data" in doc


def test_phase_table_catalog_no_irrep2_import():
    """Phase table catalog must not import irrep2."""
    src = Path("valleyscope/data/valley_irreps/catalog.py").read_text(encoding="utf-8")
    assert "irrep2" not in src, "catalog.py must not import irrep2"
