"""Auto-canonical reduced EBR table builder tests.

Covers the automatic derivation path from canonical irrep labels through
irreptables EBR data to exact reduced EBR decomposition.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.workflows.analyze_hsp import analyze_hsp, _resolve_auto_canonical_ebr_table


# -----------------------------------------------------------------------
# Unit: _resolve_auto_canonical_ebr_table
# -----------------------------------------------------------------------

def test_auto_canonical_resolver_blocks_on_none_bundle():
    table, inp = _resolve_auto_canonical_ebr_table(
        ebr_export_bundle=None, spinor_wf=True,
    )
    assert table is None
    assert inp["source"] == "auto_canonical_blocked"
    assert "no export bundle" in inp["reason"]


def test_auto_canonical_resolver_blocks_on_empty_bundles():
    table, inp = _resolve_auto_canonical_ebr_table(
        ebr_export_bundle={"bundles": []}, spinor_wf=True,
    )
    assert table is None
    assert inp["source"] == "auto_canonical_blocked"
    assert "no ready bundles" in inp["reason"]


def test_auto_canonical_resolver_blocks_on_no_subspace_sg():
    table, inp = _resolve_auto_canonical_ebr_table(
        ebr_export_bundle={
            "bundles": [{
                "bundle_id": "b_001",
                "ready_for_external_solver": True,
                "subspace_group_candidate": "P3",
                "irreps_by_kpoint": {"GammaM": ["-GM4"]},
                "expected_hsps": ["GammaM"],
            }],
        },
        spinor_wf=True,
    )
    assert table is None
    assert inp["source"] == "auto_canonical_blocked"
    assert "resolved subspace space group number" in inp["reason"]


def test_auto_canonical_resolver_blocks_on_no_irreps_by_kpoint():
    table, inp = _resolve_auto_canonical_ebr_table(
        ebr_export_bundle={
            "bundles": [{
                "bundle_id": "b_001",
                "ready_for_external_solver": True,
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "status": "resolved",
                    "candidate_space_group_number": 143,
                    "candidate_space_group_symbol": "P3",
                },
                "expected_hsps": ["GammaM"],
            }],
        },
        spinor_wf=True,
    )
    assert table is None
    assert inp["source"] == "auto_canonical_blocked"
    assert "irreps_by_kpoint" in inp["reason"]


# -----------------------------------------------------------------------
# Real label sets from irreptables (verified)
# -----------------------------------------------------------------------

# SG 75 (P4) spinor labels.
_P4_GM_LABELS = ["-GM5", "-GM6", "-GM7", "-GM8"]
_P4_X_LABELS = ["-X3", "-X4"]

# SG 143 (P3) spinor labels.
_P3_GM_LABELS = ["-GM4", "-GM5", "-GM6"]
_P3_K_LABELS = ["-K4", "-K5", "-K6"]

# SG 143 (P3) non-spinor labels.
_P3_GM_NONSPINOR = ["GM1", "GM2", "GM3"]


def _make_p4_fake_ebr_loader(gm_labels=None, x_labels=None):
    """Fake irreptables EBR loader for SG 75 P4 spinor."""
    gm = gm_labels if gm_labels is not None else _P4_GM_LABELS
    x = x_labels if x_labels is not None else _P4_X_LABELS
    all_labels = list(gm) + list(x)
    # Partition GM labels into two groups so each EBR targets a subset.
    gm_half = len(gm) // 2 if len(gm) >= 2 else 1

    def _loader(sg, spin):
        assert sg == 75
        assert spin is True
        return {
            "basis": {"irrep_labels": all_labels},
            "ebrs": [
                {"ebr_name": "EBR_GM_A", "vector": [
                    1 if l in gm[:gm_half] else 0 for l in all_labels
                ]},
                {"ebr_name": "EBR_GM_B", "vector": [
                    1 if l in gm[gm_half:] else 0 for l in all_labels
                ]},
                {"ebr_name": "EBR_X", "vector": [
                    1 if l in x else 0 for l in all_labels
                ]},
            ],
        }

    return _loader


def _make_p3_fake_ebr_loader(gm_labels=None, k_labels=None):
    """Fake irreptables EBR loader for SG 143 P3 spinor."""
    gm = gm_labels if gm_labels is not None else _P3_GM_LABELS
    k = k_labels if k_labels is not None else _P3_K_LABELS
    all_labels = list(gm) + list(k)

    def _loader(sg, spin):
        assert sg == 143
        assert spin is True
        return {
            "basis": {"irrep_labels": all_labels},
            "ebrs": [
                {"ebr_name": "EBR_GM", "vector": [
                    1 if l in gm else 0 for l in all_labels
                ]},
                {"ebr_name": "EBR_K", "vector": [
                    1 if l in k else 0 for l in all_labels
                ]},
            ],
        }

    return _loader


# -----------------------------------------------------------------------
# Synthetic P4 E2E: auto-canonical table → exact decomposition
# -----------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("irreptables"),
    reason="irreptables not installed",
)
class TestAutoCanonicalP4E2E:
    """Synthetic P4 E2E: canonical irrep labels → auto table → exact solve."""

    def test_auto_table_synthetic_p4_gm_only(self):
        from valleyscope.analysis.irreptables_runtime_table_builder import (
            build_auto_canonical_reduced_ebr_table,
        )

        table = build_auto_canonical_reduced_ebr_table(
            subspace_sg_number=75,
            spinor=True,
            bundle_irreps_by_kpoint={
                "GammaM": ["-GM5", "-GM6"],
            },
            expected_hsps=["GammaM"],
            subspace_group_candidate="P4",
            source_loader=_make_p4_fake_ebr_loader(),
        )

        assert table["subspace_group_candidate"] == "P4"
        assert table["expected_hsps"] == ["GammaM"]
        for irrep_key in table["irreps"]:
            assert irrep_key.startswith("GammaM:")
        assert len(table["irreps"]) == len(_P4_GM_LABELS)
        assert len(table["ebrs"]) == 2  # X EBR filtered to zero

        prov = table.get("provenance", {})
        assert prov.get("auto_canonical") is True
        assert prov.get("space_group_number") == 75
        assert prov.get("spinful") is True
        assert prov.get("valleyscope_reduction") == "sampled_hsp_valley_preserving"

    def test_auto_table_synthetic_p4_gm_and_x(self):
        from valleyscope.analysis.irreptables_runtime_table_builder import (
            build_auto_canonical_reduced_ebr_table,
        )

        table = build_auto_canonical_reduced_ebr_table(
            subspace_sg_number=75,
            spinor=True,
            bundle_irreps_by_kpoint={
                "GammaM": ["-GM5", "-GM6"],
                "XM": ["-X3"],
            },
            expected_hsps=["GammaM", "XM"],
            subspace_group_candidate="P4",
            source_loader=_make_p4_fake_ebr_loader(),
        )

        assert table["expected_hsps"] == ["GammaM", "XM"]
        irrep_keys = table["irreps"]
        gm_keys = [k for k in irrep_keys if k.startswith("GammaM:")]
        x_keys = [k for k in irrep_keys if k.startswith("XM:")]
        assert len(gm_keys) == len(_P4_GM_LABELS)
        assert len(x_keys) == len(_P4_X_LABELS)

    def test_auto_table_feeds_reduced_ebr_mapping_exact_solve(self, tmp_path):
        from valleyscope.analysis.irreptables_runtime_table_builder import (
            build_auto_canonical_reduced_ebr_table,
        )
        from valleyscope.analysis.reduced_ebr_mapping import (
            build_reduced_ebr_mapping, load_reduced_ebr_table,
        )

        table = build_auto_canonical_reduced_ebr_table(
            subspace_sg_number=75,
            spinor=True,
            bundle_irreps_by_kpoint={
                "GammaM": ["-GM5", "-GM6"],
            },
            expected_hsps=["GammaM"],
            subspace_group_candidate="P4",
            source_loader=_make_p4_fake_ebr_loader(),
        )

        table_path = tmp_path / "auto_table.json"
        table_path.write_text(json.dumps(table), encoding="utf-8")
        validated = load_reduced_ebr_table(table_path)
        assert validated["subspace_group_candidate"] == "P4"

        export_bundle = {
            "bundles": [{
                "bundle_id": "b_p4_gm",
                "valley": "K_valley",
                "subspace_group_candidate": "P4",
                "subspace_space_group": {
                    "status": "resolved",
                    "candidate_space_group_number": 75,
                    "candidate_space_group_symbol": "P4",
                },
                "ready_for_external_solver": True,
                "expected_hsps": ["GammaM"],
                "irreps_by_kpoint": {
                    "GammaM": ["-GM5", "-GM6"],
                },
            }],
        }

        result = build_reduced_ebr_mapping(
            ebr_export_bundle=export_bundle,
            table=validated,
        )
        assert result["mapping_status"] == "solved_exact"
        sol = result["solutions"][0]
        assert sol["classification"] == "atomic-compatible-candidate"
        assert sol["subspace_group_candidate"] == "P4"

    def test_auto_table_hsp_mismatch_excluded(self, tmp_path):
        from valleyscope.analysis.irreptables_runtime_table_builder import (
            build_auto_canonical_reduced_ebr_table,
        )
        from valleyscope.analysis.reduced_ebr_mapping import (
            build_reduced_ebr_mapping, load_reduced_ebr_table,
        )

        table = build_auto_canonical_reduced_ebr_table(
            subspace_sg_number=75,
            spinor=True,
            bundle_irreps_by_kpoint={
                "GammaM": ["-GM5"],
            },
            expected_hsps=["GammaM"],
            subspace_group_candidate="P4",
            source_loader=_make_p4_fake_ebr_loader(),
        )

        table_path = tmp_path / "auto_table_hsp.json"
        table_path.write_text(json.dumps(table), encoding="utf-8")
        validated = load_reduced_ebr_table(table_path)

        export_bundle = {
            "bundles": [{
                "bundle_id": "b_mismatch",
                "valley": "K_valley",
                "subspace_group_candidate": "P4",
                "ready_for_external_solver": True,
                "expected_hsps": ["GammaM", "XM"],
                "irreps_by_kpoint": {
                    "GammaM": ["-GM5"],
                    "XM": ["-X3"],
                },
            }],
        }

        result = build_reduced_ebr_mapping(
            ebr_export_bundle=export_bundle,
            table=validated,
        )
        assert len(result["solutions"]) == 0
        assert len(result["excluded_bundles"]) == 1
        assert "expected_hsps mismatch" in result["excluded_bundles"][0]["reason"]

    def test_conflicting_hsp_mapping_raises(self):
        from valleyscope.analysis.irreptables_runtime_table_builder import (
            build_auto_canonical_reduced_ebr_table,
        )

        with pytest.raises(ValueError, match="conflicting HSP mapping"):
            build_auto_canonical_reduced_ebr_table(
                subspace_sg_number=75,
                spinor=True,
                bundle_irreps_by_kpoint={
                    "GammaM": ["-GM5"],
                    "XM": ["-GM5"],
                },
                expected_hsps=["GammaM", "XM"],
                subspace_group_candidate="P4",
                source_loader=_make_p4_fake_ebr_loader(),
            )


# -----------------------------------------------------------------------
# Parent/subspace mismatch
# -----------------------------------------------------------------------

def test_auto_canonical_uses_subspace_sg_not_parent():
    """Auto-canonical table uses the bundle's subspace SG number,
    not the parent moire space group number."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    called_sg = []

    def _record_loader(sg, spin):
        called_sg.append(sg)
        gm_labels = ["-GM4", "-GM5", "-GM6"]
        k_labels = ["-K4", "-K5", "-K6"]
        all_labels = gm_labels + k_labels
        return {
            "basis": {"irrep_labels": all_labels},
            "ebrs": [
                {"ebr_name": "EBR_A", "vector": [1 if l in gm_labels else 0 for l in all_labels]},
                {"ebr_name": "EBR_B", "vector": [1 if l in k_labels else 0 for l in all_labels]},
            ],
        }

    build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=143,
        spinor=True,
        bundle_irreps_by_kpoint={
            "GammaM": ["-GM4"],
            "KM": ["-K5"],
        },
        expected_hsps=["GammaM", "KM"],
        subspace_group_candidate="P3",
        source_loader=_record_loader,
    )

    assert called_sg == [143], (
        f"auto table requested SG {called_sg}, expected [143] (P3), "
        f"not 150 (P321)"
    )


# -----------------------------------------------------------------------
# Spinful source selection
# -----------------------------------------------------------------------

def test_auto_canonical_spinor_flag_true():
    """Spinor=True passes True to the irreptables loader."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    called_spin = []

    def _record_loader(sg, spin):
        called_spin.append(spin)
        return {
            "basis": {"irrep_labels": ["-GM4", "-GM5", "-GM6"]},
            "ebrs": [{"ebr_name": "EBR_A", "vector": [1, 0, 0]}],
        }

    build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=143,
        spinor=True,
        bundle_irreps_by_kpoint={"GammaM": ["-GM4"]},
        expected_hsps=["GammaM"],
        subspace_group_candidate="P3",
        source_loader=_record_loader,
    )
    assert called_spin == [True]


def test_auto_canonical_spinor_flag_false():
    """Spinor=False passes False to the irreptables loader."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    called_spin = []

    def _record_loader(sg, spin):
        called_spin.append(spin)
        return {
            "basis": {"irrep_labels": ["GM1", "GM2", "GM3"]},
            "ebrs": [{"ebr_name": "EBR_A", "vector": [1, 0, 0]}],
        }

    build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=143,
        spinor=False,
        bundle_irreps_by_kpoint={"GammaM": ["GM1"]},
        expected_hsps=["GammaM"],
        subspace_group_candidate="P3",
        source_loader=_record_loader,
    )
    assert called_spin == [False]


# -----------------------------------------------------------------------
# Ordered basis equality
# -----------------------------------------------------------------------

def test_auto_canonical_irrep_keys_are_ordered_and_unique():
    """Auto-canonical table irrep keys must be ordered and unique."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    table = build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=75,
        spinor=True,
        bundle_irreps_by_kpoint={"GammaM": ["-GM5"]},
        expected_hsps=["GammaM"],
        subspace_group_candidate="P4",
        source_loader=_make_p4_fake_ebr_loader(),
    )

    irrep_keys = table["irreps"]
    assert len(irrep_keys) == len(set(irrep_keys)), "irrep keys must be unique"
    assert irrep_keys == sorted(irrep_keys), "irrep keys must be sorted"
    for key in irrep_keys:
        assert key.startswith("GammaM:"), f"unexpected irrep key: {key!r}"


def test_auto_table_irrep_basis_matches_bundle_exactly():
    """The reduced table irrep basis matches bundle irreps_by_kpoint exactly."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    # Use only the subset of labels that appear in the bundle as the
    # complete EBR basis (2 labels instead of 4).
    gm_labels = ["-GM5", "-GM6"]
    table = build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=75,
        spinor=True,
        bundle_irreps_by_kpoint={"GammaM": gm_labels},
        expected_hsps=["GammaM"],
        subspace_group_candidate="P4",
        source_loader=_make_p4_fake_ebr_loader(gm_labels=gm_labels, x_labels=[]),
    )

    export_bundle = {
        "bundles": [{
            "bundle_id": "b_exact",
            "valley": "K",
            "subspace_group_candidate": "P4",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM"],
            "irreps_by_kpoint": {"GammaM": gm_labels},
        }],
    }
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle, table=table,
    )
    assert result["mapping_status"] == "solved_exact"
    assert result["solutions"][0]["irrep_vector"] == [1, 1]


# -----------------------------------------------------------------------
# Missing / ambiguous labels block
# -----------------------------------------------------------------------

def test_auto_canonical_blocks_when_label_not_in_irreptables_table():
    """Labels not found in the irreptables irrep table raise an error."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    with pytest.raises(ValueError, match="could not derive Bilbao"):
        build_auto_canonical_reduced_ebr_table(
            subspace_sg_number=75,
            spinor=True,
            bundle_irreps_by_kpoint={"GammaM": ["-ZZ99"]},
            expected_hsps=["GammaM"],
            subspace_group_candidate="P4",
            source_loader=lambda sg, spin: {
                "basis": {"irrep_labels": ["-ZZ99"]},
                "ebrs": [{"ebr_name": "EBR_A", "vector": [1]}],
            },
        )


def test_auto_canonical_blocks_when_no_labels_map_to_expected_hsps():
    """If no irreptables labels map to expected HSPs, clear error raised."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    with pytest.raises(ValueError, match="no irreptables EBR basis labels map"):
        build_auto_canonical_reduced_ebr_table(
            subspace_sg_number=75,
            spinor=True,
            bundle_irreps_by_kpoint={
                "GammaM": ["-GM5"],
            },
            expected_hsps=["XM"],
            subspace_group_candidate="P4",
            source_loader=_make_p4_fake_ebr_loader(),
        )


# -----------------------------------------------------------------------
# Multiple exact decompositions
# -----------------------------------------------------------------------

def test_auto_canonical_multiple_exact_solutions_reported():
    """Multiple EBR vectors with identical patterns — first exact solution found."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    gm_labels = ["-GM5", "-GM6"]
    table = build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=75,
        spinor=True,
        bundle_irreps_by_kpoint={"GammaM": gm_labels},
        expected_hsps=["GammaM"],
        subspace_group_candidate="P4",
        source_loader=lambda sg, spin: {
            "basis": {"irrep_labels": gm_labels},
            "ebrs": [
                {"ebr_name": "EBR_A", "vector": [1, 0]},
                {"ebr_name": "EBR_B", "vector": [1, 0]},
            ],
        },
    )

    export_bundle = {
        "bundles": [{
            "bundle_id": "b_multi",
            "valley": "K",
            "subspace_group_candidate": "P4",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM"],
            "irreps_by_kpoint": {"GammaM": [gm_labels[0]]},
        }],
    }
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle, table=table,
    )
    assert result["mapping_status"] == "solved_exact"
    sol = result["solutions"][0]
    assert sol["classification"] == "atomic-compatible-candidate"


# -----------------------------------------------------------------------
# Disabled default
# -----------------------------------------------------------------------

def test_disabled_reduced_ebr_produces_no_auto_canonical_artifacts(tmp_path):
    """When reduced_ebr is disabled, no auto-canonical table is built."""
    from tests.helpers_io_workflow import write_fixture, write_config

    h5_path = tmp_path / "wf.h5"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    config_path = tmp_path / "cfg.yaml"
    write_config(config_path, h5_path, out_dir)

    outputs = analyze_hsp(config_path)

    assert not (out_dir / "valley_reduced_ebr_mapping.json").exists()
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "valley_reduced_ebr_mapping" not in summary


# -----------------------------------------------------------------------
# No Cn-like logic
# -----------------------------------------------------------------------

def test_auto_canonical_table_contains_no_cn_like_labels():
    """Auto-canonical table provenance must not contain Cn-like labels."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    table = build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=75,
        spinor=True,
        bundle_irreps_by_kpoint={"GammaM": ["-GM5"]},
        expected_hsps=["GammaM"],
        subspace_group_candidate="P4",
        source_loader=_make_p4_fake_ebr_loader(gm_labels=["-GM5", "-GM6"], x_labels=[]),
    )

    raw = json.dumps(table)
    for cn in ("C2_like", "C3_like", "C4_like"):
        assert cn not in raw, f"{cn} must not appear in auto-canonical table"

    assert "P4" in raw
    prov = table.get("provenance", {})
    assert prov.get("auto_canonical") is True
