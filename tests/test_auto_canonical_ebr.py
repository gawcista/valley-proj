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
from valleyscope.workflows.analyze_hsp import analyze_hsp, _build_auto_canonical_mapping


# -----------------------------------------------------------------------
# Unit: _build_auto_canonical_mapping
# -----------------------------------------------------------------------

def test_auto_canonical_mapping_returns_none_for_none_bundle():
    result = _build_auto_canonical_mapping(
        ebr_export_bundle=None, spinor_wf=True,
    )
    assert result is None


def test_auto_canonical_mapping_returns_none_for_empty_bundles():
    result = _build_auto_canonical_mapping(
        ebr_export_bundle={"bundles": []}, spinor_wf=True,
    )
    assert result is None


def test_auto_canonical_mapping_returns_none_for_no_subspace_sg():
    result = _build_auto_canonical_mapping(
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
    assert result is None


def test_auto_canonical_mapping_returns_none_for_no_irreps_by_kpoint():
    result = _build_auto_canonical_mapping(
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
    assert result is None


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
# Missing / ambiguous labels block (Finding 3 — strict resolution)
# -----------------------------------------------------------------------

def test_unknown_bundle_label_blocks():
    """One unknown canonical label alongside known labels blocks the table."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    with pytest.raises(ValueError, match="not found in irreptables irrep table"):
        build_auto_canonical_reduced_ebr_table(
            subspace_sg_number=75,
            spinor=True,
            # -GM5 is valid, -ZZ99 is not.
            bundle_irreps_by_kpoint={"GammaM": ["-GM5", "-ZZ99"]},
            expected_hsps=["GammaM"],
            subspace_group_candidate="P4",
            source_loader=_make_p4_fake_ebr_loader(),
        )


def test_unknown_ebr_basis_label_blocks():
    """An unknown EBR source-basis label blocks the table."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    with pytest.raises(ValueError, match="could not be resolved to a Bilbao kpoint"):
        build_auto_canonical_reduced_ebr_table(
            subspace_sg_number=75,
            spinor=True,
            bundle_irreps_by_kpoint={"GammaM": ["-GM5"]},
            expected_hsps=["GammaM"],
            subspace_group_candidate="P4",
            source_loader=lambda sg, spin: {
                "basis": {"irrep_labels": ["-GM5", "-UNKNOWN99"]},
                "ebrs": [
                    {"ebr_name": "EBR_A", "vector": [1, 0]},
                ],
            },
        )


def test_unsampled_hsp_labels_in_provenance():
    """Known labels at genuinely unsampled HSPs are tracked in provenance."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    # Bundle only declares GammaM labels.  X labels exist in the EBR
    # data at an unsampled HSP (XM not in expected_hsps).
    table = build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=75,
        spinor=True,
        bundle_irreps_by_kpoint={"GammaM": ["-GM5"]},
        expected_hsps=["GammaM"],
        subspace_group_candidate="P4",
        source_loader=_make_p4_fake_ebr_loader(),
    )

    prov = table.get("provenance", {})
    unsampled = prov.get("unsampled_hsp_labels", [])
    # X labels should appear as unsampled (Bilbao X maps to no ValleyScope HSP).
    assert len(unsampled) > 0, "unsampled HSP labels must be tracked"
    assert any("Bilbao X" in u for u in unsampled), (
        f"unsampled labels should mention Bilbao X: {unsampled}"
    )
    assert prov.get("unsampled_hsp_count", 0) == len(unsampled)


def test_conflicting_duplicate_label_hsp_raises():
    """Duplicate canonical label mapped to conflicting HSPs raises clear error."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    # -GM5 is a GM (Gamma) irrep.  Claiming it at GammaM and XM is a conflict.
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
# Missing / ambiguous labels block (legacy tests)
# -----------------------------------------------------------------------

def test_auto_canonical_blocks_when_label_not_in_irreptables_table():
    """Labels not found in the irreptables irrep table raise an error."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )

    with pytest.raises(ValueError, match="not found in irreptables irrep table"):
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

def test_auto_canonical_unique_exact_decomposition():
    """Unique exact nonnegative decomposition reports uniqueness correctly."""
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
                {"ebr_name": "EBR_B", "vector": [0, 1]},
            ],
        },
    )

    export_bundle = {
        "bundles": [{
            "bundle_id": "b_unique",
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
    assert sol["decomposition_uniqueness"] == "unique"
    assert "decomposition_witnesses" not in sol


def test_auto_canonical_non_unique_reports_witnesses():
    """Non-unique decomposition reports at least two witnesses."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    gm_labels = ["-GM5", "-GM6"]
    # Two EBRs with identical vectors → multiple ways to decompose.
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
            "bundle_id": "b_nonunique",
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
    assert sol["decomposition_uniqueness"] == "non_unique"
    # Must report at least two witnesses.
    witnesses = sol.get("decomposition_witnesses", [])
    assert len(witnesses) >= 2, (
        f"non_unique must report at least 2 witnesses, got {witnesses}"
    )


def test_truncated_search_reports_unknown_uniqueness():
    """Truncated search reports unknown_truncated, not fragile or unique."""
    from valleyscope.analysis.reduced_ebr_solver import classify_bundle

    # [100] = 100 × [1] is in integer span, but max_coefficient=0
    # truncates the search to coefficient 0, so no nonnegative solution
    # is found within the truncated bound.
    target = [100]
    ebr_vectors = [[1]]
    ebr_labels = ["EBR_unit"]

    result = classify_bundle(target, ebr_vectors, ebr_labels, max_coefficient=0)

    assert result["integer_span_status"] == "in_integer_span"
    assert result["classification"] == "indeterminate_truncated"
    assert result["decomposition_uniqueness"] == "unknown_truncated"
    assert "search_status" in result
    # Must not claim fragile.
    assert result["classification"] != "fragile-topology-candidate"


def test_truncated_search_finds_solution_reports_unknown():
    """Truncated search that finds solutions still reports unknown_truncated."""
    from valleyscope.analysis.reduced_ebr_solver import classify_bundle

    target = [2]
    ebr_vectors = [[1], [1]]  # can decompose with 2 = 2*EBR0 or 2*EBR1
    ebr_labels = ["EBR_A", "EBR_B"]

    # max_coefficient=2 is sufficient but physical bound is also 2.
    # Use a case where max_coefficient is small enough.
    result = classify_bundle(target, ebr_vectors, ebr_labels, max_coefficient=5)

    assert result["integer_span_status"] == "in_integer_span"
    assert result["classification"] == "atomic-compatible-candidate"
    assert result["decomposition_uniqueness"] == "non_unique"
    assert len(result.get("decomposition_witnesses", [])) >= 2


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

# -----------------------------------------------------------------------
# Irrep-key format validation (Finding 4)
# -----------------------------------------------------------------------

def test_irrep_key_regex_accepts_leading_minus():
    """Leading - for spinor labels is valid."""
    from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table
    import tempfile, json
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:-GM5", "GammaM:-GM6"],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0]}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        f.close()
        try:
            validated = load_reduced_ebr_table(f.name)
            assert validated["irreps"] == ["GammaM:-GM5", "GammaM:-GM6"]
        finally:
            import os; os.unlink(f.name)


def test_irrep_key_regex_accepts_non_spinor_labels():
    """Non-spinor labels without leading - are valid."""
    from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table
    import tempfile, json
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P3",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:GM1", "GammaM:GM2"],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0]}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        f.close()
        try:
            validated = load_reduced_ebr_table(f.name)
            assert validated["irreps"] == ["GammaM:GM1", "GammaM:GM2"]
        finally:
            import os; os.unlink(f.name)


def test_irrep_key_regex_rejects_digit_first_char():
    """Irrep label starting with digit after : is invalid."""
    from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table
    import tempfile, json, pytest
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:1GM"],
        "ebrs": [{"label": "EBR_A", "vector": [1]}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        f.close()
        try:
            with pytest.raises(ValueError, match="invalid irrep key format"):
                load_reduced_ebr_table(f.name)
        finally:
            import os; os.unlink(f.name)


def test_irrep_key_regex_rejects_plus_first_char():
    """Irrep label starting with + after : is invalid."""
    from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table
    import tempfile, json, pytest
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:+GM"],
        "ebrs": [{"label": "EBR_A", "vector": [1]}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        f.close()
        try:
            with pytest.raises(ValueError, match="invalid irrep key format"):
                load_reduced_ebr_table(f.name)
        finally:
            import os; os.unlink(f.name)


def test_irrep_key_regex_rejects_slash_first_char():
    """Irrep label starting with / after : is invalid."""
    from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table
    import tempfile, json, pytest
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:/GM"],
        "ebrs": [{"label": "EBR_A", "vector": [1]}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        f.close()
        try:
            with pytest.raises(ValueError, match="invalid irrep key format"):
                load_reduced_ebr_table(f.name)
        finally:
            import os; os.unlink(f.name)


def test_irrep_key_regex_accepts_minus_letter_label():
    """Leading - followed by letter is valid (spinor convention)."""
    from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table
    import tempfile, json
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P3",
        "expected_hsps": ["KM"],
        "irreps": ["KM:-K5", "KM:-K6"],
        "ebrs": [{"label": "EBR_K", "vector": [1, 0]}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        f.close()
        try:
            validated = load_reduced_ebr_table(f.name)
            assert len(validated["irreps"]) == 2
        finally:
            import os; os.unlink(f.name)


# -----------------------------------------------------------------------
# Multi-group auto-canonical (Finding 1)
# -----------------------------------------------------------------------

def test_two_bundles_same_sg_same_hsp_share_table():
    """Two ready bundles with same SG and HSP basis share one auto table."""
    result = _build_auto_canonical_mapping(
        ebr_export_bundle={
            "bundles": [
                {
                    "bundle_id": "b_001", "valley": "K_valley",
                    "subspace_group_candidate": "P4",
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                    },
                    "ready_for_external_solver": True,
                    "expected_hsps": ["GammaM"],
                    "irreps_by_kpoint": {"GammaM": ["-GM5"]},
                },
                {
                    "bundle_id": "b_002", "valley": "Kp_valley",
                    "subspace_group_candidate": "P4",
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                    },
                    "ready_for_external_solver": True,
                    "expected_hsps": ["GammaM"],
                    "irreps_by_kpoint": {"GammaM": ["-GM6"]},
                },
            ],
        },
        spinor_wf=True,
    )
    assert result is not None
    assert result["reduced_ebr_input"]["source"] == "auto_canonical"
    # One group, both bundles evaluated.
    groups = result.get("auto_canonical_groups", [])
    assert len(groups) == 1
    assert groups[0]["bundle_count"] == 2
    assert result["mapping_status"] == "solved_exact"


def test_two_bundles_different_sg_separate_groups():
    """Two bundles with different subspace SGs form separate table groups."""
    result = _build_auto_canonical_mapping(
        ebr_export_bundle={
            "bundles": [
                {
                    "bundle_id": "b_p4", "valley": "K_valley",
                    "subspace_group_candidate": "P4",
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                    },
                    "ready_for_external_solver": True,
                    "expected_hsps": ["GammaM"],
                    "irreps_by_kpoint": {"GammaM": ["-GM5"]},
                },
                {
                    "bundle_id": "b_p3", "valley": "K_valley",
                    "subspace_group_candidate": "P3",
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 143,
                        "candidate_space_group_symbol": "P3",
                    },
                    "ready_for_external_solver": True,
                    "expected_hsps": ["GammaM"],
                    "irreps_by_kpoint": {"GammaM": ["-GM4"]},
                },
            ],
        },
        spinor_wf=True,
    )
    assert result is not None
    groups = result.get("auto_canonical_groups", [])
    assert len(groups) == 2, f"expected 2 groups, got {groups}"
    sg_nums = {g["sg_number"] for g in groups}
    assert sg_nums == {75, 143}


def test_same_sg_incompatible_hsp_bases_separate_groups():
    """Same SG but different HSP bases form separate groups."""
    result = _build_auto_canonical_mapping(
        ebr_export_bundle={
            "bundles": [
                {
                    "bundle_id": "b_gm", "valley": "K_valley",
                    "subspace_group_candidate": "P4",
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                    },
                    "ready_for_external_solver": True,
                    "expected_hsps": ["GammaM"],
                    "irreps_by_kpoint": {"GammaM": ["-GM5"]},
                },
                {
                    "bundle_id": "b_gm_xm", "valley": "K_valley",
                    "subspace_group_candidate": "P4",
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                    },
                    "ready_for_external_solver": True,
                    "expected_hsps": ["GammaM", "XM"],
                    "irreps_by_kpoint": {
                        "GammaM": ["-GM5"],
                        "XM": ["-X3"],
                    },
                },
            ],
        },
        spinor_wf=True,
    )
    assert result is not None
    groups = result.get("auto_canonical_groups", [])
    assert len(groups) == 2, f"incompatible HSP bases must form separate groups"


def test_failed_group_preserves_global_status():
    """A failed group prevents global solved_exact."""
    # This bundle has an invalid SG number that will fail to load.
    result = _build_auto_canonical_mapping(
        ebr_export_bundle={
            "bundles": [
                {
                    "bundle_id": "b_ok", "valley": "K_valley",
                    "subspace_group_candidate": "P4",
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 75,
                        "candidate_space_group_symbol": "P4",
                    },
                    "ready_for_external_solver": True,
                    "expected_hsps": ["GammaM"],
                    "irreps_by_kpoint": {"GammaM": ["-GM5"]},
                },
                {
                    "bundle_id": "b_bad", "valley": "Kp_valley",
                    "subspace_group_candidate": "P3",
                    "subspace_space_group": {
                        "status": "resolved",
                        "candidate_space_group_number": 9999,
                        "candidate_space_group_symbol": "XX",
                    },
                    "ready_for_external_solver": True,
                    "expected_hsps": ["GammaM"],
                    "irreps_by_kpoint": {"GammaM": ["-GM4"]},
                },
            ],
        },
        spinor_wf=True,
    )
    assert result is not None
    # One group succeeded, one failed.
    groups = result.get("auto_canonical_groups", [])
    assert any(g["status"] == "auto_canonical_failed" for g in groups)
    # Global status must NOT be solved_exact when a group failed.
    assert result["mapping_status"] != "solved_exact"
    # The failed group's bundles are in excluded.
    excluded = result.get("excluded_bundles", [])
    assert any("b_bad" in str(e.get("bundle_id", "")) for e in excluded)


# -----------------------------------------------------------------------
# Explicit table/spec regression
# -----------------------------------------------------------------------

def test_explicit_table_file_behavior_unchanged(tmp_path):
    """Explicit table_file still works independently of auto-canonical."""
    from valleyscope.analysis.reduced_ebr_mapping import (
        build_reduced_ebr_mapping, load_reduced_ebr_table,
    )

    table_def = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:-GM5", "GammaM:-GM6"],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0]}],
    }
    table_path = tmp_path / "explicit.json"
    table_path.write_text(json.dumps(table_def), encoding="utf-8")
    loaded = load_reduced_ebr_table(table_path)

    export_bundle = {
        "bundles": [{
            "bundle_id": "b_explicit",
            "valley": "K_valley",
            "subspace_group_candidate": "P4",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM"],
            "irreps_by_kpoint": {"GammaM": ["-GM5"]},
        }],
    }
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=loaded,
        reduced_ebr_input={"source": "table_file", "table_file_stem": "explicit"},
    )
    assert result["mapping_status"] == "solved_exact"


def test_explicit_spec_file_behavior_unchanged(tmp_path):
    """Explicit spec_file still works independently of auto-canonical."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_reduced_table_from_spec_file,
    )
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping

    spec = {
        "schema_version": "1.1.0",
        "data_source": "irreptables",
        "space_group_number": 75,
        "spinful": True,
        "source_hsp_by_irrep": {"-GM5": "GammaM", "-GM6": "GammaM"},
        "valleyscope_irrep_multiplicity_by_source_irrep": {
            "-GM5": {"GammaM:-GM5": 1},
            "-GM6": {"GammaM:-GM6": 1},
        },
        "expected_hsps": ["GammaM"],
        "allowed_irrep_keys": ["GammaM:-GM5", "GammaM:-GM6"],
        "subspace_group_candidate": "P4",
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    # Fake loader must return only the labels declared in the spec.
    table = build_reduced_table_from_spec_file(
        str(spec_path),
        source_loader=_make_p4_fake_ebr_loader(
            gm_labels=["-GM5", "-GM6"], x_labels=[],
        ),
    )

    export_bundle = {
        "bundles": [{
            "bundle_id": "b_spec",
            "valley": "K_valley",
            "subspace_group_candidate": "P4",
            "ready_for_external_solver": True,
            "expected_hsps": ["GammaM"],
            "irreps_by_kpoint": {"GammaM": ["-GM5"]},
        }],
    }
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=table,
        reduced_ebr_input={"source": "spec_file", "spec_file_stem": "spec"},
    )
    assert result["mapping_status"] == "solved_exact"


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
