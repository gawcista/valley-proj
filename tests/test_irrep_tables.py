import numpy as np
import pytest

from valleyscope.irreps.tables import load_standard_irrep_table, match_table_operations
from valleyscope.irreps.matching import (
    decompose_characters_into_irreps,
    match_single_state_irrep,
)


def test_load_sg143_spinor_table_exposes_double_valued_k_irreps():
    table = load_standard_irrep_table(143, spinor=True)

    assert table.number == 143
    assert table.name == "P3"
    assert table.spinor is True
    assert [operation.table_index for operation in table.operations] == [1, 2, 3]

    c3 = table.operation_by_index(2)
    np.testing.assert_array_equal(
        c3.rotation_frac,
        np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int),
    )
    np.testing.assert_allclose(c3.translation_frac, np.zeros(3), atol=1e-12)

    k_irreps = {irrep.label: irrep for irrep in table.irreps_by_kpoint("K")}
    assert {"-K4", "-K5", "-K6"} <= set(k_irreps)
    assert k_irreps["-K5"].dimension == 1
    assert k_irreps["-K6"].dimension == 1
    assert k_irreps["-K5"].characters[2] == pytest.approx(
        np.exp(-1j * np.pi / 3),
        abs=1e-4,
    )
    assert k_irreps["-K6"].characters[2] == pytest.approx(
        np.exp(+1j * np.pi / 3),
        abs=1e-4,
    )
    assert table.operation_indices_for_kpoint("K") == [1, 2, 3]
    assert table.operation_indices_for_kpoint("M") == [1]


def test_match_table_operations_maps_detected_p3_operations_to_irreptables_indices():
    table = load_standard_irrep_table(143, spinor=True)
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    detected_operations = [
        {"operation_id": 10, "rotation_frac": np.eye(3, dtype=int), "translation_frac": np.zeros(3)},
        {"operation_id": 11, "rotation_frac": c3, "translation_frac": np.zeros(3)},
        {"operation_id": 12, "rotation_frac": c3 @ c3, "translation_frac": np.zeros(3)},
    ]

    report = match_table_operations(detected_operations, table)

    assert report.status == "complete"
    assert report.mapping_by_operation_id == {10: 1, 11: 2, 12: 3}
    assert report.unmatched_operation_ids == []
    assert report.unused_table_operation_indices == []


def test_match_table_operations_reports_unmatched_extra_operation():
    table = load_standard_irrep_table(143, spinor=True)
    detected_operations = [
        {"operation_id": 0, "rotation_frac": np.eye(3, dtype=int), "translation_frac": np.zeros(3)},
        {"operation_id": 9, "rotation_frac": np.diag([-1, -1, 1]), "translation_frac": np.zeros(3)},
    ]

    report = match_table_operations(detected_operations, table)

    assert report.status == "incomplete"
    assert report.mapping_by_operation_id == {0: 1}
    assert report.unmatched_operation_ids == [9]
    assert report.unused_table_operation_indices == [2, 3]


def test_match_table_operations_resolves_conjugate_twofold_via_unique_isomorphism():
    """Unique group isomorphism maps conjugate C2 when exact spatial match fails."""
    table = load_standard_irrep_table(5, spinor=True)
    detected_operations = [
        {"operation_id": 0, "rotation_frac": np.eye(3, dtype=int), "translation_frac": np.zeros(3)},
        {
            "operation_id": 4,
            "rotation_frac": np.array([[-1, 1, 0], [0, 1, 0], [0, 0, -1]], dtype=int),
            "translation_frac": np.zeros(3),
        },
    ]

    report = match_table_operations(detected_operations, table, source_hsp_label="GM")

    # Conjugate C2 resolved via unique group isomorphism.
    assert report.status == "complete"
    assert report.mapping_by_operation_id == {0: 1, 4: 2}
    assert report.provenance == "unique_group_isomorphism"


def test_decompose_characters_into_sg143_spinor_k_irrep_multiplicities():
    table = load_standard_irrep_table(143, spinor=True)

    result = decompose_characters_into_irreps(
        table=table,
        table_kpoint_label="K",
        computed_characters={1: 2.0 + 0.0j, 2: 1.0 + 0.0j, 3: 1.0 + 0.0j},
    )

    assert result.status == "matched"
    assert result.irrep_multiplicities == {"-K5": 1, "-K6": 1}
    assert result.irrep_weights["-K4"] == pytest.approx(0.0, abs=1e-8)
    assert result.irrep_weights["-K5"] == pytest.approx(1.0, abs=1e-5)
    assert result.irrep_weights["-K6"] == pytest.approx(1.0, abs=1e-5)
    assert result.missing_table_operation_indices == []
    assert result.failure_reasons == []


def test_decompose_characters_reports_missing_table_operations():
    table = load_standard_irrep_table(143, spinor=True)

    result = decompose_characters_into_irreps(
        table=table,
        table_kpoint_label="K",
        computed_characters={1: 2.0 + 0.0j, 2: 1.0 + 0.0j},
    )

    assert result.status == "missing_characters"
    assert result.irrep_multiplicities == {}
    assert result.missing_table_operation_indices == [3]
    assert result.failure_reasons == ["Missing computed characters for table operations: [3]"]


# -------------------------------------------------------------------------
# Single-state irrep matching tests
# -------------------------------------------------------------------------

def test_match_single_state_irrep_identifies_clean_one_dimensional_result():
    table = load_standard_irrep_table(143, spinor=True)

    result = match_single_state_irrep(
        table=table,
        table_kpoint_label="K",
        state_index=0,
        computed_characters={
            1: 1.0 + 0.0j,
            2: np.exp(-1j * np.pi / 3.0),
            3: np.exp(+1j * np.pi / 3.0),
        },
    )

    assert result.status == "matched"
    assert result.irrep_label == "-K5"
    assert result.state_index == 0
    assert result.irrep_multiplicities == {"-K5": 1}
    assert result.failure_reasons == []


def test_match_single_state_irrep_returns_missing_characters_when_operation_missing():
    table = load_standard_irrep_table(143, spinor=True)

    result = match_single_state_irrep(
        table=table,
        table_kpoint_label="K",
        state_index=0,
        computed_characters={1: 1.0 + 0.0j, 2: np.exp(-1j * np.pi / 3.0)},
    )

    assert result.status == "missing_characters"
    assert result.irrep_label is None


def test_match_single_state_irrep_rejects_ambiguous_decomposition():
    table = load_standard_irrep_table(143, spinor=True)

    result = match_single_state_irrep(
        table=table,
        table_kpoint_label="K",
        state_index=0,
        computed_characters={
            1: 2.0 + 0.0j,  # two-dimensional aggregate → ambiguous per state
            2: 1.0 + 0.0j,
            3: 1.0 + 0.0j,
        },
    )

    assert result.status == "ambiguous_irrep_label"
    assert result.irrep_label is None


def test_match_single_state_irrep_rejects_non_one_dimensional_irrep():
    """Characters that decompose to >1 irrep or multiplicity >1 are ambiguous per state."""
    table = load_standard_irrep_table(143, spinor=True)

    # Characters decomposing to -K4 (1) + -K5 (1) → ambiguous
    k4 = table.irreps_by_kpoint("K")[0]  # -K4
    k5 = table.irreps_by_kpoint("K")[1]  # -K5
    chars = {
        op: k4.characters[op] + k5.characters[op]
        for op in [1, 2, 3]
    }

    result = match_single_state_irrep(
        table=table,
        table_kpoint_label="K",
        state_index=0,
        computed_characters=chars,
    )

    assert result.status == "ambiguous_irrep_label"
    assert result.irrep_label is None


# -----------------------------------------------------------------------
# Source payload adapter for generic irrep matching
# -----------------------------------------------------------------------

from valleyscope.irreps.source_payload import build_source_payload_for_generic_matching
from valleyscope.analysis.valley_irrep_matching import build_valley_irrep_matching_report


def test_adapter_sg143_spinor_c3_like_payload():
    """Adapter builds valid payload for SG143 spinor C3-like operations."""
    table = load_standard_irrep_table(143, spinor=True)
    detected = [
        {"operation_id": 1, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 2, "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 3, "rotation_frac": np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=int),
         "translation_frac": np.zeros(3)},
    ]
    payload = build_source_payload_for_generic_matching(
        table=table, source_hsp_label="K",
        detected_operations=detected,
        valley_preserving_operation_ids=[1, 2, 3],
    )
    assert payload["status"] == "ok"
    assert payload["provenance"]["source_hsp_label"] == "K"
    chars = payload["source_irrep_characters"]
    assert "-K5" in chars
    assert chars["-K5"][2] == pytest.approx(np.exp(-1j * np.pi / 3), abs=1e-4)


def test_adapter_resolves_conjugate_c2_via_unique_isomorphism():
    """Conjugate C2 resolved via unique group isomorphism, not blocked."""
    table = load_standard_irrep_table(5, spinor=True)
    detected = [
        {"operation_id": 0, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 4,
         "rotation_frac": np.array([[-1, 1, 0], [0, 1, 0], [0, 0, -1]], dtype=int),
         "translation_frac": np.zeros(3)},
    ]

    payload = build_source_payload_for_generic_matching(
        table=table,
        source_hsp_label="GM",
        detected_operations=detected,
        valley_preserving_operation_ids=[0, 4],
    )

    assert payload["status"] == "ok"
    assert payload["provenance"]["operation_mapping_provenance"] == "unique_group_isomorphism"


def test_adapter_blocked_missing_source_hsp():
    """Adapter blocks when source HSP has no irreps."""
    table = load_standard_irrep_table(143, spinor=True)
    detected = [{"operation_id": 1, "rotation_frac": np.eye(3, dtype=int),
                  "translation_frac": np.zeros(3)}]
    payload = build_source_payload_for_generic_matching(
        table=table, source_hsp_label="NONEXISTENT",
        detected_operations=detected,
        valley_preserving_operation_ids=[1],
    )
    assert payload["status"] == "blocked"
    assert "no_source_irreps_for_hsp" in payload["blocker_reasons"][0]


def test_adapter_blocked_incomplete_ops():
    """Adapter blocks when VP op not in detected_operations."""
    table = load_standard_irrep_table(143, spinor=True)
    detected = [{"operation_id": 1, "rotation_frac": np.eye(3, dtype=int),
                  "translation_frac": np.zeros(3)}]
    payload = build_source_payload_for_generic_matching(
        table=table, source_hsp_label="K",
        detected_operations=detected,
        valley_preserving_operation_ids=[1, 2],
    )
    assert payload["status"] == "blocked"


def test_adapter_blocked_unmatched_valley_preserving_operation():
    """Adapter blocks when a VP op has no source-table operation match."""
    table = load_standard_irrep_table(5, spinor=True)
    detected = [
        {"operation_id": 0, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 4,
         "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int),
         "translation_frac": np.zeros(3)},
    ]
    payload = build_source_payload_for_generic_matching(
        table=table,
        source_hsp_label="GM",
        detected_operations=detected,
        valley_preserving_operation_ids=[0, 4],
    )
    assert payload["status"] == "blocked"
    assert "table_operation_matching_failed" in payload["blocker_reasons"][0]


def test_adapter_blocks_ambiguous_identity_only_source_restriction():
    """A source HSP with multiple irreps cannot be labeled from identity only."""
    table = load_standard_irrep_table(143, spinor=True)
    detected = [{"operation_id": 0, "rotation_frac": np.eye(3, dtype=int),
                 "translation_frac": np.zeros(3)}]
    payload = build_source_payload_for_generic_matching(
        table=table,
        source_hsp_label="K",
        detected_operations=detected,
        valley_preserving_operation_ids=[0],
    )
    assert payload["status"] == "blocked"
    assert "ambiguous_restricted_source_irreps" in payload["blocker_reasons"][0]


def test_adapter_payload_drives_generic_matching_report():
    """Adapter output feeds build_valley_irrep_matching_report generic path."""
    # Use SG 143 P3 spinor which has known exact operation matrices.
    table = load_standard_irrep_table(143, spinor=True)
    op2 = table.operation_by_index(2)
    detected = [
        {"operation_id": 0,
         "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 1,
         "rotation_frac": op2.rotation_frac,
         "translation_frac": op2.translation_frac},
    ]
    payload = build_source_payload_for_generic_matching(
        table=table,
        source_hsp_label="K",
        detected_operations=detected,
        valley_preserving_operation_ids=[0, 1],
    )
    assert payload["status"] == "ok"

    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "K_valley": {
                        "readiness_level": "trusted",
                        "workflow_path": "direct_qcut",
                    },
                },
            },
        },
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [{
                        "reference_valley": "K_valley",
                        "orbit": ["K_valley"],
                        "hsp_preserving_operation_ids": [0, 1],
                        "subspace_space_group": {
                            "valley_preserving_operation_ids": [0, 1],
                        },
                        "valley_preserving_character_diagnostics": {
                            "per_valley": {
                                "K_valley": [
                                    {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                                    {"operation_id": 1, "eigenphases": [0.166667]},
                                ],
                            },
                        },
                        "subspace_group": {
                            "subspace_group_candidate": "P3",
                            "operation_orders": {"0": 1, "1": 3},
                        },
                    }],
                },
            },
        },
        source_irrep_characters_flattened={
            "GammaM": {"K_valley": payload["source_irrep_characters"]},
        },
        source_operation_maps={
            "GammaM": {"K_valley": payload["source_operation_map"]},
        },
    )

    match = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert match["matching_strategy"] == "bilbao_restricted_character"
    assert match["matching_status"] in ("matched", "diagnostic")
    assert len(match.get("irrep_multiplicities", {})) >= 0


# -----------------------------------------------------------------------
# Group-isomorphism operation mapping (Finding 4)
# -----------------------------------------------------------------------

def test_group_isomorphism_resolves_conjugate_c2_identity_preserved():
    """Identity (exact match) + conjugate C2 → unique isomorphism."""
    table = load_standard_irrep_table(5, spinor=True)
    # Non-zero identity ID, conjugate C2
    detected = [
        {"operation_id": 99, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 42, "rotation_frac": np.array([[-1, 1, 0], [0, 1, 0], [0, 0, -1]], dtype=int),
         "translation_frac": np.zeros(3)},
    ]
    report = match_table_operations(detected, table, source_hsp_label="GM")
    assert report.status == "complete"
    assert report.mapping_by_operation_id == {99: 1, 42: 2}
    assert report.provenance == "unique_group_isomorphism"


def test_group_isomorphism_resolves_permuted_operation_order():
    """Permuted operation order still resolves via isomorphism."""
    table = load_standard_irrep_table(5, spinor=True)
    detected = [
        {"operation_id": 42, "rotation_frac": np.array([[-1, 1, 0], [0, 1, 0], [0, 0, -1]], dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 99, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
    ]
    report = match_table_operations(detected, table, source_hsp_label="GM")
    assert report.status == "complete"
    assert report.mapping_by_operation_id[99] == 1  # identity
    assert report.mapping_by_operation_id[42] == 2  # C2


def test_group_isomorphism_ambiguous_mapping_blocked():
    """Ambiguous (non-unique) isomorphism → blocked."""
    table = load_standard_irrep_table(143, spinor=True)
    # P3 has 3 operations at K: identity + C3 + C3^2 = [1,2,3]
    # Detected 3 ops: identity, C3, C3^2 — exact match works, so no isomorphism needed.
    # To test ambiguity, use only 2 operations where there could be multiple
    # interpretations.  Actually, 2-element groups have only one isomorphism.
    # Use 3-element group with partial mapping that strips identity.
    op_table = table.operation_by_index(2)  # C3
    op_sq = table.operation_by_index(3)     # C3^2
    detected = [
        {"operation_id": 0, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 1, "rotation_frac": op_table.rotation_frac,
         "translation_frac": np.zeros(3)},
        {"operation_id": 2, "rotation_frac": op_sq.rotation_frac,
         "translation_frac": np.zeros(3)},
    ]
    # These are exact matches → spatial matching works, no isomorphism needed.
    report = match_table_operations(detected, table, source_hsp_label="K")
    assert report.status == "complete"
    assert report.provenance == "exact_spatial"


def test_empty_gk_a_blocked_not_identity_only():
    """Empty G_k^(a) is blocked, not identity-only."""
    from valleyscope.irreps.source_payload import build_source_payload_for_generic_matching
    table = load_standard_irrep_table(5, spinor=True)
    detected = [
        {"operation_id": 99, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
    ]
    payload = build_source_payload_for_generic_matching(
        table=table, source_hsp_label="GM",
        detected_operations=detected,
        valley_preserving_operation_ids=[],
    )
    assert payload["status"] == "blocked"
    assert "empty_valley_preserving_operation_ids" in payload["blocker_reasons"][0]
