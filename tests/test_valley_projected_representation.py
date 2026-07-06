from valleyscope.analysis.valley_projected_representation import (
    build_valley_projected_representation_report,
)


def test_resolved_matching_group_overlays_unresolved_subspace_report():
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM"],
        valley_names=["K_valley"],
        symmetry_eigenvalue_rows=[{
            "kpoint": "GammaM",
            "target_valley": "K_valley",
            "operation_id": 1,
            "order": 3,
            "diagnostic_only": False,
            "topology_input_ready": True,
            "rotation_ready": True,
        }],
        symmetry_adapted_valley_report={"by_kpoint": {"GammaM": {
            "valley_preserving_subspaces": [{
                "orbit": ["K_valley"],
                "hsp_preserving_operation_ids": [0, 1, 2],
                "subspace_space_group": {
                    "status": "unresolved",
                    "candidate_space_group_symbol": None,
                    "valley_preserving_operation_ids": [0, 1, 2],
                    "valley_changing_operation_ids": [3, 4, 5],
                },
            }],
        }}},
        irrep_workflow_decisions={"by_kpoint": {"GammaM": {
            "K_valley": {
                "readiness_level": "trusted",
                "workflow_path": "direct_qcut",
            },
        }}},
        valley_irrep_matching={"generic_matches_by_kpoint": {"GammaM": {
            "K_valley": {
                "matching_status": "matched",
                "matching_strategy": "bilbao_restricted_character",
                "irrep_multiplicities": {"-GM4": 1},
                "subspace_space_group": {
                    "status": "resolved",
                    "candidate_space_group_number": 143,
                    "candidate_space_group_symbol": "P3",
                    "valley_preserving_operation_ids": [0, 1, 2],
                },
            },
        }}},
    )

    row = report["rows"][0]
    assert row["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
    assert row["subspace_space_group"]["candidate_space_group_number"] == 143
    assert row["valley_sewing_operation_ids"] == [3, 4, 5]
    assert report["representation_records"][0]["subspace_space_group"][
        "candidate_space_group_symbol"
    ] == "P3"


def test_representation_report_uses_subspace_space_group_as_primary():
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM"],
        valley_names=["M1_valley"],
        symmetry_eigenvalue_rows=[
            {
                "kpoint": "GammaM",
                "target_valley": "M1_valley",
                "operation_id": 4,
                "order": 2,
                "state_index": 0,
                "eigenvalue_real": -1.0,
                "eigenvalue_imag": 0.0,
                "phase_2pi": 0.5,
                "character_raw": "-1.000000+0.000000j",
                "character_valley": "-1.000000+0.000000j",
                "diagnostic_only": False,
                "topology_input_ready": True,
                "rotation_ready": True,
            }
        ],
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [
                        {
                            "orbit": ["M1_valley"],
                            "hsp_preserving_operation_ids": [0, 4],
                            "subspace_space_group": {
                                "candidate_space_group_symbol": "P2",
                                "candidate_space_group_number": 3,
                                "valley_preserving_operation_ids": [0, 4],
                                "valley_changing_operation_ids": [5],
                                "status": "candidate",
                            },
                            "subspace_group": {
                                "subspace_group_candidate": "P2",
                            },
                        }
                    ],
                }
            }
        },
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "M1_valley": {
                        "readiness_level": "trusted",
                        "workflow_path": "direct_qcut",
                    },
                }
            }
        },
    )

    # Per-row fields.
    row = report["rows"][0]
    assert row["subspace_space_group"]["candidate_space_group_symbol"] == "P2"
    assert row["hsp_little_group_operation_ids"] == [0, 4]
    assert row["subspace_hsp_little_group_operation_ids"] == [0, 4]
    assert row["valley_preserving_operation_ids"] == [0, 4]
    assert row["valley_sewing_operation_ids"] == [5]
    # removed
    assert report["subspace_space_group_counts"] == {"P2": 1}
    # removed

    # Grouped representation records.
    assert report["grouped_record_count"] == 1
    recs = report["representation_records"]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["kpoint"] == "GammaM"
    assert rec["valley"] == "M1_valley"
    assert rec["subspace_space_group"]["candidate_space_group_symbol"] == "P2"
    assert rec["hsp_little_group_operation_ids"] == [0, 4]
    assert rec["subspace_hsp_little_group_operation_ids"] == [0, 4]
    assert rec["valley_preserving_operation_ids"] == [0, 4]
    assert rec["valley_sewing_operation_ids"] == [5]
    assert rec["readiness_level"] == "trusted"
    assert rec["workflow_path"] == "direct_qcut"
    # removed
    assert len(rec["valley_preserving_operations"]) == 1
    op = rec["valley_preserving_operations"][0]
    assert op["operation_id"] == 4
    assert op["operation_order"] == 2
    assert op["character_valley"] == "-1.000000+0.000000j"
    assert op["eigenphases"] == [0.5]
    assert op["eigenvalues"] == [{"real": -1.0, "imag": 0.0}]
    assert op["source_row_count"] == 1
    assert op["diagnostic_only"] is False
    assert op["topology_input_ready"] is True


def test_representation_records_group_by_kpoint_valley():
    """Multiple rows for same (kpoint, valley) are grouped into one record."""
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM"],
        valley_names=["K_valley"],
        symmetry_eigenvalue_rows=[
            {
                "kpoint": "GammaM",
                "target_valley": "K_valley",
                "operation_id": 1,
                "order": 3,
                "diagnostic_only": False,
                "topology_input_ready": True,
                "rotation_ready": True,
            },
            {
                "kpoint": "GammaM",
                "target_valley": "K_valley",
                "operation_id": 2,
                "order": 3,
                "diagnostic_only": False,
                "topology_input_ready": True,
                "rotation_ready": True,
            },
        ],
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [
                        {
                            "orbit": ["K_valley"],
                            "hsp_preserving_operation_ids": [0, 1, 2],
                            "subspace_space_group": {
                                "candidate_space_group_symbol": "P3",
                                "valley_preserving_operation_ids": [0, 1, 2],
                                "valley_changing_operation_ids": [],
                                "status": "candidate",
                            },
                            "subspace_group": {
                                "subspace_group_candidate": "P3",
                            },
                        }
                    ],
                }
            }
        },
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "K_valley": {
                        "readiness_level": "trusted",
                        "workflow_path": "direct_qcut",
                    },
                }
            }
        },
    )

    assert report["grouped_record_count"] == 1
    rec = report["representation_records"][0]
    assert len(rec["valley_preserving_operations"]) == 2
    op_ids = {op["operation_id"] for op in rec["valley_preserving_operations"]}
    assert op_ids == {1, 2}


def test_representation_records_aggregate_rank_two_operation_data():
    """One operation with two eigenstate rows becomes one operation entry."""
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM"],
        valley_names=["M_valley"],
        symmetry_eigenvalue_rows=[
            {
                "kpoint": "GammaM",
                "target_valley": "M_valley",
                "operation_id": 4,
                "order": 2,
                "state_index": 1,
                "eigenvalue_real": 0.0,
                "eigenvalue_imag": 1.0,
                "phase_2pi": 0.25,
                "character_raw": "0.000000+0.000000j",
                "character_valley": "0.000000+0.000000j",
                "root_deviation": 2.0e-8,
                "basis": "valley_adapted",
                "diagnostic_only": False,
                "topology_input_ready": True,
                "rotation_ready": True,
            },
            {
                "kpoint": "GammaM",
                "target_valley": "M_valley",
                "operation_id": 4,
                "order": 2,
                "state_index": 0,
                "eigenvalue_real": 0.0,
                "eigenvalue_imag": -1.0,
                "phase_2pi": -0.25,
                "character_raw": "0.000000+0.000000j",
                "character_valley": "0.000000+0.000000j",
                "root_deviation": 1.0e-8,
                "basis": "valley_adapted",
                "diagnostic_only": False,
                "topology_input_ready": True,
                "rotation_ready": True,
            },
        ],
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [
                        {
                            "orbit": ["M_valley"],
                            "hsp_preserving_operation_ids": [0, 4],
                            "subspace_space_group": {
                                "candidate_space_group_symbol": "P2",
                                "candidate_space_group_number": 3,
                                "valley_preserving_operation_ids": [0, 4],
                                "valley_changing_operation_ids": [],
                                "status": "candidate",
                            },
                            "subspace_group": {
                                "subspace_group_candidate": "P2",
                            },
                        }
                    ],
                }
            }
        },
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "M_valley": {
                        "readiness_level": "trusted",
                        "workflow_path": "symmetry_adapted",
                    },
                }
            }
        },
    )

    rec = report["representation_records"][0]
    assert len(rec["valley_preserving_operations"]) == 1
    op = rec["valley_preserving_operations"][0]
    assert op["operation_id"] == 4
    assert op["source_row_count"] == 2
    assert op["character_valley"] == "0.000000+0.000000j"
    assert op["eigenphases"] == [-0.25, 0.25]
    assert op["eigenvalues"] == [
        {"real": 0.0, "imag": -1.0},
        {"real": 0.0, "imag": 1.0},
    ]
    assert op["max_root_deviation"] == 2.0e-8
    assert op["basis"] == "valley_adapted"


def test_representation_records_with_generic_irrep_matching():
    """Generic bilbao_restricted_character matching populates irrep_matching."""
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM"],
        valley_names=["K_valley"],
        symmetry_eigenvalue_rows=[
            {
                "kpoint": "GammaM",
                "target_valley": "K_valley",
                "operation_id": 1,
                "order": 3,
                "diagnostic_only": False,
                "topology_input_ready": True,
                "rotation_ready": True,
            },
        ],
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [
                        {
                            "orbit": ["K_valley"],
                            "hsp_preserving_operation_ids": [0, 1],
                            "subspace_space_group": {
                                "candidate_space_group_symbol": "P3",
                                "valley_preserving_operation_ids": [0, 1],
                                "valley_changing_operation_ids": [],
                                "status": "candidate",
                            },
                            "subspace_group": {
                                "subspace_group_candidate": "P3",
                            },
                        }
                    ],
                }
            }
        },
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "K_valley": {
                        "readiness_level": "trusted",
                        "workflow_path": "direct_qcut",
                    },
                }
            }
        },
        valley_irrep_matching={
            "generic_matches_by_kpoint": {
                "GammaM": {
                    "K_valley": {
                        "matching_status": "matched",
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": {"-GM6_a": 1},
                        "source_operation_map": {0: 1, 1: 2},
                        "diagnostic_only": False,
                    },
                },
            },
        },
    )

    rec = report["representation_records"][0]
    assert rec["irrep_matching"] is not None
    assert rec["irrep_matching"]["matching_status"] == "matched"
    assert rec["irrep_matching"]["matching_strategy"] == "bilbao_restricted_character"
    assert rec["irrep_matching"]["irrep_multiplicities"] == {"-GM6_a": 1}


def test_representation_records_p4_order4_group_agnostic():
    """P4/order-4 synthetic data: group-agnostic, non-C2, non-C3 proof."""
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM"],
        valley_names=["M_valley"],
        symmetry_eigenvalue_rows=[
            {
                "kpoint": "GammaM",
                "target_valley": "M_valley",
                "operation_id": 3,
                "order": 4,
                "diagnostic_only": False,
                "topology_input_ready": True,
                "rotation_ready": True,
            },
        ],
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [
                        {
                            "orbit": ["M_valley"],
                            "hsp_preserving_operation_ids": [0, 3],
                            "subspace_space_group": {
                                "candidate_space_group_symbol": "P4",
                                "candidate_space_group_number": 75,
                                "valley_preserving_operation_ids": [0, 3],
                                "valley_changing_operation_ids": [],
                                "status": "candidate",
                            },
                            "subspace_group": {
                                "subspace_group_candidate": "P4",
                            },
                        }
                    ],
                }
            }
        },
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "M_valley": {
                        "readiness_level": "trusted",
                        "workflow_path": "direct_qcut",
                    },
                }
            }
        },
        valley_irrep_matching={
            "generic_matches_by_kpoint": {
                "GammaM": {
                    "M_valley": {
                        "matching_status": "matched",
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": {"-GM_plus_1over4": 1},
                        "source_operation_map": {0: 1, 3: 2},
                        "diagnostic_only": False,
                    },
                },
            },
        },
    )

    assert report["subspace_space_group_counts"] == {"P4": 1}
    assert report["grouped_record_count"] == 1
    rec = report["representation_records"][0]
    assert rec["subspace_space_group"]["candidate_space_group_symbol"] == "P4"
    # removed
    assert rec["kpoint"] == "GammaM"
    assert rec["valley"] == "M_valley"
    assert rec["valley_preserving_operation_ids"] == [0, 3]
    assert rec["readiness_level"] == "trusted"
    assert rec["workflow_path"] == "direct_qcut"
    assert len(rec["valley_preserving_operations"]) == 1
    op = rec["valley_preserving_operations"][0]
    assert op["operation_id"] == 3
    assert op["operation_order"] == 4
    assert op["topology_input_ready"] is True
    assert rec["irrep_matching"] is not None
    assert rec["irrep_matching"]["matching_strategy"] == "bilbao_restricted_character"
    assert rec["irrep_matching"]["irrep_multiplicities"] == {"-GM_plus_1over4": 1}
    # Physical identifier is P4, not C4_like.
    assert rec["subspace_space_group"]["candidate_space_group_symbol"] != "C4_like"


# ---------------------------------------------------------------------------
# Public representation completeness tests (subspace-first semantics)
# ---------------------------------------------------------------------------

def _symmetry_analysis_stub(*, by_kpoint: dict) -> dict:
    """Minimal symmetry_analysis payload with valley_preserving_subgroup_report."""
    return {
        "valley_preserving_subgroup_report": {
            "by_kpoint": by_kpoint,
        },
    }


def test_blocked_identity_only_pair_produces_record_without_eigenvalue_rows():
    """MM with identity-only G_k^(a) and no eigenvalue rows still gets a record."""
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM", "MM"],
        valley_names=["K_valley", "Kp_valley"],
        symmetry_eigenvalue_rows=[
            {
                "kpoint": "GammaM", "target_valley": "K_valley",
                "operation_id": 1, "order": 3,
                "diagnostic_only": False, "topology_input_ready": True,
                "rotation_ready": True,
            },
        ],
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K_valley"],
                        "hsp_preserving_operation_ids": [0, 1, 2],
                        "subspace_space_group": {
                            "candidate_space_group_symbol": "P3",
                            "valley_preserving_operation_ids": [0, 1, 2],
                            "valley_changing_operation_ids": [3, 4, 5],
                            "status": "candidate",
                        },
                    }],
                },
                "MM": {
                    "valley_preserving_subspaces": [],
                },
            },
        },
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
                },
                "MM": {
                    "K_valley": {"readiness_level": "blocked", "workflow_path": "blocked"},
                    "Kp_valley": {"readiness_level": "blocked", "workflow_path": "blocked"},
                },
            },
        },
        valley_irrep_matching={
            "generic_matches_by_kpoint": {
                "GammaM": {
                    "K_valley": {
                        "matching_status": "matched",
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": {"-GM4": 1},
                        "subspace_space_group": {
                            "status": "resolved",
                            "candidate_space_group_number": 143,
                            "candidate_space_group_symbol": "P3",
                            "valley_preserving_operation_ids": [0, 1, 2],
                        },
                    },
                },
                "MM": {
                    "K_valley": {
                        "matching_status": "identity_only_not_irrep_distinguishing",
                        "matching_strategy": "bilbao_restricted_character",
                        "subspace_space_group": {
                            "status": "resolved",
                            "candidate_space_group_number": 143,
                            "candidate_space_group_symbol": "P3",
                            "valley_preserving_operation_ids": [0, 1, 2],
                        },
                    },
                },
            },
        },
        symmetry_analysis=_symmetry_analysis_stub(by_kpoint={
            "GammaM": {
                "K_valley": {
                    "little_group_operation_ids": [0, 1, 2, 3, 4, 5],
                    "allowed_operation_ids": [0, 1, 2],
                    "valley_changing_operation_ids": [3, 4, 5],
                    "identity_operation_id": 0,
                },
            },
            "MM": {
                "K_valley": {
                    "little_group_operation_ids": [0, 5],
                    "allowed_operation_ids": [0],
                    "valley_changing_operation_ids": [5],
                    "identity_operation_id": 0,
                },
                "Kp_valley": {
                    "little_group_operation_ids": [0, 5],
                    "allowed_operation_ids": [0],
                    "valley_changing_operation_ids": [5],
                    "identity_operation_id": 0,
                },
            },
        }),
    )

    # MM/K_valley must appear — no eigenvalue rows, but symmetry + workflow exist.
    mm_records = [
        r for r in report["representation_records"]
        if r["kpoint"] == "MM"
    ]
    assert len(mm_records) >= 1, "MM must produce a representation record"

    mm_k = [r for r in mm_records if r["valley"] == "K_valley"]
    assert len(mm_k) == 1
    mm = mm_k[0]
    # Public irrep group = subspace HSP little group G_k^(a).
    assert mm["subspace_hsp_little_group_operation_ids"] == [0]
    assert mm["hsp_little_group_operation_ids"] == [0]
    assert mm["valley_preserving_operation_ids"] == [0]
    # Parent provenance — NOT the irrep group.
    assert mm["parent_hsp_little_group_operation_ids"] == [0, 5]
    assert mm["valley_sewing_operation_ids"] == [5]
    assert mm["valley_preserving_operations"] == []
    assert mm["workflow_path"] == "blocked"
    assert mm["readiness_level"] == "blocked"
    assert any("identity_only" in b for b in mm["blocking_reasons"])
    assert mm["irrep_matching"] is not None
    assert mm["irrep_matching"]["matching_status"] == "identity_only_not_irrep_distinguishing"

    # GammaM record: public irrep group = [0,1,2], parent = [0,1,2,3,4,5].
    gm_records = [
        r for r in report["representation_records"]
        if r["kpoint"] == "GammaM"
    ]
    assert len(gm_records) == 1
    gm = gm_records[0]
    assert gm["subspace_hsp_little_group_operation_ids"] == [0, 1, 2]
    assert gm["hsp_little_group_operation_ids"] == [0, 1, 2]
    assert gm["parent_hsp_little_group_operation_ids"] == [0, 1, 2, 3, 4, 5]
    assert gm["valley_sewing_operation_ids"] == [3, 4, 5]

    # kpoint_labels includes all sampled kpoints.
    assert "MM" in report["kpoint_labels"]
    assert "GammaM" in report["kpoint_labels"]
    assert report["blocked_representation_count"] >= 2  # MM K + MM K'


def test_subspace_hsp_little_group_is_public_irrep_group():
    """Public irrep group = subspace G_k^(a); parent is provenance only."""
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM", "KM"],
        valley_names=["K_valley"],
        symmetry_eigenvalue_rows=[
            {
                "kpoint": "GammaM", "target_valley": "K_valley",
                "operation_id": 1, "order": 3,
                "diagnostic_only": False, "topology_input_ready": True,
                "rotation_ready": True,
            },
            {
                "kpoint": "KM", "target_valley": "K_valley",
                "operation_id": 1, "order": 3,
                "diagnostic_only": False, "topology_input_ready": True,
                "rotation_ready": True,
            },
        ],
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K_valley"],
                        "hsp_preserving_operation_ids": [0, 1, 2],
                        "subspace_space_group": {
                            "candidate_space_group_symbol": "P3",
                            "valley_preserving_operation_ids": [0, 1, 2],
                            "valley_changing_operation_ids": [3, 4, 5],
                            "status": "candidate",
                        },
                    }],
                },
                "KM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K_valley"],
                        "hsp_preserving_operation_ids": [0, 1, 2],
                        "subspace_space_group": {
                            "candidate_space_group_symbol": "P3",
                            "valley_preserving_operation_ids": [0, 1, 2],
                            "valley_changing_operation_ids": [3, 4, 5],
                            "status": "candidate",
                        },
                    }],
                },
            },
        },
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
                },
                "KM": {
                    "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
                },
            },
        },
        symmetry_analysis=_symmetry_analysis_stub(by_kpoint={
            "GammaM": {
                "K_valley": {
                    "little_group_operation_ids": [0, 1, 2, 3, 4, 5],
                    "allowed_operation_ids": [0, 1, 2],
                    "valley_changing_operation_ids": [3, 4, 5],
                    "identity_operation_id": 0,
                },
            },
            "KM": {
                "K_valley": {
                    "little_group_operation_ids": [0, 1, 2, 3, 4, 5],
                    "allowed_operation_ids": [0, 1, 2],
                    "valley_changing_operation_ids": [3, 4, 5],
                    "identity_operation_id": 0,
                },
            },
        }),
    )

    for rec in report["representation_records"]:
        # Public irrep group = subspace HSP little group G_k^(a).
        assert rec["subspace_hsp_little_group_operation_ids"] == [0, 1, 2], (
            f"{rec['kpoint']}: subspace HSP little group must be [0,1,2]"
        )
        assert rec["hsp_little_group_operation_ids"] == [0, 1, 2], (
            f"{rec['kpoint']}: legacy alias must match subspace value"
        )
        assert rec["valley_preserving_operation_ids"] == [0, 1, 2], (
            f"{rec['kpoint']}: valley_preserving = subspace HSP little group"
        )
        # Parent provenance — must NOT be the irrep group.
        assert rec["parent_hsp_little_group_operation_ids"] == [0, 1, 2, 3, 4, 5], (
            f"{rec['kpoint']}: parent full G_k is provenance only"
        )
        assert rec["valley_sewing_operation_ids"] == [3, 4, 5], (
            f"{rec['kpoint']}: valley-sewing ops are provenance only"
        )
        # Parent is a strict superset of the subspace HSP little group.
        assert set(rec["subspace_hsp_little_group_operation_ids"]).issubset(
            set(rec["parent_hsp_little_group_operation_ids"])
        )
        # Valley-sewing ops must not appear in the subspace irrep group.
        for op in rec["valley_sewing_operation_ids"]:
            assert op not in rec["subspace_hsp_little_group_operation_ids"], (
                f"{rec['kpoint']}: sewing op {op} must not be in "
                f"subspace irrep group"
            )


def test_identity_only_gka_no_false_irrep_no_ebr_candidate():
    """Identity-only G_k^(a) must not invent irrep labels or become EBR candidate."""
    report = build_valley_projected_representation_report(
        kpoint_names=["MM"],
        valley_names=["K_valley"],
        # No eigenvalue rows.
        symmetry_eigenvalue_rows=None,
        irrep_workflow_decisions={
            "by_kpoint": {
                "MM": {
                    "K_valley": {
                        "readiness_level": "blocked",
                        "workflow_path": "blocked",
                    },
                },
            },
        },
        valley_irrep_matching={
            "generic_matches_by_kpoint": {
                "MM": {
                    "K_valley": {
                        "matching_status": "identity_only_not_irrep_distinguishing",
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": None,
                        "subspace_space_group": {
                            "status": "resolved",
                            "candidate_space_group_number": 143,
                            "candidate_space_group_symbol": "P3",
                            "valley_preserving_operation_ids": [0, 1, 2],
                        },
                    },
                },
            },
        },
        symmetry_analysis=_symmetry_analysis_stub(by_kpoint={
            "MM": {
                "K_valley": {
                    "little_group_operation_ids": [0, 5],
                    "allowed_operation_ids": [0],
                    "valley_changing_operation_ids": [5],
                    "identity_operation_id": 0,
                },
            },
        }),
    )

    assert report["grouped_record_count"] == 1
    rec = report["representation_records"][0]
    assert rec["kpoint"] == "MM"
    assert rec["valley"] == "K_valley"

    # Public irrep group = G_k^(a) = [0].
    assert rec["subspace_hsp_little_group_operation_ids"] == [0]
    assert rec["hsp_little_group_operation_ids"] == [0]
    assert rec["valley_preserving_operation_ids"] == [0]
    # Parent provenance.
    assert rec["parent_hsp_little_group_operation_ids"] == [0, 5]
    assert rec["valley_sewing_operation_ids"] == [5]

    # Must NOT claim irrep labels.
    assert rec["valley_preserving_operations"] == []
    if rec["irrep_matching"]:
        status = rec["irrep_matching"].get("matching_status", "")
        mults = rec["irrep_matching"].get("irrep_multiplicities")
        # No positive irrep multiplicities.
        assert not mults or mults == {}, (
            f"identity-only must not have positive irrep multiplicities: {mults}"
        )

    # Must have blocking status — not trusted, not ready.
    assert rec["readiness_level"] != "trusted"
    assert rec["workflow_path"] == "blocked"
    assert any("identity_only" in b for b in rec["blocking_reasons"])

    # The physical subspace space group is still present.
    assert rec["subspace_space_group"]["candidate_space_group_symbol"] == "P3"


def test_target_kpoints_not_silently_dropped_from_records():
    """All target kpoints with sym/workflow data appear in representation_records."""
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM", "KM", "MM"],
        valley_names=["K_valley", "Kp_valley"],
        symmetry_eigenvalue_rows=[
            {
                "kpoint": "GammaM", "target_valley": "K_valley",
                "operation_id": 1, "order": 3,
                "diagnostic_only": False, "topology_input_ready": True,
                "rotation_ready": True,
            },
            {
                "kpoint": "KM", "target_valley": "K_valley",
                "operation_id": 1, "order": 3,
                "diagnostic_only": False, "topology_input_ready": True,
                "rotation_ready": True,
            },
        ],
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K_valley"],
                        "hsp_preserving_operation_ids": [0, 1, 2],
                        "subspace_space_group": {
                            "candidate_space_group_symbol": "P3",
                            "valley_preserving_operation_ids": [0, 1, 2],
                            "valley_changing_operation_ids": [3, 4, 5],
                            "status": "candidate",
                        },
                    }],
                },
                "KM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K_valley"],
                        "hsp_preserving_operation_ids": [0, 1, 2],
                        "subspace_space_group": {
                            "candidate_space_group_symbol": "P3",
                            "valley_preserving_operation_ids": [0, 1, 2],
                            "valley_changing_operation_ids": [3, 4, 5],
                            "status": "candidate",
                        },
                    }],
                },
            },
        },
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
                    "Kp_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
                },
                "KM": {
                    "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
                    "Kp_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
                },
                "MM": {
                    "K_valley": {"readiness_level": "blocked", "workflow_path": "blocked"},
                    "Kp_valley": {"readiness_level": "blocked", "workflow_path": "blocked"},
                },
            },
        },
        symmetry_analysis=_symmetry_analysis_stub(by_kpoint={
            "GammaM": {
                "K_valley": {
                    "little_group_operation_ids": [0, 1, 2, 3, 4, 5],
                    "allowed_operation_ids": [0, 1, 2],
                    "valley_changing_operation_ids": [3, 4, 5],
                    "identity_operation_id": 0,
                },
                "Kp_valley": {
                    "little_group_operation_ids": [0, 1, 2, 3, 4, 5],
                    "allowed_operation_ids": [0, 1, 2],
                    "valley_changing_operation_ids": [3, 4, 5],
                    "identity_operation_id": 0,
                },
            },
            "KM": {
                "K_valley": {
                    "little_group_operation_ids": [0, 1, 2, 3, 4, 5],
                    "allowed_operation_ids": [0, 1, 2],
                    "valley_changing_operation_ids": [3, 4, 5],
                    "identity_operation_id": 0,
                },
                "Kp_valley": {
                    "little_group_operation_ids": [0, 1, 2, 3, 4, 5],
                    "allowed_operation_ids": [0, 1, 2],
                    "valley_changing_operation_ids": [3, 4, 5],
                    "identity_operation_id": 0,
                },
            },
            "MM": {
                "K_valley": {
                    "little_group_operation_ids": [0, 5],
                    "allowed_operation_ids": [0],
                    "valley_changing_operation_ids": [5],
                    "identity_operation_id": 0,
                },
                "Kp_valley": {
                    "little_group_operation_ids": [0, 5],
                    "allowed_operation_ids": [0],
                    "valley_changing_operation_ids": [5],
                    "identity_operation_id": 0,
                },
            },
        }),
    )

    rec_kpoints = sorted(set(r["kpoint"] for r in report["representation_records"]))
    assert "MM" in rec_kpoints, "MM must not be silently dropped"
    assert "GammaM" in rec_kpoints
    assert "KM" in rec_kpoints
    assert sorted(report["kpoint_labels"]) == sorted(["GammaM", "KM", "MM"])

    # MM records: blocked, identity-only, subspace HSP little group = [0].
    mm_recs = [r for r in report["representation_records"] if r["kpoint"] == "MM"]
    assert len(mm_recs) == 2  # K_valley, Kp_valley
    for mm in mm_recs:
        assert mm["subspace_hsp_little_group_operation_ids"] == [0]
        assert mm["hsp_little_group_operation_ids"] == [0]
        assert mm["valley_preserving_operation_ids"] == [0]
        assert mm["parent_hsp_little_group_operation_ids"] == [0, 5]
        assert mm["valley_sewing_operation_ids"] == [5]
        assert mm["workflow_path"] == "blocked"
        assert mm["valley_preserving_operations"] == []

    # GammaM/KM records: subspace HSP little group = [0,1,2].
    gm_recs = [r for r in report["representation_records"] if r["kpoint"] == "GammaM"]
    km_recs = [r for r in report["representation_records"] if r["kpoint"] == "KM"]
    for recs in (gm_recs, km_recs):
        for rec in recs:
            assert rec["subspace_hsp_little_group_operation_ids"] == [0, 1, 2]
            assert rec["hsp_little_group_operation_ids"] == [0, 1, 2]
            assert rec["valley_preserving_operation_ids"] == [0, 1, 2]
            assert rec["parent_hsp_little_group_operation_ids"] == [0, 1, 2, 3, 4, 5]
            assert rec["valley_sewing_operation_ids"] == [3, 4, 5]


def test_representation_records_empty_when_no_rows():
    report = build_valley_projected_representation_report(
        kpoint_names=[],
        valley_names=[],
    )
    assert report["grouped_record_count"] == 0
    assert report["representation_records"] == []
    assert report["rows"] == []
    assert report["trusted_representation_count"] == 0
