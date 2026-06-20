from valleyscope.analysis.valley_projected_representation import (
    build_valley_projected_representation_report,
)


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
                                "subspace_group_candidate": "C2_like",
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
    assert row["valley_preserving_operation_ids"] == [0, 4]
    assert row["valley_changing_operation_ids"] == [5]
    assert row["legacy_subspace_group_candidate"] == "C2_like"
    assert report["subspace_space_group_counts"] == {"P2": 1}
    assert report["legacy_subspace_group_candidate_counts"] == {"C2_like": 1}

    # Grouped representation records.
    assert report["grouped_record_count"] == 1
    recs = report["representation_records"]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["kpoint"] == "GammaM"
    assert rec["valley"] == "M1_valley"
    assert rec["subspace_space_group"]["candidate_space_group_symbol"] == "P2"
    assert rec["hsp_little_group_operation_ids"] == [0, 4]
    assert rec["valley_preserving_operation_ids"] == [0, 4]
    assert rec["valley_changing_operation_ids"] == [5]
    assert rec["readiness_level"] == "trusted"
    assert rec["workflow_path"] == "direct_qcut"
    assert rec["legacy_subspace_group_candidate"] == "C2_like"
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
                                "subspace_group_candidate": "C3_like",
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
                                "subspace_group_candidate": "C2_like",
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
                                "subspace_group_candidate": "C3_like",
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
                                "subspace_group_candidate": "C4_like",
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
    assert rec["legacy_subspace_group_candidate"] == "C4_like"
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


def test_representation_records_empty_when_no_rows():
    report = build_valley_projected_representation_report(
        kpoint_names=[],
        valley_names=[],
    )
    assert report["grouped_record_count"] == 0
    assert report["representation_records"] == []
    assert report["rows"] == []
    assert report["trusted_representation_count"] == 0
