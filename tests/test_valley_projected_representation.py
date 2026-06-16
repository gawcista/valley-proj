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

    row = report["rows"][0]
    assert row["subspace_space_group"]["candidate_space_group_symbol"] == "P2"
    assert row["hsp_little_group_operation_ids"] == [0, 4]
    assert row["valley_preserving_operation_ids"] == [0, 4]
    assert row["valley_changing_operation_ids"] == [5]
    assert row["legacy_subspace_group_candidate"] == "C2_like"
    assert report["subspace_space_group_counts"] == {"P2": 1}
    assert report["legacy_subspace_group_candidate_counts"] == {"C2_like": 1}
