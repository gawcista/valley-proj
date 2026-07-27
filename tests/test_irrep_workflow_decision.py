import json
import pytest

from valleyscope.analysis.irrep_workflow_decision import (
    decide_irrep_workflow,
    build_irrep_workflow_decisions,
    PATH_DIRECT_QCUT,
    PATH_SYMMETRY_ADAPTED,
    PATH_BLOCKED,
    READINESS_TRUSTED,
    READINESS_USABLE_WITH_CAUTION,
    READINESS_BLOCKED,
)


# -----------------------------------------------------------------------
# 1. Direct q-cut path
# -----------------------------------------------------------------------

def test_direct_qcut_waits_for_cprime_evidence():
    d = decide_irrep_workflow(
        seed_symmetry_status="passed",
        seed_symmetry_failed_count=0,
        closure_quality="ok",
        qcut_eigenvalue_ready_count=3,
        qcut_eigenvalue_total_count=3,
    )
    assert d["workflow_path"] == PATH_DIRECT_QCUT
    assert d["readiness_level"] == READINESS_USABLE_WITH_CAUTION
    assert "scoped_representation_evidence" in d["reason"]
    assert d["uses_symmetry_adapted_projector"] is False
    assert d["direct_qcut_allowed"] is True


def test_direct_qcut_is_cautious_before_cprime_promotion():
    d = decide_irrep_workflow(
        seed_symmetry_status="passed",
        seed_symmetry_failed_count=0,
        closure_quality="ok",
        qcut_eigenvalue_ready_count=2,
        qcut_eigenvalue_total_count=2,
    )
    assert d["workflow_path"] == PATH_DIRECT_QCUT
    assert d["readiness_level"] == READINESS_USABLE_WITH_CAUTION
    assert "scoped_representation_evidence" in d["reason"]


# -----------------------------------------------------------------------
# 2. Symmetry-adapted path
# -----------------------------------------------------------------------

def test_symmetry_adapted_waits_for_cprime_evidence():
    d = decide_irrep_workflow(
        seed_symmetry_status="failed",
        seed_symmetry_failed_count=1,
        closure_quality="ok",
        qcut_eigenvalue_ready_count=0,
        qcut_eigenvalue_total_count=2,
        sym_adapted_proj_status="ok",
        sym_adapted_local_irrep_ready=True,
        sym_adapted_diagnostic_only=False,
    )
    assert d["workflow_path"] == PATH_SYMMETRY_ADAPTED
    assert d["readiness_level"] == READINESS_USABLE_WITH_CAUTION
    assert d["uses_symmetry_adapted_projector"] is True


def test_symmetry_adapted_numerical_path_is_cautious():
    d = decide_irrep_workflow(
        seed_symmetry_status="failed",
        seed_symmetry_failed_count=1,
        closure_quality="usable_with_caution",
        qcut_eigenvalue_ready_count=0,
        qcut_eigenvalue_total_count=2,
        sym_adapted_proj_status="ok",
        sym_adapted_local_irrep_ready=True,
        sym_adapted_diagnostic_only=False,
    )
    assert d["workflow_path"] == PATH_SYMMETRY_ADAPTED
    assert d["readiness_level"] == READINESS_USABLE_WITH_CAUTION


def test_symmetry_adapted_not_ready():
    d = decide_irrep_workflow(
        seed_symmetry_status="failed",
        seed_symmetry_failed_count=1,
        closure_quality="usable_with_caution",
        qcut_eigenvalue_ready_count=0,
        qcut_eigenvalue_total_count=2,
        sym_adapted_proj_status="warn",
        sym_adapted_local_irrep_ready=False,
        sym_adapted_diagnostic_only=True,
    )
    assert d["workflow_path"] == PATH_SYMMETRY_ADAPTED
    assert d["readiness_level"] == READINESS_USABLE_WITH_CAUTION


# -----------------------------------------------------------------------
# 3. Blocked path
# -----------------------------------------------------------------------

def test_blocked_closure():
    d = decide_irrep_workflow(
        seed_symmetry_status="passed",
        seed_symmetry_failed_count=0,
        closure_quality="blocked",
        qcut_eigenvalue_ready_count=1,
        qcut_eigenvalue_total_count=1,
    )
    assert d["workflow_path"] == PATH_BLOCKED
    assert d["readiness_level"] == READINESS_BLOCKED


def test_blocked_projector_failed():
    d = decide_irrep_workflow(
        seed_symmetry_status="failed",
        seed_symmetry_failed_count=1,
        closure_quality="usable_with_caution",
        qcut_eigenvalue_ready_count=0,
        qcut_eigenvalue_total_count=1,
        sym_adapted_proj_status="failed",
    )
    assert d["workflow_path"] == PATH_BLOCKED
    assert d["readiness_level"] == READINESS_BLOCKED


def test_blocked_all_closed():
    d = decide_irrep_workflow(
        seed_symmetry_status="not_evaluated",
        closure_quality="not_evaluated",
        qcut_eigenvalue_ready_count=0,
        qcut_eigenvalue_total_count=0,
        sym_adapted_proj_status="not_evaluated",
    )
    assert d["workflow_path"] == PATH_BLOCKED
    assert d["readiness_level"] == READINESS_BLOCKED


# -----------------------------------------------------------------------
# 4. Readiness levels only three values
# -----------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expected_readiness", [
    (dict(seed_symmetry_status="passed", seed_symmetry_failed_count=0,
          closure_quality="ok", qcut_eigenvalue_ready_count=1,
          qcut_eigenvalue_total_count=1),
     READINESS_USABLE_WITH_CAUTION),
    (dict(seed_symmetry_status="passed", seed_symmetry_failed_count=0,
          closure_quality="ok", qcut_eigenvalue_ready_count=1,
          qcut_eigenvalue_total_count=1),
     READINESS_USABLE_WITH_CAUTION),
    (dict(seed_symmetry_status="failed", seed_symmetry_failed_count=1,
          closure_quality="usable_with_caution",
          qcut_eigenvalue_ready_count=0, qcut_eigenvalue_total_count=1,
          sym_adapted_proj_status="ok", sym_adapted_local_irrep_ready=True,
          sym_adapted_diagnostic_only=False),
     READINESS_USABLE_WITH_CAUTION),
    (dict(seed_symmetry_status="failed", seed_symmetry_failed_count=1,
          closure_quality="blocked",
          qcut_eigenvalue_ready_count=0, qcut_eigenvalue_total_count=1),
     READINESS_BLOCKED),
])
def test_readiness_level_only_three(kwargs, expected_readiness):
    d = decide_irrep_workflow(**kwargs)
    assert d["readiness_level"] in {
        READINESS_TRUSTED, READINESS_USABLE_WITH_CAUTION, READINESS_BLOCKED
    }
    assert d["readiness_level"] == expected_readiness


# -----------------------------------------------------------------------
# 5. usable_with_caution never becomes trusted
# -----------------------------------------------------------------------

def test_usable_with_caution_never_trusted():
    d = decide_irrep_workflow(
        seed_symmetry_status="warn",
        seed_symmetry_max_epsilon=0.05,
        seed_symmetry_failed_count=0,
        closure_quality="usable_with_caution",
        qcut_eigenvalue_ready_count=1,
        qcut_eigenvalue_total_count=1,
        sym_adapted_proj_status="warn",
        sym_adapted_local_irrep_ready=False,
        sym_adapted_diagnostic_only=True,
    )
    assert d["readiness_level"] != READINESS_TRUSTED


def test_seed_warning_stays_cautious():
    d = decide_irrep_workflow(
        seed_symmetry_status="warn",
        seed_symmetry_max_epsilon=0.05,
        seed_symmetry_failed_count=0,
        seed_symmetry_warn_count=1,
        closure_quality="clean",
        qcut_eigenvalue_ready_count=2,
        qcut_eigenvalue_total_count=2,
    )
    assert d["workflow_path"] == PATH_DIRECT_QCUT
    assert d["readiness_level"] == READINESS_USABLE_WITH_CAUTION
    assert d["direct_qcut_allowed"] is True


def test_partial_qcut_readiness_does_not_take_direct_path():
    d = decide_irrep_workflow(
        seed_symmetry_status="passed",
        seed_symmetry_failed_count=0,
        closure_quality="clean",
        qcut_eigenvalue_ready_count=1,
        qcut_eigenvalue_total_count=2,
    )
    assert d["workflow_path"] == PATH_BLOCKED
    assert d["readiness_level"] == READINESS_BLOCKED
    assert d["direct_qcut_allowed"] is False


# -----------------------------------------------------------------------
# 6. build_irrep_workflow_decisions integration
# -----------------------------------------------------------------------

def test_build_decisions_empty():
    result = build_irrep_workflow_decisions(
        projector_symmetry_report=None,
        target_subspace_closure_report=None,
        symmetry_adapted_valley_report=None,
        symmetry_rows=[],
        valley_names=["K", "Kp"],
    )
    assert result["status"] == "ok"
    assert result["by_kpoint"] == {}


def test_build_decisions_direct_qcut_toy():
    result = build_irrep_workflow_decisions(
        projector_symmetry_report={
            "by_kpoint": {
                "Gamma": {
                    "seed_projector_symmetry": [
                        {
                            "operation_id": 1, "source_valley": "K",
                            "mapped_valley": "K", "epsilon_seed": 0.01,
                            "status": "passed",
                        },
                    ],
                },
            },
        },
        target_subspace_closure_report={
            "by_kpoint": {
                "Gamma": [{
                    "operation_id": 1, "kpoint": "Gamma",
                    "closure_quality": "ok",
                    "raw_unitarity_error": 1e-5,
                    "max_closure_residual": 0.0,
                }],
            },
        },
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "Gamma": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K"],
                        "local_irrep_ready": True,
                        "diagnostic_only": False,
                        "hsp_preserving_operation_ids": [1],
                        "subspace_group": {
                            "valley_preserving_operation_ids": [1],
                        },
                        "symmetry_adapted_projectors": {
                            "status": "ok",
                            "seed_overlap": {"K": 0.95},
                        },
                    }],
                },
            },
        },
        symmetry_rows=[
            {"kpoint": "Gamma", "target_valley": "K",
             "operation_id": 1, "numerical_input_ready": True},
        ],
        valley_names=["K"],
    )
    assert "Gamma" in result["by_kpoint"]
    d = result["by_kpoint"]["Gamma"]["K"]
    assert d["workflow_path"] == PATH_DIRECT_QCUT
    assert d["readiness_level"] == READINESS_USABLE_WITH_CAUTION
    assert "scoped_representation_evidence" in d["reason"]


def test_build_decisions_blocked_toy():
    result = build_irrep_workflow_decisions(
        projector_symmetry_report={
            "by_kpoint": {
                "MM": {
                    "seed_projector_symmetry": [
                        {"operation_id": 5, "source_valley": "M3",
                         "mapped_valley": "M3", "epsilon_seed": 1.2,
                         "status": "failed"},
                    ],
                },
            },
        },
        target_subspace_closure_report={
            "by_kpoint": {
                "MM": [{"operation_id": 5, "kpoint": "MM",
                        "closure_quality": "blocked",
                        "raw_unitarity_error": 1.41}],
            },
        },
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "MM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["M3"],
                        "local_irrep_ready": False,
                        "diagnostic_only": True,
                        "hsp_preserving_operation_ids": [5],
                        "subspace_group": {
                            "valley_preserving_operation_ids": [5],
                        },
                        "symmetry_adapted_projectors": {
                            "status": "failed",
                            "seed_overlap": {},
                        },
                    }],
                },
            },
        },
        symmetry_rows=[],
        valley_names=["M3"],
    )
    d = result["by_kpoint"]["MM"]["M3"]
    assert d["workflow_path"] == PATH_BLOCKED
    assert d["readiness_level"] == READINESS_BLOCKED


def test_identity_only_uses_algebraic_seed_and_closure_not_out_of_group_rows():
    result = build_irrep_workflow_decisions(
        projector_symmetry_report={
            "by_kpoint": {
                "MM": {
                    "seed_projector_symmetry": [],
                },
            },
        },
        target_subspace_closure_report={
            "by_kpoint": {
                "MM": [{
                    "operation_id": 5,
                    "kpoint": "MM",
                    "little_group_passed": False,
                    "closure_quality": "blocked",
                    "status": "not_evaluated",
                    "reason": "D_raw not available",
                }],
            },
        },
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "MM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["M1"],
                        "local_irrep_ready": True,
                        "diagnostic_only": False,
                        "hsp_preserving_operation_ids": [0],
                        "subspace_group": {
                            "valley_preserving_operation_ids": [0],
                        },
                        "symmetry_adapted_projectors": {
                            "status": "ok",
                            "seed_overlap": {"M1": 0.9},
                        },
                    }],
                },
            },
        },
        symmetry_rows=[],
        valley_names=["M1"],
    )
    d = result["by_kpoint"]["MM"]["M1"]
    assert d["workflow_path"] == PATH_DIRECT_QCUT
    assert d["readiness_level"] == READINESS_USABLE_WITH_CAUTION
    assert d["uses_symmetry_adapted_projector"] is False
    assert d["identity_readiness_evidence"] == {
        "seed_projector_symmetry": "algebraic_identity",
        "target_subspace_closure": "algebraic_identity",
        "local_representation_dimension": "available",
    }
    assert "identity operation" in d["reason"]


def test_identity_only_seed_path_stays_direct_qcut_without_nonidentity_phase():
    result = build_irrep_workflow_decisions(
        projector_symmetry_report={
            "by_kpoint": {
                "MM": {
                    "seed_projector_symmetry": [{
                        "operation_id": 0,
                        "source_valley": "K",
                        "mapped_valley": "K",
                        "epsilon_seed": 0.0,
                        "status": "passed",
                    }],
                },
            },
        },
        target_subspace_closure_report={
            "by_kpoint": {
                "MM": [{
                    "operation_id": 0,
                    "little_group_passed": True,
                    "closure_quality": "clean",
                    "raw_unitarity_error": 0.0,
                    "max_closure_residual": 0.0,
                }],
            },
        },
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "MM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K"],
                        "local_irrep_ready": True,
                        "diagnostic_only": False,
                        "hsp_preserving_operation_ids": [0],
                        "subspace_group": {
                            "valley_preserving_operation_ids": [0],
                        },
                        "symmetry_adapted_projectors": {
                            "status": "not_evaluated",
                            "seed_overlap": {},
                        },
                        "valley_preserving_character_diagnostics": {
                            "per_valley": {"K": [{"operation_id": 0}]},
                        },
                    }],
                },
            },
        },
        symmetry_rows=[],
        valley_names=["K"],
    )

    decision = result["by_kpoint"]["MM"]["K"]
    assert decision["workflow_path"] == PATH_DIRECT_QCUT
    assert decision["readiness_level"] == READINESS_USABLE_WITH_CAUTION
    assert decision["uses_symmetry_adapted_projector"] is False
    assert decision["direct_qcut_allowed"] is True
    assert decision["identity_only_valley_preserving_subgroup"] is True
    assert "non-identity eigenphase not required" in decision["reason"]


# -----------------------------------------------------------------------
# 7. Schema
# -----------------------------------------------------------------------

def test_schema_json_serializable():
    d = decide_irrep_workflow(
        seed_symmetry_status="passed",
        seed_symmetry_failed_count=0,
        closure_quality="ok",
        qcut_eigenvalue_ready_count=2,
        qcut_eigenvalue_total_count=2,
    )
    encoded = json.dumps(d)
    assert len(encoded) > 0
    assert "dtype" not in encoded


def test_schema_no_forbidden_terms():
    result = build_irrep_workflow_decisions(
        projector_symmetry_report={
            "by_kpoint": {
                "Gamma": {
                    "seed_projector_symmetry": [
                        {"operation_id": 1, "source_valley": "K",
                         "mapped_valley": "K", "epsilon_seed": 0.01,
                         "status": "passed"},
                    ],
                },
            },
        },
        target_subspace_closure_report={
            "by_kpoint": {
                "Gamma": [{"operation_id": 1, "closure_quality": "ok"}],
            },
        },
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "Gamma": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K"],
                        "local_irrep_ready": True,
                        "diagnostic_only": False,
                        "hsp_preserving_operation_ids": [1],
                        "subspace_group": {
                            "valley_preserving_operation_ids": [1],
                        },
                        "symmetry_adapted_projectors": {
                            "status": "ok", "seed_overlap": {"K": 0.99},
                        },
                    }],
                },
            },
        },
        symmetry_rows=[
            {"kpoint": "Gamma", "target_valley": "K",
             "operation_id": 1, "topology_input_ready": True},
        ],
        valley_names=["K"],
    )
    encoded = json.dumps(result)
    for forbidden in [
        "covariance", "equivariance", "stabilizer", "valley_little_group",
    ]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"
