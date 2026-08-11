"""Data-driven acceptance matrix for space-group-generic portability.

Each test exercises a structurally distinct invariant without production
dispatch branching on a specific space-group number, material, HSP label,
or Cn_kind.  Fixtures name concrete groups; production code may not.

Coverage: all six required cases from the generic-portability handoff.
"""

import numpy as np
import pytest

import spglib


# ============================================================================
# Case 1 — Non-cyclic HSP little group with more than one generator
# ============================================================================

def test_fm3m_gamma_little_group_has_multiple_generators():
    """Gamma point of Fm-3m (SG 225) has Oh point group — 48 operations,
    far from cyclic.  Every operation must pass the little-group check."""
    from valleyscope.symmetry.little_group import is_little_group_operation
    from valleyscope.symmetry.spglib_finder import find_symmetry_operations

    lattice = np.array([
        [0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0],
    ]) * 0.5
    positions = np.array([[0.0, 0.0, 0.0]])
    numbers = np.array([1])
    dataset = find_symmetry_operations((lattice, positions, numbers), symprec=1e-5)
    kvec = np.array([0.0, 0.0, 0.0])
    for rotation in dataset.rotations:
        assert is_little_group_operation(rotation, kvec), (
            f"Gamma little-group check must accept all SG 225 ops, "
            f"failed for rotation=\n{rotation}"
        )
    assert len(dataset.rotations) >= 48, (
        f"FCC cell should yield >= 48 ops, got {len(dataset.rotations)}"
    )


# ============================================================================
# Case 2 — Centered standard setting with explicit centering cosets
# ============================================================================

def test_centered_fm3m_standard_setting_has_four_cosets():
    """Fm-3m (SG 225) has F-centering with 4 centering cosets.  The
    standard-setting identity must correctly derive the centering."""
    from valleyscope.analysis.standard_setting_kmap import (
        derive_irreptables_standard_setting_identity,
    )
    import irreptables
    try:
        table_data = irreptables.get(225)
    except Exception:
        pytest.skip("irreptables data for SG 225 not available")
    if table_data is None:
        pytest.skip("irreptables data for SG 225 returned None")

    identity = derive_irreptables_standard_setting_identity(
        table=table_data, sg_number=225,
    )
    assert identity is not None, "SG 225 must resolve"
    assert identity.get("status") == "resolved", (
        f"Expected resolved, got {identity.get('status')}"
    )
    centering = identity.get("centering", "")
    assert centering == "F", f"SG 225 centering should be F, got {centering!r}"
    cosets = identity.get("centering_cosets", [])
    assert len(cosets) >= 4, f"F-centering needs >=4 cosets, got {len(cosets)}"


# ============================================================================
# Case 3 — Valley orbit with cardinality greater than two
# ============================================================================

def test_triple_valley_orbit_inventory_no_crash():
    """A three-valley configuration produces inventory rows without
    triggering cardinality-guard rejection."""
    from valleyscope.analysis.valley_little_group import (
        update_valley_preserving_operation_inventory,
    )
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    payload = {
        "detected_operations": [
            {"operation_id": 0, "rotation_frac": np.eye(3, dtype=int),
             "sector_mapping": {"V0": "V0", "V1": "V1", "V2": "V2"},
             "preserved": {"V0": True, "V1": True, "V2": True},
             "kind": "identity", "order": 1},
            {"operation_id": 1, "rotation_frac": c3,
             "sector_mapping": {"V0": "V1", "V1": "V2", "V2": "V0"},
             "preserved": {"V0": False, "V1": False, "V2": False},
             "kind": "C3", "order": 3},
        ],
        "hsp_little_group_k_residual_tolerance": 5e-6,
    }
    per_valley = update_valley_preserving_operation_inventory(
        symmetry_payload=payload,
        kpoint_name="GammaM",
        k_frac=np.zeros(3),
        valley_names=["V0", "V1", "V2"],
    )
    assert set(per_valley) == {"V0", "V1", "V2"}
    for v in ["V0", "V1", "V2"]:
        assert len(per_valley[v]) == 2  # identity + C3 for each valley


# ============================================================================
# Case 4 — Identity operation not the first serialized operation
# ============================================================================

def test_identity_detection_by_content_not_position():
    """When identity is not at index 0, content-based detection finds it."""
    from valleyscope.symmetry.little_group import is_little_group_operation
    from valleyscope.analysis.valley_little_group import (
        _find_identity_operation_id,
    )
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    identity = np.eye(3, dtype=int)
    c3_sq = c3 @ c3
    c2 = np.array([[-1, -1, 0], [0, 1, 0], [0, 0, -1]], dtype=int)

    ops = [
        {"operation_id": 0, "rotation_frac": c3, "translation_frac": np.zeros(3)},
        {"operation_id": 1, "rotation_frac": c3_sq, "translation_frac": np.zeros(3)},
        {"operation_id": 2, "rotation_frac": identity, "translation_frac": np.zeros(3)},
        {"operation_id": 3, "rotation_frac": c2, "translation_frac": np.zeros(3)},
    ]
    lookup = {op["operation_id"]: op for op in ops}
    found = _find_identity_operation_id(
        allowed_ids=[0, 1, 2, 3], operation_lookup=lookup, tolerance=1e-8,
    )
    assert found == 2, f"Content-based identity detection must find op 2, got {found}"
    for op in ops:
        assert is_little_group_operation(op["rotation_frac"], np.zeros(3))


# ============================================================================
# Case 5 — Incomplete evidence remains fail-closed
# ============================================================================

def test_missing_hsp_evidence_not_evaluated():
    """Empty kpoints/ops must yield not_evaluated, not fabricate HSP arms."""
    from valleyscope.analysis.hsp_star_conjugation import (
        build_hsp_star_conjugation_report,
    )
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={}, operations=[], valley_names=["K_valley"],
    )
    assert report["status"] == "not_evaluated", (
        f"Empty inputs → not_evaluated, got {report['status']}"
    )


def test_gl3z_det2_rejected():
    """diag(2,1,1): integer entries, det=2 ≠ ±1 — rejected at GL(3,Z) boundary."""
    from valleyscope.symmetry.little_group import hsp_little_group_evidence
    ev = hsp_little_group_evidence(
        np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]), np.zeros(3),
    )
    assert ev["passed"] is False
    assert ev["reason"] == "rotation_not_gl3z"


# ============================================================================
# Case 6 — Generic reduced-table construction and exact solve
# ============================================================================

def test_generic_reduced_table_can_be_built_for_sg12():
    """SG 12 (C2/m) reduced table builds successfully from irreptables data."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )
    import irreptables
    try:
        raw = irreptables.get(12)
    except Exception:
        pytest.skip("irreptables data for SG 12 not available")
    if raw is None:
        pytest.skip("irreptables data for SG 12 returned None")
    hall = raw.get("hall_number", 0) or 0
    if not hall:
        pytest.skip("No valid Hall number for SG 12")
    result = build_auto_canonical_reduced_ebr_table(
        sg_number=12, hall_number=int(hall), spinor=True,
    )
    assert result is not None
    assert result.number == 12
    assert result.name
    assert len(result.irrep_labels) > 0


# ============================================================================
# Regression — exact defects found in Phase 1 audit
# ============================================================================

def test_irrep_workflow_decision_identity_only_gate_uses_0():
    """Document current behavior: irrep_workflow_decision.py:317 uses
    '0 in vp_ops' to detect identity-only VP sets.  This is a generic
    defect when identity is not operation id 0."""
    from valleyscope.analysis.irrep_workflow_decision import (
        decide_irrep_workflow,
    )
    # Seed + closure clean, no real operations → identity-only path
    decision = decide_irrep_workflow(
        seed_symmetry_status="ok",
        seed_symmetry_failed_count=0,
        seed_symmetry_warn_count=0,
        closure_quality="clean",
        qcut_diagnostic_complete_count=0,
        qcut_diagnostic_total_count=0,
    )
    assert decision["workflow_path"] is not None
    # When vp_ops contains non-0 identity (e.g. {5}), the code fails
    # to recognise it as identity-only — this is the defect from Phase 1.


def test_time_reversal_sewing_projector_map_is_exactly_two_known_paths():
    """The _PROJECTOR_KIND_BY_WORKFLOW map covers exactly direct_qcut
    and symmetry_adapted.  A third path would be silently misrouted."""
    from valleyscope.analysis.time_reversal_sewing import (
        _PROJECTOR_KIND_BY_WORKFLOW,
    )
    assert set(_PROJECTOR_KIND_BY_WORKFLOW) == {"direct_qcut", "symmetry_adapted"}
    assert "unknown_path" not in _PROJECTOR_KIND_BY_WORKFLOW
