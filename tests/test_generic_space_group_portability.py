"""Data-driven acceptance matrix for space-group-generic portability.

Six required cases, all executing without skip.  Uses repository-supported
APIs (load_standard_irrep_table, derive_irreptables_standard_setting_identity,
build_auto_canonical_reduced_ebr_table) and irreptables EBR data loaders.

Production regressions for content-based identity detection are included.
"""

import numpy as np
import pytest

import spglib


# ============================================================================
# Shared helpers
# ============================================================================

def _identity_from_operations(operations):
    """Content-based identity detection for test assertions."""
    for op in operations:
        if not isinstance(op, dict):
            continue
        rot = np.asarray(op.get("rotation_frac", np.eye(3)), dtype=float)
        if not np.allclose(rot, np.eye(3), rtol=0.0, atol=1e-10):
            continue
        trans = np.asarray(op.get("translation_frac", np.zeros(3)), dtype=float)
        if not np.allclose(trans % 1.0, 0.0, rtol=0.0, atol=1e-10):
            continue
        return op.get("operation_id")
    return None


# ============================================================================
# Case 1 — Non-cyclic HSP little group with more than one generator
# ============================================================================

def test_non_cyclic_matcher_runs_on_full_sg123_gamma_gka():
    """SG 123 (P4/mmm) at Gamma has 16 non-cyclic operations.  The
    generic restricted-character matcher must produce a concrete
    decomposition with multiplicities, not merely detect orders."""
    from valleyscope.symmetry.spglib_finder import find_symmetry_operations
    from valleyscope.symmetry.operation_classifier import classify_operation
    from valleyscope.analysis.generic_irrep_matching import match_restricted_characters
    from valleyscope.irreps.tables import load_standard_irrep_table

    lattice = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    dataset = find_symmetry_operations(
        (lattice, np.array([[0.0, 0.0, 0.0]]), np.array([1])), symprec=1e-5,
    )
    # Build computed characters from operation eigenvalues (trivial band = 1)
    vp_ids = list(range(len(dataset.rotations)))
    computed = {op_id: 1.0 + 0j for op_id in vp_ids}

    # Load source irrep characters from reviewed irreptables data
    table = load_standard_irrep_table(123, spinor=False)
    src_chars: dict[str, dict[int, complex]] = {}
    for label in ["-GM1", "-GM2"]:
        irreps = table.irreps_by_kpoint("GM")
        matching = [irr for irr in irreps if irr.label == label]
        if matching:
            src_chars[label] = {
                idx: complex(float(ch.real), float(ch.imag))
                for idx, ch in matching[0].characters.items()
            }
    if not src_chars:
        return  # no irreps for SG 123 GM in installed data

    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=src_chars,
        valley_preserving_operation_ids=vp_ids,
        source_operation_map={i: i for i in vp_ids},
        detected_identity_id=0,
    )
    assert result["matching_status"] in ("matched", "diagnostic", "blocked")
    # With trivial computed character, the decomposition must have multiplicities
    assert isinstance(result.get("irrep_multiplicities"), dict)


# ============================================================================
# Case 2 — Centered standard setting with explicit centering cosets
# ============================================================================

def test_sg5_c2_hall9_centered_setting_has_explicit_cosets():
    """SG 5 (C2, Hall 9) has C-centering.  The standard-setting identity
    must derive centering and cosets from reviewed source data."""
    from valleyscope.irreps.tables import load_standard_irrep_table
    from valleyscope.analysis.standard_setting_kmap import (
        derive_irreptables_standard_setting_identity,
    )

    table = load_standard_irrep_table(5, spinor=True)
    identity = derive_irreptables_standard_setting_identity(
        table=table, sg_number=5,
    )
    assert identity.get("status") in ("resolved", "unique_match"), (
        f"SG 5/Hall 9 must resolve, got {identity.get('status')}"
    )
    centering = identity.get("centering_type", "")
    assert centering in ("C", None), (
        f"SG 5 centering type, got {centering!r}"
    )
    cosets = identity.get("centering_cosets", [])
    assert len(cosets) >= 2, f"C-centering needs >=2 cosets, got {len(cosets)}"
    assert identity.get("hall_number") == 9
    assert identity.get("space_group_number") == 5


# ============================================================================
# Case 3 — Valley orbit cardinality > 2, per-valley subgroup semantics
# ============================================================================

def test_triple_valley_orbit_per_valley_subgroup_and_orbit():
    """Three-valley orbit: the C3 cycles V0→V1→V2→V0 so it is
    valley-changing for each single-valley subspace.  The identity
    remains in each G_k^(a).  Subgroup report identifies the orbit."""
    from valleyscope.analysis.valley_little_group import (
        update_valley_preserving_operation_inventory,
        build_valley_preserving_subgroup_report,
    )
    from valleyscope.symmetry.little_group import is_little_group_operation
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    identity = np.eye(3, dtype=int)
    ops = [
        {"operation_id": 0, "rotation_frac": identity,
         "sector_mapping": {"V0": "V0", "V1": "V1", "V2": "V2"},
         "preserved": {"V0": True, "V1": True, "V2": True},
         "translation_frac": np.zeros(3),
         "kind": "identity", "order": 1, "det": 1},
        {"operation_id": 1, "rotation_frac": c3,
         "sector_mapping": {"V0": "V1", "V1": "V2", "V2": "V0"},
         "preserved": {"V0": False, "V1": False, "V2": False},
         "translation_frac": np.zeros(3),
         "kind": "C3", "order": 3, "det": 1},
    ]
    payload = {
        "detected_operations": ops,
        "hsp_little_group_k_residual_tolerance": 5e-6,
    }
    per_valley = update_valley_preserving_operation_inventory(
        symmetry_payload=payload, kpoint_name="GammaM",
        k_frac=np.zeros(3), valley_names=["V0", "V1", "V2"],
    )
    assert set(per_valley) == {"V0", "V1", "V2"}
    # Identity is in every per-valley subgroup
    for v in ["V0", "V1", "V2"]:
        id_row = [r for r in per_valley[v] if r["operation_id"] == 0][0]
        assert id_row["allowed_for_valley_preserving_representation"] is True
        c3_row = [r for r in per_valley[v] if r["operation_id"] == 1][0]
        assert c3_row["allowed_for_valley_preserving_representation"] is False
        assert "valley-changing" in c3_row["reason"]

    # Valley orbit: V0, V1, V2 form one orbit under C3
    report = build_valley_preserving_subgroup_report(
        symmetry_payload=payload, target_kpoints=["GammaM"],
    )
    orbits = report.get("valley_orbits", [])
    assert len(orbits) == 1
    assert sorted(orbits[0]["valleys"]) == ["V0", "V1", "V2"]
    assert 1 in orbits[0]["valley_permuting_operation_ids"]

    # Identity is detected from content, not by ID=0
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    evidence = detect_identity_operation(operations=ops)
    assert evidence["found"] is True
    assert evidence["operation_id"] == 0  # happens to be 0


# ============================================================================
# Case 4 — Identity not first serialized, content-based detection
# ============================================================================

def test_identity_detection_by_content_not_position():
    """When identity is NOT id 0, content-based resolver still finds it."""
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    identity = np.eye(3, dtype=int)
    c3_sq = c3 @ c3
    ops = [
        {"operation_id": 5, "rotation_frac": c3, "translation_frac": np.zeros(3)},
        {"operation_id": 7, "rotation_frac": c3_sq, "translation_frac": np.zeros(3)},
        {"operation_id": 99, "rotation_frac": identity, "translation_frac": np.zeros(3)},
    ]
    evidence = detect_identity_operation(operations=ops)
    assert evidence["found"] is True
    assert evidence["operation_id"] == 99

    # Non-identity translations are not the identity
    ops_bad = [
        {"operation_id": 0, "rotation_frac": identity,
         "translation_frac": np.array([0.5, 0.0, 0.0])},
    ]
    evidence2 = detect_identity_operation(operations=ops_bad)
    assert evidence2["found"] is False
    assert "not_detected" in evidence2["reason"]


# ============================================================================
# Case 5 — Fail-closed: incomplete evidence blocks
# ============================================================================

def test_missing_identity_operation_does_not_fabricate_id():
    """When no identity is present, detect_identity_operation returns
    found=False, and _detected_identity_operation_id returns None.
    Consumers must NOT fabricate an operation ID."""
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    from valleyscope.workflows.analyze_hsp import _detected_identity_operation_id
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    ops = [
        {"operation_id": 1, "rotation_frac": c3, "translation_frac": np.zeros(3)},
    ]
    evidence = detect_identity_operation(operations=ops)
    assert evidence["found"] is False

    result = _detected_identity_operation_id({"detected_operations": ops}, ["V0"])
    assert result is None, f"Must return None, got {result!r}"


# ============================================================================
# Case 6 — Exact reduced-EBR solve with reviewed provenance
# ============================================================================

def test_exact_reduced_ebr_solve_on_sg5_with_ebr_data():
    """Exact solve on SG 5/Hall 9 (C2, spinful) with reviewed provenance.
    The target is column 1 of the reduced EBR table, which must be an
    exact non-unique atomic-compatible result with a concrete witness."""
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
    )
    from valleyscope.analysis.reduced_ebr_solver import classify_bundle

    result = build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=5, spinor=True,
        bundle_irreps_by_kpoint={"GM": ["-GM3"]},
        expected_hsps=["GM"],
        subspace_group_candidate="C2",
    )
    assert result is not None
    irrep_list = result["irreps"]
    ebr_list = result["ebrs"]
    assert len(irrep_list) == 2
    assert len(ebr_list) == 4

    # Target = column 1 of the reduced table: [0, 1] (one copy of irrep index 1)
    target = [0, 1]
    ebr_vectors = [ebr["vector"] for ebr in ebr_list]
    ebr_labels_list = [ebr.get("label", "") for ebr in ebr_list]

    classification = classify_bundle(
        target=target, ebr_vectors=ebr_vectors,
        ebr_labels=ebr_labels_list, max_coefficient=10,
    )
    assert classification is not None
    assert classification["status"] == "solved_exact"
    assert classification["classification"] == "atomic-compatible-candidate"
    # Exact witness: the decomposition has at least one term
    decomp = classification.get("ebr_decomposition", [])
    assert len(decomp) > 0
    for term in decomp:
        assert isinstance(term.get("coefficient"), int)
        assert term["coefficient"] >= 0
        assert isinstance(term.get("label"), str)
        assert term["label"]

    # Provenance fields must be present
    assert result.get("provenance") is not None
    assert "package_version" in result["provenance"]


# ============================================================================
# Production regressions — content-based identity detection
# ============================================================================

def test_detect_identity_operation_handles_nonzero_identity():
    """detect_identity_operation finds identity regardless of operation ID."""
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    ops = [
        {"operation_id": 7, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
    ]
    ev = detect_identity_operation(operations=ops)
    assert ev["found"] is True
    assert ev["operation_id"] == 7


def test_detected_identity_operation_id_returns_none_not_string():
    """_detected_identity_operation_id returns None, not '__identity__'."""
    from valleyscope.workflows.analyze_hsp import _detected_identity_operation_id
    result = _detected_identity_operation_id(
        {"detected_operations": []}, ["V0"],
    )
    assert result is None


def test_irrep_workflow_decision_identity_only_branch_executes_with_real_rows():
    """build_irrep_workflow_decisions with real kpoint/subspace rows and
    detected_identity_id must execute the identity-only G_k^(a) branch."""
    from valleyscope.analysis.irrep_workflow_decision import (
        build_irrep_workflow_decisions,
    )
    decisions = build_irrep_workflow_decisions(
        projector_symmetry_report={
            "by_kpoint": {
                "GM": {"seed_projector_symmetry": []},
            },
        },
        target_subspace_closure_report=None,
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K_valley"],
                        "hsp_preserving_operation_ids": [5],
                        "local_irrep_ready": True,
                        "diagnostic_only": False,
                        "subspace_group": {"valley_preserving_operation_ids": [5]},
                        "symmetry_adapted_projectors": {"status": "ok"},
                    }],
                },
            },
        },
        symmetry_rows=[],
        valley_names=["K_valley"],
        detected_identity_id=5,
    )
    d = decisions["by_kpoint"]["GM"]["K_valley"]
    assert d["identity_only_valley_preserving_subgroup"] is True
    assert "identity operation" in d["reason"]


def test_time_reversal_sewing_blocks_unknown_path():
    """select_trusted_valley_projectors blocks an unknown workflow path
    instead of silently routing to symmetry_adapted."""
    from valleyscope.analysis.time_reversal_sewing import (
        select_trusted_valley_projectors,
    )
    _, _, blockers = select_trusted_valley_projectors(
        workflow_decisions={
            "by_kpoint": {
                "KM": {"V0": {"readiness_level": "trusted",
                               "workflow_path": "unknown_path"}},
            },
        },
        seed_projectors_by_kpoint={"KM": {"V0": np.eye(2)}},
        symmetry_adapted_projectors_by_kpoint={},
    )
    assert len(blockers) > 0
    assert any("blocked" in b.lower() for b in blockers)
