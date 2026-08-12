"""Data-driven acceptance matrix for space-group-generic portability.

Six required cases, all executing without skip.  Uses repository-supported
APIs (load_standard_irrep_table, derive_irreptables_standard_setting_identity,
build_auto_canonical_reduced_ebr_table) and irreptables EBR data loaders.

Production regressions for content-based identity detection are included.
"""

import numpy as np
import pytest


# ============================================================================
# Case 1 — Non-cyclic HSP little group with more than one generator
# ============================================================================

def test_non_cyclic_matcher_runs_on_full_sg123_gamma_gka():
    """SG 123 (P4/mmm) at Gamma has 16 non-cyclic operations.  The generic
    restricted-character matcher must decompose the trivial computed
    character into exactly one copy of the reviewed GM1+ irrep.

    All source table indices are remapped to shuffled opaque ValleyScope
    IDs with the identity NOT first; no early return is allowed.
    """
    from valleyscope.analysis.generic_irrep_matching import match_restricted_characters
    from valleyscope.irreps.tables import load_standard_irrep_table

    table = load_standard_irrep_table(123, spinor=False)
    op_indices = table.operation_indices_for_kpoint("GM")
    assert len(op_indices) == 16

    # Reviewed operation inventory: rotations must form a NON-cyclic set.
    # Proof: two distinct reviewed rotations do not commute.
    rotations: dict[int, np.ndarray] = {}
    for idx in op_indices:
        op = table.operation_by_index(idx)
        rotations[idx] = np.asarray(op.rotation_frac, dtype=int)
    noncommuting = False
    for i, left in enumerate(op_indices):
        for right in op_indices[i + 1:]:
            if not np.array_equal(
                rotations[left] @ rotations[right],
                rotations[right] @ rotations[left],
            ):
                noncommuting = True
                break
        if noncommuting:
            break
    assert noncommuting, "SG 123 Gamma little group must be non-cyclic"

    # Identity detected by content (rotation=I, translation=0), not by index.
    identity_idx = next(
        idx for idx in op_indices
        if np.array_equal(rotations[idx], np.eye(3, dtype=int))
        and not np.any(
            np.asarray(
                table.operation_by_index(idx).translation_frac, dtype=float
            ) % 1.0
        )
    )

    # Shuffled opaque ValleyScope IDs; the identity is NOT the first/minimum.
    vs_id: dict[int, int] = {idx: 31 + (3 * idx) % 16 for idx in op_indices}
    assert len(set(vs_id.values())) == 16
    assert vs_id[identity_idx] != min(vs_id.values())

    vp_ids = [vs_id[idx] for idx in op_indices]
    # All Gamma irreps of the reviewed SG 123 table, keyed by SOURCE
    # table index (the explicit operation map translates VS IDs).
    src_chars: dict[str, dict[int, complex]] = {}
    for irr in table.irreps_by_kpoint("GM"):
        src_chars[irr.label] = {
            idx: complex(float(ch.real), float(ch.imag))
            for idx, ch in irr.characters.items()
        }
    assert len(src_chars) == 10

    # Computed characters equal the reviewed trivial irrep GM1+.
    computed = {vs: 1.0 + 0j for vs in vp_ids}

    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=src_chars,
        valley_preserving_operation_ids=vp_ids,
        source_operation_map={vs_id[idx]: idx for idx in op_indices},
        detected_identity_id=vs_id[identity_idx],
    )
    assert result["matching_status"] == "matched"
    assert result["irrep_multiplicities"] == {"GM1+": 1}
    assert result["diagnostic_only"] is False


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
    The target is a reviewed reduced-EBR table column; the solve must be
    exactly the reviewed non-unique atomic-compatible result with the two
    exact witnesses and full setting provenance."""
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
    assert irrep_list == ["GM:-GM3", "GM:-GM4"]
    assert len(ebr_list) == 4

    # Bind the target to an actual reviewed table column, not a handwritten
    # vector: column 1 is shared by the 2a and 2b -1E↑G(1) EBRs.
    witness_2b = [e for e in ebr_list if e["label"] == "-1E↑G(1) @ 2b(2,2)"]
    witness_2a = [e for e in ebr_list if e["label"] == "-1E↑G(1) @ 2a(2,2)"]
    assert len(witness_2b) == 1 and len(witness_2a) == 1
    target = list(witness_2b[0]["vector"])
    assert target == list(witness_2a[0]["vector"]) == [0, 1]
    ebr_vectors = [ebr["vector"] for ebr in ebr_list]
    ebr_labels_list = [ebr.get("label", "") for ebr in ebr_list]

    classification = classify_bundle(
        target=target, ebr_vectors=ebr_vectors,
        ebr_labels=ebr_labels_list, max_coefficient=10,
    )
    assert classification is not None
    assert classification["status"] == "solved_exact"
    assert classification["classification"] == "atomic-compatible-candidate"
    assert classification["decomposition_uniqueness"] == "non_unique"
    # Exact witnesses: primary 2b, second 2a, each with coefficient 1.
    decomp = classification["ebr_decomposition"]
    assert decomp == [{"label": "-1E↑G(1) @ 2b(2,2)", "coefficient": 1}]
    witnesses = classification["decomposition_witnesses"]
    assert witnesses == [
        [{"label": "-1E↑G(1) @ 2b(2,2)", "coefficient": 1}],
        [{"label": "-1E↑G(1) @ 2a(2,2)", "coefficient": 1}],
    ]

    # Convention provenance: reviewed irreptables source, SG 5, spinful,
    # Hall 9 with Hall symbol, C centering, two centering cosets,
    # primitive/conventional index 2, validated standard-operation closure,
    # sampled HSP basis, and reduced irrep basis.
    provenance = result["provenance"]
    assert provenance is not None
    assert provenance["package"] == "irreptables"
    assert provenance["data_source"] == "irreptables"
    assert provenance["space_group_number"] == 5
    assert provenance["spinful"] is True
    assert provenance["sampled_bilbao_hsps"] == ["GM"]
    assert provenance["valleyscope_reduction"] == "sampled_hsp_valley_preserving"
    assert provenance["reduction_basis_count"] == 2
    setting = provenance["standard_setting_identity"]
    assert setting["status"] == "unique_match"
    assert setting["hall_number"] == 9
    assert setting["hall_symbol"] == "C 2y"
    assert setting["space_group_number"] == 5
    assert setting["space_group_symbol"] == "C2"
    assert setting["centering_type"] == "C"
    assert len(setting["centering_cosets"]) == 2
    assert setting["primitive_conventional_index"] == 2
    assert setting["standard_setting_operation_count"] == 4
    assert setting["standard_operation_closure_validated"] is True
    assert "irreptables.StandardIrrepTable" in setting["source"]


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


def test_detect_identity_missing_translation_is_not_identity():
    """Missing translation_frac is unknown evidence, NOT zero translation."""
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    ev = detect_identity_operation(operations=[
        {"operation_id": 7, "rotation_frac": np.eye(3, dtype=int)},
    ])
    assert ev["found"] is False
    assert "not_detected" in ev["reason"]


def test_detect_identity_missing_rotation_is_not_identity():
    """Missing rotation_frac is unknown evidence, NOT identity."""
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    ev = detect_identity_operation(operations=[
        {"operation_id": 7, "translation_frac": np.zeros(3)},
    ])
    assert ev["found"] is False
    assert "not_detected" in ev["reason"]


def test_detect_identity_malformed_shapes_are_not_identity():
    """Malformed rotation/translation shapes are non-identity evidence."""
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    for op in (
        {"operation_id": 1, "rotation_frac": np.eye(2),
         "translation_frac": np.zeros(3)},
        {"operation_id": 2, "rotation_frac": np.eye(3),
         "translation_frac": np.zeros(2)},
        {"operation_id": 3, "rotation_frac": [[1, 0], [0, 1], [0, 0]],
         "translation_frac": np.zeros(3)},
    ):
        ev = detect_identity_operation(operations=[op])
        assert ev["found"] is False
        assert "not_detected" in ev["reason"]


def test_detect_identity_nonfinite_values_are_not_identity():
    """NaN/Inf rotation or translation entries are non-identity evidence."""
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    nan_rot = np.eye(3)
    nan_rot[0, 0] = np.nan
    inf_trans = np.array([np.inf, 0.0, 0.0])
    ev_rot = detect_identity_operation(operations=[
        {"operation_id": 1, "rotation_frac": nan_rot,
         "translation_frac": np.zeros(3)},
    ])
    ev_trans = detect_identity_operation(operations=[
        {"operation_id": 2, "rotation_frac": np.eye(3),
         "translation_frac": inf_trans},
    ])
    assert ev_rot["found"] is False
    assert ev_trans["found"] is False


def test_detect_identity_centering_coset_translation_is_not_identity():
    """A centering-coset translation (0.5, 0.5, 0) with rotation I is not
    the identity."""
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    ev = detect_identity_operation(operations=[
        {"operation_id": 9, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.array([0.5, 0.5, 0.0])},
    ])
    assert ev["found"] is False
    assert "not_detected" in ev["reason"]


def test_detect_identity_duplicate_affine_identities_ambiguous():
    """Two operations with the same affine identity content are ambiguous."""
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    ev = detect_identity_operation(operations=[
        {"operation_id": 3, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
        {"operation_id": 8, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
    ])
    assert ev["found"] is False
    assert ev["reason"] == "identity_operation_ambiguous"
    assert sorted(ev["candidate_ids"], key=str) == [3, 8]


def test_detect_identity_invalid_tolerance_raises():
    """Nonfinite or negative tolerance raises ValueError."""
    from valleyscope.analysis.valley_little_group import detect_identity_operation
    ops = [
        {"operation_id": 1, "rotation_frac": np.eye(3, dtype=int),
         "translation_frac": np.zeros(3)},
    ]
    for bad_tol in (np.nan, np.inf, -1.0):
        with pytest.raises(ValueError, match="tolerance"):
            detect_identity_operation(operations=ops, tolerance=bad_tol)


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
    """Coverage of an existing invariant: select_trusted_valley_projectors
    fails closed on a workflow path outside the trusted projector registry
    (unknown_path) instead of silently routing to a source map.  This is
    not a regression for the (removed) inner unreachable branch."""
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
