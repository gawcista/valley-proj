from __future__ import annotations

import numpy as np

from valleyscope.analysis.projected_hsp_coverage import (
    build_projected_hsp_coverage_report,
    classify_projected_subspace_kpoint,
    derive_projected_subspace_source_hsp_basis,
)
from valleyscope.irreps.ebr_data_adapter import load_ebr_source_data
from valleyscope.irreps.source_payload import (
    build_source_payload_for_projected_hsp_matching,
)
from valleyscope.irreps.tables import (
    StandardIrrep,
    StandardIrrepTable,
    StandardTableOperation,
    load_standard_irrep_table,
)


def _operation(index: int, rotation: list[list[int]]) -> StandardTableOperation:
    return StandardTableOperation(
        table_index=index,
        rotation_frac=np.asarray(rotation, dtype=int),
        translation_frac=np.zeros(3),
        spin_rotation=np.eye(2, dtype=complex),
        time_reversal=False,
    )


def _irrep(
    label: str,
    hsp: str,
    k_frac: list[float],
    operation_indices: list[int],
) -> StandardIrrep:
    return StandardIrrep(
        label=label,
        kpoint_label=hsp,
        k_frac=np.asarray(k_frac, dtype=float),
        dimension=1,
        characters={index: 1.0 + 0.0j for index in operation_indices},
    )


def _certificate(
    transform: list[list[float]] | None = None,
    *,
    centering_vectors: list[list[float]] | None = None,
    sg_number: int = 5,
    sg_symbol: str = "C2",
    hall_number: int = 9,
    hall_symbol: str = "C 2y",
) -> dict[str, object]:
    return {
        "validation_status": "validated",
        "subspace_sg_number": sg_number,
        "subspace_sg_symbol": sg_symbol,
        "hall_number": hall_number,
        "hall_symbol": hall_symbol,
        "parent_to_standard_direct_transform": transform or np.eye(3).tolist(),
        "normalized_centering_vectors": centering_vectors
        or [[0.0, 0.0, 0.0]],
        "centering_coset_count": len(
            centering_vectors or [[0.0, 0.0, 0.0]]
        ),
        "standard_operation_closure_validated": True,
    }


def _centered_c2_table() -> StandardIrrepTable:
    identity = _operation(1, np.eye(3, dtype=int).tolist())
    c2y = _operation(2, [[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    return StandardIrrepTable(
        number=5,
        name="C2",
        spinor=True,
        operations=(identity, c2y),
        irreps=(
            _irrep("-GM3", "GM", [0.0, 0.0, 0.0], [1, 2]),
            _irrep("-V2", "V", [0.5, 0.5, 0.0], [1]),
            _irrep("-Y3", "Y", [0.0, 1.0, 0.0], [1, 2]),
            _irrep("-A3", "A", [0.0, 0.0, 0.5], [1, 2]),
        ),
    )


def test_primitive_representative_and_centered_star_arm_are_distinct():
    table = _centered_c2_table()
    certificate = _certificate(
        centering_vectors=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
    )
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=["-GM3", "-V2", "-Y3", "-A3"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )

    assert basis["status"] == "validated"
    assert basis["required_source_hsp_labels"] == ["GM", "V", "Y"]

    representative = classify_projected_subspace_kpoint(
        parent_k_frac=[0.5, 0.5, 0.0],
        table=table,
        source_hsp_basis=basis,
        standard_setting_certificate=certificate,
        mapped_standard_little_group_operation_ids=[1],
    )
    assert representative["classification"] == "representative"
    assert representative["source_hsp_label"] == "V"
    assert representative["reciprocal_shift"] == [0, 0, 0]
    assert representative["projected_subspace_space_group"] == {
        "number": 5,
        "symbol": "C2",
        "hall_number": 9,
        "hall_symbol": "C 2y",
    }

    star = classify_projected_subspace_kpoint(
        parent_k_frac=[0.5, -0.5, 0.0],
        table=table,
        source_hsp_basis=basis,
        standard_setting_certificate=certificate,
        mapped_standard_little_group_operation_ids=[1],
    )
    assert star["classification"] == "star_equivalent"
    assert star["source_hsp_label"] == "V"
    assert star["standard_operation_witness"]["table_index"] == 2
    assert star["reciprocal_shift"] == [1, -1, 0]
    assert star["representation_transport_status"] == "validated"


def test_star_geometry_rejects_little_group_content_mismatch():
    table = _centered_c2_table()
    certificate = _certificate(
        centering_vectors=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
    )
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=["-GM3", "-V2", "-Y3"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )

    result = classify_projected_subspace_kpoint(
        parent_k_frac=[0.5, -0.5, 0.0],
        table=table,
        source_hsp_basis=basis,
        standard_setting_certificate=certificate,
        mapped_standard_little_group_operation_ids=[1, 2],
    )

    assert result["classification"] == "unresolved"
    assert result["geometric_classification"] == "star_equivalent"
    assert result["validation_status"] == "blocked"
    assert "little_group_operation_mismatch" in result["blocker"]


def test_source_payload_retains_evaluated_operation_map_on_little_group_mismatch():
    table = _centered_c2_table()
    payload = build_source_payload_for_projected_hsp_matching(
        table=table,
        projected_hsp_classification={
            "classification": "representative",
            "source_hsp_label": "GM",
            "source_irrep_labels": ["-GM3"],
            "representation_transport_status": "validated",
            "standard_little_group_operation_ids": [1, 2],
        },
        detected_operations=[{
            "operation_id": 10,
            "rotation_frac": table.operation_by_index(1).rotation_frac,
            "translation_frac": table.operation_by_index(1).translation_frac,
        }],
        valley_preserving_operation_ids=[10],
    )

    assert payload["status"] == "blocked"
    assert payload["operation_mapping_evaluated"] is True
    assert payload["source_operation_map"] == {10: 1}
    assert payload["blocker_reasons"][0].startswith(
        "little_group_operation_mismatch:"
    )


def test_generic_identity_row_is_valid_local_but_not_source_hsp():
    table = _centered_c2_table()
    certificate = _certificate(
        centering_vectors=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
    )
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=["-GM3", "-V2", "-Y3"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )

    result = classify_projected_subspace_kpoint(
        parent_k_frac=[1.0 / 3.0, 1.0 / 3.0, 0.0],
        table=table,
        source_hsp_basis=basis,
        standard_setting_certificate=certificate,
        mapped_standard_little_group_operation_ids=[1],
    )

    assert result["classification"] == "generic"
    assert result["source_hsp_membership"] is False
    assert result["local_representation_status"] == "valid"
    assert result["validation_status"] == "validated"
    assert result["blocker"] == ""


def test_missing_parent_k_coordinate_is_unresolved_not_gamma():
    table = _centered_c2_table()
    certificate = _certificate()
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=["-GM3", "-V2", "-Y3"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )

    result = classify_projected_subspace_kpoint(
        parent_k_frac=None,
        table=table,
        source_hsp_basis=basis,
        standard_setting_certificate=certificate,
    )

    assert result["classification"] == "unresolved"
    assert result["standard_k_frac"] is None
    assert result["validation_status"] == "blocked"
    assert result["blocker"] == "missing_parent_k_coordinate"


def test_source_basis_requires_affine_operation_closure_evidence():
    certificate = _certificate()
    certificate.pop("standard_operation_closure_validated")

    basis = derive_projected_subspace_source_hsp_basis(
        table=_centered_c2_table(),
        ebr_source_basis_labels=["-GM3", "-V2", "-Y3"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )

    assert basis["status"] == "blocked"
    assert basis["blocker"] == (
        "standard_setting_operation_closure_not_validated"
    )


def test_source_basis_rejects_table_certificate_space_group_mismatch():
    certificate = _certificate()
    certificate["subspace_sg_number"] = 143

    basis = derive_projected_subspace_source_hsp_basis(
        table=_centered_c2_table(),
        ebr_source_basis_labels=["-GM3", "-V2", "-Y3"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )

    assert basis["status"] == "blocked"
    assert basis["blocker"] == (
        "source_table_certificate_space_group_mismatch: table=5, "
        "certificate=143"
    )


def test_unresolved_rational_plane_membership_blocks_source_basis():
    transform = np.eye(3)
    transform[0, 2] = 1.0 / 97.0

    basis = derive_projected_subspace_source_hsp_basis(
        table=_centered_c2_table(),
        ebr_source_basis_labels=["-GM3", "-V2", "-Y3"],
        standard_setting_certificate=_certificate(transform.tolist()),
        use_2d_momentum_only=True,
    )

    assert basis["status"] == "blocked"
    assert basis["blocker"].startswith(
        "source_hsp_plane_membership_unresolved:"
    )


def test_source_plane_uses_reciprocal_transform_not_standard_kz_zero():
    # Swapping parent y/z maps the parent 2D plane to standard ky=0.
    transform = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    table = StandardIrrepTable(
        number=75,
        name="P4",
        spinor=False,
        operations=(_operation(1, np.eye(3, dtype=int).tolist()),),
        irreps=(
            _irrep("Q1", "Q", [0.5, 0.0, 0.5], [1]),
            _irrep("R1", "R", [0.0, 0.5, 0.0], [1]),
        ),
    )

    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=["Q1", "R1"],
        standard_setting_certificate=_certificate(
            transform,
            sg_number=75,
            sg_symbol="P4",
            hall_number=349,
            hall_symbol="P 4",
        ),
        use_2d_momentum_only=True,
    )

    assert basis["required_source_hsp_labels"] == ["Q"]
    q = basis["source_hsps"][0]
    assert q["standard_representative_k_frac"] == [0.5, 0.0, 0.5]
    assert np.allclose(q["inverse_parent_k_frac"], [0.5, 0.5, 0.0])
    assert basis["standard_plane_basis"] == [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_per_valley_coverage_never_combines_complementary_rows():
    table = _centered_c2_table()
    certificate = _certificate(
        centering_vectors=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
    )
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=["-GM3", "-V2", "-Y3"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )

    def row(kpoint: str, valley: str, hsp: str) -> dict[str, object]:
        return {
            "kpoint": kpoint,
            "valley": valley,
            "classification": "representative",
            "geometric_classification": "representative",
            "source_hsp_label": hsp,
            "source_hsp_membership": True,
            "validation_status": "validated",
            "representation_transport_status": "validated",
        }

    report = build_projected_hsp_coverage_report(
        source_hsp_basis_by_valley={"v1": basis, "v2": basis},
        classifications=[
            row("G", "v1", "GM"), row("V", "v1", "V"),
            row("G", "v2", "GM"), row("Y", "v2", "Y"),
        ],
        matching_by_kpoint={
            "G": {"v1": {"matching_status": "matched"}, "v2": {"matching_status": "matched"}},
            "V": {"v1": {"matching_status": "matched"}},
            "Y": {"v2": {"matching_status": "matched"}},
        },
        workflow_decisions_by_kpoint={
            "G": {"v1": {"readiness_level": "trusted"}, "v2": {"readiness_level": "trusted"}},
            "V": {"v1": {"readiness_level": "trusted"}},
            "Y": {"v2": {"readiness_level": "trusted"}},
        },
    )

    v1 = report["by_valley"]["v1"]
    v2 = report["by_valley"]["v2"]
    assert v1["covered_source_hsp_labels"] == ["GM", "V"]
    assert v1["missing_source_hsp_labels"] == ["Y"]
    assert v2["covered_source_hsp_labels"] == ["GM", "Y"]
    assert v2["missing_source_hsp_labels"] == ["V"]
    assert v1["complete"] is False
    assert v2["complete"] is False
    assert report["all_valleys_complete"] is False


def test_missing_hsp_guidance_has_deterministic_inverse_parent_coordinate():
    table = _centered_c2_table()
    certificate = _certificate(
        transform=[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        centering_vectors=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
    )
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=["-GM3", "-V2", "-Y3"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )
    report = build_projected_hsp_coverage_report(
        source_hsp_basis_by_valley={"v": basis},
        classifications=[],
        matching_by_kpoint={},
        workflow_decisions_by_kpoint={},
    )

    missing = report["by_valley"]["v"]["missing_source_hsp_representatives"]
    assert [entry["source_hsp_label"] for entry in missing] == ["GM", "V", "Y"]
    assert all(len(entry["inverse_parent_k_frac"]) == 3 for entry in missing)
    assert all("inverse_reciprocal_shift" in entry for entry in missing)


def test_source_basis_order_comes_from_ebr_basis_not_table_or_sample_order():
    table = _centered_c2_table()
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=["-V2", "-GM3", "-Y3", "-A3"],
        standard_setting_certificate=_certificate(),
        use_2d_momentum_only=True,
    )

    assert basis["required_source_hsp_labels"] == ["V", "GM", "Y"]
    assert basis["provenance"]["basis_order_source"] == (
        "irreptables EBR source basis order"
    )


def test_missing_ebr_source_irrep_label_blocks_basis_derivation():
    basis = derive_projected_subspace_source_hsp_basis(
        table=_centered_c2_table(),
        ebr_source_basis_labels=["-GM3", "missing-source-irrep"],
        standard_setting_certificate=_certificate(),
        use_2d_momentum_only=True,
    )

    assert basis["status"] == "blocked"
    assert basis["blocker"].startswith(
        "ebr_source_irrep_labels_missing_from_standard_table: "
        "['missing-source-irrep']"
    )
    assert "compatible auxiliary source tables=[]" in basis["blocker"]


def test_compatible_auxiliary_source_table_validates_noncanonical_ebr_rows():
    table = load_standard_irrep_table(143, spinor=True)
    source = load_ebr_source_data(143, True)
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=source["source_basis_labels"],
        standard_setting_certificate=_certificate(
            sg_number=143,
            sg_symbol="P3",
            hall_number=430,
            hall_symbol="P 3",
        ),
        use_2d_momentum_only=True,
    )

    assert basis["status"] == "validated"
    assert basis["required_source_hsp_labels"] == ["GM", "K", "M"]
    excluded = basis["provenance"][
        "excluded_noncanonical_ebr_source_rows"
    ]
    assert [row["label"] for row in excluded] == [
        "-HA4", "-HA5", "-HA6", "-KA4", "-KA5", "-KA6"
    ]
    assert {row["source_table"] for row in excluded} == {
        "irreps-SG=143.1-spin.dat"
    }


def test_non_material_synthetic_star_transports_full_little_group_characters():
    identity = _operation(1, np.eye(3, dtype=int).tolist())
    mirror_x = _operation(2, [[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mirror_y = _operation(3, [[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    c2 = _operation(4, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    table = StandardIrrepTable(
        number=25,
        name="Pmm2",
        spinor=False,
        operations=(identity, mirror_x, mirror_y, c2),
        irreps=(
            StandardIrrep(
                label="Q1",
                kpoint_label="Q",
                k_frac=np.asarray([0.25, 0.0, 0.0]),
                dimension=1,
                characters={1: 1.0 + 0.0j, 3: -1.0 + 0.0j},
            ),
            StandardIrrep(
                label="Q2",
                kpoint_label="Q",
                k_frac=np.asarray([0.25, 0.0, 0.0]),
                dimension=1,
                characters={1: 1.0 + 0.0j, 3: 1.0 + 0.0j},
            ),
        ),
    )
    certificate = _certificate(
        sg_number=25,
        sg_symbol="Pmm2",
        hall_number=125,
        hall_symbol="P 2 -2",
    )
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=["Q1"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )
    classification = classify_projected_subspace_kpoint(
        parent_k_frac=[-0.25, 0.0, 0.0],
        table=table,
        source_hsp_basis=basis,
        standard_setting_certificate=certificate,
    )
    assert classification["classification"] == "star_equivalent"
    assert classification["standard_little_group_operation_ids"] == [1, 3]

    payload = build_source_payload_for_projected_hsp_matching(
        table=table,
        projected_hsp_classification=classification,
        detected_operations=[
            {
                "operation_id": 10,
                "rotation_frac": identity.rotation_frac,
                "translation_frac": identity.translation_frac,
            },
            {
                "operation_id": 11,
                "rotation_frac": mirror_y.rotation_frac,
                "translation_frac": mirror_y.translation_frac,
            },
        ],
        valley_preserving_operation_ids=[10, 11],
    )

    assert payload["status"] == "ok"
    assert set(payload["source_irrep_characters"]) == {"Q1"}
    assert payload["source_operation_map"] == {10: 1, 11: 3}
    assert payload["source_irrep_characters"]["Q1"] == {
        1: 1.0 + 0.0j,
        3: -1.0 + 0.0j,
    }
    assert payload["provenance"]["character_transport_status"] == "validated"


def test_spinful_star_transport_applies_double_group_lift_factor():
    sigma_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sigma_z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    def spin_op(
        index: int,
        rotation: list[list[int]],
        spin_rotation: np.ndarray,
    ) -> StandardTableOperation:
        return StandardTableOperation(
            table_index=index,
            rotation_frac=np.asarray(rotation, dtype=int),
            translation_frac=np.zeros(3),
            spin_rotation=spin_rotation,
            time_reversal=False,
        )

    identity = spin_op(1, np.eye(3, dtype=int).tolist(), np.eye(2))
    c2x = spin_op(2, [[1, 0, 0], [0, -1, 0], [0, 0, -1]], -1.0j * sigma_x)
    c2y = spin_op(3, [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], -1.0j * sigma_y)
    c2z = spin_op(4, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], -1.0j * sigma_z)
    table = StandardIrrepTable(
        number=16,
        name="P222",
        spinor=True,
        operations=(identity, c2x, c2y, c2z),
        irreps=(StandardIrrep(
            label="-Q1",
            kpoint_label="Q",
            k_frac=np.asarray([0.0, 0.25, 0.0]),
            dimension=1,
            characters={1: 1.0 + 0.0j, 3: 0.0 + 1.0j},
        ),),
    )
    certificate = _certificate(
        sg_number=16,
        sg_symbol="P222",
        hall_number=108,
        hall_symbol="P 2 2",
    )
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=["-Q1"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )
    classification = classify_projected_subspace_kpoint(
        parent_k_frac=[0.0, -0.25, 0.0],
        table=table,
        source_hsp_basis=basis,
        standard_setting_certificate=certificate,
    )

    assert classification["classification"] == "star_equivalent"
    assert classification["standard_operation_witness"]["table_index"] == 2
    transport = {
        row["star_arm_operation_id"]: row
        for row in classification["character_transport_map"]
    }
    assert transport[1]["spin_lift_factor"] == 1
    assert transport[3]["spin_lift_factor"] == -1

    payload = build_source_payload_for_projected_hsp_matching(
        table=table,
        projected_hsp_classification=classification,
        detected_operations=[
            {
                "operation_id": 10,
                "rotation_frac": identity.rotation_frac,
                "translation_frac": identity.translation_frac,
            },
            {
                "operation_id": 11,
                "rotation_frac": c2y.rotation_frac,
                "translation_frac": c2y.translation_frac,
            },
        ],
        valley_preserving_operation_ids=[10, 11],
    )

    assert payload["status"] == "ok"
    assert payload["source_irrep_characters"]["-Q1"] == {
        1: 1.0 + 0.0j,
        3: 0.0 - 1.0j,
    }


def test_star_character_transport_with_lattice_phase_fails_closed():
    table = _centered_c2_table()
    payload = build_source_payload_for_projected_hsp_matching(
        table=table,
        projected_hsp_classification={
            "classification": "star_equivalent",
            "source_hsp_label": "V",
            "source_irrep_labels": ["-V2"],
            "representation_transport_status": "validated",
            "standard_little_group_operation_ids": [1],
            "character_transport_map": [{
                "source_representative_operation_id": 1,
                "star_arm_operation_id": 1,
                "affine_lattice_translation": [1, 0, 0],
            }],
        },
        detected_operations=[{
            "operation_id": 10,
            "rotation_frac": table.operation_by_index(1).rotation_frac,
            "translation_frac": table.operation_by_index(1).translation_frac,
        }],
        valley_preserving_operation_ids=[10],
    )

    assert payload["status"] == "blocked"
    assert payload["blocker_reasons"][0].startswith(
        "star_character_transport_requires_bloch_phase:"
    )
