"""Translation-lattice equivalence: T Z^3 == L_standard.

A parent-to-standard direct transform may be trusted only when it maps the
primitive parent translation lattice onto the standard-setting lattice
``L_standard = Z^3 + sum_i Z c_i``.  ``|det T| = 1/m`` alone is not that
statement: rescalings with the right index map onto a different lattice.

Expected lattices in this file are built from reviewed centering cosets and
from exact mod-1 membership arithmetic; the production Hermite-normal-form
basis is never used as its own oracle.
"""

from itertools import product

import numpy as np
import pytest
import spglib

from valleyscope.analysis.standard_setting_kmap import (
    _centering_cosets_from_hall_database,
    _standard_translation_lattice_basis,
    _validate_translation_lattice_equivalence,
    resolve_standard_setting_hsp_label,
)
from valleyscope.irreps.tables import load_standard_irrep_table


P_COSETS = [[0.0, 0.0, 0.0]]
C_COSETS = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
# Column basis of the C-centered lattice: Z(1/2,1/2,0) + Z(-1/2,1/2,0) + Z(0,0,1).
C_LATTICE_BASIS = np.array(
    [[0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
)


# --- the lattice gate matrix -------------------------------------------------

@pytest.mark.parametrize(
    "cosets,transform,accepted",
    [
        # Primitive standard setting: only unimodular integer rebases survive.
        (P_COSETS, np.diag([2.0, 0.5, 1.0]), False),
        (P_COSETS, np.diag([2.0, 2.0, 0.25]), False),
        (P_COSETS, np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), True),
        (P_COSETS, np.diag([-1.0, 1.0, 1.0]), True),
        # C-centered standard setting: the transform must generate the
        # centered lattice itself, not one of the other index-2 lattices.
        (C_COSETS, C_LATTICE_BASIS, True),
        (C_COSETS, np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]), True),
        (C_COSETS, np.array([[1.0, 1.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]), True),
        (C_COSETS, np.diag([0.5, 1.0, 1.0]), False),
        (C_COSETS, np.diag([2.0, 2.0, 0.25]), False),
        (C_COSETS, np.eye(3), False),
        (C_COSETS, C_LATTICE_BASIS @ np.diag([2.0, 1.0, 1.0]), False),
    ],
)
def test_translation_lattice_gate_matrix(cosets, transform, accepted):
    """Explicit gate matrix: index alone must not certify the lattice."""
    result = _validate_translation_lattice_equivalence(
        np.asarray(transform, dtype=float), cosets,
    )
    assert (result["status"] == "passed") is accepted
    assert result["centering_index"] == len(cosets)
    if accepted:
        assert result["lattice_index"] == 1
        assert result["integer_residual"] <= 1e-9
        assert result["rebase_residual"] <= 1e-9
    else:
        assert result["missing_ingredient"] == "non_lattice_direct_transform"
        assert result["reason"]


def test_index_preserving_rescaling_commutes_with_the_standard_rotations():
    """diag(2, 2, 1/4) leaves every P3 rotation integral and still fails.

    This is the determinant-plus-rotation-matching gap: conjugating a
    hexagonal operation set by that diagonal preserves its in-plane integers.
    """
    transform = np.diag([2.0, 2.0, 0.25])
    transform_inv = np.linalg.inv(transform)
    rotations = spglib.get_symmetry_from_database(430)["rotations"]
    for rotation in rotations:
        conjugated = transform @ np.asarray(rotation, float) @ transform_inv
        assert np.allclose(conjugated, np.rint(conjugated), atol=1e-12)
    assert abs(float(np.linalg.det(transform))) == pytest.approx(1.0)

    result = _validate_translation_lattice_equivalence(transform, P_COSETS)
    assert result["status"] == "failed"
    assert result["missing_ingredient"] == "non_lattice_direct_transform"


def test_non_integral_combination_index_is_rejected():
    """A transform that is an integer combination but not unit index fails."""
    transform = C_LATTICE_BASIS @ np.diag([2.0, 1.0, 1.0])
    result = _validate_translation_lattice_equivalence(transform, C_COSETS)
    assert result["status"] == "failed"
    assert result["integer_residual"] <= 1e-9
    assert result["lattice_index"] == 2
    assert "lattice_index_mismatch" in result["reason"]


# --- independent oracle for the standard lattice basis -----------------------

def _membership_in_coset_lattice(vector, cosets, tolerance=1e-8):
    """Brute-force membership of one fractional vector in Z^3 + sum Z c_i."""
    reduced = np.asarray(vector, dtype=float)
    reduced = reduced - np.floor(reduced + tolerance)
    index = len(cosets)
    for counts in product(range(index), repeat=index):
        candidate = np.zeros(3)
        for count, coset in zip(counts, cosets):
            candidate = candidate + count * np.asarray(coset, dtype=float)
        difference = reduced - candidate
        if np.allclose(difference, np.rint(difference), atol=tolerance):
            return True
    return False


@pytest.mark.parametrize(
    "hall_number,centering",
    [(430, "P"), (9, "C"), (353, "I"), (122, "F"), (458, "R")],
)
def test_hnf_basis_generates_exactly_the_reviewed_centering_lattice(
    hall_number, centering,
):
    """Index plus mutual containment, without trusting the HNF result itself."""
    evidence = _centering_cosets_from_hall_database(hall_number)
    assert evidence["status"] == "passed", evidence
    cosets = evidence["centering_cosets"]
    index = evidence["primitive_conventional_index"]
    assert len(cosets) == index

    basis_evidence = _standard_translation_lattice_basis(cosets)
    assert basis_evidence["status"] == "passed", basis_evidence
    assert basis_evidence["basis_determinant_abs"] == pytest.approx(1.0 / index)
    basis = np.asarray(basis_evidence["basis"], dtype=float)

    # Every basis column lies in the reviewed coset lattice ...
    for column in range(3):
        assert _membership_in_coset_lattice(basis[:, column], cosets), (
            f"{centering}: basis column {column} is outside the coset lattice"
        )
    # ... and every coset is an integer combination of the basis columns.
    for coset in cosets:
        combination = np.linalg.solve(basis, np.asarray(coset, dtype=float))
        assert np.allclose(combination, np.rint(combination), atol=1e-9), (
            f"{centering}: coset {coset} is not generated by the basis"
        )


# --- reviewed source-derived centering cases ---------------------------------

@pytest.mark.parametrize(
    "hall_number,centering",
    [(430, "P"), (9, "C"), (353, "I"), (122, "F"), (458, "R")],
)
def test_source_derived_centering_lattices_accept_and_reject(
    hall_number, centering,
):
    """Every centering type: the true rebase passes, a same-volume decoy fails.

    The decoy ``diag(1/m, 1, 1)`` has the determinant of the true primitive
    basis of an index-``m`` centered lattice but generates a different lattice,
    so only the explicit lattice equality separates the two.
    """
    evidence = _centering_cosets_from_hall_database(hall_number)
    assert evidence["status"] == "passed", evidence
    cosets = evidence["centering_cosets"]
    index = evidence["primitive_conventional_index"]
    basis = np.asarray(
        _standard_translation_lattice_basis(cosets)["basis"], dtype=float
    )
    shear = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    for transform in (basis, basis @ shear, -basis):
        result = _validate_translation_lattice_equivalence(transform, cosets)
        assert result["status"] == "passed", (centering, transform, result)

    # A diagonal sign flip preserves the primitive, C, I and F lattices, but
    # not the rhombohedral lattice: (2/3, 2/3, 2/3) is not a coset of R.
    flipped = _validate_translation_lattice_equivalence(
        np.diag([-1.0, 1.0, 1.0]) @ basis, cosets,
    )
    if centering == "R":
        assert flipped["status"] == "failed", flipped
        assert flipped["missing_ingredient"] == "non_lattice_direct_transform"
    else:
        assert flipped["status"] == "passed", (centering, flipped)

    decoy = np.diag([1.0 / index, 1.0, 1.0])
    if index > 1:
        result = _validate_translation_lattice_equivalence(decoy, cosets)
        assert result["status"] == "failed", (centering, result)
        assert result["missing_ingredient"] == "non_lattice_direct_transform"
        assert abs(result["transform_determinant"]) == pytest.approx(
            result["expected_transform_abs_determinant"]
        )


def test_malformed_centering_evidence_is_not_trusted():
    """Unknown or malformed cosets are unresolved, never assumed primitive."""
    for cosets in (None, [], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.0, 0.0]]):
        result = _validate_translation_lattice_equivalence(
            np.eye(3), cosets,
        )
        assert result["status"] != "passed"
    # Missing cosets are unknown, not primitive: identity is not accepted.
    assert _validate_translation_lattice_equivalence(np.eye(3), None)["status"] == "unresolved"


# --- full resolver -----------------------------------------------------------

def test_resolver_rejects_non_lattice_transform_with_lattice_blocker():
    """The audit probe: a real table + complete operations + illegal T."""
    operations = [
        {
            "operation_id": index,
            "rotation_frac": np.asarray(rotation, dtype=float).tolist(),
            "translation_frac": np.asarray(translation, dtype=float).tolist(),
        }
        for index, (rotation, translation) in enumerate(
            zip(
                spglib.get_symmetry_from_database(430)["rotations"],
                spglib.get_symmetry_from_database(430)["translations"],
            )
        )
    ]
    table = load_standard_irrep_table(143, spinor=True)
    k_frac = np.array([1.0, 0.0, 0.0])
    assert table.match_kpoint_label(k_frac) == "GM"

    label, blocker, provenance = resolve_standard_setting_hsp_label(
        k_frac=k_frac,
        table=table,
        standard_match=dict(
            number=143,
            hall_number=430,
            hall_symbol="P 3",
            international_short="P3",
            operation_ids=[0, 1, 2],
        ),
        lattice_direct_cart=np.array(
            [[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]]
        ),
        detected_operations=operations,
        parent_to_standard_direct_transform=np.diag([2.0, 2.0, 0.25]),
        transform_provenance="audit_invalid_lattice_transform",
    )

    assert label is None
    assert blocker is not None
    assert "translation-lattice rebase" in blocker
    certificate = provenance["standard_setting_certificate"]
    assert certificate["validation_status"] != "validated"
    assert certificate["translation_validation_status"] != "passed"
    assert "non_lattice_direct_transform" in certificate[
        "missing_affine_ingredients"
    ]
    assert certificate.get("resolved_hsp_label") is None


def _parent_operations_from_table(table, transform, operation_ids):
    """Parent-frame operation inventory for a consistent basis transform."""
    transform_inv = np.linalg.inv(transform)
    operations = []
    for operation_id, operation in zip(operation_ids, table.operations):
        rotation = transform_inv @ operation.rotation_frac @ transform
        assert np.allclose(rotation, np.rint(rotation), atol=1e-12)
        operations.append({
            "operation_id": operation_id,
            "rotation_frac": np.rint(rotation).astype(int).tolist(),
            "translation_frac": (
                transform_inv @ operation.translation_frac
            ).tolist(),
        })
    return operations


def test_resolver_accepts_unimodular_rebase_of_the_same_lattice():
    """A shear is a lattice-preserving rebase and must still resolve."""
    table = load_standard_irrep_table(143, spinor=False)
    transform = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    operation_ids = [5, 11, 17]
    operations = _parent_operations_from_table(table, transform, operation_ids)
    k_standard = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
    k_parent = transform.T @ k_standard

    label, blocker, provenance = resolve_standard_setting_hsp_label(
        k_frac=k_parent,
        table=table,
        standard_match=dict(
            number=143,
            hall_number=430,
            hall_symbol="P 3",
            international_short="P3",
            operation_ids=operation_ids,
        ),
        detected_operations=operations,
        parent_to_standard_direct_transform=transform,
        transform_provenance="reviewed_test_unimodular_rebase",
    )

    assert blocker is None, provenance
    assert label == table.match_kpoint_label(k_standard)
    certificate = provenance["standard_setting_certificate"]
    assert certificate["validation_status"] == "validated"
    assert certificate["translation_validation_status"] == "passed"
    assert certificate["missing_affine_ingredients"] == []
    lattice = certificate.get("translation_lattice_equivalence")
    assert not isinstance(lattice, dict) or lattice["status"] == "passed"
