"""Tests for standard-setting HSP k-coordinate mapping."""

import numpy as np
import pytest
import spglib



def _p3_ops_from_spglib(*, override: dict | None = None, skip_id: int | None = None):
    import numpy as np
    sym = spglib.get_symmetry_from_database(430)
    ops = []
    for i in range(len(sym["rotations"])):
        if skip_id is not None and i == skip_id:
            continue
        op = {"operation_id": i,
              "rotation_frac": np.asarray(sym["rotations"][i], dtype=float).tolist(),
              "translation_frac": np.asarray(sym["translations"][i], dtype=float).tolist()}
        ops.append(op)
    if override is not None:
        found = False
        for op in ops:
            if op["operation_id"] == override.get("operation_id"):
                op.update(override)
                found = True
                break
        if not found:
            ops.append(override)
    return ops


def _detected_std_ops(hall_number, ids):
    sym = spglib.get_symmetry_from_database(int(hall_number))
    return [{"operation_id": i,
             "rotation_frac": np.asarray(sym["rotations"][i], float).tolist(),
             "translation_frac": np.asarray(sym["translations"][i], float).tolist()}
            for i in ids]

from valleyscope.analysis.standard_setting_kmap import (
    resolve_standard_setting_hsp_label,
    _attempt_setting_transform,
    _verify_operation_basis,
)


# ---------------------------------------------------------------------------
# Synthetic coordinate-match tests
# ---------------------------------------------------------------------------

class _FakeTable:
    """Minimal mock matching the match_kpoint_label interface."""
    def __init__(self, *, labels: dict[str, tuple[float, ...]]):
        self._labels = {k: np.asarray(v, dtype=float) for k, v in labels.items()}

    def match_kpoint_label(self, k_frac, *, tolerance=1e-6):
        k = np.asarray(k_frac, dtype=float)
        for label, coord in self._labels.items():
            delta = k - coord
            delta -= np.rint(delta)
            if np.linalg.norm(delta) <= tolerance:
                return label
        return None

    number = 143
    name = "P3"
    spinor = True


def _table_p3():
    return _FakeTable(labels={"GM": (0, 0, 0), "K": (1/3, 1/3, 0), "M": (1/2, 0, 0)})


def _table_c2():
    return _FakeTable(labels={
        "GM": (0, 0, 0), "Y": (0, 1/2, 0), "M": (1/2, 1/2, 0),
        "A": (0, 0, 1/2), "L": (0, 1/2, 1/2), "V": (1/2, 1/2, 1/2),
    })


# --- Direct match tests ---

def test_direct_coordinate_match_succeeds():
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=_table_p3(),
        standard_match=None,
    )
    assert label == "GM"
    assert blocker is None
    assert prov["direct_match_succeeded"] is True


def test_direct_coordinate_match_near_zero_mod_lattice():
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([1.0, 1.0, 0.0]),
        table=_table_p3(),
        standard_match=None,
    )
    assert label == "GM"


def test_direct_coordinate_match_m_point():
    label, _, _ = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_table_p3(),
        standard_match=None,
    )
    assert label == "M"


# --- Unresolved with provenance tests ---

def test_no_match_no_standard_match_returns_blocker():
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match=None,
    )
    assert label is None
    assert blocker is not None
    assert "standard_setting_hsp_mapping_unresolved" in blocker
    assert prov["direct_match_succeeded"] is False


def test_no_match_with_standard_match_returns_detailed_blocker():
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5,
            "international_short": "C2",
            "hall_number": 9,
            "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert label is None
    assert blocker is not None
    assert "standard_setting_hsp_mapping_unresolved" in blocker
    assert "C2" in blocker
    assert "No. 5" in blocker
    assert prov["hall_number"] == 9
    assert prov["hall_symbol"] == "C 2y"
    assert "setting_transform" in prov


# --- P3 direct match with standard_match still works ---

def test_p3_m_direct_match_with_standard_match():
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 143,
            "international_short": "P3",
            "hall_number": 430,
            "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        detected_operations=_detected_std_ops(430, [0, 1, 2]),
    )
    assert label == "M"
    assert blocker is None
    assert prov["direct_match_succeeded"] is True


# --- Centered-setting transform provenance tests ---

def test_centered_hall_symbol_documented_as_unavailable():
    result = _attempt_setting_transform(
        k_frac=np.array([0.5, 0.0, 0.0]),
        hall_number=9,
        hall_symbol="C 2y",
        sg_number=5,
    )
    assert result["sg_number"] == 5
    assert result["reason"] is not None
    assert "centered" in str(result["reason"]).lower()


def test_centered_setting_documented():
    result = _attempt_setting_transform(
        k_frac=np.array([0.5, 0.0, 0.0]),
        hall_number=9,
        hall_symbol="C 2y",
        sg_number=5,
    )
    assert "C 2y" in str(result.get("reason", ""))


# --- Primitive-setting direct match remains coordinate based ---

def test_p3_m_point_matches_direct():
    """M-point (1/2, 0, 0) in P3 matches directly; no transform needed."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 143,
            "international_short": "P3",
            "hall_number": 430,
            "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        detected_operations=_detected_std_ops(430, [0, 1, 2]),
    )
    assert label == "M"
    assert blocker is None


# ---------------------------------------------------------------------------
# Basis transform integration tests
# ---------------------------------------------------------------------------

def test_basis_transform_no_lattice_returns_unavailable():
    from valleyscope.analysis.standard_setting_kmap import (
        _compute_standard_setting_basis_transform,
    )
    result = _compute_standard_setting_basis_transform(
        lattice_direct_cart=None,
        vp_operations=None,
        standard_match={"operation_ids": [0, 4]},
    )
    assert result["status"] == "unavailable"
    assert "lattice" in result.get("reason", "")


def test_basis_transform_centered_setting_requires_explicit_cell_transform():
    """Centered settings cannot be accepted from rotation axes alone."""
    from valleyscope.analysis.standard_setting_kmap import (
        _compute_standard_setting_basis_transform,
    )
    result = _compute_standard_setting_basis_transform(
        lattice_direct_cart=np.eye(3),
        vp_operations=[{"operation_id": 4, "order": 2,
                         "rotation_frac": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]}],
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert result["status"] == "unavailable"
    assert "transform_matrix" not in result
    assert result["operation_basis_verification"]["status"] == "not_attempted"
    assert "rotation matrices alone" in result["reason"]


def test_resolver_includes_basis_transform_in_provenance():
    _, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert blocker is not None
    bt = prov.get("basis_transform")
    assert bt is not None
    assert bt.get("status") == "unavailable"
    assert "reason" in bt


def test_hexagonal_lattice_subgroup_reconstruction_requires_verification():
    """Centered subgroup mapping stays blocked without an explicit cell transform."""
    from valleyscope.analysis.standard_setting_kmap import (
        _compute_standard_setting_basis_transform,
    )
    a = 3.5; c = 20.0
    hex_lattice = np.array([[a, 0, 0], [-a/2, a*np.sqrt(3)/2, 0], [0, 0, c]])
    result = _compute_standard_setting_basis_transform(
        lattice_direct_cart=hex_lattice,
        vp_operations=[{"operation_id": 4, "order": 2,
                         "rotation_frac": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]}],
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert result["status"] == "unavailable"
    assert result["operation_basis_verification"]["status"] == "not_attempted"
    assert "centered" in result["operation_basis_verification"]["reason"]


def test_operation_basis_verification_does_not_round_away_shear():
    """A sheared transform that only matches after rounding must fail."""
    rot = np.diag([-1.0, 1.0, -1.0])
    sheared_transform = np.array([
        [1.0, 0.2, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    result = _verify_operation_basis(
        parent_rotations=[rot],
        std_rotations=[rot],
        transform_matrix=sheared_transform,
    )

    assert result["status"] == "failed"
    assert result["unmatched_count"] == 1


# ---------------------------------------------------------------------------
# Explicit transform source interface tests
# ---------------------------------------------------------------------------

def test_explicit_transform_used_even_when_parent_coordinate_matches():
    """Explicit standard-setting transform defines the trusted convention."""
    T = np.eye(3)
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        parent_to_standard_direct_transform=T,
    )
    assert label == "M"
    assert blocker is None
    assert prov["direct_match_succeeded"] is False
    assert "explicit_transform" in prov


def test_explicit_transform_resolves_nonmatching_parent_kpoint():
    """Explicit direct transform uses k_std = T^(-T) k_parent."""
    # T maps direct coordinates x_parent -> x_std.  For reciprocal
    # coordinates, k_std = T^(-T) k_parent.  This transform maps
    # parent k=(1/4,0,0) to standard k=(1/2,0,0), i.e. the P3 M label.
    T = np.diag([0.5, 1.0, 1.0])
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.25, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        parent_to_standard_direct_transform=T,
    )
    assert label == "M"
    assert blocker is None
    tf = prov.get("explicit_transform", {})
    assert tf.get("status") == "valid"
    assert prov["explicit_transformed_match_succeeded"] is True
    np.testing.assert_allclose(prov["transformed_k_frac"], [0.5, 0.0, 0.0])
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "validated"
    assert cert["transform_provenance"] == "explicit_user_input"
    np.testing.assert_allclose(
        cert["parent_to_standard_direct_transform"],
        T,
    )


def test_explicit_transform_success_records_transform_candidate():
    """Accepted explicit transform carries transform_candidate provenance."""
    T = np.diag([0.5, 1.0, 1.0])
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.25, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        parent_to_standard_direct_transform=T,
        detected_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.0, 0.0, 0.0],
        }],
    )
    # Bijection: only 1 of 3 ops → label not trusted (unresolved).
    tc = prov["transform_candidate"]
    assert tc["transform_provenance"] == "explicit_user_input"


def test_explicit_transform_takes_precedence_over_direct_parent_match():
    """Supplied standard-cell transform defines the trusted HSP coordinates."""
    T = np.diag([2.0, 1.0, 1.0])
    table = _FakeTable(labels={
        "PARENT_M": (0.5, 0.0, 0.0),
        "STD_X": (0.25, 0.0, 0.0),
    })

    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=table,
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        parent_to_standard_direct_transform=T,
        transform_provenance="unit-test parent-to-standard transform",
        origin_shift_fractional=np.array([0.25, 0.0, 0.0]),
        detected_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.0, 0.0, 0.0],
        }],
    )

    # Bijection: only 1 of 3 P3 ops → label is not trusted.
    # Coordinate resolution via explicit transform still succeeds.
    assert prov["direct_match_succeeded"] is False
    assert "skipped" in prov["direct_match_reason"]
    cert = prov["standard_setting_certificate"]
    assert cert["transform_provenance"] == "unit-test parent-to-standard transform"
    assert cert["origin_shift_status"] == "explicit"
    assert cert["origin_shift_fractional"] == [0.25, 0.0, 0.0]
    np.testing.assert_allclose(
        cert["parent_to_standard_direct_transform"],
        T,
    )


def test_singular_transform_is_rejected():
    """Singular (zero-determinant) transform is rejected before k-point use."""
    T_bad = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float)
    # Use a non-matching k_frac so direct match doesn't catch it.
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_p3(),
        standard_match={"operation_ids": [0, 1]},
        parent_to_standard_direct_transform=T_bad,
    )
    assert label is None
    tf = prov.get("explicit_transform", {})
    assert tf.get("status") == "rejected"
    assert "singular" in tf.get("rejection_reason", "").lower()


def test_nonfinite_transform_is_rejected():
    """Transform containing NaN or inf is rejected."""
    T_bad = np.array([[1, 0, 0], [0, np.nan, 0], [0, 0, 1]], dtype=float)
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_p3(),
        standard_match={"operation_ids": [0, 1]},
        parent_to_standard_direct_transform=T_bad,
    )
    assert label is None
    tf = prov.get("explicit_transform", {})
    assert tf.get("status") == "rejected"


def test_centered_setting_without_explicit_transform_remains_blocked():
    """Without explicit transform, centered settings stay unresolved."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert label is None
    assert blocker is not None
    assert "standard_setting_hsp_mapping_unresolved" in blocker


def test_centered_direct_coordinate_match_is_not_trusted():
    """Centered HSP coordinate coincidences need a validated standard transform."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.5, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert label is None
    assert blocker is not None
    assert prov["direct_match_succeeded"] is False
    assert "not trusted" in prov["direct_match_reason"]
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "unresolved"
    assert cert["primitive_conventional_relation"] == "centered_unresolved"


def test_override_still_blocked_when_kmap_unresolved():
    """Manual HSP override cannot bypass unresolved standard-setting mapping."""
    from valleyscope.workflows.analyze_hsp import _resolve_generic_irrep_hsp_label

    class _T:
        number = 5; name = "C2"; spinor = True
        def match_kpoint_label(self, k, tolerance=1e-6): return None

    label, blocker = _resolve_generic_irrep_hsp_label(
        table=_T(), k_frac=np.array([0.123, 0.456, 0.0]),
        override_label="M",
        standard_match={"number": 5, "hall_number": 9, "hall_symbol": "C 2y"},
    )
    assert label is None
    assert blocker is not None
    assert "cannot be applied" in blocker

def test_basis_transform_matrix_requires_operation_verification(monkeypatch):
    """A transform matrix is not accepted without operation-basis validation."""
    import valleyscope.analysis.standard_setting_kmap as kmap

    class _SecondCallTable:
        def __init__(self):
            self.calls = 0

        def match_kpoint_label(self, k_frac, *, tolerance=1e-6):
            self.calls += 1
            return "M" if self.calls > 1 else None

    monkeypatch.setattr(
        kmap,
        "_compute_standard_setting_basis_transform",
        lambda **kwargs: {
            "status": "accepted",
            "transform_matrix": np.eye(3).tolist(),
        },
    )

    label, blocker, prov = kmap.resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_SecondCallTable(),
        standard_match={
            "number": 143,
            "international_short": "P3",
            "hall_number": 430,
            "hall_symbol": "P 3",
            "operation_ids": [0, 1],
        },
    )

    assert label is None
    assert blocker is not None
    assert prov["basis_transform"]["status"] == "accepted"
    assert "operation_basis_verification" not in prov["basis_transform"]


# ---------------------------------------------------------------------------
# Standard-setting certificate tests
# ---------------------------------------------------------------------------

def test_direct_match_produces_validated_certificate():
    """Direct coordinate match produces a validated certificate."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        detected_operations=_detected_std_ops(430, [0, 1, 2]),
    )
    assert label == "GM"
    cert = prov.get("standard_setting_certificate")
    assert cert is not None
    assert cert["validation_status"] == "validated"
    assert cert["subspace_sg_number"] == 143
    assert cert["subspace_sg_symbol"] == "P3"
    assert cert["hall_number"] == 430
    assert cert["resolved_hsp_label"] == "GM"


def test_unresolved_blocker_produces_certificate():
    """Unresolved setting produces a certificate with blocker reason."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert label is None
    cert = prov.get("standard_setting_certificate")
    assert cert is not None
    assert cert["validation_status"] == "unresolved"
    assert cert["unresolved_reason"] is not None
    assert "standard_setting_hsp_mapping_unresolved" in cert["unresolved_reason"]
    assert cert["subspace_sg_number"] == 5
    assert cert["hall_number"] == 9


def test_no_standard_match_certificate_still_produced():
    """No standard_match still produces a certificate with unresolved status."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match=None,
    )
    cert = prov.get("standard_setting_certificate")
    assert cert is not None
    assert cert["validation_status"] == "unresolved"


def test_centered_setting_certificate_has_blocker():
    """Centered setting without explicit transform has certificate blocker."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    cert = prov.get("standard_setting_certificate")
    assert cert["validation_status"] == "unresolved"
    assert "standard_setting" in cert.get("unresolved_reason", "")
    assert cert["centering_type"] == "C"


def test_certificate_parent_basis_operation_ids():
    """Certificate records parent-basis operation IDs."""
    _, _, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
    )
    cert = prov["standard_setting_certificate"]
    assert cert["parent_basis_operation_ids"] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Affine certificate tests
# ---------------------------------------------------------------------------

def test_direct_match_certificate_has_affine_status():
    """Direct match certificate carries affine validation fields."""
    _, _, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        detected_operations=[
            {"operation_id": 0, "order": 1,
             "rotation_frac": [[1,0,0],[0,1,0],[0,0,1]],
             "translation_frac": [0.0, 0.0, 0.0]},
        ],
    )
    cert = prov["standard_setting_certificate"]
    # Bijection: only 1 of 3 required P3 ops → validation fails closed.
    assert cert["validation_status"] == "rejected"


def test_unresolved_certificate_has_missing_ingredients():
    """Unresolved certificate lists missing affine ingredients."""
    _, _, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
        detected_operations=[
            {"operation_id": 0, "order": 1,
             "rotation_frac": [[1,0,0],[0,1,0],[0,0,1]],
             "translation_frac": [0.0, 0.0, 0.0]},
            {"operation_id": 4, "order": 2,
             "rotation_frac": [[-1,0,0],[0,-1,0],[0,0,1]],
             "translation_frac": [0.0, 0.0, 0.0]},
        ],
    )
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "unresolved"
    # C-centering cosets are now explicit — centering_vectors is present.
    assert cert["centering_type"] == "C"
    assert cert["centering_status"] == "centered_unresolved"


def test_centered_setting_blocked_with_affine_reason():
    """Centered setting blocker references affine convention gap, not material name."""
    _, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert blocker is not None
    assert "C2" in blocker
    cert = prov["standard_setting_certificate"]
    assert cert["centering_status"] == "centered_unresolved"
    # Blocker must not name any real material.
    for mat in ("MoTe2", "ZrSe2", "tMoTe2", "tZrSe2"):
        assert mat.lower() not in str(blocker).lower()


# ---------------------------------------------------------------------------
# Affine inconsistency rejection tests
# ---------------------------------------------------------------------------

def _fake_table_m():
    """Table that matches (0.5, 0, 0) as HSP 'M'."""
    return _FakeTable(labels={"M": (0.5, 0.0, 0.0)})


def test_explicit_transform_rejected_when_translations_inconsistent():
    """Matching rotations but inconsistent translations → rejected, not validated."""
    T = np.diag([0.5, 1.0, 1.0])
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.25, 0.0, 0.0]),
        table=_fake_table_m(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        parent_to_standard_direct_transform=T,
        detected_operations=_p3_ops_from_spglib(override={
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.25, 0.0, 0.0],
        }),
    )
    # Must be rejected — identity operation with non-zero translation
    # is not affine-equivalent to the standard identity operation.
    assert label is None
    assert blocker is not None
    tf = prov.get("explicit_transform", {})
    assert tf.get("status") == "rejected"
    # Under bijection, insufficient ops → transform is unresolved.
    cert = prov.get("standard_setting_certificate", {})
    assert cert.get("validation_status") == "unresolved"
    tc = prov["transform_candidate"]


def test_direct_match_rejected_when_translations_inconsistent():
    """Direct coordinate match cannot bypass affine translation validation."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_fake_table_m(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        detected_operations=_p3_ops_from_spglib(override={
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.25, 0.0, 0.0],
        }),
    )
    assert label is None
    assert blocker is not None
    assert "direct coordinate match rejected" in blocker
    cert = prov.get("standard_setting_certificate", {})
    assert cert.get("validation_status") == "rejected"
    assert cert.get("translation_validation_status") == "failed"


def test_basis_reconstruction_rejected_when_translations_inconsistent(monkeypatch):
    """Operation-basis reconstruction cannot bypass affine translations."""
    import valleyscope.analysis.standard_setting_kmap as kmap

    class _SecondCallTable:
        def __init__(self):
            self.calls = 0

        def match_kpoint_label(self, k_frac, *, tolerance=1e-6):
            self.calls += 1
            return "M" if self.calls > 1 else None

    monkeypatch.setattr(
        kmap,
        "_compute_standard_setting_basis_transform",
        lambda **kwargs: {
            "status": "accepted",
            "transform_matrix": np.eye(3).tolist(),
            "operation_basis_verification": {"status": "passed"},
        },
    )

    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_SecondCallTable(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        detected_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.25, 0.0, 0.0],
        }],
    )

    assert label is None
    assert blocker is not None
    assert "operation-basis reconstruction rejected" in blocker
    cert = prov.get("standard_setting_certificate", {})
    assert cert.get("validation_status") == "rejected"
    assert cert.get("translation_validation_status") == "failed"
    assert (
        cert.get("primitive_conventional_relation")
        == "operation_basis_reconstruction"
    )
    tc = prov["transform_candidate"]
    assert tc["validation_status"] == "rejected"
    assert tc["transform_provenance"] == "operation_basis_reconstruction"
    assert tc["affine_validation_status"] == "failed"


def test_basis_reconstruction_certificate_records_relation(monkeypatch):
    """Accepted operation-basis reconstruction records primitive relation provenance."""
    import valleyscope.analysis.standard_setting_kmap as kmap

    class _SecondCallTable:
        def __init__(self):
            self.calls = 0

        def match_kpoint_label(self, k_frac, *, tolerance=1e-6):
            self.calls += 1
            return "M" if self.calls > 1 else None

    monkeypatch.setattr(
        kmap,
        "_compute_standard_setting_basis_transform",
        lambda **kwargs: {
            "status": "accepted",
            "transform_matrix": np.eye(3).tolist(),
            "operation_basis_verification": {"status": "passed"},
        },
    )

    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_SecondCallTable(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        detected_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.0, 0.0, 0.0],
        }],
    )

    # Bijection: only 1 of 3 ops → operation basis reconstruction is
    # unresolved (full affine evidence is required).
    assert label is None
    cert = prov["standard_setting_certificate"]
    assert (
        cert["primitive_conventional_relation"]
        == "operation_basis_reconstruction"
    )
    tc = prov["transform_candidate"]
    assert tc["transform_provenance"] == "operation_basis_reconstruction"


# ---------------------------------------------------------------------------
# Hall-number convention guard tests
# ---------------------------------------------------------------------------

def test_hall_number_mismatch_with_sg_number_is_consistency_checked():
    """Hall for SG 143 is Hall 430, not 143. Mismatch must be flagged."""
    from valleyscope.analysis.standard_setting_kmap import (
        _validate_hall_sg_consistency,
    )
    ok, blocker = _validate_hall_sg_consistency(
        hall_number=143, sg_number=143,
    )
    assert not ok
    assert blocker is not None
    assert "mismatch" in blocker


def test_hall_number_mismatch_blocks_trusted_hsp_label():
    """A Hall/SG mismatch cannot produce a trusted standard-setting HSP label."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_fake_table_m(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 143, "hall_symbol": "P 3",
            "operation_ids": [0],
        },
        detected_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.0, 0.0, 0.0],
        }],
    )
    assert label is None
    assert blocker is not None
    assert "standard_setting_hall_number_mismatch" in blocker
    cert = prov.get("standard_setting_certificate", {})
    assert cert.get("validation_status") == "rejected"
    assert cert.get("translation_validation_status") == "failed"


def test_valid_hall_sg_pair_passes_consistency_check():
    """Hall 430 <-> SG 143 (P3) is a valid pair."""
    from valleyscope.analysis.standard_setting_kmap import (
        _validate_hall_sg_consistency,
    )
    ok, blocker = _validate_hall_sg_consistency(
        hall_number=430, sg_number=143,
    )
    assert ok
    assert blocker is None


def test_hall_9_sg_5_passes_consistency():
    """Hall 9 <-> SG 5 (C2) is a valid pair."""
    from valleyscope.analysis.standard_setting_kmap import (
        _validate_hall_sg_consistency,
    )
    ok, blocker = _validate_hall_sg_consistency(
        hall_number=9, sg_number=5,
    )
    assert ok
    assert blocker is None


# ---------------------------------------------------------------------------
# Primitive-conventional relation tests
# ---------------------------------------------------------------------------

def test_direct_match_certificate_has_primitive_conventional_relation():
    """Direct coordinate match records primitive_conventional_relation."""
    _, _, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
    )
    cert = prov["standard_setting_certificate"]
    assert cert["primitive_conventional_relation"] == "direct_coordinate_match"
    assert cert["centering_status"] == "primitive_direct_match"
    assert cert["centering_type"] == "P"


def test_negative_hall_prefix_primitive_direct_match_is_trusted():
    """Hall symbols like -P 1 are primitive despite the leading inversion sign."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 2, "international_short": "P-1",
            "hall_number": 2, "hall_symbol": "-P 1",
            "operation_ids": [0, 1],
        },
        detected_operations=_detected_std_ops(2, [0, 1]),
    )
    assert label == "GM"
    assert blocker is None
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "validated"
    assert cert["centering_type"] == "P"
    assert cert["centering_status"] == "primitive_direct_match"


def test_centered_setting_certificate_relation_is_centered_unresolved():
    """Centered setting records primitive_conventional_relation=centered_unresolved."""
    _, _, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    cert = prov["standard_setting_certificate"]
    assert cert["primitive_conventional_relation"] == "centered_unresolved"
    assert cert["centering_status"] == "centered_unresolved"
    assert cert["centering_type"] == "C"


# ---------------------------------------------------------------------------
# Transform candidate tests
# ---------------------------------------------------------------------------

def test_transform_candidate_in_provenance_for_unresolved_centered():
    """Unresolved centered setting produces transform_candidate in provenance."""
    _, _, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    tc = prov.get("transform_candidate")
    assert tc is not None
    assert tc["validation_status"] == "unresolved"
    assert tc["centering_type"] == "C"
    assert tc["centering_status"] == "centered_unresolved"
    assert tc.get("centering_vectors") is not None


def test_centered_unresolved_transform_candidate_has_centering_vectors():
    """Unresolved centered: centering vectors from Hall symbol are present."""
    _, _, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.123, 0.456, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    tc = prov.get("transform_candidate")
    assert tc is not None
    assert tc["validation_status"] == "unresolved"
    assert tc["centering_type"] == "C"
    assert tc["centering_status"] == "centered_unresolved"
    cv = tc.get("centering_vectors")
    assert cv is not None
    assert len(cv) == 2  # identity + C-center (1/2, 1/2, 0)


def test_centering_affine_validation_with_explicit_cosets_passes_toy():
    """C-centered toy: transform + centering cosets → affine validation can pass."""
    from valleyscope.analysis.standard_setting_kmap import (
        _validate_affine_operation_equivalence,
    )
    T_id = np.eye(3)
    # C-centered identity operation: the parent translation differs from the
    # standard identity by the C-centering coset.
    result = _validate_affine_operation_equivalence(
        vp_operations=[{
            "operation_id": 0, "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.5, 0.5, 0.0],
        }],
        vp_operation_ids=[0],
        standard_match={
            "number": 5, "hall_number": 9, "hall_symbol": "C 2y",
        },
        parent_to_standard_direct_transform=T_id,
    )
    # Centering cosets allow comparison modulo C-centered lattice.
    assert result["status"] == "failed"
    assert result["centering_cosets_count"] == 2
    assert result["matched_affine_operations"] == 1


def test_centering_cosets_c_type():
    """C-centering returns identity + (1/2, 1/2, 0)."""
    from valleyscope.analysis.standard_setting_kmap import _centering_cosets
    cosets = _centering_cosets("C 2y")
    assert len(cosets) == 2
    assert np.allclose(cosets[1], [0.5, 0.5, 0.0])


def test_centering_cosets_i_type():
    """I-centering returns identity + (1/2, 1/2, 1/2)."""
    from valleyscope.analysis.standard_setting_kmap import _centering_cosets
    cosets = _centering_cosets("I 4")
    assert len(cosets) == 2
    assert np.allclose(cosets[1], [0.5, 0.5, 0.5])


def test_centering_cosets_f_type():
    """F-centering returns 4 cosets."""
    from valleyscope.analysis.standard_setting_kmap import _centering_cosets
    cosets = _centering_cosets("F d 3")
    assert len(cosets) == 4


def test_centering_cosets_p_type():
    """P-centering returns identity only."""
    from valleyscope.analysis.standard_setting_kmap import _centering_cosets
    cosets = _centering_cosets("P 3")
    assert len(cosets) == 1


# ---------------------------------------------------------------------------
# Centered explicit-transform E2E contract tests
# ---------------------------------------------------------------------------

def test_centered_without_explicit_transform_remains_blocked():
    """C-centered without explicit transform: blocked with provenance."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert label is None
    assert blocker is not None
    assert "standard_setting_hsp_mapping_unresolved" in blocker
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "unresolved"
    tc = prov.get("transform_candidate")
    assert tc["centering_type"] == "C"
    assert tc["centering_status"] == "centered_unresolved"


def test_centered_with_explicit_transform_and_valid_affine_becomes_validated():
    """C-centered with T=identity + valid affine → validated certificate."""
    T = np.eye(3)
    k = np.array([0.0, 0.0, 0.0])
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=k,
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0],
        },
        parent_to_standard_direct_transform=T,
        detected_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.0, 0.0, 0.0],
        }],
    )
    # C-centered: Phase E (1 of 4 ops).  Promoter blocks; transform is unresolved.
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "rejected"
    assert cert["primitive_conventional_relation"] == "explicit_transform"
    assert cert["centering_type"] == "C"
    assert cert["centering_vectors"] == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]


def test_centered_with_explicit_transform_fails_with_bad_affine_translation():
    """C-centered with bad affine translation → rejected, not trusted."""
    T = np.eye(3)
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0],
        },
        parent_to_standard_direct_transform=T,
        detected_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.3, 0.0, 0.0],
        }],
    )
    # Translation [0.3, 0, 0] does not match identity or C-center (0.5, 0.5, 0).
    assert label is None
    assert blocker is not None
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "rejected"
    assert cert["translation_validation_status"] == "failed"
    assert cert["matched_affine_operations"] == 0
    assert cert["mismatched_translation_count"] == 4
    tc = prov.get("transform_candidate", {})
    assert tc["validation_status"] == "rejected"
    assert tc["affine_validation_status"] == "failed"


def test_centered_transform_certificate_has_centering_vectors():
    """Explicit transform for C-centered: certificate records centering vectors."""
    T = np.eye(3)
    _, _, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=_table_c2(),
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0],
        },
        parent_to_standard_direct_transform=T,
        detected_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.0, 0.0, 0.0],
        }],
    )
    tc = prov.get("transform_candidate", {})
    cv = tc.get("centering_vectors")
    assert cv is not None
    assert len(cv) == 2  # identity + C-center
    cert = prov.get("standard_setting_certificate", {})
    assert cert["centering_vectors"] == cv
    assert cert["centering_type"] == "C"
    assert cert["validation_status"] == "rejected"


def test_centering_cosets_negative_p_type():
    """Leading Hall inversion sign does not change primitive centering."""
    from valleyscope.analysis.standard_setting_kmap import _centering_cosets
    cosets = _centering_cosets("-P 1")
    assert len(cosets) == 1


def test_centering_cosets_rhombohedral_requires_explicit_convention():
    """R centering stays unresolved until obverse/reverse convention is explicit."""
    from valleyscope.analysis.standard_setting_kmap import _centering_cosets
    assert _centering_cosets("R 3") is None
    assert _centering_cosets("-R 3") is None


# ---------------------------------------------------------------------------
# Affine transform derivation tests
# ---------------------------------------------------------------------------

def test_derive_transform_c2_identity_pair_is_ambiguous():
    """C2 parent with identity transform remains basis-orientation ambiguous."""
    from valleyscope.analysis.standard_setting_kmap import (
        _derive_transform_candidate,
    )
    result = _derive_transform_candidate(
        vp_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.0, 0.0, 0.0],
        }, {
            "operation_id": 4,
            "rotation_frac": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            "translation_frac": [0.0, 0.0, 0.0],
        }],
        vp_operation_ids=[0, 4],
        standard_match={
            "number": 5, "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert result["status"] == "ambiguous"
    assert result["candidate_count"] > 1


def test_derive_transform_no_vp_ops_returns_unresolved():
    """No VP ops → derivation unresolved."""
    from valleyscope.analysis.standard_setting_kmap import (
        _derive_transform_candidate,
    )
    result = _derive_transform_candidate(
        vp_operations=[],
        vp_operation_ids=[],
        standard_match={"hall_number": 9},
    )
    assert result["status"] == "unresolved"


def test_derive_transform_no_hall_number_returns_unresolved():
    """No Hall number → unresolved."""
    from valleyscope.analysis.standard_setting_kmap import (
        _derive_transform_candidate,
    )
    result = _derive_transform_candidate(
        vp_operations=[{"operation_id": 0,
                         "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         "translation_frac": [0.0, 0.0, 0.0]}],
        vp_operation_ids=[0],
        standard_match={"operation_ids": [0]},
    )
    assert result["status"] == "unresolved"
    assert "hall_number" in result.get("missing_ingredients", [])


def test_derive_transform_r_centering_without_cosets_returns_unresolved():
    """R-centered setting needs an explicit obverse/reverse centering convention."""
    from valleyscope.analysis.standard_setting_kmap import (
        _derive_transform_candidate,
    )
    result = _derive_transform_candidate(
        vp_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.0, 0.0, 0.0],
        }],
        vp_operation_ids=[0],
        standard_match={
            "number": 146,
            "hall_number": 433,
            "hall_symbol": "R 3",
            "operation_ids": [0],
        },
    )
    assert result["status"] == "unresolved"
    assert "conventional_centering_vectors" in result.get("missing_ingredients", [])


# ---------------------------------------------------------------------------
# Derived transform downstream provenance tests
# ---------------------------------------------------------------------------

def test_plumbing_derived_transform_provenance_reaches_ebr_candidate():
    """Plumbing test: manually-injected derived transform provenance flows to EBR.

    This is a plumbing-only test — it injects a certificate payload with
    transform_provenance='affine_operation_derivation' and proves the
    plumbing preserves it into EBR candidate irrep_source_provenance.
    It does NOT claim that _derive_transform_candidate() produced this
    certificate automatically.
    """
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates

    workflow = {"by_kpoint": {"GM": {"K_valley": {
        "readiness_level": "trusted", "workflow_path": "direct_qcut",
    }}}}
    sa_report = {"by_kpoint": {"GM": {"valley_preserving_subspaces": [{
        "orbit": ["K_valley"],
        "hsp_preserving_operation_ids": [0, 1],
        "subspace_space_group": {
            "status": "resolved",
            "candidate_space_group_symbol": "P4",
            "candidate_space_group_number": 75,
            "valley_preserving_operation_ids": [0, 1],
        },
        "valley_preserving_character_diagnostics": {
            "per_valley": {"K_valley": [
                {"operation_id": 0, "eigenphases": [0.0]},
                {"operation_id": 1, "eigenphases": [0.5]},
            ]},
        },
    }]}}}
    # Standard-setting provenance with derived transform certificate.
    kmap_prov: dict = {
        "standard_setting_certificate": {
            "validation_status": "validated",
            "subspace_sg_number": 75,
            "subspace_sg_symbol": "P4",
            "centering_type": "P",
            "primitive_conventional_relation": "operation_basis_reconstruction",
            "transform_provenance": "affine_operation_derivation",
        },
    }
    matching = build_valley_irrep_matching_report(
        irrep_workflow_decisions=workflow,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters_flattened={"GM": {"K_valley": {
            "A": {1: 1.0 + 0j, 2: -1.0 + 0j},
        }}},
        source_operation_maps={"GM": {"K_valley": {0: 1, 1: 2}}},
        source_payload_provenance={"GM": {"K_valley": {
            "standard_setting_hsp_mapping": kmap_prov,
        }}},
    )
    candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    assert candidates["candidate_count"] == 1
    c = candidates["candidates"][0]
    prov = c.get("irrep_source_provenance", {})
    kmap = prov.get("standard_setting_hsp_mapping", {})
    cert = kmap.get("standard_setting_certificate", {})
    assert cert["validation_status"] == "validated"
    assert cert["transform_provenance"] == "affine_operation_derivation"


def test_ambiguous_derivation_preserves_provenance():
    """Ambiguous derivation always records derivation_attempt in result."""
    from valleyscope.analysis.standard_setting_kmap import (
        _compute_standard_setting_basis_transform,
    )
    # C2 with only 2 ops → derivation is ambiguous (multiple valid T).
    result = _compute_standard_setting_basis_transform(
        lattice_direct_cart=np.eye(3),
        vp_operations=[
            {"operation_id": 0, "order": 1,
             "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {"operation_id": 4, "order": 2,
             "rotation_frac": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]},
        ],
        standard_match={
            "number": 5, "international_short": "C2",
            "hall_number": 9, "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    da = result.get("derivation_attempt")
    assert da is not None
    assert da["status"] == "ambiguous"
    assert da.get("candidate_count", 0) > 1


def test_primitive_direct_match_resolves_label_in_resolver():
    """P3 resolver-level: direct match returns label, no spurious blocker."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_table_p3(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0, 1, 2],
        },
        detected_operations=_detected_std_ops(430, [0, 1, 2]),
    )
    assert label == "M"
    assert blocker is None
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "validated"
    assert "standard_setting_hsp_mapping_unresolved" not in str(prov)
    # Direct coordinate match bypasses derivation entirely — no basis_transform.


def test_malformed_nonrequired_operation_is_ignored_before_affine_validation():
    """Parent operations outside G_k^(a) do not enter affine field checks."""
    detected = _detected_std_ops(1, [0])
    detected.append({
        "operation_id": 99,
        "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        # Deliberately no translation_frac: irrelevant outside required {0}.
    })
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.zeros(3),
        table=_FakeTable(labels={"GM": (0.0, 0.0, 0.0)}),
        standard_match={
            "number": 1,
            "international_short": "P1",
            "hall_number": 1,
            "hall_symbol": "P 1",
            "operation_ids": [0],
        },
        detected_operations=detected,
    )
    assert label == "GM"
    assert blocker is None
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "validated"
    assert cert["missing_affine_ingredients"] == []
    assert cert["unmatched_parent_operations"] == []
    assert cert["unused_standard_operation_indices"] == []


def test_malformed_required_operation_blocks_affine_validation():
    from valleyscope.analysis.standard_setting_kmap import (
        _validate_affine_operation_equivalence,
    )
    result = _validate_affine_operation_equivalence(
        vp_operations=[{
            "operation_id": 0,
            "rotation_frac": np.eye(3).tolist(),
        }],
        vp_operation_ids=[0],
        standard_match={
            "number": 1, "hall_number": 1, "hall_symbol": "P 1",
        },
        parent_to_standard_direct_transform=np.eye(3),
    )
    assert result["status"] == "failed"
    assert result["missing_required_operation_ids"] == [0]
    assert "malformed_detected_operations" in result["missing_ingredients"]


def test_missing_required_operation_id_blocks_affine_validation():
    from valleyscope.analysis.standard_setting_kmap import (
        _validate_affine_operation_equivalence,
    )
    result = _validate_affine_operation_equivalence(
        vp_operations=_detected_std_ops(430, [0, 1]),
        vp_operation_ids=[0, 1, 2],
        standard_match={
            "number": 143, "hall_number": 430, "hall_symbol": "P 3",
        },
        parent_to_standard_direct_transform=np.eye(3),
    )
    assert result["status"] == "failed"
    assert result["missing_required_operation_ids"] == [2]


def test_duplicate_required_operation_id_blocks_affine_validation():
    from valleyscope.analysis.standard_setting_kmap import (
        _validate_affine_operation_equivalence,
    )
    result = _validate_affine_operation_equivalence(
        vp_operations=_detected_std_ops(1, [0]),
        vp_operation_ids=[0, 0],
        standard_match={
            "number": 1, "hall_number": 1, "hall_symbol": "P 1",
        },
        parent_to_standard_direct_transform=np.eye(3),
    )
    assert result["status"] == "failed"
    assert result["required_operation_ids"] is None
    assert result["missing_ingredients"] == ["duplicate_required_operation_id"]


def test_duplicate_affine_content_under_distinct_ids_blocks_bijection():
    from valleyscope.analysis.standard_setting_kmap import (
        _validate_affine_operation_equivalence,
    )
    identity = {
        "rotation_frac": np.eye(3).tolist(),
        "translation_frac": [0.0, 0.0, 0.0],
    }
    result = _validate_affine_operation_equivalence(
        vp_operations=[
            {"operation_id": 0, **identity},
            {"operation_id": 4, **identity},
        ],
        vp_operation_ids=[0, 4],
        standard_match={
            "number": 3, "hall_number": 3, "hall_symbol": "P 2y",
        },
        parent_to_standard_direct_transform=np.eye(3),
    )
    assert result["status"] == "failed"
    assert len(result["unmatched_parent_operations"]) == 1
    assert result["operation_closure_validated"] is False


def test_early_unmatched_operation_preserves_later_map_provenance():
    from valleyscope.analysis.standard_setting_kmap import (
        _validate_affine_operation_equivalence,
    )
    ops = _detected_std_ops(430, [0, 1, 2])
    ops[0]["translation_frac"] = [0.25, 0.0, 0.0]
    result = _validate_affine_operation_equivalence(
        vp_operations=ops,
        vp_operation_ids=[0, 1, 2],
        standard_match={
            "number": 143, "hall_number": 430, "hall_symbol": "P 3",
        },
        parent_to_standard_direct_transform=np.eye(3),
    )
    assert result["status"] == "failed"
    assert result["operation_map"] == {"1": 1, "2": 2}
    assert result["unmatched_parent_operations"] == [{
        "operation_id": 0,
        "parent_translation_frac": [0.25, 0.0, 0.0],
    }]


@pytest.mark.parametrize("required_ids", [None, (0,), [True], [0.0], ["0"]])
def test_malformed_required_operation_id_collection_is_unknown(required_ids):
    from valleyscope.analysis.standard_setting_kmap import (
        _validate_affine_operation_equivalence,
    )
    result = _validate_affine_operation_equivalence(
        vp_operations=_detected_std_ops(1, [0]),
        vp_operation_ids=required_ids,
        standard_match={
            "number": 1, "hall_number": 1, "hall_symbol": "P 1",
        },
        parent_to_standard_direct_transform=np.eye(3),
    )
    assert result["status"] == "failed"
    assert result["required_operation_ids"] is None
    assert result["missing_ingredients"] == ["malformed_required_operation_ids"]
