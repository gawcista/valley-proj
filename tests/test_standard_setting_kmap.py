"""Tests for standard-setting HSP k-coordinate mapping."""

import numpy as np

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
        },
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

def test_explicit_transform_not_needed_when_direct_match_succeeds():
    """Direct HSP coordinate match short-circuits before explicit transform use."""
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
    assert prov["direct_match_succeeded"] is True
    assert "explicit_transform" not in prov


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
            "operation_ids": [0],
        },
        detected_operations=[
            {"operation_id": 0, "order": 1,
             "rotation_frac": [[1,0,0],[0,1,0],[0,0,1]],
             "translation_frac": [0.0, 0.0, 0.0]},
        ],
    )
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "validated"
    assert cert["translation_validation_status"] == "passed"


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
    assert "conventional_centering_vectors" in cert.get(
        "missing_affine_ingredients", []
    ), (
        f"should list conventional_centering_vectors as missing, "
        f"got {cert.get('missing_affine_ingredients')}"
    )
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
            "operation_ids": [0],
        },
        parent_to_standard_direct_transform=T,
        detected_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.25, 0.0, 0.0],
        }],
    )
    # Must be rejected — identity operation with non-zero translation
    # is not affine-equivalent to the standard identity operation.
    assert label is None
    assert blocker is not None
    tf = prov.get("explicit_transform", {})
    assert tf.get("status") == "rejected"
    assert "affine operation" in tf.get("rejection_reason", "").lower()
    # Certificate must show the rejection reason.
    cert = prov.get("standard_setting_certificate", {})
    assert cert.get("validation_status") == "rejected"
    assert cert.get("translation_validation_status") == "failed"
    assert cert.get("mismatched_translation_count") == 1


def test_direct_match_rejected_when_translations_inconsistent():
    """Direct coordinate match cannot bypass affine translation validation."""
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.5, 0.0, 0.0]),
        table=_fake_table_m(),
        standard_match={
            "number": 143, "international_short": "P3",
            "hall_number": 430, "hall_symbol": "P 3",
            "operation_ids": [0],
        },
        detected_operations=[{
            "operation_id": 0,
            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_frac": [0.25, 0.0, 0.0],
        }],
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
            "operation_ids": [0],
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
