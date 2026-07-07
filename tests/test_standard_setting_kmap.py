"""Tests for standard-setting HSP k-coordinate mapping."""

import numpy as np

from valleyscope.analysis.standard_setting_kmap import (
    resolve_standard_setting_hsp_label,
    _attempt_setting_transform,
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
            "hall_number": 1,
            "hall_symbol": "C 2y",
            "operation_ids": [0, 4],
        },
    )
    assert label is None
    assert blocker is not None
    assert "standard_setting_hsp_mapping_unresolved" in blocker
    assert "C2" in blocker
    assert "No. 5" in blocker
    assert prov["hall_number"] == 1
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
            "hall_number": 143,
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
        hall_number=1,
        hall_symbol="C 2y",
        sg_number=5,
    )
    assert result["sg_number"] == 5
    assert result["reason"] is not None
    assert "centered" in str(result["reason"]).lower()


def test_centered_setting_documented():
    result = _attempt_setting_transform(
        k_frac=np.array([0.5, 0.0, 0.0]),
        hall_number=1,
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
            "hall_number": 143,
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


def test_basis_transform_centered_setting_triggers_reconstruction():
    """Subgroup cell reconstruction is attempted for centered settings."""
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
    # Reconstruction was attempted; transform accepted only with verification.
    assert result["status"] in ("accepted", "rejected", "unavailable")
    assert "operation_basis_verification" in result


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
    """Subgroup reconstruction may be rejected when VP ops don't match standard ops
    after transformation (e.g., hexagonal lattice with C2 subgroup)."""
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
    # With hexagonal lattice, the operation-basis check should run.
    assert "operation_basis_verification" in result


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
            "hall_number": 143,
            "hall_symbol": "P 3",
            "operation_ids": [0, 1],
        },
    )

    assert label is None
    assert blocker is not None
    assert prov["basis_transform"]["status"] == "accepted"
    assert "operation_basis_verification" not in prov["basis_transform"]
