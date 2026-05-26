import numpy as np
import pytest

from valleyscope.analysis.layer_pairing_diagnostic import (
    build_layer_pairing_permutation_diagnostic,
)
from valleyscope.geometry.valley_centers import ValleyCenter


def _mstar_centers():
    reciprocal = np.eye(3, dtype=float)
    coords = {
        "M1": np.array([0.5, 0.0, 0.0], dtype=float),
        "M2": np.array([0.0, 0.5, 0.0], dtype=float),
        "M3": np.array([-0.5, 0.5, 0.0], dtype=float),
    }
    centers = []
    for layer in ("top", "bottom"):
        for label, coord in coords.items():
            centers.append(
                ValleyCenter(
                    name=f"{layer}_{label}",
                    cart=coord,
                    layer=layer,
                    reciprocal_cart=reciprocal,
                )
            )
    return centers


def _toy_coefficients():
    coeffs = np.zeros((3, 1, 6), dtype=np.complex128)
    inv = 1.0 / np.sqrt(2.0)
    coeffs[0, 0, 0] = inv  # top_M1
    coeffs[0, 0, 3] = inv  # bottom_M1
    coeffs[1, 0, 1] = inv  # top_M2
    coeffs[1, 0, 4] = inv  # bottom_M2
    coeffs[2, 0, 2] = inv  # top_M3
    coeffs[2, 0, 5] = inv  # bottom_M3
    return coeffs


def _cyclic_representation():
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )


def test_layer_pairing_diagnostic_identifies_same_name_pairing_as_best():
    centers = _mstar_centers()
    q_cart = np.array([center.cart for center in centers], dtype=float)
    d_c3 = _cyclic_representation()

    report = build_layer_pairing_permutation_diagnostic(
        coefficients_by_kpoint={"GammaM": _toy_coefficients()},
        q_cart_by_kpoint={"GammaM": q_cart},
        raw_representations_by_kpoint={
            "GammaM": {
                1: {
                    "D_raw": d_c3,
                    "kind": "C3",
                    "order": 3,
                    "little_group_passed": True,
                }
            }
        },
        operations=[
            {
                "operation_id": 1,
                "kind": "C3",
                "order": 3,
                "rotation_cart": np.array(
                    [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=float,
                ),
            }
        ],
        centers=centers,
        valley_names=["M1_valley", "M2_valley", "M3_valley"],
        top_centers_by_valley={
            "M1_valley": "top_M1",
            "M2_valley": "top_M2",
            "M3_valley": "top_M3",
        },
        bottom_centers_by_valley={
            "M1_valley": "bottom_M1",
            "M2_valley": "bottom_M2",
            "M3_valley": "bottom_M3",
        },
        monolayer_reciprocal_cart=np.eye(3),
        qcut=0.05,
    )

    assert len(report["pairings"]) == 6
    best = report["pairings"][0]
    assert best["is_current_pairing"] is True
    assert best["bottom_assignment"] == {
        "M1_valley": "bottom_M1",
        "M2_valley": "bottom_M2",
        "M3_valley": "bottom_M3",
    }
    assert best["score"]["min_s_min"] == pytest.approx(1.0)
    assert best["score"]["min_valley_concentration"] == pytest.approx(1.0)
    assert best["score"]["max_seed_projector_symmetry_error"] < 1e-12

    non_current = [item for item in report["pairings"] if not item["is_current_pairing"]]
    assert all(item["score"]["min_s_min"] < 1.0 for item in non_current)


def test_layer_pairing_diagnostic_keeps_projector_quality_per_kpoint():
    centers = _mstar_centers()
    q_cart = np.array([center.cart for center in centers], dtype=float)

    report = build_layer_pairing_permutation_diagnostic(
        coefficients_by_kpoint={"GammaM": _toy_coefficients()},
        q_cart_by_kpoint={"GammaM": q_cart},
        raw_representations_by_kpoint={},
        operations=[],
        centers=centers,
        valley_names=["M1_valley", "M2_valley", "M3_valley"],
        top_centers_by_valley={
            "M1_valley": "top_M1",
            "M2_valley": "top_M2",
            "M3_valley": "top_M3",
        },
        bottom_centers_by_valley={
            "M1_valley": "bottom_M1",
            "M2_valley": "bottom_M2",
            "M3_valley": "bottom_M3",
        },
        monolayer_reciprocal_cart=np.eye(3),
        qcut=0.05,
    )

    current = next(item for item in report["pairings"] if item["is_current_pairing"])
    gamma = current["by_kpoint"]["GammaM"]
    assert gamma["projector_quality"]["expected_rank"] == 1
    assert gamma["projector_quality"]["per_valley"]["M1_valley"]["rank_estimate"] == 1
