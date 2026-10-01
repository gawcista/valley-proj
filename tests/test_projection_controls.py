"""Negative-control checks must reject lost capture and false valley capture."""

import copy

import numpy as np
import pytest


def _rows():
    return [
        {"kpoint": "point", "family": family,
         "subspace_capture_eigenvalues": [weight],
         "weights": [{"band": 1, "norm": 1.0, "W_val": weight,
                      "W_res": 1.0 - weight, "W_overlap": 0.0}]}
        for family, weight in (("reference", 0.98), ("control_a", 0.00001),
                               ("control_b", 0.14))
    ]


def _check(rows):
    from tests.projection_control_checks import check_capture_controls

    return check_capture_controls(
        rows, reference_family="reference",
        excluded_families=("control_a", "control_b"), w_val_min=0.8,
    )


def test_controls_allow_nonzero_window_alias_below_capture_threshold():
    assert _check(_rows()) == {"point_count": 1, "band_count": 1}


@pytest.mark.parametrize("change", ["reference_lost", "false_capture", "basis_hidden",
                                   "missing_family", "missing_band", "nan", "norm"])
def test_controls_reject_incomplete_or_physically_failed_results(change):
    rows = copy.deepcopy(_rows())
    if change == "reference_lost":
        rows[0]["subspace_capture_eigenvalues"] = [0.2]
    elif change == "false_capture":
        rows[2]["weights"][0].update(W_val=0.9, W_res=0.1)
    elif change == "basis_hidden":
        rows[2]["subspace_capture_eigenvalues"] = [0.9]
    elif change == "missing_family":
        rows.pop()
    elif change == "missing_band":
        rows[1]["weights"] = []
    elif change == "nan":
        rows[1]["weights"][0]["W_val"] = float("nan")
    elif change == "norm":
        rows[1]["weights"][0]["W_res"] = 0.5
    with pytest.raises(AssertionError):
        _check(rows)


def test_cross_layer_periodic_windows_can_capture_without_same_layer_overlap():
    """Using one fallback lattice for every layer would lose the 0.14 alias."""
    from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
    from valleyscope.projection.sector_projectors import build_sector_projectors
    from valleyscope.projection.weights import compute_valley_weights
    from valleyscope.subspace.valley_basis import build_valley_subspace_matrices

    top, bottom = np.diag([10.0, 10.0, 1.0]), np.diag([6.0, 6.0, 1.0])
    q = np.array([[9.0, 0, 0], [-1.0, 0, 0], [2.0, 0, 0]])
    coefficients = np.sqrt([0.14, 0.84, 0.02]).reshape(1, 1, 3).astype(complex)
    rows, projectors = [], {}
    for family, locations in (("reference", (9, 1)), ("control_a", (0, 0)),
                               ("control_b", (3, 3))):
        centers = [ValleyCenter(name, [location, 0, 0], name, lattice)
                   for name, location, lattice in zip(("top", "bottom"), locations,
                                                       (top, bottom))]
        projection = build_sector_projectors(
            q, centers, [ValleySector(family, ["top", "bottom"])], top, qcut=0.1,
        )
        projectors[family] = projection
        weight = compute_valley_weights(coefficients, projection)[0]
        capture = build_valley_subspace_matrices(coefficients, projection.sector_masks)
        rows.append({"kpoint": "point", "family": family,
                     "subspace_capture_eigenvalues": capture.s_eigenvalues.tolist(),
                     "weights": [{"band": 1, "norm": weight.norm, "W_val": weight.w_val,
                                  "W_res": weight.residual_weight,
                                  "W_overlap": weight.overlap_weight}]})
    assert _check(rows) == {"point_count": 1, "band_count": 1}
    assert [row["weights"][0]["W_val"] for row in rows] == pytest.approx([0.98, 0, 0.14])
    for layer in ("top", "bottom"):
        assert not (projectors["reference"].center_masks[layer]
                    & projectors["control_b"].center_masks[layer]).any()
    assert (projectors["reference"].center_masks["top"]
            & projectors["control_b"].center_masks["bottom"]).tolist() == [True, False, False]
