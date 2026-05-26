import numpy as np

from valleyscope.analysis.hsp_star import build_hsp_star_report


def _p312_operations():
    return [
        {
            "operation_id": 0,
            "kind": "E",
            "order": 1,
            "rotation_frac": np.eye(3, dtype=int),
        },
        {
            "operation_id": 1,
            "kind": "C3",
            "order": 3,
            "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
        },
        {
            "operation_id": 2,
            "kind": "C3^2",
            "order": 3,
            "rotation_frac": np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        },
        {
            "operation_id": 5,
            "kind": "C2",
            "order": 2,
            "rotation_frac": np.array([[1, 0, 0], [1, -1, 0], [0, 0, -1]]),
        },
    ]


def test_hsp_star_report_marks_mm_representatives_as_symmetry_derivable():
    report = build_hsp_star_report(
        kpoint_frac_by_name={"MM": [0.5, 0.0, 0.0]},
        operations=_p312_operations(),
    )

    mm = report["by_kpoint"]["MM"]
    assert mm["complete"] is False
    assert mm["requires_additional_dft"] is False
    assert mm["requires_symmetry_derivation"] is True
    assert mm["star_size"] == 3
    assert mm["explicit_count"] == 1
    assert mm["symmetry_derivable_count"] == 2
    assert "missing_representatives" not in mm
    assert "available_representatives" not in mm
    assert len(mm["symmetry_derivable_representatives"]) == 2
    missing = {
        tuple(round(value, 6) for value in item["canonical_frac"])
        for item in mm["symmetry_derivable_representatives"]
    }
    assert missing == {
        (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.0),
    }


def test_hsp_star_report_complete_when_all_mm_representatives_are_present():
    report = build_hsp_star_report(
        kpoint_frac_by_name={
            "MM1": [0.5, 0.0, 0.0],
            "MM2": [0.0, 0.5, 0.0],
            "MM3": [0.5, 0.5, 0.0],
        },
        operations=_p312_operations(),
    )

    for label in ("MM1", "MM2", "MM3"):
        data = report["by_kpoint"][label]
        assert data["complete"] is True
        assert data["star_size"] == 3
        assert data["explicit_count"] == 3
        assert data["symmetry_derivable_representatives"] == []
        assert data["requires_additional_dft"] is False


def test_hsp_star_report_keeps_gamma_complete():
    report = build_hsp_star_report(
        kpoint_frac_by_name={"GammaM": [0.0, 0.0, 0.0]},
        operations=_p312_operations(),
    )

    gamma = report["by_kpoint"]["GammaM"]
    assert gamma["complete"] is True
    assert gamma["star_size"] == 1
    assert gamma["symmetry_derivable_representatives"] == []


def test_hsp_star_report_merges_rounded_one_third_coordinates():
    report = build_hsp_star_report(
        kpoint_frac_by_name={"KM": [0.333333, 0.333333, 0.0]},
        operations=_p312_operations(),
    )

    km = report["by_kpoint"]["KM"]
    assert km["star_size"] == 2
    assert km["explicit_count"] == 1
    assert km["symmetry_derivable_count"] == 1
    missing = [
        item["canonical_frac"]
        for item in km["symmetry_derivable_representatives"]
    ]
    assert len(missing) == 1
    assert np.allclose(missing[0], [2.0 / 3.0, 2.0 / 3.0, 0.0], atol=1e-5)
