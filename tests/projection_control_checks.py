"""Reusable acceptance checks, not production valley-readiness policy."""

import math


def check_capture_controls(
    rows, *, reference_family, excluded_families, w_val_min,
    weight_conservation_atol=1e-12,
):
    """Check fixed-radius controls in both sampled bands and their full span.

    The cutoff is supplied by the existing material configuration. Nonzero
    capture in an excluded family is allowed below that cutoff; momentum
    windows are not layer-resolved Bloch unfolding.
    """
    families = (reference_family, *excluded_families)
    assert len(set(families)) == len(families) and len(families) > 1
    assert math.isfinite(w_val_min) and 0 < w_val_min < 1
    assert math.isfinite(weight_conservation_atol) and weight_conservation_atol > 0
    assert rows, "missing projection controls"
    by_point = {}
    for row in rows:
        point = row["kpoint"]
        family = row["family"]
        assert family in families
        point_rows = by_point.setdefault(point, {})
        assert family not in point_rows, "duplicate control"
        point_rows[family] = row
    band_count = 0
    for point_rows in by_point.values():
        assert set(point_rows) == set(families), "missing control family"
        reference_bands = None
        for family, row in point_rows.items():
            weights = row["weights"]
            assert weights, "missing control bands"
            bands = [weight["band"] for weight in weights]
            assert len(set(bands)) == len(bands), "duplicate band"
            if reference_bands is None:
                reference_bands = bands
            assert bands == reference_bands, "different control bands"
            captures = []
            for weight in weights:
                norm, captured, residual, overlap = (
                    weight[key] for key in ("norm", "W_val", "W_res", "W_overlap")
                )
                assert all(math.isfinite(value) for value in (norm, captured, residual, overlap))
                assert norm > 0 and min(captured, residual, overlap) >= 0
                assert abs(norm - captured - residual - overlap) <= weight_conservation_atol
                captures.append(captured / norm)
            spectrum = row["subspace_capture_eigenvalues"]
            assert len(spectrum) == len(bands), "capture spectrum rank mismatch"
            assert all(math.isfinite(value) and -weight_conservation_atol <= value
                       <= 1 + weight_conservation_atol for value in spectrum)
            if family == reference_family:
                assert min(captures + list(spectrum)) >= w_val_min, "reference capture lost"
            else:
                assert max(captures + list(spectrum)) < w_val_min, "false valley capture"
        band_count += len(reference_bands)
    return {"point_count": len(by_point), "band_count": band_count}
