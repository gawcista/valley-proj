"""Focused diagnostics.h5 tests: canonical valley_masks group with a
sector_masks hard-link compatibility alias (no data copy)."""

import h5py
import numpy as np

from valleyscope.projection.sector_projectors import SectorProjectors
from valleyscope.reports.h5_report import write_diagnostics_h5


def _projectors_by_kpoint():
    return {
        "GammaM": SectorProjectors(
            sector_masks={
                "K_sector": np.array([True, True, False, False]),
                "Kp_sector": np.array([False, False, True, True]),
            },
            center_masks={
                "K": np.array([True, True, False, False]),
                "Kp": np.array([False, False, True, True]),
            },
            overlap_mask=np.zeros(4, dtype=bool),
            qcut=0.1,
            warnings=[],
        )
    }


def test_valley_masks_canonical_with_sector_masks_hard_link_alias(tmp_path):
    out = write_diagnostics_h5(tmp_path / "diagnostics.h5", _projectors_by_kpoint())

    with h5py.File(out, "r") as h5:
        group = h5["projectors"]["GammaM"]
        # Both the canonical valley-named path and the legacy alias exist.
        assert "valley_masks" in group
        assert "sector_masks" in group

        valley = group["valley_masks"]
        sector = group["sector_masks"]

        # Same valley labels with identical masks.
        assert set(valley) == {"K_sector", "Kp_sector"}
        assert set(sector) == {"K_sector", "Kp_sector"}
        for name in valley:
            np.testing.assert_array_equal(valley[name][...], sector[name][...])

        # Hard link: both names refer to the same HDF5 object, not a copy.
        assert valley.id == sector.id
