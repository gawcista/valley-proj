from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from valley_proj.projection.sector_projectors import SectorProjectors


def write_diagnostics_h5(
    path: str | Path,
    projectors_by_kpoint: dict[str, SectorProjectors],
    qcut_scan_payload: dict[str, object] | None = None,
) -> Path:
    out = Path(path)
    with h5py.File(out, "w") as h5:
        projectors_group = h5.create_group("projectors")
        for kpoint_name, projectors in projectors_by_kpoint.items():
            group = projectors_group.create_group(kpoint_name)
            group.attrs["qcut"] = projectors.qcut
            group["ambiguous_mask"] = projectors.ambiguous_mask.astype(np.uint8)
            sectors = group.create_group("sector_masks")
            for name, mask in projectors.sector_masks.items():
                sectors[name] = mask.astype(np.uint8)
            centers = group.create_group("center_masks")
            for name, mask in projectors.center_masks.items():
                centers[name] = mask.astype(np.uint8)
        if qcut_scan_payload is not None:
            scan = h5.create_group("qcut_scan")
            scan.attrs["has_payload"] = True
    return out


def write_basis_transform_h5(path: str | Path, transforms: dict[str, np.ndarray]) -> Path:
    out = Path(path)
    with h5py.File(out, "w") as h5:
        for name, matrix in transforms.items():
            h5[name] = np.asarray(matrix)
    return out
