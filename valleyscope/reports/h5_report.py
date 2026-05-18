from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from valleyscope.projection.sector_projectors import SectorProjectors


def write_diagnostics_h5(
    path: str | Path,
    projectors_by_kpoint: dict[str, SectorProjectors],
    qcut_scan_payload: dict[str, object] | None = None,
    symmetry_representation_payload: dict[str, object] | None = None,
    symmetry_payload: dict[str, object] | None = None,
) -> Path:
    out = Path(path)
    with h5py.File(out, "w") as h5:
        projectors_group = h5.create_group("projectors")
        for kpoint_name, projectors in projectors_by_kpoint.items():
            group = projectors_group.create_group(kpoint_name)
            group.attrs["qcut"] = projectors.qcut
            group["overlap_mask"] = projectors.overlap_mask.astype(np.uint8)
            sectors = group.create_group("sector_masks")
            for name, mask in projectors.sector_masks.items():
                sectors[name] = mask.astype(np.uint8)
            centers = group.create_group("center_masks")
            for name, mask in projectors.center_masks.items():
                centers[name] = mask.astype(np.uint8)
        if qcut_scan_payload is not None:
            scan = h5.create_group("qcut_scan")
            scan.attrs["has_payload"] = True
            for kpoint_name, payload in qcut_scan_payload.items():
                group = scan.create_group(kpoint_name)
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if isinstance(value, bool):
                            group.attrs[key] = value
                        else:
                            group[key] = np.asarray(value)
        if symmetry_representation_payload is not None:
            representation_group = h5.create_group("symmetry_representations")
            for kpoint_name, operations in symmetry_representation_payload.items():
                kpoint_group = representation_group.create_group(kpoint_name)
                if not isinstance(operations, dict):
                    continue
                for operation_id, payload in operations.items():
                    op_group = kpoint_group.create_group(str(operation_id))
                    if not isinstance(payload, dict):
                        continue
                    for key, value in payload.items():
                        if isinstance(value, str):
                            op_group.attrs[key] = value
                        else:
                            op_group[key] = np.asarray(value)
        if symmetry_payload is not None:
            symmetry_group = h5.create_group("symmetry")
            symmetry_group.attrs["status"] = str(symmetry_payload.get("status", "unknown"))
            symmetry_group.attrs["operation_detection_backend"] = str(
                symmetry_payload.get("operation_detection_backend", "")
            )
            symmetry_group.attrs["structure_file"] = str(symmetry_payload.get("structure_file", ""))
            symmetry_group.attrs["detected_operation_count"] = int(symmetry_payload.get("detected_operation_count", 0))
            symmetry_group.attrs["candidate_rotation_count"] = len(symmetry_payload.get("candidate_rotations", []))
    return out


def write_basis_transform_h5(path: str | Path, transforms: dict[str, object]) -> Path:
    out = Path(path)
    with h5py.File(out, "w") as h5:
        for name, payload in transforms.items():
            if isinstance(payload, dict):
                group = h5.create_group(name)
                for dataset_name, value in payload.items():
                    group[dataset_name] = np.asarray(value)
            else:
                h5[name] = np.asarray(payload)
    return out
