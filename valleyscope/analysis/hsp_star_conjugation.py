"""HSP-star conjugation graph: map valley-preserving operations across HSP-star members.

For source k0 with explicit valley-preserving operation g, and a space-group
operation r that maps k0 -> k1 = r k0, the conjugate h = r g r^{-1} should be
a valley-preserving operation at k1 for the mapped valley pi_r(a0).

This module builds the conjugation graph and identifies which derived operations
are available without additional DFT.

Only unitary space-group operations (det=1) are supported.  Improper unitary
operations (det=-1) are schema-recognised but marked not supported; they are
distinct from antiunitary (TRS) operations which are not represented in the
current spglib unitary operation list.
"""

from __future__ import annotations

import numpy as np

from valleyscope.analysis.hsp_star import _canonical_frac
from valleyscope.symmetry.little_group import (
    reciprocal_transform,
    is_little_group_operation,
)


def build_hsp_star_conjugation_report(
    *,
    kpoint_frac_by_name: dict[str, list[float]],
    operations: list[dict[str, object]],
    valley_names: list[str],
    tolerance: float = 1e-5,
) -> dict[str, object]:
    """Build the HSP-star conjugation graph.

    For each source k-point, examines each space-group operation r that maps it
    to another HSP representative.  Target representatives that are not in the
    explicit HDF5 set are still included with a stable generated key and
    target_kpoint_label=None.

    Returns a report with per-source-kpoint conjugation entries.
    """
    kpoints = {
        str(name): np.asarray(frac, dtype=float)
        for name, frac in kpoint_frac_by_name.items()
    }
    by_source: dict[str, object] = {}

    for source_label, source_frac in kpoints.items():
        entries: list[dict[str, object]] = []
        for op_r in operations:
            r_rotation = op_r.get("rotation_frac")
            r_op_id = op_r.get("operation_id")
            if r_rotation is None or r_op_id is None:
                continue
            r_rot = np.asarray(r_rotation, dtype=float)
            target_frac_raw = reciprocal_transform(r_rot, source_frac)
            target_frac = _canonical_frac(target_frac_raw)
            target_label = _match_kpoint(target_frac, kpoints, tolerance=tolerance)

            # Always proceed — even if target is not in the explicit kpoint set.
            if target_label is not None and target_label == source_label:
                continue

            target_key = _make_target_key(
                explicit_label=target_label,
                canonical_frac=target_frac,
            )

            r_valley_mapping = op_r.get("sector_mapping", {})
            if not isinstance(r_valley_mapping, dict):
                continue

            for source_valley in valley_names:
                target_valley = r_valley_mapping.get(source_valley)
                if target_valley is None:
                    continue
                target_valley = str(target_valley)

                for op_g in operations:
                    g_op_id = op_g.get("operation_id")
                    g_rotation = op_g.get("rotation_frac")
                    if g_op_id is None or g_rotation is None:
                        continue
                    g_rot = np.asarray(g_rotation, dtype=float)

                    g_valley_mapping = op_g.get("sector_mapping", {})
                    if not isinstance(g_valley_mapping, dict):
                        continue
                    g_mapped = g_valley_mapping.get(source_valley)
                    if g_mapped is None or str(g_mapped) != str(source_valley):
                        continue

                    # g must belong to the source kpoint's HSP little group.
                    # Valley-preserving operations that map the kpoint to
                    # another star member are not valid source operations for
                    # character derivation at this kpoint.
                    if not is_little_group_operation(g_rot, source_frac):
                        entries.append(_conjugation_entry(
                            source_kpoint=source_label,
                            source_frac=source_frac,
                            target_kpoint_label=None,
                            target_kpoint_key="",
                            target_frac=np.zeros(3),
                            mapping_operation_id=r_op_id,
                            source_valley=source_valley,
                            target_valley=target_valley,
                            source_operation_id=g_op_id,
                            conjugation_status="source_not_in_hsp_little_group",
                            reason=(
                                f"g={g_op_id} preserves {source_valley} but is "
                                f"not in the HSP little group at {source_label}"
                            ),
                        ))
                        continue

                    # Det != 1 means improper unitary.  We do not
                    # support improper unitary conjugations yet.
                    r_det = int(op_r.get("det", 1))
                    g_det = int(op_g.get("det", 1))
                    if r_det != 1 or g_det != 1:
                        entries.append(_conjugation_entry(
                            source_kpoint=source_label,
                            source_frac=source_frac,
                            target_kpoint_label=target_label,
                            target_kpoint_key=target_key,
                            target_frac=target_frac,
                            mapping_operation_id=r_op_id,
                            source_valley=source_valley,
                            target_valley=target_valley,
                            source_operation_id=g_op_id,
                            conjugation_status="improper_unitary_not_supported",
                            reason="improper unitary conjugation not yet supported (det != 1)",
                        ))
                        continue

                    h_rotation = r_rot @ g_rot @ np.linalg.inv(r_rot)
                    h_translation = _compute_conjugate_translation(
                        r_rot,
                        np.asarray(op_r.get("translation_frac", np.zeros(3)), dtype=float),
                        g_rot,
                        np.asarray(op_g.get("translation_frac", np.zeros(3)), dtype=float),
                    )

                    h_match = _find_operation_by_rotation_translation(
                        operations=operations,
                        rotation=h_rotation,
                        translation=h_translation,
                        tolerance=tolerance,
                    )

                    if h_match is None:
                        entries.append(_conjugation_entry(
                            source_kpoint=source_label,
                            source_frac=source_frac,
                            target_kpoint_label=target_label,
                            target_kpoint_key=target_key,
                            target_frac=target_frac,
                            mapping_operation_id=r_op_id,
                            source_valley=source_valley,
                            target_valley=target_valley,
                            source_operation_id=g_op_id,
                            conjugation_status="missing_operation_product",
                            reason=(
                                f"h = r g r^-1 not found in detected operations; "
                                f"h_rotation={_format_rotation(h_rotation)}"
                            ),
                        ))
                        continue

                    if isinstance(h_match, list):
                        entries.append(_conjugation_entry(
                            source_kpoint=source_label,
                            source_frac=source_frac,
                            target_kpoint_label=target_label,
                            target_kpoint_key=target_key,
                            target_frac=target_frac,
                            mapping_operation_id=r_op_id,
                            source_valley=source_valley,
                            target_valley=target_valley,
                            source_operation_id=g_op_id,
                            derived_target_operation_ids=h_match,
                            conjugation_status="diagnostic_only",
                            reason=f"ambiguous product: {len(h_match)} candidate operations match h",
                        ))
                        continue

                    h_op_id = h_match
                    h_valley_mapping = _get_valley_mapping(operations, h_op_id)
                    h_mapped = h_valley_mapping.get(target_valley) if h_valley_mapping else None

                    if h_mapped is None or str(h_mapped) != str(target_valley):
                        entries.append(_conjugation_entry(
                            source_kpoint=source_label,
                            source_frac=source_frac,
                            target_kpoint_label=target_label,
                            target_kpoint_key=target_key,
                            target_frac=target_frac,
                            mapping_operation_id=r_op_id,
                            source_valley=source_valley,
                            target_valley=target_valley,
                            source_operation_id=g_op_id,
                            derived_target_operation_id=h_op_id,
                            conjugation_status="valley_mapping_mismatch",
                            reason=(
                                f"h={h_op_id} does not preserve {target_valley} "
                                f"(maps to {h_mapped})"
                            ),
                        ))
                        continue

                    entries.append(_conjugation_entry(
                        source_kpoint=source_label,
                        source_frac=source_frac,
                        target_kpoint_label=target_label,
                        target_kpoint_key=target_key,
                        target_frac=target_frac,
                        mapping_operation_id=r_op_id,
                        source_valley=source_valley,
                        target_valley=target_valley,
                        source_operation_id=g_op_id,
                        derived_target_operation_id=h_op_id,
                        conjugation_status="matched",
                        reason="",
                    ))

        if entries:
            by_source[source_label] = entries

    status = "ok" if by_source else "not_evaluated"
    return {
        "status": status,
        "tolerance": tolerance,
        "interpretation": (
            "For each HSP-star pair (k0 -> k1 = r k0), conjugate source "
            "valley-preserving operations g to target operations h = r g r^-1. "
            "matched: h found and preserves mapped valley. "
            "missing_operation_product: h not in detected operations. "
            "improper_unitary_not_supported: det != 1 operations not yet handled. "
            "diagnostic_only: ambiguous match (multiple candidates). "
            "target_kpoint_label is None for symmetry-derivable targets that are "
            "not in the explicit HDF5 k-point set."
        ),
        "by_source_kpoint": by_source,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _conjugation_entry(
    *,
    source_kpoint: str,
    source_frac: np.ndarray,
    target_kpoint_label: str | None,
    target_kpoint_key: str,
    target_frac: np.ndarray,
    mapping_operation_id: object,
    source_valley: str,
    target_valley: str,
    source_operation_id: object,
    derived_target_operation_id: object | None = None,
    derived_target_operation_ids: list[object] | None = None,
    conjugation_status: str,
    reason: str,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "source_kpoint": source_kpoint,
        "source_frac": [float(v) for v in np.asarray(source_frac, dtype=float).tolist()],
        "target_kpoint_label": target_kpoint_label,
        "target_kpoint_key": target_kpoint_key,
        "target_frac": [float(v) for v in np.asarray(target_frac, dtype=float).tolist()],
        "mapping_operation_id": mapping_operation_id,
        "source_valley": source_valley,
        "target_valley": target_valley,
        "source_preserving_operation_id": source_operation_id,
        "conjugation_status": conjugation_status,
        "reason": reason,
    }
    if derived_target_operation_id is not None:
        entry["derived_target_operation_id"] = derived_target_operation_id
    if derived_target_operation_ids is not None:
        entry["derived_target_operation_ids"] = derived_target_operation_ids
    return entry


def _make_target_key(
    *,
    explicit_label: str | None,
    canonical_frac: np.ndarray,
) -> str:
    if explicit_label is not None:
        return explicit_label
    frac_list = [round(float(v), 6) for v in np.asarray(canonical_frac, dtype=float).tolist()]
    return f"derived:{frac_list}"



def _match_kpoint(
    frac: np.ndarray,
    kpoints: dict[str, np.ndarray],
    tolerance: float,
) -> str | None:
    arr = _canonical_frac(frac)
    for label, candidate in kpoints.items():
        delta = arr - _canonical_frac(candidate)
        delta_mod = delta - np.rint(delta)
        if np.allclose(delta_mod, 0.0, atol=tolerance):
            return label
    return None


def _compute_conjugate_translation(
    r_rot: np.ndarray,
    r_trans: np.ndarray,
    g_rot: np.ndarray,
    g_trans: np.ndarray,
) -> np.ndarray:
    r_inv = np.linalg.inv(r_rot)
    return r_rot @ g_rot @ r_inv @ (-r_trans) + r_rot @ g_trans + r_trans


def _find_operation_by_rotation_translation(
    *,
    operations: list[dict[str, object]],
    rotation: np.ndarray,
    translation: np.ndarray,
    tolerance: float,
) -> object | list[object] | None:
    candidates: list[object] = []
    for op in operations:
        op_rot = op.get("rotation_frac")
        op_trans = op.get("translation_frac")
        if op_rot is None:
            continue
        op_rot = np.asarray(op_rot, dtype=float)
        op_trans = np.asarray(op_trans, dtype=float) if op_trans is not None else np.zeros(3)
        if not np.allclose(rotation, op_rot, atol=tolerance):
            continue
        delta = translation - op_trans
        delta_mod = delta - np.rint(delta)
        if np.linalg.norm(delta_mod) <= tolerance:
            candidates.append(op.get("operation_id"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return candidates


def _get_valley_mapping(
    operations: list[dict[str, object]],
    op_id: object,
) -> dict[str, str] | None:
    for op in operations:
        if op.get("operation_id") == op_id:
            mapping = op.get("sector_mapping", {})
            if isinstance(mapping, dict):
                return {str(k): str(v) for k, v in mapping.items() if v is not None}
    return None


def _format_rotation(mat: np.ndarray) -> str:
    arr = np.asarray(mat, dtype=float)
    rows = ["[" + ",".join(f"{v:.1f}" for v in row) + "]" for row in arr]
    return "[" + ";".join(rows) + "]"
