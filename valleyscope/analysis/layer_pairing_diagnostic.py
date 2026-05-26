from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np

from valleyscope.analysis.projector_symmetry import build_projector_symmetry_report
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.projection.sector_projectors import build_sector_projectors
from valleyscope.projection.weights import compute_valley_weights
from valleyscope.subspace.valley_basis import (
    build_valley_adapted_basis,
    build_valley_subspace_matrices,
    summarize_valley_projector_quality,
)
from valleyscope.symmetry.valley_preservation import map_valley_sectors


def build_layer_pairing_permutation_diagnostic(
    *,
    coefficients_by_kpoint: dict[str, np.ndarray],
    q_cart_by_kpoint: dict[str, np.ndarray],
    raw_representations_by_kpoint: dict[str, dict[object, dict[str, object]]],
    operations: list[dict[str, object]],
    centers: list[ValleyCenter],
    valley_names: list[str],
    top_centers_by_valley: dict[str, str],
    bottom_centers_by_valley: dict[str, str],
    monolayer_reciprocal_cart: np.ndarray,
    qcut: float,
    use_2d: bool = True,
    overlap_policy: str = "warn_exclude",
) -> dict[str, object]:
    """Enumerate bottom-layer label pairings and score q-cut seed quality.

    This diagnostic is intentionally generic: it does not assume a material
    name or a particular M-star.  It answers whether a different pairing of
    bottom-layer valley centers with fixed top-layer valley labels would make
    the q-cut seed projectors more complete, more rank-like, and more
    symmetry-consistent.
    """
    _validate_pairing_inputs(
        valley_names=valley_names,
        top_centers_by_valley=top_centers_by_valley,
        bottom_centers_by_valley=bottom_centers_by_valley,
    )
    current_bottom_order = [bottom_centers_by_valley[valley] for valley in valley_names]
    pairings: list[dict[str, object]] = []

    for bottom_order in permutations(current_bottom_order):
        bottom_assignment = {
            valley: bottom_order[index] for index, valley in enumerate(valley_names)
        }
        sectors = [
            ValleySector(
                name=valley,
                centers=[top_centers_by_valley[valley], bottom_assignment[valley]],
            )
            for valley in valley_names
        ]
        valley_matrices_by_kpoint: dict[str, dict[str, np.ndarray]] = {}
        by_kpoint: dict[str, object] = {}
        for kpoint, coefficients in coefficients_by_kpoint.items():
            q_cart = q_cart_by_kpoint[kpoint]
            projectors = build_sector_projectors(
                q_cart,
                centers,
                sectors,
                monolayer_reciprocal_cart,
                qcut,
                use_2d=use_2d,
                overlap_policy=overlap_policy,
                emit_warnings=False,
            )
            weights = compute_valley_weights(coefficients, projectors)
            subspace = build_valley_subspace_matrices(
                coefficients,
                projectors.sector_masks,
            )
            adapted = build_valley_adapted_basis(
                coefficients,
                projectors.sector_masks,
            )
            expected_rank = (
                int(coefficients.shape[0] // len(valley_names))
                if valley_names and coefficients.shape[0] % len(valley_names) == 0
                else None
            )
            quality = summarize_valley_projector_quality(
                subspace.valley_matrices,
                expected_rank=expected_rank,
            )
            valley_matrices_by_kpoint[kpoint] = subspace.valley_matrices
            by_kpoint[kpoint] = {
                "W_val_min": min((item.w_val for item in weights), default=0.0),
                "W_val_max": max((item.w_val for item in weights), default=0.0),
                "W_res_max": max((item.residual_weight for item in weights), default=0.0),
                "W_overlap_max": max((item.overlap_weight for item in weights), default=0.0),
                "P_v_min": min((item.purity for item in weights), default=0.0),
                "P_v_max": max((item.purity for item in weights), default=0.0),
                "S_eigenvalues": [float(value) for value in subspace.s_eigenvalues],
                "S_min": float(subspace.s_min),
                "S_max": float(subspace.s_max),
                "assigned_valleys": list(adapted.assigned_valleys),
                "min_valley_concentration": float(adapted.min_valley_concentration),
                "projector_quality": quality,
            }

        operation_mappings = _operation_mappings_for_sectors(
            operations=operations,
            centers=centers,
            sectors=sectors,
            monolayer_reciprocal_cart=monolayer_reciprocal_cart,
        )
        raw_with_pairing = _raw_representations_with_pairing(
            raw_representations_by_kpoint=raw_representations_by_kpoint,
            operation_mappings=operation_mappings,
        )
        projector_symmetry = build_projector_symmetry_report(
            valley_matrices_by_kpoint=valley_matrices_by_kpoint,
            raw_representations_by_kpoint=raw_with_pairing,
            valley_names=valley_names,
        )
        pairings.append(
            {
                "bottom_assignment": dict(bottom_assignment),
                "is_current_pairing": all(
                    bottom_assignment[valley] == bottom_centers_by_valley[valley]
                    for valley in valley_names
                ),
                "operation_mappings": operation_mappings,
                "by_kpoint": by_kpoint,
                "projector_symmetry": _compact_pairing_projector_symmetry(
                    projector_symmetry
                ),
                "score": _score_pairing(
                    by_kpoint=by_kpoint,
                    projector_symmetry=projector_symmetry,
                ),
            }
        )

    pairings.sort(key=_pairing_sort_key)
    return {
        "status": "ok" if pairings else "no_data",
        "valley_names": list(valley_names),
        "current_bottom_assignment": {
            valley: bottom_centers_by_valley[valley] for valley in valley_names
        },
        "pairings": pairings,
    }


def _validate_pairing_inputs(
    *,
    valley_names: list[str],
    top_centers_by_valley: dict[str, str],
    bottom_centers_by_valley: dict[str, str],
) -> None:
    if not valley_names:
        raise ValueError("valley_names must not be empty")
    for valley in valley_names:
        if valley not in top_centers_by_valley:
            raise ValueError(f"missing top center for valley {valley}")
        if valley not in bottom_centers_by_valley:
            raise ValueError(f"missing bottom center for valley {valley}")
    if len(set(bottom_centers_by_valley.values())) != len(valley_names):
        raise ValueError("bottom centers must be unique")


def _operation_mappings_for_sectors(
    *,
    operations: list[dict[str, object]],
    centers: list[ValleyCenter],
    sectors: list[ValleySector],
    monolayer_reciprocal_cart: np.ndarray,
) -> dict[object, dict[str, str | None]]:
    mappings: dict[object, dict[str, str | None]] = {}
    for operation in operations:
        operation_id = operation.get("operation_id")
        if operation_id is None:
            continue
        rotation_cart = operation.get("rotation_cart")
        if rotation_cart is None:
            continue
        mapping = map_valley_sectors(
            np.eye(3, dtype=float),
            np.asarray(rotation_cart, dtype=float),
            centers,
            sectors,
            monolayer_reciprocal_cart,
            tolerance=1e-6,
        )
        mappings[operation_id] = dict(mapping.sector_mapping)
    return mappings


def _raw_representations_with_pairing(
    *,
    raw_representations_by_kpoint: dict[str, dict[object, dict[str, object]]],
    operation_mappings: dict[object, dict[str, str | None]],
) -> dict[str, dict[object, dict[str, object]]]:
    result: dict[str, dict[object, dict[str, object]]] = {}
    for kpoint, op_payloads in raw_representations_by_kpoint.items():
        result[kpoint] = {}
        for operation_id, payload in op_payloads.items():
            copied = dict(payload)
            if operation_id in operation_mappings:
                copied["sector_mapping"] = dict(operation_mappings[operation_id])
            result[kpoint][operation_id] = copied
    return result


def _compact_pairing_projector_symmetry(
    report: dict[str, object],
) -> dict[str, object]:
    by_kpoint: dict[str, object] = {}
    for kpoint, data in report.get("by_kpoint", {}).items():
        if not isinstance(data, dict):
            continue
        rows = data.get("seed_projector_symmetry", [])
        evaluated = [
            row for row in rows
            if isinstance(row, dict) and row.get("epsilon_seed") is not None
        ]
        eps = [float(row["epsilon_seed"]) for row in evaluated]
        by_kpoint[str(kpoint)] = {
            "evaluated_count": len(eps),
            "failed_count": sum(value > 1.0e-1 for value in eps),
            "max_seed_projector_symmetry_error": max(eps) if eps else None,
            "min_seed_projector_symmetry_error": min(eps) if eps else None,
        }
    return {
        "status": report.get("status"),
        "by_kpoint": by_kpoint,
    }


def _score_pairing(
    *,
    by_kpoint: dict[str, object],
    projector_symmetry: dict[str, object],
) -> dict[str, object]:
    s_mins: list[float] = []
    concentrations: list[float] = []
    w_res: list[float] = []
    eps: list[float] = []
    for data in by_kpoint.values():
        if not isinstance(data, dict):
            continue
        s_mins.append(float(data.get("S_min", 0.0)))
        concentrations.append(float(data.get("min_valley_concentration", 0.0)))
        w_res.append(float(data.get("W_res_max", 0.0)))
    for data in projector_symmetry.get("by_kpoint", {}).values():
        if not isinstance(data, dict):
            continue
        for row in data.get("seed_projector_symmetry", []):
            if isinstance(row, dict) and row.get("epsilon_seed") is not None:
                eps.append(float(row["epsilon_seed"]))
    return {
        "min_s_min": min(s_mins) if s_mins else 0.0,
        "min_valley_concentration": min(concentrations) if concentrations else 0.0,
        "max_W_res": max(w_res) if w_res else 0.0,
        "max_seed_projector_symmetry_error": max(eps) if eps else None,
    }


def _pairing_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, int]:
    score = item.get("score", {})
    if not isinstance(score, dict):
        score = {}
    max_eps = score.get("max_seed_projector_symmetry_error")
    eps_value = float(max_eps) if max_eps is not None else float("inf")
    return (
        -float(score.get("min_s_min", 0.0)),
        -float(score.get("min_valley_concentration", 0.0)),
        eps_value,
        float(score.get("max_W_res", 0.0)),
        0 if item.get("is_current_pairing") else 1,
    )
