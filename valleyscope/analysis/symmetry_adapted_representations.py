"""Toy-only helper: extract valley-preserving representations and valley sewing
matrices from symmetry-adapted valley projector bases U_a.

Does NOT integrate into the production analyze_hsp workflow.
All inputs are synthetic matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_ORTHONORMALITY_TOL = 1e-8
DEFAULT_UNITARITY_TOL = 1e-8


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_valley_preserving_representations(
    *,
    valley_bases: dict[str, np.ndarray],
    representations: dict[object, np.ndarray],
    valley_mappings: dict[object, dict[str, str]],
    orbit: list[str],
    orthonormality_tol: float = DEFAULT_ORTHONORMALITY_TOL,
) -> dict[str, object]:
    """Build valley-preserving representation matrices D_a(g) = U_a^dag D_g U_a.

    Returns dict with keys: representations, unitarity_error, missing_mapping,
    shape_mismatch, status, reason.
    """
    result: dict[str, object] = {
        "representations": {},
        "unitarity_error": {},
        "missing_mapping": [],
        "shape_mismatch": [],
        "status": "ok",
        "reason": "",
    }

    for valley in orbit:
        u_a = _get_validated_basis(valley_bases, valley)
        if u_a is None:
            result["shape_mismatch"].append(
                f"{valley}: U_a missing or shape-incompatible"
            )
            continue
        if not _check_orthonormality(u_a, orthonormality_tol):
            result["shape_mismatch"].append(
                f"{valley}: U_a not orthonormal within tol={orthonormality_tol}"
            )
            continue

        vp_reps: dict[object, np.ndarray] = {}
        vp_errors: dict[object, float] = {}

        for op_id, mapping in valley_mappings.items():
            if op_id not in representations:
                continue
            mapped = mapping.get(valley)
            if mapped is None:
                result["missing_mapping"].append(
                    f"op_{op_id}: pi_g({valley}) not in valley_mapping"
                )
                continue
            if str(mapped) != str(valley):
                continue  # valley-changing, not valley-preserving

            d_g = np.asarray(representations[op_id], dtype=np.complex128)
            d_a = u_a.conj().T @ d_g @ u_a
            vp_reps[op_id] = d_a
            r = d_a.shape[0]
            vp_errors[op_id] = float(
                np.linalg.norm(d_a.conj().T @ d_a - np.eye(r, dtype=np.complex128), ord="fro")
            )

        result["representations"][valley] = vp_reps
        result["unitarity_error"][valley] = vp_errors

    # Status
    if result["shape_mismatch"]:
        result["status"] = "failed"
        result["reason"] = "; ".join(result["shape_mismatch"])
    elif result["missing_mapping"]:
        result["status"] = "partial"
        result["reason"] = "; ".join(result["missing_mapping"])
    return result


def build_valley_sewing_matrices(
    *,
    valley_bases: dict[str, np.ndarray],
    representations: dict[object, np.ndarray],
    valley_mappings: dict[object, dict[str, str]],
    orbit: list[str],
    orthonormality_tol: float = DEFAULT_ORTHONORMALITY_TOL,
) -> dict[str, object]:
    """Build valley sewing matrices B_{ba}(g) = U_b^dag D_g U_a, b = pi_g(a).

    Keyed by (operation_id, source_valley, target_valley).
    """
    result: dict[str, object] = {
        "sewing_matrices": {},
        "unitarity_error": {},
        "missing_mapping": [],
        "shape_mismatch": [],
        "status": "ok",
        "reason": "",
    }

    orbit_set = set(orbit)
    for op_id, mapping in valley_mappings.items():
        if op_id not in representations:
            continue
        d_g = np.asarray(representations[op_id], dtype=np.complex128)
        for src in orbit:
            tgt = mapping.get(src)
            if tgt is None:
                result["missing_mapping"].append(
                    f"op_{op_id}: pi_g({src}) not in valley_mapping"
                )
                continue
            tgt = str(tgt)
            if tgt not in orbit_set:
                result["missing_mapping"].append(
                    f"op_{op_id}: pi_g({src})={tgt} not in orbit"
                )
                continue

            if src == tgt:
                continue  # valley-preserving, not a sewing matrix

            u_src = _get_validated_basis(valley_bases, src)
            u_tgt = _get_validated_basis(valley_bases, tgt)
            if u_src is None or u_tgt is None:
                result["shape_mismatch"].append(
                    f"op_{op_id}: {src}->{tgt}: U_a missing"
                )
                continue
            if not _check_orthonormality(u_src, orthonormality_tol):
                result["shape_mismatch"].append(
                    f"op_{op_id}: {src}->{tgt}: U_{src} not orthonormal"
                )
                continue
            if not _check_orthonormality(u_tgt, orthonormality_tol):
                result["shape_mismatch"].append(
                    f"op_{op_id}: {src}->{tgt}: U_{tgt} not orthonormal"
                )
                continue
            if u_src.shape[1] != u_tgt.shape[1]:
                result["shape_mismatch"].append(
                    f"op_{op_id}: {src}->{tgt}: rank mismatch "
                    f"({u_src.shape[1]} vs {u_tgt.shape[1]})"
                )
                continue

            b = u_tgt.conj().T @ d_g @ u_src
            key = (op_id, src, tgt)
            result["sewing_matrices"][key] = b
            r = b.shape[0]
            result["unitarity_error"][key] = float(
                np.linalg.norm(b.conj().T @ b - np.eye(r, dtype=np.complex128), ord="fro")
            )

    if result["shape_mismatch"]:
        result["status"] = "failed"
        result["reason"] = "; ".join(result["shape_mismatch"])
    elif result["missing_mapping"]:
        result["status"] = "partial"
        result["reason"] = "; ".join(result["missing_mapping"])
    return result


def build_symmetry_adapted_representation_diagnostics(
    *,
    valley_bases: dict[str, np.ndarray],
    representations: dict[object, np.ndarray],
    valley_mappings: dict[object, dict[str, str]],
    orbit: list[str],
    orthonormality_tol: float = DEFAULT_ORTHONORMALITY_TOL,
    unitarity_tol: float = DEFAULT_UNITARITY_TOL,
    closure_mapping: dict[tuple[object, object], object] | None = None,
) -> dict[str, object]:
    """Build full diagnostics for symmetry-adapted representations.

    Combines valley-preserving representations, valley sewing matrices,
    and optional representation closure checks.
    """
    vp_result = build_valley_preserving_representations(
        valley_bases=valley_bases,
        representations=representations,
        valley_mappings=valley_mappings,
        orbit=orbit,
        orthonormality_tol=orthonormality_tol,
    )
    sewing_result = build_valley_sewing_matrices(
        valley_bases=valley_bases,
        representations=representations,
        valley_mappings=valley_mappings,
        orbit=orbit,
        orthonormality_tol=orthonormality_tol,
    )

    # Aggregate errors
    max_vp_unitarity = 0.0
    for vp_errors in vp_result.get("unitarity_error", {}).values():
        if isinstance(vp_errors, dict):
            for err in vp_errors.values():
                max_vp_unitarity = max(max_vp_unitarity, float(err))

    max_sewing_unitarity = 0.0
    for err in sewing_result.get("unitarity_error", {}).values():
        max_sewing_unitarity = max(max_sewing_unitarity, float(err))

    # Readiness
    local_irrep_ready = True
    diagnostic_only = False
    reasons: list[str] = []

    if vp_result["status"] == "failed" or sewing_result["status"] == "failed":
        local_irrep_ready = False
        diagnostic_only = True
        reasons.append("shape_mismatch_in_valley_bases")
    if max_vp_unitarity > unitarity_tol:
        local_irrep_ready = False
        diagnostic_only = True
        reasons.append(
            f"valley_preserving_unitarity_error={max_vp_unitarity:.2e} > tol={unitarity_tol:.2e}"
        )
    if max_sewing_unitarity > unitarity_tol:
        local_irrep_ready = False
        diagnostic_only = True
        reasons.append(
            f"sewing_unitarity_error={max_sewing_unitarity:.2e} > tol={unitarity_tol:.2e}"
        )

    # Valley-preserving and valley-changing operation classification
    vp_ops: dict[str, list[object]] = {}
    vc_ops: dict[str, list[object]] = {}
    for valley in orbit:
        vp_ops[valley] = []
        vc_ops[valley] = []
        for op_id, mapping in valley_mappings.items():
            mapped = mapping.get(valley)
            if mapped is None:
                continue
            if str(mapped) == str(valley):
                vp_ops[valley].append(op_id)
            else:
                vc_ops[valley].append(op_id)

    # Rank by valley
    rank_by_valley: dict[str, int] = {}
    for valley in orbit:
        u = valley_bases.get(valley)
        rank_by_valley[valley] = u.shape[1] if u is not None else 0

    # Closure diagnostics (only if closure_mapping provided)
    closure_status = "not_evaluated"
    missing_products: list[dict[str, object]] = []
    if closure_mapping is not None:
        closure_status, missing_products = _check_representation_closure(
            valley_bases=valley_bases,
            representations=representations,
            valley_mappings=valley_mappings,
            orbit=orbit,
            closure_mapping=closure_mapping,
            unitarity_tol=unitarity_tol,
        )

    reason = "; ".join(reasons) if reasons else "all diagnostics within tolerance"

    return {
        "status": "ok" if local_irrep_ready else "failed",
        "reason": reason,
        "local_irrep_ready": local_irrep_ready,
        "diagnostic_only": diagnostic_only,
        "orbit": orbit,
        "selected_rank_by_valley": rank_by_valley,
        "valley_preserving_operations": vp_ops,
        "valley_changing_operations": vc_ops,
        "max_valley_preserving_unitarity_error": max_vp_unitarity,
        "max_sewing_unitarity_error": max_sewing_unitarity,
        "representation_closure_status": closure_status,
        "representation_closure_missing_products": (
            missing_products if missing_products else []
        ),
        "valley_preserving_representations": vp_result,
        "valley_sewing_matrices": sewing_result,
    }


def summarize_symmetry_adapted_representations(
    diagnostics: dict[str, object],
) -> dict[str, object]:
    """Produce a JSON-safe compact summary, omitting large matrices."""
    def _safe(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, complex):
            return {"real": v.real, "imag": v.imag}
        return v

    vp_result = diagnostics.get("valley_preserving_representations", {})
    sewing_result = diagnostics.get("valley_sewing_matrices", {})

    # Compact sewing summary: per key, record shape and unitarity error only
    compact_sewing: list[dict[str, object]] = []
    for key, err in sewing_result.get("unitarity_error", {}).items():
        if not isinstance(key, tuple) or len(key) != 3:
            continue
        op_id, src, tgt = key
        b = sewing_result.get("sewing_matrices", {}).get(key)
        shape = list(b.shape) if b is not None else None
        compact_sewing.append({
            "operation_id": _safe(op_id),
            "source_valley": str(src),
            "target_valley": str(tgt),
            "shape": shape,
            "sewing_unitarity_error": _safe(err),
        })

    # Compact valley-preserving summary
    compact_vp: dict[str, list[dict[str, object]]] = {}
    for valley, reps in vp_result.get("representations", {}).items():
        items: list[dict[str, object]] = []
        errors = vp_result.get("unitarity_error", {}).get(valley, {})
        for op_id, d_a in reps.items():
            items.append({
                "operation_id": _safe(op_id),
                "shape": list(d_a.shape),
                "valley_preserving_unitarity_error": _safe(errors.get(op_id, None)),
            })
        compact_vp[str(valley)] = items

    return {
        "status": diagnostics.get("status"),
        "reason": diagnostics.get("reason"),
        "local_irrep_ready": diagnostics.get("local_irrep_ready"),
        "diagnostic_only": diagnostics.get("diagnostic_only"),
        "orbit": diagnostics.get("orbit"),
        "selected_rank_by_valley": {
            str(k): int(v) for k, v in
            diagnostics.get("selected_rank_by_valley", {}).items()
        },
        "valley_preserving_operations": diagnostics.get("valley_preserving_operations"),
        "valley_changing_operations": diagnostics.get("valley_changing_operations"),
        "max_valley_preserving_unitarity_error":
            _safe(diagnostics.get("max_valley_preserving_unitarity_error", 0.0)),
        "max_sewing_unitarity_error":
            _safe(diagnostics.get("max_sewing_unitarity_error", 0.0)),
        "representation_closure_status":
            diagnostics.get("representation_closure_status"),
        "valley_preserving_representations": compact_vp,
        "valley_sewing_matrices_summary": compact_sewing,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_validated_basis(
    valley_bases: dict[str, np.ndarray], valley: str
) -> np.ndarray | None:
    u = valley_bases.get(valley)
    if u is None:
        return None
    u = np.asarray(u, dtype=np.complex128)
    if u.ndim != 2 or u.shape[0] < 1 or u.shape[1] < 1:
        return None
    return u


def _check_orthonormality(u: np.ndarray, tol: float) -> bool:
    r = u.shape[1]
    gram = u.conj().T @ u
    return bool(np.linalg.norm(gram - np.eye(r, dtype=np.complex128), ord="fro") <= tol)


def _check_representation_closure(
    *,
    valley_bases: dict[str, np.ndarray],
    representations: dict[object, np.ndarray],
    valley_mappings: dict[object, dict[str, str]],
    orbit: list[str],
    closure_mapping: dict[tuple[object, object], object],
    unitarity_tol: float,
) -> tuple[str, list[dict[str, object]]]:
    """Check that valley-preserving reps satisfy D(g1)D(g2) ~= D(g1*g2).

    Only checks pairs (g1, g2) that are both valley-preserving for the same
    valley, AND whose product is in the closure_mapping table.
    If closure_mapping is empty or no product involves valley-preserving ops,
    returns ("not_evaluated", []).
    """
    missing: list[dict[str, object]] = []
    if not closure_mapping:
        return "not_evaluated", missing

    any_checked = False
    for valley in orbit:
        vp_op_ids = []
        for op_id, mapping in valley_mappings.items():
            mapped = mapping.get(valley)
            if mapped is not None and str(mapped) == str(valley):
                vp_op_ids.append(op_id)

        u_a = valley_bases.get(valley)
        if u_a is None:
            continue
        u_a = np.asarray(u_a, dtype=np.complex128)

        for g1 in vp_op_ids:
            for g2 in vp_op_ids:
                product_key = (g1, g2)
                if product_key not in closure_mapping:
                    continue
                expected_op = closure_mapping[product_key]
                if g1 not in representations or g2 not in representations:
                    continue
                if expected_op not in representations:
                    continue
                any_checked = True
                d1 = np.asarray(representations[g1], dtype=np.complex128)
                d2 = np.asarray(representations[g2], dtype=np.complex128)
                d_expected = np.asarray(representations[expected_op], dtype=np.complex128)

                d1_a = u_a.conj().T @ d1 @ u_a
                d2_a = u_a.conj().T @ d2 @ u_a
                product = d1_a @ d2_a
                expected_a = u_a.conj().T @ d_expected @ u_a

                err = float(np.linalg.norm(product - expected_a, ord="fro"))
                if err > unitarity_tol:
                    missing.append({
                        "valley": valley,
                        "g1": g1,
                        "g2": g2,
                        "expected_product_operation_id": expected_op,
                        "error": err,
                    })

    if not any_checked:
        return "not_evaluated", missing
    return "closed" if not missing else "not_closed", missing
