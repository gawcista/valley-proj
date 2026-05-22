"""Toy-only character / eigenphase diagnostics for valley-preserving
representation matrices D_a(h) = U_a^dag D_h U_a.

Does NOT integrate into analyze_hsp.  Does NOT perform irrep table matching.
"""

from __future__ import annotations

import numpy as np

DEFAULT_UNITARITY_TOL = 1e-8
DEFAULT_MODULUS_TOL = 1e-8
SMALL_NUMBER = 1e-14


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_valley_preserving_characters(
    *,
    valley_preserving_representations: dict[str, object],
    orbit: list[str],
) -> dict[str, dict[object, complex]]:
    """Compute characters chi_a(h) = Tr[D_a(h)] for each valley.

    Returns {valley: {operation_id: character}}.
    """
    result: dict[str, dict[object, complex]] = {}
    reps = valley_preserving_representations.get("representations", {})
    for valley in orbit:
        vp_reps = reps.get(valley, {})
        chars: dict[object, complex] = {}
        for op_id, d_a in vp_reps.items():
            chars[op_id] = complex(np.trace(np.asarray(d_a, dtype=np.complex128)))
        result[valley] = chars
    return result


def compute_valley_preserving_eigenphases(
    *,
    valley_preserving_representations: dict[str, object],
    orbit: list[str],
) -> dict[str, dict[object, list[float]]]:
    """Compute eigenphases (phase / 2pi) for each D_a(h).

    For eigenvalue lambda, phase phi where lambda ~= exp(i*2pi*phi).
    phi is in (-0.5, 0.5].

    Returns {valley: {operation_id: [phi_0, phi_1, ...]}}.
    """
    result: dict[str, dict[object, list[float]]] = {}
    reps = valley_preserving_representations.get("representations", {})
    for valley in orbit:
        vp_reps = reps.get(valley, {})
        phases: dict[object, list[float]] = {}
        for op_id, d_a in vp_reps.items():
            eigvals = np.linalg.eigvals(np.asarray(d_a, dtype=np.complex128))
            phi_list: list[float] = []
            for lam in eigvals:
                phase = float(np.angle(lam)) / (2.0 * np.pi)
                # Wrap to (-0.5, 0.5]
                if phase <= -0.5:
                    phase += 1.0
                phi_list.append(phase)
            phi_list.sort()
            phases[op_id] = phi_list
        result[valley] = phases
    return result


def build_valley_preserving_character_diagnostics(
    *,
    valley_preserving_representations: dict[str, object],
    orbit: list[str],
    input_diagnostic_only: bool = False,
    input_local_irrep_ready: bool | None = None,
    unitarity_tol: float = DEFAULT_UNITARITY_TOL,
    modulus_tol: float = DEFAULT_MODULUS_TOL,
) -> dict[str, object]:
    """Build character / eigenphase diagnostics from valley-preserving reps.

    Parameters
    ----------
    valley_preserving_representations : dict
        Output of ``build_valley_preserving_representations``.
    orbit : list[str]
    input_diagnostic_only : bool
        Whether the upstream diagnostics already flagged diagnostic_only.
    input_local_irrep_ready : bool or None
        Upstream readiness flag.  If False or None, output will be diagnostic_only.
    unitarity_tol : float
        Maximum allowed VP unitarity error before flagging diagnostic_only.
    modulus_tol : float
        Maximum allowed |eigenvalue| - 1 deviation before flagging diagnostic_only.
    """
    chars = compute_valley_preserving_characters(
        valley_preserving_representations=valley_preserving_representations,
        orbit=orbit,
    )
    phases = compute_valley_preserving_eigenphases(
        valley_preserving_representations=valley_preserving_representations,
        orbit=orbit,
    )

    # --- Readiness ---
    diagnostic_only = bool(input_diagnostic_only)
    local_irrep_ready = not diagnostic_only
    reason = ""
    irrep_matching_status = "not_yet_implemented"

    if input_local_irrep_ready is False:
        diagnostic_only = True
        local_irrep_ready = False
        irrep_matching_status = "failed_input_readiness"
        reason = "upstream diagnostics marked local_irrep_ready=False"

    if not diagnostic_only:
        vp_errors = valley_preserving_representations.get("unitarity_error", {})
        max_vp_err = _max_vp_error(vp_errors)
        if max_vp_err > unitarity_tol:
            diagnostic_only = True
            local_irrep_ready = False
            irrep_matching_status = "failed_unitarity"
            reason = (
                f"valley-preserving unitarity error "
                f"{max_vp_err:.2e} > tol={unitarity_tol:.2e}"
            )

    # Check eigenvalue moduli
    max_modulus_dev = 0.0
    reps = valley_preserving_representations.get("representations", {})
    for valley in orbit:
        vp_reps = reps.get(valley, {})
        for op_id, d_a in vp_reps.items():
            eigvals = np.linalg.eigvals(np.asarray(d_a, dtype=np.complex128))
            for lam in eigvals:
                dev = abs(abs(lam) - 1.0)
                max_modulus_dev = max(max_modulus_dev, dev)

    if not diagnostic_only and max_modulus_dev > modulus_tol:
        diagnostic_only = True
        local_irrep_ready = False
        irrep_matching_status = "failed_unitarity"
        reason = (
            f"eigenvalue modulus deviation {max_modulus_dev:.2e} "
            f"> tol={modulus_tol:.2e}"
        )

    if not diagnostic_only:
        irrep_matching_status = "not_yet_implemented"
        reason = "character diagnostics ready but irrep matching not yet implemented"

    # --- Per-valley diagnostics ---
    per_valley: dict[str, object] = {}
    for valley in orbit:
        vp_reps = reps.get(valley, {})
        vp_errors = valley_preserving_representations.get("unitarity_error", {}).get(valley, {})
        items: list[dict[str, object]] = []
        for op_id in sorted(vp_reps.keys(), key=str):
            chi = chars.get(valley, {}).get(op_id, complex(0, 0))
            phi_list = phases.get(valley, {}).get(op_id, [])
            err = float(vp_errors.get(op_id, 0.0)) if isinstance(vp_errors, dict) else 0.0
            items.append({
                "operation_id": op_id,
                "valley": valley,
                "character": chi,
                "eigenphases": phi_list,
                "representation_unitarity_error": err,
            })
        per_valley[valley] = items

    # --- Max VP unitarity from all valleys ---
    max_vp_unitarity = _max_vp_error(
        valley_preserving_representations.get("unitarity_error", {})
    )

    return {
        "status": "ok" if local_irrep_ready else "diagnostic_only",
        "reason": reason,
        "local_irrep_ready": local_irrep_ready,
        "diagnostic_only": diagnostic_only,
        "irrep_matching_status": irrep_matching_status,
        "max_valley_preserving_unitarity_error": max_vp_unitarity,
        "max_eigenvalue_modulus_deviation": max_modulus_dev,
        "unitarity_tol": unitarity_tol,
        "modulus_tol": modulus_tol,
        "per_valley": per_valley,
    }


def summarize_valley_preserving_character_diagnostics(
    diagnostics: dict[str, object],
) -> dict[str, object]:
    """JSON-safe compact summary, omitting full precision complex values."""

    def _safe(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, complex):
            return {"real": round(v.real, 10), "imag": round(v.imag, 10)}
        return v

    compact_per_valley: dict[str, list[dict[str, object]]] = {}
    for valley, items in diagnostics.get("per_valley", {}).items():
        compact_items: list[dict[str, object]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            compact_items.append({
                "operation_id": _safe(item.get("operation_id")),
                "valley": str(item.get("valley", "")),
                "character": _safe(item.get("character", 0j)),
                "eigenphases": [_safe(p) for p in item.get("eigenphases", [])],
                "representation_unitarity_error":
                    _safe(item.get("representation_unitarity_error", 0.0)),
            })
        compact_per_valley[str(valley)] = compact_items

    return {
        "status": diagnostics.get("status"),
        "reason": diagnostics.get("reason"),
        "local_irrep_ready": diagnostics.get("local_irrep_ready"),
        "diagnostic_only": diagnostics.get("diagnostic_only"),
        "irrep_matching_status": diagnostics.get("irrep_matching_status"),
        "max_valley_preserving_unitarity_error":
            _safe(diagnostics.get("max_valley_preserving_unitarity_error", 0.0)),
        "max_eigenvalue_modulus_deviation":
            _safe(diagnostics.get("max_eigenvalue_modulus_deviation", 0.0)),
        "per_valley": compact_per_valley,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _max_vp_error(vp_errors: object) -> float:
    max_err = 0.0
    if not isinstance(vp_errors, dict):
        return max_err
    for valley_errors in vp_errors.values():
        if isinstance(valley_errors, dict):
            for err in valley_errors.values():
                max_err = max(max_err, float(err))
    return max_err
