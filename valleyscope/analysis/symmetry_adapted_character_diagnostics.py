"""Toy-only character / eigenphase diagnostics for valley-preserving
representation matrices D_a(h) = U_a^dag D_h U_a.

Does NOT integrate into analyze_hsp.  Does NOT perform irrep table matching.
"""

from __future__ import annotations

import numpy as np

DEFAULT_UNITARITY_TOL = 1e-8
DEFAULT_MODULUS_TOL = 1e-8
PHASE_CONVENTION = "phase_over_2pi_in_interval_minus_half_to_half"


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
            matrix = _as_square_matrix(d_a)
            if matrix is None:
                continue
            chars[op_id] = complex(np.trace(matrix))
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
            matrix = _as_square_matrix(d_a)
            if matrix is None:
                continue
            eigvals = np.linalg.eigvals(matrix)
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
        Upstream readiness flag. If False, output will be diagnostic_only. None
        means the caller did not provide this upstream gate.
    unitarity_tol : float
        Maximum allowed VP unitarity error before flagging diagnostic_only.
    modulus_tol : float
        Maximum allowed |eigenvalue| - 1 deviation before flagging diagnostic_only.
    """
    matrix_errors, computed_max_unitarity, max_modulus_dev, matrix_issues = (
        _validate_valley_preserving_matrices(
            valley_preserving_representations=valley_preserving_representations,
            orbit=orbit,
        )
    )

    # --- Readiness ---
    diagnostic_only = False
    local_irrep_ready = True
    reasons: list[str] = []
    irrep_matching_status = "not_yet_implemented"

    source_status = str(valley_preserving_representations.get("status", "ok"))
    if input_diagnostic_only:
        diagnostic_only = True
        local_irrep_ready = False
        irrep_matching_status = "failed_input_readiness"
        reasons.append("upstream diagnostics marked diagnostic_only=True")
    if input_local_irrep_ready is False:
        diagnostic_only = True
        local_irrep_ready = False
        irrep_matching_status = "failed_input_readiness"
        reasons.append("upstream diagnostics marked local_irrep_ready=False")
    if source_status in {"failed", "partial", "diagnostic_only"}:
        diagnostic_only = True
        local_irrep_ready = False
        irrep_matching_status = "failed_input_readiness"
        source_reason = valley_preserving_representations.get("reason", "")
        reasons.append(
            f"valley-preserving representation diagnostics status={source_status}"
            + (f": {source_reason}" if source_reason else "")
        )
    if matrix_issues:
        diagnostic_only = True
        local_irrep_ready = False
        irrep_matching_status = "failed_input_readiness"
        reasons.append(
            "invalid valley-preserving representation matrices: "
            + "; ".join(matrix_issues)
        )

    vp_errors = valley_preserving_representations.get("unitarity_error", {})
    max_vp_err = max(_max_vp_error(vp_errors), computed_max_unitarity)
    if max_vp_err > unitarity_tol:
        diagnostic_only = True
        local_irrep_ready = False
        if irrep_matching_status != "failed_input_readiness":
            irrep_matching_status = "failed_unitarity"
        reasons.append(
            f"valley-preserving unitarity error "
            f"{max_vp_err:.2e} > tol={unitarity_tol:.2e}"
        )

    if max_modulus_dev > modulus_tol:
        diagnostic_only = True
        local_irrep_ready = False
        if irrep_matching_status != "failed_input_readiness":
            irrep_matching_status = "failed_unitarity"
        reasons.append(
            f"eigenvalue modulus deviation {max_modulus_dev:.2e} "
            f"> tol={modulus_tol:.2e}"
        )

    if not diagnostic_only:
        irrep_matching_status = "not_yet_implemented"
        reasons.append("character diagnostics ready but irrep matching not yet implemented")

    reason = "; ".join(reasons)

    chars = compute_valley_preserving_characters(
        valley_preserving_representations=valley_preserving_representations,
        orbit=orbit,
    )
    phases = compute_valley_preserving_eigenphases(
        valley_preserving_representations=valley_preserving_representations,
        orbit=orbit,
    )

    # --- Per-valley diagnostics ---
    reps = valley_preserving_representations.get("representations", {})
    per_valley: dict[str, object] = {}
    for valley in orbit:
        vp_reps = reps.get(valley, {})
        vp_errors = valley_preserving_representations.get("unitarity_error", {}).get(valley, {})
        items: list[dict[str, object]] = []
        for op_id in sorted(vp_reps.keys(), key=str):
            chi = chars.get(valley, {}).get(op_id, complex(0, 0))
            phi_list = phases.get(valley, {}).get(op_id, [])
            err = matrix_errors.get((valley, op_id))
            if err is None:
                err = (
                    float(vp_errors.get(op_id, 0.0))
                    if isinstance(vp_errors, dict)
                    else 0.0
                )
            items.append({
                "operation_id": op_id,
                "valley": valley,
                "character": chi,
                "eigenphases": phi_list,
                "representation_unitarity_error": err,
            })
        per_valley[valley] = items

    return {
        "status": "ok" if local_irrep_ready else "diagnostic_only",
        "reason": reason,
        "local_irrep_ready": local_irrep_ready,
        "diagnostic_only": diagnostic_only,
        "irrep_matching_status": irrep_matching_status,
        "max_valley_preserving_unitarity_error": max_vp_err,
        "max_eigenvalue_modulus_deviation": max_modulus_dev,
        "unitarity_tol": unitarity_tol,
        "modulus_tol": modulus_tol,
        "phase_convention": PHASE_CONVENTION,
        "per_valley": per_valley,
    }


def summarize_valley_preserving_character_diagnostics(
    diagnostics: dict[str, object],
) -> dict[str, object]:
    """JSON-safe compact summary, omitting full precision complex values."""

    def _safe(v):
        if isinstance(v, np.ndarray):
            return _safe(v.tolist())
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, complex):
            return {"real": round(v.real, 10), "imag": round(v.imag, 10)}
        if isinstance(v, tuple):
            return [_safe(item) for item in v]
        if isinstance(v, list):
            return [_safe(item) for item in v]
        if isinstance(v, dict):
            return {str(_safe(k)): _safe(item) for k, item in v.items()}
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
        "phase_convention": diagnostics.get("phase_convention", PHASE_CONVENTION),
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


def _as_square_matrix(matrix: object) -> np.ndarray | None:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    return arr


def _validate_valley_preserving_matrices(
    *,
    valley_preserving_representations: dict[str, object],
    orbit: list[str],
) -> tuple[dict[tuple[str, object], float], float, float, list[str]]:
    errors: dict[tuple[str, object], float] = {}
    max_unitarity = 0.0
    max_modulus_dev = 0.0
    issues: list[str] = []

    reps = valley_preserving_representations.get("representations", {})
    if not isinstance(reps, dict):
        return errors, max_unitarity, max_modulus_dev, ["representations is not a dict"]

    for valley in orbit:
        vp_reps = reps.get(valley, {})
        if not isinstance(vp_reps, dict):
            issues.append(f"{valley}: representations entry is not a dict")
            continue

        for op_id, d_a in vp_reps.items():
            matrix = _as_square_matrix(d_a)
            if matrix is None:
                shape = np.asarray(d_a).shape
                issues.append(f"{valley}, op_{op_id}: D_a(h) must be square, got {shape}")
                continue

            rank = matrix.shape[0]
            unitary_error = float(
                np.linalg.norm(
                    matrix.conj().T @ matrix - np.eye(rank, dtype=np.complex128),
                    ord="fro",
                )
            )
            errors[(valley, op_id)] = unitary_error
            max_unitarity = max(max_unitarity, unitary_error)

            eigvals = np.linalg.eigvals(matrix)
            for lam in eigvals:
                max_modulus_dev = max(max_modulus_dev, abs(abs(lam) - 1.0))

    return errors, max_unitarity, max_modulus_dev, issues
