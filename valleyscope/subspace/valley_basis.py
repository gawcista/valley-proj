from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Conservative default thresholds (used when projection.thresholds is absent)
# ---------------------------------------------------------------------------
_DEFAULT_W_VAL_MIN = 0.8
_DEFAULT_CONCENTRATION_CLEAN = 0.95
_DEFAULT_COMMUTATOR_WARN = 1e-3
_DEFAULT_IDEMPOTENCY_WARN = 1e-3


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValleySubspaceMatrices:
    valley_matrices: dict[str, np.ndarray]
    s_matrix: np.ndarray
    s_eigenvalues: np.ndarray
    s_min: float
    s_max: float
    commutator_norm_max: float
    idempotency_deviation_max: float


@dataclass(frozen=True)
class ValleyBasisResult:
    transform: np.ndarray
    valley_labels: list[str]
    valley_label_values: np.ndarray
    valley_weights_adapted: np.ndarray  # [n_state, n_valley]
    assigned_valleys: list[str]
    valley_concentration: np.ndarray  # per adapted state
    min_valley_concentration: float
    s_expectation: np.ndarray         # diag(U^H S U)
    eta_adapted: np.ndarray | None    # only for exactly two valleys
    s_matrix: np.ndarray
    valley_matrices: dict[str, np.ndarray]
    label_operator: np.ndarray
    stably_separable: bool
    reason: str
    commutator_norm_max: float = 0.0
    idempotency_deviation_max: float = 0.0
    diagnostic_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def _projector_matrix(coefficients: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape [nb,nspinor,nG]")
    flat = coeffs.reshape(coeffs.shape[0], -1)
    expanded_mask = np.tile(mask.astype(bool), coeffs.shape[1])
    selected = flat[:, expanded_mask]
    return selected @ selected.conj().T


def _default_valley_labels(n_valleys: int) -> tuple[list[str], np.ndarray]:
    """Deterministic labels matching legacy two-valley ordering."""
    if n_valleys == 2:
        values = np.array([1.0, -1.0], dtype=float)
    else:
        values = np.arange(n_valleys, dtype=float)
    labels = [f"valley_{i}" for i in range(n_valleys)]
    return labels, values


def build_valley_subspace_matrices(
    coefficients: np.ndarray,
    valley_masks: dict[str, np.ndarray],
) -> ValleySubspaceMatrices:
    """Project valley masks into the target subspace spanned by *coefficients*.

    For each valley label *a* with mask *mask_a*:

        P_a^sub[i,j] = <psi_i | P_a | psi_j>

    S = sum_a P_a^sub
    """
    n_valleys = len(valley_masks)
    if n_valleys < 1:
        raise ValueError("At least one valley mask is required")

    names = list(valley_masks)
    matrices: dict[str, np.ndarray] = {}
    for name in names:
        matrices[name] = _projector_matrix(coefficients, valley_masks[name])

    s_matrix = sum(matrices.values())  # type: ignore[arg-type]
    s_eig = np.linalg.eigvalsh(s_matrix)
    s_min = float(np.min(s_eig)) if len(s_eig) else 0.0
    s_max = float(np.max(s_eig)) if len(s_eig) else 0.0

    # Commutator check
    commutator_norm_max = 0.0
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            a = matrices[a_name]
            b = matrices[b_name]
            commutator_norm_max = max(
                commutator_norm_max, float(np.linalg.norm(a @ b - b @ a))
            )

    # Idempotency check (projected projectors are not strict projectors, so
    # this is a diagnostic, not a hard gate)
    idempotency_deviation_max = 0.0
    for name in names:
        mat = matrices[name]
        idempotency_deviation_max = max(
            idempotency_deviation_max, float(np.linalg.norm(mat @ mat - mat))
        )

    return ValleySubspaceMatrices(
        valley_matrices=matrices,
        s_matrix=s_matrix,
        s_eigenvalues=s_eig,
        s_min=s_min,
        s_max=s_max,
        commutator_norm_max=commutator_norm_max,
        idempotency_deviation_max=idempotency_deviation_max,
    )


def summarize_valley_projector_quality(
    valley_matrices: dict[str, np.ndarray],
    *,
    expected_rank: int | None = None,
    rank_threshold: float = 0.5,
) -> dict[str, object]:
    """Return JSON-safe diagnostics for projected q-cut seed matrices.

    These matrices are P_a^sub = <psi_i|P_a^0|psi_j> inside the target DFT
    subspace. They need not be exact projectors, so all values here are
    diagnostics rather than readiness gates.
    """
    if not valley_matrices:
        return {
            "expected_rank": expected_rank,
            "rank_threshold": float(rank_threshold),
            "per_valley": {},
            "pairwise": {},
            "sum_projector": {
                "eigenvalues": [],
                "trace": 0.0,
                "identity_deviation_fro": None,
                "idempotency_deviation_fro": 0.0,
            },
            "max_trace_overlap": 0.0,
            "max_commutator_norm": 0.0,
            "max_idempotency_deviation": 0.0,
        }

    names = list(valley_matrices)
    matrices = {
        name: np.asarray(matrix, dtype=np.complex128)
        for name, matrix in valley_matrices.items()
    }
    first_shape = matrices[names[0]].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError("valley matrices must be square")
    dim = first_shape[0]
    for name, matrix in matrices.items():
        if matrix.shape != first_shape:
            raise ValueError(f"valley matrix {name} has inconsistent shape")

    per_valley: dict[str, object] = {}
    max_idempotency = 0.0
    for name in names:
        matrix = (matrices[name] + matrices[name].conj().T) / 2.0
        eigvals = np.linalg.eigvalsh(matrix)
        eig_desc = np.sort(np.real(eigvals))[::-1]
        rank_estimate = int(np.sum(eig_desc > rank_threshold))
        rank_gap = None
        if expected_rank is not None and 0 < expected_rank < len(eig_desc):
            rank_gap = float(eig_desc[expected_rank - 1] - eig_desc[expected_rank])
        idempotency = float(np.linalg.norm(matrix @ matrix - matrix, ord="fro"))
        max_idempotency = max(max_idempotency, idempotency)
        per_valley[name] = {
            "trace": float(np.real(np.trace(matrix))),
            "eigenvalues": [float(value) for value in eig_desc],
            "rank_estimate": rank_estimate,
            "rank_threshold": float(rank_threshold),
            "rank_gap": rank_gap,
            "idempotency_deviation_fro": idempotency,
        }

    pairwise: dict[str, object] = {}
    max_trace_overlap = 0.0
    max_commutator = 0.0
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            a = matrices[a_name]
            b = matrices[b_name]
            trace_overlap = float(np.real(np.trace(a @ b)))
            commutator = float(np.linalg.norm(a @ b - b @ a, ord="fro"))
            max_trace_overlap = max(max_trace_overlap, abs(trace_overlap))
            max_commutator = max(max_commutator, commutator)
            pairwise[f"{a_name}__{b_name}"] = {
                "trace_overlap": trace_overlap,
                "commutator_norm": commutator,
            }

    s_matrix = sum(matrices.values())  # type: ignore[arg-type]
    s_hermitian = (s_matrix + s_matrix.conj().T) / 2.0
    s_eigs = np.sort(np.real(np.linalg.eigvalsh(s_hermitian)))[::-1]
    sum_projector = {
        "trace": float(np.real(np.trace(s_matrix))),
        "eigenvalues": [float(value) for value in s_eigs],
        "identity_deviation_fro": float(
            np.linalg.norm(s_matrix - np.eye(dim, dtype=np.complex128), ord="fro")
        ),
        "idempotency_deviation_fro": float(
            np.linalg.norm(s_matrix @ s_matrix - s_matrix, ord="fro")
        ),
    }

    return {
        "expected_rank": expected_rank,
        "rank_threshold": float(rank_threshold),
        "per_valley": per_valley,
        "pairwise": pairwise,
        "sum_projector": sum_projector,
        "max_trace_overlap": float(max_trace_overlap),
        "max_commutator_norm": float(max_commutator),
        "max_idempotency_deviation": float(max_idempotency),
    }


def build_valley_adapted_basis(
    coefficients: np.ndarray,
    valley_masks: dict[str, np.ndarray],
    *,
    valley_label_values: np.ndarray | None = None,
) -> ValleyBasisResult:
    """General multi-valley adapted basis construction.

    Given *N_v* valley projectors P_a, construct the projected matrices
    P_a^sub in the target subspace and then the valley-label operator

        L = sum_a lambda_a P_a^sub

    where *lambda_a* are distinct real numbers distinguishing valleys
    numerically.  Diagonalising L yields the valley-adapted basis.

    For exactly two valleys the compat field ``eta_adapted`` is also
    populated.
    """
    n_states = coefficients.shape[0]
    names = list(valley_masks)
    n_valleys = len(names)
    if n_valleys < 1:
        raise ValueError("At least one valley mask is required")

    if valley_label_values is None:
        _, valley_label_values = _default_valley_labels(n_valleys)
    else:
        valley_label_values = np.asarray(valley_label_values, dtype=float)
        if valley_label_values.shape != (n_valleys,):
            raise ValueError("valley_label_values must have shape (n_valleys,)")

    labels, _ = _default_valley_labels(n_valleys)

    # Build subspace matrices
    matrices: dict[str, np.ndarray] = {}
    for name in names:
        matrices[name] = _projector_matrix(coefficients, valley_masks[name])
    s_matrix = sum(matrices.values())  # type: ignore[arg-type]

    # Label operator L = sum_a lambda_a P_a^sub
    label_op = sum(
        float(valley_label_values[i]) * matrices[names[i]]
        for i in range(n_valleys)
    )

    # Diagonalise L to obtain valley-adapted basis
    eigvals, eigvecs = np.linalg.eigh(label_op)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Valley weights in the adapted basis: w_alpha_a = <phi_alpha|P_a|phi_alpha>
    valley_weights = np.zeros((n_states, n_valleys), dtype=float)
    for a_idx, name in enumerate(names):
        mat = matrices[name]
        for alpha in range(n_states):
            vec = eigvecs[:, alpha]
            valley_weights[alpha, a_idx] = float(np.real(vec.conj() @ mat @ vec))

    # s_expectation = diag(U^H S U)
    s_expectation = np.array(
        [float(np.real(eigvecs[:, i].conj() @ s_matrix @ eigvecs[:, i]))
         for i in range(n_states)]
    )

    # Valley assignment and concentration
    assigned_valleys: list[str] = []
    valley_concentration = np.zeros(n_states, dtype=float)
    for alpha in range(n_states):
        a_idx = int(np.argmax(valley_weights[alpha]))
        assigned_valleys.append(names[a_idx])
        s_a = float(np.sum(valley_weights[alpha]))
        valley_concentration[alpha] = (
            float(np.max(valley_weights[alpha])) / s_a if s_a > 1e-14 else 0.0
        )

    min_valley_concentration = (
        float(np.min(valley_concentration)) if len(valley_concentration) else 0.0
    )

    # eta_adapted only for exactly two valleys
    eta_adapted: np.ndarray | None = None
    if n_valleys == 2:
        eta_adapted = np.zeros(n_states, dtype=float)
        for alpha in range(n_states):
            w0 = valley_weights[alpha, 0]
            w1 = valley_weights[alpha, 1]
            denom = w0 + w1
            eta_adapted[alpha] = (
                float((w0 - w1) / denom) if denom > 1e-14 else 0.0
            )

    # Commutator and idempotency diagnostics
    commutator_norm_max = 0.0
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            a = matrices[a_name]
            b = matrices[b_name]
            commutator_norm_max = max(
                commutator_norm_max, float(np.linalg.norm(a @ b - b @ a))
            )

    idempotency_deviation_max = 0.0
    for name in names:
        mat = matrices[name]
        idempotency_deviation_max = max(
            idempotency_deviation_max, float(np.linalg.norm(mat @ mat - mat))
        )

    # s_eigenvalues from S matrix
    s_eig = np.linalg.eigvalsh(s_matrix)
    s_min = float(np.min(s_eig)) if len(s_eig) else 0.0

    return ValleyBasisResult(
        transform=eigvecs,
        valley_labels=names,
        valley_label_values=valley_label_values,
        valley_weights_adapted=valley_weights,
        assigned_valleys=assigned_valleys,
        valley_concentration=valley_concentration,
        min_valley_concentration=min_valley_concentration,
        s_expectation=s_expectation,
        eta_adapted=eta_adapted,
        s_matrix=s_matrix,
        valley_matrices=matrices,
        label_operator=label_op,
        stably_separable=False,  # filled by diagnose_valley_separability
        reason="not_evaluated",
        commutator_norm_max=commutator_norm_max,
        idempotency_deviation_max=idempotency_deviation_max,
    )


def diagnose_valley_separability(
    result: ValleyBasisResult,
    *,
    w_val_min: float | None = None,
    concentration_threshold: float | None = None,
    commutator_tol: float | None = None,
    idempotency_tol: float | None = None,
) -> ValleyBasisResult:
    """Attach ``stably_separable``, ``reason``, and ``diagnostic_notes``.

    The idempotency deviation of projected valley matrices is collected as a
    diagnostic note rather than a hard rejection gate.  In a finite target-band
    subspace, P_a^sub is generally not a strict projector, so idempotency
    failure alone does not invalidate valley separability.
    """
    w_val_min = _DEFAULT_W_VAL_MIN if w_val_min is None else w_val_min
    concentration_threshold = (
        _DEFAULT_CONCENTRATION_CLEAN if concentration_threshold is None
        else concentration_threshold
    )
    commutator_tol = (
        _DEFAULT_COMMUTATOR_WARN if commutator_tol is None else commutator_tol
    )
    idempotency_tol = (
        _DEFAULT_IDEMPOTENCY_WARN if idempotency_tol is None else idempotency_tol
    )

    s_eig = np.linalg.eigvalsh(result.s_matrix)
    s_min = float(np.min(s_eig)) if len(s_eig) else 0.0
    diagnostic_notes: list[str] = []

    if result.idempotency_deviation_max > idempotency_tol:
        diagnostic_notes.append(
            f"projected valley matrix idempotency deviation "
            f"({result.idempotency_deviation_max:.2e}) exceeds "
            f"diagnostic tol ({idempotency_tol:.2e}); "
            f"this is expected in finite target-band subspaces and is "
            f"not a hard separability rejection"
        )

    def _final(stably_separable: bool, reason: str) -> ValleyBasisResult:
        return ValleyBasisResult(
            transform=result.transform,
            valley_labels=result.valley_labels,
            valley_label_values=result.valley_label_values,
            valley_weights_adapted=result.valley_weights_adapted,
            assigned_valleys=result.assigned_valleys,
            valley_concentration=result.valley_concentration,
            min_valley_concentration=result.min_valley_concentration,
            s_expectation=result.s_expectation,
            eta_adapted=result.eta_adapted,
            s_matrix=result.s_matrix,
            valley_matrices=result.valley_matrices,
            label_operator=result.label_operator,
            stably_separable=stably_separable,
            reason=reason,
            commutator_norm_max=result.commutator_norm_max,
            idempotency_deviation_max=result.idempotency_deviation_max,
            diagnostic_notes=diagnostic_notes,
        )

    if s_min < w_val_min:
        return _final(False, "insufficient_valley_derived_score")

    if result.commutator_norm_max > commutator_tol:
        return _final(False, "non_commuting_valley_projectors")

    if result.min_valley_concentration < concentration_threshold:
        return _final(False, "insufficient_valley_concentration")

    return _final(True, "stably_separable")


# ---------------------------------------------------------------------------
# Legacy compatibility wrappers (keep existing callers working)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoValleyBasisResult:
    """Legacy result, kept for backward compatibility.

    Prefer ``ValleyBasisResult`` for new code.
    """
    transform: np.ndarray
    eta: np.ndarray
    s_matrix: np.ndarray
    v_matrix: np.ndarray


@dataclass(frozen=True)
class MultiValleyDiagnostic:
    """Legacy diagnostic, kept for backward compatibility.

    Prefer ``diagnose_valley_separability`` on a ``ValleyBasisResult``.
    """
    stably_separable: bool
    reason: str
    eigenvalues: dict[str, np.ndarray]
    max_commutator_norm: float


def build_two_valley_adapted_basis(
    coefficients: np.ndarray,
    sector_masks: dict[str, np.ndarray],
    first_sector: str,
    second_sector: str,
) -> TwoValleyBasisResult:
    """Legacy entry point for two-valley adapted basis.

    Delegates to the general ``build_valley_adapted_basis`` and returns
    the old ``TwoValleyBasisResult`` shape for compatibility.
    """
    if first_sector not in sector_masks or second_sector not in sector_masks:
        raise ValueError("Both valley sectors must be present")
    masks = {first_sector: sector_masks[first_sector],
             second_sector: sector_masks[second_sector]}
    result = build_valley_adapted_basis(coefficients, masks)
    v_matrix = result.valley_matrices[first_sector] - result.valley_matrices[second_sector]
    return TwoValleyBasisResult(
        transform=result.transform,
        eta=result.eta_adapted if result.eta_adapted is not None else np.zeros(result.transform.shape[1]),
        s_matrix=result.s_matrix,
        v_matrix=v_matrix,
    )


def diagnose_multivalley_subspace(
    sector_matrices: dict[str, np.ndarray],
    *,
    eig_tol: float = 1e-6,
    commutator_tol: float = 1e-6,
) -> MultiValleyDiagnostic:
    """Legacy diagnostic — delegates to general checks."""
    eigenvalues: dict[str, np.ndarray] = {}
    max_non_idempotent = 0.0
    for name, matrix in sector_matrices.items():
        mat = np.asarray(matrix, dtype=np.complex128)
        eigenvalues[name] = np.linalg.eigvalsh(mat)
        max_non_idempotent = max(max_non_idempotent, float(np.linalg.norm(mat @ mat - mat)))
    names = list(sector_matrices)
    max_commutator = 0.0
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            a = np.asarray(sector_matrices[name_a], dtype=np.complex128)
            b = np.asarray(sector_matrices[name_b], dtype=np.complex128)
            max_commutator = max(max_commutator, float(np.linalg.norm(a @ b - b @ a)))
    if max_non_idempotent > eig_tol:
        return MultiValleyDiagnostic(False, "non_idempotent_sector_projector", eigenvalues, max_commutator)
    if max_commutator > commutator_tol:
        return MultiValleyDiagnostic(False, "non_commuting_sector_projectors", eigenvalues, max_commutator)
    return MultiValleyDiagnostic(True, "stably_separable", eigenvalues, max_commutator)
