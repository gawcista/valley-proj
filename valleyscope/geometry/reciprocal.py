from __future__ import annotations

from itertools import product

import numpy as np

# Node budget for one complete nearest-image sphere search.  Exceeding it is
# reported as unresolved input, never as a partial minimum.
_MAX_SEARCH_NODES = 4_000_000
_BOUND_SLACK = 1e-9


def reciprocal_grid(reciprocal_cart: np.ndarray, shell: int = 2, use_2d: bool = True) -> np.ndarray:
    basis = np.asarray(reciprocal_cart, dtype=float)
    if basis.shape != (3, 3):
        raise ValueError("reciprocal_cart must have shape [3,3]")
    if shell < 0:
        raise ValueError("shell must be non-negative")
    if use_2d:
        coeffs = [(i, j, 0) for i, j in product(range(-shell, shell + 1), repeat=2)]
    else:
        coeffs = list(product(range(-shell, shell + 1), repeat=3))
    return np.asarray(coeffs, dtype=float) @ basis


def _validate_inputs(
    q_cart: np.ndarray,
    center_cart: np.ndarray,
    reciprocal_cart: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.asarray(reciprocal_cart, dtype=float)
    if basis.shape != (3, 3):
        raise ValueError("reciprocal_cart must have shape [3,3]")
    q = np.asarray(q_cart, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q_cart must have shape [nG,3]")
    center = np.asarray(center_cart, dtype=float)
    if center.shape != (3,):
        raise ValueError("center_cart must have shape [3]")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(center)) and np.all(np.isfinite(basis))):
        raise ValueError("reciprocal geometry inputs must be finite")
    return q, center, basis


def _in_plane_basis(basis: np.ndarray, use_2d: bool) -> np.ndarray:
    dim = 2 if use_2d else 3
    sub = np.array(basis[:dim, :dim], dtype=float)
    singular = np.linalg.svd(sub, compute_uv=False)
    scale = float(singular[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("reciprocal_cart is rank-deficient in the projected plane")
    if float(singular[-1]) <= np.finfo(float).eps * dim * scale:
        raise ValueError("reciprocal_cart is rank-deficient in the projected plane")
    return sub


def _babai_integers(y: np.ndarray, upper: np.ndarray, diag: np.ndarray, dim: int) -> np.ndarray:
    x = np.zeros(dim, dtype=np.int64)
    for k in range(dim - 1, -1, -1):
        tail = float(upper[k, k + 1:] @ x[k + 1:]) if k + 1 < dim else 0.0
        x[k] = int(np.rint((y[k] - tail) / diag[k]))
    return x


def _nearest_lattice_distances(deltas: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Exact min over integer n of ||delta - n @ lattice|| for each row delta.

    Complete finite sphere enumeration after QR reduction of ``lattice.T``;
    Q is orthogonal and R is upper triangular with positive diagonal, so the
    search bound is a hard proof of minimality rather than a fixed shell.
    """
    dim = lattice.shape[0]
    q_mat, r_mat = np.linalg.qr(lattice.T)
    signs = np.sign(np.diag(r_mat))
    signs[signs == 0.0] = 1.0
    q_mat = q_mat * signs[None, :]
    upper = signs[:, None] * r_mat
    diag = np.diag(upper)
    if not np.all(np.isfinite(diag)) or np.any(diag <= 0.0):
        raise ValueError("reciprocal_cart does not admit a positive-definite QR reduction")
    row_scale = float(np.max(np.linalg.norm(lattice, axis=1)))

    out = np.empty(deltas.shape[0], dtype=float)
    for row in range(deltas.shape[0]):
        delta = deltas[row]
        y = q_mat.T @ delta
        x_cur = _babai_integers(y, upper, diag, dim)
        best2 = float(np.sum((delta - x_cur @ lattice) ** 2))
        # Pad the initial upper bound outward so no lattice point at the true
        # minimum can be pruned by floating-point bound evaluation.
        radius = np.sqrt(best2)
        bound2 = (radius * (1.0 + _BOUND_SLACK) + _BOUND_SLACK * (row_scale + radius)) ** 2

        nodes = 0

        def _enumerate(k: int, partial: float) -> None:
            nonlocal best2, nodes
            nodes += 1
            if nodes > _MAX_SEARCH_NODES:
                raise RuntimeError(
                    "nearest lattice search exceeded its node budget; the reciprocal "
                    "basis is too ill-conditioned for a complete minimum-distance proof"
                )
            if k < 0:
                residual = float(np.sum((delta - x_cur @ lattice) ** 2))
                if residual < best2:
                    best2 = residual
                return
            tail = float(upper[k, k + 1:] @ x_cur[k + 1:]) if k + 1 < dim else 0.0
            center = (y[k] - tail) / diag[k]
            remaining = bound2 - partial
            if remaining < 0.0:
                return
            limit = np.sqrt(remaining) / diag[k]
            slack = _BOUND_SLACK * (1.0 + abs(center) + limit)
            low = int(np.floor(center - limit - slack))
            high = int(np.ceil(center + limit + slack))
            for value in range(low, high + 1):
                x_cur[k] = value
                _enumerate(k - 1, partial + (y[k] - tail - diag[k] * value) ** 2)

        _enumerate(dim - 1, 0.0)
        if not np.isfinite(best2):
            raise RuntimeError("nearest lattice search produced no finite candidate")
        out[row] = float(np.sqrt(best2))
    return out


def minimum_periodic_distance(
    q_cart: np.ndarray,
    center_cart: np.ndarray,
    reciprocal_cart: np.ndarray,
    shell: int = 2,
    use_2d: bool = True,
) -> np.ndarray:
    """Minimum distance from each q to the periodic images of one center.

    d(q, Q) = min_{n in Z^d} || (q - Q) - n @ B ||_2

    ``use_2d`` restricts both the metric and the basis to Cartesian xy
    (``B[:2, :2]``); otherwise the complete three-dimensional vectors and
    basis are used.  ``shell`` is retained for call-site compatibility and
    does not affect the result: the search is a complete nearest-image
    enumeration, not a fixed-shell approximation.
    """
    q, center, basis = _validate_inputs(q_cart, center_cart, reciprocal_cart)
    del shell
    dim = 2 if use_2d else 3
    lattice = _in_plane_basis(basis, use_2d)
    deltas = q[:, :dim] - center[:dim]
    # Recenter by a nearby lattice translate to limit cancellation for large
    # integer reciprocal shifts; the lattice is invariant under that shift.
    fractional = np.linalg.solve(lattice.T, deltas.T).T
    shift = np.rint(fractional)
    deltas = deltas - shift @ lattice
    return _nearest_lattice_distances(deltas, lattice)


def equivalent_mod_reciprocal(
    q_cart: np.ndarray,
    center_cart: np.ndarray,
    reciprocal_cart: np.ndarray,
    tolerance: float,
    shell: int = 2,
    use_2d: bool = True,
) -> bool:
    q = np.asarray(q_cart, dtype=float).reshape(1, 3)
    distance = minimum_periodic_distance(q, center_cart, reciprocal_cart, shell=shell, use_2d=use_2d)[0]
    return bool(distance <= tolerance)
