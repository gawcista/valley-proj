from __future__ import annotations

import hashlib
from collections import OrderedDict
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

    Exact duplicate rows are compressed before the search and expanded
    afterwards, so repeated plane-wave momenta share one complete
    nearest-image proof without any rounding, snapping, or coefficient/grid
    compression.  Row order is preserved.
    """
    q, center, basis = _validate_inputs(q_cart, center_cart, reciprocal_cart)
    del shell
    dim = 2 if use_2d else 3
    lattice = _in_plane_basis(basis, use_2d)
    deltas = q[:, :dim] - center[:dim]
    # Compress exact duplicate rows before the search; np.unique groups by
    # bit-equal coordinates only, and the inverse expansion restores the
    # original row order.  No rounding or snapping is involved.
    unique_deltas, inverse = np.unique(deltas, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    # Recenter by a nearby lattice translate to limit cancellation for large
    # integer reciprocal shifts; the lattice is invariant under that shift.
    fractional = np.linalg.solve(lattice.T, unique_deltas.T).T
    shift = np.rint(fractional)
    unique_reduced = unique_deltas - shift @ lattice
    unique_distances = _nearest_lattice_distances(unique_reduced, lattice)
    return unique_distances[inverse]


class RunLocalPeriodicDistanceCache:
    """Bounded, input-bound reuse of exact nearest-image distances.

    A hit requires exact float equality of the complete q batch, center,
    reciprocal basis, and dimensional mode -- the numeric geometry itself,
    never labels, object identities, or cutoff values.  Strict threshold
    decisions stay with the caller; only the qcut-independent distance
    arrays are reused, and they are exposed read-only.
    """

    def __init__(self, max_entries: int = 64) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = int(max_entries)
        self._entries: OrderedDict[bytes, tuple] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _key(
        q: np.ndarray,
        center: np.ndarray,
        basis: np.ndarray,
        use_2d: bool,
    ) -> bytes:
        digest = hashlib.sha256()
        digest.update(str(int(use_2d)).encode("ascii"))
        for array in (q, center, basis):
            digest.update(str(array.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(array, dtype=float).tobytes())
        return digest.digest()

    def distances(
        self,
        q_cart: np.ndarray,
        center_cart: np.ndarray,
        reciprocal_cart: np.ndarray,
        *,
        use_2d: bool = True,
    ) -> np.ndarray:
        q, center, basis = _validate_inputs(
            q_cart, center_cart, reciprocal_cart
        )
        key = self._key(q, center, basis, use_2d)
        entry = self._entries.get(key)
        if entry is not None:
            cached_q, cached_center, cached_basis, cached_use_2d, distances = (
                entry
            )
            if (
                cached_use_2d == use_2d
                and cached_q.shape == q.shape
                and cached_center.shape == center.shape
                and cached_basis.shape == basis.shape
                and np.array_equal(cached_q, q)
                and np.array_equal(cached_center, center)
                and np.array_equal(cached_basis, basis)
            ):
                self._entries.move_to_end(key)
                return distances
        distances = minimum_periodic_distance(
            q, center, basis, use_2d=use_2d
        )
        distances.setflags(write=False)
        self._entries[key] = (q, center, basis, use_2d, distances)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return distances


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
