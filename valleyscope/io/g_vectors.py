"""Bounded enumeration of the WAVECAR plane-wave basis in VASP order."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Executor
from numbers import Integral
from typing import Callable, Iterable, Iterator, TypeVar

import numpy as np

_T = TypeVar("_T")
_R = TypeVar("_R")
_KINETIC = 3.80998212


def ordered_bounded_map(
    executor: Executor, function: Callable[[_T], _R], tasks: Iterable[_T], *, limit: int
) -> Iterator[_R]:
    """Yield ordered results while retaining at most ``limit`` in-flight tasks."""
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError("max_in_flight must be a positive integer")
    iterator = iter(tasks)
    pending = deque()
    try:
        for _ in range(limit):
            task = next(iterator, None)
            if task is None:
                break
            pending.append(executor.submit(function, task))
        while pending:
            future = pending.popleft()
            result = future.result()
            del future
            yield result
            del result
            task = next(iterator, None)
            if task is not None:
                pending.append(executor.submit(function, task))
    finally:
        for future in pending:
            future.cancel()


def _evaluate_chunk(task):
    start, stop, bounds, reciprocal, kpoint, encut, include_vectors = task
    dimensions = 2 * bounds + 1
    index = np.arange(start, stop, dtype=np.int64)
    vectors = np.empty((len(index), 3), dtype=np.int64)
    vectors[:, 0] = index % dimensions[0]
    vectors[:, 1] = (index // dimensions[0]) % dimensions[1]
    vectors[:, 2] = index // (dimensions[0] * dimensions[1])
    del index
    np.subtract(vectors, dimensions, out=vectors, where=vectors > bounds)
    shifted = vectors + kpoint
    # Batched vector products retain the original per-vector arithmetic rather
    # than replacing it by a large GEMM with a different reduction order.
    q = np.matmul(shifted[:, None, :], reciprocal)[:, 0, :]
    energies = _KINETIC * np.matmul(q[:, None, :], q[:, :, None])[:, 0, 0]
    # Re-evaluate cutoff-near candidates with the scalar reference expression.
    # This preserves the strict boundary decision even across BLAS builds.
    scale = max(1., encut, float(np.max(np.abs(shifted))) ** 2 * float(np.sum(np.abs(reciprocal))) ** 2)
    nearby = np.flatnonzero(np.abs(energies - encut) <= 128 * np.finfo(float).eps * scale)
    for row in nearby:
        q_scalar = shifted[row] @ reciprocal
        energies[row] = _KINETIC * float(q_scalar @ q_scalar)
    del shifted, q
    keep = energies < encut
    return (vectors[keep] if include_vectors else None), energies[keep]


def enumerate_g_vectors(
    direct: np.ndarray,
    reciprocal: np.ndarray,
    kpoint: np.ndarray,
    encut: float,
    *,
    chunk_size: int = 65536,
    executor: Executor | None = None,
    max_in_flight: int = 1,
    include_vectors: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Enumerate bounded candidate chunks; output always follows z-y-x order."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not np.isfinite(encut) or encut <= 0:
        raise ValueError("ENCUT must be finite and positive")
    bounds = np.ceil(np.sqrt(encut / _KINETIC) * np.linalg.norm(direct, axis=1) / (2 * np.pi)).astype(np.int64) + 1
    total = int(np.prod(2 * bounds + 1))
    tasks = ((start, min(start + chunk_size, total), bounds, reciprocal, kpoint, encut, include_vectors)
             for start in range(0, total, chunk_size))
    results = (map(_evaluate_chunk, tasks) if executor is None else
               ordered_bounded_map(executor, _evaluate_chunk, tasks, limit=max_in_flight))
    vector_bytes, energy_bytes = bytearray(), bytearray()
    for chunk_vectors, chunk_energies in results:
        if include_vectors and chunk_vectors.size:
            vector_bytes.extend(memoryview(chunk_vectors).cast("B"))
        energy_bytes.extend(memoryview(chunk_energies).cast("B"))
    vectors = np.frombuffer(vector_bytes, dtype=np.int64).reshape(-1, 3) if include_vectors else None
    return vectors, np.frombuffer(energy_bytes, dtype=np.float64)
