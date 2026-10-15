from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest


def reference_vectors(direct, kpoint, encut):
    """Scalar reference with the original WAVECAR cutoff arithmetic."""
    reciprocal = 2 * np.pi * np.linalg.inv(direct).T
    bounds = np.ceil(np.sqrt(encut / 3.80998212) * np.linalg.norm(direct, axis=1) / (2 * np.pi)).astype(int) + 1
    result, energies = [], []
    axes = [list(range(n + 1)) + list(range(-n, 0)) for n in bounds]
    for z in axes[2]:
        for y in axes[1]:
            for x in axes[0]:
                g = np.array([x, y, z])
                q = (g + kpoint) @ reciprocal
                energy = 3.80998212 * float(q @ q)
                if energy < encut:
                    result.append(g)
                    energies.append(energy)
    return np.asarray(result, dtype=int).reshape(-1, 3), np.asarray(energies)


@pytest.mark.parametrize("chunk_size", [1, 7, 65536])
@pytest.mark.parametrize("kpoint", [np.zeros(3), np.array([0.17, -0.23, 0.04])])
def test_chunked_g_vectors_preserve_scalar_cutoff_and_vasp_order(chunk_size, kpoint):
    from valleyscope.io.g_vectors import enumerate_g_vectors

    direct = np.array([[13., 0., 0.], [4., 11., 0.], [1., 2., 17.]])
    reciprocal = 2 * np.pi * np.linalg.inv(direct).T
    expected, energies = reference_vectors(direct, kpoint, 1.7)
    actual, actual_energies = enumerate_g_vectors(direct, reciprocal, kpoint, 1.7, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual_energies, energies, rtol=2e-15, atol=1e-15)


@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_cutoff_boundary_matches_original_scalar_decision(direction):
    from valleyscope.io.g_vectors import enumerate_g_vectors

    direct = np.array([[13., 0., 0.], [4., 11., 0.], [1., 2., 17.]])
    reciprocal = 2 * np.pi * np.linalg.inv(direct).T
    kpoint = np.array([0.17, -0.23, 0.04])
    q = (np.array([1., -1., 0.]) + kpoint) @ reciprocal
    encut = np.nextafter(3.80998212 * float(q @ q), direction)
    expected, _ = reference_vectors(direct, kpoint, encut)
    actual, _ = enumerate_g_vectors(direct, reciprocal, kpoint, encut, chunk_size=11)
    np.testing.assert_array_equal(actual, expected)


def test_g_chunks_use_spawn_pool_without_changing_order():
    from valleyscope.io.g_vectors import enumerate_g_vectors

    direct = np.diag([12., 13., 14.])
    reciprocal = 2 * np.pi * np.linalg.inv(direct).T
    kpoint = np.array([0.1, 0.2, -0.1])
    expected, _ = reference_vectors(direct, kpoint, 2.)
    with ProcessPoolExecutor(2, mp_context=get_context("spawn")) as pool:
        actual, _ = enumerate_g_vectors(direct, reciprocal, kpoint, 2., chunk_size=37, executor=pool, max_in_flight=2)
    np.testing.assert_array_equal(actual, expected)


def test_bounded_mapping_does_not_eagerly_submit_all_tasks():
    from valleyscope.io.g_vectors import ordered_bounded_map

    submitted = []

    def tasks():
        for index in range(100):
            submitted.append(index)
            yield index

    with ThreadPoolExecutor(2) as pool:
        results = ordered_bounded_map(pool, abs, tasks(), limit=2)
        assert next(results) == 0
        assert len(submitted) == 2
        assert list(results) == list(range(1, 100))


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_g_chunk_size_rejects_invalid_values(chunk_size):
    from valleyscope.io.g_vectors import enumerate_g_vectors

    with pytest.raises(ValueError, match="chunk_size"):
        enumerate_g_vectors(np.eye(3), np.eye(3), np.zeros(3), 1., chunk_size=chunk_size)
