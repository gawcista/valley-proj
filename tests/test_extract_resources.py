"""Resource decisions use synthetic OS probes, never host hardware sizes."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def raw_resources():
    return importlib.import_module("valleyscope.workflows.extract_resources")


@pytest.fixture
def resources(monkeypatch):
    module = importlib.import_module("valleyscope.workflows.extract_resources")
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: set(range(8)))
    monkeypatch.setattr(module.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(module, "_host_available_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(module, "_cgroup_available_bytes", lambda: None)
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        monkeypatch.delenv(name, raising=False)
    return module


def test_default_is_serial_with_requested_chunk(resources):
    plan = resources.plan_extraction({}, record_length=256, max_nplane=10)
    assert (plan.workers, plan.memory_bytes, plan.chunk_size) == (1, 8589934592, 65536)
    with pytest.raises(AttributeError):
        plan.workers = 2


@pytest.mark.parametrize("key,value", [
    ("workers", True), ("workers", 0), ("workers", -2), ("workers", 1.5),
    ("workers", "2"), ("workers", "AUTO"), ("workers", None),
    ("g_chunk_size", False), ("g_chunk_size", 0), ("g_chunk_size", 1.5),
    ("g_chunk_size", "64"), ("memory_gb", True), ("memory_gb", 0),
    ("memory_gb", -1), ("memory_gb", float("nan")),
    ("memory_gb", float("inf")), ("memory_gb", "1"), ("memory_gb", None),
])
def test_rejects_invalid_options(resources, key, value):
    with pytest.raises(ValueError, match=key):
        resources.plan_extraction({key: value}, record_length=256, max_nplane=10)


@pytest.mark.parametrize("sizes", [{"record_length": 0, "max_nplane": 1},
                                  {"record_length": True, "max_nplane": 1},
                                  {"record_length": 256, "max_nplane": -1}])
def test_rejects_invalid_wavecar_sizes(resources, sizes):
    with pytest.raises(ValueError):
        resources.plan_extraction({}, **sizes)


def test_affinity_and_slurm_cap_workers(resources, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert resources.plan_extraction({"workers": "auto"}, record_length=256,
                                     max_nplane=10).workers == 3
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "20")
    assert resources.plan_extraction({"workers": 12}, record_length=256,
                                     max_nplane=10).workers == 8


def test_cpu_count_fallback_when_affinity_unavailable(resources, monkeypatch):
    def unavailable(pid):
        raise OSError("no affinity")
    monkeypatch.setattr(resources.os, "sched_getaffinity", unavailable)
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 2)
    assert resources.plan_extraction({"workers": "auto"}, record_length=256,
                                     max_nplane=10).workers == 2


@pytest.mark.parametrize("name,value", [("SLURM_CPUS_PER_TASK", "0"),
                                       ("SLURM_CPUS_PER_TASK", "2(x2)"),
                                       ("SLURM_MEM_PER_NODE", "bad"),
                                       ("SLURM_MEM_PER_CPU", "-1"),
                                       ("SLURM_MEM_PER_NODE", "")])
def test_malformed_slurm_limits_fail(resources, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        resources.plan_extraction({}, record_length=256, max_nplane=10)


def test_explicit_memory_only_caps_detected_remaining_memory(resources, monkeypatch):
    monkeypatch.setattr(resources, "_cgroup_available_bytes", lambda: 512 * 1024**2)
    plan = resources.plan_extraction({"workers": 8, "memory_gb": 4},
                                     record_length=256, max_nplane=10)
    assert (plan.workers, plan.memory_bytes) == (1, 536870912)
    plan = resources.plan_extraction({"memory_gb": 0.5}, record_length=256, max_nplane=10)
    assert plan.memory_bytes == 536870912


@pytest.mark.parametrize("name,value,expected", [("SLURM_MEM_PER_NODE", "512", 536870912),
                                                ("SLURM_MEM_PER_CPU", "128", 1073741824)])
def test_slurm_memory_limits_are_mib(resources, monkeypatch, name, value, expected):
    monkeypatch.setenv(name, value)
    assert resources.plan_extraction({}, record_length=256, max_nplane=10).memory_bytes == expected


@pytest.mark.parametrize("name", ["SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"])
def test_slurm_zero_memory_keeps_detected_host_and_cgroup_caps(resources, monkeypatch, name):
    monkeypatch.setenv(name, "0")
    assert resources.plan_extraction({}, record_length=256, max_nplane=1).memory_bytes == 8589934592


def test_exact_worker_memory_boundary(resources, monkeypatch):
    # Parent=201326736, worker=201328768 for 1 PW, RECL=256 and chunk=1.
    monkeypatch.setattr(resources, "_host_available_bytes", lambda: 603984272)
    assert resources.plan_extraction({"workers": 8, "g_chunk_size": 1},
                                     record_length=256, max_nplane=1).workers == 2
    monkeypatch.setattr(resources, "_host_available_bytes", lambda: 603984271)
    assert resources.plan_extraction({"workers": 8, "g_chunk_size": 1},
                                     record_length=256, max_nplane=1).workers == 1


def test_chunk_shrinks_to_fit_one_worker_and_rejects_one_byte_less(resources, monkeypatch):
    monkeypatch.setattr(resources, "_host_available_bytes", lambda: 402655504)
    plan = resources.plan_extraction({"g_chunk_size": 100}, record_length=256, max_nplane=1)
    assert (plan.workers, plan.chunk_size) == (1, 1)
    monkeypatch.setattr(resources, "_host_available_bytes", lambda: 402655503)
    with pytest.raises(ValueError, match="one extraction worker"):
        resources.plan_extraction({}, record_length=256, max_nplane=1)


def test_unknown_memory_requires_explicit_budget_for_parallel(resources, monkeypatch):
    monkeypatch.setattr(resources, "_host_available_bytes", lambda: None)
    assert resources.plan_extraction({}, record_length=256, max_nplane=1).workers == 1
    for workers in (2, "auto"):
        with pytest.raises(ValueError, match="memory_gb"):
            resources.plan_extraction({"workers": workers}, record_length=256, max_nplane=1)
    plan = resources.plan_extraction({"workers": 2, "memory_gb": 1},
                                     record_length=256, max_nplane=1)
    assert (plan.workers, plan.memory_bytes) == (2, 1073741824)


def _files(monkeypatch, contents):
    def read_text(path, *args, **kwargs):
        try:
            return contents[str(path)]
        except KeyError:
            raise FileNotFoundError(str(path)) from None
    monkeypatch.setattr(Path, "read_text", read_text)


def test_cgroup_v2_resolves_nested_mount_and_parent_remaining(resources, monkeypatch):
    _files(monkeypatch, {
        "/proc/self/cgroup": "0::/tenant/job/step\n",
        "/proc/self/mountinfo": "29 23 0:26 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        "/sys/fs/cgroup/job/step/memory.max": "1073741824\n",
        "/sys/fs/cgroup/job/step/memory.current": "134217728\n",
        "/sys/fs/cgroup/job/memory.max": "2147483648\n",
        "/sys/fs/cgroup/job/memory.current": "1610612736\n",
        "/sys/fs/cgroup/memory.max": "max\n",
        "/sys/fs/cgroup/memory.current": "3000000000\n",
    })
    # Undo only the environmental-probe stub; exercise real membership resolution.
    monkeypatch.setattr(resources, "_cgroup_available_bytes", resources._read_cgroup_available_bytes)
    assert resources.plan_extraction({}, record_length=256, max_nplane=1).memory_bytes == 536870912


def test_cgroup_v1_memory_mount_and_usage(resources, monkeypatch):
    _files(monkeypatch, {
        "/proc/self/cgroup": "2:cpu,cpuacct:/work\n5:memory:/batch/step\n",
        "/proc/self/mountinfo": "31 23 0:28 /batch /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n",
        "/sys/fs/cgroup/memory/step/memory.limit_in_bytes": "1073741824\n",
        "/sys/fs/cgroup/memory/step/memory.usage_in_bytes": "402653184\n",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "1000000000\n",
    })
    monkeypatch.setattr(resources, "_cgroup_available_bytes", resources._read_cgroup_available_bytes)
    assert resources.plan_extraction({}, record_length=256, max_nplane=1).memory_bytes == 671088640


def test_exhausted_cgroup_is_not_ignored(resources, monkeypatch):
    monkeypatch.setattr(resources, "_cgroup_available_bytes", lambda: 0)
    with pytest.raises(ValueError, match="one extraction worker"):
        resources.plan_extraction({}, record_length=256, max_nplane=1)


def test_large_finite_memory_budget_is_capped_without_overflow(resources):
    assert resources.plan_extraction({"memory_gb": 1e300}, record_length=256,
                                     max_nplane=1).memory_bytes == 8589934592
    assert resources.plan_extraction({"memory_gb": 10**500}, record_length=256,
                                     max_nplane=1).memory_bytes == 8589934592


def test_mountinfo_escapes_and_unreadable_cgroup_files(resources, monkeypatch):
    _files(monkeypatch, {
        "/proc/self/cgroup": "0::/tenant/job\n",
        "/proc/self/mountinfo": "29 23 0:26 /tenant /sys/fs/my\\040cgroup rw - cgroup2 cgroup rw\n",
        "/sys/fs/my cgroup/job/memory.max": "1073741824\n",
        "/sys/fs/my cgroup/job/memory.current": "402653184\n",
        "/sys/fs/my cgroup/memory.max": "500000000\n",
        # No readable ancestor usage: do not invent a remaining allowance.
    })
    monkeypatch.setattr(resources, "_cgroup_available_bytes", resources._read_cgroup_available_bytes)
    assert resources.plan_extraction({}, record_length=256, max_nplane=1).memory_bytes == 671088640


def test_host_memavailable_is_used_as_remaining_memory(raw_resources, monkeypatch):
    _files(monkeypatch, {"/proc/meminfo": "MemTotal: 10000000 kB\nMemAvailable: 786432 kB\n"})
    assert raw_resources._host_available_bytes() == 805306368


def test_host_sysconf_fallback_and_unavailable_probes(raw_resources, monkeypatch):
    _files(monkeypatch, {})
    monkeypatch.setattr(raw_resources.os, "sysconf", lambda name: {
        "SC_AVPHYS_PAGES": 131072, "SC_PAGE_SIZE": 4096}[name])
    assert raw_resources._host_available_bytes() == 536870912
    def unavailable(name):
        raise ValueError(name)
    monkeypatch.setattr(raw_resources.os, "sysconf", unavailable)
    assert raw_resources._host_available_bytes() is None


def test_missing_cgroup_probe_does_not_invent_allowance(raw_resources, monkeypatch):
    _files(monkeypatch, {})
    assert raw_resources._cgroup_available_bytes() is None
