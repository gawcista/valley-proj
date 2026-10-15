"""Conservative extraction working-set planning; estimates are not RSS guarantees.

CPU affinity and Slurm limits cap concurrency. Available host memory and each
readable cgroup allowance cap memory, then ``memory_gb`` can lower that cap.
Unknown memory permits a 1 GiB serial fallback; parallel requests need an
explicit budget. Malformed Slurm limits fail; memory zero adds no extra cap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath
import re


_MIB = 1024**2
_GIB = 1024**3
_BASE_BYTES = 192 * _MIB


@dataclass(frozen=True)
class ExtractionPlan:
    workers: int
    memory_bytes: int
    chunk_size: int


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _slurm_int(name: str, *, allow_zero: bool = False) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if re.fullmatch(r"[0-9]+", value) is None or (int(value) == 0 and not allow_zero):
        domain = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain a {domain} integer")
    return int(value)


def _cpu_limit() -> int:
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError, NotImplementedError):
        affinity = 0
    detected = affinity or os.cpu_count() or 1
    slurm = _slurm_int("SLURM_CPUS_PER_TASK")
    return min(detected, slurm) if slurm is not None else detected


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _host_available_bytes() -> int | None:
    text = _read_text(Path("/proc/meminfo"))
    if text is not None:
        match = re.search(r"^MemAvailable:\s+(\d+)\s+kB\s*$", text, re.MULTILINE)
        if match:
            return int(match[1]) * 1024
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages >= 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _remaining_at(directory: Path, *, unified: bool) -> int | None:
    limit_name = "memory.max" if unified else "memory.limit_in_bytes"
    used_name = "memory.current" if unified else "memory.usage_in_bytes"
    limit = _read_text(directory / limit_name)
    used = _read_text(directory / used_name)
    if limit is None or used is None:
        return None
    try:
        limit_bytes, used_bytes = int(limit.strip()), int(used.strip())
    except ValueError:
        return None
    # v1 represents an unlimited hierarchy with a page-aligned LONG_MAX.
    if limit_bytes < 0 or used_bytes < 0 or limit_bytes >= 2**60:
        return None
    return max(0, limit_bytes - used_bytes)


def _read_cgroup_available_bytes() -> int | None:
    memberships = _read_text(Path("/proc/self/cgroup"))
    mounts = _read_text(Path("/proc/self/mountinfo"))
    if memberships is None or mounts is None:
        return None
    groups = []
    for line in memberships.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3:
            _, controllers, group_path = fields
            if not controllers or "memory" in controllers.split(","):
                groups.append((not controllers, PurePosixPath(group_path)))
    remaining = []
    for line in mounts.splitlines():
        before, separator, after = line.partition(" - ")
        fields, extra = before.split(), after.split()
        if not separator or len(fields) < 6 or len(extra) < 3:
            continue
        unified = extra[0] == "cgroup2"
        if not unified and not (extra[0] == "cgroup" and "memory" in extra[2].split(",")):
            continue
        root = PurePosixPath(_unescape_mount(fields[3]))
        mountpoint = Path(_unescape_mount(fields[4]))
        for group_unified, group_path in groups:
            if unified != group_unified or ".." in group_path.parts:
                continue
            try:
                relative = group_path.relative_to(root)
            except ValueError:
                continue
            directory = mountpoint / str(relative)
            # Ancestors can share a tighter allocation than the leaf cgroup.
            while True:
                value = _remaining_at(directory, unified=unified)
                if value is not None:
                    remaining.append(value)
                if directory == mountpoint:
                    break
                directory = directory.parent
    return min(remaining) if remaining else None


_cgroup_available_bytes = _read_cgroup_available_bytes


def plan_extraction(options: Mapping, *, record_length: int, max_nplane: int) -> ExtractionPlan:
    """Return CPU/memory-capped workers and a bounded G-vector chunk size.

    ``memory_gb`` uses GiB; Slurm's memory environment variables use MiB.
    Reserves include one parent and every worker's Python/numerical baseline.
    They describe estimated working sets, rather than an operating-system limit.
    """
    record_length = _positive_int(record_length, "record_length")
    max_nplane = _positive_int(max_nplane, "max_nplane")
    workers = options.get("workers", 1)
    automatic = isinstance(workers, str) and workers == "auto"
    if not automatic:
        workers = _positive_int(workers, "extract.workers")
    chunk = _positive_int(options.get("g_chunk_size", 65536), "extract.g_chunk_size")
    explicit = None
    if "memory_gb" in options:
        memory = options["memory_gb"]
        if (isinstance(memory, bool) or not isinstance(memory, Real) or memory <= 0
                or (not isinstance(memory, Integral) and not math.isfinite(memory))):
            raise ValueError("extract.memory_gb must be finite and positive")
        # Split first so a finite GiB value cannot overflow during scaling.
        explicit = int(memory) * _GIB + int((memory % 1) * _GIB)
    cpus = _cpu_limit()
    limits = [value for value in (_host_available_bytes(), _cgroup_available_bytes())
              if value is not None]
    for name, multiplier in (("SLURM_MEM_PER_NODE", _MIB), ("SLURM_MEM_PER_CPU", _MIB * cpus)):
        # Slurm memory zero grants all allocation/node memory; measured caps remain.
        value = _slurm_int(name, allow_zero=True)
        if value:
            limits.append(value * multiplier)
    if explicit is not None:
        limits.append(explicit)
    if not limits:
        if automatic or workers > 1:
            raise ValueError("Cannot detect available memory; set extract.memory_gb for parallel extraction")
        limits.append(_GIB)
    memory_bytes = min(limits)
    parent = _BASE_BYTES + 144 * max_nplane
    worker_base = _BASE_BYTES + 8 * record_length
    fitting_chunk = (memory_bytes - parent - worker_base) // 128
    if fitting_chunk < 1:
        raise ValueError("Memory budget cannot fit one extraction worker")
    chunk = min(chunk, fitting_chunk)
    memory_workers = (memory_bytes - parent) // (worker_base + 128 * chunk)
    requested = cpus if automatic else workers
    return ExtractionPlan(min(requested, cpus, memory_workers), memory_bytes, chunk)
