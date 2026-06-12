from pathlib import Path

import pytest

# Benchmark ingestion record anchor tests
# -----------------------------------------------------------------------

def test_benchmark_ingestion_record_docs():
    """Smoke doc and benchmark matrix must cover ingestion record anchors and state offline-only."""
    smoke_path = Path("docs/benchmarks/database_ingestion_record_smoke.md")
    assert smoke_path.exists()
    smoke = smoke_path.read_text(encoding="utf-8")
    for phrase in ["collect-database-record", "has_ready_ebr_bundles", "no_ready_ebr_bundles",
                     "P321", "P312", "tmpdir=$(mktemp -d)", "--output \"$tmpdir/"]:
        assert phrase in smoke, f"missing '{phrase}'"
    smoke_lower = smoke.lower()
    assert ("not a default" in smoke_lower or "offline" in smoke_lower) and "explicit" in smoke_lower
    assert "spinor" in smoke_lower and ("physical" in smoke_lower or "blocker" in smoke_lower)

    matrix = Path("docs/benchmarks/benchmark_matrix.md").read_text(encoding="utf-8")
    assert ("ingestion" in matrix.lower() and "collect-database-record" in matrix
            and ("offline" in matrix.lower() or "not a default" in matrix.lower()))
    assert matrix.count("## Standard Output Contract") == 1


# -----------------------------------------------------------------------
