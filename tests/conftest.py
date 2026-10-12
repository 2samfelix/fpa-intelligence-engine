from __future__ import annotations

import pytest

from fpa_engine.loader import Benchmark, load_benchmark
from fpa_engine.pipeline import PipelineResult, run


@pytest.fixture(scope="session")
def benchmark() -> Benchmark:
    return load_benchmark()


@pytest.fixture(scope="session")
def expected(benchmark: Benchmark) -> dict:
    return benchmark.expected


@pytest.fixture(scope="session")
def result(benchmark: Benchmark) -> PipelineResult:
    return run(benchmark)


@pytest.fixture(scope="session")
def recon(result: PipelineResult) -> dict:
    return {r.id: r for r in result.reconciliations}


@pytest.fixture(scope="session")
def packet(benchmark: Benchmark):
    from datetime import datetime, timezone

    from fpa_engine.brief import build_brief_packet

    return build_brief_packet(benchmark, generated_at=datetime(2026, 10, 12, tzinfo=timezone.utc))
