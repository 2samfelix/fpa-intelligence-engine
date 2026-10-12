from __future__ import annotations

import copy
from decimal import Decimal

import pytest

from fpa_engine.loader import BenchmarkLoadError, load_benchmark, parse_benchmark, read_raw, BENCHMARK_ROOT, DEFAULT_BENCHMARK
from fpa_engine.schemas import DisclosureStatus, Origin

SOURCED_IDS = [f"S{i:02d}" for i in range(1, 41)]


def test_benchmark_counts(benchmark, expected):
    counts = expected["counts"]
    assert len(benchmark.sourced) == counts["sourced_facts"] == 40
    assert len(benchmark.derived) == counts["derived_facts"] == 25
    assert len(benchmark.narratives) == counts["narratives"]
    assert len(benchmark.gaps) == counts["known_gaps"]
    additive = [b for b in benchmark.bridges if b.bridge_type.value != "non_additive_set"]
    assert len(additive) == counts["additive_bridges"]
    assert len(benchmark.bridges) - len(additive) == counts["non_additive_sets"]
    assert sorted(f.id for f in benchmark.sourced) == SOURCED_IDS


@pytest.mark.parametrize("fact_id", SOURCED_IDS)
def test_sourced_fact_matches_benchmark(benchmark, expected, fact_id):
    fact = benchmark.fact_map[fact_id]
    exp = expected["sourced"][fact_id]
    assert fact.origin is Origin.sourced
    assert fact.disclosure_status is DisclosureStatus.disclosed
    assert fact.value == Decimal(exp["value"])
    assert fact.unit.value == exp["unit"]
    assert fact.period == exp["period"]
    assert fact.scope == exp["scope"]
    assert fact.precision == Decimal(exp["precision"])
    assert fact.source_ids == exp["source_ids"]
    resolved = benchmark.sources_for(fact_id)
    assert [s.id for s in resolved] == exp["source_ids"]


def test_uncertain_items_keep_their_uncertainty(benchmark):
    facts = benchmark.fact_map
    for fid in ("S14", "S16", "S18", "S20", "S36", "S37"):
        assert "restatement_untested" in facts[fid].review_flags
    assert "scope_period_inferred" in facts["S24"].review_flags
    for fid in ("S23", "S24", "S25"):
        assert "phrase_match_only" in facts[fid].review_flags
    assert facts["S18"].vintage_source_id is None
    assert facts["S20"].vintage_source_id is None


def test_derived_definitions_carry_no_stored_values(benchmark):
    assert all(f.value is None for f in benchmark.derived)


def _raw():
    return copy.deepcopy(read_raw(BENCHMARK_ROOT / DEFAULT_BENCHMARK))


def test_loader_rejects_malformed_unit():
    raw = _raw()
    raw["facts"]["sourced"][0]["unit"] = "usd_millions"
    with pytest.raises(BenchmarkLoadError, match="sourced fact S01"):
        parse_benchmark("bad", raw)


def test_loader_rejects_duplicate_fact_id():
    raw = _raw()
    raw["facts"]["sourced"][1]["id"] = "S01"
    with pytest.raises(BenchmarkLoadError, match="duplicate id S01"):
        parse_benchmark("bad", raw)


def test_loader_rejects_duplicate_source_id():
    raw = _raw()
    raw["sources"][1]["id"] = raw["sources"][0]["id"]
    with pytest.raises(BenchmarkLoadError, match="duplicate id"):
        parse_benchmark("bad", raw)


def test_loader_rejects_unresolved_source_reference():
    raw = _raw()
    raw["facts"]["sourced"][0]["source_ids"] = ["R-9Q99"]
    with pytest.raises(BenchmarkLoadError, match="unresolved source reference R-9Q99"):
        parse_benchmark("bad", raw)


def test_loader_rejects_derived_fact_with_smuggled_value():
    raw = _raw()
    raw["facts"]["derived"][0]["value"] = "2156"
    with pytest.raises(BenchmarkLoadError, match="derived values must be computed"):
        parse_benchmark("bad", raw)


def test_loader_rejects_sourced_fact_without_precision():
    raw = _raw()
    del raw["facts"]["sourced"][0]["precision"]
    with pytest.raises(BenchmarkLoadError, match="precision"):
        parse_benchmark("bad", raw)


def test_loader_rejects_unknown_fields():
    raw = _raw()
    raw["facts"]["sourced"][0]["confidence"] = 0.92
    with pytest.raises(BenchmarkLoadError):
        parse_benchmark("bad", raw)


def test_load_is_deterministic():
    a, b = load_benchmark(), load_benchmark()
    assert [f.model_dump() for f in a.facts] == [f.model_dump() for f in b.facts]
