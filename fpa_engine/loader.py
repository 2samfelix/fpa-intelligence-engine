from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fpa_engine.schemas import (
    AnalyticalAssumption,
    AssertionVoice,
    EvidenceType,
    NarrativeAssertion,
    BridgeDefinition,
    DisclosureGap,
    FinancialFact,
    NarrativeEvidence,
    Origin,
    SourceReference,
)

BENCHMARK_ROOT = Path(__file__).parent / "benchmark"
DEFAULT_BENCHMARK = "cat_q2_2026"


class BenchmarkLoadError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("benchmark failed to load:\n  " + "\n  ".join(problems))


@dataclass(frozen=True)
class LookupResult:
    status: str
    fact: FinancialFact | None = None
    gap: DisclosureGap | None = None


@dataclass(frozen=True)
class Benchmark:
    name: str
    sources: tuple[SourceReference, ...]
    facts: tuple[FinancialFact, ...]
    gaps: tuple[DisclosureGap, ...] = ()
    narratives: tuple[NarrativeEvidence, ...] = ()
    bridges: tuple[BridgeDefinition, ...] = ()
    assumptions: tuple[AnalyticalAssumption, ...] = ()
    assertions: tuple[NarrativeAssertion, ...] = ()
    expected: dict[str, Any] = field(default_factory=dict)

    @property
    def source_map(self) -> dict[str, SourceReference]:
        return {s.id: s for s in self.sources}

    @property
    def fact_map(self) -> dict[str, FinancialFact]:
        return {f.id: f for f in self.facts}

    @property
    def sourced(self) -> list[FinancialFact]:
        return [f for f in self.facts if f.origin is Origin.sourced]

    @property
    def derived(self) -> list[FinancialFact]:
        return [f for f in self.facts if f.origin is Origin.derived]

    def sources_for(self, fact_id: str) -> list[SourceReference]:
        smap = self.source_map
        return [smap[s] for s in self.fact_map[fact_id].source_ids if s in smap]

    def without_facts(self, *fact_ids: str) -> Benchmark:
        return replace(self, facts=tuple(f for f in self.facts if f.id not in fact_ids))

    def with_fact(self, fact: FinancialFact) -> Benchmark:
        facts = [fact if f.id == fact.id else f for f in self.facts]
        if fact.id not in {f.id for f in self.facts}:
            facts.append(fact)
        return replace(self, facts=tuple(facts))

    def lookup(self, metric: str, scope: str, period: str) -> LookupResult:
        for f in self.facts:
            if (f.metric, f.scope, f.period) == (metric, scope, period):
                return LookupResult(status="found", fact=f)
        for g in self.gaps:
            if (g.metric, g.scope, g.period) == (metric, scope, period):
                return LookupResult(status=g.disclosure_status.value, gap=g)
        return LookupResult(status="not_in_benchmark")


def _parse_list(model: type, items: list[dict], label: str, problems: list[str]) -> list:
    parsed = []
    for i, item in enumerate(items):
        try:
            parsed.append(model.model_validate(item))
        except ValidationError as exc:
            ident = item.get("id", f"#{i}") if isinstance(item, dict) else f"#{i}"
            problems.append(f"{label} {ident}: {exc.errors()[0]['msg']}")
    return parsed


def _check_assertions(narratives: dict[str, NarrativeEvidence], assertions: list[NarrativeAssertion],
                      problems: list[str]) -> None:
    """Assertions attach to one narrative each. Source-voice wording must come from an explicit narrative, and a
    quote must be an exact excerpt of that narrative's validated verbatim quote. Every narrative needs one."""
    for a in assertions:
        n = narratives.get(a.narrative_id)
        if n is None:
            problems.append(f"assertion {a.id}: unknown narrative {a.narrative_id}")
            continue
        if a.voice is AssertionVoice.source and n.evidence_type is not EvidenceType.explicit:
            problems.append(f"assertion {a.id}: source voice needs explicit evidence; {n.id} is interpretive")
        if a.quote is not None:
            if a.voice is not AssertionVoice.source:
                problems.append(f"assertion {a.id}: only a source-voice assertion may carry a quote")
            if not n.verbatim_quote or a.quote not in n.verbatim_quote:
                problems.append(f"assertion {a.id}: quote is not an excerpt of {n.id}'s verbatim quote")
    covered = {a.narrative_id for a in assertions}
    for nid in sorted(set(narratives) - covered):
        problems.append(f"narrative {nid}: has no assertion")


def parse_benchmark(name: str, raw: dict[str, Any]) -> Benchmark:
    problems: list[str] = []
    fact_doc = raw.get("facts", {})

    sources = _parse_list(SourceReference, raw.get("sources", []), "source", problems)
    sourced = _parse_list(FinancialFact, fact_doc.get("sourced", []), "sourced fact", problems)
    derived_raw = fact_doc.get("derived", [])
    for item in derived_raw:
        if isinstance(item, dict) and item.get("value") is not None:
            problems.append(f"derived fact {item.get('id')}: stores a value; derived values must be computed")
    derived = _parse_list(FinancialFact, derived_raw, "derived fact", problems)
    gaps = _parse_list(DisclosureGap, fact_doc.get("known_gaps", []), "gap", problems)
    narratives = _parse_list(NarrativeEvidence, raw.get("narratives", []), "narrative", problems)
    bridges = _parse_list(BridgeDefinition, raw.get("bridges", []), "bridge", problems)
    assumptions = _parse_list(AnalyticalAssumption, raw.get("assumptions", []), "assumption", problems)
    assertions = _parse_list(NarrativeAssertion, raw.get("assertions", []), "assertion", problems)

    for f in sourced:
        if f.origin is not Origin.sourced:
            problems.append(f"fact {f.id}: listed under 'sourced' but origin is {f.origin.value}")
    for f in derived:
        if f.origin is not Origin.derived:
            problems.append(f"fact {f.id}: listed under 'derived' but origin is {f.origin.value}")

    all_ids = [o.id for group in (sources, sourced, derived, gaps, narratives, bridges, assumptions, assertions)
               for o in group]
    for dup, n in Counter(all_ids).items():
        if n > 1:
            problems.append(f"duplicate id {dup} ({n} occurrences)")

    source_ids = {s.id for s in sources}
    for f in [*sourced, *derived]:
        for sid in f.source_ids:
            if sid not in source_ids:
                problems.append(f"fact {f.id}: unresolved source reference {sid}")
    for g in gaps:
        for sid in g.evidence_source_ids:
            if sid not in source_ids:
                problems.append(f"gap {g.id}: unresolved source reference {sid}")
    for n in narratives:
        for sid in n.source_ids:
            if sid not in source_ids:
                problems.append(f"narrative {n.id}: unresolved source reference {sid}")
    assumption_ids = {a.id for a in assumptions}
    narrative_map = {n.id: n for n in narratives}
    for n in narratives:
        for aid in n.assumption_ids:
            if aid not in assumption_ids:
                problems.append(f"narrative {n.id}: unresolved assumption reference {aid}")
    for a in assumptions:
        introducer = narrative_map.get(a.introduced_by)
        if introducer is None or a.id not in introducer.assumption_ids:
            problems.append(f"assumption {a.id}: introduced_by {a.introduced_by} must be a narrative that lists it")
    if "assertions" in raw:
        _check_assertions(narrative_map, assertions, problems)

    if problems:
        raise BenchmarkLoadError(problems)
    return Benchmark(
        name=name,
        sources=tuple(sources),
        facts=tuple([*sourced, *derived]),
        gaps=tuple(gaps),
        narratives=tuple(narratives),
        bridges=tuple(bridges),
        assumptions=tuple(assumptions),
        assertions=tuple(assertions),
        expected=raw.get("expected", {}),
    )


def read_raw(directory: Path) -> dict[str, Any]:
    def read(name: str) -> Any:
        try:
            return json.loads((directory / name).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise BenchmarkLoadError([f"missing benchmark file {name}"]) from exc
        except json.JSONDecodeError as exc:
            raise BenchmarkLoadError([f"{name}: invalid JSON ({exc.msg} at line {exc.lineno})"]) from exc

    return {
        "sources": read("sources.json"),
        "facts": read("facts.json"),
        "narratives": read("narratives.json"),
        "bridges": read("bridges.json"),
        "assumptions": read("assumptions.json"),
        "assertions": read("assertions.json"),
        "expected": read("expected.json"),
    }


def load_benchmark(name: str = DEFAULT_BENCHMARK, root: Path = BENCHMARK_ROOT) -> Benchmark:
    return parse_benchmark(name, read_raw(root / name))
