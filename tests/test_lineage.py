from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from fpa_engine.calculator import REGISTRY, dependency_order
from fpa_engine.schemas import EvidenceType, FinancialFact, ReviewStatus, Severity, VerificationMethod
from fpa_engine.validator import validate_inputs


def test_benchmark_validates_clean(result):
    assert result.validation.ok, [f"{i.code} {i.subject_id}" for i in result.validation.errors]


def test_every_sourced_fact_resolves_to_a_source(benchmark):
    smap = benchmark.source_map
    for fact in benchmark.sourced:
        assert fact.source_ids, fact.id
        assert all(s in smap for s in fact.source_ids), fact.id
        if fact.vintage_source_id:
            assert fact.vintage_source_id in smap


def test_every_derived_fact_resolves_to_inputs_and_registered_formula(benchmark):
    facts = benchmark.fact_map
    for fact in benchmark.derived:
        assert fact.formula in REGISTRY, fact.id
        assert fact.inputs and all(i in facts for i in fact.inputs), fact.id


def test_no_dependency_cycles(benchmark):
    order, cyclic = dependency_order(benchmark.fact_map)
    assert not cyclic
    assert len(order) == 25
    position = {fid: i for i, fid in enumerate(order)}
    for fact in benchmark.derived:
        for dep in fact.inputs:
            if dep in position:
                assert position[dep] < position[fact.id]


def test_phrase_match_facts_keep_verification_method(benchmark, result):
    warned = {i.subject_id for i in result.validation.warnings if i.code == "PHRASE_MATCH_ONLY"}
    for fid in ("S23", "S24", "S25"):
        methods = {s.verification_method for s in benchmark.sources_for(fid)}
        assert VerificationMethod.edgar_fts_phrase_match in methods
        assert fid in warned
    assert warned == {"S23", "S24", "S25"}


def test_required_warning_classes_are_preserved(result):
    codes = result.validation.codes(Severity.warning)
    assert {"RESTATEMENT_UNTESTED", "PHRASE_MATCH_ONLY", "MIXED_BASIS", "SCOPE_PERIOD_INFERRED"} <= codes
    mixed = {i.subject_id for i in result.validation.warnings if i.code == "MIXED_BASIS"}
    assert {"D07", "D08", "D20", "D21"} <= mixed


def test_approximate_disclosure_is_warned(benchmark):
    approx = benchmark.fact_map["S25"].model_copy(update={"approximate": True})
    report = validate_inputs(benchmark.with_fact(approx))
    assert report.ok
    assert any(i.code == "APPROXIMATE" and i.subject_id == "S25" for i in report.warnings)


def test_explicit_quotes_are_exact_substrings_of_source_excerpts(benchmark):
    smap = benchmark.source_map
    for n in benchmark.narratives:
        if n.evidence_type is EvidenceType.explicit and n.verbatim_quote:
            excerpts = [smap[s].excerpt for s in n.source_ids if smap[s].excerpt]
            assert any(n.verbatim_quote in e for e in excerpts), n.id


def test_interpretive_evidence_declares_premises_and_assumptions(benchmark):
    narratives = {n.id: n for n in benchmark.narratives}
    for n in narratives.values():
        if n.evidence_type is EvidenceType.interpretive:
            assert n.premise_ids, n.id
    assert narratives["N03"].assumption_ids == ["AS01"] and narratives["N03"].review_status is ReviewStatus.needs_review
    assert narratives["N13"].verbatim_quote is None and narratives["N13"].review_status is ReviewStatus.needs_review


def test_validator_flags_quote_not_in_source(benchmark):
    n01 = next(n for n in benchmark.narratives if n.id == "N01")
    tampered = n01.model_copy(update={"verbatim_quote": "Higher sales volume was driven entirely by end users."})
    bm = replace(benchmark, narratives=tuple(tampered if n.id == "N01" else n for n in benchmark.narratives))
    report = validate_inputs(bm)
    assert any(i.code == "QUOTE_NOT_IN_SOURCE" and i.subject_id == "N01" for i in report.errors)


def test_validator_rejects_sourced_fact_without_source(benchmark):
    orphan = benchmark.fact_map["S01"].model_copy(update={"source_ids": []})
    report = validate_inputs(benchmark.with_fact(orphan))
    assert any(i.code == "MISSING_SOURCE" and i.subject_id == "S01" for i in report.errors)


def test_validator_rejects_unknown_formula_and_missing_input(benchmark):
    bad = benchmark.fact_map["D01"].model_copy(update={"formula": "magic_uplift"})
    report = validate_inputs(benchmark.with_fact(bad))
    assert "UNKNOWN_FORMULA" in report.codes(Severity.error)
    dangling = benchmark.fact_map["D01"].model_copy(update={"inputs": ["S01", "S99"]})
    report = validate_inputs(benchmark.with_fact(dangling))
    assert any(i.code == "MISSING_INPUT" and "S99" in i.message for i in report.errors)


def test_validator_rejects_unit_and_scope_mismatch(benchmark):
    wrong_units = benchmark.fact_map["D01"].model_copy(update={"inputs": ["S01", "S17"]})
    report = validate_inputs(benchmark.with_fact(wrong_units))
    assert "UNIT_MISMATCH" in report.codes(Severity.error)
    wrong_scope = benchmark.fact_map["D19"].model_copy(update={"inputs": ["S31", "S01"]})
    report = validate_inputs(benchmark.with_fact(wrong_scope))
    assert "SCOPE_MISMATCH" in report.codes(Severity.error)


def test_validator_rejects_circular_formula_dependencies(benchmark):
    a = FinancialFact(id="DX1", metric="m", unit="usd_m", period="p", period_type="three_months", origin="derived",
                      disclosure_status="disclosed", formula="absolute_change", inputs=["DX2", "S01"], basis="b",
                      scope="CI", price_basis="current", sign_convention="level")
    b = a.model_copy(update={"id": "DX2", "inputs": ["DX1", "S01"]})
    report = validate_inputs(benchmark.with_fact(a).with_fact(b))
    cyclic = {i.subject_id for i in report.errors if i.code == "CIRCULAR_DEPENDENCY"}
    assert cyclic == {"DX1", "DX2"}


def test_validator_rejects_duplicate_ids(benchmark):
    dup = benchmark.fact_map["S01"]
    bm = replace(benchmark, facts=benchmark.facts + (dup,))
    report = validate_inputs(bm)
    assert any(i.code == "DUPLICATE_ID" and i.subject_id == "S01" for i in report.errors)


def test_validator_rejects_incompatible_additive_bridge_inputs(benchmark):
    retail_as_component = benchmark.fact_map["S06"].model_copy(update={"price_basis": "price_neutral"})
    report = validate_inputs(benchmark.with_fact(retail_as_component))
    assert any(i.code == "BRIDGE_INPUT_INCOMPATIBLE" and i.subject_id == "R01" for i in report.errors)


def test_validator_rejects_malformed_unit(benchmark):
    broken = FinancialFact.model_construct(**{**benchmark.fact_map["S01"].__dict__, "unit": "usd_millions"})
    report = validate_inputs(benchmark.with_fact(broken))
    assert any(i.code == "MALFORMED_UNIT" for i in report.errors)


def test_derived_facts_have_no_direct_source_lineage_shortcut(result):
    for fid in (f"D{i:02d}" for i in range(1, 26)):
        fact = result.facts[fid]
        assert fact.value is not None and isinstance(fact.value, Decimal)
        assert fact.source_ids == []
