from __future__ import annotations

from collections import Counter

from fpa_engine.calculator import REGISTRY, dependency_order, expected_input_units
from fpa_engine.loader import Benchmark
from fpa_engine.schemas import (
    BridgeType,
    DisclosureStatus,
    EvidenceType,
    FinancialFact,
    Origin,
    PriceBasis,
    ReviewStatus,
    Severity,
    Unit,
    ValidationIssue,
    ValidationReport,
    VerificationMethod,
)

WARNING_FLAGS = {
    "restatement_untested": "uncertain restatement: prior-period figure not tested against later release vintages",
    "scope_period_inferred": "scope/period qualifier INFERRED; amount sourced",
    "vintage_unknown": "release vintage of this figure is not recorded",
    "attribution_largely_tariffs": "manufacturing-cost line only 'largely' reflects tariffs",
    "precision_inferred": "disclosure precision inferred from the stated significant figures",
}


def _unit_label(unit: object) -> str:
    return unit.value if isinstance(unit, Unit) else repr(unit)


def _issue(severity: Severity, code: str, subject: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, subject_id=subject, message=message)


def _check_ids(bm: Benchmark, issues: list[ValidationIssue]) -> None:
    ids = [o.id for group in (bm.sources, bm.facts, bm.gaps, bm.narratives, bm.bridges) for o in group]
    for dup, n in Counter(ids).items():
        if n > 1:
            issues.append(_issue(Severity.error, "DUPLICATE_ID", dup, f"id appears {n} times"))


def _check_sourced(fact: FinancialFact, bm: Benchmark, issues: list[ValidationIssue]) -> None:
    smap = bm.source_map
    if fact.disclosure_status is not DisclosureStatus.disclosed:
        code = "BACK_SOLVED_AS_SOURCED" if fact.disclosure_status is DisclosureStatus.back_solved else "UNDISCLOSED_AS_SOURCED"
        issues.append(_issue(Severity.error, code, fact.id,
                             f"marked sourced but disclosure_status is {fact.disclosure_status.value}"))
    valid = [s for s in fact.source_ids if s in smap]
    if not valid:
        issues.append(_issue(Severity.error, "MISSING_SOURCE", fact.id, "sourced fact has no valid source reference"))
    for sid in fact.source_ids:
        if sid not in smap:
            issues.append(_issue(Severity.error, "UNKNOWN_SOURCE", fact.id, f"source {sid} does not exist"))
    if fact.vintage_source_id and fact.vintage_source_id not in smap:
        issues.append(_issue(Severity.error, "UNKNOWN_VINTAGE_SOURCE", fact.id,
                             f"vintage source {fact.vintage_source_id} does not exist"))
    if any(smap[s].verification_method is VerificationMethod.edgar_fts_phrase_match for s in valid):
        issues.append(_issue(Severity.warning, "PHRASE_MATCH_ONLY", fact.id,
                             "confirmed by EDGAR full-text phrase match; surrounding sentence not read"))
    if valid and all(smap[s].verification_method is VerificationMethod.transcript for s in valid):
        issues.append(_issue(Severity.warning, "TRANSCRIPT_ONLY", fact.id,
                             "sourced only from a third-party call transcript, not a filing"))


def _check_derived(fact: FinancialFact, facts: dict[str, FinancialFact], issues: list[ValidationIssue]) -> None:
    formula = REGISTRY.get(fact.formula or "")
    if formula is None:
        issues.append(_issue(Severity.error, "UNKNOWN_FORMULA", fact.id, f"formula '{fact.formula}' is not registered"))
        return
    missing = [i for i in fact.inputs if i not in facts]
    for i in missing:
        issues.append(_issue(Severity.error, "MISSING_INPUT", fact.id, f"input {i} does not exist"))
    if missing:
        return
    operands = [facts[i] for i in fact.inputs]
    units = expected_input_units(formula, len(operands))
    if units is None:
        issues.append(_issue(Severity.error, "ARITY_MISMATCH", fact.id,
                             f"{formula.name} cannot take {len(operands)} inputs"))
        return
    for op, unit in zip(operands, units):
        if op.unit is not unit:
            issues.append(_issue(Severity.error, "UNIT_MISMATCH", fact.id,
                                 f"input {op.id} is {_unit_label(op.unit)}; {formula.name} expects {unit.value}"))
    if fact.unit is not formula.output_unit:
        issues.append(_issue(Severity.error, "UNIT_MISMATCH", fact.id,
                             f"declared unit {_unit_label(fact.unit)}; {formula.name} returns {formula.output_unit.value}"))
    if formula.same_scope and len({op.scope for op in operands}) > 1:
        issues.append(_issue(Severity.error, "SCOPE_MISMATCH", fact.id,
                             f"{formula.name} requires one scope; got {sorted({op.scope for op in operands})}"))
    if formula.same_metric and len({op.metric for op in operands}) > 1:
        issues.append(_issue(Severity.error, "METRIC_MISMATCH", fact.id,
                             f"{formula.name} requires one metric; got {sorted({op.metric for op in operands})}"))
    if fact.disclosure_status is DisclosureStatus.back_solved:
        issues.append(_issue(Severity.warning, "BACK_SOLVED", fact.id,
                             "back-solved value: any bridge using it closes by construction"))


def _check_warnings(fact: FinancialFact, issues: list[ValidationIssue]) -> None:
    if fact.approximate:
        issues.append(_issue(Severity.warning, "APPROXIMATE", fact.id, "approximate disclosure ('about')"))
    if fact.price_basis is PriceBasis.mixed or "mixed_basis" in fact.review_flags:
        issues.append(_issue(Severity.warning, "MIXED_BASIS", fact.id,
                             "combines inputs on different price/scope bases; directional only"))
    for flag in fact.review_flags:
        if flag in WARNING_FLAGS:
            issues.append(_issue(Severity.warning, flag.upper(), fact.id, WARNING_FLAGS[flag]))


def _check_bridges(bm: Benchmark, facts: dict[str, FinancialFact], issues: list[ValidationIssue]) -> None:
    for bridge in bm.bridges:
        if bridge.bridge_type is BridgeType.non_additive_set:
            refs = list(bridge.members)
        else:
            refs = [bridge.starting_fact, bridge.ending_fact, *bridge.components.values()]
        for ref in refs:
            if ref not in facts:
                issues.append(_issue(Severity.error, "BRIDGE_INPUT_MISSING", bridge.id, f"input {ref} does not exist"))
        if bridge.bridge_type is BridgeType.non_additive_set or any(r not in facts for r in refs):
            continue
        inputs = [facts[r] for r in refs]
        for dim in ("unit", "scope", "price_basis"):
            values = {getattr(f, dim) for f in inputs}
            if len(values) > 1:
                issues.append(_issue(Severity.error, "BRIDGE_INPUT_INCOMPATIBLE", bridge.id,
                                     f"additive bridge inputs differ in {dim}: {sorted(str(v) for v in values)}"))
        if any(f.disclosure_status is DisclosureStatus.back_solved for f in inputs):
            issues.append(_issue(Severity.warning, "BRIDGE_USES_BACK_SOLVED", bridge.id,
                                 "a back-solved input makes closure circular; cannot be RECONCILED"))


def _check_narratives(bm: Benchmark, facts: dict[str, FinancialFact], issues: list[ValidationIssue]) -> None:
    smap = bm.source_map
    narrative_ids = {n.id for n in bm.narratives}
    known = set(facts) | narrative_ids
    for n in bm.narratives:
        for sid in n.source_ids:
            if sid not in smap:
                issues.append(_issue(Severity.error, "UNKNOWN_SOURCE", n.id, f"source {sid} does not exist"))
        for ref in [*n.related_fact_ids, *n.premise_ids, *n.counter_evidence_ids]:
            if ref not in known:
                issues.append(_issue(Severity.error, "NARRATIVE_DANGLING_REF", n.id, f"reference {ref} does not exist"))
        if n.evidence_type is EvidenceType.explicit and n.verbatim_quote:
            excerpts = [smap[s].excerpt for s in n.source_ids if s in smap and smap[s].excerpt]
            if not excerpts:
                issues.append(_issue(Severity.warning, "QUOTE_UNVERIFIABLE", n.id,
                                     "no cited source carries an excerpt to check the quote against"))
            elif not any(n.verbatim_quote in e for e in excerpts):
                issues.append(_issue(Severity.error, "QUOTE_NOT_IN_SOURCE", n.id,
                                     "explicit quote is not an exact substring of any cited source excerpt"))
        if n.evidence_type is EvidenceType.explicit and n.assumption_ids:
            issues.append(_issue(Severity.error, "EXPLICIT_WITH_ASSUMPTIONS", n.id,
                                 "explicit evidence cannot rest on assumptions; label it interpretive"))
        if n.review_status is ReviewStatus.needs_review:
            issues.append(_issue(Severity.warning, "NARRATIVE_NEEDS_REVIEW", n.id, "narrative evidence needs review"))


def validate_inputs(bm: Benchmark) -> ValidationReport:
    issues: list[ValidationIssue] = []
    _check_ids(bm, issues)
    facts = bm.fact_map
    for fact in bm.facts:
        if not isinstance(fact.unit, Unit):
            issues.append(_issue(Severity.error, "MALFORMED_UNIT", fact.id, f"unit {fact.unit!r} is not recognised"))
        if fact.origin is Origin.sourced:
            _check_sourced(fact, bm, issues)
        else:
            _check_derived(fact, facts, issues)
        _check_warnings(fact, issues)
    _, cyclic = dependency_order(facts)
    for fid in sorted(cyclic):
        issues.append(_issue(Severity.error, "CIRCULAR_DEPENDENCY", fid, "derived fact is part of a formula cycle"))
    for gap in bm.gaps:
        for sid in gap.evidence_source_ids:
            if sid not in bm.source_map:
                issues.append(_issue(Severity.error, "UNKNOWN_SOURCE", gap.id, f"source {sid} does not exist"))
    _check_bridges(bm, facts, issues)
    _check_narratives(bm, facts, issues)
    return ValidationReport(issues=issues)
