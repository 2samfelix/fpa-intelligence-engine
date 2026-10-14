"""Brief packet (the only data a future synthesis model may see) and the brief output contract."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal, localcontext
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from fpa_engine.calculator import DECIMAL_PRECISION, REGISTRY
from fpa_engine.loader import BENCHMARK_ROOT, Benchmark, load_benchmark
from fpa_engine.pipeline import ValidationFailed, run
from fpa_engine.schemas import (
    ID_PATTERN,
    AssumptionSourcing,
    AssumptionUse,
    BridgeType,
    CalcStatus,
    EvidenceType,
    FinancialFact,
    Origin,
    PeriodType,
    PriceBasis,
    ReconciliationStatus,
    ReviewStatus,
    SourceKind,
    StrictModel,
    Unit,
    VerificationMethod,
)
from fpa_engine.tolerance import half_unit

PACKET_VERSION = 2
DERIVED_DISPLAY_DECIMALS = {Unit.usd_m: 0, Unit.pct: 1, Unit.pts: 1, Unit.bps: 0}
APPROX_PREFIX = "~"
BASIS_FLAGS = {
    "mixed_basis_identity_test": "non_comparable",
    "directional_indicator": "directional",
    "upper_bound_ex_tariff": "upper_bound",
}


class PacketBuildError(Exception):
    pass


# ---------------------------------------------------------------- packet ---

class PacketSource(StrictModel):
    id: str
    source_kind: SourceKind
    publisher: str
    document: str
    filing_type: str | None = None
    accession_number: str | None = None
    url: str
    verification_method: VerificationMethod


class PacketFact(StrictModel):
    id: str
    metric: str
    period: str
    period_type: PeriodType
    scope: str
    unit: Unit
    origin: Origin
    value: Decimal
    display: str
    display_wording: str
    display_step: Decimal
    approximate: bool
    precision: Decimal | None = None
    max_decimals: int
    uncertainty: Decimal | None = None
    price_basis: PriceBasis
    basis: str
    flags: list[str] = Field(default_factory=list)
    requires_limitation: bool = False
    source_ids: list[str] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)
    comparability_note_ids: list[str] = Field(default_factory=list)


class PacketNarrative(StrictModel):
    id: str
    topic: str
    claim: str
    evidence_type: EvidenceType
    verbatim_quote: str | None = None
    speaker: str | None = None
    source_kind: SourceKind
    scope: str
    period: str
    source_ids: list[str] = Field(default_factory=list)
    related_fact_ids: list[str] = Field(default_factory=list)
    premise_ids: list[str] = Field(default_factory=list)
    assumption_ids: list[str] = Field(default_factory=list)
    inherited_assumption_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus
    requires_limitation: bool = False


class PacketAssumption(StrictModel):
    """An analytical premise no cited source states. Its sourcing status is structural, never prose that a
    provenance filter could remove."""
    id: str
    text: str
    sourcing: AssumptionSourcing
    introduced_by: str
    allowed_use: AssumptionUse
    must_disclose: bool


class PacketReconciliation(StrictModel):
    id: str
    name: str
    period: str
    scope: str
    bridge_type: BridgeType
    status: ReconciliationStatus
    starting_value: Decimal | None = None
    ending_value: Decimal | None = None
    reported_change: Decimal | None = None
    identified_change: Decimal | None = None
    residual: Decimal | None = None
    allowed: Decimal | None = None
    unquantified_components: list[str] = Field(default_factory=list)
    circular: bool = False
    precision_limited: bool = False
    reason: str
    input_fact_ids: list[str] = Field(default_factory=list)
    related_fact_ids: list[str] = Field(default_factory=list)
    requires_limitation: bool = False


class ComparabilityNote(StrictModel):
    id: str
    subject_ids: list[str]
    text: str


class OpenFlagKind(str, Enum):
    disclosure_gap = "disclosure_gap"
    review_required_narrative = "review_required_narrative"
    unresolved_reconciliation = "unresolved_reconciliation"
    unavailable_fact = "unavailable_fact"


class OpenFlag(StrictModel):
    id: str
    kind: OpenFlagKind
    subject_ids: list[str] = Field(default_factory=list)
    related_ids: list[str] = Field(default_factory=list)
    message: str


class ForbiddenValue(StrictModel):
    id: str
    unit: Unit
    value: Decimal | None = None
    low: Decimal | None = None
    high: Decimal | None = None
    reason: str

    @model_validator(mode="after")
    def _single_or_range(self) -> ForbiddenValue:
        if (self.value is None) == (self.low is None or self.high is None):
            raise ValueError(f"{self.id}: give either a value or a low/high range")
        return self


class SurfacingPolicy(StrictModel):
    """Which limitations a brief must carry. Mandatory items are surfaced in every brief; conditional narratives
    need a limitation only when the brief relies on them (directly or through a premise chain)."""
    mandatory_reconciliation_ids: list[str]
    mandatory_disclosure_gap_ids: list[str]
    mandatory_comparability_note_ids: list[str]
    conditional_narrative_ids: list[str]


class DisplayPolicy(StrictModel):
    """Words or symbols that may introduce an approximate figure (\"~24 pts\", \"about 24 pts\")."""
    approximation_markers: list[str] = Field(min_length=1)


class InferencePolicy(StrictModel):
    """Phrases that visibly frame a statement as the analyst's inference; a statement resting on an unsourced
    assumption must use one in its own text."""
    assumption_inference_markers: list[str] = Field(min_length=1)


class InternalTerm(StrictModel):
    concept: str
    pattern: str


class ProsePolicy(StrictModel):
    """Engine and audit vocabulary that must never reach executive prose. Patterns are case-insensitive regexes,
    written as phrases or contexts rather than bare words where the word also has a financial meaning."""
    internal_language: list[InternalTerm] = Field(min_length=1)


class StatementBounds(StrictModel):
    min: int = Field(ge=1)
    max: int = Field(ge=1)


class BriefPacket(StrictModel):
    packet_version: Literal[2] = PACKET_VERSION
    analysis_id: str
    company: str
    period: str
    scope: str
    generated_at: datetime
    packet_hash: str
    material_flags: list[str]
    surfacing_policy: SurfacingPolicy
    statement_bounds: StatementBounds
    display_policy: DisplayPolicy
    inference_policy: InferencePolicy
    prose_policy: ProsePolicy
    sources: list[PacketSource]
    facts: list[PacketFact]
    narratives: list[PacketNarrative]
    assumptions: list[PacketAssumption]
    reconciliations: list[PacketReconciliation]
    comparability_notes: list[ComparabilityNote]
    open_flags: list[OpenFlag]
    forbidden_values: list[ForbiddenValue]

    @property
    def fact_map(self) -> dict[str, PacketFact]:
        return {f.id: f for f in self.facts}

    @property
    def narrative_map(self) -> dict[str, PacketNarrative]:
        return {n.id: n for n in self.narratives}

    @property
    def assumption_map(self) -> dict[str, PacketAssumption]:
        return {a.id: a for a in self.assumptions}

    @property
    def reconciliation_map(self) -> dict[str, PacketReconciliation]:
        return {r.id: r for r in self.reconciliations}

    @property
    def flag_map(self) -> dict[str, OpenFlag]:
        return {f.id: f for f in self.open_flags}


# ---------------------------------------------------------- brief output ---

class StatementKind(str, Enum):
    fact = "fact"
    explicit_claim = "explicit_claim"
    interpretive = "interpretive"
    limitation = "limitation"


class StatementSeverity(str, Enum):
    info = "info"
    warning = "warning"
    critical = "critical"


class BriefStatement(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    text: str = Field(min_length=1)
    kind: StatementKind
    fact_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    reconciliation_ids: list[str] = Field(default_factory=list)
    premise_ids: list[str] = Field(default_factory=list)
    assumption_ids: list[str] = Field(default_factory=list)
    flag_ids: list[str] = Field(default_factory=list)
    comparability_note_ids: list[str] = Field(default_factory=list)
    severity: StatementSeverity = StatementSeverity.info


class BriefMetadata(StrictModel):
    analysis_id: str
    packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    author: str


class ExecutiveBrief(StrictModel):
    title: str = Field(min_length=1)
    statements: list[BriefStatement] = Field(min_length=1)
    validation_metadata: BriefMetadata


_LEGAL_SUFFIX = re.compile(r",?\s+(?:Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|Limited|plc|LLC|N\.V|S\.A|AG)\.?$")


def brief_title(packet: BriefPacket) -> str:
    """The only title a brief may carry: built from packet metadata, never from model output."""
    company = _LEGAL_SUFFIX.sub("", packet.company.strip())
    return f"{company} {packet.period} — {packet.scope} Executive Brief"


# --------------------------------------------------------------- builder ---

def load_policy(name: str, root: Path = BENCHMARK_ROOT) -> dict[str, Any]:
    path = root / name / "brief_policy.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PacketBuildError(f"missing brief policy: {path}") from exc


_WORKBOOK_REF = re.compile(r"[A-Z][A-Za-z ]*![A-Z]+\d+|\bworkbook\b", re.I)


def _strip_workbook_refs(text: str) -> str:
    """Drop sentences that point at the workbook; the model must not see workbook provenance notes."""
    sentences = re.split(r"(?<=\.)\s+", text.strip())
    return " ".join(s for s in sentences if not _WORKBOOK_REF.search(s))


def _relabel(text: str, terminology: dict[str, str]) -> str:
    for internal, neutral in terminology.items():
        text = text.replace(internal, neutral)
    return text


_INTERNAL_TERMS = re.compile(r"workbook", re.I)


def _surfacing_policy(policy: dict[str, Any], recs: list[PacketReconciliation], flags: list[OpenFlag],
                      notes: list[ComparabilityNote], narratives: list[PacketNarrative]) -> SurfacingPolicy:
    sp = SurfacingPolicy.model_validate(policy["surfacing_policy"])
    rec_status = {r.id: r.status for r in recs}
    gap_ids = {f.id for f in flags if f.kind is OpenFlagKind.disclosure_gap}
    problems = [f"{r} is not an UNRESOLVED reconciliation" for r in sp.mandatory_reconciliation_ids
                if rec_status.get(r) is not ReconciliationStatus.UNRESOLVED]
    problems += [f"{g} is not a disclosure gap" for g in sp.mandatory_disclosure_gap_ids if g not in gap_ids]
    problems += [f"{n} is not a comparability note in the packet" for n in sp.mandatory_comparability_note_ids
                 if n not in {c.id for c in notes}]
    needs_limitation = {n.id for n in narratives if n.requires_limitation}
    if set(sp.conditional_narrative_ids) != needs_limitation:
        problems.append(f"conditional_narrative_ids must be exactly the review-required or assumption-based "
                        f"narratives {sorted(needs_limitation)}")
    if problems:
        raise PacketBuildError("SURFACING_POLICY_INVALID: " + "; ".join(problems))
    return sp


def _decimals(precision: Decimal) -> int:
    return max(0, -precision.normalize().as_tuple().exponent)


def round_to_step(value: Decimal, step: Decimal) -> Decimal:
    """value rounded to the nearest multiple of step (half-even, like Decimal.quantize)."""
    return ((value / step).quantize(Decimal(1)) * step).quantize(Decimal(1).scaleb(-_decimals(step)))


def format_display(value: Decimal, unit: Unit, step: Decimal, approximate: bool = False) -> str:
    """The canonical public rendering of a value at its display step; approximate values carry a "~"."""
    q = round_to_step(value, step)
    decimals = _decimals(step)
    prefix = APPROX_PREFIX if approximate else ""
    if unit is Unit.usd_m:
        return f"{prefix}{'-' if q < 0 else ''}${abs(q):,.{decimals}f}M"
    suffix = {Unit.pct: "%", Unit.pts: " pts", Unit.bps: " bps"}[unit]
    return f"{prefix}{q:,.{decimals}f}{suffix}"


def uncertainty_step(uncertainty: Decimal) -> Decimal:
    """The smallest power of ten not below the uncertainty half-width: 0.5 -> 1, 5 -> 10, 10 -> 10."""
    exponent = uncertainty.adjusted()
    if uncertainty > Decimal(10) ** exponent:
        exponent += 1
    return Decimal(10) ** exponent


def _two_significant_up(value: Decimal) -> Decimal:
    return value.quantize(Decimal(1).scaleb(value.adjusted() - 1), rounding=ROUND_CEILING)


def propagate_uncertainty(fid: str, facts: dict[str, FinancialFact], calcs: dict[str, Any],
                          cache: dict[str, Decimal | None]) -> Decimal | None:
    """Worst-case half-width of a derived value from the rounding of its inputs. Uses the calculator's linear
    uncertainty where it has one; otherwise evaluates the registered formula at every corner of the input
    intervals (each formula is monotone in each input, so the corners bound it)."""
    if fid in cache:
        return cache[fid]
    fact = facts[fid]
    if fact.origin is Origin.sourced:
        cache[fid] = half_unit(fact.precision) if fact.precision is not None else None
        return cache[fid]
    if calcs[fid].uncertainty is not None:
        cache[fid] = calcs[fid].uncertainty
        return cache[fid]
    formula = REGISTRY.get(fact.formula or "")
    widths = [propagate_uncertainty(i, facts, calcs, cache) for i in fact.inputs]
    if formula is None or any(w is None for w in widths):
        cache[fid] = None
        return None
    values = [facts[i].value for i in fact.inputs]
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        corners = [formula.func(*(v + s * w for v, s, w in zip(values, signs, widths)))
                   for signs in product((-1, 1), repeat=len(values))]
        spread = max(abs(c - fact.value) for c in corners)
    cache[fid] = _two_significant_up(spread) if spread else Decimal(0)
    return cache[fid]


def _packet_assumptions(bm: Benchmark, narratives: list[PacketNarrative]) -> list[PacketAssumption]:
    """Every assumption a packet narrative uses, with its analytical text and sourcing status. Provenance stays
    behind; text that points at the workbook blocks the build instead of being silently trimmed."""
    used = {a for n in narratives for a in n.assumption_ids}
    out = []
    for a in sorted(bm.assumptions, key=lambda x: x.id):
        if a.id not in used:
            continue
        if _WORKBOOK_REF.search(a.text):
            raise PacketBuildError(f"ASSUMPTION_TEXT_INVALID: {a.id} text cites the workbook; move that to provenance")
        out.append(PacketAssumption(id=a.id, text=a.text, sourcing=a.sourcing, introduced_by=a.introduced_by,
                                    allowed_use=a.allowed_use, must_disclose=a.must_disclose))
    return out


def _prose_policy(policy: dict[str, Any]) -> ProsePolicy:
    prose = ProsePolicy.model_validate(policy["prose_policy"])
    for term in prose.internal_language:
        try:
            re.compile(term.pattern)
        except re.error as exc:
            raise PacketBuildError(f"PROSE_POLICY_INVALID: {term.concept}: {exc}") from exc
    return prose


def _display_policy(policy: dict[str, Any]) -> tuple[DisplayPolicy, dict[str, dict[str, Any]]]:
    raw = policy.get("display_policy")
    if raw is None:
        raise PacketBuildError("DISPLAY_POLICY_INVALID: brief policy has no display_policy")
    overrides = raw.get("fact_overrides", {})
    return DisplayPolicy(approximation_markers=raw["approximation_markers"]), overrides


def _presentation(fid: str, f: FinancialFact, uncertainty: Decimal | None,
                  override: dict[str, Any] | None) -> dict[str, Any]:
    """Python-owned presentation: display step, approximation status and canonical strings. Sourced figures keep
    their disclosed precision; a derived figure is rounded no finer than its input-rounding uncertainty allows."""
    sourced = f.origin is Origin.sourced
    native = Decimal(10) ** -(_decimals(f.precision) if sourced else DERIVED_DISPLAY_DECIMALS[f.unit])
    step = native if sourced or not uncertainty else max(native, uncertainty_step(uncertainty))
    approximate = step > native
    wording = None
    if override:
        problems = []
        if sourced:
            problems.append("sourced figures keep their disclosed display")
        if "display_step" in override:
            o_step = Decimal(str(override["display_step"]))
            if o_step < native or (uncertainty is not None and o_step < uncertainty):
                problems.append(f"display_step {o_step} is finer than the precision the inputs support")
            step = o_step
            approximate = approximate or step > native
        if override.get("approximate") is False and approximate:
            problems.append("an uncertainty-rounded figure cannot be marked exact")
        approximate = approximate or bool(override.get("approximate"))
        wording = override.get("display_wording")
        if wording is not None and re.search(r"\d", wording):
            problems.append("display_wording may not contain digits; numeric renderings are generated")
        if not override.get("reason"):
            problems.append("an override must give a reason")
        if problems:
            raise PacketBuildError(f"DISPLAY_POLICY_INVALID: {fid}: " + "; ".join(problems))
    display = format_display(f.value, f.unit, step, approximate)
    return dict(display=display, display_wording=wording or display, display_step=step, approximate=approximate,
                max_decimals=_decimals(step))


def _fact_flags(fact: FinancialFact) -> list[str]:
    flags = set(fact.review_flags)
    if fact.origin is Origin.derived:
        flags.add("derived_not_disclosed")
    if fact.price_basis is PriceBasis.mixed:
        flags.add("mixed_basis")
    if fact.basis in BASIS_FLAGS:
        flags.add(BASIS_FLAGS[fact.basis])
    return sorted(flags)


def _packet_hash(payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k not in ("generated_at", "packet_hash")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def build_brief_packet(
    benchmark: Benchmark | None = None,
    generated_at: datetime | None = None,
    policy: dict[str, Any] | None = None,
) -> BriefPacket:
    bm = benchmark or load_benchmark()
    try:
        result = run(bm)
    except ValidationFailed as exc:
        raise PacketBuildError(f"benchmark failed input validation; no packet built:\n{exc}") from exc
    policy = policy or load_policy(bm.name)
    material = set(policy["material_flags"])
    terminology: dict[str, str] = policy.get("terminology", {})

    available = {
        fid: f for fid, f in result.facts.items()
        if f.origin is Origin.sourced or result.calculations[fid].status is CalcStatus.computed
    }
    unavailable = {fid: result.calculations[fid] for fid in result.facts if fid not in available}

    notes = []
    note_ids_by_fact: dict[str, list[str]] = {}
    for note in policy["comparability_notes"]:
        subjects = [s for s in note["subject_ids"] if s in available]
        if not subjects:
            continue
        notes.append(ComparabilityNote(id=note["id"], subject_ids=subjects, text=note["text"]))
        for s in subjects:
            note_ids_by_fact.setdefault(s, []).append(note["id"])

    display_policy, overrides = _display_policy(policy)
    unknown = sorted(set(overrides) - set(available))
    if unknown:
        raise PacketBuildError(f"DISPLAY_POLICY_INVALID: overrides name facts not in the packet: {unknown}")
    widths: dict[str, Decimal | None] = {}
    facts = []
    for fid in sorted(available):
        f = available[fid]
        sourced = f.origin is Origin.sourced
        uncertainty = None if sourced else propagate_uncertainty(fid, result.facts, result.calculations, widths)
        flags = _fact_flags(f)
        facts.append(PacketFact(
            id=fid, metric=f.metric, period=f.period, period_type=f.period_type, scope=f.scope, unit=f.unit,
            origin=f.origin, value=f.value, **_presentation(fid, f, uncertainty, overrides.get(fid)),
            precision=f.precision, uncertainty=uncertainty,
            price_basis=f.price_basis, basis=_relabel(f.basis, terminology), flags=flags,
            requires_limitation=bool(material & set(flags)), source_ids=list(f.source_ids),
            derived_from=list(f.inputs), comparability_note_ids=note_ids_by_fact.get(fid, []),
        ))
    fact_ids = {f.id for f in facts}

    narratives_src = {n.id: n for n in bm.narratives if n.review_status is not ReviewStatus.rejected}

    def inherited(nid: str, seen: set[str]) -> list[str]:
        out: list[str] = []
        for p in narratives_src[nid].premise_ids:
            if p in narratives_src and p not in seen:
                seen.add(p)
                out += list(narratives_src[p].assumption_ids) + inherited(p, seen)
        return out

    narratives = []
    for nid in sorted(narratives_src):
        n = narratives_src[nid]
        inh = sorted(set(inherited(nid, {nid})))
        narratives.append(PacketNarrative(
            id=n.id, topic=n.topic, claim=n.claim, evidence_type=n.evidence_type, verbatim_quote=n.verbatim_quote,
            speaker=n.speaker, source_kind=n.source_kind, scope=n.scope, period=n.period,
            source_ids=list(n.source_ids), related_fact_ids=[i for i in n.related_fact_ids if i in fact_ids],
            premise_ids=list(n.premise_ids), assumption_ids=list(n.assumption_ids),
            inherited_assumption_ids=[a for a in inh if a not in n.assumption_ids],
            counter_evidence_ids=list(n.counter_evidence_ids), review_status=n.review_status,
            requires_limitation=n.review_status is ReviewStatus.needs_review or bool(n.assumption_ids or inh),
        ))
    assumptions = _packet_assumptions(bm, narratives)

    reconciliations = []
    for r in sorted(result.reconciliations, key=lambda x: x.id):
        inputs = set(r.input_fact_ids)
        if r.bridge_type is BridgeType.non_additive_set:
            related = sorted(inputs & fact_ids)
        else:
            components = set(r.component_fact_ids.values())
            derived_within = {f.id for f in facts if f.derived_from and set(f.derived_from) <= inputs}
            related = sorted((components | derived_within) & fact_ids)
        reconciliations.append(PacketReconciliation(
            id=r.id, name=r.bridge_name, period=r.period, scope=r.scope, bridge_type=r.bridge_type,
            status=r.status, starting_value=r.starting_value, ending_value=r.ending_value,
            reported_change=r.reported_change, identified_change=r.identified_change, residual=r.residual,
            allowed=r.tolerance.allowed if r.tolerance else None,
            unquantified_components=list(r.unquantified_components), circular=r.circular,
            precision_limited=r.precision_limited, reason=_relabel(r.reason, terminology),
            input_fact_ids=list(r.input_fact_ids),
            related_fact_ids=related, requires_limitation=r.status is not ReconciliationStatus.RECONCILED,
        ))
    known_ids = fact_ids | {r.id for r in reconciliations}

    open_flags = []
    for gap in sorted(bm.gaps, key=lambda g: g.id):
        links = [i for i in policy["gap_links"].get(gap.id, []) if i in known_ids]
        open_flags.append(OpenFlag(
            id=gap.id, kind=OpenFlagKind.disclosure_gap, subject_ids=[gap.metric], related_ids=links,
            message=(f"{gap.disclosure_status.value}: {gap.metric} ({gap.scope}, {gap.period}). "
                     f"{_relabel(_strip_workbook_refs(gap.reason), terminology)}").strip(),
        ))
    for n in narratives:
        if n.review_status is ReviewStatus.needs_review:
            open_flags.append(OpenFlag(id=f"RV-{n.id}", kind=OpenFlagKind.review_required_narrative,
                                       subject_ids=[n.id], message=f"Narrative {n.id} needs review."))
    for r in reconciliations:
        if r.status is ReconciliationStatus.UNRESOLVED:
            open_flags.append(OpenFlag(id=f"UR-{r.id}", kind=OpenFlagKind.unresolved_reconciliation,
                                       subject_ids=[r.id], related_ids=r.related_fact_ids, message=r.reason))
    for fid, calc in sorted(unavailable.items()):
        open_flags.append(OpenFlag(id=f"UA-{fid}", kind=OpenFlagKind.unavailable_fact, subject_ids=[fid],
                                   message=f"{fid} unavailable: {calc.reason}"))

    cited_sources = {s for f in facts for s in f.source_ids} | {s for n in narratives for s in n.source_ids}
    sources = [
        PacketSource(id=s.id, source_kind=s.source_kind, publisher=s.publisher, document=s.document,
                     filing_type=s.filing_type, accession_number=s.accession_number, url=s.url,
                     verification_method=s.verification_method)
        for s in sorted(bm.sources, key=lambda s: s.id) if s.id in cited_sources
    ]
    forbidden = [ForbiddenValue.model_validate(fv) for fv in policy["forbidden_values"]]
    surfacing = _surfacing_policy(policy, reconciliations, open_flags, notes, narratives)
    bounds = StatementBounds.model_validate(policy["statement_count"])
    if bounds.min > bounds.max:
        raise PacketBuildError(f"STATEMENT_BOUNDS_INVALID: min {bounds.min} > max {bounds.max}")

    payload = dict(
        analysis_id=policy["analysis_id"], company=policy["company"], period=policy["period"],
        scope=policy["scope"], generated_at=generated_at or datetime.now(timezone.utc), packet_hash="",
        material_flags=sorted(material), surfacing_policy=surfacing, statement_bounds=bounds,
        display_policy=display_policy, inference_policy=InferencePolicy.model_validate(policy["inference_policy"]),
        prose_policy=_prose_policy(policy), sources=sources, facts=facts, narratives=narratives,
        assumptions=assumptions,
        reconciliations=reconciliations, comparability_notes=notes, open_flags=open_flags,
        forbidden_values=forbidden,
    )
    draft = BriefPacket(**payload)
    leak = _INTERNAL_TERMS.search(draft.model_dump_json())
    if leak:
        raise PacketBuildError("INTERNAL_TERMINOLOGY: packet-visible text mentions the source workbook; "
                               "add a neutral term to the policy terminology map")
    return draft.model_copy(update={"packet_hash": _packet_hash(draft.model_dump(mode="json"))})
