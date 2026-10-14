"""Deterministic validator for synthesized briefs. A brief is VALID only if every claim traces to the packet."""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from itertools import permutations

from pydantic import ValidationError

from fpa_engine.brief import (
    BriefPacket,
    CoverageAlternative,
    CoverageRole,
    BriefStatement,
    ExecutiveBrief,
    ForbiddenValue,
    PacketFact,
    StatementKind,
    brief_title,
    round_to_step,
)
from fpa_engine.schemas import (
    AssertionVoice,
    AssumptionUse,
    BridgeType,
    EvidenceType,
    ReconciliationStatus,
    ReviewStatus,
    StrictModel,
)

HUNDRED = Decimal("100")
EPSILON = Decimal("1e-9")

_SUFFIX = (r"%|percent\b|percentage points?\b|pts?\b|points?\b|bps\b|basis points?\b"
           r"|billion\b|bn\b|B\b|million\b|mn\b|M\b")
_NUMBER = re.compile(
    r"(?P<sign>[+\-])?(?P<cur>\$)?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(?P<suffix>" + _SUFFIX + r")?"
)
_RANGE = re.compile(
    r"(?P<cur>\$)?(?P<a>\d[\d,]*(?:\.\d+)?)\s?[–\-]\s?\$?(?P<b>\d[\d,]*(?:\.\d+)?)\s?(?P<suffix>" + _SUFFIX + r")?"
)
_NON_QUANTITATIVE = [
    re.compile(r"\b(?:FY|CY)\s?'?\d{2,4}\b"),
    re.compile(r"\b[QH][1-4]\b"),
    re.compile(r"(?<![\$\d.,])\b(?:19|20)\d{2}\b(?!\s*(?:%|bps\b|basis\b|pts?\b|points?\b|percentage\b"
               r"|M\b|million\b|B\b|bn\b|billion\b))"),
    re.compile(r"\b[A-Z]{1,3}-?\d{2,3}\b"),
    re.compile(r"\b\d{1,2}-[KQ]\b"),
    re.compile(r"\bEx(?:hibit)?\.?\s?\d+(?:\.\d+)?\b"),
]

INTERPRETIVE_MARKERS = re.compile(
    r"\b(suggest\w*|appear\w*|likely|indicat\w*|impl(?:y|ies|ied)|consistent with|may|might|could"
    r"|points? to|seem\w*|interpret\w*|infer\w*|in our view|we read)\b", re.I)
DIRECTIONAL_QUALIFIERS = re.compile(
    r"\b(directional(?:ly)?|implied|indicative|not comparable|non-comparable|approximate|direction)\b", re.I)
COMPLETION_LANGUAGE = re.compile(
    r"\b(?:fully|completely|entirely|exactly)\s+(?:reconcil\w*|explain\w*|account\w*|attribut\w*|bridged)\b"
    r"|\breconciles\b|\b(?:is|are|was|were)\s+reconciled\b|\bcloses\b"
    r"|\baccounts?\s+for\s+(?:all|the\s+(?:full|entire|whole))\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|unresolved|partial(?:ly)?|partly|unassigned|cannot|incomplete)\b", re.I)
ADDITIVE_LANGUAGE = re.compile(
    r"\b(sum(?:s|med)?|adds?\s+up|added\s+up|plus|equals?|made\s+up\s+of|composed\s+of|consists?\s+of"
    r"|breaks?\s+down\s+into|decompos\w*|bridge[sd]?|reconcil\w*|accounts?\s+for)\b|(?<![<>!])=", re.I)
DISCLOSURE_VERB = re.compile(r"disclos", re.I)

@dataclass(frozen=True)
class UnitConversion:
    name: str
    source: str
    target: str
    factor: Decimal


# The only unit re-expressions a brief may use. Anything not registered here is an unsupported conversion.
UNIT_CONVERSIONS: dict[tuple[str, str], UnitConversion] = {
    (c.source, c.target): c for c in (
        UnitConversion("bps_to_percentage_points", "bps", "pts", Decimal("0.01")),
        UnitConversion("percentage_points_to_bps", "pts", "bps", Decimal("100")),
    )
}

UNIT_BY_SUFFIX = {
    "%": "pct", "percent": "pct",
    "percentage point": "pts", "percentage points": "pts", "pt": "pts", "pts": "pts", "point": "pts", "points": "pts",
    "bps": "bps", "basis point": "bps", "basis points": "bps",
    "billion": "usd_m", "bn": "usd_m", "b": "usd_m", "million": "usd_m", "mn": "usd_m", "m": "usd_m",
}
BILLIONS = {"billion", "bn", "b"}


class BriefIssue(StrictModel):
    code: str
    statement_id: str | None = None
    subject_id: str | None = None
    message: str


class BriefValidationResult(StrictModel):
    valid: bool
    errors: list[BriefIssue]

    @property
    def status(self) -> str:
        return "VALID" if self.valid else "REJECTED"

    @property
    def codes(self) -> set[str]:
        return {e.code for e in self.errors}


@dataclass(frozen=True)
class NumberToken:
    raw: str
    value: Decimal
    unit: str | None
    written_unit: Decimal
    start: int
    has_point: bool = False


@dataclass(frozen=True)
class RangeToken:
    raw: str
    low: Decimal
    high: Decimal
    unit: str | None


def _mask(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def _unit_and_scale(suffix: str | None, has_currency: bool) -> tuple[str | None, Decimal]:
    if not suffix:
        return ("usd_m" if has_currency else None), Decimal("1")
    key = suffix.lower()
    scale = Decimal("1000") if key in BILLIONS else Decimal("1")
    return UNIT_BY_SUFFIX[key], scale


def _to_decimal(num: str) -> tuple[Decimal, int]:
    clean = num.replace(",", "")
    decimals = len(clean.split(".")[1]) if "." in clean else 0
    return Decimal(clean), decimals


def extract_numbers(text: str) -> tuple[list[NumberToken], list[RangeToken]]:
    """Numeric claims in prose, ignoring periods (Q2 2026, FY2026), years and evidence IDs (S01, NA03)."""
    work = text.replace("−", "-")
    for pattern in _NON_QUANTITATIVE:
        work = pattern.sub(lambda m: " " * len(m.group(0)), work)

    ranges: list[RangeToken] = []
    for m in list(_RANGE.finditer(work)):
        unit, scale = _unit_and_scale(m.group("suffix"), bool(m.group("cur")))
        a, _ = _to_decimal(m.group("a"))
        b, _ = _to_decimal(m.group("b"))
        ranges.append(RangeToken(m.group(0), a * scale, b * scale, unit))

    tokens: list[NumberToken] = []
    range_spans = [(m.start(), m.end()) for m in _RANGE.finditer(work)]
    for m in _NUMBER.finditer(work):
        if any(s <= m.start("num") < e for s, e in range_spans):
            continue
        unit, scale = _unit_and_scale(m.group("suffix"), bool(m.group("cur")))
        value, decimals = _to_decimal(m.group("num"))
        tokens.append(NumberToken(m.group(0).strip(), value * scale, unit,
                                  Decimal(1).scaleb(-decimals) * scale, m.start(), decimals > 0))
    for r in ranges:
        for end in (r.low, r.high):
            tokens.append(NumberToken(r.raw, end, r.unit, Decimal("1"), -1))
    return tokens, ranges


def _unit_ok(token_unit: str | None, unit: str) -> bool:
    return token_unit is None or token_unit == unit


def express(value: Decimal, unit: str, max_decimals: int, token_unit: str | None) -> tuple[Decimal, int] | None:
    """Value and allowed decimals in the token's unit, via a registered conversion only; None if unsupported."""
    if _unit_ok(token_unit, unit):
        return value, max_decimals
    conversion = UNIT_CONVERSIONS.get((unit, token_unit))
    if conversion is None:
        return None
    return value * conversion.factor, max(0, max_decimals - conversion.factor.adjusted())


def _close(token: NumberToken, value: Decimal) -> bool:
    return abs(token.value - abs(value)) <= token.written_unit / 2 + EPSILON


def _expressed_with_step(fact: PacketFact, token_unit: str | None) -> tuple[Decimal, Decimal] | None:
    """The fact's value and display step in the token's unit, via a registered conversion only."""
    if _unit_ok(token_unit, fact.unit.value):
        return fact.value, fact.display_step
    conversion = UNIT_CONVERSIONS.get((fact.unit.value, token_unit))
    if conversion is None:
        return None
    return fact.value * conversion.factor, fact.display_step * conversion.factor


def matches_fact(token: NumberToken, fact: PacketFact) -> bool:
    """The token renders this fact: within its written rounding of the value, or equal to the canonical value
    rounded to the fact's display step (so "~510 bps" matches 509.07 shown at a 10 bps step)."""
    expressed = _expressed_with_step(fact, token.unit)
    if expressed is None:
        return False
    value, step = expressed
    return _close(token, value) or abs(token.value - abs(round_to_step(value, step))) <= EPSILON


def _false_precision(token: NumberToken, fact: PacketFact) -> bool:
    """Finer than the packet's display step: decimals beyond it, or an integer that is not a multiple of it."""
    expressed = _expressed_with_step(fact, token.unit)
    if expressed is None:
        return False
    step = expressed[1]
    if token.written_unit >= step:
        return False
    return token.has_point or token.value % step != 0


def _approximation_marked(text: str, start: int, markers: list[str]) -> bool:
    """An approved marker sits immediately before the figure (sign and currency symbol aside)."""
    before = text[:start].rstrip()
    for marker in markers:
        if marker[-1].isalpha():
            if re.search(r"\b" + re.escape(marker) + r"$", before, re.I):
                return True
        elif before.endswith(marker):
            return True
    return False


def _forbidden_hit(token: NumberToken, fv: ForbiddenValue) -> bool:
    if fv.value is None:
        return False
    fv_decimals = max(0, -fv.value.normalize().as_tuple().exponent)
    expressed = express(fv.value, fv.unit.value, fv_decimals, token.unit)
    if expressed is None:
        return False
    return token.written_unit <= Decimal(1).scaleb(-expressed[1]) and _close(token, expressed[0])


def _range_forbidden(rng: RangeToken, fv: ForbiddenValue) -> bool:
    return (fv.low is not None and _unit_ok(rng.unit, fv.unit.value)
            and rng.low == fv.low and rng.high == fv.high)


def _arithmetic_candidates(values: list[Decimal]) -> list[Decimal]:
    out = []
    for a, b in permutations(values, 2):
        out += [a - b, a + b, (a - b) * HUNDRED]
        if b != 0:
            out += [a / b * HUNDRED, (a / b - 1) * HUNDRED]
    return out


class _Context:
    def __init__(self, packet: BriefPacket):
        self.packet = packet
        self.facts = packet.fact_map
        self.narratives = packet.narrative_map
        self.recs = packet.reconciliation_map
        self.flags = packet.flag_map
        self.notes = {n.id for n in packet.comparability_notes}
        self.assumptions = packet.assumption_map
        markers = packet.inference_policy.assumption_inference_markers
        self.inference_framing = re.compile(r"\b(?:" + "|".join(re.escape(m) for m in markers) + r")\b", re.I)
        self.internal_language = [(t.concept, re.compile(t.pattern, re.I))
                                  for t in packet.prose_policy.internal_language]
        self.assertions = packet.assertion_map
        voice = packet.voice_policy
        self.attribution = [re.compile(p, re.I) for p in voice.attribution_patterns]
        self.characterization = re.compile(
            r"\b(?:" + "|".join(re.escape(w) for w in voice.characterization_terms) + r")\b", re.I)
        self.id_kinds = packet_id_kinds(packet)
        ids = sorted(self.id_kinds, key=len, reverse=True)
        self.packet_ids = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(i) for i in ids) + r")(?![\w-])")

    def narrative_chain(self, nid: str) -> set[str]:
        chain, pending = set(), [nid]
        while pending:
            cur = pending.pop()
            if cur in chain or cur not in self.narratives:
                continue
            chain.add(cur)
            pending += [p for p in self.narratives[cur].premise_ids if p in self.narratives]
        return chain


def packet_id_kinds(packet: BriefPacket) -> dict[str, str]:
    """Every identifier the packet defines, with the kind of item it names."""
    groups = [("source", packet.sources), ("fact", packet.facts), ("narrative", packet.narratives),
              ("assumption", packet.assumptions), ("assertion", packet.assertions),
              ("reconciliation", packet.reconciliations),
              ("comparability note", packet.comparability_notes), ("open flag", packet.open_flags),
              ("forbidden value", packet.forbidden_values)]
    return {item.id: kind for kind, items in groups for item in items}


def _check_prose(s: BriefStatement, ctx: _Context, errors: list[BriefIssue]) -> None:
    """Executive prose states the finance, not the engine: no pipeline vocabulary and no packet identifiers."""
    for concept, pattern in ctx.internal_language:
        m = pattern.search(s.text)
        if m:
            errors.append(_issue("INTERNAL_PIPELINE_LANGUAGE", s,
                                 f"'{m.group(0)}' is internal pipeline language ({concept}); state the financial "
                                 f"point or limitation itself, not how the analysis was produced or checked"))
    for found in dict.fromkeys(m.group(0) for m in ctx.packet_ids.finditer(s.text)):
        errors.append(_issue("PACKET_ID_IN_PROSE", s,
                             f"'{found}' is a packet {ctx.id_kinds[found]} ID; keep IDs in the reference fields and "
                             f"describe the item in words in the text", found))


def _issue(code: str, statement: BriefStatement | None, message: str, subject: str | None = None) -> BriefIssue:
    return BriefIssue(code=code, statement_id=statement.id if statement else None, subject_id=subject, message=message)


def _check_references(s: BriefStatement, ctx: _Context, errors: list[BriefIssue]) -> bool:
    known = True
    for fid in s.fact_ids:
        if fid not in ctx.facts:
            errors.append(_issue("UNKNOWN_FACT_ID", s, f"fact {fid} is not in the packet", fid))
            known = False
    for eid in s.evidence_ids:
        if eid not in ctx.narratives:
            errors.append(_issue("UNKNOWN_EVIDENCE_ID", s, f"evidence {eid} is not in the packet", eid))
            known = False
    for rid in s.reconciliation_ids:
        if rid not in ctx.recs:
            errors.append(_issue("UNKNOWN_RECONCILIATION_ID", s, f"reconciliation {rid} is not in the packet", rid))
            known = False
    for pid in s.premise_ids:
        if pid not in ctx.facts and pid not in ctx.narratives:
            errors.append(_issue("UNKNOWN_PREMISE_ID", s, f"premise {pid} is not in the packet", pid))
            known = False
    for gid in s.flag_ids:
        if gid not in ctx.flags:
            errors.append(_issue("UNKNOWN_FLAG_ID", s, f"open flag {gid} is not in the packet", gid))
            known = False
    for xid in s.assertion_ids:
        if xid not in ctx.assertions:
            errors.append(_issue("UNKNOWN_ASSERTION_ID", s, f"assertion {xid} is not in the packet", xid))
            known = False
    for aid in s.assumption_ids:
        if aid not in ctx.assumptions:
            errors.append(_issue("UNKNOWN_ASSUMPTION_ID", s, f"assumption {aid} is not in the packet", aid))
            known = False
    for cid in s.comparability_note_ids:
        if cid not in ctx.notes:
            errors.append(_issue("UNKNOWN_COMPARABILITY_NOTE_ID", s, f"comparability note {cid} is not in the packet",
                                 cid))
            known = False
    return known


def _check_kind(s: BriefStatement, ctx: _Context, errors: list[BriefIssue]) -> None:
    cited_narr = [ctx.narratives[e] for e in s.evidence_ids]
    if s.kind is StatementKind.fact:
        if not s.fact_ids:
            errors.append(_issue("FACT_WITHOUT_CITATION", s, "a fact statement must cite at least one fact ID"))
        for n in cited_narr:
            if n.evidence_type is EvidenceType.interpretive:
                errors.append(_issue("INTERPRETIVE_AS_FACT", s, f"{n.id} is interpretive evidence", n.id))
    elif s.kind is StatementKind.explicit_claim:
        if not s.evidence_ids:
            errors.append(_issue("EXPLICIT_WITHOUT_EVIDENCE", s, "an explicit claim must cite narrative evidence"))
        for n in cited_narr:
            if n.evidence_type is not EvidenceType.explicit:
                errors.append(_issue("INTERPRETIVE_AS_EXPLICIT", s,
                                     f"{n.id} is interpretive evidence and cannot support an explicit claim", n.id))
    elif s.kind is StatementKind.interpretive:
        if not s.premise_ids:
            errors.append(_issue("MISSING_PREMISES", s, "an interpretive statement must cite its premises"))
        if not INTERPRETIVE_MARKERS.search(s.text):
            errors.append(_issue("INTERPRETIVE_MARKER_MISSING", s,
                                 "interpretive text reads as a disclosed fact; mark it as inference"))
    elif not (s.fact_ids or s.evidence_ids or s.reconciliation_ids or s.premise_ids or s.flag_ids
              or s.comparability_note_ids):
        errors.append(_issue("LIMITATION_WITHOUT_SUBJECT", s, "a limitation must cite what it limits"))


def _cited_facts(s: BriefStatement, ctx: _Context) -> list[PacketFact]:
    ids = list(dict.fromkeys([*s.fact_ids, *[p for p in s.premise_ids if p in ctx.facts]]))
    return [ctx.facts[i] for i in ids if i in ctx.facts]


def _cited_narratives(s: BriefStatement, ctx: _Context) -> list[str]:
    return list(dict.fromkeys([*s.evidence_ids, *[p for p in s.premise_ids if p in ctx.narratives]]))


_QUOTED = re.compile(r"[\"“”]([^\"“”]+)[\"“”]")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")


def _normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip().strip(".,;: ").casefold()


def quoted_spans(text: str) -> list[str]:
    return [m.group(1) for m in _QUOTED.finditer(text)]


def _quote_matches(span: str, verbatim: str) -> bool:
    """Every fragment of the span (split at an ellipsis) is an exact excerpt of the verbatim quote, ignoring case
    and whitespace runs."""
    source = _normalize_quote(verbatim)
    fragments = [f for f in (_normalize_quote(x) for x in _ELLIPSIS.split(span)) if f]
    return bool(fragments) and all(f in source for f in fragments)


def _quotable(nid: str, ctx: _Context) -> bool:
    n = ctx.narratives.get(nid)
    return bool(n and n.evidence_type is EvidenceType.explicit and n.review_status is ReviewStatus.validated
                and n.verbatim_quote)


def _check_voice(s: BriefStatement, ctx: _Context, errors: list[BriefIssue]) -> None:
    """Assertions, source versus analyst voice, attribution, verbatim quotes and characterization."""
    cited_narratives = set(_cited_narratives(s, ctx))
    cited = [ctx.assertions[x] for x in s.assertion_ids]
    for a in cited:
        if a.narrative_id not in cited_narratives:
            errors.append(_issue("ASSERTION_WITHOUT_EVIDENCE", s,
                                 f"{a.id} belongs to {a.narrative_id}; cite that narrative in evidence_ids or "
                                 f"premise_ids", a.id))
    if s.kind is StatementKind.explicit_claim and not s.assertion_ids:
        errors.append(_issue("EXPLICIT_WITHOUT_ASSERTION", s,
                             "an explicit claim must list the assertions it restates in assertion_ids"))
    attributed = any(p.search(s.text) for p in ctx.attribution)
    for a in cited:
        if a.voice is AssertionVoice.source and not attributed:
            errors.append(_issue("SOURCE_VOICE_NOT_ATTRIBUTED", s,
                                 f"{a.id} is the source's own wording; attribute it to the source in the statement",
                                 a.id))
    denies = s.kind is StatementKind.limitation and NEGATION.search(s.text)
    backed = any(a.voice is AssertionVoice.source and a.narrative_id in cited_narratives
                 and _quotable(a.narrative_id, ctx) for a in cited)
    if attributed and not denies and not backed:
        errors.append(_issue("ATTRIBUTION_WITHOUT_EVIDENCE", s,
                             "the statement attributes words or expectations to a source; cite a source-voice "
                             "assertion of validated, quoted explicit evidence, or state it without attribution"))
    spans = quoted_spans(s.text)
    verbatims = [ctx.narratives[n].verbatim_quote for n in sorted(cited_narratives) if _quotable(n, ctx)]
    for span in spans:
        if not any(_quote_matches(span, v) for v in verbatims):
            errors.append(_issue("QUOTE_NOT_VERBATIM", s,
                                 f"\"{span}\" is not an exact excerpt of the verbatim quote of any validated "
                                 f"explicit evidence the statement cites"))
    if spans and not attributed:
        errors.append(_issue("QUOTE_NOT_ATTRIBUTED", s, "a quotation must be attributed to its source"))
    if s.kind is StatementKind.limitation:
        return
    analyst = [a.text for a in cited if a.voice is AssertionVoice.analyst]
    if s.kind is StatementKind.interpretive:
        analyst += [ctx.assumptions[x].text for x in s.assumption_ids if x in ctx.assumptions]
    source = [f"{a.text} {a.quote or ''}" for a in cited if a.voice is AssertionVoice.source]
    for term in dict.fromkeys(m.group(0).lower() for m in ctx.characterization.finditer(s.text)):
        word = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
        if any(word.search(x) for x in analyst) or (attributed and any(word.search(x) for x in source)):
            continue
        owners = sorted(a.id for n in cited_narratives for a in ctx.assertions.values()
                        if a.narrative_id == n and word.search(f"{a.text} {a.quote or ''}"))
        hint = (f"; only {', '.join(owners)} uses it: cite it and, if it is source wording, attribute it"
                if owners else "; no cited assertion supports it")
        errors.append(_issue("UNSUPPORTED_CHARACTERIZATION", s, f"'{term}' characterizes the evidence{hint}", term))


def assumption_dependencies(s: BriefStatement, ctx: _Context) -> set[str]:
    """Assumptions the statement rests on: those of every narrative it cites (evidence or premise) and of every
    narrative in their premise chains. Derived from references only, never from prose."""
    return {a for nid in _cited_narratives(s, ctx) for chained in ctx.narrative_chain(nid)
            for a in ctx.narratives[chained].assumption_ids}


def _used_assumptions(s: BriefStatement, ctx: _Context) -> set[str]:
    return assumption_dependencies(s, ctx) | set(s.assumption_ids)


def _check_assumptions(s: BriefStatement, ctx: _Context, errors: list[BriefIssue]) -> None:
    if s.kind is StatementKind.limitation:
        return
    required = assumption_dependencies(s, ctx)
    for aid in sorted(required - set(s.assumption_ids)):
        errors.append(_issue("HIDDEN_ASSUMPTION", s,
                             f"the statement's premises rest on {aid} ({ctx.assumptions[aid].text}); list it in "
                             f"assumption_ids", aid))
    for aid in sorted(_used_assumptions(s, ctx)):
        a = ctx.assumptions[aid]
        if a.allowed_use is AssumptionUse.interpretive_only and s.kind is not StatementKind.interpretive:
            errors.append(_issue("ASSUMPTION_AS_FACT", s,
                                 f"{aid} is {a.sourcing.value} and may support only an interpretive statement, "
                                 f"not a {s.kind.value} statement", aid))
        elif (s.kind is StatementKind.interpretive and (a.must_disclose or a.sourcing.value == "unsourced")
              and not ctx.inference_framing.search(s.text)):
            errors.append(_issue("MISSING_ASSUMPTION_DISCLOSURE", s,
                                 f"the statement rests on {a.sourcing.value} assumption {aid}; frame it in the "
                                 f"statement itself as inference (for example 'we read', 'consistent with')", aid))


def _check_numbers(s: BriefStatement, ctx: _Context, errors: list[BriefIssue]) -> None:
    tokens, ranges = extract_numbers(s.text)
    forbidden = ctx.packet.forbidden_values

    blocked_ranges = set()
    for rng in ranges:
        hit = next((fv for fv in forbidden if _range_forbidden(rng, fv)), None)
        if hit:
            blocked_ranges.add(rng.raw)
            errors.append(_issue("FORBIDDEN_VALUE", s, f"'{rng.raw}' is a forbidden value: {hit.reason}", hit.id))

    cited = _cited_facts(s, ctx)
    markers = ctx.packet.display_policy.approximation_markers
    rec_values = [v for rid in s.reconciliation_ids for v in (
        ctx.recs[rid].residual, ctx.recs[rid].allowed, ctx.recs[rid].reported_change,
        ctx.recs[rid].identified_change, ctx.recs[rid].starting_value, ctx.recs[rid].ending_value) if v is not None]
    quote_tokens: list[NumberToken] = []
    if s.kind in (StatementKind.explicit_claim, StatementKind.limitation):
        for nid in s.evidence_ids:
            n = ctx.narratives[nid]
            quote_tokens += extract_numbers(" ".join(filter(None, [n.verbatim_quote, n.claim])))[0]
    arithmetic = _arithmetic_candidates([f.value for f in cited])

    for token in tokens:
        if token.raw in blocked_ranges:
            continue
        hit = next((fv for fv in forbidden if _forbidden_hit(token, fv)), None)
        if hit:
            errors.append(_issue("FORBIDDEN_VALUE", s, f"'{token.raw}' is a forbidden value: {hit.reason}", hit.id))
            continue
        matched = [f for f in cited if matches_fact(token, f)]
        if matched:
            fact = matched[0]
            precise_ok = [f for f in matched if not _false_precision(token, f)]
            if not precise_ok:
                errors.append(_issue("FALSE_PRECISION", s,
                                     f"'{token.raw}' is more precise than {fact.id} allows; write it as "
                                     f"'{fact.display}'", fact.id))
            elif (token.start >= 0 and all(f.approximate for f in precise_ok)
                  and not _approximation_marked(s.text, token.start, markers)):
                approx = precise_ok[0]
                errors.append(_issue("APPROXIMATION_MARKER_MISSING", s,
                                     f"'{token.raw}' renders {approx.id}, which is approximate; write it as "
                                     f"'{approx.display}' or with another approved approximation marker", approx.id))
            window = s.text[max(0, token.start - 40):token.start] if token.start >= 0 else ""
            if all(f.origin.value == "derived" for f in matched) and DISCLOSURE_VERB.search(window):
                errors.append(_issue("DERIVED_PRESENTED_AS_DISCLOSED", s,
                                     f"'{token.raw}' is derived ({fact.id}), not a disclosed figure", fact.id))
            continue
        if any(_unit_ok(token.unit, "usd_m") and _close(token, v) for v in rec_values):
            continue
        if any(q.value == token.value and (token.unit is None or q.unit in (None, token.unit)) for q in quote_tokens):
            continue
        if any(_close(token, c) for c in arithmetic):
            errors.append(_issue("ARITHMETIC_IN_PROSE", s,
                                 f"'{token.raw}' is computed from cited facts in prose; it must exist as a derived fact"))
            continue
        uncited = next((f for f in ctx.packet.facts if matches_fact(token, f)), None)
        if uncited:
            errors.append(_issue("UNCITED_FACT_VALUE", s, f"'{token.raw}' matches {uncited.id}, which is not cited",
                                 uncited.id))
            continue
        errors.append(_issue("UNSUPPORTED_NUMBER", s, f"'{token.raw}' does not match any cited fact or evidence"))


def _check_semantics(s: BriefStatement, ctx: _Context, errors: list[BriefIssue]) -> None:
    cited = _cited_facts(s, ctx)
    cited_ids = {f.id for f in cited}
    if s.kind in (StatementKind.fact, StatementKind.interpretive):
        for f in cited:
            if {"non_comparable", "directional"} & set(f.flags) and not DIRECTIONAL_QUALIFIERS.search(s.text):
                errors.append(_issue("NON_COMPARABLE_PRESENTED_AS_PRECISE", s,
                                     f"{f.id} is directional/non-comparable; say so in the statement", f.id))
    if s.kind is StatementKind.limitation:
        return
    involved = {rid for rid in s.reconciliation_ids}
    involved |= {r.id for r in ctx.recs.values() if set(r.related_fact_ids) & cited_ids}
    for rid in sorted(involved):
        r = ctx.recs[rid]
        if (r.status is ReconciliationStatus.UNRESOLVED and COMPLETION_LANGUAGE.search(s.text)
                and not NEGATION.search(s.text)):
            errors.append(_issue("UNRESOLVED_PRESENTED_AS_RECONCILED", s,
                                 f"{rid} is UNRESOLVED ({r.reason}); it cannot be presented as complete", rid))
    for r in ctx.recs.values():
        if r.bridge_type is not BridgeType.non_additive_set:
            continue
        co_cited = len(set(r.related_fact_ids) & cited_ids) >= 2 or r.id in s.reconciliation_ids
        if co_cited and ADDITIVE_LANGUAGE.search(s.text):
            errors.append(_issue("NA_RELATIONSHIP_AS_ADDITIVE", s,
                                 f"{r.id} ({r.name}) is NOT_APPLICABLE and cannot be treated as additive", r.id))


def _check_coverage(statements: list[BriefStatement], ctx: _Context, errors: list[BriefIssue]) -> None:
    lims = [s for s in statements if s.kind is StatementKind.limitation]
    lim_facts = {i for s in lims for i in [*s.fact_ids, *s.premise_ids]}
    lim_narr = {i for s in lims for i in [*s.evidence_ids, *s.premise_ids]}
    lim_recs = {i for s in lims for i in s.reconciliation_ids}
    lim_flags = {i for s in lims for i in s.flag_ids}
    lim_assumptions = {i for s in lims for i in s.assumption_ids}
    reported: set[tuple[str, str]] = set()

    def report(code: str, s: BriefStatement, subject: str, message: str) -> None:
        if (code, subject) not in reported:
            reported.add((code, subject))
            errors.append(_issue(code, s, message, subject))

    for s in statements:
        if s.kind is StatementKind.limitation:
            continue
        cited = {f.id for f in _cited_facts(s, ctx)}
        for fid in sorted(cited):
            f = ctx.facts[fid]
            if f.requires_limitation and fid not in lim_facts:
                flags = sorted(set(f.flags) & set(ctx.packet.material_flags))
                report("MISSING_FLAG_LIMITATION", s, fid, f"{fid} carries {', '.join(flags)}; add a limitation citing it")
        involved = set(s.reconciliation_ids)
        for r in ctx.recs.values():
            members = set(r.related_fact_ids) & cited
            if r.bridge_type is BridgeType.non_additive_set:
                if len(members) >= 2:
                    involved.add(r.id)
            elif members and r.status is not ReconciliationStatus.RECONCILED:
                involved.add(r.id)
        for rid in sorted(involved):
            r = ctx.recs[rid]
            if r.requires_limitation and rid not in lim_recs:
                code = ("MISSING_NA_LIMITATION" if r.status is ReconciliationStatus.NOT_APPLICABLE
                        else "MISSING_RECONCILIATION_LIMITATION")
                report(code, s, rid, f"{rid} is {r.status.value}; add a limitation citing it")
        conditional = set(ctx.packet.surfacing_policy.conditional_narrative_ids)
        for nid in _cited_narratives(s, ctx):
            problems = ctx.narrative_chain(nid) & conditional
            if problems and not ({nid} | problems) & lim_narr:
                report("MISSING_REVIEW_LIMITATION", s, nid,
                       f"{nid} depends on review-required or assumption-based evidence ({', '.join(sorted(problems))})")
        for aid in sorted(_used_assumptions(s, ctx)):
            if ctx.assumptions[aid].must_disclose and aid not in lim_assumptions:
                report("MISSING_ASSUMPTION_LIMITATION", s, aid,
                       f"{aid} is {ctx.assumptions[aid].sourcing.value}; add a limitation citing it in assumption_ids")
        touched = cited | involved
        for flag in ctx.packet.open_flags:
            if flag.kind.value == "disclosure_gap" and set(flag.related_ids) & touched and flag.id not in lim_flags:
                report("MISSING_DISCLOSURE_GAP", s, flag.id, f"disclosure gap {flag.id} applies; add a limitation citing it")


def validate_statements(packet: BriefPacket, statements: list[BriefStatement]) -> BriefValidationResult:
    ctx = _Context(packet)
    errors: list[BriefIssue] = []
    if not statements:
        errors.append(_issue("EMPTY_BRIEF", None, "the brief has no statements"))
    seen: set[str] = set()
    for s in statements:
        if s.id in seen:
            errors.append(_issue("DUPLICATE_STATEMENT_ID", s, f"statement id {s.id} is repeated"))
        seen.add(s.id)

    resolvable = []
    for s in statements:
        if _check_references(s, ctx, errors):
            _check_kind(s, ctx, errors)
            _check_prose(s, ctx, errors)
            _check_assumptions(s, ctx, errors)
            _check_voice(s, ctx, errors)
            _check_numbers(s, ctx, errors)
            _check_semantics(s, ctx, errors)
            resolvable.append(s)
    _check_coverage(resolvable, ctx, errors)
    return BriefValidationResult(valid=not errors, errors=errors)


def statement_references(s: BriefStatement, packet: BriefPacket) -> dict[str, set[str]]:
    """The structured references a statement carries, grouped by entity type. Coverage reads only these."""
    facts, narratives = {f.id for f in packet.facts}, {n.id for n in packet.narratives}
    return {
        "fact_ids": set(s.fact_ids) | {p for p in s.premise_ids if p in facts},
        "reconciliation_ids": set(s.reconciliation_ids),
        "narrative_ids": set(s.evidence_ids) | {p for p in s.premise_ids if p in narratives},
        "assumption_ids": set(s.assumption_ids),
    }


def _covers(refs: dict[str, set[str]], alt: CoverageAlternative) -> bool:
    return all(set(getattr(alt, field)) <= ids for field, ids in refs.items())


def _describe_alternative(alt: CoverageAlternative) -> str:
    parts = [f"{field} {' + '.join(getattr(alt, field))}" for field in
             ("fact_ids", "reconciliation_ids", "narrative_ids", "assumption_ids") if getattr(alt, field)]
    return " and ".join(parts)


def covered_roles(packet: BriefPacket, statements: list[BriefStatement]) -> dict[str, list[str]]:
    """Role -> IDs of the statements covering it. A statement may cover several roles; each role needs one
    statement of an allowed kind that cites every ID of one alternative."""
    out: dict[str, list[str]] = {}
    refs = [(s, statement_references(s, packet)) for s in statements]
    for role in packet.coverage_policy.roles:
        if role.from_surfacing_policy:
            continue
        out[role.role] = [s.id for s, r in refs
                          if s.kind in role.statement_kinds and any(_covers(r, alt) for alt in role.any_of)]
    return out


def _coverage_issue(role: CoverageRole) -> BriefIssue:
    kinds = " or ".join(k.value for k in role.statement_kinds)
    options = "; or ".join(_describe_alternative(a) for a in role.any_of)
    return _issue("MISSING_REQUIRED_COVERAGE", None,
                  f"required role '{role.role}' is not covered ({role.purpose}). Cover it with one {kinds} "
                  f"statement whose reference fields include all of: {options}", role.role)


def _check_brief_policy(packet: BriefPacket, brief: ExecutiveBrief) -> list[BriefIssue]:
    """Brief-level rules from the packet policy: deterministic title, statement-count bounds, mandatory surfacing."""
    issues: list[BriefIssue] = []
    expected = brief_title(packet)
    if brief.title != expected:
        issues.append(_issue("TITLE_NOT_DETERMINISTIC", None,
                             f"title must be exactly '{expected}'; titles are generated from packet metadata"))
    n, bounds = len(brief.statements), packet.statement_bounds
    if not bounds.min <= n <= bounds.max:
        issues.append(_issue("STATEMENT_COUNT_OUT_OF_RANGE", None,
                             f"brief has {n} statements; {bounds.min} to {bounds.max} are allowed"))
    lims = [s for s in brief.statements if s.kind is StatementKind.limitation]
    sp = packet.surfacing_policy
    required = (
        [(r, "reconciliation_ids", "reconciliation") for r in sp.mandatory_reconciliation_ids]
        + [(g, "flag_ids", "disclosure gap") for g in sp.mandatory_disclosure_gap_ids]
        + [(c, "comparability_note_ids", "comparability limitation") for c in sp.mandatory_comparability_note_ids]
    )
    for subject, field, label in required:
        if not any(subject in getattr(s, field) for s in lims):
            issues.append(_issue("MISSING_MANDATORY_LIMITATION", None,
                                 f"{label} {subject} is designated mandatory by policy; a limitation must cite it "
                                 f"in {field}", subject))
    covered = covered_roles(packet, brief.statements)
    roles = {r.role: r for r in packet.coverage_policy.roles}
    issues += [_coverage_issue(roles[name]) for name, by in covered.items() if not by]
    return issues


def validate_brief(packet: BriefPacket, brief: ExecutiveBrief) -> BriefValidationResult:
    meta = brief.validation_metadata
    pre: list[BriefIssue] = []
    if meta.analysis_id != packet.analysis_id:
        pre.append(_issue("PACKET_MISMATCH", None, f"brief targets {meta.analysis_id}, packet is {packet.analysis_id}"))
    if meta.packet_hash != packet.packet_hash:
        pre.append(_issue("PACKET_HASH_MISMATCH", None,
                          "brief was written against a different packet version", meta.packet_hash))
    result = validate_statements(packet, brief.statements)
    errors = pre + result.errors + _check_brief_policy(packet, brief)
    return BriefValidationResult(valid=not errors, errors=errors)


def validate_brief_payload(packet: BriefPacket, payload: dict | str) -> BriefValidationResult:
    """Validate raw brief JSON: schema failures become deterministic rejections instead of exceptions."""
    try:
        brief = (ExecutiveBrief.model_validate_json(payload) if isinstance(payload, str)
                 else ExecutiveBrief.model_validate(payload))
    except ValidationError as exc:
        return BriefValidationResult(valid=False, errors=schema_issues(exc))
    return validate_brief(packet, brief)


def schema_issues(exc: ValidationError) -> list[BriefIssue]:
    errors = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        code = ("MISSING_PACKET_HASH" if loc == "validation_metadata.packet_hash" and err["type"] == "missing"
                else "SCHEMA_INVALID")
        errors.append(BriefIssue(code=code, message=f"{loc}: {err['msg']}" if loc else err["msg"]))
    return errors
