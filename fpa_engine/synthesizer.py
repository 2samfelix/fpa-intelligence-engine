"""Gate C: structured executive-brief synthesis. The model writes language; the deterministic validator decides.

Flow: BriefPacket -> LLM structured synthesis -> schema validation -> deterministic brief validation -> ACCEPT/REJECT.
The model writes statements only; the title is built from packet metadata. Nothing in this module imports a
provider SDK; provider code lives in fpa_engine.llm.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from pydantic import ValidationError

from fpa_engine.brief import (
    BriefMetadata,
    BriefPacket,
    BriefStatement,
    ExecutiveBrief,
    brief_title,
    build_brief_packet,
)
from fpa_engine.brief_validator import BriefIssue, BriefValidationResult, schema_issues, validate_brief
from fpa_engine.schemas import StrictModel

PROMPT_VERSION = "synthesis_v2_1"
MAX_ATTEMPTS = 2  # one initial attempt plus at most one corrective retry
AUTHOR = PROMPT_VERSION

logger = logging.getLogger("fpa_engine.synthesis")


# ------------------------------------------------------------ interface ---

class LLMProviderError(Exception):
    """The provider could not produce output (transport failure, API error, refusal). Not retried here."""


class LLMTimeoutError(LLMProviderError):
    pass


class LLMRefusalError(LLMProviderError):
    """The configured model declined. There is no fallback to another model."""


class MalformedOutputError(Exception):
    """The model answered, but not with a single JSON object. Treated as a rejected attempt."""

    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


class SynthesisOutput(StrictModel):
    """Exactly what the model may return. No title: titles are generated deterministically."""
    statements: list[BriefStatement]
    validation_metadata: BriefMetadata


class LLMClient(Protocol):
    def generate_structured_brief(
        self, packet: BriefPacket, *, corrections: Sequence[BriefIssue] = ()
    ) -> dict: ...


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    keys = [k for k, _ in pairs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"duplicate keys {dupes}")
    return dict(pairs)


def parse_model_output(text: str) -> dict:
    """Strictly parse raw model text: exactly one JSON object, no markdown fences, no surrounding prose."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        raise MalformedOutputError("output must be a single JSON object with no prose or markdown around it", text)
    try:
        obj = json.loads(stripped, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise MalformedOutputError(f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})", text) from exc
    except ValueError as exc:
        raise MalformedOutputError(f"invalid JSON: {exc}", text) from exc
    return obj


# --------------------------------------------------------------- prompt ---

SYSTEM_PROMPT = """\
You write executive briefs for FP&A analysis. You are given a BRIEF PACKET: a validated, allowlisted set of \
facts, narrative evidence, reconciliation results, comparability notes, open flags and forbidden values. \
The packet is the only source of truth. Your output is checked by a deterministic validator; any violation \
rejects the whole brief.

YOUR ROLE
You write language only. You do not calculate, estimate, or add knowledge. Use only facts and evidence in the \
packet. Do not use outside knowledge about the company, the industry, or the period. Do not invent sources, IDs, \
facts, causes, or numbers.

OUTPUT FORMAT
Return exactly one JSON object and nothing else: no markdown, no code fences, no commentary. Shape:
{
  "statements": [
    {
      "id": string (B1, B2, ... unique),
      "text": string,
      "kind": "fact" | "explicit_claim" | "interpretive" | "limitation",
      "fact_ids": [packet fact IDs],
      "evidence_ids": [packet narrative IDs],
      "reconciliation_ids": [packet reconciliation IDs],
      "premise_ids": [packet fact or narrative IDs],
      "flag_ids": [packet open-flag IDs],
      "comparability_note_ids": [packet comparability-note IDs],
      "severity": "info" | "warning" | "critical"
    }
  ],
  "validation_metadata": {"analysis_id": string, "packet_hash": string, "author": string}
}
No other fields are allowed anywhere; in particular do not write a title (it is generated from packet metadata). \
Copy analysis_id and packet_hash exactly as given in the request.

BRIEF SHAPE
Write concise statements in an executive register. The number of statements must lie within \
statement_bounds (min to max, inclusive) in the packet; aim for the low end. Order them by priority: (1) what changed, \
(2) what appears to drive it, (3) what is directly disclosed, (4) what is interpretive, (5) what remains \
unresolved. Use a mix of statement kinds as the evidence warrants; do not force every kind.

STATEMENT KINDS
- fact: states packet facts. fact_ids must list every fact whose value appears in the text. Never cite \
narrative evidence whose evidence_type is "interpretive".
- explicit_claim: restates what a source explicitly says. evidence_ids must cite only narratives whose \
evidence_type is "explicit".
- interpretive: an inference drawn from packet items. premise_ids must list every fact and narrative it rests \
on. The text must signal inference with words such as "appears", "suggests", "indicates", "likely", "may", or \
"is consistent with". Never present an inference as disclosed or explicit. Narratives with evidence_type \
"interpretive" may only support interpretive statements (as premise_ids) or limitations.
- limitation: states what is uncertain, unresolved, non-comparable, untested, or missing. It must cite what it \
limits in fact_ids, evidence_ids, reconciliation_ids, premise_ids, flag_ids, or comparability_note_ids.

NUMBERS (strict)
- Every amount, percentage, point, or basis-point figure in a statement must be the "display" value of a fact \
cited by that same statement (in fact_ids or premise_ids). Copy the display value exactly; never add precision.
- A limitation that cites a reconciliation in reconciliation_ids may quote that reconciliation's residual. A \
statement that cites explicit narrative evidence may quote a figure that appears in that narrative's claim or \
verbatim_quote.
- Do not calculate. No differences, sums, ratios, growth rates, averages, shares, or unit conversions, even \
simple ones. If a figure is not already a packet fact, describe the movement in words ("rose", "fell", \
"narrowed") without a number.
- Never write any value listed in forbidden_values, in any statement of any kind, in any form - not even in a \
limitation that rejects it.
- Facts with origin "derived" were calculated, not disclosed by the company. Never call them disclosed.
- Fiscal periods (Q2 2026) and years are not quantitative claims and may be used. Packet IDs belong only in the \
ID fields; never write them in statement text.

HONESTY RULES
- Facts flagged "non_comparable" or "directional" must be described as directional or not comparable in the \
same statement.
- Never describe an UNRESOLVED reconciliation as reconciled, closed, fully explained, or fully accounted for.
- Members of a NOT_APPLICABLE reconciliation are not on a common basis. Never describe them as adding up, \
summing, bridging, equalling, decomposing, or accounting for one another.
- Never relabel interpretive evidence as explicit, and never present review-required evidence as settled.

REQUIRED LIMITATIONS (every one is checked)
1. For every fact with requires_limitation true that you cite in a non-limitation statement, include a \
limitation statement citing that fact ID in fact_ids.
2. For every reconciliation with requires_limitation true that your non-limitation statements involve - cited in \
reconciliation_ids; an additive bridge that is not RECONCILED whose related_fact_ids you cite; or a \
NOT_APPLICABLE set where one statement cites two or more of its related_fact_ids - include a limitation citing \
that reconciliation ID in reconciliation_ids.
3. For every narrative listed in surfacing_policy.conditional_narrative_ids that your brief relies on (cited \
directly, or reached through the premise_ids of a narrative you cite), include a limitation citing it in \
evidence_ids. Do not add limitations for conditional narratives the brief does not rely on.
4. For every open flag of kind "disclosure_gap" whose related_ids include a fact or reconciliation you cite, \
include a limitation citing that flag ID in flag_ids.
5. MANDATORY, in every brief, whatever else it says (surfacing_policy): every ID in \
mandatory_reconciliation_ids must be cited by a limitation in reconciliation_ids; every ID in \
mandatory_disclosure_gap_ids by a limitation in flag_ids; every ID in mandatory_comparability_note_ids by a \
limitation in comparability_note_ids. Do not hide unresolved items.
Limitations may combine several subjects into one statement.
"""


def packet_for_prompt(packet: BriefPacket) -> dict[str, Any]:
    """The packet exactly as validated, minus the build timestamp (so the prompt is a pure function of the hash)."""
    return packet.model_dump(mode="json", exclude={"generated_at"})


def build_user_prompt(packet: BriefPacket) -> str:
    body = json.dumps(packet_for_prompt(packet), sort_keys=True, indent=1, ensure_ascii=False)
    return (
        f"Write the executive brief for analysis_id \"{packet.analysis_id}\" "
        f"({packet.company}, {packet.scope}, {packet.period}).\n"
        f"Set validation_metadata to analysis_id \"{packet.analysis_id}\", packet_hash \"{packet.packet_hash}\", "
        f"author \"{AUTHOR}\".\n\n"
        f"<brief_packet>\n{body}\n</brief_packet>"
    )


def build_correction_prompt(errors: Sequence[BriefIssue]) -> str:
    lines = []
    for e in errors:
        where = ", ".join(p for p in (f"statement {e.statement_id}" if e.statement_id else None,
                                      f"subject {e.subject_id}" if e.subject_id else None) if p)
        lines.append(f"- {e.code}{f' ({where})' if where else ''}: {e.message}")
    return (
        "A previous brief written from this packet was REJECTED by the deterministic validator for these "
        "violations:\n" + "\n".join(lines) + "\n\n"
        "Write the brief again from the packet. Correct only these violations; keep every other rule. "
        "Return only the JSON object."
    )


def build_messages(packet: BriefPacket, corrections: Sequence[BriefIssue] = ()) -> tuple[str, list[dict]]:
    """(system, messages) for one attempt. A retry sees only the packet, the validator errors, and the instruction."""
    content = build_user_prompt(packet)
    if corrections:
        content += "\n\n" + build_correction_prompt(corrections)
    return SYSTEM_PROMPT, [{"role": "user", "content": content}]


# ---------------------------------------------------------------- flow ---

class SynthesisStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    SYNTHESIS_REJECTED = "SYNTHESIS_REJECTED"
    PROVIDER_ERROR = "PROVIDER_ERROR"


class AttemptRecord(StrictModel):
    attempt: int
    status: str  # VALID | REJECTED | PROVIDER_ERROR
    errors: list[BriefIssue]
    output: Any = None


class SynthesisResult(StrictModel):
    status: SynthesisStatus
    brief: ExecutiveBrief | None = None
    errors: list[BriefIssue]
    attempts: list[AttemptRecord]
    model: str
    packet_hash: str
    prompt_version: str

    @property
    def accepted(self) -> bool:
        return self.status is SynthesisStatus.ACCEPTED


def evaluate_output(packet: BriefPacket, payload: Any) -> tuple[BriefValidationResult, ExecutiveBrief | None]:
    """Schema-validate the model output, attach the deterministic title, then run the deterministic validator.
    The validator is the final authority; output is never repaired."""
    if not isinstance(payload, dict):
        issue = BriefIssue(code="SCHEMA_INVALID", message=f"expected a JSON object, got {type(payload).__name__}")
        return BriefValidationResult(valid=False, errors=[issue]), None
    try:
        output = SynthesisOutput.model_validate(payload)
    except ValidationError as exc:
        return BriefValidationResult(valid=False, errors=schema_issues(exc)), None
    brief = ExecutiveBrief(title=brief_title(packet), statements=output.statements,
                           validation_metadata=output.validation_metadata)
    return validate_brief(packet, brief), brief


def _model_name(client: object) -> str:
    return str(getattr(client, "model", type(client).__name__))


def _log(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True, default=str))


def synthesize(
    packet: BriefPacket,
    client: LLMClient,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> SynthesisResult:
    if not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError(f"max_attempts must be 1..{MAX_ATTEMPTS}")
    model = _model_name(client)
    attempts: list[AttemptRecord] = []
    corrections: list[BriefIssue] = []

    def finish(status: SynthesisStatus, errors: list[BriefIssue], brief: ExecutiveBrief | None = None):
        served = getattr(client, "last_served_model", None)
        _log("synthesis_result", model=model, served_model=served, timestamp=now().isoformat(),
             packet_hash=packet.packet_hash, prompt_version=PROMPT_VERSION, attempt_count=len(attempts),
             validation=status.value, error_codes=sorted({e.code for e in errors}))
        return SynthesisResult(status=status, brief=brief, errors=errors, attempts=attempts, model=model,
                               packet_hash=packet.packet_hash, prompt_version=PROMPT_VERSION)

    for n in range(1, max_attempts + 1):
        try:
            payload: Any = client.generate_structured_brief(packet, corrections=corrections)
            if isinstance(payload, str):
                payload = parse_model_output(payload)
        except LLMProviderError as exc:
            code = ("PROVIDER_TIMEOUT" if isinstance(exc, LLMTimeoutError)
                    else "PROVIDER_REFUSAL" if isinstance(exc, LLMRefusalError) else "PROVIDER_ERROR")
            issue = BriefIssue(code=code, message=str(exc))
            attempts.append(AttemptRecord(attempt=n, status="PROVIDER_ERROR", errors=[issue]))
            _log("synthesis_attempt", model=model, timestamp=now().isoformat(), packet_hash=packet.packet_hash,
                 prompt_version=PROMPT_VERSION, attempt=n, validation="PROVIDER_ERROR", error_codes=[issue.code])
            prior = attempts[-2].errors if len(attempts) > 1 else []
            return finish(SynthesisStatus.PROVIDER_ERROR, [*prior, issue])
        except MalformedOutputError as exc:
            result = BriefValidationResult(valid=False, errors=[BriefIssue(code="MALFORMED_OUTPUT", message=str(exc))])
            attempts.append(AttemptRecord(attempt=n, status=result.status, errors=result.errors, output=exc.raw))
            brief = None
        else:
            result, brief = evaluate_output(packet, payload)
            attempts.append(AttemptRecord(attempt=n, status=result.status, errors=result.errors, output=payload))
        _log("synthesis_attempt", model=model, timestamp=now().isoformat(), packet_hash=packet.packet_hash,
             prompt_version=PROMPT_VERSION, attempt=n, validation=result.status, error_codes=sorted(result.codes))
        if result.valid:
            return finish(SynthesisStatus.ACCEPTED, [], brief)
        corrections = list(result.errors)
    return finish(SynthesisStatus.SYNTHESIS_REJECTED, corrections)


# ------------------------------------------------------------------ cli ---

def render(result: SynthesisResult) -> str:
    lines = [
        "CAT Q2 2026 — Structured Synthesis",
        "",
        f"Model: {result.model}",
        f"Prompt version: {result.prompt_version}",
        f"Packet hash: {result.packet_hash}",
        f"Attempts: {len(result.attempts)}",
        f"Result: {result.status.value}",
    ]
    for a in result.attempts:
        lines.append(f"Attempt {a.attempt}: {a.status}")
        for e in a.errors:
            parts = " ".join(p for p in (e.subject_id, f"[{e.statement_id}]" if e.statement_id else None) if p)
            lines.append(f"- {e.code}: {parts + ' ' if parts else ''}{e.message}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one live structured synthesis on the brief packet.")
    parser.add_argument("--out", type=Path, help="write the full synthesis result JSON here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from fpa_engine.llm import AnthropicLLMClient, LLMConfigError, LLMSettings

    try:
        client = AnthropicLLMClient(LLMSettings.from_env())
    except LLMConfigError as exc:
        print(f"CONFIG_ERROR: {exc}", file=sys.stderr)
        return 2
    result = synthesize(build_brief_packet(), client)
    if result.brief is not None:
        print(result.brief.model_dump_json(indent=2))
    print(render(result))
    if args.out:
        args.out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return {SynthesisStatus.ACCEPTED: 0, SynthesisStatus.SYNTHESIS_REJECTED: 1}.get(result.status, 2)


if __name__ == "__main__":
    sys.exit(main())
