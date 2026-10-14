"""Gate C2: N independent live syntheses of one pinned packet, prompt and model, plus an evaluation report.

Refuses to start unless the packet hash and prompt version match the gate's pins. Each run is a fresh client and a
fresh `synthesize` call (at most one corrective retry, enforced by the synthesizer). The report is built from an
explicit allow-list of fields, never from client or request objects, so it carries no key and no request headers.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fpa_engine.brief import BriefPacket, StatementKind, build_brief_packet
from fpa_engine.brief_validator import validate_brief
from fpa_engine.synthesizer import PROMPT_VERSION, LLMClient, SynthesisResult, SynthesisStatus, synthesize

EXPECTED_PACKET_HASH = "d14c0d3d63beeba2ac2bfdd2f4c7ef8337d376adee2aef1a2c56e45ac3abd8d8"
EXPECTED_PROMPT_VERSION = "synthesis_v2"
MAX_RUNS = 5
GATE_RUNS = 5  # the gate passes only on GATE_RUNS of GATE_RUNS finally accepted
NUMBER_CODES = ("UNSUPPORTED_NUMBER", "FORBIDDEN_VALUE", "ARITHMETIC_IN_PROSE")
_SECRET_MARKERS = ("x-api-key", "authorization: bearer", "anthropic_custom_headers", "anthropic_api_key")


class GateError(Exception):
    pass


def preflight(packet: BriefPacket, runs: int) -> None:
    problems = []
    if packet.packet_hash != EXPECTED_PACKET_HASH:
        problems.append(f"packet hash {packet.packet_hash} != pinned {EXPECTED_PACKET_HASH}")
    if PROMPT_VERSION != EXPECTED_PROMPT_VERSION:
        problems.append(f"prompt version {PROMPT_VERSION} != pinned {EXPECTED_PROMPT_VERSION}")
    if not 1 <= runs <= MAX_RUNS:
        problems.append(f"runs must be 1..{MAX_RUNS}, got {runs}")
    if problems:
        raise GateError("; ".join(problems))


def banner(model: str, packet: BriefPacket, runs: int) -> str:
    return (f"Gate C2 | configured model: {model} | packet hash: {packet.packet_hash} | "
            f"prompt version: {PROMPT_VERSION} | runs: {runs}")


def _mandatory(packet: BriefPacket) -> list[str]:
    sp = packet.surfacing_policy
    return [*sp.mandatory_reconciliation_ids, *sp.mandatory_disclosure_gap_ids, *sp.mandatory_comparability_note_ids]


def run_record(n: int, result: SynthesisResult, client: LLMClient, packet: BriefPacket) -> dict[str, Any]:
    brief = result.brief
    statements = brief.statements if brief else []

    def cited(field: str, *, limitations_only: bool = False) -> list[str]:
        return sorted({i for s in statements if not limitations_only or s.kind is StatementKind.limitation
                       for i in getattr(s, field)})

    surfaced = sorted({*cited("reconciliation_ids", limitations_only=True), *cited("flag_ids", limitations_only=True),
                       *cited("comparability_note_ids", limitations_only=True)})
    # Independent re-check of the accepted brief with the unchanged validator; must be clean by construction.
    recheck = validate_brief(packet, brief).codes if brief else set()
    attempt_codes = [Counter(e.code for e in a.errors) for a in result.attempts]
    served = getattr(client, "last_served_model", None)
    return {
        "run": n,
        "configured_model": result.model,
        "served_model": served,
        "model_match": served == result.model,
        "packet_hash": result.packet_hash,
        "prompt_version": result.prompt_version,
        "attempt_1_status": result.attempts[0].status if result.attempts else None,
        "attempt_1_errors": [e.model_dump() for e in result.attempts[0].errors] if result.attempts else [],
        "retry_used": len(result.attempts) > 1,
        "final_status": result.status.value,
        "final_errors": [e.model_dump() for e in result.errors],
        "attempts": [{"attempt": a.attempt, "status": a.status, "error_codes": dict(c)}
                     for a, c in zip(result.attempts, attempt_codes)],
        "statement_count": len(statements),
        "fact_ids": cited("fact_ids"),
        "evidence_ids": cited("evidence_ids"),
        "reconciliation_ids": cited("reconciliation_ids"),
        "limitation_ids_surfaced": surfaced,
        "missing_mandatory_limitations": sorted(set(_mandatory(packet)) - set(surfaced)) if brief else None,
        "number_issues_by_attempt": [{k: c.get(k, 0) for k in NUMBER_CODES} for c in attempt_codes],
        "accepted_brief_recheck_codes": sorted(recheck),
        "forbidden_value_count": sum(c.get("FORBIDDEN_VALUE", 0) for c in attempt_codes),
        "unsupported_number_count": sum(c.get("UNSUPPORTED_NUMBER", 0) for c in attempt_codes),
        "brief": brief.model_dump(mode="json") if brief else None,
    }


def summarize(records: list[dict[str, Any]], packet: BriefPacket) -> dict[str, Any]:
    n = len(records)
    accepted = [r for r in records if r["final_status"] == SynthesisStatus.ACCEPTED.value]
    code_runs: Counter[str] = Counter()
    code_total: Counter[str] = Counter()
    for r in records:
        seen = set()
        for a in r["attempts"]:
            code_total.update(a["error_codes"])
            seen |= set(a["error_codes"])
        code_runs.update(seen)

    def ids_across(field: str) -> dict[str, list[str]]:
        sets = [set(r[field]) for r in accepted]
        return {"in_every_accepted_run": sorted(set.intersection(*sets)) if sets else [],
                "in_any_accepted_run": sorted(set.union(*sets)) if sets else []}

    criteria = {
        f"{GATE_RUNS}_of_{GATE_RUNS}_final_accepted": n == GATE_RUNS and len(accepted) == GATE_RUNS,
        "zero_missing_mandatory_limitations": all(not r["missing_mandatory_limitations"] for r in accepted),
        "zero_model_mismatches": all(r["model_match"] for r in records),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packet_hash": packet.packet_hash,
        "prompt_version": PROMPT_VERSION,
        "runs": n,
        "configured_models": sorted({r["configured_model"] for r in records}),
        "served_models": sorted({str(r["served_model"]) for r in records}),
        "gate_criteria": criteria,
        "gate_c2_passed": all(criteria.values()),
        "report_only": {
            "first_pass_acceptance_rate": sum(r["attempt_1_status"] == "VALID" for r in records) / n,
            "retry_rate": sum(r["retry_used"] for r in records) / n,
            "recurring_validator_errors": sorted(c for c, k in code_runs.items() if k >= 2),
        },
        "final_acceptance_rate": len(accepted) / n,
        "hard_rejection_rate": sum(r["final_status"] == SynthesisStatus.SYNTHESIS_REJECTED.value for r in records) / n,
        "provider_error_rate": sum(r["final_status"] == SynthesisStatus.PROVIDER_ERROR.value for r in records) / n,
        "validator_errors": {code: {"runs": code_runs[code], "occurrences": code_total[code]}
                             for code in sorted(code_total)},
        "mandatory_limitations": _mandatory(packet),
        "number_issues_any_attempt": {k: sum(a[k] for r in records for a in r["number_issues_by_attempt"])
                                      for k in NUMBER_CODES},
        "number_issues_in_accepted_briefs": sorted({c for r in accepted for c in r["accepted_brief_recheck_codes"]
                                                    if c in NUMBER_CODES}),
        "consistency": {f: ids_across(f) for f in ("fact_ids", "reconciliation_ids", "limitation_ids_surfaced")},
        "per_run": [{k: r[k] for k in ("run", "final_status", "attempt_1_status", "retry_used", "statement_count",
                                       "served_model", "forbidden_value_count", "unsupported_number_count")}
                    for r in records],
    }


def _write_json(path: Path, payload: dict[str, Any], secrets: tuple[str, ...]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    lowered = text.lower()
    if any(s and s in text for s in secrets) or any(m in lowered for m in _SECRET_MARKERS):
        raise GateError(f"refusing to write {path.name}: it would contain a credential or header marker")
    path.write_text(text + "\n", encoding="utf-8")


def evaluate(runs: int, out_dir: Path, make_client: Callable[[], LLMClient], secrets: tuple[str, ...] = ()) -> dict:
    packet = build_brief_packet()
    preflight(packet, runs)
    out_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for n in range(1, runs + 1):
        client = make_client()
        result = synthesize(packet, client)
        record = run_record(n, result, client, packet)
        _write_json(out_dir / f"run_{n:02d}.json", record, secrets)
        records.append(record)
    summary = summarize(records, packet)
    _write_json(out_dir / "summary.json", summary, secrets)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate C2: N independent live syntheses of the pinned packet.")
    parser.add_argument("--runs", type=int, default=1, help=f"number of independent syntheses (1..{MAX_RUNS})")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/live_synthesis") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args(argv)

    from fpa_engine.llm import AnthropicLLMClient, LLMConfigError, LLMSettings

    try:
        packet = build_brief_packet()
        preflight(packet, args.runs)  # before any client exists
        settings = LLMSettings.from_env()  # read once: every run uses the same model and settings
        print(banner(settings.model, packet, args.runs), flush=True)
        summary = evaluate(args.runs, args.out, lambda: AnthropicLLMClient(settings), (settings.api_key,))
    except (GateError, LLMConfigError, FileExistsError) as exc:
        print(f"GATE_C2_REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"runs": summary["runs"], "final_acceptance_rate": summary["final_acceptance_rate"],
                      "gate_criteria": summary["gate_criteria"], "gate_c2_passed": summary["gate_c2_passed"],
                      "report_only": summary["report_only"]}, indent=2))
    print(f"report: {args.out / 'summary.json'}")
    return 0 if summary["gate_c2_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
