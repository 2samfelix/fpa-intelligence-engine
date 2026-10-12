from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal

from fpa_engine.calculator import calculate_all, materialize
from fpa_engine.loader import DEFAULT_BENCHMARK, Benchmark, load_benchmark
from fpa_engine.reconciliation import evaluate_all
from fpa_engine.schemas import (
    BridgeType,
    CalcStatus,
    CalculatedValue,
    FinancialFact,
    ReconciliationResult,
    Unit,
    ValidationReport,
)
from fpa_engine.validator import validate_inputs


class ValidationFailed(Exception):
    def __init__(self, report: ValidationReport):
        self.report = report
        lines = [f"{i.code} {i.subject_id}: {i.message}" for i in report.errors]
        super().__init__("input validation failed:\n  " + "\n  ".join(lines))


@dataclass(frozen=True)
class PipelineResult:
    benchmark: Benchmark
    validation: ValidationReport
    calculations: dict[str, CalculatedValue]
    facts: dict[str, FinancialFact]
    reconciliations: list[ReconciliationResult]


def run(benchmark: Benchmark | None = None, strict: bool = True) -> PipelineResult:
    bm = benchmark or load_benchmark()
    report = validate_inputs(bm)
    if strict and not report.ok:
        raise ValidationFailed(report)
    calculations = calculate_all(bm.fact_map)
    facts = materialize(bm.fact_map, calculations)
    reconciliations = evaluate_all(bm.bridges, facts)
    return PipelineResult(bm, report, calculations, facts, reconciliations)


def _num(value: Decimal, places: int = 2) -> str:
    q = value.quantize(Decimal(1).scaleb(-places))
    text = f"{q:,.{places}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _signed_usd(value: Decimal) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}${_num(abs(value))}M"


def _format_value(fact: FinancialFact) -> str:
    if fact.value is None:
        return "n/a"
    if fact.unit is Unit.usd_m:
        return f"{'-' if fact.value < 0 else ''}${_num(abs(fact.value))}M"
    suffix = {Unit.pct: "%", Unit.pts: " pts", Unit.bps: " bps"}[fact.unit]
    return f"{_num(fact.value)}{suffix}"


def summarize(result: PipelineResult) -> str:
    bm, report = result.benchmark, result.validation
    statuses = Counter(c.status for c in result.calculations.values())
    lines = [
        f"CAT Q2 2026 — Construction Industries  [{bm.name}]",
        "",
        f"Facts loaded: {len(bm.sourced)} sourced, {len(bm.derived)} derived definitions, {len(bm.gaps)} known gaps",
        (f"Derived calculations: {statuses[CalcStatus.computed]} computed, "
         f"{statuses[CalcStatus.unavailable]} unavailable, {statuses[CalcStatus.not_computable]} not computable"),
        (f"Validation: {'PASS' if report.ok else 'FAIL'} "
         f"({len(report.errors)} errors, {len(report.warnings)} warnings)"),
    ]
    for issue in report.errors:
        lines.append(f"  ERROR {issue.code} {issue.subject_id}: {issue.message}")
    if report.warnings:
        counts = Counter(w.code for w in report.warnings)
        lines.append("  warnings: " + ", ".join(f"{code} x{n}" for code, n in sorted(counts.items())))

    lines += ["", "Derived facts:"]
    for fact in result.benchmark.derived:
        calc = result.calculations[fact.id]
        shown = _format_value(result.facts[fact.id])
        extra = f" (+/-{_num(calc.uncertainty)})" if calc.uncertainty else ""
        note = "" if calc.status is CalcStatus.computed else f"  [{calc.status.value}: {calc.reason}]"
        lines.append(f"  {fact.id} {fact.metric} {fact.period}: {shown}{extra}{note}")

    lines += ["", "Reconciliations:"]
    for r in result.reconciliations:
        lines.append(f"  {r.id} {r.bridge_name} {r.period}: {r.status.value}")
        if r.bridge_type is not BridgeType.non_additive_set and r.residual is not None:
            flags = [f for f, on in (("circular", r.circular), ("precision-limited", r.precision_limited)) if on]
            allowed = _signed_usd(r.tolerance.allowed).lstrip("+") if r.tolerance else "n/a"
            lines.append(f"    Residual: {_signed_usd(r.residual)} (allowed {allowed})"
                         + (f" [{', '.join(flags)}]" if flags else ""))
        lines.append(f"    {r.reason}")
    return "\n".join(lines)


def to_json(result: PipelineResult) -> dict:
    return {
        "benchmark": result.benchmark.name,
        "validation": {
            "ok": result.validation.ok,
            "errors": [i.model_dump(mode="json") for i in result.validation.errors],
            "warnings": [i.model_dump(mode="json") for i in result.validation.warnings],
        },
        "derived": {fid: c.model_dump(mode="json") for fid, c in result.calculations.items()},
        "reconciliations": [r.model_dump(mode="json") for r in result.reconciliations],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic FP&A benchmark engine.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args(argv)
    try:
        result = run(load_benchmark(args.benchmark))
    except ValidationFailed as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(to_json(result), indent=2) if args.json else summarize(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
