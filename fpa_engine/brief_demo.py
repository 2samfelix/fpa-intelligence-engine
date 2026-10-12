"""Local demo: build the brief packet and validate the hand-written brief fixtures. No model is called."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fpa_engine.brief import BriefPacket, ExecutiveBrief, build_brief_packet
from fpa_engine.brief_validator import BriefValidationResult, validate_brief_payload
from fpa_engine.loader import BENCHMARK_ROOT, DEFAULT_BENCHMARK, load_benchmark


def fixture_dir(name: str = DEFAULT_BENCHMARK) -> Path:
    return BENCHMARK_ROOT / name / "briefs"


def load_brief(path: Path) -> ExecutiveBrief:
    return ExecutiveBrief.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_fixture(packet: BriefPacket, path: Path) -> BriefValidationResult:
    return validate_brief_payload(packet, json.loads(path.read_text(encoding="utf-8")))


def render(packet: BriefPacket, results: dict[str, BriefValidationResult]) -> str:
    lines = [
        "CAT Q2 2026 — Brief Contract",
        "",
        f"Packet facts: {len(packet.facts)}",
        f"Narratives: {len(packet.narratives)}",
        f"Reconciliations: {len(packet.reconciliations)}",
        f"Open flags: {len(packet.open_flags)}",
        f"Forbidden values: {len(packet.forbidden_values)}",
        f"Packet hash: {packet.packet_hash[:16]}",
    ]
    for name, result in results.items():
        lines += ["", f"{name}:", result.status]
        for e in result.errors:
            parts = [p for p in (e.subject_id, f"[{e.statement_id}]" if e.statement_id else None) if p]
            lines.append(f"- {e.code}: {' '.join(parts)} {e.message}".replace(":  ", ": "))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the brief packet and validate brief fixtures.")
    parser.add_argument("briefs", nargs="*", type=Path, help="brief JSON files (default: all bundled fixtures)")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--dump-packet", type=Path, help="write the packet JSON to this path")
    args = parser.parse_args(argv)

    packet = build_brief_packet(load_benchmark(args.benchmark))
    if args.dump_packet:
        args.dump_packet.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
    paths = args.briefs or sorted(fixture_dir(args.benchmark).glob("*.json"))
    results = {p.name: validate_fixture(packet, p) for p in paths}
    print(render(packet, results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
