from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from fpa_engine.brief import BriefPacket, OpenFlagKind, PacketBuildError, PacketFact, build_brief_packet, load_policy
from fpa_engine.schemas import DisclosureStatus, Origin, ReconciliationStatus

T0 = datetime(2026, 10, 12, tzinfo=timezone.utc)
FORBIDDEN_KEYS = {"benchmark_cell", "formula", "inputs", "excerpt", "comparability_notes_raw", "expected"}


def _all_keys(obj) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in _all_keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in _all_keys(v)}
    return set()


def test_packet_counts(packet):
    assert len(packet.facts) == 65
    assert sum(f.origin is Origin.sourced for f in packet.facts) == 40
    assert sum(f.origin is Origin.derived for f in packet.facts) == 25
    assert len(packet.narratives) == 17
    assert len(packet.reconciliations) == 13
    kinds = [f.kind for f in packet.open_flags]
    assert kinds.count(OpenFlagKind.disclosure_gap) == 7
    assert kinds.count(OpenFlagKind.review_required_narrative) == 6
    assert kinds.count(OpenFlagKind.unresolved_reconciliation) == 3
    assert OpenFlagKind.unavailable_fact not in kinds
    assert len(packet.comparability_notes) == 13
    assert len(packet.forbidden_values) == 5


def _free_text(packet) -> list[str]:
    texts = [n.text for n in packet.comparability_notes]
    for n in packet.narratives:
        texts += [n.claim, n.verbatim_quote or "", *n.assumptions, *n.inherited_assumptions]
    texts += [f.message for f in packet.open_flags]
    texts += [fv.reason for fv in packet.forbidden_values]
    return texts


def test_packet_excludes_workbook_material_and_formulas(packet):
    payload = packet.model_dump(mode="json")
    assert not (_all_keys(payload) & FORBIDDEN_KEYS)
    without_urls = json.dumps({**payload, "sources": [{**s, "url": ""} for s in payload["sources"]]})
    assert not re.search(r"[A-Z][A-Za-z ]+![A-Z]+\d+", without_urls), "workbook cell reference leaked"
    for text in _free_text(packet):
        assert "workbook" not in text.lower(), text
        assert not re.search(r"\d\s*/\s*\d", text), f"arithmetic expression leaked: {text}"


def test_packet_fact_model_has_no_formula_fields():
    fields = set(PacketFact.model_fields)
    assert {"formula", "inputs", "benchmark_cell", "comparability_notes"}.isdisjoint(fields)
    assert {"value", "display", "precision", "max_decimals", "flags", "derived_from"} <= fields


def test_narrative_quotes_are_the_validated_excerpts(packet, benchmark):
    smap = benchmark.source_map
    for n in packet.narratives:
        if n.verbatim_quote:
            assert any(n.verbatim_quote in (smap[s].excerpt or "") for s in n.source_ids), n.id


def test_packet_values_match_engine_outputs(packet, result):
    for f in packet.facts:
        assert f.value == result.facts[f.id].value


def test_display_and_precision_metadata(packet):
    fm = packet.fact_map
    assert (fm["S23"].display, fm["S23"].precision, fm["S23"].max_decimals) == ("$1,500M", Decimal("100"), 0)
    assert (fm["S17"].display, fm["S17"].max_decimals) == ("23.3%", 1)
    assert (fm["D03"].display, fm["D03"].max_decimals) == ("34.8%", 2)
    assert fm["D07"].display == "7.8 pts" and fm["D07"].uncertainty == Decimal("0.5")
    assert fm["D09"].display == "320 bps" and fm["D09"].uncertainty == Decimal("10")


def test_d15_stays_derived_not_disclosed(packet):
    d15 = packet.fact_map["D15"]
    assert d15.origin is Origin.derived
    assert "derived_not_disclosed" in d15.flags
    assert d15.source_ids == []
    assert d15.display == "$284M"
    assert "CN10" in d15.comparability_note_ids
    assert packet.fact_map["S29"].value == Decimal("340")


def test_material_flags_require_limitations(packet):
    required = {f.id for f in packet.facts if f.requires_limitation}
    assert {"D07", "D08", "D20", "D21", "D16", "D17", "D24", "D11", "D12", "D22", "D14", "S24"} <= required
    assert {"S01", "S23", "S25", "D03", "D15"}.isdisjoint(required)
    assert set(packet.fact_map["D20"].flags) >= {"mixed_basis", "non_comparable"}


def test_reconciliations_carry_status_and_limitation_requirements(packet):
    rm = packet.reconciliation_map
    assert rm["R01"].status is ReconciliationStatus.RECONCILED and not rm["R01"].requires_limitation
    for rid, residual in (("R03", "12"), ("R04", "703"), ("R05", "14")):
        assert rm[rid].status is ReconciliationStatus.UNRESOLVED
        assert rm[rid].residual == Decimal(residual)
        assert rm[rid].requires_limitation
        assert f"UR-{rid}" in packet.flag_map
    assert rm["R04"].related_fact_ids == ["D11"]
    for i in range(1, 9):
        na = rm[f"NA0{i}"]
        assert na.status is ReconciliationStatus.NOT_APPLICABLE and na.residual is None and na.requires_limitation


def test_disclosure_gaps_and_review_narratives_surface_as_open_flags(packet):
    flags = packet.flag_map
    assert flags["G01"].related_ids == ["S24", "NA05"]
    assert set(flags["G04"].related_ids) == {"D11", "R04"}
    for nid in ("N03", "N06", "N09", "N13", "N14", "N15"):
        assert f"RV-{nid}" in flags
    n15 = packet.narrative_map["N15"]
    assert n15.inherited_assumptions and n15.requires_limitation


def test_forbidden_values_cover_gate_b1_rejections(packet, expected):
    singles = {fv.value for fv in packet.forbidden_values if fv.value is not None}
    for rejected in expected["rejected_values"]:
        assert Decimal(rejected["value"]) in singles
    assert {Decimal("343"), Decimal("570"), Decimal("3113")} <= singles
    ranges = {(fv.low, fv.high) for fv in packet.forbidden_values if fv.low is not None}
    assert (Decimal("100"), Decimal("150")) in ranges


def test_policy_ids_resolve_on_the_full_benchmark(packet):
    policy = load_policy("cat_q2_2026")
    known = set(packet.fact_map) | set(packet.reconciliation_map)
    for note in policy["comparability_notes"]:
        assert set(note["subject_ids"]) <= known, note["id"]
    for gid, links in policy["gap_links"].items():
        assert gid in packet.flag_map and set(links) <= known
    assert set(policy["material_flags"]) == set(packet.material_flags)


def test_packet_is_deterministic(benchmark, packet):
    again = build_brief_packet(benchmark, generated_at=T0)
    assert again == packet
    later = build_brief_packet(benchmark, generated_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert later.packet_hash == packet.packet_hash


def test_packet_round_trips_as_json(packet):
    assert BriefPacket.model_validate_json(packet.model_dump_json()) == packet


def test_missing_input_blocks_packet_build(benchmark):
    with pytest.raises(PacketBuildError, match="MISSING_INPUT"):
        build_brief_packet(benchmark.without_facts("S29"), generated_at=T0)


def test_not_computable_derived_facts_are_excluded_and_flagged(benchmark):
    s02 = benchmark.fact_map["S02"]
    degraded = build_brief_packet(benchmark.with_fact(s02.model_copy(update={"value": Decimal("0.4")})),
                                  generated_at=T0)
    for fid in ("D03", "D05", "D07"):
        assert fid not in degraded.fact_map
        assert f"UA-{fid}" in degraded.flag_map
    assert "D01" in degraded.fact_map


def test_invalid_benchmark_cannot_produce_a_packet(benchmark):
    s02 = benchmark.fact_map["S02"]
    bad = benchmark.with_fact(s02.model_copy(update={"disclosure_status": DisclosureStatus.back_solved}))
    with pytest.raises(PacketBuildError, match="BACK_SOLVED_AS_SOURCED"):
        build_brief_packet(bad, generated_at=T0)
