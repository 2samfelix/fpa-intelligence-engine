"""Gate C4 step 2: unsourced analytical assumptions are first-class, declared, interpretive-only and disclosed."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from fpa_engine.brief import BriefStatement, PacketBuildError, build_brief_packet, load_policy
from fpa_engine.brief_demo import fixture_dir
from fpa_engine.brief_validator import assumption_dependencies, validate_statements
from fpa_engine.brief_validator import _Context as ValidatorContext
from fpa_engine.llm import MockLLMClient
from fpa_engine.loader import BenchmarkLoadError, load_benchmark, parse_benchmark, read_raw
from fpa_engine.loader import BENCHMARK_ROOT, DEFAULT_BENCHMARK
from fpa_engine.schemas import AssumptionSourcing, AssumptionUse
from fpa_engine.synthesizer import MAX_ATTEMPTS, PROMPT_VERSION, SynthesisStatus, synthesize

T0 = datetime(2026, 10, 12, tzinfo=timezone.utc)
ASSUMPTION_CODES = {"HIDDEN_ASSUMPTION", "ASSUMPTION_AS_FACT", "MISSING_ASSUMPTION_DISCLOSURE",
                    "MISSING_ASSUMPTION_LIMITATION", "UNKNOWN_ASSUMPTION_ID"}


def st(sid: str, kind: str, text: str, **ids) -> BriefStatement:
    return BriefStatement(id=sid, kind=kind, text=text, **ids)


def codes(packet, *statements) -> set[str]:
    return validate_statements(packet, list(statements)).codes


# A limitation that covers N03's review status and discloses AS01.
DISCLOSED = dict(evidence_ids=["N03"], assumption_ids=["AS01"])
LIMITATION = st("L", "limitation", "Reading the order of mention as a ranking by size rests on an unsourced "
                                   "assumption, and the driver-order evidence still needs review.", **DISCLOSED)


# ---------------------------------------------------------------- packet ---

def test_assumption_entities_appear_in_the_packet(packet):
    am = packet.assumption_map
    assert set(am) == {"AS01", "AS02"}
    as01 = am["AS01"]
    assert as01.text == "Caterpillar lists volume drivers in order of magnitude."
    assert (as01.sourcing, as01.allowed_use, as01.must_disclose, as01.introduced_by) == (
        AssumptionSourcing.unsourced, AssumptionUse.interpretive_only, True, "N03")
    assert am["AS02"].introduced_by == "N06" and am["AS02"].sourcing is AssumptionSourcing.unsourced
    assert am["AS02"].text == ("Any undisclosed Construction Industries share of the $392M IEEPA tariff recoveries "
                               "is not large enough to materially change the reported CI segment margin.")
    assert "corporate items" not in am["AS02"].text


def test_unsourced_status_survives_packet_building(packet):
    payload = packet.model_dump(mode="json")
    assert all(a["sourcing"] == "unsourced" for a in payload["assumptions"])
    assert "Caterpillar lists volume drivers in order of magnitude." in json.dumps(payload)


def test_workbook_provenance_does_not_leak(packet, benchmark):
    as01 = next(a for a in benchmark.assumptions if a.id == "AS01")
    assert "Analysis!A31" in as01.provenance
    text = packet.model_dump_json()
    assert "A31" not in text and "provenance" not in text and "Gate C3" not in text


def test_narratives_reference_assumptions_by_id(packet):
    nm = packet.narrative_map
    assert nm["N03"].assumption_ids == ["AS01"] and nm["N06"].assumption_ids == ["AS02"]
    assert nm["N15"].assumption_ids == [] and nm["N15"].inherited_assumption_ids == ["AS01"]
    assert "assumptions" not in type(nm["N03"]).model_fields


def test_assumption_text_that_cites_the_workbook_blocks_the_packet(benchmark):
    bad = [a.model_copy(update={"text": a.text + " See Analysis!A31."}) if a.id == "AS01" else a
           for a in benchmark.assumptions]
    with pytest.raises(PacketBuildError, match="ASSUMPTION_TEXT_INVALID"):
        build_brief_packet(replace(benchmark, assumptions=tuple(bad)), generated_at=T0)


def test_inference_markers_are_policy_owned(benchmark, packet):
    assert {"we interpret", "we read", "suggests", "consistent with", "inference", "may indicate",
            "points to"} <= set(packet.inference_policy.assumption_inference_markers)
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["inference_policy"]["assumption_inference_markers"] = ["we interpret"]
    strict = build_brief_packet(benchmark, generated_at=T0, policy=policy)
    s = st("A", "interpretive", "We read the reversal as consistent with a shift in emphasis.",
           premise_ids=["N03"], assumption_ids=["AS01"])
    assert codes(packet, s, LIMITATION) == set()
    assert codes(strict, s, LIMITATION) == {"MISSING_ASSUMPTION_DISCLOSURE"}


# ---------------------------------------------------------------- loader ---

def _raw():
    return read_raw(BENCHMARK_ROOT / DEFAULT_BENCHMARK)


def test_dangling_assumption_reference_fails_to_load():
    raw = _raw()
    n03 = next(n for n in raw["narratives"] if n["id"] == "N03")
    n03["assumption_ids"] = ["AS99"]
    with pytest.raises(BenchmarkLoadError, match="unresolved assumption reference AS99"):
        parse_benchmark("x", raw)


def test_introducer_must_list_its_assumption():
    raw = _raw()
    raw["assumptions"][0]["introduced_by"] = "N01"
    with pytest.raises(BenchmarkLoadError, match="AS01: introduced_by N01"):
        parse_benchmark("x", raw)


def test_explicit_evidence_cannot_carry_assumptions():
    raw = _raw()
    n01 = next(n for n in raw["narratives"] if n["id"] == "N01")
    n01["assumption_ids"] = ["AS01"]
    with pytest.raises(BenchmarkLoadError, match="explicit evidence cannot rest on assumptions"):
        parse_benchmark("x", raw)


def test_benchmark_loads_assumptions():
    assert [a.id for a in load_benchmark().assumptions] == ["AS01", "AS02"]


# ------------------------------------------------------- dependency chain ---

@pytest.mark.parametrize("ids, expected", [
    (dict(premise_ids=["N03"]), {"AS01"}),
    (dict(premise_ids=["N15"]), {"AS01"}),
    (dict(evidence_ids=["N03"]), {"AS01"}),
    (dict(premise_ids=["N06"]), {"AS02"}),
    (dict(premise_ids=["N01", "N02"]), set()),
    (dict(premise_ids=["N16"]), set()),
    (dict(premise_ids=["S23", "S25", "D07"]), set()),
])
def test_dependencies_follow_references_not_prose(packet, ids, expected):
    s = st("A", "interpretive", "Caterpillar lists volume drivers in order of magnitude.", **ids)
    assert assumption_dependencies(s, ValidatorContext(packet)) == expected


# ------------------------------------------------------------- validator ---

def test_valid_interpretive_statement_with_disclosure_passes(packet):
    s = st("A", "interpretive", "We read the reversal in order of mention as consistent with a shift in emphasis.",
           premise_ids=["N03"], assumption_ids=["AS01"])
    assert codes(packet, s, LIMITATION) == set()


def test_undeclared_dependency_is_a_hidden_assumption(packet):
    s = st("A", "interpretive", "We read the reversal in order of mention as consistent with a shift in emphasis.",
           premise_ids=["N03"])
    assert codes(packet, s, LIMITATION) == {"HIDDEN_ASSUMPTION"}


def test_inherited_dependency_is_a_hidden_assumption(packet):
    s = st("A", "interpretive", "The Q1 to Q2 2026 pattern is consistent with healthier demand.", premise_ids=["N15"])
    lim = st("L2", "limitation", "The demand-quality thesis still needs review.", evidence_ids=["N15"])
    assert codes(packet, s, LIMITATION, lim) == {"HIDDEN_ASSUMPTION"}


def test_assumption_in_a_fact_statement(packet):
    s = st("A", "fact", "Caterpillar lists its drivers in order of magnitude.", fact_ids=["S23"],
           assumption_ids=["AS01"])
    assert codes(packet, s, LIMITATION) == {"ASSUMPTION_AS_FACT"}


def test_assumption_reached_through_a_fact_statement_premise(packet):
    s = st("A", "fact", "Dealer inventories were the larger Q1 driver because Caterpillar listed them first.",
           fact_ids=["S23"], premise_ids=["N03"])
    assert codes(packet, s, LIMITATION) == {"HIDDEN_ASSUMPTION", "ASSUMPTION_AS_FACT"}


@pytest.mark.parametrize("text", [
    "Caterpillar lists its drivers in order of magnitude.",
    "Dealer inventories were the larger Q1 driver because Caterpillar listed them first.",
])
def test_assumption_in_an_explicit_claim(packet, text):
    s = st("A", "explicit_claim", text, evidence_ids=["N01"], assumption_ids=["AS01"])
    assert codes(packet, s, LIMITATION) == {"ASSUMPTION_AS_FACT"}


@pytest.mark.parametrize("text, extra", [
    ("Management's ordering shows dealer inventory was the larger driver.", {"INTERPRETIVE_MARKER_MISSING"}),
    ("Dealer inventories were the larger Q1 driver because Caterpillar listed them first.",
     {"INTERPRETIVE_MARKER_MISSING"}),
    ("Dealer inventory was likely the larger Q1 driver.", set()),
])
def test_declared_assumption_without_inference_framing(packet, text, extra):
    """A later limitation does not rescue an unqualified statement, and generic hedges are not enough."""
    s = st("A", "interpretive", text, premise_ids=["N03"], assumption_ids=["AS01"])
    assert codes(packet, s, LIMITATION) == {"MISSING_ASSUMPTION_DISCLOSURE"} | extra


@pytest.mark.parametrize("text", [
    "We interpret the reversal in order of mention as a shift in emphasis.",
    "We read the reversal in order of mention as a shift in emphasis.",
    "The reversal in order of mention suggests a shift in emphasis.",
    "By inference, the reversal in order of mention reflects a shift in emphasis.",
    "The reversal in order of mention may indicate a shift in emphasis.",
    "The reversal in order of mention points to a shift in emphasis.",
])
def test_each_policy_marker_discloses_the_inference(packet, text):
    s = st("A", "interpretive", text, premise_ids=["N03"], assumption_ids=["AS01"])
    assert codes(packet, s, LIMITATION) == set()


def test_missing_assumption_limitation_is_rejected(packet):
    s = st("A", "interpretive", "We read the reversal in order of mention as consistent with a shift in emphasis.",
           premise_ids=["N03"], assumption_ids=["AS01"])
    review_only = st("L", "limitation", "The driver-order evidence still needs review.", evidence_ids=["N03"])
    assert codes(packet, s, review_only) == {"MISSING_ASSUMPTION_LIMITATION"}


def test_both_disclosure_levels_are_required(packet):
    unframed = st("A", "interpretive", "Dealer inventory was likely the larger Q1 driver.", premise_ids=["N03"],
                  assumption_ids=["AS01"])
    review_only = st("L", "limitation", "The driver-order evidence still needs review.", evidence_ids=["N03"])
    assert codes(packet, unframed, review_only) == {"MISSING_ASSUMPTION_DISCLOSURE", "MISSING_ASSUMPTION_LIMITATION"}


def test_a_limitation_may_describe_an_assumption(packet):
    assert codes(packet, LIMITATION) == set()


def test_unknown_assumption_id_is_rejected(packet):
    s = st("A", "interpretive", "We read this as a shift.", premise_ids=["N01"], assumption_ids=["AS99"])
    assert codes(packet, s) == {"UNKNOWN_ASSUMPTION_ID"}


@pytest.mark.parametrize("text, premises", [
    ("Cash flow appears not to confirm the stocking thesis.", ["N16"]),
    ("Ex-tariff margin likely overstates underlying performance.", ["N17"]),
    ("The gap appears to have narrowed, a directional reading.", ["D07", "D08"]),
])
def test_unrelated_interpretive_statements_need_no_assumption(packet, text, premises):
    s = st("A", "interpretive", text, premise_ids=premises)
    result = validate_statements(packet, [s])
    assert not result.codes & ASSUMPTION_CODES, result.codes


def test_ieepa_assumption_is_enforced_the_same_way(packet):
    s = st("A", "interpretive", "CI segment margin is not materially inflated by the recoveries.",
           premise_ids=["N06"])
    assert {"HIDDEN_ASSUMPTION", "MISSING_ASSUMPTION_LIMITATION"} <= codes(packet, s)


def test_existing_valid_brief_declares_its_assumption_and_passes(packet):
    from fpa_engine.brief import ExecutiveBrief
    from fpa_engine.brief_validator import validate_brief

    brief = ExecutiveBrief.model_validate(json.loads((fixture_dir() / "valid_brief.json").read_text(encoding="utf-8")))
    v4 = next(s for s in brief.statements if s.id == "V4")
    assert v4.assumption_ids == ["AS01"]
    assert validate_brief(packet, brief).valid


def test_briefs_without_assumption_dependency_are_unaffected(packet):
    s = st("A", "fact", "Disclosed CI dealer inventory rose $1.5 billion in Q1 2026.", fact_ids=["S23"])
    explicit = st("B", "explicit_claim", "Caterpillar's Q2 2026 release named end users first.", evidence_ids=["N02"])
    assert codes(packet, s, explicit) == set()


# ------------------------------------------------------------ synthesis ---

def _model_output(name: str = "valid_brief.json") -> dict:
    brief = json.loads((fixture_dir() / name).read_text(encoding="utf-8"))
    del brief["title"]
    brief["validation_metadata"]["author"] = PROMPT_VERSION
    return brief


def _hide_assumption(brief: dict) -> dict:
    bad = copy.deepcopy(brief)
    for s in bad["statements"]:
        if s["id"] == "V4":
            del s["assumption_ids"]
    return bad


def test_hidden_assumption_is_corrected_in_the_single_retry(packet):
    good = _model_output()
    client = MockLLMClient([_hide_assumption(good), good])
    result = synthesize(packet, client, now=lambda: T0)
    assert result.accepted and [a.status for a in result.attempts] == ["REJECTED", "VALID"]
    assert [e.code for e in client.calls[1].corrections] == ["HIDDEN_ASSUMPTION"]


def test_retry_limit_is_unchanged(packet):
    bad = _hide_assumption(_model_output())
    client = MockLLMClient([bad, bad, bad])
    result = synthesize(packet, client, now=lambda: T0)
    assert MAX_ATTEMPTS == 2
    assert result.status is SynthesisStatus.SYNTHESIS_REJECTED and len(client.calls) == 2
    assert {e.code for e in result.errors} == {"HIDDEN_ASSUMPTION"}
