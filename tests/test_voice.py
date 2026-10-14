"""Gate C4 step 5: assertions, source versus analyst voice, verbatim quotes, attribution and characterization."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest

from fpa_engine.brief import BriefStatement, PacketBuildError, build_brief_packet, load_policy
from fpa_engine.brief_demo import fixture_dir
from fpa_engine.brief_validator import validate_statements
from fpa_engine.llm import MockLLMClient
from fpa_engine.loader import BENCHMARK_ROOT, DEFAULT_BENCHMARK, BenchmarkLoadError, parse_benchmark, read_raw
from fpa_engine.schemas import AssertionVoice, EvidenceType
from fpa_engine.synthesizer import PROMPT_VERSION, SynthesisStatus, synthesize

T0 = datetime(2026, 10, 12, tzinfo=timezone.utc)
VOICE_CODES = {"UNKNOWN_ASSERTION_ID", "ASSERTION_WITHOUT_EVIDENCE", "EXPLICIT_WITHOUT_ASSERTION",
               "SOURCE_VOICE_NOT_ATTRIBUTED", "ATTRIBUTION_WITHOUT_EVIDENCE", "QUOTE_NOT_VERBATIM",
               "QUOTE_NOT_ATTRIBUTED", "UNSUPPORTED_CHARACTERIZATION"}


def st(sid: str, kind: str, text: str, **ids) -> BriefStatement:
    return BriefStatement(id=sid, kind=kind, text=text, **ids)


def voice_codes(packet, kind: str, text: str, **ids) -> set[str]:
    return validate_statements(packet, [st("A", kind, text, **ids)]).codes & VOICE_CODES


def issues(packet, kind: str, text: str, **ids):
    return [e for e in validate_statements(packet, [st("A", kind, text, **ids)]).errors if e.code in VOICE_CODES]


# ----------------------------------------------------------------- schema ---

def test_every_narrative_carries_assertions(packet):
    assert len(packet.assertions) == 26
    for n in packet.narratives:
        assert n.assertion_ids, n.id
        assert all(packet.assertion_map[a].narrative_id == n.id for a in n.assertion_ids)


def test_interpretive_narratives_speak_only_in_analyst_voice(packet):
    for a in packet.assertions:
        if packet.narrative_map[a.narrative_id].evidence_type is EvidenceType.interpretive:
            assert a.voice is AssertionVoice.analyst and a.quote is None, a.id


def test_n07_separates_the_fact_from_the_cfo_characterization(packet):
    a1, a2 = packet.assertion_map["N07-A01"], packet.assertion_map["N07-A02"]
    assert a1.voice is AssertionVoice.analyst and "typical" not in a1.text and "slight" not in a1.text
    assert a2.voice is AssertionVoice.source and "more typical" in a2.quote
    assert a2.quote in packet.narrative_map["N07"].verbatim_quote


def test_quotes_are_excerpts_of_the_narrative_quote(packet):
    for a in packet.assertions:
        if a.quote:
            assert a.quote in packet.narrative_map[a.narrative_id].verbatim_quote, a.id


def _raw():
    return read_raw(BENCHMARK_ROOT / DEFAULT_BENCHMARK)


@pytest.mark.parametrize("change, message", [
    (lambda r: r["assertions"].append({"id": "N15-A09", "narrative_id": "N15", "text": "x", "voice": "source"}),
     "source voice needs explicit evidence"),
    (lambda r: r["assertions"].append({"id": "N07-A09", "narrative_id": "N07", "text": "x", "voice": "source",
                                       "quote": "a typical increase"}), "not an excerpt"),
    (lambda r: r["assertions"].append({"id": "N07-A09", "narrative_id": "N07", "text": "x", "voice": "analyst",
                                       "quote": "a more typical"}), "only a source-voice assertion may carry a quote"),
    (lambda r: r["assertions"].append({"id": "N99-A01", "narrative_id": "N99", "text": "x", "voice": "analyst"}),
     "unknown narrative N99"),
    (lambda r: r.__setitem__("assertions", [a for a in r["assertions"] if a["narrative_id"] != "N11"]),
     "narrative N11: has no assertion"),
])
def test_assertion_integrity_is_checked_at_load(change, message):
    raw = _raw()
    change(raw)
    with pytest.raises(BenchmarkLoadError, match=message):
        parse_benchmark("x", raw)


# ------------------------------------------------- characterization ---

def test_gate_c2_more_typical_in_analyst_voice_fails(packet):
    found = issues(packet, "fact", "The Q1 2026 CI build of $1.5 billion was a more typical first-quarter increase.",
                   fact_ids=["S23"], evidence_ids=["N07"], assertion_ids=["N07-A01"])
    assert [(e.code, e.subject_id) for e in found] == [("UNSUPPORTED_CHARACTERIZATION", "typical")]
    assert "N07-A02" in found[0].message


def test_citing_the_source_wording_without_attribution_fails(packet):
    assert voice_codes(packet, "fact", "The Q1 2026 CI build of $1.5 billion was a more typical first-quarter "
                       "increase.", fact_ids=["S23"], evidence_ids=["N07"],
                       assertion_ids=["N07-A02"]) == {"SOURCE_VOICE_NOT_ATTRIBUTED", "UNSUPPORTED_CHARACTERIZATION"}


def test_attributed_source_characterization_passes(packet):
    s = st("A", "explicit_claim", 'The CFO called the Q1 2026 build "a more typical $1.5 billion increase in the '
           'first quarter."', evidence_ids=["N07"], assertion_ids=["N07-A02"])
    assert validate_statements(packet, [s]).codes == set()


@pytest.mark.parametrize("text, ids", [
    ("Reported CI sales growth rose sharply to 34.8% in Q2 2026.", {"fact_ids": ["D03"]}),
    ("The prior-year quarter saw a slight decrease in the CI build.",
     {"fact_ids": ["S23"], "evidence_ids": ["N07"], "assertion_ids": ["N07-A01"]}),
    ("Retail demand was robust in Q2 2026, with growth of 22%.", {"fact_ids": ["S22"]}),
    ("The Q2 2026 dealer build of $400 million was unusually modest.", {"fact_ids": ["S25"]}),
])
def test_unsupported_characterization_fails(packet, text, ids):
    assert voice_codes(packet, "fact", text, **ids) == {"UNSUPPORTED_CHARACTERIZATION"}


def test_analyst_assertion_supports_its_own_wording(packet):
    text = "We read the Q1-to-Q2 2026 pattern as consistent with a mix that shifted materially toward end-user demand."
    assert voice_codes(packet, "interpretive", text, premise_ids=["N15"], assumption_ids=["AS01"]) == {
        "UNSUPPORTED_CHARACTERIZATION"}
    assert voice_codes(packet, "interpretive", text, premise_ids=["N15"], assumption_ids=["AS01"],
                       assertion_ids=["N15-A01"]) == set()


def test_declared_assumption_supports_interpretive_wording(packet):
    text = "We read any undisclosed CI share of the recoveries as not large enough to materially change CI margin."
    assert voice_codes(packet, "interpretive", text, premise_ids=["N06"], assumption_ids=["AS02"]) == set()


def test_limitations_may_describe_materiality(packet):
    assert voice_codes(packet, "limitation", "Whether the CI share is material is not disclosed.",
                       evidence_ids=["N06"]) == set()


def test_characterization_terms_are_policy_owned(benchmark):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["voice_policy"]["characterization_terms"].remove("sharply")
    relaxed = build_brief_packet(benchmark, generated_at=T0, policy=policy)
    assert voice_codes(relaxed, "fact", "Reported CI sales growth rose sharply to 34.8% in Q2 2026.",
                       fact_ids=["D03"]) == set()


# ------------------------------------------------------------------ quotes ---

@pytest.mark.parametrize("text", [
    'The Q2 2026 release said higher sales volume was "mainly driven by higher sales of equipment to end users".',
    'The Q2 2026 release said higher sales volume was “mainly driven by higher sales of equipment to end users.”',
    'The Q2 2026 release said volume was "Mainly driven by higher sales  of equipment to end users".',
])
def test_verbatim_quote_passes(packet, text):
    assert voice_codes(packet, "explicit_claim", text, evidence_ids=["N02"], assertion_ids=["N02-A02"]) == set()


def test_quote_with_ellipsis_passes(packet):
    assert voice_codes(packet, "explicit_claim", 'The CFO said CI margin was "impacted by ... 340 basis points within '
                       'Construction Industries".', evidence_ids=["N04"], assertion_ids=["N04-A02"]) == set()


@pytest.mark.parametrize("quoted, ok", [
    ("a more typical ... in the first quarter", True),
    ("in the first quarter ... a more typical", False),
    ("a more typical … first quarter … prior year", True),
    ("a more typical ... prior year ... first quarter", False),
    ("first quarter ... a more typical ... prior year", False),
    ("a slight decrease ... a slight decrease", False),
])
def test_ellipsis_fragments_must_keep_source_order(packet, quoted, ok):
    found = voice_codes(packet, "explicit_claim", f'The CFO called the Q1 2026 build "{quoted}".',
                        evidence_ids=["N07"], assertion_ids=["N07-A02"])
    assert found == (set() if ok else {"QUOTE_NOT_VERBATIM"})


@pytest.mark.parametrize("text, evidence", [
    ('The Q2 2026 release said volume was "mainly driven by end users".', "N02"),
    ('The Q2 2026 release said volume was "mainly driven by the impact from changes in dealer inventories and '
     'higher sales of equipment to end users".', "N02"),  # N01's wording under N02
    ('The release said retail is reported in "price-neutral" dollars.', "N02"),
])
def test_misquote_fails(packet, text, evidence):
    assert voice_codes(packet, "explicit_claim", text, evidence_ids=[evidence],
                       assertion_ids=[f"{evidence}-A02"]) == {"QUOTE_NOT_VERBATIM"}


def test_quote_from_unvalidated_evidence_fails(packet):
    found = voice_codes(packet, "explicit_claim", 'The release said costs "largely reflected the impact of higher '
                        'tariff costs".', evidence_ids=["N09"], assertion_ids=["N09-A01"])
    assert found == {"QUOTE_NOT_VERBATIM", "ATTRIBUTION_WITHOUT_EVIDENCE"}


def test_unattributed_quote_fails(packet):
    found = voice_codes(packet, "fact", 'The Q2 2026 build was "about $400 million in Construction Industries".',
                        fact_ids=["S25"], evidence_ids=["N08"], assertion_ids=["N08-A01"])
    assert found == {"QUOTE_NOT_ATTRIBUTED"}


# ------------------------------------------------------------- attribution ---

@pytest.mark.parametrize("kind, text, ids", [
    ("explicit_claim", "Management expects a dealer-inventory drawdown in Q4 2026.",
     {"evidence_ids": ["N13"], "assertion_ids": ["N13-A01"]}),
    ("explicit_claim", "Management guided full-year 2026 tariff cost higher.",
     {"evidence_ids": ["N14"], "assertion_ids": ["N14-A01"]}),
    ("interpretive", "Management said demand is improving, which suggests durable growth.",
     {"premise_ids": ["S22"]}),
    ("limitation", "Management expects a Q4 2026 drawdown.", {"evidence_ids": ["N13"]}),
    ("fact", "According to the CFO, the Q2 2026 build was $400 million.", {"fact_ids": ["S25"]}),
])
def test_attribution_without_validated_quoted_evidence_fails(packet, kind, text, ids):
    assert "ATTRIBUTION_WITHOUT_EVIDENCE" in voice_codes(packet, kind, text, **ids)


@pytest.mark.parametrize("kind, text, ids", [
    ("limitation", "Management's Q4 2026 drawdown commentary was referenced but not captured verbatim, so it is not "
     "used as evidence.", {"evidence_ids": ["N13"]}),
    ("interpretive", "We expect retail growth to remain the better guide to end demand, which suggests watching H2.",
     {"premise_ids": ["S22"]}),
    ("fact", "Q2 2026 retail growth of 22% was higher than expected from the Q1 2026 rate of 7%.",
     {"fact_ids": ["S21", "S22"]}),
    ("interpretive", "The cash-flow pattern suggests the stocking reading should be called into question.",
     {"premise_ids": ["N16"], "assertion_ids": ["N16-A01"]}),
    ("explicit_claim", "Caterpillar's Q2 2026 release said higher sales volume was mainly driven by higher sales of "
     "equipment to end users.", {"evidence_ids": ["N02"], "assertion_ids": ["N02-A02"]}),
])
def test_legitimate_attribution_or_analyst_voice_passes(packet, kind, text, ids):
    assert voice_codes(packet, kind, text, **ids) == set()


def test_source_voice_must_be_attributed(packet):
    assert voice_codes(packet, "explicit_claim", "Higher sales volume was mainly driven by end users in Q2 2026.",
                       evidence_ids=["N02"], assertion_ids=["N02-A02"]) == {"SOURCE_VOICE_NOT_ATTRIBUTED"}


def test_analyst_voice_restatement_needs_no_attribution(packet):
    assert voice_codes(packet, "explicit_claim", "Caterpillar's Q2 2026 release named end users before dealer "
                       "inventories as volume drivers.", evidence_ids=["N02"], assertion_ids=["N02-A01"]) == set()


# --------------------------------------------------------- structure rules ---

def test_explicit_claim_must_list_its_assertion(packet):
    assert voice_codes(packet, "explicit_claim", "Caterpillar's Q2 2026 release named end users first.",
                       evidence_ids=["N02"]) == {"EXPLICIT_WITHOUT_ASSERTION"}


def test_assertion_must_belong_to_cited_evidence(packet):
    assert voice_codes(packet, "explicit_claim", "Caterpillar's Q2 2026 release named end users first.",
                       evidence_ids=["N02"], assertion_ids=["N01-A01"]) == {"ASSERTION_WITHOUT_EVIDENCE"}


def test_unknown_assertion_id(packet):
    assert voice_codes(packet, "explicit_claim", "Caterpillar's Q2 2026 release named end users first.",
                       evidence_ids=["N02"], assertion_ids=["N02-A99"]) == {"UNKNOWN_ASSERTION_ID"}


def test_assertion_ids_stay_out_of_prose(packet):
    s = st("A", "explicit_claim", "Caterpillar's Q2 2026 release named end users first (N02-A01).",
           evidence_ids=["N02"], assertion_ids=["N02-A01"])
    assert "PACKET_ID_IN_PROSE" in validate_statements(packet, [s]).codes


def test_invalid_attribution_pattern_blocks_the_packet(benchmark):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["voice_policy"]["attribution_patterns"].append("(unclosed")
    with pytest.raises(PacketBuildError, match="VOICE_POLICY_INVALID"):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


def test_missing_voice_policy_blocks_the_packet(benchmark):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    del policy["voice_policy"]
    with pytest.raises(PacketBuildError, match="VOICE_POLICY_INVALID"):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


# --------------------------------------------------------------- synthesis ---

def _model_output() -> dict:
    brief = json.loads((fixture_dir() / "valid_brief.json").read_text(encoding="utf-8"))
    del brief["title"]
    brief["validation_metadata"]["author"] = PROMPT_VERSION
    return brief


def _typical(brief: dict) -> dict:
    bad = copy.deepcopy(brief)
    bad["statements"][0].update(text="Disclosed CI dealer inventory increased a more typical $1.5 billion in Q1 2026 "
                                     "and $400 million in Q2 2026.")
    return bad


def test_characterization_is_fixed_on_the_single_retry(packet):
    good = _model_output()
    client = MockLLMClient([_typical(good), good])
    result = synthesize(packet, client, now=lambda: T0)
    assert result.accepted and [a.status for a in result.attempts] == ["REJECTED", "VALID"]
    assert [(e.code, e.subject_id) for e in client.calls[1].corrections] == [
        ("UNSUPPORTED_CHARACTERIZATION", "typical")]
    assert "a more typical $1.5 billion in Q1 2026" not in client.calls[1].messages[0]["content"]


def test_second_bad_response_hard_rejects_after_two_calls(packet):
    bad = _typical(_model_output())
    client = MockLLMClient([bad, bad, bad])
    result = synthesize(packet, client, now=lambda: T0)
    assert result.status is SynthesisStatus.SYNTHESIS_REJECTED and len(client.calls) == 2
