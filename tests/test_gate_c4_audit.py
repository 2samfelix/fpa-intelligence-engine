"""Gate C4 final audit: each Gate C2 lesson is now rejected deterministically, and the contract pins agree."""
from __future__ import annotations

import copy
import json

from fpa_engine import gate_c2
from fpa_engine.brief import BriefStatement, ExecutiveBrief
from fpa_engine.brief_demo import fixture_dir
from fpa_engine.brief_validator import validate_brief, validate_statements
from fpa_engine.synthesizer import PROMPT_VERSION

C4_PACKET_HASH = "d726abc65d9d39f8077406bfcc3603596fd0dad36adb5e2421695d9f491409bc"


def st(kind: str, text: str, **ids) -> BriefStatement:
    return BriefStatement(id="A", kind=kind, text=text, **ids)


def codes(packet, *statements) -> set[str]:
    return validate_statements(packet, list(statements)).codes


def _valid() -> dict:
    return json.loads((fixture_dir() / "valid_brief.json").read_text(encoding="utf-8"))


def test_contract_pins_agree(packet):
    assert packet.packet_version == 2
    assert packet.packet_hash == gate_c2.EXPECTED_PACKET_HASH == C4_PACKET_HASH
    assert PROMPT_VERSION == gate_c2.EXPECTED_PROMPT_VERSION == "synthesis_v3"
    for path in fixture_dir().glob("*.json"):
        brief = json.loads(path.read_text(encoding="utf-8"))
        assert brief["validation_metadata"]["packet_hash"] == C4_PACKET_HASH, path.name


def test_valid_brief_is_accepted(packet):
    assert validate_brief(packet, ExecutiveBrief.model_validate(_valid())).valid


# A. Semantic fidelity: the CFO's "more typical" stated in the analyst's own voice.
def test_a_more_typical_in_analyst_voice_is_rejected(packet):
    text = "The Q1 2026 CI build of $1.5 billion was a more typical first-quarter increase."
    assert codes(packet, st("fact", text, fact_ids=["S23"], evidence_ids=["N07"], assertion_ids=["N07-A01"])) == {
        "UNSUPPORTED_CHARACTERIZATION"}
    assert codes(packet, st("fact", text, fact_ids=["S23"], evidence_ids=["N07"], assertion_ids=["N07-A02"])) == {
        "SOURCE_VOICE_NOT_ATTRIBUTED", "UNSUPPORTED_CHARACTERIZATION"}
    attributed = st("explicit_claim", 'The CFO called the Q1 2026 build "a more typical $1.5 billion increase in the '
                    'first quarter."', evidence_ids=["N07"], assertion_ids=["N07-A02"])
    assert codes(packet, attributed) == set()


# B. False precision: whole-percent retail inputs rendered to a tenth of a point.
def test_b_false_precision_is_rejected_and_canonical_forms_pass(packet):
    lim = st("limitation", "The gap mixes price bases and is a directional reading only.", fact_ids=["D07", "D08"])
    lim = lim.model_copy(update={"id": "L"})
    for text, fid in (("The Q1 2026 gap, a directional reading, was 24.3 pts.", "D08"),
                      ("The Q2 2026 gap, a directional reading, was 7.8 pts.", "D07")):
        assert codes(packet, st("fact", text, fact_ids=[fid]), lim) == {"FALSE_PRECISION"}
    for text, fid in (("The Q1 2026 gap, a directional reading, was ~24 pts.", "D08"),
                      ("The Q1 2026 gap, a directional reading, was about 24 pts.", "D08"),
                      ("The Q2 2026 gap, a directional reading, was ~8 pts.", "D07"),
                      ("The Q2 2026 gap, a directional reading, was roughly 8 pts.", "D07")):
        assert codes(packet, st("fact", text, fact_ids=[fid]), lim) == set()


# C. Hidden assumptions: the unsourced driver-order premise.
def test_c_driver_order_assumption_cannot_be_fact_or_hidden(packet):
    lim = st("limitation", "Reading the order of mention as a ranking by size rests on an unsourced assumption, and "
             "the driver-order evidence still needs review.", evidence_ids=["N03"], assumption_ids=["AS01"])
    lim = lim.model_copy(update={"id": "L"})
    assert codes(packet, st("fact", "Caterpillar lists its drivers in order of magnitude.", fact_ids=["S23"],
                            assumption_ids=["AS01"]), lim) == {"ASSUMPTION_AS_FACT"}
    assert codes(packet, st("interpretive", "We read the reversal in order of mention as consistent with a shift in "
                            "emphasis.", premise_ids=["N03"]), lim) == {"HIDDEN_ASSUMPTION"}
    assert "MISSING_ASSUMPTION_DISCLOSURE" in codes(packet, st(
        "interpretive", "Management's ordering shows dealer inventory was the larger driver.", premise_ids=["N03"],
        assumption_ids=["AS01"]), lim)


# D. Internal leakage: the accepted Gate C2 brief narrated the engine.
def test_d_internal_leakage_is_rejected(packet):
    assert codes(packet, st("limitation", "A stale alternative estimate was excluded.", fact_ids=["S29"])) == {
        "INTERNAL_PIPELINE_LANGUAGE"}


# E. Coverage: a structurally valid brief that leaves out the analytical spine.
def test_e_missing_required_role_is_rejected(packet):
    brief = copy.deepcopy(_valid())
    for s in brief["statements"]:
        if s["id"] == "V2":
            s["text"] = "Reported CI sales growth was 34.8% in Q2 2026, compared with 38.1% in Q1 2026."
            s["reconciliation_ids"] = []
    result = validate_brief(packet, ExecutiveBrief.model_validate(brief))
    assert result.codes == {"MISSING_REQUIRED_COVERAGE"}
    assert [e.subject_id for e in result.errors] == ["primary_reconciliation"]
