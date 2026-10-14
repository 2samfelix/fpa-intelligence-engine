"""Gate C4 step 3: executive prose carries no engine vocabulary and no packet identifiers."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest

from fpa_engine.brief import BriefStatement, PacketBuildError, build_brief_packet, load_policy
from fpa_engine.brief_demo import fixture_dir
from fpa_engine.brief_validator import packet_id_kinds, validate_statements
from fpa_engine.llm import MockLLMClient
from fpa_engine.synthesizer import PROMPT_VERSION, SYSTEM_PROMPT, SynthesisStatus, synthesize

T0 = datetime(2026, 10, 12, tzinfo=timezone.utc)
PROSE_CODES = {"INTERNAL_PIPELINE_LANGUAGE", "PACKET_ID_IN_PROSE"}


def st(sid: str, kind: str, text: str, **ids) -> BriefStatement:
    return BriefStatement(id=sid, kind=kind, text=text, **ids)


def prose_codes(packet, text: str, kind: str = "limitation", **ids) -> set[str]:
    ids = ids or {"reconciliation_ids": ["R04"]}
    return validate_statements(packet, [st("A", kind, text, **ids)]).codes & PROSE_CODES


# ------------------------------------------------------------- must fail ---

@pytest.mark.parametrize("text", [
    "A stale alternative estimate was excluded.",
    "The validator rejected the 570 bps estimate.",
    "The packet does not support that conclusion.",
    "An unsupported estimate of the residual was dropped.",
    "The alternative estimate was excluded from this brief.",
    "This figure is a forbidden value.",
    "The rejected estimates are not shown.",
    "The validation errors were fixed on retry.",
    "The output schema has no field for this.",
    "The test fixture covers this case.",
    "This is not quantified in the benchmark.",
    "The open flags list this gap.",
    "The fact IDs cited here are derived.",
])
def test_internal_pipeline_language_is_rejected(packet, text):
    assert "INTERNAL_PIPELINE_LANGUAGE" in prose_codes(packet, text)


@pytest.mark.parametrize("text, found", [
    ("R04 remains unresolved.", "R04"),
    ("D08 shows approximately 24 points.", "D08"),
    ("AS01 is an unsourced assumption.", "AS01"),
    ("The driver-order reading (N03) needs review.", "N03"),
    ("Disclosed dealer inventory (S23) rose.", "S23"),
    ("See G04 and CN06.", "G04"),
])
def test_packet_ids_in_prose_are_rejected(packet, text, found):
    result = validate_statements(packet, [st("A", "limitation", text, reconciliation_ids=["R04"])])
    hits = [e for e in result.errors if e.code == "PACKET_ID_IN_PROSE"]
    assert found in {e.subject_id for e in hits}


def test_every_current_packet_id_is_caught(packet):
    ids = packet_id_kinds(packet)
    assert {"S23", "D08", "N03", "AS01", "R04", "G04", "CN06", "FV01", "RV-N03", "UR-R04", "NA01"} <= set(ids)
    for pid, kind in ids.items():
        result = validate_statements(packet, [st("A", "limitation", f"The item {pid} is open.",
                                                 reconciliation_ids=["R04"])])
        hit = next(e for e in result.errors if e.code == "PACKET_ID_IN_PROSE")
        assert hit.subject_id == pid and kind in hit.message, pid


def test_each_id_is_reported_once_per_statement(packet):
    result = validate_statements(packet, [st("A", "limitation", "R04, R04 and R04.", reconciliation_ids=["R04"])])
    assert [e.subject_id for e in result.errors if e.code == "PACKET_ID_IN_PROSE"] == ["R04"]


@pytest.mark.parametrize("kind, ids", [
    ("fact", {"fact_ids": ["S23"]}),
    ("explicit_claim", {"evidence_ids": ["N02"]}),
    ("interpretive", {"premise_ids": ["S23", "S25"]}),
    ("limitation", {"reconciliation_ids": ["R04"]}),
])
def test_rules_apply_to_every_statement_kind(packet, kind, ids):
    assert prose_codes(packet, "S25 shows the packet view.", kind=kind, **ids) == PROSE_CODES


# ------------------------------------------------------------- must pass ---

@pytest.mark.parametrize("text, ids", [
    ("The Q2 profit bridge remains unresolved.", {"reconciliation_ids": ["R04"]}),
    ("The available disclosures do not support a consistent split.", {"reconciliation_ids": ["R04"]}),
    ("The workbook method is not comparable with the disclosed 340 bps figure.", {"fact_ids": ["S29"]}),
    ("Inter-segment eliminations are excluded from this chart for legibility.", {"reconciliation_ids": ["R04"]}),
    ("The calculation excludes inter-segment eliminations for legibility.", {"reconciliation_ids": ["R04"]}),
    ("One-time items were excluded from the estimate by management.", {"reconciliation_ids": ["R04"]}),
    ("Retail growth is compared against a peer benchmark index.", {"reconciliation_ids": ["R04"]}),
    ("Regression analysis is not used here.", {"reconciliation_ids": ["R04"]}),
    ("Dealers reported stale inventory in some regions.", {"reconciliation_ids": ["R04"]}),
    ("Fixtures and fittings are outside this review.", {"reconciliation_ids": ["R04"]}),
    ("The validated quote names end users first.", {"reconciliation_ids": ["R04"]}),
])
def test_legitimate_financial_language_passes(packet, text, ids):
    assert prose_codes(packet, text, **ids) == set()


@pytest.mark.parametrize("text", [
    "The D11T dozer, the S100 series and the 10-Q filing are outside this review.",
    "Lowercase d08 or r04 are not identifiers.",
    "Model RV-N99 and item AS011 are unrelated.",
    "The G4 platform and the CN6 code are unrelated.",
])
def test_lookalike_identifiers_pass(packet, text):
    assert prose_codes(packet, text) == set()


def test_ids_remain_allowed_in_reference_fields(packet):
    s = st("A", "interpretive", "We read the reversal in order of mention as consistent with a shift in emphasis.",
           premise_ids=["N03", "S23", "S25"], assumption_ids=["AS01"])
    lim = st("L", "limitation", "The driver-order reading rests on an unsourced assumption and needs review.",
             evidence_ids=["N03"], assumption_ids=["AS01"], flag_ids=["RV-N03"])
    assert validate_statements(packet, [s, lim]).codes == set()


# ---------------------------------------------------------------- policy ---

def test_internal_language_is_policy_owned(benchmark, packet):
    concepts = {t.concept for t in packet.prose_policy.internal_language}
    assert {"validator", "the brief packet", "stale estimates", "excluded or unsupported estimates"} <= concepts
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["prose_policy"]["internal_language"] = [t for t in policy["prose_policy"]["internal_language"]
                                                   if t["concept"] != "the brief packet"]
    relaxed = build_brief_packet(benchmark, generated_at=T0, policy=policy)
    text = "The packet does not support that conclusion."
    assert prose_codes(packet, text) == {"INTERNAL_PIPELINE_LANGUAGE"}
    assert prose_codes(relaxed, text) == set()


def test_invalid_internal_language_pattern_blocks_the_packet(benchmark):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["prose_policy"]["internal_language"].append({"concept": "broken", "pattern": "(unclosed"})
    with pytest.raises(PacketBuildError, match="PROSE_POLICY_INVALID"):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


def test_audit_metadata_is_kept_internally(packet):
    assert len(packet.forbidden_values) == 5 and packet.flag_map["UR-R04"]
    assert all(f.id for f in packet.facts) and packet.assumption_map["AS01"]


def test_revised_prompt_has_its_own_version():
    """The Step 3 prompt text differs from synthesis_v2; synthesis_v3 is reserved for the full Gate C4 contract."""
    from fpa_engine import gate_c2
    assert PROMPT_VERSION == gate_c2.EXPECTED_PROMPT_VERSION == "synthesis_v2_1"


def test_prompt_no_longer_invites_pipeline_narration():
    assert "estimate was excluded" not in SYSTEM_PROMPT
    assert "packet IDs are not quantitative claims and may be used" not in SYSTEM_PROMPT
    assert "never write them in statement text" in SYSTEM_PROMPT


# ------------------------------------------------------------- synthesis ---

def _model_output() -> dict:
    brief = json.loads((fixture_dir() / "valid_brief.json").read_text(encoding="utf-8"))
    del brief["title"]
    brief["validation_metadata"]["author"] = PROMPT_VERSION
    return brief


LEAK_ID = "Disclosed CI dealer inventory (S23, S25) increased $1.5 billion in Q1 2026 and $400 million in Q2 2026."
LEAK_TERM = "A stale alternative estimate was excluded."


def _leaky(brief: dict) -> dict:
    bad = copy.deepcopy(brief)
    bad["statements"][0]["text"] = LEAK_ID
    bad["statements"].append({"id": "V10", "kind": "limitation", "text": LEAK_TERM, "fact_ids": ["S29"]})
    return bad


def test_leak_is_actionable_and_fixed_on_the_single_retry(packet):
    good = _model_output()
    client = MockLLMClient([_leaky(good), good])
    result = synthesize(packet, client, now=lambda: T0)
    assert result.accepted and [a.status for a in result.attempts] == ["REJECTED", "VALID"]
    corrections = client.calls[1].corrections
    assert {e.code for e in corrections} == PROSE_CODES
    assert {e.subject_id for e in corrections if e.code == "PACKET_ID_IN_PROSE"} == {"S23", "S25"}
    content = client.calls[1].messages[0]["content"]
    assert "INTERNAL_PIPELINE_LANGUAGE" in content and "PACKET_ID_IN_PROSE" in content
    assert "keep IDs in the reference fields" in content and "state the financial point" in content
    assert LEAK_ID not in content and LEAK_TERM not in content


def test_second_leaky_response_hard_rejects_after_two_calls(packet):
    bad = _leaky(_model_output())
    client = MockLLMClient([bad, bad, bad])
    result = synthesize(packet, client, now=lambda: T0)
    assert result.status is SynthesisStatus.SYNTHESIS_REJECTED and len(client.calls) == 2
    assert {e.code for e in result.errors} == PROSE_CODES
