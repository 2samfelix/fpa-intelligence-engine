from __future__ import annotations

import copy
import json
import logging
import re
from datetime import datetime, timezone

import pytest

from fpa_engine.brief import brief_title, build_brief_packet
from fpa_engine.brief_demo import fixture_dir
from fpa_engine.brief_validator import BriefIssue
from fpa_engine.llm import DEFAULT_MODEL, LLMConfigError, LLMSettings, MockLLMClient
from fpa_engine.synthesizer import (
    MAX_ATTEMPTS,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    LLMProviderError,
    LLMRefusalError,
    LLMTimeoutError,
    SynthesisStatus,
    build_messages,
    build_user_prompt,
    parse_model_output,
    synthesize,
)

FIXTURES = fixture_dir()
T0 = datetime(2026, 10, 12, tzinfo=timezone.utc)


def _fixture(name: str) -> dict:
    """A B2 fixture in model-output shape: statements and metadata only (the model never writes the title)."""
    brief = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    del brief["title"]
    brief["validation_metadata"]["author"] = PROMPT_VERSION
    return brief


def _codes(result) -> set[tuple[str, str | None]]:
    return {(e.code, e.subject_id) for e in result.errors}


@pytest.fixture
def valid() -> dict:
    return _fixture("valid_brief.json")


def _run(packet, *responses, **kw):
    client = MockLLMClient(list(responses))
    return synthesize(packet, client, now=lambda: T0, **kw), client


def _assert_rejected(result):
    assert result.status is SynthesisStatus.SYNTHESIS_REJECTED
    assert result.brief is None and not result.accepted
    assert result.errors


# --------------------------------------------------------------- prompt ---

def test_prompt_is_deterministic_and_bound_to_the_packet(packet, benchmark):
    later = build_brief_packet(benchmark, generated_at=datetime(2031, 1, 1, tzinfo=timezone.utc))
    assert build_messages(packet) == build_messages(later)
    system, messages = build_messages(packet)
    assert system == SYSTEM_PROMPT and len(messages) == 1 and messages[0]["role"] == "user"
    user = messages[0]["content"]
    assert packet.packet_hash in user and f'"{packet.analysis_id}"' in user and f'author "{PROMPT_VERSION}"' in user
    assert "generated_at" not in user
    body = json.loads(re.search(r"<brief_packet>\n(.*)\n</brief_packet>", user, re.S).group(1))
    assert body == json.loads(packet.model_dump_json(exclude={"generated_at"}))


def test_prompt_states_the_synthesis_rules():
    for phrase in ("Use only facts and evidence in the packet", "Do not calculate", "Never write any value listed in "
                   "forbidden_values, in any statement of any kind", "not even in a limitation",
                   "Copy the display value exactly; never add precision", "premise_ids must list",
                   "Never relabel interpretive evidence as explicit", "mandatory_reconciliation_ids",
                   "mandatory_disclosure_gap_ids", "mandatory_comparability_note_ids", "conditional_narrative_ids",
                   "Do not add limitations for conditional narratives the brief does not rely on",
                   "statement_bounds", "do not write a title", "REQUIRED LIMITATIONS",
                   "Return exactly one JSON object and nothing else"):
        assert phrase in SYSTEM_PROMPT, phrase


def test_prompt_excludes_raw_sources_workbook_material_and_answers(packet, benchmark):
    system, messages = build_messages(packet)
    prompt = system + messages[0]["content"]
    for key in ("benchmark_cell", "formula", "excerpt", "expected", "rejected_values"):
        assert f'"{key}"' not in prompt, key
    assert not re.search(r"[A-Z][A-Za-z ]+![A-Z]+\d+", prompt), "workbook cell reference in prompt"
    quotes = [n.verbatim_quote for n in packet.narratives if n.verbatim_quote]
    for src in benchmark.sources:
        if src.excerpt and not any(src.excerpt == q for q in quotes):
            assert src.excerpt not in prompt, f"raw excerpt of {src.id} leaked"
    for path in FIXTURES.glob("*.json"):
        for s in json.loads(path.read_text(encoding="utf-8"))["statements"]:
            assert s["text"] not in prompt, f"fixture text from {path.name} leaked"
    assert "workbook" not in system.lower()


def test_correction_prompt_carries_only_errors_and_instruction(packet):
    errors = [BriefIssue(code="UNSUPPORTED_NUMBER", statement_id="B3", message="'$9M' does not match any cited fact")]
    _, messages = build_messages(packet, errors)
    user = messages[0]["content"]
    assert user.startswith(build_user_prompt(packet))
    tail = user[len(build_user_prompt(packet)):]
    assert "- UNSUPPORTED_NUMBER (statement B3): '$9M' does not match any cited fact" in tail
    assert "Correct only these violations" in tail


# ------------------------------------------------------------ mock flow ---

def test_valid_output_is_accepted(packet, valid):
    result, client = _run(packet, valid)
    assert result.status is SynthesisStatus.ACCEPTED and result.accepted
    assert result.brief is not None and result.brief.validation_metadata.packet_hash == packet.packet_hash
    assert result.brief.title == brief_title(packet)
    assert [a.status for a in result.attempts] == ["VALID"]
    assert len(client.calls) == 1 and client.calls[0].corrections == ()
    assert result.model == "mock-deterministic" and result.prompt_version == PROMPT_VERSION


def test_valid_json_text_is_accepted(packet, valid):
    result, _ = _run(packet, json.dumps(valid))
    assert result.accepted


@pytest.mark.parametrize("raw", [
    "```json\n{}\n```",
    "Here is the brief: {\"title\": \"t\"}",
    '{"title": "t", "statements": [',
    '{"title": "a", "title": "b"}',
    "[1, 2]",
    "",
])
def test_malformed_output_fails_safely(packet, raw):
    result, client = _run(packet, raw, raw)
    _assert_rejected(result)
    assert [a.status for a in result.attempts] == ["REJECTED", "REJECTED"]
    assert {e.code for e in result.errors} == {"MALFORMED_OUTPUT"}
    assert result.attempts[0].output == raw
    assert len(client.calls) == MAX_ATTEMPTS


def test_parse_model_output_is_strict():
    assert parse_model_output('  {"a": 1}\n') == {"a": 1}


@pytest.mark.parametrize("mutate, code", [
    (lambda b: b.update(confidence=0.9), "SCHEMA_INVALID"),
    (lambda b: b["statements"][0].update(kind="opinion"), "SCHEMA_INVALID"),
    (lambda b: b["statements"][0].update(notes="extra"), "SCHEMA_INVALID"),
    (lambda b: b["validation_metadata"].pop("packet_hash"), "MISSING_PACKET_HASH"),
    (lambda b: b.update(title="Caterpillar Q2 2026 — Construction Industries Executive Brief"), "SCHEMA_INVALID"),
])
def test_schema_violations_are_rejected(packet, valid, mutate, code):
    mutate(valid)
    result, _ = _run(packet, valid, max_attempts=1)
    _assert_rejected(result)
    assert code in {e.code for e in result.errors}


def test_non_object_payload_is_rejected(packet):
    result, _ = _run(packet, lambda p: ["not", "an", "object"], max_attempts=1)
    _assert_rejected(result)
    assert {e.code for e in result.errors} == {"SCHEMA_INVALID"}


def test_validator_rejection_is_reported_exactly(packet):
    result, _ = _run(packet, _fixture("invalid_01_unknown_fact_id.json"), max_attempts=1)
    _assert_rejected(result)
    assert {(e.code, e.subject_id) for e in result.errors} == {("UNKNOWN_FACT_ID", "D999")}


def test_hash_mismatch_fails(packet, valid):
    valid["validation_metadata"]["packet_hash"] = "0" * 64
    result, _ = _run(packet, valid, max_attempts=1)
    _assert_rejected(result)
    assert {e.code for e in result.errors} == {"PACKET_HASH_MISMATCH"}


@pytest.mark.parametrize("name, codes", [
    ("invalid_03_unsupported_number.json", {"UNSUPPORTED_NUMBER"}),
    ("invalid_04_arithmetic_in_prose.json", {"ARITHMETIC_IN_PROSE"}),
    ("invalid_08_stale_343_bps.json", {"FORBIDDEN_VALUE"}),
    ("invalid_05_interpretive_as_explicit.json", {"INTERPRETIVE_AS_EXPLICIT", "ASSUMPTION_AS_FACT"}),
])
def test_contract_violations_are_rejected(packet, name, codes):
    result, _ = _run(packet, _fixture(name), max_attempts=1)
    _assert_rejected(result)
    assert {e.code for e in result.errors} == codes


def test_omitted_limitations_fail(packet, valid):
    valid["statements"] = [s for s in valid["statements"] if s["id"] not in ("V8", "V9")]
    result, _ = _run(packet, valid, max_attempts=1)
    _assert_rejected(result)
    found = _codes(result)
    assert ("MISSING_RECONCILIATION_LIMITATION", "R04") in found
    mandatory = ("R03", "R04", "R05", "G04", "G05", "G06", "G07", "CN06")
    assert {("MISSING_MANDATORY_LIMITATION", m) for m in mandatory} <= found


def test_unresolved_items_cannot_be_hidden_even_if_uncited(packet):
    hidden = _fixture("valid_brief.json")
    hidden["statements"] = [s for s in hidden["statements"] if s["id"] != "V9"]
    result, _ = _run(packet, hidden, max_attempts=1)
    assert _codes(result) == {("MISSING_MANDATORY_LIMITATION", "R03"), ("MISSING_MANDATORY_LIMITATION", "R05")}


# ---------------------------------------------------------- hardening ---

def test_model_cannot_supply_a_title(packet, valid):
    for title in ("CI profit up $999M", "Tariff drag of 343 bps", brief_title(packet)):
        result, _ = _run(packet, {**valid, "title": title}, max_attempts=1)
        _assert_rejected(result)
        assert {e.code for e in result.errors} == {"SCHEMA_INVALID"}
        assert any("title" in e.message for e in result.errors)


def test_title_is_deterministic_and_carries_no_model_text(packet, benchmark, valid):
    assert brief_title(packet) == "Caterpillar Q2 2026 — Construction Industries Executive Brief"
    later = build_brief_packet(benchmark, generated_at=datetime(2031, 1, 1, tzinfo=timezone.utc))
    assert brief_title(later) == brief_title(packet)
    result, _ = _run(packet, valid)
    assert result.brief.title == brief_title(packet)


@pytest.mark.parametrize("keep", [4, 13])
def test_statement_count_is_bounded(packet, valid, keep):
    by_id = {s["id"]: s for s in valid["statements"]}
    filler = {"kind": "limitation", "text": "Dealer-inventory dollars change scope between quarters.",
              "comparability_note_ids": ["CN04"]}
    if keep == 4:  # two facts plus the two mandatory-surfacing limitations: valid apart from its length
        valid["statements"] = [by_id[i] for i in ("V1", "V2", "V8", "V9")]
    else:
        valid["statements"] += [{**filler, "id": f"F{i}"} for i in range(keep - len(by_id))]
    assert len(valid["statements"]) == keep
    result, _ = _run(packet, valid, max_attempts=1)
    _assert_rejected(result)
    assert {e.code for e in result.errors} == {"STATEMENT_COUNT_OUT_OF_RANGE"}


@pytest.mark.parametrize("keep", [5, 12])
def test_statement_count_bounds_are_inclusive(packet, valid, keep):
    by_id = {s["id"]: s for s in valid["statements"]}
    filler = {"kind": "limitation", "text": "Dealer-inventory dollars change scope between quarters.",
              "comparability_note_ids": ["CN04"]}
    base = [by_id[i] for i in ("V1", "V2", "V8", "V9")] if keep == 5 else list(by_id.values())
    valid["statements"] = base + [{**filler, "id": f"F{i}"} for i in range(keep - len(base))]
    assert (packet.statement_bounds.min, packet.statement_bounds.max) == (5, 12)
    assert _run(packet, valid)[0].accepted


def test_forbidden_value_cannot_enter_through_a_limitation(packet, valid):
    valid["statements"].append({"id": "V10", "kind": "limitation", "fact_ids": ["S29"],
                                "text": "A stale 343 bps estimate was excluded from the validated analysis."})
    result, _ = _run(packet, valid, max_attempts=1)
    _assert_rejected(result)
    assert _codes(result) == {("FORBIDDEN_VALUE", "FV01")}
    valid["statements"][-1]["text"] = "A stale prior estimate was excluded from the validated analysis."
    assert _run(packet, valid)[0].accepted


def test_irrelevant_review_flags_are_not_forced(packet, valid):
    cited = {i for s in valid["statements"] for i in [*s.get("evidence_ids", []), *s.get("premise_ids", [])]}
    unused = set(packet.surfacing_policy.conditional_narrative_ids) - cited
    assert unused == {"N06", "N09", "N13", "N14", "N15"}
    assert _run(packet, valid)[0].accepted


def test_relied_on_review_evidence_requires_a_limitation(packet, valid):
    valid["statements"] = [s for s in valid["statements"] if s["id"] != "V6"]
    result, _ = _run(packet, valid, max_attempts=1)
    assert _codes(result) == {("MISSING_REVIEW_LIMITATION", "N03"), ("MISSING_ASSUMPTION_LIMITATION", "AS01")}


# ---------------------------------------------------------------- retry ---

def test_corrective_retry_recovers(packet, valid):
    bad = copy.deepcopy(valid)
    bad["statements"][6]["text"] = "Q2 2026 CI segment profit rose $704M year over year."
    result, client = _run(packet, bad, valid)
    assert result.accepted
    assert [a.status for a in result.attempts] == ["REJECTED", "VALID"]
    retry = client.calls[1]
    assert [e.code for e in retry.corrections] == ["UNSUPPORTED_NUMBER"]
    content = retry.messages[0]["content"]
    assert "- UNSUPPORTED_NUMBER (statement V7): '$704M'" in content
    assert "$704M year over year" not in content, "the rejected output must not be fed back"
    assert result.attempts[0].output == bad


def test_retry_is_limited_to_one(packet, valid):
    bad = copy.deepcopy(valid)
    bad["validation_metadata"]["packet_hash"] = "f" * 64
    result, client = _run(packet, bad, bad, valid)
    _assert_rejected(result)
    assert len(client.calls) == MAX_ATTEMPTS == 2
    assert [a.status for a in result.attempts] == ["REJECTED", "REJECTED"]
    assert {e.code for e in result.errors} == {"PACKET_HASH_MISMATCH"}


def test_max_attempts_is_bounded(packet, valid):
    with pytest.raises(ValueError):
        synthesize(packet, MockLLMClient([valid]), max_attempts=3)


def test_malformed_then_valid_recovers(packet, valid):
    result, client = _run(packet, "not json", valid)
    assert result.accepted
    assert [e.code for e in client.calls[1].corrections] == ["MALFORMED_OUTPUT"]


# ------------------------------------------------------- provider errors ---

def test_timeout_is_a_provider_error_and_not_retried(packet, valid):
    result, client = _run(packet, LLMTimeoutError("request timed out after 300s"), valid)
    assert result.status is SynthesisStatus.PROVIDER_ERROR and result.brief is None
    assert [e.code for e in result.errors] == ["PROVIDER_TIMEOUT"]
    assert len(client.calls) == 1


def test_provider_failure_during_retry_keeps_the_validator_errors(packet, valid):
    bad = copy.deepcopy(valid)
    bad["validation_metadata"]["packet_hash"] = "f" * 64
    result, _ = _run(packet, bad, LLMProviderError("HTTP 529: overloaded"))
    assert result.status is SynthesisStatus.PROVIDER_ERROR and result.brief is None
    assert [e.code for e in result.errors] == ["PACKET_HASH_MISMATCH", "PROVIDER_ERROR"]


def test_refusal_is_reported_and_not_retried(packet, valid):
    result, client = _run(packet, LLMRefusalError("claude-opus-5-5 declined the request"), valid)
    assert result.status is SynthesisStatus.PROVIDER_ERROR and result.brief is None
    assert [e.code for e in result.errors] == ["PROVIDER_REFUSAL"] and len(client.calls) == 1


def test_exhausted_mock_is_a_provider_error(packet):
    result, _ = _run(packet)
    assert result.status is SynthesisStatus.PROVIDER_ERROR


# -------------------------------------------------------- config & logs ---

def test_settings_come_from_the_environment():
    s = LLMSettings.from_env({"ANTHROPIC_API_KEY": "sk-test-secret", "ANTHROPIC_MODEL": "claude-sonnet-5-5"})
    assert s.model == "claude-sonnet-5-5" and "sk-test-secret" not in repr(s)
    assert LLMSettings.from_env({"ANTHROPIC_API_KEY": "k"}).model == DEFAULT_MODEL
    with pytest.raises(LLMConfigError):
        LLMSettings.from_env({"ANTHROPIC_MODEL": "claude-opus-5-5"})


def test_every_attempt_is_logged_without_secrets(packet, valid, caplog, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret-value")
    bad = copy.deepcopy(valid)
    bad["validation_metadata"]["packet_hash"] = "f" * 64
    with caplog.at_level(logging.INFO, logger="fpa_engine.synthesis"):
        _run(packet, bad, valid)
    events = [json.loads(r.getMessage()) for r in caplog.records]
    assert [e["event"] for e in events] == ["synthesis_attempt", "synthesis_attempt", "synthesis_result"]
    for e in events:
        assert e["model"] == "mock-deterministic" and e["packet_hash"] == packet.packet_hash
        assert e["prompt_version"] == "synthesis_v2" and e["timestamp"] == T0.isoformat()
    assert [e.get("attempt") for e in events[:2]] == [1, 2]
    assert events[0]["validation"] == "REJECTED" and events[0]["error_codes"] == ["PACKET_HASH_MISMATCH"]
    assert events[2]["validation"] == "ACCEPTED" and events[2]["attempt_count"] == 2
    assert "sk-test-secret-value" not in caplog.text
