"""Gate C4 step 6: the synthesis_v3 prompt states exactly the contract the packet and validator enforce."""
from __future__ import annotations

import re

import pytest

from fpa_engine import gate_c2
from fpa_engine.brief import (
    BriefPacket,
    BriefStatement,
    CoverageAlternative,
    CoverageRole,
    DisplayPolicy,
    InferencePolicy,
    PacketAssertion,
    PacketAssumption,
    PacketFact,
    PacketNarrative,
    ProsePolicy,
    StatementKind,
    VoicePolicy,
)
from fpa_engine.brief_validator import packet_id_kinds, validate_statements
from fpa_engine.synthesizer import MAX_ATTEMPTS, PROMPT_VERSION, SYSTEM_PROMPT, build_messages

# Every packet field the prompt tells the model to read, with the model that must define it.
REFERENCED_FIELDS = [
    (PacketFact, "display"), (PacketFact, "display_step"), (PacketFact, "approximate"),
    (PacketFact, "display_wording"), (PacketFact, "requires_limitation"), (PacketFact, "origin"),
    (DisplayPolicy, "approximation_markers"),
    (PacketAssumption, "sourcing"), (PacketAssumption, "allowed_use"), (PacketAssumption, "must_disclose"),
    (PacketNarrative, "assumption_ids"), (PacketNarrative, "inherited_assumption_ids"),
    (PacketNarrative, "assertion_ids"), (PacketNarrative, "verbatim_quote"), (PacketNarrative, "evidence_type"),
    (InferencePolicy, "assumption_inference_markers"),
    (PacketAssertion, "voice"),
    (VoicePolicy, "characterization_terms"),
    (ProsePolicy, "internal_language"),
    (CoverageRole, "statement_kinds"), (CoverageRole, "any_of"), (CoverageRole, "from_surfacing_policy"),
    (BriefPacket, "coverage_policy"), (BriefPacket, "statement_bounds"), (BriefPacket, "forbidden_values"),
    (BriefPacket, "surfacing_policy"), (BriefPacket, "display_policy"), (BriefPacket, "inference_policy"),
    (BriefPacket, "voice_policy"), (BriefPacket, "prose_policy"), (BriefPacket, "assumptions"),
]


def test_prompt_version_and_retry_limit_are_the_c4_contract():
    assert PROMPT_VERSION == gate_c2.EXPECTED_PROMPT_VERSION == "synthesis_v3"
    assert MAX_ATTEMPTS == 2


def test_output_shape_lists_exactly_the_statement_fields():
    shape = SYSTEM_PROMPT[SYSTEM_PROMPT.index('"statements": ['):SYSTEM_PROMPT.index('"validation_metadata"')]
    assert set(re.findall(r'"(\w+)":', shape)) - {"statements"} == set(BriefStatement.model_fields)


def test_prompt_names_exactly_four_statement_kinds():
    assert '"fact" | "explicit_claim" | "interpretive" | "limitation"' in SYSTEM_PROMPT
    assert {k.value for k in StatementKind} == {"fact", "explicit_claim", "interpretive", "limitation"}
    assert "STATEMENT KINDS (there are exactly four)" in SYSTEM_PROMPT


@pytest.mark.parametrize("model, field", REFERENCED_FIELDS)
def test_every_field_the_prompt_names_exists(model, field):
    assert field in model.model_fields, f"{model.__name__}.{field}"
    assert field in SYSTEM_PROMPT, field


def test_coverage_alternative_fields_are_explained():
    for field in ("fact_ids", "narrative_ids", "reconciliation_ids", "assumption_ids"):
        assert field in CoverageAlternative.model_fields and field in SYSTEM_PROMPT


@pytest.mark.parametrize("phrase", [
    # division of labour
    "Python owns", "all arithmetic", "display precision and approximation status", "assumption status",
    "required coverage", "the final decision to accept or reject the brief", "structured executive synthesis only",
    # numbers
    "Never write a fact's raw \"value\"", "never round on your own", "Never write more digits (\"24.3 pts\")",
    "never drop the marker (\"24 pts\")", "(for example \"about 24 pts\")",
    # assumptions
    "appearing in the packet does not make it sourced", "must list that assumption in its own assumption_ids",
    "never a fact or an explicit_claim", "A later limitation does not make up for an unframed statement",
    # assertions and voice
    "do not paraphrase beyond it", "Use it only attributed to the source",
    "must never be turned into a management claim", "use neutral financial language",
    # quotations
    "Do not use quotation marks unless you are reproducing validated source language",
    "No scare quotes", "kept in their original order",
    # executive language
    "Never refer to the validator", "Do not write, for example, that a stale estimate was excluded",
    "Packet IDs belong only in the ID fields",
    # coverage
    "leaving out required analytical content rejects the brief", "The IDs of one alternative must sit in the same",
    "One statement may cover several roles; do not add redundant statements",
])
def test_prompt_states_each_c4_rule(phrase):
    assert phrase in SYSTEM_PROMPT, phrase


def test_prompt_hard_codes_no_packet_ids(packet):
    ids = packet_id_kinds(packet)
    pattern = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(i) for i in ids) + r")(?![\w-])")
    assert pattern.findall(SYSTEM_PROMPT) == []


def test_prompt_numeric_examples_match_the_validator(packet):
    """The prompt's own examples for an approximate fact behave as the prompt says."""
    def number_codes(text):
        found = validate_statements(packet, [BriefStatement(id="A", kind="fact", text=text, fact_ids=["D08"])]).codes
        return found & {"FALSE_PRECISION", "APPROXIMATION_MARKER_MISSING"}
    assert packet.fact_map["D08"].display == "~24 pts"
    assert number_codes("The Q1 2026 gap was ~24 pts.") == set()
    assert number_codes("The Q1 2026 gap was about 24 pts.") == set()
    assert number_codes("The Q1 2026 gap was 24.3 pts.") == {"FALSE_PRECISION"}
    assert number_codes("The Q1 2026 gap was 24 pts.") == {"APPROXIMATION_MARKER_MISSING"}


def test_retry_message_carries_errors_not_the_rejected_output(packet):
    from fpa_engine.brief_validator import BriefIssue

    error = BriefIssue(code="MISSING_REQUIRED_COVERAGE", subject_id="demand_signal", message="role missing")
    system, messages = build_messages(packet, [error])
    assert system == SYSTEM_PROMPT and len(messages) == 1
    content = messages[0]["content"]
    assert "MISSING_REQUIRED_COVERAGE" in content and "Write the brief again from the packet" in content
    assert content.count("<brief_packet>") == 1
