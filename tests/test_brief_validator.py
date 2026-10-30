from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fpa_engine.brief import BriefMetadata, BriefStatement, ExecutiveBrief, brief_title
from fpa_engine.brief_demo import fixture_dir
from fpa_engine.brief_validator import (
    UNIT_CONVERSIONS,
    extract_numbers,
    validate_brief,
    validate_brief_payload,
    validate_statements,
)


def st(sid: str, kind: str, text: str, **ids) -> BriefStatement:
    return BriefStatement(id=sid, kind=kind, text=text, **ids)


def codes(packet, *statements) -> set[str]:
    return validate_statements(packet, list(statements)).codes


# ------------------------------------------------------------ tokenizer ---

def test_periods_years_and_ids_are_not_numeric_claims():
    tokens, _ = extract_numbers("In Q2 2026 (FY2026, H2, 10-Q, Ex 99.2) S01, D11, R04 and NA03 were cited.")
    assert tokens == []


def test_currency_scaling_and_written_precision():
    tokens, _ = extract_numbers("a $1.5 billion build, $400 million, $8,346M and 7.0%")
    by_raw = {t.raw: t for t in tokens}
    assert by_raw["$1.5 billion"].value == Decimal("1500") and by_raw["$1.5 billion"].written_unit == Decimal("100")
    assert by_raw["$400 million"].value == Decimal("400") and by_raw["$400 million"].unit == "usd_m"
    assert by_raw["$8,346M"].value == Decimal("8346")
    assert by_raw["7.0%"].unit == "pct" and by_raw["7.0%"].written_unit == Decimal("0.1")


def test_signed_tokens_get_a_unit_and_ranges_are_parsed():
    tokens, ranges = extract_numbers("+$12M, -470 bps, 7.8 percentage points and $100–150M")
    units = {t.raw: t.unit for t in tokens}
    assert units["+$12M"] == "usd_m" and units["-470 bps"] == "bps" and units["7.8 percentage points"] == "pts"
    assert len(ranges) == 1 and (ranges[0].low, ranges[0].high) == (Decimal("100"), Decimal("150"))


# ------------------------------------------------------- numeric rules ---

def test_matching_values_at_written_precision_pass(packet):
    assert codes(packet, st("A", "fact", "CI sales were $8,346M, about $8.3 billion.", fact_ids=["S01"])) == set()
    assert codes(packet, st("A", "fact", "Reported growth was 34.8%, or roughly 35%.", fact_ids=["D03"])) == set()


def test_false_precision_is_rejected(packet):
    assert "FALSE_PRECISION" in codes(packet, st("A", "fact", "World retail grew 7.0% in Q1 2026.", fact_ids=["S21"]))


def test_uncited_fact_value_is_rejected(packet):
    assert codes(packet, st("A", "fact", "CI sales growth was 34.8% in Q2 2026.", fact_ids=["S01"])) == {
        "UNCITED_FACT_VALUE"}


def test_ratio_arithmetic_requires_the_derived_fact(packet):
    prose = "The Q2 2026 CI build was 26.7% of the Q1 2026 build."
    assert codes(packet, st("A", "fact", prose, fact_ids=["S23", "S25"])) == {"ARITHMETIC_IN_PROSE"}
    assert codes(packet, st("A", "fact", "The Q2 2026 CI build was about 25% of the Q1 2026 build.",
                            fact_ids=["D13"])) == set()


def test_disclosed_rounded_figure_is_not_a_forbidden_back_solve(packet):
    tokens, _ = extract_numbers("volume of $3.1 billion")
    assert tokens[0].written_unit == Decimal("100")
    statement = st("A", "limitation", "Consolidated volume was disclosed only as $3.1 billion.", fact_ids=["S01"])
    assert "FORBIDDEN_VALUE" not in codes(packet, statement)


@pytest.mark.parametrize("text", [
    "The 343 bps figure is a stale artifact and should not be used.",
    "A rejected 342.85 bps estimate was superseded.",
    "The 570 bps plug is not a measured attribution.",
    "An unsupported $100–150M estimate was excluded.",
    "The back-solved $3,113M volume is not a disclosed figure.",
])
def test_forbidden_values_are_rejected_even_in_limitations(packet, text):
    assert codes(packet, st("L", "limitation", text, fact_ids=["S29"])) == {"FORBIDDEN_VALUE"}


def test_limitation_may_not_narrate_an_excluded_estimate(packet):
    """Gate C4 step 3 withdrew the old permission: a brief states the limitation, not the engine's handling."""
    s = st("L", "limitation", "A stale prior estimate was excluded from the validated analysis.", fact_ids=["S29"])
    assert codes(packet, s) == {"INTERNAL_PIPELINE_LANGUAGE"}


def test_explicit_claim_numbers_must_come_from_the_quote_or_facts(packet):
    assert codes(packet, st("A", "explicit_claim", "Management said CI dealer inventory rose by about $400 million.",
                            evidence_ids=["N08"], assertion_ids=["N08-A02"])) == set()
    assert codes(packet, st("A", "explicit_claim", "Management said CI dealer inventory rose by $450 million.",
                            evidence_ids=["N08"], assertion_ids=["N08-A02"])) == {"UNSUPPORTED_NUMBER"}


N14_FIGURE = st("L", "limitation", "Full-year 2026 tariff cost of about $2.2 billion is not yet reviewed.",
                evidence_ids=["N14"])


def test_needs_review_evidence_does_not_support_its_claim_figures(packet):
    """Accepted before: a limitation citing N14 could restate the $2.2 billion in N14's unreviewed claim. Now the
    figure needs other support; "$2.2 billion" (written to the nearest $100M) rounds to D01's $2,156M, which the
    statement does not cite, so it is reported as an uncited fact value."""
    assert packet.narrative_map["N14"].review_status.value == "needs_review"
    assert codes(packet, N14_FIGURE) == {"UNCITED_FACT_VALUE"}


def test_validated_evidence_still_supports_its_quoted_figures(packet):
    assert packet.narrative_map["N04"].review_status.value == "validated"
    assert codes(packet, st("L", "limitation", "The 340 basis points CI tariff impact comes from the call alone.",
                            evidence_ids=["N04"])) == set()


def test_needs_review_claim_figure_rejects_the_whole_brief(packet):
    result = validate_brief_payload(packet, _payload(packet, [N14_FIGURE.model_dump(mode="json")]))
    assert result.codes == {"UNCITED_FACT_VALUE"}


def test_reported_adjective_is_not_a_disclosure_claim(packet):
    assert codes(packet, st("A", "fact", "Reported CI sales growth was 34.8%.", fact_ids=["D03"])) == set()


# ------------------------------------------------------ citation rules ---

def test_fact_statement_needs_fact_citation(packet):
    assert "FACT_WITHOUT_CITATION" in codes(packet, st("A", "fact", "Volume drove growth.", evidence_ids=["N02"]))


def test_fact_cannot_lean_on_interpretive_evidence(packet):
    found = codes(packet, st("A", "fact", "CI sales were $8,346M.", fact_ids=["S01"], evidence_ids=["N16"]))
    assert "INTERPRETIVE_AS_FACT" in found


def test_explicit_claim_needs_evidence(packet):
    assert codes(packet, st("A", "explicit_claim", "Management was upbeat.")) == {"EXPLICIT_WITHOUT_EVIDENCE",
                                                                                   "EXPLICIT_WITHOUT_ASSERTION"}


def test_interpretive_must_read_as_inference(packet):
    found = codes(packet, st("A", "interpretive", "Operating cash flow was $1,870M in Q1 2026.",
                             premise_ids=["S31"]))
    assert found == {"INTERPRETIVE_MARKER_MISSING"}


def test_limitation_must_cite_its_subject(packet):
    assert codes(packet, st("A", "limitation", "Some figures are uncertain.")) == {"LIMITATION_WITHOUT_SUBJECT"}


@pytest.mark.parametrize("field, code", [
    ("reconciliation_ids", "UNKNOWN_RECONCILIATION_ID"),
    ("premise_ids", "UNKNOWN_PREMISE_ID"),
    ("flag_ids", "UNKNOWN_FLAG_ID"),
])
def test_unknown_references_are_rejected(packet, field, code):
    assert codes(packet, st("A", "limitation", "Gap.", **{field: ["ZZ99"]})) == {code}


def test_duplicate_statement_ids_are_rejected(packet):
    s = st("A", "fact", "CI sales were $8,346M.", fact_ids=["S01"])
    assert "DUPLICATE_STATEMENT_ID" in codes(packet, s, s)


# ------------------------------------------------- relationship rules ---

def test_non_comparable_metric_must_be_qualified(packet):
    unqualified = st("A", "fact", "The Q2 2026 stocking swing was ~$500M.", fact_ids=["D20"])
    lim = st("L", "limitation", "Mixed basis.", fact_ids=["D20"], flag_ids=["G02"])
    assert codes(packet, unqualified, lim) == {"NON_COMPARABLE_PRESENTED_AS_PRECISE"}


def test_material_flag_needs_limitation(packet):
    assert codes(packet, st("A", "fact", "Q1 2026 ex-tariff CI margin was 26.5%.", fact_ids=["D17"])) == {
        "MISSING_FLAG_LIMITATION"}


def test_co_citing_na_members_requires_na_limitation(packet):
    s = st("A", "fact", "World retail grew 22% while the CI build was $400M.", fact_ids=["S22", "S25"])
    assert codes(packet, s) == {"MISSING_NA_LIMITATION"}
    assert codes(packet, s, st("L", "limitation", "Different bases.", reconciliation_ids=["NA01"])) == set()


def test_reconciled_bridge_may_be_described_as_reconciled(packet):
    s = st("A", "fact", "The Q2 2026 CI sales bridge reconciles to the $2,156M change.",
           fact_ids=["D01"], reconciliation_ids=["R01"])
    assert codes(packet, s) == set()


def test_negated_completion_language_is_not_a_reconciliation_claim(packet):
    s = st("A", "fact", "The Q1 2026 CI profit bridge does not fully reconcile.", reconciliation_ids=["R03"],
           fact_ids=["S26"])
    lim = st("L", "limitation", "Unassigned $12M residual.", reconciliation_ids=["R03"])
    assert codes(packet, s, lim) == set()


def test_review_and_assumption_dependencies_need_limitations(packet):
    interp = st("A", "interpretive", "We read this as consistent with a shift toward end users.", premise_ids=["N03"],
                assumption_ids=["AS01"])
    assert codes(packet, interp) == {"MISSING_REVIEW_LIMITATION", "MISSING_ASSUMPTION_LIMITATION"}
    inherited = st("A", "interpretive", "Growth composition is consistent with improvement.", premise_ids=["N15"],
                   assumption_ids=["AS01"])
    assert codes(packet, inherited) == {"MISSING_REVIEW_LIMITATION", "MISSING_ASSUMPTION_LIMITATION"}
    covered = st("L", "limitation", "Driver-order reading is an unsourced assumption.", evidence_ids=["N03"],
                 assumption_ids=["AS01"])
    assert codes(packet, inherited, covered) == set()


# Numberless limitations that satisfy the packet's mandatory-surfacing policy and the 5-statement minimum.
MANDATORY = [
    {"id": "M1", "kind": "limitation", "text": "The CI segment profit bridges remain unresolved.",
     "reconciliation_ids": ["R03", "R04", "R05"], "flag_ids": ["G04", "G05", "G06"]},
    {"id": "M2", "kind": "limitation", "text": "The prior-year CI profit comparative has not been tested for restatement.",
     "flag_ids": ["G07"], "comparability_note_ids": ["CN06"]},
    {"id": "M3", "kind": "limitation", "text": "Dealer-inventory dollars change scope between quarters.",
     "comparability_note_ids": ["CN04"]},
    {"id": "M4", "kind": "limitation", "text": "Price realization is one item in two bridges.",
     "comparability_note_ids": ["CN12"]},
]


# The analytical spine required by the coverage policy, taken from the valid brief fixture.
SPINE = [{**s, "id": f"C{i}"} for i, s in enumerate(
    (s for s in json.loads((fixture_dir() / "valid_brief.json").read_text(encoding="utf-8"))["statements"]
     if s["id"] in ("V2", "V4", "V5", "V6")), 1)]


def _payload(packet, statements, **meta) -> dict:
    metadata = {"analysis_id": packet.analysis_id, "packet_hash": packet.packet_hash, "author": "test", **meta}
    return {"title": brief_title(packet), "statements": [*statements, *MANDATORY, *SPINE],
            "validation_metadata": {k: v for k, v in metadata.items() if v is not None}}


SIMPLE = [{"id": "A", "kind": "fact", "text": "CI sales were $8,346M.", "fact_ids": ["S01"]}]


def test_analysis_id_mismatch_is_rejected(packet):
    assert validate_brief_payload(packet, _payload(packet, SIMPLE, analysis_id="other")).codes == {"PACKET_MISMATCH"}


# ------------------------------------------------------ packet hashing ---

def test_correct_packet_hash_passes(packet):
    result = validate_brief_payload(packet, _payload(packet, SIMPLE))
    assert result.valid and result.status == "VALID"


def test_missing_packet_hash_is_rejected(packet):
    result = validate_brief_payload(packet, _payload(packet, SIMPLE, packet_hash=None))
    assert result.codes == {"MISSING_PACKET_HASH"}
    with pytest.raises(ValidationError):
        BriefMetadata(analysis_id=packet.analysis_id, author="test")


@pytest.mark.parametrize("stale", ["0" * 64, "9a07a9113b5847badf8f4191d71b29bf4f238a7f1d362fe373e9c4981a29b4fc"])
def test_stale_or_wrong_packet_hash_is_rejected(packet, stale):
    assert stale != packet.packet_hash
    assert validate_brief_payload(packet, _payload(packet, SIMPLE, packet_hash=stale)).codes == {
        "PACKET_HASH_MISMATCH"}


def test_malformed_packet_hash_is_a_schema_rejection(packet):
    assert validate_brief_payload(packet, _payload(packet, SIMPLE, packet_hash="abc")).codes == {"SCHEMA_INVALID"}


def test_payload_validation_accepts_json_text(packet):
    assert validate_brief_payload(packet, json.dumps(_payload(packet, SIMPLE))).valid


def test_brief_objects_must_carry_the_hash(packet):
    statements = [BriefStatement.model_validate(s) for s in [*SIMPLE, *MANDATORY, *SPINE]]
    brief = ExecutiveBrief(title=brief_title(packet), statements=statements,
                           validation_metadata=BriefMetadata(analysis_id=packet.analysis_id,
                                                             packet_hash=packet.packet_hash, author="test"))
    assert validate_brief(packet, brief).valid


# ----------------------------------------------- registered conversions ---

def test_registered_conversions_are_the_only_ones():
    assert {c.name for c in UNIT_CONVERSIONS.values()} == {"bps_to_percentage_points", "percentage_points_to_bps"}


def test_approved_bps_to_points_conversion_passes(packet):
    s = st("A", "fact", "CI margin rose about 3.2 percentage points year over year in Q2 2026.", fact_ids=["D09"])
    assert codes(packet, s) == set()
    disclosed = st("A", "fact", "Tariffs cut Q2 2026 CI margin by 3.4 percentage points.", fact_ids=["S29"])
    assert codes(packet, disclosed) == set()


def test_approved_points_to_bps_conversion_passes(packet):
    s = st("A", "fact", "The Q2 2026 adjusted gap was about 800 bps, a directional reading.", fact_ids=["D07"])
    lim = st("L", "limitation", "Directional only.", fact_ids=["D07"])
    assert codes(packet, s, lim) == set()


@pytest.mark.parametrize("text, fact_id", [
    ("CI margin rose 3.2% year over year in Q2 2026.", "D09"),
    ("Q2 2026 CI segment sales were 8.3%.", "S01"),
    ("CI margin rose 0.032 in Q2 2026.", "D09"),
])
def test_unsupported_conversion_fails(packet, text, fact_id):
    assert codes(packet, st("A", "fact", text, fact_ids=[fact_id])) & {"UNSUPPORTED_NUMBER", "UNCITED_FACT_VALUE"}


def test_conversion_preserves_precision(packet):
    assert codes(packet, st("A", "fact", "Tariffs cut Q2 2026 CI margin by 3.40 percentage points.",
                            fact_ids=["S29"])) == set()
    assert codes(packet, st("A", "fact", "Tariffs cut Q2 2026 CI margin by 3.400 percentage points.",
                            fact_ids=["S29"])) == {"FALSE_PRECISION"}
    assert "UNSUPPORTED_NUMBER" in codes(packet, st("A", "fact", "CI margin rose 3.25 percentage points.",
                                                    fact_ids=["D09"]))


def test_converted_forbidden_value_is_still_forbidden(packet):
    s = st("A", "fact", "Tariffs cut Q2 2026 CI margin by 3.43 percentage points.", fact_ids=["S29"])
    assert codes(packet, s) == {"FORBIDDEN_VALUE"}


# -------------------------------------------- limitation-trigger policy ---

def test_material_flag_policy_is_exactly_the_approved_set(packet):
    assert set(packet.material_flags) == {"mixed_basis", "non_comparable", "directional", "upper_bound",
                                          "scope_period_inferred", "restatement_untested"}


def test_source_quality_flags_are_visible_but_do_not_force_limitations(packet):
    fm = packet.fact_map
    assert "phrase_match_only" in fm["S25"].flags and not fm["S25"].requires_limitation
    assert "transcript_source" in fm["S29"].flags and not fm["S29"].requires_limitation
    assert codes(packet, st("A", "fact", "CI dealer inventory rose $400 million in Q2 2026.", fact_ids=["S25"])) == set()
    assert codes(packet, st("A", "fact", "Tariffs cut Q2 2026 CI margin by 340 bps.", fact_ids=["S29"])) == set()


def test_phrase_match_fact_with_another_material_flag_still_needs_limitation(packet):
    s24 = packet.fact_map["S24"]
    assert {"phrase_match_only", "scope_period_inferred"} <= set(s24.flags)
    s = st("A", "fact", "Company-wide dealer inventory rose $2.0 billion in Q1 2026.", fact_ids=["S24"])
    assert codes(packet, s) == {"MISSING_FLAG_LIMITATION", "MISSING_DISCLOSURE_GAP"}


@pytest.mark.parametrize("fact_id, flag", [
    ("D07", "mixed_basis"), ("D20", "non_comparable"), ("D08", "directional"), ("D16", "upper_bound"),
    ("D14", "scope_period_inferred"), ("D12", "restatement_untested"),
])
def test_each_material_flag_forces_a_limitation(packet, fact_id, flag):
    fact = packet.fact_map[fact_id]
    assert flag in fact.flags and fact.requires_limitation
