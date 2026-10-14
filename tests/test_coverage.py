"""Gate C4 step 4: role-based required coverage, read only from structured references, and feasibility checks."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from fpa_engine.brief import (
    ExecutiveBrief,
    PacketBuildError,
    StatementKind,
    build_brief_packet,
    load_policy,
    minimum_statement_slots,
)
from fpa_engine.brief_demo import fixture_dir
from fpa_engine.brief_validator import covered_roles, validate_brief
from fpa_engine.llm import MockLLMClient
from fpa_engine.synthesizer import PROMPT_VERSION, SynthesisStatus, synthesize

T0 = datetime(2026, 10, 12, tzinfo=timezone.utc)
ROLES = {"headline_evidence", "demand_signal", "primary_reconciliation", "material_unresolved_item",
         "key_interpretation", "mandatory_limitation"}
K = StatementKind


def _valid() -> dict:
    return json.loads((fixture_dir() / "valid_brief.json").read_text(encoding="utf-8"))


def _edit(brief: dict, sid: str, **changes) -> dict:
    out = copy.deepcopy(brief)
    for s in out["statements"]:
        if s["id"] == sid:
            s.update(changes)
    return out


def _drop(brief: dict, *sids: str) -> dict:
    out = copy.deepcopy(brief)
    out["statements"] = [s for s in out["statements"] if s["id"] not in sids]
    return out


def _add(brief: dict, *statements: dict) -> dict:
    out = copy.deepcopy(brief)
    out["statements"] += list(statements)
    return out


def _result(packet, brief: dict):
    return validate_brief(packet, ExecutiveBrief.model_validate(brief))


def _missing_roles(result) -> set[str]:
    return {e.subject_id for e in result.errors if e.code == "MISSING_REQUIRED_COVERAGE"}


def _covered(packet, brief: dict) -> dict[str, list[str]]:
    return covered_roles(packet, ExecutiveBrief.model_validate(brief).statements)


V4_NO_DEALER = dict(text="We read the change between Q1 and Q2 2026 as consistent with a shift toward end-user demand: "
                         "World retail growth rose from 7% to 22%.", premise_ids=["S21", "S22", "N15"])


# ---------------------------------------------------------------- policy ---

def test_caterpillar_policy_builds_with_every_role(packet):
    cp = packet.coverage_policy
    assert {r.role for r in cp.roles} == ROLES
    assert cp.min_statement_slots == 2  # one interpretive statement and one limitation can carry every role
    alts = {r.role: [a.all_ids() for a in r.any_of] for r in cp.roles}
    assert alts["headline_evidence"] == [["S23", "S25"]]
    assert alts["demand_signal"] == [["S21", "S22"], ["D08", "D07"]]
    assert alts["primary_reconciliation"] == [["R01"], ["R02"]]
    assert alts["material_unresolved_item"] == [["R04"]]
    assert alts["key_interpretation"] == [["N15"]]
    assert next(r for r in cp.roles if r.role == "mandatory_limitation").from_surfacing_policy


def test_valid_brief_covers_every_role(packet):
    covered = _covered(packet, _valid())
    assert covered == {"headline_evidence": ["V1", "V4"], "demand_signal": ["V4"],
                       "primary_reconciliation": ["V2"], "material_unresolved_item": ["V8"],
                       "key_interpretation": ["V4"]}
    assert _result(packet, _valid()).valid


# -------------------------------------------------------- missing coverage ---

def test_missing_dealer_build_change(packet):
    brief = _edit(_drop(_valid(), "V1"), "V4", **V4_NO_DEALER)
    result = _result(packet, brief)
    assert result.codes == {"MISSING_REQUIRED_COVERAGE"} and _missing_roles(result) == {"headline_evidence"}


def test_only_the_q1_dealer_build(packet):
    brief = _edit(_edit(_valid(), "V1", text="Disclosed CI dealer inventory increased $1.5 billion in Q1 2026.",
                        fact_ids=["S23"]), "V4", **V4_NO_DEALER)
    result = _result(packet, brief)
    assert result.codes == {"MISSING_REQUIRED_COVERAGE"} and _missing_roles(result) == {"headline_evidence"}


def test_only_one_retail_quarter(packet):
    brief = _edit(_valid(), "V4", text="We read the change between Q1 and Q2 2026 as consistent with a shift toward "
                  "end-user demand: the disclosed CI dealer build fell from $1.5 billion to $400 million while World "
                  "retail growth reached 22% in Q2 2026.", premise_ids=["S23", "S25", "S22", "N15"])
    result = _result(packet, brief)
    assert result.codes == {"MISSING_REQUIRED_COVERAGE"} and _missing_roles(result) == {"demand_signal"}


def test_only_one_gap_quarter_does_not_cover_demand(packet):
    brief = _edit(_valid(), "V4", premise_ids=["S23", "S25", "D07", "N15"])
    assert "demand_signal" not in {r for r, by in _covered(packet, brief).items() if by}


def test_no_primary_reconciliation(packet):
    brief = _edit(_valid(), "V2", text="Reported CI sales growth was 34.8% in Q2 2026, compared with 38.1% in Q1 2026.",
                  reconciliation_ids=[])
    result = _result(packet, brief)
    assert result.codes == {"MISSING_REQUIRED_COVERAGE"} and _missing_roles(result) == {"primary_reconciliation"}


def test_no_r04_unresolved_status(packet):
    brief = _edit(_valid(), "V8", reconciliation_ids=[])
    result = _result(packet, brief)
    assert _missing_roles(result) == {"material_unresolved_item"}
    # coverage supplements, and does not replace, the existing limitation rules
    assert {"MISSING_MANDATORY_LIMITATION", "MISSING_RECONCILIATION_LIMITATION"} <= result.codes


def test_no_key_interpretation(packet):
    """The Step 2 valid brief rested its interpretation on N03 only; it is now rejected."""
    brief = _edit(_valid(), "V4", premise_ids=["S23", "S25", "S21", "S22", "N03"])
    result = _result(packet, brief)
    assert result.codes == {"MISSING_REQUIRED_COVERAGE"} and _missing_roles(result) == {"key_interpretation"}


def test_error_names_the_role_and_its_alternatives(packet):
    brief = _edit(_valid(), "V4", premise_ids=["S23", "S25", "N15"],
                  text="We read the change between Q1 and Q2 2026 as consistent with a shift toward end-user demand: "
                       "the disclosed CI dealer build fell from $1.5 billion to $400 million.")
    issue = next(e for e in _result(packet, brief).errors if e.code == "MISSING_REQUIRED_COVERAGE")
    assert issue.subject_id == "demand_signal"
    assert "fact_ids S21 + S22" in issue.message and "fact_ids D08 + D07" in issue.message
    assert "fact or interpretive statement" in issue.message


# ------------------------------------------------------- valid alternatives ---

GAP_V4 = dict(text="We read the change between Q1 and Q2 2026 as consistent with a shift toward end-user demand: the "
                   "disclosed CI dealer build fell from $1.5 billion to $400 million while the adjusted "
                   "reported-vs-retail gap, a directional reading, narrowed from ~24 pts to ~8 pts.",
              premise_ids=["S23", "S25", "D08", "D07", "N15"])
GAP_LIMITATION = {"id": "V10", "kind": "limitation", "fact_ids": ["D07", "D08"],
                  "text": "The adjusted reported-vs-retail gap mixes price bases and is a directional reading only."}


def test_retail_pair_satisfies_demand_signal(packet):
    assert _covered(packet, _valid())["demand_signal"] == ["V4"]


def test_gap_pair_satisfies_demand_signal(packet):
    brief = _add(_edit(_valid(), "V4", **GAP_V4), GAP_LIMITATION)
    assert _covered(packet, brief)["demand_signal"] == ["V4"]
    assert _result(packet, brief).valid


@pytest.mark.parametrize("rid", ["R01", "R02"])
def test_either_sales_bridge_satisfies_primary_reconciliation(packet, rid):
    brief = _edit(_valid(), "V2", reconciliation_ids=[rid])
    assert _covered(packet, brief)["primary_reconciliation"] == ["V2"]
    assert _result(packet, brief).valid


# ------------------------------------------------------------- multi-role ---

def test_one_statement_may_cover_several_roles(packet):
    covered = _covered(packet, _drop(_valid(), "V1"))
    assert {r for r, by in covered.items() if by == ["V4"]} == {"headline_evidence", "demand_signal",
                                                                "key_interpretation"}


def test_a_pair_split_across_statements_does_not_count(packet):
    split = [{"id": "H1", "kind": "fact", "text": "Disclosed CI dealer inventory increased $1.5 billion in Q1 2026.",
              "fact_ids": ["S23"]},
             {"id": "H2", "kind": "fact", "text": "Disclosed CI dealer inventory increased $400 million in Q2 2026.",
              "fact_ids": ["S25"]}]
    brief = _add(_edit(_drop(_valid(), "V1"), "V4", **V4_NO_DEALER), *split)
    result = _result(packet, brief)
    assert result.codes == {"MISSING_REQUIRED_COVERAGE"} and _missing_roles(result) == {"headline_evidence"}


def test_a_statement_of_a_disallowed_kind_does_not_count(packet):
    lim = {"id": "V10", "kind": "limitation", "text": "The demand-quality thesis still needs review.",
           "evidence_ids": ["N15"]}
    brief = _add(_edit(_valid(), "V4", premise_ids=["S23", "S25", "S21", "S22", "N03"]), lim)
    assert _missing_roles(_result(packet, brief)) == {"key_interpretation"}


# ---------------------------------------------- structured references only ---

def test_prose_without_ids_does_not_cover(packet):
    brief = _edit(_valid(), "V2", reconciliation_ids=[])
    assert "match the reported sales change" in next(s["text"] for s in brief["statements"] if s["id"] == "V2")
    assert _missing_roles(_result(packet, brief)) == {"primary_reconciliation"}


def test_ids_cover_whatever_the_valid_wording(packet):
    brief = _edit(_valid(), "V4", text="The Q1-to-Q2 2026 pattern suggests demand quality improved: the disclosed CI "
                  "dealer build fell from $1.5 billion to $400 million as World retail growth rose from 7% to 22%.")
    assert _result(packet, brief).valid


def test_covering_statements_still_face_every_other_rule(packet):
    brief = _edit(_valid(), "V4", text="Demand shifted to end users: the CI dealer build fell from $1.5 billion to "
                  "$400 million while World retail growth rose from 7.0% to 22% (N15).")
    codes = _result(packet, brief).codes
    assert {"PACKET_ID_IN_PROSE", "FALSE_PRECISION", "MISSING_ASSUMPTION_DISCLOSURE"} <= codes
    assert "MISSING_REQUIRED_COVERAGE" not in codes


# ------------------------------------------------------------ feasibility ---

def _policy_with(**role_changes) -> dict:
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    for role in policy["coverage_policy"]["roles"]:
        if role["role"] in role_changes:
            role.update(role_changes[role["role"]])
    return policy


@pytest.mark.parametrize("changes, message", [
    ({"headline_evidence": {"any_of": [{"fact_ids": ["S23", "S99"]}]}}, "fact_ids S99"),
    ({"primary_reconciliation": {"any_of": [{"reconciliation_ids": ["R09"]}]}}, "reconciliation_ids R09"),
    ({"key_interpretation": {"any_of": [{"narrative_ids": ["N99"]}]}}, "narrative_ids N99"),
    ({"headline_evidence": {"any_of": [{}]}}, "alternative 1 is empty"),
    ({"headline_evidence": {"any_of": []}}, "headline_evidence: no alternatives"),
    ({"key_interpretation": {"statement_kinds": ["fact"]}}, "cannot legally be cited"),
    ({"key_interpretation": {"statement_kinds": ["explicit_claim"], "any_of": [{"assumption_ids": ["AS01"]}]}},
     "cannot legally be cited"),
    ({"mandatory_limitation": {"any_of": [{"fact_ids": ["S23"]}]}}, "takes no alternatives"),
    ({"mandatory_limitation": {"statement_kinds": ["fact"]}}, "can only be limitation statements"),
    ({"headline_evidence": {"statement_kinds": ["opinion"]}}, "COVERAGE_POLICY_INVALID"),
])
def test_invalid_coverage_policy_blocks_the_packet(benchmark, changes, message):
    with pytest.raises(PacketBuildError, match="COVERAGE_POLICY_INVALID") as exc:
        build_brief_packet(benchmark, generated_at=T0, policy=_policy_with(**changes))
    assert message in str(exc.value)


def test_duplicate_role_blocks_the_packet(benchmark):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["coverage_policy"]["roles"].append(policy["coverage_policy"]["roles"][0])
    with pytest.raises(PacketBuildError, match="duplicate role"):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


def test_missing_coverage_policy_blocks_the_packet(benchmark):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    del policy["coverage_policy"]
    with pytest.raises(PacketBuildError, match="COVERAGE_POLICY_INVALID"):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


def test_impossible_statement_bounds_block_the_packet(benchmark):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["statement_count"] = {"min": 1, "max": 1}
    with pytest.raises(PacketBuildError, match="COVERAGE_INFEASIBLE: required coverage needs at least 2"):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


def test_distinct_kind_requirements_raise_the_minimum(benchmark):
    policy = _policy_with(primary_reconciliation={"statement_kinds": ["fact"]},
                          material_unresolved_item={"statement_kinds": ["explicit_claim"],
                                                    "any_of": [{"reconciliation_ids": ["R04"]}]})
    assert build_brief_packet(benchmark, generated_at=T0, policy=policy).coverage_policy.min_statement_slots == 4
    policy["statement_count"] = {"min": 1, "max": 3}
    with pytest.raises(PacketBuildError, match="COVERAGE_INFEASIBLE: required coverage needs at least 4"):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


@pytest.mark.parametrize("achievable, slots", [
    ([{K.fact, K.interpretive}, {K.interpretive}, {K.limitation}], 2),
    ([{K.fact}, {K.limitation}, {K.interpretive}, {K.explicit_claim}], 4),
    ([{K.fact, K.interpretive, K.limitation}], 1),
])
def test_minimum_statement_slots(achievable, slots):
    assert minimum_statement_slots([frozenset(a) for a in achievable]) == slots


def test_uncomputable_fact_drops_only_its_alternative(benchmark):
    s02 = benchmark.fact_map["S02"]
    degraded = build_brief_packet(benchmark.with_fact(s02.model_copy(update={"value": Decimal("0.4")})),
                                  generated_at=T0)
    demand = next(r for r in degraded.coverage_policy.roles if r.role == "demand_signal")
    assert [a.all_ids() for a in demand.any_of] == [["S21", "S22"]]


def test_role_with_only_uncomputable_alternatives_blocks_the_packet(benchmark):
    s02 = benchmark.fact_map["S02"]
    policy = _policy_with(demand_signal={"any_of": [{"fact_ids": ["D08", "D07"]}]})
    with pytest.raises(PacketBuildError, match="every alternative names a fact this packet cannot compute"):
        build_brief_packet(benchmark.with_fact(s02.model_copy(update={"value": Decimal("0.4")})),
                           generated_at=T0, policy=policy)


# -------------------------------------------------------------- synthesis ---

def _model_output(brief: dict) -> dict:
    out = copy.deepcopy(brief)
    del out["title"]
    out["validation_metadata"]["author"] = PROMPT_VERSION
    return out


MISSING_KEY = _edit(_valid(), "V4", premise_ids=["S23", "S25", "S21", "S22", "N03"])


def test_missing_role_is_named_on_the_single_retry_and_fixed(packet):
    client = MockLLMClient([_model_output(MISSING_KEY), _model_output(_valid())])
    result = synthesize(packet, client, now=lambda: T0)
    assert result.accepted and [a.status for a in result.attempts] == ["REJECTED", "VALID"]
    corrections = client.calls[1].corrections
    assert [(e.code, e.subject_id) for e in corrections] == [("MISSING_REQUIRED_COVERAGE", "key_interpretation")]
    content = client.calls[1].messages[0]["content"]
    assert "key_interpretation" in content and "narrative_ids N15" in content
    assert "We read the change between Q1 and Q2 2026" not in content  # the rejected brief is never resent


def test_second_response_missing_coverage_hard_rejects_after_two_calls(packet):
    bad = _model_output(MISSING_KEY)
    client = MockLLMClient([bad, bad, bad])
    result = synthesize(packet, client, now=lambda: T0)
    assert result.status is SynthesisStatus.SYNTHESIS_REJECTED and len(client.calls) == 2
    assert {(e.code, e.subject_id) for e in result.errors} == {("MISSING_REQUIRED_COVERAGE", "key_interpretation")}
