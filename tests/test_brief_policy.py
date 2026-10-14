"""Gate C hardening of the brief contract: deterministic titles, statement bounds, policy-driven surfacing,
no forbidden-value exemption, and neutral packet terminology."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest

from fpa_engine.brief import ExecutiveBrief, PacketBuildError, brief_title, build_brief_packet, load_policy
from fpa_engine.brief_demo import fixture_dir

T0 = datetime(2026, 10, 12, tzinfo=timezone.utc)
CURRENT_HASH = "2e45fe103a5146b4c3aa8d3bce07b0da7b60ca17b516e06c3306ad59864f2dec"


def _valid_brief() -> ExecutiveBrief:
    return ExecutiveBrief.model_validate(json.loads((fixture_dir() / "valid_brief.json").read_text(encoding="utf-8")))


def _validate(packet, brief):
    from fpa_engine.brief_validator import validate_brief

    return validate_brief(packet, brief)


def test_packet_hash_is_pinned(packet):
    assert packet.packet_hash == CURRENT_HASH


# ---------------------------------------------------------------- title ---

def test_title_is_built_from_packet_metadata(packet):
    assert brief_title(packet) == "Caterpillar Q2 2026 — Construction Industries Executive Brief"
    renamed = packet.model_copy(update={"company": "Example Holdings Corp.", "period": "Q3 2026"})
    assert brief_title(renamed) == "Example Holdings Q3 2026 — Construction Industries Executive Brief"


@pytest.mark.parametrize("title", [
    "Caterpillar Q2 2026 — CI profit rose $999M",
    "Caterpillar Q2 2026 — tariff drag of 343 bps",
    "Caterpillar Q2 2026 — Construction Industries Executive Brief ",
])
def test_validator_rejects_any_other_title(packet, title):
    brief = _valid_brief().model_copy(update={"title": title})
    assert _validate(packet, brief).codes == {"TITLE_NOT_DETERMINISTIC"}


# ---------------------------------------------------------- terminology ---

def test_packet_carries_no_workbook_terminology(packet):
    dumped = packet.model_dump_json()
    assert "workbook" not in dumped.lower()
    fm = packet.fact_map
    assert fm["D18"].basis == fm["D25"].basis == "derived_distortion_method"
    assert "derived-method bps" in packet.reconciliation_map["NA06"].reason


def test_relabeling_does_not_change_values(packet, result):
    for fid in ("D18", "D25"):
        assert packet.fact_map[fid].value == result.facts[fid].value


def test_unmapped_internal_terms_block_the_packet(benchmark):
    policy = load_policy("cat_q2_2026")
    policy = {**policy, "terminology": {}}
    with pytest.raises(PacketBuildError, match="INTERNAL_TERMINOLOGY"):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


# ------------------------------------------------------ surfacing policy ---

def test_surfacing_policy_is_carried_in_the_packet(packet):
    sp = packet.surfacing_policy
    assert sp.mandatory_reconciliation_ids == ["R03", "R04", "R05"]
    assert sp.mandatory_disclosure_gap_ids == ["G04", "G05", "G06", "G07"]
    assert sp.mandatory_comparability_note_ids == ["CN06"]
    assert set(sp.conditional_narrative_ids) == {n.id for n in packet.narratives if n.requires_limitation}
    assert (packet.statement_bounds.min, packet.statement_bounds.max) == (5, 12)


@pytest.mark.parametrize("field, value, message", [
    ("mandatory_reconciliation_ids", ["R01"], "R01 is not an UNRESOLVED reconciliation"),
    ("mandatory_disclosure_gap_ids", ["RV-N03"], "RV-N03 is not a disclosure gap"),
    ("mandatory_comparability_note_ids", ["CN99"], "CN99 is not a comparability note"),
    ("conditional_narrative_ids", ["N03"], "conditional_narrative_ids must be exactly"),
])
def test_invalid_surfacing_policy_blocks_the_packet(benchmark, field, value, message):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["surfacing_policy"][field] = value
    with pytest.raises(PacketBuildError, match=message):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


def test_invalid_statement_bounds_block_the_packet(benchmark):
    policy = copy.deepcopy(load_policy("cat_q2_2026"))
    policy["statement_count"] = {"min": 9, "max": 5}
    with pytest.raises(PacketBuildError, match="STATEMENT_BOUNDS_INVALID"):
        build_brief_packet(benchmark, generated_at=T0, policy=policy)


@pytest.mark.parametrize("subject, field, side_effects", [
    # V9 quotes R05's $14M residual, so dropping the citation also orphans that number.
    ("R05", "reconciliation_ids", {("UNSUPPORTED_NUMBER", None)}),
    # V7 cites D11, which G05 is linked to, so the existing linked-gap rule fires too.
    ("G05", "flag_ids", {("MISSING_DISCLOSURE_GAP", "G05")}),
    ("CN06", "comparability_note_ids", set()),
])
def test_each_mandatory_item_is_enforced(packet, subject, field, side_effects):
    brief = _valid_brief()
    statements = [s.model_copy(update={field: [i for i in getattr(s, field) if i != subject]})
                  for s in brief.statements]
    result = _validate(packet, brief.model_copy(update={"statements": statements}))
    assert {(e.code, e.subject_id) for e in result.errors} == {("MISSING_MANDATORY_LIMITATION", subject),
                                                               *side_effects}


def test_mandatory_items_count_only_in_limitations(packet):
    brief = _valid_brief()
    v9 = next(s for s in brief.statements if s.id == "V9")
    statements = [s.model_copy(update={"kind": "fact", "fact_ids": ["D12"]}) if s is v9 else s
                  for s in brief.statements]
    found = {(e.code, e.subject_id) for e in _validate(packet, brief.model_copy(update={"statements": statements})).errors}
    assert {("MISSING_MANDATORY_LIMITATION", "R03"), ("MISSING_MANDATORY_LIMITATION", "R05")} <= found


def test_unknown_comparability_note_is_rejected(packet):
    brief = _valid_brief()
    extra = brief.statements[-1].model_copy(update={"id": "V10", "text": "A note that does not exist.",
                                                    "reconciliation_ids": [], "comparability_note_ids": ["CN99"]})
    result = _validate(packet, brief.model_copy(update={"statements": [*brief.statements, extra]}))
    assert {(e.code, e.subject_id) for e in result.errors} == {("UNKNOWN_COMPARABILITY_NOTE_ID", "CN99")}
