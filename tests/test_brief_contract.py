from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from fpa_engine.brief import ExecutiveBrief, StatementKind
from fpa_engine.brief_demo import fixture_dir, load_brief, main, validate_fixture
from fpa_engine.schemas import Origin

FIXTURES = fixture_dir()
PACKAGE = Path(__file__).resolve().parents[1] / "fpa_engine"

EXPECTED_REJECTIONS = {
    "invalid_01_unknown_fact_id.json": {"UNKNOWN_FACT_ID"},
    "invalid_02_unknown_evidence_id.json": {"UNKNOWN_EVIDENCE_ID"},
    "invalid_03_unsupported_number.json": {"UNSUPPORTED_NUMBER"},
    "invalid_04_arithmetic_in_prose.json": {"ARITHMETIC_IN_PROSE"},
    "invalid_05_interpretive_as_explicit.json": {"INTERPRETIVE_AS_EXPLICIT", "ASSUMPTION_AS_FACT"},
    "invalid_06_unresolved_as_reconciled.json": {"UNRESOLVED_PRESENTED_AS_RECONCILED"},
    "invalid_07_na_relationship_additive.json": {"NA_RELATIONSHIP_AS_ADDITIVE"},
    "invalid_08_stale_343_bps.json": {"FORBIDDEN_VALUE"},
    "invalid_09_stale_570_bps.json": {"FORBIDDEN_VALUE"},
    "invalid_10_missing_r04_limitation.json": {"MISSING_RECONCILIATION_LIMITATION", "MISSING_MANDATORY_LIMITATION"},
    "invalid_11_omitted_disclosure_gap.json": {"MISSING_DISCLOSURE_GAP"},
    "invalid_12_missing_premises.json": {"MISSING_PREMISES"},
    "invalid_13_derived_as_disclosed.json": {"DERIVED_PRESENTED_AS_DISCLOSED"},
    "invalid_14_unsupported_range_estimate.json": {"FORBIDDEN_VALUE"},
}
EXPECTED_SUBJECTS = {
    "invalid_01_unknown_fact_id.json": "D999",
    "invalid_08_stale_343_bps.json": "FV01",
    "invalid_09_stale_570_bps.json": "FV03",
    "invalid_10_missing_r04_limitation.json": "R04",
    "invalid_11_omitted_disclosure_gap.json": "G02",
    "invalid_14_unsupported_range_estimate.json": "FV05",
}


def test_every_fixture_has_an_expectation():
    on_disk = {p.name for p in FIXTURES.glob("invalid_*.json")}
    assert on_disk == set(EXPECTED_REJECTIONS)
    assert len(on_disk) >= 12


def test_valid_brief_passes(packet):
    result = validate_fixture(packet, FIXTURES / "valid_brief.json")
    assert result.valid, [f"{e.code} {e.subject_id} {e.message}" for e in result.errors]
    assert result.status == "VALID"


def test_valid_brief_exercises_every_statement_kind(packet):
    brief = load_brief(FIXTURES / "valid_brief.json")
    kinds = {s.kind for s in brief.statements}
    assert kinds == set(StatementKind)
    fm = packet.fact_map
    fact_ids = {fid for s in brief.statements if s.kind is StatementKind.fact for fid in s.fact_ids}
    assert any(fm[f].origin is Origin.sourced for f in fact_ids)
    assert any(fm[f].origin is Origin.derived for f in fact_ids)


@pytest.mark.parametrize("name", sorted(EXPECTED_REJECTIONS))
def test_invalid_brief_is_rejected_for_the_expected_reason(packet, name):
    result = validate_fixture(packet, FIXTURES / name)
    assert not result.valid
    assert result.codes == EXPECTED_REJECTIONS[name], [f"{e.code}: {e.message}" for e in result.errors]
    if name in EXPECTED_SUBJECTS:
        assert EXPECTED_SUBJECTS[name] in {e.subject_id for e in result.errors}


def _brief_json(**statement) -> dict:
    base = {"id": "A", "kind": "fact", "text": "CI sales were $8,346M.", "fact_ids": ["S01"]}
    return {"title": "t", "validation_metadata": {"analysis_id": "cat-q2-2026-ci", "packet_hash": "a" * 64,
                                                   "author": "test"},
            "statements": [{**base, **statement}]}


def test_schema_baseline_is_well_formed():
    ExecutiveBrief.model_validate(_brief_json())


def test_every_fixture_is_bound_to_the_current_packet(packet):
    for path in FIXTURES.glob("*.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))["validation_metadata"]
        assert meta["packet_hash"] == packet.packet_hash, (
            f"{path.name} carries a stale packet_hash; re-stamp fixtures after any packet change")


@pytest.mark.parametrize("override", [
    {"kind": "opinion"},
    {"confidence": 0.9},
    {"text": ""},
    {"fact_ids": "S01"},
])
def test_statement_schema_rejects_malformed_output(override):
    with pytest.raises(ValidationError):
        ExecutiveBrief.model_validate(_brief_json(**override))


def test_brief_requires_statements_and_metadata():
    with pytest.raises(ValidationError):
        ExecutiveBrief.model_validate({"title": "t", "statements": [], "validation_metadata": {
            "analysis_id": "cat-q2-2026-ci", "packet_hash": "a" * 64, "author": "test"}})
    with pytest.raises(ValidationError):
        ExecutiveBrief.model_validate({"title": "t", "statements": [_brief_json()["statements"][0]]})


def test_brief_is_structured_json_not_markdown():
    assert set(ExecutiveBrief.model_fields) == {"title", "statements", "validation_metadata"}
    with pytest.raises(ValidationError):
        ExecutiveBrief.model_validate({**_brief_json(), "markdown": "# Brief"})


def test_demo_cli_reports_real_results(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("CAT Q2 2026 — Brief Contract")
    assert "Packet facts: 65" in out and "Narratives: 17" in out and "Reconciliations: 13" in out
    assert "Open flags: 16" in out
    assert "valid_brief.json:\nVALID" in out
    assert "invalid_01_unknown_fact_id.json:\nREJECTED\n- UNKNOWN_FACT_ID: D999" in out


def test_demo_can_dump_the_packet(tmp_path, capsys):
    target = tmp_path / "packet.json"
    assert main([str(FIXTURES / "valid_brief.json"), "--dump-packet", str(target)]) == 0
    dumped = json.loads(target.read_text(encoding="utf-8"))
    assert len(dumped["facts"]) == 65
    assert "valid_brief.json:\nVALID" in capsys.readouterr().out


def test_no_llm_integration_in_package():
    # Gate C: the provider SDK is an optional extra and may be imported only by fpa_engine/llm.py.
    pyproject = (PACKAGE.parent / "pyproject.toml").read_text(encoding="utf-8")
    deps = re.search(r"^dependencies = \[(.*?)\]", pyproject, re.M).group(1)
    assert deps.strip() == '"pydantic>=2,<3"'
    pattern = re.compile(r"^\s*(?:import|from)\s+(anthropic|openai|httpx|requests|langchain)\b", re.M)
    for path in PACKAGE.rglob("*.py"):
        if path.name != "llm.py":
            assert not pattern.search(path.read_text(encoding="utf-8")), path.name
