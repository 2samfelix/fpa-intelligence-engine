"""Gate C2 runner, mock clients only: no test here builds a live client or opens a connection."""
from __future__ import annotations

import json

import pytest

import fpa_engine.llm as llm
from fpa_engine import gate_c2
from fpa_engine.brief import brief_title
from fpa_engine.brief_demo import fixture_dir
from fpa_engine.llm import MockLLMClient
from fpa_engine.synthesizer import PROMPT_VERSION

FAKE_KEY = "fake-test-key-not-a-secret-0000"
MANDATORY = ["R03", "R04", "R05", "G04", "G05", "G06", "G07", "CN06"]


def _fixture(name: str) -> dict:
    """A B2 fixture in model-output shape: statements and metadata only."""
    brief = json.loads((fixture_dir() / name).read_text(encoding="utf-8"))
    del brief["title"]
    brief["validation_metadata"]["author"] = PROMPT_VERSION
    return brief


GOOD = _fixture("valid_brief.json")
STALE = _fixture("invalid_08_stale_343_bps.json")
UNSUPPORTED = _fixture("invalid_03_unsupported_number.json")


def _clients(scripts, served=None, model="mock-model"):
    """A make_client factory handing out one scripted MockLLMClient per run."""
    scripts, served, made = iter(scripts), iter(served or []), []

    def make():
        client = MockLLMClient(next(scripts), model=model)
        client.last_served_model = next(served, model)
        made.append(client)
        return client

    make.made = made
    return make


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------ preflight ---

def test_pins_match_the_current_packet_and_prompt(packet):
    assert packet.packet_hash == gate_c2.EXPECTED_PACKET_HASH
    assert PROMPT_VERSION == gate_c2.EXPECTED_PROMPT_VERSION


@pytest.mark.parametrize("runs", [0, -1, 6])
def test_run_count_outside_bounds_is_refused_before_any_client_or_file(tmp_path, runs):
    make = _clients([])
    with pytest.raises(gate_c2.GateError, match="runs must be 1..5"):
        gate_c2.evaluate(runs, tmp_path / "out", make)
    assert make.made == [] and not (tmp_path / "out").exists()


def test_packet_hash_mismatch_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_c2, "EXPECTED_PACKET_HASH", "0" * 64)
    make = _clients([])
    with pytest.raises(gate_c2.GateError, match="packet hash"):
        gate_c2.evaluate(1, tmp_path / "out", make)
    assert make.made == [] and not (tmp_path / "out").exists()


def test_prompt_version_mismatch_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_c2, "PROMPT_VERSION", "synthesis_v3")
    make = _clients([])
    with pytest.raises(gate_c2.GateError, match="prompt version"):
        gate_c2.evaluate(1, tmp_path / "out", make)
    assert make.made == [] and not (tmp_path / "out").exists()


def test_existing_output_directory_is_not_overwritten(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        gate_c2.evaluate(1, tmp_path / "out", _clients([[GOOD]]))


# -------------------------------------------------------------- report ---

def test_mixed_runs_are_scored_and_written(tmp_path):
    make = _clients([[GOOD], [STALE, GOOD], [UNSUPPORTED, STALE], [GOOD]],
                    served=["mock-model", "mock-model", "mock-model", "other-model"])
    summary = gate_c2.evaluate(4, tmp_path / "out", make)

    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "run_01.json", "run_02.json", "run_03.json", "run_04.json", "summary.json"]
    assert _read(tmp_path / "out" / "summary.json") == json.loads(json.dumps(summary, default=str))
    assert summary["report_only"] == {"first_pass_acceptance_rate": 0.5, "retry_rate": 0.5,
                                      "recurring_validator_errors": ["FORBIDDEN_VALUE"]}
    assert summary["final_acceptance_rate"] == 0.75
    assert summary["hard_rejection_rate"] == 0.25
    assert summary["number_issues_any_attempt"] == {"UNSUPPORTED_NUMBER": 1, "FORBIDDEN_VALUE": 2,
                                                    "ARITHMETIC_IN_PROSE": 0}
    assert summary["number_issues_in_accepted_briefs"] == []
    assert summary["gate_criteria"] == {"5_of_5_final_accepted": False, "zero_missing_mandatory_limitations": True,
                                        "zero_model_mismatches": False}
    assert summary["gate_c2_passed"] is False

    run2, run3, run4 = (_read(tmp_path / "out" / f"run_0{n}.json") for n in (2, 3, 4))
    assert run2["attempt_1_status"] == "REJECTED" and run2["retry_used"] and run2["final_status"] == "ACCEPTED"
    assert run3["final_status"] == "SYNTHESIS_REJECTED" and run3["brief"] is None
    assert run3["missing_mandatory_limitations"] is None
    assert len(run3["attempts"]) == 2
    assert run4["served_model"] == "other-model" and run4["model_match"] is False


def test_accepted_run_surfaces_every_mandatory_limitation(tmp_path):
    gate_c2.evaluate(1, tmp_path / "out", _clients([[GOOD]]))
    run = _read(tmp_path / "out" / "run_01.json")
    assert set(MANDATORY) <= set(run["limitation_ids_surfaced"])
    assert run["missing_mandatory_limitations"] == []
    assert run["accepted_brief_recheck_codes"] == []
    assert run["forbidden_value_count"] == run["unsupported_number_count"] == 0
    assert run["brief"]["title"] == "Caterpillar Q2 2026 — Construction Industries Executive Brief"


def test_five_of_five_accepted_passes_with_retry_as_report_only(tmp_path, packet):
    make = _clients([[GOOD], [STALE, GOOD], [GOOD], [GOOD], [GOOD]])
    summary = gate_c2.evaluate(5, tmp_path / "out", make)
    assert summary["gate_c2_passed"] is True
    assert all(summary["gate_criteria"].values())
    assert summary["report_only"]["first_pass_acceptance_rate"] == 0.8
    assert summary["report_only"]["retry_rate"] == 0.2
    assert set(MANDATORY) <= set(summary["consistency"]["limitation_ids_surfaced"]["in_every_accepted_run"])
    assert all(len(c.calls) <= 2 for c in make.made)
    assert {c.calls[0].packet_hash for c in make.made} == {packet.packet_hash}


def test_fewer_than_five_runs_never_pass_the_gate(tmp_path):
    summary = gate_c2.evaluate(1, tmp_path / "out", _clients([[GOOD]]))
    assert summary["final_acceptance_rate"] == 1.0
    assert summary["gate_criteria"]["5_of_5_final_accepted"] is False
    assert summary["gate_c2_passed"] is False


def test_missing_mandatory_limitation_fails_the_gate(packet):
    record = {"final_status": "ACCEPTED", "attempt_1_status": "VALID", "retry_used": False, "attempts": [],
              "missing_mandatory_limitations": ["R04"], "model_match": True, "configured_model": "m",
              "served_model": "m", "number_issues_by_attempt": [], "accepted_brief_recheck_codes": [],
              "fact_ids": [], "reconciliation_ids": [], "limitation_ids_surfaced": [], "run": 1,
              "statement_count": 1, "forbidden_value_count": 0, "unsupported_number_count": 0}
    summary = gate_c2.summarize([dict(record, run=n) for n in range(1, 6)], packet)
    assert summary["gate_criteria"]["5_of_5_final_accepted"] is True
    assert summary["gate_criteria"]["zero_missing_mandatory_limitations"] is False
    assert summary["gate_c2_passed"] is False


# ------------------------------------------------------------- secrets ---

def test_report_containing_the_key_is_refused(tmp_path):
    make = _clients([[GOOD]], served=[FAKE_KEY])
    with pytest.raises(gate_c2.GateError, match="credential"):
        gate_c2.evaluate(1, tmp_path / "out", make, (FAKE_KEY,))
    assert list((tmp_path / "out").iterdir()) == []


@pytest.mark.parametrize("text", ["x-api-key: abc", "Authorization: Bearer abc", "ANTHROPIC_CUSTOM_HEADERS=a"])
def test_header_markers_are_refused(tmp_path, text):
    with pytest.raises(gate_c2.GateError):
        gate_c2._write_json(tmp_path / "r.json", {"text": text}, ())
    assert not (tmp_path / "r.json").exists()


def test_ordinary_prose_about_authorizations_is_written(tmp_path):
    gate_c2._write_json(tmp_path / "r.json", {"text": "The share repurchase authorization — unchanged."}, (FAKE_KEY,))
    assert _read(tmp_path / "r.json")["text"].endswith("authorization — unchanged.")


# ----------------------------------------------------------------- cli ---

def test_cli_without_key_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert gate_c2.main(["--runs", "1", "--out", str(tmp_path / "out")]) == 2
    out, err = capsys.readouterr()
    assert "GATE_C2_REFUSED: ANTHROPIC_API_KEY is not set" in err
    assert "Gate C2 |" not in out and not (tmp_path / "out").exists()


def test_cli_refuses_bad_run_count_before_reading_the_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert gate_c2.main(["--runs", "6", "--out", str(tmp_path / "out")]) == 2
    assert "runs must be 1..5" in capsys.readouterr().err


def test_cli_prints_banner_before_first_call_and_never_the_key(tmp_path, monkeypatch, capsys, packet):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("ANTHROPIC_MODEL", "mock-model")
    seen_before_call = []

    class FakeLiveClient(MockLLMClient):
        def __init__(self, settings):
            super().__init__([GOOD], model=settings.model)
            self.last_served_model = settings.model

        def generate_structured_brief(self, packet, *, corrections=()):
            seen_before_call.append(capsys.readouterr().out)
            return super().generate_structured_brief(packet, corrections=corrections)

    monkeypatch.setattr(llm, "AnthropicLLMClient", FakeLiveClient)
    assert gate_c2.main(["--runs", "1", "--out", str(tmp_path / "out")]) == 1  # 1 run cannot pass a 5-run gate
    out, err = capsys.readouterr()

    assert seen_before_call[0].strip() == (
        f"Gate C2 | configured model: mock-model | packet hash: {packet.packet_hash} | "
        f"prompt version: {PROMPT_VERSION} | runs: 1")
    texts = [seen_before_call[0], out, err, *(p.read_text(encoding="utf-8") for p in (tmp_path / "out").iterdir())]
    assert all(FAKE_KEY not in t for t in texts)
    assert brief_title(packet) in (tmp_path / "out" / "run_01.json").read_text(encoding="utf-8")
