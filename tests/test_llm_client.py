"""The live Anthropic client, exercised against an in-process fake transport. No network, no key."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")
import httpx2  # noqa: E402  (installed with the anthropic SDK)

from fpa_engine.brief import ExecutiveBrief  # noqa: E402
from fpa_engine.llm import ANTHROPIC_API_URL, AnthropicLLMClient, LLMSettings, brief_output_schema  # noqa: E402
from fpa_engine.synthesizer import (  # noqa: E402
    SYSTEM_PROMPT,
    LLMProviderError,
    LLMRefusalError,
    LLMTimeoutError,
    MalformedOutputError,
    SynthesisStatus,
    synthesize,
)
from test_synthesis import _fixture  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1] / "fpa_engine"
SETTINGS = LLMSettings(model="claude-opus-5-5", api_key="sk-test-not-a-real-key")
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class FakeSDK:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params):
        self.requests.append(params)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _response(text: str, stop_reason: str = "end_turn", model: str = "claude-opus-5-5"):
    return SimpleNamespace(model=model, stop_reason=stop_reason,
                           content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)])


def _client(*outcomes) -> tuple[AnthropicLLMClient, FakeSDK]:
    fake = FakeSDK(*outcomes)
    return AnthropicLLMClient(SETTINGS, sdk_client=fake), fake


def test_request_uses_structured_output_and_central_settings(packet):
    client, fake = _client(_response(json.dumps(_fixture("valid_brief.json"))))
    client.generate_structured_brief(packet)
    params = fake.requests[0]
    assert params["model"] == "claude-opus-5-5" and params["system"] == SYSTEM_PROMPT
    assert params["output_config"]["format"] == {"type": "json_schema", "schema": brief_output_schema()}
    assert "fallbacks" not in params and "betas" not in params, "refusal fallback must stay disabled"
    assert "sk-test-not-a-real-key" not in json.dumps(params)


def test_output_schema_is_closed_and_matches_the_contract():
    schema = brief_output_schema()
    objects = [schema, *schema["$defs"].values()]
    for node in objects:
        if node.get("type") == "object":
            assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
    assert set(schema["properties"]) == set(ExecutiveBrief.model_fields) - {"title"}
    assert "pattern" not in json.dumps(schema) and "minLength" not in json.dumps(schema)


def test_live_client_end_to_end_with_fake_transport(packet):
    client, _ = _client(_response(json.dumps(_fixture("valid_brief.json")), model="claude-opus-5-5"))
    result = synthesize(packet, client)
    assert result.status is SynthesisStatus.ACCEPTED and result.model == "claude-opus-5-5"


def test_live_client_retry_sends_errors_not_previous_output(packet):
    bad = _fixture("valid_brief.json")
    bad["validation_metadata"]["author"] = "REJECTED-DRAFT-MARKER"
    bad["statements"][6]["text"] = "Q2 2026 CI segment profit rose $704M year over year."
    client, fake = _client(_response(json.dumps(bad)), _response(json.dumps(_fixture("valid_brief.json"))))
    result = synthesize(packet, client)
    assert result.accepted and len(fake.requests) == 2
    retry = fake.requests[1]["messages"]
    assert len(retry) == 1 and "UNSUPPORTED_NUMBER" in retry[0]["content"]
    assert "REJECTED-DRAFT-MARKER" not in retry[0]["content"] and "rose $704M" not in retry[0]["content"]


@pytest.mark.parametrize("exc, expected", [
    (anthropic.APITimeoutError(request=REQUEST), LLMTimeoutError),
    (anthropic.APIConnectionError(request=REQUEST), LLMProviderError),
    (anthropic.InternalServerError("overloaded", response=httpx2.Response(529, request=REQUEST), body=None),
     LLMProviderError),
    (anthropic.AuthenticationError("invalid x-api-key", response=httpx2.Response(401, request=REQUEST), body=None),
     LLMProviderError),
])
def test_sdk_errors_become_provider_errors(packet, exc, expected):
    client, _ = _client(exc)
    with pytest.raises(expected) as info:
        client.generate_structured_brief(packet)
    assert "sk-test-not-a-real-key" not in str(info.value)


def test_401_invalid_key_run_record_is_written_without_secret_markers(tmp_path):
    """A 401 body says "invalid x-api-key"; only status, class and structured error type reach the run record, so
    the unchanged secret guard writes it instead of aborting the gate. Real SDK client, fake transport, no network."""
    from fpa_engine import gate_c2

    key = "sk-test-not-a-real-key"
    body = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    transport = httpx2.MockTransport(lambda request: httpx2.Response(401, json=body))
    sdk = anthropic.Anthropic(api_key=key, base_url=ANTHROPIC_API_URL, max_retries=0,
                              http_client=httpx2.Client(transport=transport))
    settings = LLMSettings(api_key=key, model="claude-opus-5-5")
    out = tmp_path / "out"
    summary = gate_c2.evaluate(1, out, lambda: AnthropicLLMClient(settings, sdk_client=sdk), (key,))
    assert summary["provider_error_rate"] == 1.0
    text = (out / "run_01.json").read_text(encoding="utf-8")
    assert key not in text and not any(m in text.lower() for m in gate_c2._SECRET_MARKERS)
    record = json.loads(text)
    assert [e["message"] for e in record["final_errors"]] == ["HTTP 401: AuthenticationError (authentication_error)"]


def test_refusal_and_truncation_are_not_accepted(packet):
    client, _ = _client(_response("", stop_reason="refusal"))
    with pytest.raises(LLMRefusalError, match="no fallback"):
        client.generate_structured_brief(packet)
    client, _ = _client(_response('{"title": "t", "statem', stop_reason="max_tokens"))
    with pytest.raises(MalformedOutputError, match="max_tokens"):
        client.generate_structured_brief(packet)


def test_timeout_through_the_flow_returns_provider_error(packet):
    client, fake = _client(anthropic.APITimeoutError(request=REQUEST))
    result = synthesize(packet, client)
    assert result.status is SynthesisStatus.PROVIDER_ERROR and result.brief is None
    assert [e.code for e in result.errors] == ["PROVIDER_TIMEOUT"] and len(fake.requests) == 1


def test_provider_sdk_is_isolated_to_llm_module():
    pattern = re.compile(r"^\s*(?:import|from)\s+(anthropic|openai|httpx2?|requests|langchain|llama_index)\b", re.M)
    importers = {p.name for p in PACKAGE.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))}
    assert importers == {"llm.py"}


def test_refusal_through_the_flow_is_an_explicit_error(packet):
    client, fake = _client(_response("", stop_reason="refusal"), _response(json.dumps(_fixture("valid_brief.json"))))
    result = synthesize(packet, client)
    assert result.status is SynthesisStatus.PROVIDER_ERROR and result.brief is None
    assert [e.code for e in result.errors] == ["PROVIDER_REFUSAL"] and len(fake.requests) == 1


def test_a_different_served_model_is_rejected(packet):
    client, _ = _client(_response(json.dumps(_fixture("valid_brief.json")), model="claude-sonnet-5-5"))
    with pytest.raises(LLMProviderError, match="differs from configured"):
        client.generate_structured_brief(packet)


def test_inherited_base_url_does_not_redirect_the_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://attacker.example.invalid")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token")
    client = AnthropicLLMClient(SETTINGS)
    sdk = client._client
    assert str(sdk.base_url).rstrip("/") == ANTHROPIC_API_URL == "https://api.anthropic.com"
    assert sdk.api_key == SETTINGS.api_key and sdk.auth_token is None


def test_ambient_custom_headers_do_not_reach_the_production_client(monkeypatch):
    hostile = "X-Exfil: harvest\nAuthorization: Bearer stolen\nanthropic-beta: server-side-fallback-2026-07-01"
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", hostile)
    sdk = AnthropicLLMClient(SETTINGS)._client
    sent = {k.lower(): str(v) for k, v in sdk.default_headers.items()}
    assert "x-exfil" not in sent and "authorization" not in sent and "anthropic-beta" not in sent
    assert "stolen" not in json.dumps(sent) and "harvest" not in json.dumps(sent)
    assert sent["x-api-key"] == SETTINGS.api_key


def test_the_sdk_does_read_custom_headers_when_not_isolated(monkeypatch):
    """Control: proves the previous test is meaningful, i.e. a plain SDK client would have sent the header."""
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Exfil: harvest")
    plain = anthropic.Anthropic(api_key="sk-test-not-a-real-key")
    assert plain.default_headers.get("X-Exfil") == "harvest"


def test_header_injection_remains_possible_through_explicit_injection(packet):
    injected = FakeSDK(_response(json.dumps(_fixture("valid_brief.json"))))
    client = AnthropicLLMClient(SETTINGS, sdk_client=injected)
    assert client._client is injected and client.generate_structured_brief(packet)
