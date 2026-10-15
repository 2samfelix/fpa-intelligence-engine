"""LLM clients for structured synthesis. All provider-specific code lives in this module.

The provider SDK is an optional extra (`pip install -e .[llm]`) and is imported only when a live client is built.
"""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, Union

from fpa_engine.brief import BriefPacket
from fpa_engine.brief_validator import BriefIssue
from fpa_engine.synthesizer import (
    LLMProviderError,
    LLMRefusalError,
    LLMTimeoutError,
    MalformedOutputError,
    SynthesisOutput,
    build_messages,
    parse_model_output,
)

DEFAULT_MODEL = "claude-opus-5-5"
# Production calls always go here. An inherited ANTHROPIC_BASE_URL is ignored; tests inject a client instead.
ANTHROPIC_API_URL = "https://api.anthropic.com"


class LLMConfigError(Exception):
    pass


@dataclass(frozen=True)
class LLMSettings:
    """The one place model configuration is read. The key is never part of repr or logs."""
    model: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LLMSettings:
        env = os.environ if environ is None else environ
        key = env.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise LLMConfigError("ANTHROPIC_API_KEY is not set")
        return cls(model=env.get("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL, api_key=key)


# ----------------------------------------------------------------- mock ---

MockResponse = Union[dict, str, Exception, Callable[[BriefPacket], Any]]


@dataclass(frozen=True)
class MockCall:
    packet_hash: str
    corrections: tuple[BriefIssue, ...]
    system: str
    messages: list[dict]


class MockLLMClient:
    """Deterministic scripted client: returns its responses in order and records the exact prompt of every call."""

    def __init__(self, responses: Sequence[MockResponse], model: str = "mock-deterministic"):
        self.model = model
        self._responses = list(responses)
        self.calls: list[MockCall] = []

    def generate_structured_brief(self, packet: BriefPacket, *, corrections: Sequence[BriefIssue] = ()) -> dict:
        system, messages = build_messages(packet, corrections)
        self.calls.append(MockCall(packet.packet_hash, tuple(corrections), system, messages))
        if not self._responses:
            raise LLMProviderError("mock client has no scripted response left")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(packet)
        if isinstance(response, str):
            return parse_model_output(response)
        return copy.deepcopy(response)


# ------------------------------------------------------------ anthropic ---

_UNSUPPORTED_SCHEMA_KEYS = {"pattern", "minLength", "maxLength", "minItems", "maxItems", "default", "title"}


def brief_output_schema() -> dict:
    """SynthesisOutput (statements and metadata, no title) as a structured-output JSON schema: every object
    closed and fully required, and constraints the API does not enforce dropped. Those are still enforced
    afterwards by schema validation and the deterministic validator."""
    def clean(node: Any) -> Any:
        if isinstance(node, list):
            return [clean(n) for n in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for key, value in node.items():
            if key == "properties":
                out[key] = {name: clean(sub) for name, sub in value.items()}
            elif key == "$defs":
                out[key] = {name: clean(sub) for name, sub in value.items()}
            elif key not in _UNSUPPORTED_SCHEMA_KEYS:
                out[key] = clean(value)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return clean(SynthesisOutput.model_json_schema())


def _production_sdk_client(anthropic: Any, settings: LLMSettings, timeout: float, max_retries: int) -> Any:
    """The SDK client used in production, configured only from engine settings.

    The SDK reads ANTHROPIC_BASE_URL and ANTHROPIC_CUSTOM_HEADERS from the environment. The endpoint is pinned
    explicitly; ambient custom headers are discarded and the result verified, failing closed if the SDK changes."""
    client = anthropic.Anthropic(api_key=settings.api_key, base_url=ANTHROPIC_API_URL, timeout=timeout,
                                 max_retries=max_retries, default_headers={})
    if not isinstance(getattr(client, "_custom_headers", None), dict):
        raise LLMConfigError("cannot isolate the SDK client from ANTHROPIC_CUSTOM_HEADERS; check the anthropic version")
    client._custom_headers = {}
    ambient = _parse_header_env(os.environ.get("ANTHROPIC_CUSTOM_HEADERS", ""))
    sent = {k.lower(): v for k, v in client.default_headers.items()}
    leaked = sorted(k for k, v in ambient.items() if sent.get(k.lower()) == v)
    if leaked:
        raise LLMConfigError(f"ambient ANTHROPIC_CUSTOM_HEADERS still applied to the client: {leaked}")
    return client


def _parse_header_env(raw: str) -> dict[str, str]:
    """Header names/values as the SDK parses ANTHROPIC_CUSTOM_HEADERS (one 'Name: value' per line)."""
    headers = {}
    for line in raw.split("\n"):
        name, sep, value = line.partition(":")
        if sep:
            headers[name.strip()] = value.strip()
    return headers


def _error_type(exc: Any) -> str:
    """The provider's structured error type (e.g. "authentication_error"), never its free-text message: error
    bodies can name request headers, which the live-run secret guard rightly refuses to write."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    kind = error.get("type") if isinstance(error, dict) else None
    return kind if isinstance(kind, str) and re.fullmatch(r"[a-z_]{1,64}", kind) else "unknown"


class AnthropicLLMClient:
    """Live client on the official Anthropic SDK. Returns the parsed JSON object; never repairs it.

    No refusal fallback: the configured model either answers or the call fails. The endpoint is pinned to the
    official API and ambient custom headers are ignored; `sdk_client` is the only way to substitute a transport
    or headers, and it is meant for tests."""

    def __init__(self, settings: LLMSettings, *, timeout: float = 300.0, max_retries: int = 2,
                 max_tokens: int = 16000, effort: str = "high", sdk_client: Any = None):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMConfigError("the anthropic SDK is not installed; pip install -e '.[llm]'") from exc
        self._sdk = anthropic
        self._client = sdk_client or _production_sdk_client(anthropic, settings, timeout, max_retries)
        self.model = settings.model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.effort = effort
        self.last_served_model: str | None = None

    def request_params(self, packet: BriefPacket, corrections: Sequence[BriefIssue] = ()) -> dict:
        system, messages = build_messages(packet, corrections)
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": self.effort,
                              "format": {"type": "json_schema", "schema": brief_output_schema()}},
        }

    def generate_structured_brief(self, packet: BriefPacket, *, corrections: Sequence[BriefIssue] = ()) -> dict:
        sdk = self._sdk
        try:
            response = self._client.messages.create(**self.request_params(packet, corrections))
        except sdk.APITimeoutError as exc:
            raise LLMTimeoutError(f"request timed out after {self.timeout}s") from exc
        except sdk.APIConnectionError as exc:
            raise LLMProviderError(f"connection error: {type(exc).__name__}") from exc
        except sdk.APIStatusError as exc:
            raise LLMProviderError(f"HTTP {exc.status_code}: {type(exc).__name__} ({_error_type(exc)})") from exc
        self.last_served_model = getattr(response, "model", None)
        if self.last_served_model != self.model:
            raise LLMProviderError(f"served model {self.last_served_model!r} differs from configured {self.model!r}")
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        if response.stop_reason == "refusal":
            raise LLMRefusalError(f"{self.model} declined the request (stop_reason=refusal); no fallback model is used")
        if response.stop_reason == "max_tokens":
            raise MalformedOutputError(f"output truncated at max_tokens={self.max_tokens}", text)
        return parse_model_output(text)
