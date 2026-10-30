"""Provider-agnostic LLM layer.

One interface, `Provider.complete()`, hides the vendor. Tiers (listen, speak,
sense, teacher) each pick a provider/model/base URL/key independently, and a
voice may override its listen/speak tier in its front matter. Embeddings go
through `Provider.embed()` where the vendor has one.

Every call is traced: `set_tracer()` installs a callback that receives one
`Trace` per call (main.py writes them to `llm_calls`).

Structured output: `complete_json(schema=...)` passes the JSON schema to
vendors that enforce it (OpenAI, Gemini, Ollama `format`); the reply is
always run through `extract_json` as well, so a vendor without enforcement
still works.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import TierConfig

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


@dataclass
class Completion:
    """What a provider returns when it knows more than the text."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    server_ms: int | None = None  # time the server spent on the request (Ollama total_duration)


@dataclass
class Trace:
    tier: str
    provider: str
    model: str
    purpose: str
    voice: str | None
    latency_ms: int
    queue_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    ok: bool = True
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        json: bool = False,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> str | Completion: ...


# --------------------------------------------------------------------------- OpenAI-compatible


class OpenAIProvider:
    """OpenAI, and anything that speaks its API (DeepSeek, vLLM, ...)."""

    def __init__(self, base_url: str | None, api_key: str | None, *, official: bool = True):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.official = official

    async def complete(self, *, model, system, user, json=False, schema=None, temperature=0.8,
                       max_tokens=1024) -> Completion:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if self.official:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        if schema and self.official:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": _strict(schema), "strict": True},
            }
        elif json or schema:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # Some reasoning models reject a custom temperature; drop it once.
            if "temperature" in str(e) and getattr(e, "status_code", None) == 400:
                kwargs.pop("temperature")
                resp = await self.client.chat.completions.create(**kwargs)
            else:
                raise
        u = getattr(resp, "usage", None)
        cached = None
        if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
            cached = getattr(u.prompt_tokens_details, "cached_tokens", None)
        return Completion(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(u, "prompt_tokens", None),
            output_tokens=getattr(u, "completion_tokens", None),
            cached_input_tokens=cached,
        )

    async def embed(self, *, model: str, texts: list[str]) -> list[list[float]]:
        resp = await self.client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI strict mode wants additionalProperties: false on every object."""
    s = dict(schema)
    if s.get("type") == "object":
        s.setdefault("additionalProperties", False)
        s["properties"] = {k: _strict(v) for k, v in s.get("properties", {}).items()}
    if "items" in s and isinstance(s["items"], dict):
        s["items"] = _strict(s["items"])
    return s


# --------------------------------------------------------------------------- Ollama (native API)


class OllamaProvider:
    """Ollama's native API: structured `format`, `keep_alive`, and server-side timings.

    The native API reports `total_duration` (time the server spent on the
    request); client latency minus that is time spent queued, which is what
    `OLLAMA_NUM_PARALLEL` controls.
    """

    def __init__(self, base_url: str | None):
        import httpx

        base = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        self.base = base.rstrip("/").removesuffix("/v1")
        self.client = httpx.AsyncClient(base_url=self.base, timeout=300)
        self.keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        try:
            r = await self.client.post(path, json=body)
        except httpx.ConnectError as e:
            raise LLMError(
                f"Ollama is not reachable at {self.base} — start Ollama or set LISTEN_BASE_URL / "
                f"EMBED_BASE_URL / OLLAMA_BASE_URL (on macOS in Docker: "
                f"http://host.docker.internal:11434). Polyphony never falls back to a paid tier."
            ) from e
        if r.status_code >= 400:
            raise LLMError(f"Ollama {path} {r.status_code}: {r.text[:300]}")
        return r.json()

    async def complete(self, *, model, system, user, json=False, schema=None, temperature=0.8,
                       max_tokens=1024) -> Completion:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if schema:
            body["format"] = schema
        elif json:
            body["format"] = "json"
        d = await self._post("/api/chat", body)
        return Completion(
            text=(d.get("message") or {}).get("content", ""),
            input_tokens=d.get("prompt_eval_count"),
            output_tokens=d.get("eval_count"),
            server_ms=int(d["total_duration"] / 1e6) if d.get("total_duration") else None,
        )

    async def embed(self, *, model: str, texts: list[str]) -> list[list[float]]:
        d = await self._post("/api/embed", {"model": model, "input": texts,
                                            "keep_alive": self.keep_alive})
        return d["embeddings"]


# --------------------------------------------------------------------------- Anthropic


class AnthropicProvider:
    def __init__(self, base_url: str | None, api_key: str | None):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(base_url=base_url, api_key=api_key)

    async def complete(self, *, model, system, user, json=False, schema=None, temperature=0.8,
                       max_tokens=1024) -> Completion:
        # The system prompt is frame + engine (+ voice): mark it cacheable so the shared
        # prefix is billed once. Temperature is not sent; default sampling is fine.
        resp = await self.client.messages.create(
            model=model,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        u = resp.usage
        return Completion(
            text="".join(b.text for b in resp.content if getattr(b, "type", None) == "text"),
            input_tokens=(u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
            + (getattr(u, "cache_creation_input_tokens", 0) or 0),
            output_tokens=u.output_tokens,
            cached_input_tokens=getattr(u, "cache_read_input_tokens", None),
        )


# --------------------------------------------------------------------------- Gemini


class GeminiProvider:
    def __init__(self, base_url: str | None, api_key: str | None):
        from google import genai

        http_options = {"base_url": base_url} if base_url else None
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            http_options=http_options,
        )

    async def complete(self, *, model, system, user, json=False, schema=None, temperature=0.8,
                       max_tokens=1024) -> Completion:
        from google.genai import types

        cfg: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json or schema:
            cfg["response_mime_type"] = "application/json"
        if schema:
            cfg["response_json_schema"] = schema
        resp = await self.client.aio.models.generate_content(
            model=model, contents=user, config=types.GenerateContentConfig(**cfg)
        )
        u = getattr(resp, "usage_metadata", None)
        return Completion(
            text=resp.text or "",
            input_tokens=getattr(u, "prompt_token_count", None),
            output_tokens=getattr(u, "candidates_token_count", None),
            cached_input_tokens=getattr(u, "cached_content_token_count", None),
        )

    async def embed(self, *, model: str, texts: list[str]) -> list[list[float]]:
        resp = await self.client.aio.models.embed_content(model=model, contents=texts)
        return [e.values for e in resp.embeddings]


# --------------------------------------------------------------------------- registry

ProviderFactory = Callable[[str | None, str | None], Provider]


def _openai(base_url, api_key):
    return OpenAIProvider(base_url, api_key or os.environ.get("OPENAI_API_KEY"), official=not base_url)


def _deepseek(base_url, api_key):
    return OpenAIProvider(
        base_url or "https://api.deepseek.com",
        api_key or os.environ.get("DEEPSEEK_API_KEY"),
        official=False,
    )


def _ollama(base_url, api_key):
    return OllamaProvider(base_url)


def _anthropic(base_url, api_key):
    return AnthropicProvider(base_url, api_key or os.environ.get("ANTHROPIC_API_KEY"))


def _gemini(base_url, api_key):
    return GeminiProvider(base_url, api_key)


PROVIDERS: dict[str, ProviderFactory] = {
    "openai": _openai,
    "deepseek": _deepseek,
    "ollama": _ollama,
    "anthropic": _anthropic,
    "gemini": _gemini,
}

_cache: dict[tuple, Provider] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Add a provider (used by tests, or to plug in a vendor without editing this file)."""
    PROVIDERS[name] = factory
    for key in [k for k in _cache if k[0] == name]:
        del _cache[key]


def get_provider(tier: TierConfig) -> Provider:
    key = (tier.provider, tier.base_url, tier.api_key)
    if key not in _cache:
        try:
            factory = PROVIDERS[tier.provider]
        except KeyError:
            raise LLMError(f"unknown provider {tier.provider!r}; known: {sorted(PROVIDERS)}")
        _cache[key] = factory(tier.base_url, tier.api_key)
    return _cache[key]


def resolve_tier(base: TierConfig, override: dict[str, Any] | None) -> TierConfig:
    """Apply a voice's per-tier override; missing keys fall back to the global tier.

    If the override switches provider, the global key/base URL are not carried
    over (they belong to another vendor); the provider's own env defaults apply.
    """
    if not override:
        return base
    provider = override.get("provider", base.provider)
    same = provider == base.provider
    return TierConfig(
        provider=provider,
        model=override.get("model", base.model),
        base_url=override.get("base_url", base.base_url if same else None),
        api_key=override.get("api_key", base.api_key if same else None),
    )


# --------------------------------------------------------------------------- tracing

Tracer = Callable[[Trace], Awaitable[None]]
_tracer: Tracer | None = None
_trace_tasks: set[asyncio.Task] = set()


def set_tracer(fn: Tracer | None) -> None:
    global _tracer
    _tracer = fn


def _emit(trace: Trace) -> None:
    if _tracer is None:
        return
    try:
        t = asyncio.get_running_loop().create_task(_tracer(trace))
    except RuntimeError:
        return
    _trace_tasks.add(t)
    t.add_done_callback(_trace_tasks.discard)


# --------------------------------------------------------------------------- helpers


def _transient(e: Exception) -> bool:
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(status, int) and (status in (408, 409, 429) or status >= 500):
        return True
    name = type(e).__name__
    return any(s in name for s in ("Connection", "Timeout", "ServerError", "Overloaded"))


async def _once(tier: TierConfig, tier_name: str, purpose: str, voice: str | None,
                **kw) -> str:
    provider = get_provider(tier)
    start = time.perf_counter()
    try:
        out = await provider.complete(model=tier.model, **kw)
    except Exception as e:
        _emit(Trace(tier_name, tier.provider, tier.model, purpose, voice,
                    int((time.perf_counter() - start) * 1000), ok=False, error=str(e)[:500]))
        raise
    latency = int((time.perf_counter() - start) * 1000)
    c = out if isinstance(out, Completion) else Completion(text=out)
    queue = max(0, latency - c.server_ms) if c.server_ms is not None else None
    _emit(Trace(tier_name, tier.provider, tier.model, purpose, voice, latency, queue,
                c.input_tokens, c.output_tokens, c.cached_input_tokens))
    return c.text


async def _call(tier: TierConfig, *, tier_name: str = "?", purpose: str = "?",
                voice: str | None = None, **kw) -> str:
    try:
        return await _once(tier, tier_name, purpose, voice, **kw)
    except Exception as e:
        if not _transient(e):
            raise
        log.warning("transient LLM error (%s), retrying once: %s", tier.provider, e)
        await asyncio.sleep(1.5)
        return await _once(tier, tier_name, purpose, voice, **kw)


def extract_json(text: str) -> dict[str, Any]:
    """Return the first balanced `{...}` object in `text`, parsed. Raises on failure."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    raise LLMError(f"no JSON object in model reply: {text[:200]!r}")


async def complete_json(
    tier: TierConfig, system: str, user: str, *, temperature: float = 0.7, max_tokens: int = 600,
    schema: dict[str, Any] | None = None, tier_name: str = "?", purpose: str = "?",
    voice: str | None = None,
) -> dict[str, Any]:
    text = await _call(tier, tier_name=tier_name, purpose=purpose, voice=voice, system=system,
                       user=user, json=True, schema=schema, temperature=temperature,
                       max_tokens=max_tokens)
    return extract_json(text)


async def complete_text(
    tier: TierConfig, system: str, user: str, *, temperature: float = 0.9, max_tokens: int = 1024,
    tier_name: str = "?", purpose: str = "?", voice: str | None = None,
) -> str:
    text = await _call(tier, tier_name=tier_name, purpose=purpose, voice=voice, system=system,
                       user=user, temperature=temperature, max_tokens=max_tokens)
    return text.strip()


async def embed(tier: TierConfig, texts: list[str]) -> list[list[float]]:
    provider = get_provider(tier)
    fn = getattr(provider, "embed", None)
    if fn is None:
        raise LLMError(f"provider {tier.provider!r} has no embeddings")
    start = time.perf_counter()
    try:
        out = await fn(model=tier.model, texts=texts)
    except Exception as e:
        _emit(Trace("embed", tier.provider, tier.model, "embed", None,
                    int((time.perf_counter() - start) * 1000), ok=False, error=str(e)[:500]))
        raise
    _emit(Trace("embed", tier.provider, tier.model, "embed", None,
                int((time.perf_counter() - start) * 1000)))
    return out
