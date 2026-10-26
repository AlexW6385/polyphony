"""Provider-agnostic LLM layer.

One interface, `Provider.complete()`, hides the vendor. Tiers (listen, speak,
sense) each pick a provider/model/base URL/key independently, and a voice may
override either of its tiers in its front matter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from typing import Any, Protocol

from .config import TierConfig

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class Provider(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        json: bool = False,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        web_search: bool = False,
    ) -> str: ...


# --------------------------------------------------------------------------- OpenAI-compatible


class OpenAIProvider:
    """OpenAI, and anything that speaks its API (DeepSeek, Ollama, vLLM, ...)."""

    def __init__(self, base_url: str | None, api_key: str | None, *, official: bool = True):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.official = official  # only api.openai.com has the Responses web_search tool

    async def complete(
        self,
        *,
        model,
        system,
        user,
        json=False,
        temperature=0.8,
        max_tokens=1024,
        web_search=False,
    ) -> str:
        if web_search:
            if not self.official:
                raise LLMError("this OpenAI-compatible endpoint has no web search")
            resp = await self.client.responses.create(
                model=model,
                instructions=system,
                input=user,
                tools=[{"type": "web_search"}],
                max_output_tokens=max_tokens,
            )
            return resp.output_text or ""

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
        if json:
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
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- Anthropic


class AnthropicProvider:
    def __init__(self, base_url: str | None, api_key: str | None):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(base_url=base_url, api_key=api_key)

    async def complete(
        self,
        *,
        model,
        system,
        user,
        json=False,
        temperature=0.8,
        max_tokens=1024,
        web_search=False,
    ) -> str:
        # `temperature` is deliberately not sent: current Claude models and SDKs
        # no longer take it, and the default sampling is fine for voices.
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,  # system prompt goes here, not in messages
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
        }
        if web_search:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
        resp = await self.client.messages.create(**kwargs)
        texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        # With tools, the model's final answer is the last text block.
        return texts[-1] if web_search and texts else "".join(texts)


# --------------------------------------------------------------------------- Gemini


class GeminiProvider:
    def __init__(self, base_url: str | None, api_key: str | None):
        from google import genai

        http_options = {"base_url": base_url} if base_url else None
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            http_options=http_options,
        )

    async def complete(
        self,
        *,
        model,
        system,
        user,
        json=False,
        temperature=0.8,
        max_tokens=1024,
        web_search=False,
    ) -> str:
        from google.genai import types

        cfg: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if web_search:
            cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        elif json:
            cfg["response_mime_type"] = "application/json"
        resp = await self.client.aio.models.generate_content(
            model=model, contents=user, config=types.GenerateContentConfig(**cfg)
        )
        return resp.text or ""


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
    base = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    return OpenAIProvider(base, api_key or "ollama", official=False)


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


# --------------------------------------------------------------------------- helpers


def _transient(e: Exception) -> bool:
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(status, int) and (status in (408, 409, 429) or status >= 500):
        return True
    name = type(e).__name__
    return any(s in name for s in ("Connection", "Timeout", "ServerError", "Overloaded"))


async def _call(tier: TierConfig, **kw) -> str:
    provider = get_provider(tier)
    try:
        return await provider.complete(model=tier.model, **kw)
    except Exception as e:
        if not _transient(e):
            raise
        log.warning("transient LLM error (%s), retrying once: %s", tier.provider, e)
        await asyncio.sleep(1.5)
        return await provider.complete(model=tier.model, **kw)


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
    tier: TierConfig, system: str, user: str, *, temperature: float = 0.7, max_tokens: int = 600
) -> dict[str, Any]:
    text = await _call(
        tier, system=system, user=user, json=True, temperature=temperature, max_tokens=max_tokens
    )
    return extract_json(text)


async def complete_text(
    tier: TierConfig, system: str, user: str, *, temperature: float = 0.9, max_tokens: int = 1024
) -> str:
    text = await _call(tier, system=system, user=user, temperature=temperature, max_tokens=max_tokens)
    return text.strip()


async def search_json(
    tier: TierConfig, system: str, user: str, *, max_tokens: int = 2000
) -> dict[str, Any]:
    """Ask a model with web search for a JSON object (used by senses)."""
    text = await _call(
        tier, system=system, user=user, web_search=True, temperature=0.3, max_tokens=max_tokens
    )
    return extract_json(text)
