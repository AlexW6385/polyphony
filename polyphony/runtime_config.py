"""Model settings saved from the settings page.

One provider, one API key, and two model names: listen (cheap, runs on every new
thought for every voice) and speak (what voices say, cadences, digests). The teacher
and the senses follow speak. Nothing saved means `.env` decides, as before.

The saved values live in one `app_settings` row and are applied onto the shared
`Settings` object: components that read `settings.*_tier` at call time see a change at
once, and voices restart with it because their fingerprint includes `fingerprint()`.
The key is never sent back to the page: it only learns whether one is set and its last
four characters.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from .config import Settings
from .db import AppSetting

ROW = "runtime"
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
           "deepseek": "DEEPSEEK_API_KEY", "gemini": "GEMINI_API_KEY"}
PROVIDERS = tuple(KEY_ENV)
TIERS = ("listen", "speak")  # what the page sets; teacher and sense follow speak
# Suggestions only: any model name the provider accepts can be typed in.
MODEL_SUGGESTIONS = {
    "anthropic": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "openai": [],
    "gemini": [],
}


async def load(sessions: async_sessionmaker) -> dict[str, Any]:
    async with sessions() as s:
        row = await s.get(AppSetting, ROW)
    return dict(row.value) if row and row.value else {}


async def save(sessions: async_sessionmaker, data: dict[str, Any]) -> None:
    async with sessions() as s:
        row = await s.get(AppSetting, ROW)
        if row is None:
            s.add(AppSetting(key=ROW, value=data))
        else:
            row.value = data
        await s.commit()


def env_base(settings: Settings) -> Settings:
    if settings._env_values is None:
        settings._env_values = settings.model_copy()
    return settings._env_values


def apply(settings: Settings, data: dict[str, Any]) -> None:
    """Overlay saved settings on the `.env` values; an empty `data` restores `.env`."""
    base = env_base(settings)
    provider = data.get("provider")
    for tier in ("listen", "speak", "teacher", "sense"):
        fields = ("provider", "model", "base_url", "api_key")
        if not provider:
            for f in fields:
                setattr(settings, f"{tier}_{f}", getattr(base, f"{tier}_{f}"))
        elif tier in TIERS:
            same_vendor = provider == getattr(base, f"{tier}_provider")
            setattr(settings, f"{tier}_provider", provider)
            setattr(settings, f"{tier}_model", data.get(f"{tier}_model") or "")
            # A tier's own .env key / base URL belong to the vendor .env named for it.
            setattr(settings, f"{tier}_base_url",
                    getattr(base, f"{tier}_base_url") if same_vendor else "")
            setattr(settings, f"{tier}_api_key", data.get("api_key")
                    or (getattr(base, f"{tier}_api_key") if same_vendor else ""))
        else:  # teacher, sense: an empty provider means "same as speak"
            for f in fields:
                setattr(settings, f"{tier}_{f}", "")


def fingerprint(settings: Settings) -> str:
    """Changes whenever a voice would talk to a different model or key."""
    parts = [(t.provider, t.model, t.base_url,
              hashlib.sha256((t.api_key or "").encode()).hexdigest()[:12])
             for t in (settings.listen_tier, settings.speak_tier, settings.teacher_tier)]
    return json.dumps(parts)


def view(settings: Settings, data: dict[str, Any]) -> dict[str, Any]:
    """What the settings page shows. Never contains the key."""
    provider = settings.speak_provider
    saved_key = data.get("api_key") if data.get("provider") == provider else ""
    env_key = os.environ.get(KEY_ENV.get(provider, ""), "") or settings.speak_api_key
    key, source = (saved_key, "page") if saved_key else (env_key, "env") if env_key else ("", None)
    return {
        "provider": provider,
        "providers": list(PROVIDERS),
        "api_key": {"set": bool(key), "source": source, "hint": f"…{key[-4:]}" if key else ""},
        "listen_model": settings.listen_model,
        "speak_model": settings.speak_model,
        "listen_provider": settings.listen_provider,  # differs from speak only via .env
        "suggestions": MODEL_SUGGESTIONS,
        "saved": bool(data.get("provider")),
    }


def merge(data: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a page edit: provider, api_key ("" removes it), listen_model, speak_model.

    Switching provider drops a saved key unless a new one comes with it: a key belongs
    to one vendor.
    """
    out = dict(data)
    if "provider" in patch:
        provider = str(patch["provider"] or "")
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        if provider != out.get("provider"):
            out.pop("api_key", None)
        out["provider"] = provider
    if "api_key" in patch:
        key = str(patch["api_key"] or "").strip()
        if key:
            out["api_key"] = key
        else:
            out.pop("api_key", None)
    for tier in TIERS:
        if f"{tier}_model" in patch:
            out[f"{tier}_model"] = str(patch[f"{tier}_model"] or "").strip()
    if not out.get("provider"):
        raise ValueError("choose a provider")
    for tier in TIERS:
        if not out.get(f"{tier}_model"):
            raise ValueError(f"enter a {tier} model")
    return out
