"""One sense run as a LangGraph workflow.

    collect -> triage -> [search] -> read -> summarize -> emit

- collect:   new entries from the sense's feeds (our feeds MCP server); seen ones dropped
- triage marks the entries it passes over as seen, so each entry is judged once
- triage:    the model picks entries and optional search queries, as JSON (this is how it
             chooses tools: a schema, not native tool calling, so every provider works)
- search:    only when queries were requested and a search server is configured
- read:      full text of picked articles via the official Fetch MCP server; feed summary
             as fallback
- summarize: 0..N items, grounded only in what was read
- emit:      v0 dedup (URL and title fingerprints, claim before append), one `world`
             thought per new item

Fetched text is untrusted: it is delimited and marked as data in every prompt, and can
never change which tools run beyond the triage schema.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from . import llm, prompts
from .config import Settings
from .frame import Frame

log = logging.getLogger(__name__)

MAX_ENTRIES_FOR_TRIAGE = 40
MAX_ARTICLE_CHARS = 12000


class MCPCaller(Protocol):
    def has(self, server: str) -> bool: ...
    async def call(self, server: str, tool: str, args: dict[str, Any]) -> str: ...


class SenseState(TypedDict, total=False):
    entries: list[dict[str, Any]]
    pick: list[int]
    queries: list[str]
    search_results: list[str]
    articles: list[dict[str, Any]]
    items: list[dict[str, Any]]
    appended: list[dict[str, Any]]


def _escape(text: str) -> str:
    return text.replace("</article>", "</ article>").replace("</entry>", "</ entry>")


def build(*, sense, settings: Settings, frame: Frame, mcp: MCPCaller,
          search: tuple[str, str] | None):
    """`sense` is a sense.Sense (for dedup and append); `search` is (tool, query arg)."""
    spec = sense.spec
    max_items = settings.sense_max_items

    async def collect(state: SenseState) -> SenseState:
        entries: list[dict[str, Any]] = []
        for url in spec.feeds:
            try:
                raw = await mcp.call("feeds", "fetch_feed", {"url": url, "limit": 30})
                data = json.loads(raw)
            except Exception as e:
                log.warning("sense %s: feed %s failed: %s", spec.name, url, e)
                continue
            for e in data.get("entries", []):
                item = {"title": e.get("title"), "url": e.get("link"), "summary": e.get("summary"),
                        "published": e.get("published"), "feed": data.get("feed") or url}
                if not await sense.seen(item):
                    entries.append(item)
        entries = entries[:MAX_ENTRIES_FOR_TRIAGE]
        for i, e in enumerate(entries):
            e["id"] = i
        return {"entries": entries}

    async def triage(state: SenseState) -> SenseState:
        entries = state.get("entries") or []
        if not entries:
            return {"pick": [], "queries": []}
        listing = "\n".join(
            f'<entry id="{e["id"]}">\n{_escape(e["title"] or "")}\n{e["feed"]} · '
            f'{e["published"] or ""}\n{_escape((e["summary"] or "")[:400])}\n</entry>'
            for e in entries)
        reply = await llm.complete_json(
            settings.sense_tier,
            prompts.triage_system(frame, spec.instruction, bool(search and spec.search), max_items),
            f"Today is {prompts.today()}.\n\n{listing}",
            schema=prompts.TRIAGE_SCHEMA, temperature=0.2, tier_name="sense", purpose="triage",
        )
        ids = {e["id"] for e in entries}
        pick = [i for i in reply.get("pick") or [] if isinstance(i, int) and i in ids][:max_items]
        queries = [q for q in reply.get("queries") or [] if isinstance(q, str) and q.strip()][:2]
        if not (search and spec.search):
            queries = []
        # Entries the model passed over count as seen, so they are not triaged again.
        for e in entries:
            if e["id"] not in pick:
                await sense.mark_seen(e)
        return {"pick": pick, "queries": queries}

    def after_triage(state: SenseState) -> str:
        if not state.get("pick"):
            return END
        return "search" if state.get("queries") else "read"

    async def do_search(state: SenseState) -> SenseState:
        tool, arg = search
        results = []
        for q in state.get("queries") or []:
            try:
                results.append(await mcp.call("search", tool, {arg: q}))
            except Exception as e:
                log.warning("sense %s: search %r failed: %s", spec.name, q, e)
        return {"search_results": results}

    async def read(state: SenseState) -> SenseState:
        by_id = {e["id"]: e for e in state.get("entries") or []}
        articles = []
        for i in (state.get("pick") or [])[: settings.sense_read_max_articles]:
            e = by_id[i]
            body = e.get("summary") or ""
            if e.get("url") and mcp.has("fetch"):
                try:
                    body = await mcp.call("fetch", "fetch",
                                          {"url": e["url"], "max_length": MAX_ARTICLE_CHARS})
                except Exception as ex:
                    log.info("sense %s: fetch %s failed, using feed summary: %s",
                             spec.name, e["url"], ex)
            articles.append({**e, "body": body[:MAX_ARTICLE_CHARS]})
        return {"articles": articles}

    async def summarize(state: SenseState) -> SenseState:
        articles = state.get("articles") or []
        if not articles:
            return {"items": []}
        blocks = "\n\n".join(
            f'<article url="{a.get("url") or ""}">\n{_escape(a.get("title") or "")}\n\n'
            f'{_escape(a["body"])}\n</article>' for a in articles)
        if state.get("search_results"):
            blocks += "\n\n" + "\n\n".join(f"<article url=\"search\">\n{_escape(r[:4000])}\n"
                                           f"</article>" for r in state["search_results"])
        reply = await llm.complete_json(
            settings.sense_tier, prompts.summarize_system(frame, spec.instruction, max_items),
            f"Today is {prompts.today()}.\n\n{blocks}", temperature=0.3, max_tokens=1500,
            tier_name="sense", purpose="summarize",
        )
        items = reply.get("items") or []
        return {"items": [i for i in items if isinstance(i, dict)][:max_items]}

    async def emit(state: SenseState) -> SenseState:
        appended = []
        for item in state.get("items") or []:
            t = await sense.emit(item)
            if t:
                appended.append(t)
        return {"appended": appended}

    g = StateGraph(SenseState)
    g.add_node("collect", collect)
    g.add_node("triage", triage)
    g.add_node("search", do_search)
    g.add_node("read", read)
    g.add_node("summarize", summarize)
    g.add_node("emit", emit)
    g.add_edge(START, "collect")
    g.add_edge("collect", "triage")
    g.add_conditional_edges("triage", after_triage, ["search", "read", END])
    g.add_edge("search", "read")
    g.add_edge("read", "summarize")
    g.add_edge("summarize", "emit")
    g.add_edge("emit", END)
    return g.compile()
