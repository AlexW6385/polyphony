"""The order of record: PostgreSQL decides; Redis is only a cache."""

import asyncio
import json

from polyphony.stream import K_RECENT


async def test_concurrent_appends_get_distinct_increasing_seq(stream):
    results = await asyncio.gather(*[
        stream.append(f"v{i % 5}", "voice", f"thought {i}") for i in range(50)
    ])
    seqs = [t["seq"] for t in results]
    assert len(set(seqs)) == 50
    listed = await stream.thoughts()
    assert [t["seq"] for t in listed] == sorted(seqs)
    recent = await stream.recent()
    assert [t["seq"] for t in recent] == sorted(seqs)[-stream.window:]


async def test_recent_sorts_out_of_order_pushes(stream):
    a = await stream.append("a", "voice", "first")
    b = await stream.append("b", "voice", "second")
    # Simulate two voices whose Redis pushes landed in the wrong order.
    await stream.redis.delete(K_RECENT)
    await stream.redis.zadd(K_RECENT, {json.dumps(b): b["seq"]})
    await stream.redis.zadd(K_RECENT, {json.dumps(a): a["seq"]})
    assert [t["seq"] for t in await stream.recent()] == [a["seq"], b["seq"]]


async def test_window_keeps_newest_seq_regardless_of_push_order(stream):
    stream.window = 3
    ts = [await stream.append("a", "voice", str(i)) for i in range(5)]
    await stream.redis.delete(K_RECENT)
    for t in [ts[4], ts[3], ts[2], ts[1], ts[0]]:  # worst case: arrive newest-first
        await stream.redis.zadd(K_RECENT, {json.dumps(t): t["seq"]})
        await stream.redis.zremrangebyrank(K_RECENT, 0, -stream.window - 1)
    assert [t["content"] for t in await stream.recent()] == ["2", "3", "4"]


async def test_cold_window_refills_from_postgres(stream):
    for i in range(5):
        await stream.append("a", "voice", str(i))
    await stream.redis.delete(K_RECENT)
    recent = await stream.recent()
    assert [t["content"] for t in recent] == ["0", "1", "2", "3", "4"]
    assert await stream.redis.zcard(K_RECENT) == 5


async def test_subscribe_sees_appends(stream):
    ready = asyncio.Event()
    got = []

    async def watch():
        async for t in stream.subscribe(ready=ready):
            got.append(t)
            if len(got) == 2:
                return

    task = asyncio.create_task(watch())
    await ready.wait()
    await stream.append("conscious", "conscious", "one")
    await stream.append("rest", "voice", "two")
    await asyncio.wait_for(task, 5)
    assert [t["content"] for t in got] == ["one", "two"]


async def test_strays_are_older_only(stream):
    for i in range(10):
        await stream.append("a", "voice", str(i))
    strays = await stream.strays(before_seq=6, n=3)
    assert len(strays) == 3 and all(t["seq"] < 6 for t in strays)


async def test_listens_and_forming(stream):
    await stream.record_listen("rest", 1, {"speak": False, "mood": "tired", "reason": "loop",
                                           "pause_seconds": 0})
    await stream.set_forming("make", True)
    assert (await stream.listens_latest())["rest"]["mood"] == "tired"
    assert await stream.forming() == ["make"]
    await stream.set_forming("make", False)
    assert await stream.forming() == []
