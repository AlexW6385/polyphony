"""The order of record: PostgreSQL decides; Redis is only a cache."""

import asyncio
import json

from polyphony.score import K_RECENT


async def test_concurrent_appends_get_distinct_increasing_seq(score):
    results = await asyncio.gather(*[
        score.append(f"v{i % 5}", "voice", f"thought {i}") for i in range(50)
    ])
    seqs = [t["seq"] for t in results]
    assert len(set(seqs)) == 50
    listed = await score.thoughts()
    assert [t["seq"] for t in listed] == sorted(seqs)
    recent = await score.recent()
    assert [t["seq"] for t in recent] == sorted(seqs)[-score.window:]


async def test_recent_sorts_out_of_order_pushes(score):
    a = await score.append("a", "voice", "first")
    b = await score.append("b", "voice", "second")
    # Simulate two voices whose Redis pushes landed in the wrong order.
    await score.redis.delete(score.redis_key(K_RECENT))
    await score.redis.zadd(score.redis_key(K_RECENT), {json.dumps(b): b["seq"]})
    await score.redis.zadd(score.redis_key(K_RECENT), {json.dumps(a): a["seq"]})
    assert [t["seq"] for t in await score.recent()] == [a["seq"], b["seq"]]


async def test_window_keeps_newest_seq_regardless_of_push_order(score):
    score.window = 3
    ts = [await score.append("a", "voice", str(i)) for i in range(5)]
    await score.redis.delete(score.redis_key(K_RECENT))
    for t in [ts[4], ts[3], ts[2], ts[1], ts[0]]:  # worst case: arrive newest-first
        await score.redis.zadd(score.redis_key(K_RECENT), {json.dumps(t): t["seq"]})
        await score.redis.zremrangebyrank(score.redis_key(K_RECENT), 0, -score.window - 1)
    assert [t["content"] for t in await score.recent()] == ["2", "3", "4"]


async def test_cold_window_refills_from_postgres(score):
    for i in range(5):
        await score.append("a", "voice", str(i))
    await score.redis.delete(score.redis_key(K_RECENT))
    recent = await score.recent()
    assert [t["content"] for t in recent] == ["0", "1", "2", "3", "4"]
    assert await score.redis.zcard(score.redis_key(K_RECENT)) == 5


async def test_subscribe_sees_appends(score):
    ready = asyncio.Event()
    got = []

    async def watch():
        async for t in score.subscribe(ready=ready):
            got.append(t)
            if len(got) == 2:
                return

    task = asyncio.create_task(watch())
    await ready.wait()
    await score.append("conscious", "conscious", "one")
    await score.append("rest", "voice", "two")
    await asyncio.wait_for(task, 5)
    assert [t["content"] for t in got] == ["one", "two"]


async def test_strays_are_older_only(score):
    for i in range(10):
        await score.append("a", "voice", str(i))
    strays = await score.strays(before_seq=6, n=3)
    assert len(strays) == 3 and all(t["seq"] < 6 for t in strays)


async def test_listens_and_forming(score):
    await score.record_listen("rest", 1, {"speak": False, "mood": "tired", "reason": "loop",
                                           "pause_seconds": 0})
    await score.set_forming("make", True)
    assert (await score.listens_latest())["rest"]["mood"] == "tired"
    assert await score.forming() == ["make"]
    await score.set_forming("make", False)
    assert await score.forming() == []
