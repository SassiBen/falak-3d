import asyncio
import json

import pytest

from falak3d.generator import FakeGenerator
from falak3d.jobs import JobQueue, QueueFull
from falak3d.presets import load_presets


def make_queue(settings, **kw):
    presets = load_presets(settings.presets_dir)
    return JobQueue(kw.pop("generator", FakeGenerator()), presets, settings.data_dir, **kw)


def test_jobs_run_in_order_and_produce_glb(settings):
    async def scenario():
        q = make_queue(settings)
        q.start()
        a = q.submit("mug", "ecommerce", "a mug")
        b = q.submit("box", "packaging", "a box packaging")
        await q.drain()
        await q.stop()
        return q, a, b

    q, a, b = asyncio.run(scenario())
    assert q.jobs[a.id].status == q.jobs[b.id].status == "done"
    assert q.glb_path(a.id).stat().st_size > 1000
    assert q.jobs[b.id].report["size_m"][0] == pytest.approx(0.25, abs=1e-3)


def test_failed_job_does_not_stop_the_worker(settings):
    async def scenario():
        q = make_queue(settings, generator=FakeGenerator(fail_on="cursed"))
        q.start()
        bad = q.submit("x", "ecommerce", "a cursed box")
        good = q.submit("y", "ecommerce", "a box")
        await q.drain()
        alive = q.alive
        await q.stop()
        return q, bad, good, alive

    q, bad, good, alive = asyncio.run(scenario())
    assert q.jobs[bad.id].status == "error"
    assert q.jobs[good.id].status == "done"
    assert alive


def test_cancel_and_positions(settings):
    async def scenario():
        q = make_queue(settings)  # worker non démarré : tout reste en attente
        jobs = [q.submit(f"p{i}", "jeu", f"low poly box {i}") for i in range(3)]
        assert [q.position(j.id) for j in jobs] == [1, 2, 3]
        assert q.cancel(jobs[1].id)
        assert q.position(jobs[2].id) == 2
        assert not q.cancel(jobs[1].id)
        return q

    asyncio.run(scenario())


def test_queue_limit(settings):
    async def scenario():
        q = make_queue(settings, max_queue=2)
        q.submit("a", "jeu", "a")
        q.submit("b", "jeu", "b")
        with pytest.raises(QueueFull):
            q.submit("c", "jeu", "c")

    asyncio.run(scenario())


def test_interrupted_jobs_are_marked_after_restart(settings):
    async def first_run():
        q = make_queue(settings)
        return q.submit("mug", "ecommerce", "a mug").id

    job_id = asyncio.run(first_run())
    restarted = make_queue(settings)
    assert restarted.jobs[job_id].status == "error"
    saved = json.loads((settings.data_dir / "jobs" / f"{job_id}.json").read_text())
    assert saved["prompt"] == "mug"
