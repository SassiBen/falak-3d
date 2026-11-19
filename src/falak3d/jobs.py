"""File de jobs à un seul worker.

Le GPU ne traite qu'une génération à la fois. Les jobs attendent dans une file
asyncio, leur état est recopié sur disque (écriture atomique) pour survivre à un
redémarrage, et les fichiers sont purgés au-delà de la durée de rétention.
"""

import asyncio
import contextlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .generator import Generator
from .postprocess import process
from .presets import Preset
from .vram import NoVram


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    prompt: str
    preset: str
    caption: str
    status: str = "queued"  # queued | running | done | error | cancelled
    created_at: str = field(default_factory=now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    report: dict | None = None


class QueueFull(Exception):
    pass


class JobQueue:
    def __init__(self, generator: Generator, presets: dict[str, Preset], data_dir: Path,
                 max_queue: int = 10, retention_days: int = 7, vram=None):
        self.generator = generator
        self.presets = presets
        self.models_dir = data_dir / "models"
        self.jobs_dir = data_dir / "jobs"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.max_queue = max_queue
        self.retention_days = retention_days
        self.vram = vram or NoVram()
        self.jobs: dict[str, Job] = {}
        self.waiting: list[str] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.worker_errors = 0
        self._restore()

    # -- cycle de vie -----------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    # -- API ---------------------------------------------------------------

    def submit(self, prompt: str, preset: str, caption: str) -> Job:
        if len(self.waiting) >= self.max_queue:
            raise QueueFull
        job = Job(id=uuid.uuid4().hex[:12], prompt=prompt, preset=preset, caption=caption)
        self.jobs[job.id] = job
        self.waiting.append(job.id)
        self._save(job)
        self._queue.put_nowait(job.id)
        return job

    def position(self, job_id: str) -> int | None:
        return self.waiting.index(job_id) + 1 if job_id in self.waiting else None

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status != "queued":
            return False
        job.status, job.finished_at = "cancelled", now()
        self.waiting.remove(job_id)
        self._save(job)
        return True

    def glb_path(self, job_id: str) -> Path:
        return self.models_dir / f"{job_id}.glb"

    async def drain(self) -> None:
        await self._queue.join()

    # -- worker -----------------------------------------------------------

    async def _run(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                job = self.jobs.get(job_id)
                if job is None or job.status == "cancelled":
                    continue
                self.waiting.remove(job_id)
                job.status, job.started_at = "running", now()
                self._save(job)
                await asyncio.to_thread(self.vram.release)
                try:
                    job.report = await asyncio.to_thread(self._generate, job)
                    job.status = "done"
                except Exception as exc:  # noqa: BLE001 - un job raté ne doit pas tuer la file
                    job.status, job.error = "error", str(exc)[:300]
                job.finished_at = now()
                self._save(job)
                if self._queue.empty():
                    await asyncio.to_thread(self.vram.warm)
                    self.purge()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                # Typiquement un disque plein pendant _save. Le worker doit
                # survivre : sinon /health reste vert pendant que plus rien
                # ne sort de la file.
                self.worker_errors += 1
            finally:
                self._queue.task_done()

    def _generate(self, job: Job) -> dict:
        preset = self.presets[job.preset]
        mesh = self.generator.generate(job.caption, preset)
        mesh, report = process(mesh, preset.target_faces, preset.size_m)
        mesh.export(self.glb_path(job.id), file_type="glb")
        return asdict(report)

    # -- persistance ------------------------------------------------------

    def _save(self, job: Job) -> None:
        target = self.jobs_dir / f"{job.id}.json"
        tmp = target.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(asdict(job), f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)

    def _restore(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = Job(**json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, TypeError):
                continue
            # Un job interrompu par l'arrêt du service ne reprendra pas tout seul.
            if job.status in ("queued", "running"):
                job.status, job.error = "error", "interrompu par un redémarrage"
            self.jobs[job.id] = job

    def purge(self) -> int:
        if self.retention_days <= 0:
            return 0
        cutoff = time.time() - self.retention_days * 86400
        removed = 0
        for path in [*self.models_dir.glob("*.glb"), *self.jobs_dir.glob("*.json")]:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                self.jobs.pop(path.stem, None)
                removed += 1
        return removed
