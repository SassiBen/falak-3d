import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .config import Settings
from .generator import Generator, ShapEGenerator
from .jobs import JobQueue, QueueFull
from .presets import load_presets
from .prompting import to_english
from .vram import NoVram, OllamaVram


class JobIn(BaseModel):
    prompt: str = Field(min_length=3, max_length=200)
    preset: str = "ecommerce"


def create_app(settings: Settings | None = None, generator: Generator | None = None,
               translate=None) -> FastAPI:
    settings = settings or Settings.from_env()
    if not settings.api_key:
        raise RuntimeError("FALAK3D_API_KEY manquant : démarrage refusé")
    presets = load_presets(settings.presets_dir)
    generator = generator or ShapEGenerator(settings.data_dir / "shap_e_cache")
    vram = (OllamaVram(settings.ollama_url, settings.ollama_model)
            if settings.ollama_url else NoVram())
    translate = translate or (lambda text: to_english(text, settings.ollama_url,
                                                      settings.ollama_model))
    queue = JobQueue(generator, presets, settings.data_dir, settings.max_queue,
                     settings.retention_days, vram)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        queue.start()
        yield
        await queue.stop()

    app = FastAPI(title="falak-3d", version="0.3.0", lifespan=lifespan)
    app.state.queue = queue

    def authorized(x_api_key: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "clé d'API invalide")

    def view(job_id: str) -> dict:
        job = queue.jobs[job_id]
        out = {"id": job.id, "status": job.status, "prompt": job.prompt, "preset": job.preset,
               "caption": job.caption, "created_at": job.created_at,
               "started_at": job.started_at, "finished_at": job.finished_at}
        if job.status == "queued":
            out["position"] = queue.position(job_id)
        if job.status == "done":
            out["file"] = f"/v1/files/{job.id}.glb"
            out["report"] = job.report
        if job.error:
            out["error"] = job.error
        return out

    @app.get("/health")
    async def health():
        return {"status": "ok" if queue.alive else "worker_down", "waiting": len(queue.waiting)}

    @app.get("/v1/presets")
    async def list_presets():
        return [p.model_dump(include={"name", "title", "description"}) for p in presets.values()]

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(authorized)])
    async def create_job(req: JobIn):
        if req.preset not in presets:
            raise HTTPException(404, f"preset inconnu : {req.preset}")
        caption = presets[req.preset].prompt(translate(req.prompt))
        try:
            job = queue.submit(req.prompt, req.preset, caption)
        except QueueFull as exc:
            raise HTTPException(429, "file pleine, réessayer plus tard") from exc
        return view(job.id)

    @app.get("/v1/jobs", dependencies=[Depends(authorized)])
    async def list_jobs(limit: int = 50):
        recent = sorted(queue.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [view(j.id) for j in recent[:limit]]

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorized)])
    async def get_job(job_id: str):
        if job_id not in queue.jobs:
            raise HTTPException(404, "job inconnu")
        return view(job_id)

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(authorized)])
    async def cancel_job(job_id: str):
        if not queue.cancel(job_id):
            raise HTTPException(409, "seul un job en attente peut être annulé")
        return view(job_id)

    @app.get("/v1/files/{job_id}.glb", dependencies=[Depends(authorized)])
    async def get_file(job_id: str):
        path = queue.glb_path(job_id)
        if not job_id.isalnum() or not path.exists():
            raise HTTPException(404, "fichier introuvable")
        return FileResponse(path, media_type="model/gltf-binary")

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        by_status: dict[str, int] = {}
        for job in queue.jobs.values():
            by_status[job.status] = by_status.get(job.status, 0) + 1
        lines = [f'falak3d_jobs{{status="{s}"}} {n}' for s, n in sorted(by_status.items())]
        lines += [f"falak3d_waiting {len(queue.waiting)}",
                  f"falak3d_worker_alive {int(queue.alive)}",
                  f"falak3d_worker_errors_total {queue.worker_errors}"]
        return "\n".join(lines) + "\n"

    return app
