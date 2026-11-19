import time

from fastapi.testclient import TestClient

from falak3d.api import create_app
from falak3d.generator import FakeGenerator

H = {"X-API-Key": "test-key"}


def wait_done(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}", headers=H).json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job toujours en cours")


def test_full_cycle(settings):
    app = create_app(settings, FakeGenerator(), translate=lambda t: "mug")
    with TestClient(app) as c:
        assert c.post("/v1/jobs", json={"prompt": "une tasse"}).status_code == 401
        r = c.post("/v1/jobs", headers=H, json={"prompt": "une tasse", "preset": "ecommerce"})
        assert r.status_code == 202
        job = wait_done(c, r.json()["id"])
        assert job["status"] == "done"
        assert job["caption"].startswith("a mug")
        glb = c.get(job["file"], headers=H)
        assert glb.status_code == 200
        assert glb.content[:4] == b"glTF"
        assert "falak3d_worker_alive 1" in c.get("/metrics").text


def test_unknown_preset(settings):
    app = create_app(settings, FakeGenerator(), translate=lambda t: t)
    with TestClient(app) as c:
        r = c.post("/v1/jobs", headers=H, json={"prompt": "chair", "preset": "inconnu"})
        assert r.status_code == 404


def test_presets_are_public(settings):
    app = create_app(settings, FakeGenerator(), translate=lambda t: t)
    with TestClient(app) as c:
        names = {p["name"] for p in c.get("/v1/presets").json()}
    assert names == {"ecommerce", "packaging", "jeu", "amenagement"}
