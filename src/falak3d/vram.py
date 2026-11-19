"""Partage de la carte graphique avec le LLM local.

Sur une carte de 6 Go, le LLM de l'assistant (≈4,5 Go résident) et Shap-E
(≈3 Go en fp16) ne tiennent pas ensemble. Avant un job, on demande à Ollama de
décharger son modèle (keep_alive=0) ; quand la file est vide, on le recharge
pour que l'assistant ne reparte pas à froid, ce qui coûte plus d'une minute.
"""

import httpx


class OllamaVram:
    def __init__(self, url: str, model: str, timeout: float = 30.0):
        self.url, self.model, self.timeout = url, model, timeout

    def _keep_alive(self, value: int) -> bool:
        try:
            r = httpx.post(f"{self.url}/api/generate", timeout=self.timeout,
                           json={"model": self.model, "keep_alive": value})
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def release(self) -> bool:
        return self._keep_alive(0)

    def warm(self) -> bool:
        return self._keep_alive(-1)


class NoVram:
    def release(self) -> bool:
        return True

    def warm(self) -> bool:
        return True
