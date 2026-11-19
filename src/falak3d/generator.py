"""Générateurs de maillages.

ShapEGenerator charge les poids au premier appel (~30 s, ~3 Go de VRAM en fp16)
puis les garde. FakeGenerator produit des primitives : il sert aux tests et à
développer l'API sur une machine sans GPU.
"""

from pathlib import Path
from typing import Protocol

import numpy as np
import trimesh

from .presets import Preset


class Generator(Protocol):
    def generate(self, prompt: str, preset: Preset) -> trimesh.Trimesh: ...


class ShapEGenerator:
    def __init__(self, cache_dir: Path):
        # Sans cache_dir explicite, shap-e télécharge ses 2,7 Go de poids dans le
        # répertoire courant, donc dans la couche du conteneur : à chaque
        # recréation, onze minutes de téléchargement.
        self.cache_dir = str(cache_dir)
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            import torch
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.diffusion.sample import sample_latents
            from shap_e.models.download import load_config, load_model
            from shap_e.util.notebooks import decode_latent_mesh

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._loaded = {
                "torch": torch,
                "device": device,
                "xm": load_model("transmitter", device=device, cache_dir=self.cache_dir),
                "model": load_model("text300M", device=device, cache_dir=self.cache_dir),
                "diffusion": diffusion_from_config(
                    load_config("diffusion", cache_dir=self.cache_dir)),
                "sample": sample_latents,
                "decode": decode_latent_mesh,
            }
        return self._loaded

    def generate(self, prompt: str, preset: Preset) -> trimesh.Trimesh:
        p = self._load()
        cuda = p["device"].type == "cuda"
        try:
            latents = p["sample"](
                batch_size=1, model=p["model"], diffusion=p["diffusion"],
                guidance_scale=preset.guidance_scale, model_kwargs={"texts": [prompt]},
                progress=False, clip_denoised=True, use_fp16=cuda, use_karras=True,
                karras_steps=preset.steps, sigma_min=1e-3, sigma_max=160, s_churn=0,
            )
            raw = p["decode"](p["xm"], latents[0]).tri_mesh()
            mesh = trimesh.Trimesh(vertices=raw.verts, faces=raw.faces, process=False)
            if all(c in raw.vertex_channels for c in "RGB"):
                rgb = np.stack([raw.vertex_channels[c] for c in "RGB"], axis=1)
                mesh.visual.vertex_colors = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            return mesh
        finally:
            if cuda:
                p["torch"].cuda.empty_cache()


class FakeGenerator:
    """Primitive choisie d'après le texte, en Z vers le haut comme Shap-E."""

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.calls: list[str] = []

    def generate(self, prompt: str, preset: Preset) -> trimesh.Trimesh:
        self.calls.append(prompt)
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("génération impossible")
        if "mug" in prompt or "bottle" in prompt or "barrel" in prompt:
            mesh = trimesh.creation.cylinder(radius=0.4, height=1.0, sections=64)
        elif "lamp" in prompt or "tree" in prompt:
            mesh = trimesh.creation.cone(radius=0.5, height=1.5, sections=64)
        else:
            mesh = trimesh.creation.box(extents=[1.0, 0.6, 0.8])
        mesh = mesh.subdivide().subdivide()
        mesh.visual.vertex_colors = np.tile([142, 27, 46, 255], (len(mesh.vertices), 1))
        return mesh
