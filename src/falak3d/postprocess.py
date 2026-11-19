"""Nettoyage des maillages produits par Shap-E.

Le maillage brut sort de marching cubes : des centaines de milliers de faces,
quelques îlots parasites, l'axe Z vers le haut et une échelle arbitraire. Pour
qu'il soit utilisable (visionneuse web, moteur de jeu, configurateur), on le
réduit, on le réoriente en Y vers le haut comme l'attend glTF, on le met à
l'échelle réelle demandée et on le pose sur le sol.
"""

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass
class MeshReport:
    faces_in: int
    faces_out: int
    components_dropped: int
    size_m: tuple[float, float, float]
    watertight: bool


def keep_main_part(mesh: trimesh.Trimesh, min_ratio: float = 0.02) -> tuple[trimesh.Trimesh, int]:
    """Supprime les îlots dont l'aire est négligeable devant la pièce principale."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh, 0
    biggest = max(p.area for p in parts)
    kept = [p for p in parts if p.area >= biggest * min_ratio]
    return trimesh.util.concatenate(kept), len(parts) - len(kept)


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    colors = _vertex_colors(mesh)
    reduced = mesh.simplify_quadric_decimation(face_count=target_faces)
    if colors is not None:
        # La décimation perd les couleurs : chaque nouveau sommet reprend celle
        # du sommet d'origine le plus proche.
        _, idx = cKDTree(mesh.vertices).query(reduced.vertices)
        reduced.visual.vertex_colors = colors[idx]
    return reduced


def z_up_to_y_up(mesh: trimesh.Trimesh) -> None:
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))


def fit(mesh: trimesh.Trimesh, size_m: float) -> None:
    """Met la plus grande dimension à size_m, centre en X/Z, pose sur Y=0."""
    extent = mesh.extents.max()
    if extent > 0:
        mesh.apply_scale(size_m / extent)
    lo, hi = mesh.bounds
    center = (lo + hi) / 2
    mesh.apply_translation([-center[0], -lo[1], -center[2]])


def process(mesh: trimesh.Trimesh, target_faces: int, size_m: float,
            z_up: bool = True) -> tuple[trimesh.Trimesh, MeshReport]:
    faces_in = len(mesh.faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh, dropped = keep_main_part(mesh)
    mesh = decimate(mesh, target_faces)
    if z_up:
        z_up_to_y_up(mesh)
    fit(mesh, size_m)
    report = MeshReport(
        faces_in=faces_in,
        faces_out=len(mesh.faces),
        components_dropped=dropped,
        size_m=tuple(round(float(x), 3) for x in mesh.extents),
        watertight=bool(mesh.is_watertight),
    )
    return mesh, report


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if mesh.visual.kind != "vertex":
        return None
    return np.asarray(mesh.visual.vertex_colors)
