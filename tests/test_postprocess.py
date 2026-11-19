import numpy as np
import trimesh

from falak3d.postprocess import decimate, keep_main_part, process


def tall_box():
    # 2 m de haut le long de Z, comme sort un objet Shap-E.
    return trimesh.creation.box(extents=[0.5, 0.5, 2.0]).subdivide().subdivide()


def test_z_up_becomes_y_up_and_scaled():
    mesh, report = process(tall_box(), target_faces=10_000, size_m=1.0)
    x, y, z = mesh.extents
    assert np.isclose(y, 1.0) and np.isclose(x, 0.25) and np.isclose(z, 0.25)
    assert report.size_m == (0.25, 1.0, 0.25)


def test_object_sits_on_the_ground_and_is_centered():
    mesh, _ = process(tall_box(), target_faces=10_000, size_m=1.0)
    lo, hi = mesh.bounds
    assert np.isclose(lo[1], 0.0)
    assert np.allclose((lo[[0, 2]] + hi[[0, 2]]) / 2, 0.0)


def test_small_islands_are_removed():
    main = trimesh.creation.box(extents=[1, 1, 1])
    speck = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    speck.apply_translation([3, 0, 0])
    mesh, dropped = keep_main_part(trimesh.util.concatenate([main, speck]))
    assert dropped == 1
    assert np.isclose(mesh.extents.max(), 1.0)


def test_decimation_keeps_vertex_colors():
    sphere = trimesh.creation.icosphere(subdivisions=5)
    sphere.visual.vertex_colors = np.tile([142, 27, 46, 255], (len(sphere.vertices), 1))
    reduced = decimate(sphere, target_faces=2000)
    assert len(reduced.faces) <= 2100
    assert (reduced.visual.vertex_colors[:, :3] == [142, 27, 46]).all()
