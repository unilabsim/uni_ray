"""Minimal installed-distribution smoke test used by packaging workflows."""

from __future__ import annotations

import numpy as np
import unisim

scene = unisim.RaySceneDescription(
    num_bodies=1,
    geom_types=("plane",),
    geom_sizes=np.zeros((1, 3), dtype=np.float64),
    geom_local_pos=np.zeros((1, 3), dtype=np.float64),
    geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    geom_body_ids=np.array([0], dtype=np.intp),
    body_names=("body",),
)
caster = unisim.create_ray_caster(
    "uni_ray",
    num_envs=1,
    num_rays=1,
    device="cpu",
)
caster.materialize(scene)
caster.update_pose(
    np.zeros((1, 1, 3), dtype=np.float64),
    np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (1, 1, 4)).copy(),
)
result = caster.trace(
    np.array([[0.0, 0.0, 2.0]], dtype=np.float64),
    np.array([[0.0, 0.0, -1.0]], dtype=np.float64),
    max_distance=4.0,
)
assert result.hit.item()
assert abs(float(result.distance.item()) - 2.0) < 1.0e-5
