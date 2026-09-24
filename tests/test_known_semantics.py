"""Pinned resolutions of the known semantic differences (#299).

(a) Rays originating inside a primitive report the forward *exit* distance,
    matching ``mujoco.mj_ray`` (the upstream MuJoCo-LiDAR kernels reported
    0.0; uni_ray deliberately deviates, see the header of
    ``src/uni_ray/geometry.py``).
(b) Contract planes are the infinite, double-sided local ``z = 0`` plane and
    ignore their sizes, even if a ``RaySceneDescription`` carries a nonzero
    (rendering-grid) size. ``mujoco.mj_ray`` instead treats a nonzero plane
    size as a finite grid and only reports hits approached from the upper
    (+normal) side; uni_ray follows the contract and ``FakeRayCaster``.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco is required for the semantics tests")
wp = pytest.importorskip("warp", reason="warp-lang is required for the semantics tests")

from unisim.ray_query import RaySceneDescription  # noqa: E402

import uni_ray  # noqa: E402
from uni_ray.mjbatch import build_collision_description  # noqa: E402

ATOL = 1e-4

INSIDE_XML = """
<mujoco>
  <worldbody>
    <geom type="sphere" size="0.5" pos="0 0 1"/>
    <geom type="box" size="0.2 0.3 0.4" pos="3 0 1"/>
    <geom type="capsule" size="0.1 0.3" pos="6 0 1"/>
    <geom type="cylinder" size="0.1 0.3" pos="9 0 1"/>
    <geom type="ellipsoid" size="0.2 0.3 0.1" pos="12 0 1"/>
  </worldbody>
</mujoco>
"""


def test_inside_origin_rays_return_exit_distance_like_mj_ray() -> None:
    model = mujoco.MjModel.from_xml_string(INSIDE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    collision = build_collision_description(model)
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=5, collision=collision)
    caster.materialize(collision.scene)
    try:
        caster.update_pose(data.xpos[None], data.xquat[None])
        # Rays from each geom's center straight down; expected exit distances
        # come from mj_ray itself (sphere radius 0.5, box half-height 0.4,
        # capsule half_length+radius 0.4, cylinder half_length 0.3, ellipsoid
        # z semi-axis 0.1).
        origins = np.array(
            [[0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [6.0, 0.0, 1.0], [9.0, 0.0, 1.0], [12.0, 0.0, 1.0]]
        )
        directions = np.tile(np.array([[0.0, 0.0, -1.0]]), (5, 1))
        result = caster.trace(origins, directions, max_distance=10.0)
        assert result.hit.all()
        for index in range(5):
            geom_id = np.zeros(1, dtype=np.int32)
            cpu = mujoco.mj_ray(
                model, data, origins[index], directions[index], None, 1, -1, geom_id
            )
            assert result.distance[0, index] == pytest.approx(cpu, abs=ATOL)
            assert geom_id[0] == index
    finally:
        caster.close()


def test_plane_sizes_are_ignored_and_hits_are_double_sided() -> None:
    # A descriptor whose plane carries a nonzero size (a rendering grid in
    # MuJoCo) must still behave as the infinite contract plane.
    scene = RaySceneDescription(
        num_bodies=1,
        geom_types=("plane",),
        geom_sizes=np.array([[0.5, 0.5, 0.0]]),
        geom_local_pos=np.zeros((1, 3)),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.array([0]),
    )
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=3)
    caster.materialize(scene)
    try:
        origins = np.array(
            [
                [50.0, 50.0, 2.0],  # far outside the 0.5 grid, above: hits at 2.0
                [0.0, 0.0, -2.0],  # below the plane pointing up: contract hits at 2.0
                [0.0, 0.0, 2.0],  # above pointing up: miss
            ]
        )
        directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        result = caster.trace(origins, directions, max_distance=10.0)
        np.testing.assert_allclose(result.distance[0], [2.0, 2.0, 10.0], atol=ATOL)
        assert result.hit[0].tolist() == [True, True, False]
    finally:
        caster.close()


def test_mjbatch_zeroes_mjcf_plane_rendering_grid() -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="5 5 0.1"/></worldbody></mujoco>'
    )
    collision = build_collision_description(model)
    np.testing.assert_array_equal(collision.scene.geom_sizes[0], np.zeros(3))
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=1, collision=collision)
    caster.materialize(collision.scene)
    try:
        # mj_ray would miss here (finite 5x5 grid, single-sided); the contract
        # plane is infinite, so the caster hits at distance 4.0.
        result = caster.trace(
            np.array([[8.0, 8.0, 4.0]]), np.array([[0.0, 0.0, -1.0]]), max_distance=10.0
        )
        assert result.hit[0, 0]
        assert result.distance[0, 0] == pytest.approx(4.0, abs=ATOL)
    finally:
        caster.close()
