"""Hfield support tests (#2): cold-path triangulation, lifecycle, mj_ray parity.

Hfield geoms are triangulated once in the geom-local frame at descriptor build
and bound as static meshes, so the runtime geom-pose transform applies
unchanged. ``mujoco.mj_ray`` intersects the same per-cell triangle pair the
triangulation produces, so the fixed tolerance is the float32 device tolerance
used by the other sweeps (``ATOL = 1e-4``), not an approximation budget; the
worst observed error is recorded in docs/conformance.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from unisim.ray_query import RayTraceOutputs

mujoco = pytest.importorskip("mujoco", reason="mujoco is required for the hfield tests")
wp = pytest.importorskip("warp", reason="warp-lang is required to create the caster")

import uni_ray  # noqa: E402
from uni_ray.mjbatch import build_collision_description  # noqa: E402

ATOL = 1e-4
NUM_RAYS = 512

# Non-flat terrain (a bump over a tilted base, deliberately non-symmetric so
# the hfield_data row orientation is exercised) so every grid cell is hit.
TERRAIN_XML = """
<mujoco>
  <asset>
    <hfield name="terrain" nrow="5" ncol="5" size="2 2 0.5 0.1"
            elevation="0.0 0.1 0.2 0.3 0.4
                       0.1 0.3 0.5 0.35 0.1
                       0.2 0.5 0.9 0.5 0.2
                       0.15 0.3 0.55 0.3 0.1
                       0.0 0.05 0.15 0.25 0.35"/>
  </asset>
  <worldbody>
    <geom type="hfield" hfield="terrain"/>
  </worldbody>
</mujoco>
"""

MOVING_TERRAIN_XML = """
<mujoco>
  <asset>
    <hfield name="terrain" nrow="4" ncol="4" size="1 1 0.4 0.1"
            elevation="0.0 0.2 0.4 0.6
                       0.2 0.6 0.8 0.4
                       0.4 0.8 0.6 0.2
                       0.6 0.4 0.2 0.0"/>
  </asset>
  <worldbody>
    <body pos="0 0 1.0">
      <freejoint/>
      <geom type="hfield" hfield="terrain"/>
    </body>
  </worldbody>
</mujoco>
"""

POSE_A = np.array([0.4, -0.3, 1.2, 0.9659258, 0.258819, 0.0, 0.0])
POSE_B = np.array([-0.6, 0.5, 0.4, 1.0, 0.0, 0.0, 0.0])


def _rays() -> tuple[np.ndarray, np.ndarray]:
    """Mixed ray families: from above (top surface), from the sides at all
    heights aimed at the terrain center (skirt walls), from below the base
    (bottom face), and guaranteed sky misses."""
    rng = np.random.default_rng(11)
    origins = np.zeros((NUM_RAYS, 3))
    directions = np.zeros((NUM_RAYS, 3))
    third = NUM_RAYS // 3
    rest = NUM_RAYS - 2 * third
    origins[:third, 0] = rng.uniform(-2.5, 2.5, third)
    origins[:third, 1] = rng.uniform(-2.5, 2.5, third)
    origins[:third, 2] = rng.uniform(0.8, 3.0, third)
    directions[:third, 0] = rng.uniform(-0.4, 0.4, third)
    directions[:third, 1] = rng.uniform(-0.4, 0.4, third)
    directions[:third, 2] = -1.0
    directions[::16, :] = [0.0, 0.0, 1.0]  # guaranteed misses pointing at the sky
    side = slice(third, 2 * third)
    origins[side, 0] = rng.uniform(-4.0, 4.0, third)
    origins[side, 1] = rng.uniform(-4.0, 4.0, third)
    origins[side, 2] = rng.uniform(-0.4, 0.8, third)
    directions[side] = -origins[side]
    below = slice(2 * third, NUM_RAYS)
    origins[below, 0] = rng.uniform(-2.5, 2.5, rest)
    origins[below, 1] = rng.uniform(-2.5, 2.5, rest)
    origins[below, 2] = rng.uniform(-1.0, -0.2, rest)
    directions[below, 0] = rng.uniform(-0.4, 0.4, rest)
    directions[below, 1] = rng.uniform(-0.4, 0.4, rest)
    directions[below, 2] = 1.0
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origins, directions


def _mj_ray_reference(model, data, origins, directions, max_distance):
    distances = np.full(origins.shape[0], -1.0)
    geom_ids = np.full(origins.shape[0], -1, dtype=np.int32)
    for index in range(origins.shape[0]):
        geom_id = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            model, data, origins[index].astype(np.float64), directions[index], None, 1, -1, geom_id
        )
        # mj_ray has no cutoff; clip to the contract's max_distance window.
        if 0.0 <= distance < max_distance:
            distances[index] = distance
            geom_ids[index] = geom_id[0]
    return distances, geom_ids


def _sync_pose(model, data, caster, qpos) -> None:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    caster.update_pose(
        np.array(data.xpos, copy=True)[None], np.array(data.xquat, copy=True)[None]
    )


def test_hfield_lifecycle_and_correctness_vs_mj_ray() -> None:
    model = mujoco.MjModel.from_xml_string(TERRAIN_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    collision = build_collision_description(model)
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=NUM_RAYS, collision=collision)
    caster.materialize(collision.scene)

    origins, directions = _rays()
    _sync_pose(model, data, caster, data.qpos)
    result = caster.trace(
        origins, directions, max_distance=10.0, outputs=RayTraceOutputs(geom_id=True)
    )
    ref_distances, ref_geom_ids = _mj_ray_reference(model, data, origins, directions, 10.0)

    uni_hit = np.array(result.hit)[0]
    uni_distances = np.array(result.distance)[0]
    uni_geom_ids = np.array(result.geom_id)[0]
    ref_hit = ref_geom_ids >= 0
    np.testing.assert_array_equal(uni_hit, ref_hit)
    np.testing.assert_array_equal(uni_geom_ids[ref_hit], ref_geom_ids[ref_hit])
    np.testing.assert_allclose(uni_distances[ref_hit], ref_distances[ref_hit], atol=ATOL)
    assert ref_hit.any() and (~ref_hit).any()  # the sweep exercises hits and misses
    caster.close()
    with pytest.raises(Exception, match="closed"):
        caster.trace(origins, directions, max_distance=10.0)


def test_hfield_geom_pose_follows_body_pose() -> None:
    model = mujoco.MjModel.from_xml_string(MOVING_TERRAIN_XML)
    data = mujoco.MjData(model)
    collision = build_collision_description(model)
    caster = uni_ray.create_ray_caster(num_envs=2, num_rays=NUM_RAYS, collision=collision)
    caster.materialize(collision.scene)
    origins, directions = _rays()

    references = []
    body_pos, body_quat = [], []
    for pose in (POSE_A, POSE_B):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        body_pos.append(np.array(data.xpos, copy=True))
        body_quat.append(np.array(data.xquat, copy=True))
        references.append(_mj_ray_reference(model, data, origins, directions, 10.0))
    caster.update_pose(np.stack(body_pos), np.stack(body_quat))

    for env_id, (ref_distances, ref_geom_ids) in enumerate(references):
        result = caster.trace(
            origins,
            directions,
            max_distance=10.0,
            env_ids=[env_id],
            outputs=RayTraceOutputs(geom_id=True),
        )
        ref_hit = ref_geom_ids >= 0
        np.testing.assert_array_equal(np.array(result.hit)[0], ref_hit)
        np.testing.assert_allclose(
            np.array(result.distance)[0][ref_hit], ref_distances[ref_hit], atol=ATOL
        )
    caster.close()
