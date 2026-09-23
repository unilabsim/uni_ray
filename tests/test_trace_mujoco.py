"""End-to-end mjbatch pose-sync trace tests against MuJoCo CPU ray casting (#298).

The scene is a ground plane, a static mesh geom, and a freejoint body carrying
one geom of every supported primitive type. Correctness is checked against
``mujoco.mj_ray`` with a fixed float32 tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from unisim.ray_query import RayTraceOutputs

mujoco = pytest.importorskip("mujoco", reason="mujoco is required for the mjbatch caster tests")
wp = pytest.importorskip("warp", reason="warp-lang is required to create the caster")

import uni_ray  # noqa: E402
from uni_ray.mjbatch import build_collision_description  # noqa: E402

SCENE_XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
  </asset>
  <worldbody>
    <geom type="plane" size="0 0 0.1"/>
    <geom type="mesh" mesh="tet" pos="3 0 0"/>
    <body pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.3" pos="0.4 0 0"/>
      <geom type="box" size="0.1 0.1 0.1" pos="-0.4 0 0"/>
      <geom type="capsule" size="0.05 0.2" pos="0 0.5 0"/>
      <geom type="cylinder" size="0.08 0.15" pos="0 -0.5 0" quat="0.7071068 0.7071068 0 0"/>
      <geom type="ellipsoid" size="0.1 0.2 0.05" pos="0.8 0.5 0"/>
    </body>
  </worldbody>
</mujoco>
"""

# Rays aimed at each primitive, the static mesh, the far-out infinite plane,
# and straight up (a guaranteed miss).
RAY_ORIGINS = np.array(
    [
        [0.7, -0.2, 4.0],
        [-0.1, -0.2, 4.0],
        [0.3, 0.3, 4.0],
        [0.3, -0.7, 4.0],
        [1.1, 0.3, 4.0],
        [3.5, 0.4, 3.0],
        [8.0, 8.0, 4.0],
        [0.0, 0.0, 4.0],
    ]
)
RAY_DIRECTIONS = np.tile(np.array([[0.0, 0.0, -1.0]]), (8, 1))
RAY_DIRECTIONS[7] = [0.0, 0.0, 1.0]

POSE_A = np.array([0.3, -0.2, 1.5, 0.9659258, 0.258819, 0.0, 0.0])
POSE_B = np.array([-0.5, 0.4, 0.8, 1.0, 0.0, 0.0, 0.0])

ATOL = 1e-4


@pytest.fixture()
def model_and_data():
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    return model, mujoco.MjData(model)


@pytest.fixture()
def caster(model_and_data):
    model, _ = model_and_data
    collision = build_collision_description(model)
    caster = uni_ray.create_ray_caster(
        num_envs=2, num_rays=RAY_ORIGINS.shape[0], collision=collision
    )
    caster.materialize(collision.scene)
    yield caster
    caster.close()


def _sync_pose(model, data, caster, qpos: np.ndarray, env_ids) -> None:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    caster.update_pose(
        np.tile(data.xpos[None], (len(env_ids), 1, 1)),
        np.tile(data.xquat[None], (len(env_ids), 1, 1)),
        env_ids=env_ids,
    )


def _mj_ray_distances(model, data) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full(RAY_ORIGINS.shape[0], -1.0)
    geom_ids = np.full(RAY_ORIGINS.shape[0], -1, dtype=np.int32)
    for index in range(RAY_ORIGINS.shape[0]):
        geom_id = np.zeros(1, dtype=np.int32)
        distances[index] = mujoco.mj_ray(
            model,
            data,
            RAY_ORIGINS[index].astype(np.float64),
            RAY_DIRECTIONS[index],
            None,
            1,
            -1,
            geom_id,
        )
        geom_ids[index] = geom_id[0]
    return distances, geom_ids


def test_end_to_end_lifecycle(model_and_data, caster) -> None:
    model, data = model_and_data
    # descriptor built from MjModel -> create -> materialize (fixture) ->
    # update_pose from plain numpy body pos/quat -> trace -> host readback.
    _sync_pose(model, data, caster, POSE_A, env_ids=[0, 1])
    result = caster.trace(RAY_ORIGINS, RAY_DIRECTIONS, max_distance=10.0)
    assert result.distance.shape == (2, 8)
    assert result.hit.dtype == np.bool_
    assert np.isfinite(result.distance).all()
    caster.close()  # explicit close; the fixture close afterwards is idempotent


def test_matches_mujoco_mj_ray(model_and_data, caster) -> None:
    model, data = model_and_data
    _sync_pose(model, data, caster, POSE_A, env_ids=[0, 1])
    result = caster.trace(
        RAY_ORIGINS, RAY_DIRECTIONS, max_distance=10.0, outputs=RayTraceOutputs(geom_id=True)
    )
    expected_dist, expected_geom = _mj_ray_distances(model, data)
    for env_row in range(2):
        for ray in range(RAY_ORIGINS.shape[0]):
            if expected_dist[ray] < 0.0:
                assert not result.hit[env_row, ray]
                assert result.distance[env_row, ray] == pytest.approx(10.0)
                assert result.geom_id[env_row, ray] == -1
            else:
                assert result.hit[env_row, ray]
                assert result.distance[env_row, ray] == pytest.approx(
                    expected_dist[ray], abs=ATOL
                )
                assert result.geom_id[env_row, ray] == expected_geom[ray]


def test_pose_sync_updates_geometry_per_env(model_and_data, caster) -> None:
    model, data = model_and_data
    _sync_pose(model, data, caster, POSE_A, env_ids=[0])
    _sync_pose(model, data, caster, POSE_B, env_ids=[1])
    result = caster.trace(RAY_ORIGINS, RAY_DIRECTIONS, max_distance=10.0)

    data.qpos[:] = POSE_A
    mujoco.mj_forward(model, data)
    dist_a, _ = _mj_ray_distances(model, data)
    data.qpos[:] = POSE_B
    mujoco.mj_forward(model, data)
    dist_b, _ = _mj_ray_distances(model, data)

    for ray in range(RAY_ORIGINS.shape[0]):
        if dist_a[ray] >= 0.0:
            assert result.distance[0, ray] == pytest.approx(dist_a[ray], abs=ATOL)
        else:
            assert not result.hit[0, ray]
        if dist_b[ray] >= 0.0:
            assert result.distance[1, ray] == pytest.approx(dist_b[ray], abs=ATOL)
        else:
            assert not result.hit[1, ray]


def test_miss_and_max_distance_cutoff(model_and_data, caster) -> None:
    model, data = model_and_data
    _sync_pose(model, data, caster, POSE_A, env_ids=[0, 1])
    # The up-pointing ray (index 7) misses: distance is clipped to
    # max_distance and hit is False.
    result = caster.trace(RAY_ORIGINS, RAY_DIRECTIONS, max_distance=7.5)
    assert not result.hit[0, 7]
    assert result.distance[0, 7] == pytest.approx(7.5)
    # A max_distance below every real hit distance turns hits into misses.
    cutoff = caster.trace(RAY_ORIGINS, RAY_DIRECTIONS, max_distance=0.5)
    assert not cutoff.hit.any()
    np.testing.assert_allclose(cutoff.distance, 0.5)


def test_hit_points_land_on_surfaces(model_and_data, caster) -> None:
    model, data = model_and_data
    _sync_pose(model, data, caster, POSE_A, env_ids=[0, 1])
    result = caster.trace(
        RAY_ORIGINS, RAY_DIRECTIONS, max_distance=10.0, outputs=RayTraceOutputs(hit_point=True)
    )
    # The plane ray (index 6) hits z = 0; every downward hit point is finite.
    np.testing.assert_allclose(result.hit_point[0, 6], [8.0, 8.0, 0.0], atol=ATOL)
    assert np.isfinite(result.hit_point[0][result.hit[0]]).all()


def test_selected_rows_trace(model_and_data, caster) -> None:
    model, data = model_and_data
    _sync_pose(model, data, caster, POSE_A, env_ids=[0])
    _sync_pose(model, data, caster, POSE_B, env_ids=[1])
    selected = caster.trace(RAY_ORIGINS, RAY_DIRECTIONS, max_distance=10.0, env_ids=[1])
    assert selected.distance.shape == (1, 8)
    # Copy before the next trace: result rows are views into reused buffers.
    selected_distance = selected.distance[0].copy()
    full = caster.trace(RAY_ORIGINS, RAY_DIRECTIONS, max_distance=10.0)
    np.testing.assert_allclose(selected_distance, full.distance[1], atol=ATOL)
