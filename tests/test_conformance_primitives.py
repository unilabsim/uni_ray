"""Primitive-scene conformance sweep: Warp caster vs MuJoCo CPU mj_ray (#299).

Both sweeps use fixed seeds and a fixed documented distance tolerance of
``ATOL = 1e-4`` (the caster computes in float32; mj_ray is float64). Hit/miss
and nearest-geom agreement are asserted strictly; a geom-id tie is only
accepted when both backends report the same distance within tolerance (two
surfaces at equal distance along the ray).
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco is required for the conformance sweep")
wp = pytest.importorskip("warp", reason="warp-lang is required for the conformance sweep")

from unisim.ray_query import RayTraceOutputs  # noqa: E402

import uni_ray  # noqa: E402
from uni_ray.mjbatch import build_collision_description  # noqa: E402

ATOL = 1e-4

SWEEP_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="0 0 0.1"/>
    <body pos="0 0 1.2">
      <freejoint/>
      <geom type="sphere" size="0.25" pos="0.45 0 0"/>
      <geom type="box" size="0.15 0.2 0.1" pos="-0.45 0 0" quat="0.9238795 0 0 0.3826834"/>
      <geom type="capsule" size="0.08 0.25" pos="0 0.55 0" quat="0.7071068 0.7071068 0 0"/>
      <geom type="cylinder" size="0.1 0.2" pos="0 -0.55 0" quat="0.7071068 0 0.7071068 0"/>
      <geom type="ellipsoid" size="0.12 0.2 0.07" pos="0 0 0.45"/>
    </body>
  </worldbody>
</mujoco>
"""

# Mirrors MuJoCo-LiDAR's test_warp_backend_capsule_cylinder_match_cpu scene:
# rotated capsule, upright capsule, upright cylinder, rays from a site at the
# origin.
CAPSULE_CYLINDER_XML = """
<mujoco>
  <worldbody>
    <geom type="capsule" pos="2 0 0.5" quat="0.7071068 0 0.7071068 0" size="0.3 0.8"/>
    <geom type="capsule" pos="0 3 -0.2" size="0.2 0.5"/>
    <geom type="cylinder" pos="-2 0 0" size="0.4 0.9"/>
  </worldbody>
</mujoco>
"""


def _random_quaternion(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    return quat / np.linalg.norm(quat)


def _random_rays(
    rng: np.random.Generator, count: int, center: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Origins on a shell around ``center``, aimed at jittered points near it."""
    directions_on_sphere = rng.normal(size=(count, 3))
    directions_on_sphere /= np.linalg.norm(directions_on_sphere, axis=1, keepdims=True)
    radii = rng.uniform(2.5, 4.0, size=(count, 1))
    origins = center + directions_on_sphere * radii
    targets = center + rng.uniform(-1.0, 1.0, size=(count, 3))
    directions = targets - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origins, directions


def _mj_ray_all(
    model, data, origins: np.ndarray, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full(origins.shape[0], -1.0)
    geom_ids = np.full(origins.shape[0], -1, dtype=np.int32)
    for index in range(origins.shape[0]):
        geom_id = np.zeros(1, dtype=np.int32)
        distances[index] = mujoco.mj_ray(
            model, data, origins[index], directions[index], None, 1, -1, geom_id
        )
        geom_ids[index] = geom_id[0]
    return distances, geom_ids


def _cpu_reference(
    model,
    data,
    origins: np.ndarray,
    directions: np.ndarray,
    max_distance: float,
    plane_geom_ids: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Contract-faithful CPU reference: mj_ray plus double-sided analytic planes.

    Two deliberate adaptations: mj_ray has no cutoff (its hits beyond
    ``max_distance`` are misses under the contract), and mj_ray only reports
    plane hits approached from the upper (+normal) side while the contract
    plane is the double-sided infinite local z = 0 plane (as in
    ``FakeRayCaster``). All test scenes attach their plane to the worldbody at
    the identity pose, so the analytic plane is world ``z = 0``.
    """
    distances, geom_ids = _mj_ray_all(model, data, origins, directions)
    for index in range(origins.shape[0]):
        if distances[index] < 0.0 or distances[index] > max_distance:
            distances[index] = -1.0
            geom_ids[index] = -1
        for plane_geom_id in plane_geom_ids:
            dz = directions[index, 2]
            if abs(dz) <= 1e-12:
                continue
            t = -origins[index, 2] / dz
            if 0.0 <= t <= max_distance and (distances[index] < 0.0 or t < distances[index]):
                distances[index] = t
                geom_ids[index] = plane_geom_id
    return distances, geom_ids


def _assert_matches_cpu(
    result, cpu_distances: np.ndarray, cpu_geom_ids: np.ndarray, max_distance: float
) -> None:
    assert result.distance.shape == cpu_distances.shape
    cpu_hit = cpu_distances >= 0.0
    np.testing.assert_array_equal(result.hit, cpu_hit)
    assert np.all(result.distance[~cpu_hit] == np.float32(max_distance))
    if np.any(cpu_hit):
        np.testing.assert_allclose(result.distance[cpu_hit], cpu_distances[cpu_hit], atol=ATOL)
        # Geom ids must agree unless the two backends report tied distances.
        tied = np.abs(result.distance[cpu_hit] - cpu_distances[cpu_hit]) <= ATOL
        geom_ids = np.where(tied, result.geom_id[cpu_hit], cpu_geom_ids[cpu_hit])
        np.testing.assert_array_equal(geom_ids, cpu_geom_ids[cpu_hit])


def test_primitive_sweep_matches_mj_ray_over_random_poses() -> None:
    model = mujoco.MjModel.from_xml_string(SWEEP_XML)
    data = mujoco.MjData(model)
    collision = build_collision_description(model)
    caster = uni_ray.create_ray_caster(num_envs=2, num_rays=512, collision=collision)
    caster.materialize(collision.scene)
    try:
        rng = np.random.default_rng(299)
        for _ in range(6):
            origins, directions = _random_rays(rng, 512, center=np.array([0.0, 0.0, 1.0]))
            for env_id in range(2):
                data.qpos[:3] = rng.uniform([-0.8, -0.8, 0.6], [0.8, 0.8, 1.8])
                data.qpos[3:7] = _random_quaternion(rng)
                mujoco.mj_forward(model, data)
                caster.update_pose(data.xpos[None], data.xquat[None], env_ids=[env_id])
                cpu_distances, cpu_geom_ids = _cpu_reference(
                    model, data, origins, directions, max_distance=10.0, plane_geom_ids=(0,)
                )
                result = caster.trace(
                    origins,
                    directions,
                    max_distance=10.0,
                    env_ids=[env_id],
                    outputs=RayTraceOutputs(geom_id=True),
                )
                _assert_matches_cpu(result, cpu_distances[None], cpu_geom_ids[None], 10.0)
    finally:
        caster.close()


def test_capsule_cylinder_rotated_axis_matches_cpu_2000_rays() -> None:
    model = mujoco.MjModel.from_xml_string(CAPSULE_CYLINDER_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    collision = build_collision_description(model)
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=2000, collision=collision)
    caster.materialize(collision.scene)
    try:
        caster.update_pose(data.xpos[None], data.xquat[None])
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, np.pi, 2000)
        phi = rng.uniform(-np.pi / 2, np.pi / 2, 2000)
        directions = np.stack(
            (np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)), axis=-1
        )
        origins = np.zeros((2000, 3))
        cpu_distances, cpu_geom_ids = _mj_ray_all(model, data, origins, directions)
        result = caster.trace(
            origins, directions, max_distance=100.0, outputs=RayTraceOutputs(geom_id=True)
        )
        _assert_matches_cpu(result, cpu_distances[None], cpu_geom_ids[None], 100.0)
    finally:
        caster.close()
