"""mjwarp-profile adapter tests (#300).

The mjwarp adapter (``uni_ray.mjwarp``) sources the collision descriptor from
a ``mujoco_warp.Model`` and feeds the shared profile-neutral descriptor build,
so the caster, pose-sync contract, and kernels are identical to the mjbatch
profile. These tests pin: the minimal trace through the UniSim plugin
contract, trace parity with the mjbatch profile on the same static scene, the
fail-closed surface, and the adapter lifecycle.
"""

from __future__ import annotations

import numpy as np
import pytest
from unisim.errors import BackendError, UnsupportedCapabilityError
from unisim.ray_query import RayCaster, RayGeomType, RayTraceOutputs, RayTraceResult

mujoco = pytest.importorskip("mujoco", reason="mujoco is required to compile the test models")
mjw = pytest.importorskip("mujoco_warp", reason="mujoco-warp is required for the mjwarp profile")
wp = pytest.importorskip("warp", reason="warp-lang is required to create the caster")

import uni_ray  # noqa: E402
from uni_ray.mjbatch import build_collision_description as build_from_mjmodel  # noqa: E402
from uni_ray.mjwarp import (  # noqa: E402
    build_collision_description,
    create_ray_caster,
    rebuild_caster_from_mjwarp,
)

ATOL = 1e-4

MINI_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="0 0 0.1"/>
    <body pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""

# Exercises every descriptor path: plane, hfield (triangulated), static mesh,
# and primitive geoms on a freejoint body.
SCENE_XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
    <hfield name="hf" nrow="5" ncol="5" size="2 2 0.5 0.1"
            elevation="0.0 0.1 0.2 0.3 0.4
                       0.1 0.3 0.5 0.35 0.1
                       0.2 0.5 0.9 0.5 0.2
                       0.15 0.3 0.55 0.3 0.1
                       0.0 0.05 0.15 0.25 0.35"/>
  </asset>
  <worldbody>
    <geom type="plane" size="0 0 0.1"/>
    <geom type="hfield" hfield="hf" pos="0 5 0"/>
    <geom type="mesh" mesh="tet" pos="3 0 0"/>
    <geom type="mesh" mesh="tet" pos="6 0 0"/>
    <body pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.3" pos="0.4 0 0"/>
      <geom type="capsule" size="0.05 0.2" pos="0 0.5 0"/>
    </body>
  </worldbody>
</mujoco>
"""

V2_XML = SCENE_XML.replace(
    'vertex="0 0 0  1 0 0  0 1 0  0 0 1"', 'vertex="0 0 0  2 0 0  0 2 0  0 0 2"'
)


def _models(xml: str) -> tuple:
    mj_model = mujoco.MjModel.from_xml_string(xml)
    return mj_model, mjw.put_model(mj_model)


def _poses(rng: np.random.Generator, num_envs: int, num_bodies: int) -> tuple:
    body_pos = np.zeros((num_envs, num_bodies, 3))
    body_pos[:, 1:] = rng.normal(size=(num_envs, num_bodies - 1, 3)) * 0.2
    body_pos[:, 1:, 2] += 1.0
    body_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_envs, num_bodies, 1))
    body_quat[:, 1:] = rng.normal(size=(num_envs, num_bodies - 1, 4))
    body_quat[:, 1:] /= np.linalg.norm(body_quat[:, 1:], axis=-1, keepdims=True)
    return body_pos, body_quat


def _rays(num_rays: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Down-rays over the plane/body/mesh area plus rays aimed at the hfield."""
    rng = np.random.default_rng(11)
    origins = np.zeros((num_rays, 3))
    directions = np.tile([[0.0, 0.0, -1.0]], (num_rays, 1))
    half = num_rays // 2
    origins[:half, 0] = rng.uniform(-1.5, 7.0, half)
    origins[:half, 1] = rng.uniform(-1.5, 1.5, half)
    origins[:half, 2] = rng.uniform(2.0, 4.0, half)
    origins[half:, 0] = rng.uniform(-2.5, 2.5, num_rays - half)
    origins[half:, 1] = 5.0 + rng.uniform(-2.5, 2.5, num_rays - half)
    origins[half:, 2] = rng.uniform(1.5, 3.0, num_rays - half)
    directions[half:, 0] = rng.uniform(-0.3, 0.3, num_rays - half)
    directions[::16] = [0.0, 0.0, 1.0]  # guaranteed misses pointing at the sky
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origins, directions


def test_descriptor_matches_mjbatch_profile() -> None:
    mj_model, mjw_model = _models(SCENE_XML)
    ref = build_from_mjmodel(mj_model)
    collision = build_collision_description(mjw_model)
    scene, ref_scene = collision.scene, ref.scene

    assert scene.num_bodies == ref_scene.num_bodies
    assert scene.geom_types == ref_scene.geom_types
    assert scene.geom_types == (
        RayGeomType.PLANE,
        RayGeomType.MESH,
        RayGeomType.MESH,
        RayGeomType.MESH,
        RayGeomType.SPHERE,
        RayGeomType.CAPSULE,
    )
    np.testing.assert_array_equal(scene.geom_body_ids, ref_scene.geom_body_ids)
    # The mjwarp profile sources float32 model fields (mjbatch reads float64),
    # so sizes/local poses match up to float32 rounding.
    np.testing.assert_allclose(scene.geom_sizes, ref_scene.geom_sizes, atol=1e-6)
    np.testing.assert_allclose(scene.geom_local_pos, ref_scene.geom_local_pos, atol=1e-6)
    np.testing.assert_allclose(scene.geom_local_quat, ref_scene.geom_local_quat, atol=1e-6)
    # Mesh data (incl. the triangulated hfield) is float32 on both profiles
    # and must be bitwise identical; dedup by (geom type, data id) matches.
    assert len(collision.meshes) == len(ref.meshes) == 2
    np.testing.assert_array_equal(collision.geom_mesh_ids, ref.geom_mesh_ids)
    for mesh, ref_mesh in zip(collision.meshes, ref.meshes):
        np.testing.assert_array_equal(mesh.points, ref_mesh.points)
        np.testing.assert_array_equal(mesh.indices, ref_mesh.indices)


def test_minimal_trace_through_plugin_contract() -> None:
    _, mjw_model = _models(MINI_XML)
    collision = build_collision_description(mjw_model)

    from unisim.factory import create_ray_caster as factory_create

    caster = factory_create("uni_ray", num_envs=1, num_rays=3, collision=collision)
    assert isinstance(caster, RayCaster)  # no warp/mujoco_warp types leak out
    caster.materialize(collision.scene)
    origins = np.array([[0.0, 0.0, 2.0], [4.0, 4.0, 2.0], [0.0, 0.0, 2.0]])
    directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    # A freshly materialized scene has identity body poses; place the body at
    # its XML pose (z=1, sphere spans 0.5..1.5) before the first trace.
    body_pos = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    body_quat = np.array([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    caster.update_pose(body_pos, body_quat)
    result = caster.trace(origins, directions, max_distance=10.0)
    assert isinstance(result, RayTraceResult)
    np.testing.assert_allclose(result.distance[0], [0.5, 2.0, 10.0], atol=1e-5)
    assert result.hit[0].tolist() == [True, True, False]

    body_pos[0, 1] = [0.0, 0.0, 3.0]
    caster.update_pose(body_pos, body_quat)
    moved = caster.trace(origins, directions, max_distance=10.0)
    np.testing.assert_allclose(moved.distance[0], [2.0, 2.0, 0.5], atol=1e-5)
    caster.close()


def test_trace_parity_with_mjbatch() -> None:
    mj_model, mjw_model = _models(SCENE_XML)
    ref_collision = build_from_mjmodel(mj_model)
    collision = build_collision_description(mjw_model)
    num_envs = 2
    rng = np.random.default_rng(7)
    body_pos, body_quat = _poses(rng, num_envs, mj_model.nbody)
    origins, directions = _rays()

    results = []
    for profile_collision in (ref_collision, collision):
        caster = uni_ray.create_ray_caster(
            num_envs=num_envs, num_rays=origins.shape[0], collision=profile_collision
        )
        caster.materialize(profile_collision.scene)
        caster.update_pose(body_pos, body_quat)
        result = caster.trace(
            origins, directions, max_distance=10.0, outputs=RayTraceOutputs(geom_id=True)
        )
        results.append(
            (
                np.array(result.hit, copy=True),
                np.array(result.distance, copy=True),
                np.array(result.geom_id, copy=True),
            )
        )
        caster.close()

    (ref_hit, ref_dist, ref_geom), (hit, dist, geom) = results
    np.testing.assert_array_equal(hit, ref_hit)
    np.testing.assert_array_equal(geom[ref_hit], ref_geom[ref_hit])
    np.testing.assert_allclose(dist[ref_hit], ref_dist[ref_hit], atol=ATOL)
    assert ref_hit.any() and (~ref_hit).any()


def test_adapter_create_ray_caster_lifecycle() -> None:
    _, mjw_model = _models(MINI_XML)
    caster = create_ray_caster(mjw_model, num_envs=2, num_rays=3)
    assert isinstance(caster, RayCaster)

    # Bound already (materialized by the convenience): a second materialize
    # fails closed, per the contract.
    with pytest.raises(BackendError, match="materialized"):
        caster.materialize(build_collision_description(mjw_model).scene)

    origins = np.array([[0.0, 0.0, 2.0], [4.0, 4.0, 2.0], [0.0, 0.0, 2.0]])
    directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    # Freshly materialized scenes have identity body poses: the freejoint
    # body sits at the world origin (sphere spans z -0.5..0.5).
    result = caster.trace(origins, directions, max_distance=10.0)
    np.testing.assert_allclose(result.distance[0], [1.5, 2.0, 10.0], atol=1e-5)

    body_pos = np.zeros((2, 2, 3))
    body_pos[:, 1] = [0.0, 0.0, 1.0]
    body_quat = np.tile([[1.0, 0.0, 0.0, 0.0]], (2, 2, 1))
    caster.update_pose(body_pos[1:], body_quat[1:], env_ids=[1])
    moved = caster.trace(origins, directions, max_distance=10.0)
    np.testing.assert_allclose(moved.distance[0], [1.5, 2.0, 10.0], atol=1e-5)
    np.testing.assert_allclose(moved.distance[1], [0.5, 2.0, 10.0], atol=1e-5)

    caster.close()
    with pytest.raises(BackendError, match="closed"):
        caster.trace(origins, directions, max_distance=10.0)


def test_rebuild_caster_from_mjwarp() -> None:
    mj_model, mjw_model = _models(SCENE_XML)
    _, mjw_model_v2 = _models(V2_XML)
    num_envs = 1
    rng = np.random.default_rng(7)
    body_pos, body_quat = _poses(rng, num_envs, mj_model.nbody)
    origins, directions = _rays()

    caster = create_ray_caster(mjw_model, num_envs=num_envs, num_rays=origins.shape[0])
    caster.update_pose(body_pos, body_quat)
    before = np.array(
        caster.trace(origins, directions, max_distance=10.0).distance, copy=True
    )

    collision_v2 = rebuild_caster_from_mjwarp(caster, mjw_model_v2)
    assert caster._collision is collision_v2
    caster.update_pose(body_pos, body_quat)
    after = np.array(
        caster.trace(origins, directions, max_distance=10.0).distance, copy=True
    )
    # The scaled tet mesh moves the hit distances of the rays aimed at it.
    mesh_rays = (origins[:, 0] > 2.0) & (np.abs(origins[:, 1]) < 1.5)
    assert np.any(~np.isclose(before[:, mesh_rays], after[:, mesh_rays], atol=ATOL))
    caster.close()


def test_unknown_geom_code_fails_closed() -> None:
    _, mjw_model = _models(MINI_XML)
    mjw_model.geom_type.fill_(42)  # not a real mjtGeom code
    with pytest.raises(UnsupportedCapabilityError, match="geom type codes"):
        build_collision_description(mjw_model)


def test_non_mjwarp_model_input_rejected() -> None:
    with pytest.raises(TypeError, match="mujoco_warp.Model"):
        build_collision_description(object())
    mj_model, _ = _models(MINI_XML)
    with pytest.raises(TypeError, match="mujoco_warp.Model"):
        build_collision_description(mj_model)


def test_device_output_and_dynamic_geometry_fail_closed() -> None:
    _, mjw_model = _models(MINI_XML)
    collision = build_collision_description(mjw_model)

    # Device output is not a contract capability: declared unsupported, and
    # requesting it at creation fails closed on the unexpected kwarg.
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=1, collision=collision)
    assert not caster.get_ray_capabilities().supports_device_output
    caster.close()
    with pytest.raises(TypeError, match="device_output"):
        uni_ray.create_ray_caster(
            num_envs=1, num_rays=1, collision=collision, device_output=True
        )

    # Dynamic geometry outside the explicit cold-path rebuild has no API:
    # re-materializing a bound caster (geometry change without rebuild) fails.
    bound = create_ray_caster(mjw_model, num_envs=1, num_rays=1)
    with pytest.raises(BackendError, match="materialized"):
        bound.materialize(collision.scene)
    bound.close()
