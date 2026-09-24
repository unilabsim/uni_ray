"""Explicit rebuild path tests (#2): cold-path scene replacement without
recreating the caster.

``WarpRayCaster.rebuild`` re-binds geom sizes/local poses, mesh data, and the
BVH from an updated descriptor; the hot path (``update_pose``/``trace``) keeps
its body-pose-only contract and the fixed ``(num_envs, num_rays)`` batch
shape. Results before and after a rebuild are checked against
``mujoco.mj_ray`` on the corresponding model.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from unisim.errors import BackendError
from unisim.ray_query import RayGeomType, RayTraceOutputs

mujoco = pytest.importorskip("mujoco", reason="mujoco is required for the rebuild tests")
wp = pytest.importorskip("warp", reason="warp-lang is required to create the caster")

import uni_ray  # noqa: E402
from uni_ray.mjbatch import (  # noqa: E402
    build_collision_description,
    rebuild_caster_from_model,
)

ATOL = 1e-4
NUM_ENVS = 2

BASE_XML = """
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
    </body>
  </worldbody>
</mujoco>
"""

# Same scene with a scaled tet mesh and an extra freejoint body + capsule, so
# num_bodies/num_geoms change across the rebuild.
V2_XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  2 0 0  0 2 0  0 0 2" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
  </asset>
  <worldbody>
    <geom type="plane" size="0 0 0.1"/>
    <geom type="mesh" mesh="tet" pos="3 0 0"/>
    <body pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.3" pos="0.4 0 0"/>
    </body>
    <body pos="-2 0 0.8">
      <freejoint/>
      <geom type="capsule" size="0.1 0.3"/>
    </body>
  </worldbody>
</mujoco>
"""

# Aimed at the sphere (pose-dependent), the tet mesh, the far plane, the sky.
RAY_ORIGINS = np.array(
    [
        [0.7, -0.2, 4.0],
        [3.5, 0.4, 3.0],
        [8.0, 8.0, 4.0],
        [0.0, 0.0, 4.0],
        [-2.0, 0.0, 4.0],
    ]
)
RAY_DIRECTIONS = np.tile(np.array([[0.0, 0.0, -1.0]]), (5, 1))
RAY_DIRECTIONS[3] = [0.0, 0.0, 1.0]

POSE_A = np.array([0.3, -0.2, 1.5, 1.0, 0.0, 0.0, 0.0])
POSE_B = np.array([-2.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])

SPHERE_GEOM = 2
SPHERE_RAY = 0


def _make_caster(xml: str) -> tuple:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    collision = build_collision_description(model)
    caster = uni_ray.create_ray_caster(
        num_envs=NUM_ENVS, num_rays=RAY_ORIGINS.shape[0], collision=collision
    )
    caster.materialize(collision.scene)
    return model, data, collision, caster


def _reference(model, data) -> tuple[np.ndarray, np.ndarray]:
    """Per-ray mj_ray distances/geom ids clipped to the max_distance window."""
    distances = np.full(RAY_ORIGINS.shape[0], -1.0)
    geom_ids = np.full(RAY_ORIGINS.shape[0], -1, dtype=np.int32)
    for index in range(RAY_ORIGINS.shape[0]):
        geom_id = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            model, data, RAY_ORIGINS[index], RAY_DIRECTIONS[index], None, 1, -1, geom_id
        )
        if 0.0 <= distance < 10.0:
            distances[index] = distance
            geom_ids[index] = geom_id[0]
    return distances, geom_ids


def _pose_and_reference(model, data, caster, qposes: list[np.ndarray]) -> tuple:
    """Apply per-env poses (body rows from mj_forward) and return mj_ray refs."""
    body_pos, body_quat = [], []
    references = []
    for qpos in qposes:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        body_pos.append(np.array(data.xpos, copy=True))
        body_quat.append(np.array(data.xquat, copy=True))
        references.append(_reference(model, data))
    caster.update_pose(np.stack(body_pos), np.stack(body_quat))
    return references


def _assert_matches(caster, references) -> np.ndarray:
    result = caster.trace(
        RAY_ORIGINS, RAY_DIRECTIONS, max_distance=10.0, outputs=RayTraceOutputs(geom_id=True)
    )
    distances = np.array(result.distance, copy=True)
    assert distances.shape == (NUM_ENVS, RAY_ORIGINS.shape[0])
    for env_id, (ref_distances, ref_geom_ids) in enumerate(references):
        ref_hit = ref_geom_ids >= 0
        np.testing.assert_array_equal(np.array(result.hit)[env_id], ref_hit)
        np.testing.assert_array_equal(
            np.array(result.geom_id)[env_id][ref_hit], ref_geom_ids[ref_hit]
        )
        np.testing.assert_allclose(distances[env_id][ref_hit], ref_distances[ref_hit], atol=ATOL)
    return distances


def test_rebuild_tracks_geom_size_and_local_pose_change() -> None:
    model, data, _, caster = _make_caster(BASE_XML)
    references = _pose_and_reference(model, data, caster, [POSE_A] * NUM_ENVS)
    before = _assert_matches(caster, references)

    # Runtime geometry randomization: grow the sphere and move it in the
    # body's local frame, then rebuild from the mutated model.
    model.geom_size[SPHERE_GEOM] = [0.55, 0.0, 0.0]
    model.geom_pos[SPHERE_GEOM] = [-0.3, 0.3, 0.0]
    new_collision = rebuild_caster_from_model(caster, model)
    assert caster._collision is new_collision

    references = _pose_and_reference(model, data, caster, [POSE_A] * NUM_ENVS)
    after = _assert_matches(caster, references)
    assert not np.isclose(before[0, SPHERE_RAY], after[0, SPHERE_RAY], atol=ATOL)
    caster.close()


def test_rebuild_replaces_mesh_and_allows_scene_shape_change() -> None:
    model, data, _, caster = _make_caster(BASE_XML)
    _pose_and_reference(model, data, caster, [POSE_A] * NUM_ENVS)

    model_v2 = mujoco.MjModel.from_xml_string(V2_XML)
    data_v2 = mujoco.MjData(model_v2)
    collision_v2 = build_collision_description(model_v2)
    caster.rebuild(collision_v2)
    assert caster._scene is collision_v2.scene
    # Batch shape is untouched; body count follows the new descriptor.
    assert caster.num_envs == NUM_ENVS
    assert caster._num_bodies == model_v2.nbody

    references = _pose_and_reference(
        model_v2, data_v2, caster, [POSE_A.tolist() + POSE_B.tolist()] * NUM_ENVS
    )
    _assert_matches(caster, references)
    caster.close()


def test_rebuild_scene_only_reuses_mesh_data() -> None:
    model, data, collision, caster = _make_caster(BASE_XML)
    _pose_and_reference(model, data, caster, [POSE_A] * NUM_ENVS)

    new_sizes = np.array(collision.scene.geom_sizes)
    new_sizes[SPHERE_GEOM] = [0.5, 0.0, 0.0]
    new_scene = dataclasses.replace(collision.scene, geom_sizes=new_sizes)
    caster.rebuild(scene=new_scene)
    # The bound descriptor (and its mesh data) is reused, not replaced.
    assert caster._collision is collision

    model.geom_size[SPHERE_GEOM] = [0.5, 0.0, 0.0]
    references = _pose_and_reference(model, data, caster, [POSE_A] * NUM_ENVS)
    _assert_matches(caster, references)
    caster.close()


def test_rebuild_resets_body_poses_to_identity() -> None:
    model, data, _, caster = _make_caster(BASE_XML)
    _pose_and_reference(model, data, caster, [POSE_A] * NUM_ENVS)
    # Rebuilding the unchanged descriptor drops the pose write: a trace
    # before the next update_pose must see the identity-pose scene.
    rebuild_caster_from_model(caster, model)
    data.qpos[:] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    _assert_matches(caster, [_reference(model, data)] * NUM_ENVS)
    caster.close()


def test_rebuild_validation() -> None:
    model, _, collision, caster = _make_caster(BASE_XML)

    with pytest.raises(ValueError, match="exactly one"):
        caster.rebuild()
    with pytest.raises(ValueError, match="exactly one"):
        caster.rebuild(collision, scene=collision.scene)
    with pytest.raises(TypeError, match="MjBatchCollision"):
        caster.rebuild(object())

    # scene= with a geom count that no longer matches the bound collision.
    shrunk = dataclasses.replace(
        collision.scene,
        geom_types=(RayGeomType.PLANE,),
        geom_sizes=collision.scene.geom_sizes[:1],
        geom_local_pos=collision.scene.geom_local_pos[:1],
        geom_local_quat=collision.scene.geom_local_quat[:1],
        geom_body_ids=collision.scene.geom_body_ids[:1],
    )
    with pytest.raises(ValueError, match="does not match"):
        caster.rebuild(scene=shrunk)

    fresh = uni_ray.create_ray_caster(num_envs=1, num_rays=1, collision=collision)
    with pytest.raises(BackendError, match="materialized"):
        fresh.rebuild(collision)
    fresh.close()

    caster.close()
    with pytest.raises(BackendError, match="closed"):
        caster.rebuild(collision)
    with pytest.raises(BackendError, match="closed"):
        rebuild_caster_from_model(caster, model)


def test_rebuild_caster_from_model_rejects_wrong_caster_type() -> None:
    model = mujoco.MjModel.from_xml_string(BASE_XML)
    with pytest.raises(TypeError, match="WarpRayCaster"):
        rebuild_caster_from_model(object(), model)
