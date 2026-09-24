"""Contract lifecycle/semantics edge cases (#299).

Complements tests/test_lifecycle.py with the miss convention details, batch
shape validation, empty selections, zero-geom scenes, and post-close
behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
from unisim.errors import BackendError, UnsupportedCapabilityError
from unisim.ray_query import RaySceneDescription, RayTraceOutputs

wp = pytest.importorskip("warp", reason="warp-lang is required to create the caster")

import uni_ray  # noqa: E402


@pytest.fixture()
def scene() -> RaySceneDescription:
    return RaySceneDescription(
        num_bodies=1,
        geom_types=("sphere",),
        geom_sizes=np.array([[0.5, 0.0, 0.0]]),
        geom_local_pos=np.array([[0.0, 0.0, 1.0]]),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.array([0]),
    )


@pytest.fixture()
def caster(scene: RaySceneDescription):
    caster = uni_ray.create_ray_caster(num_envs=3, num_rays=2)
    caster.materialize(scene)
    yield caster
    caster.close()


def test_miss_convention_is_exact(caster) -> None:
    # Rays pointing up miss: distance is exactly max_distance, hit False, ids -1.
    origins = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    result = caster.trace(
        origins,
        directions,
        max_distance=3.5,
        outputs=RayTraceOutputs(geom_id=True, body_id=True, hit_point=True),
    )
    assert result.distance.shape == (3, 2)
    assert not result.hit.any()
    assert np.all(result.distance == np.float32(3.5))
    assert np.all(result.geom_id == -1)
    assert np.all(result.body_id == -1)
    # hit_point is the clipped ray endpoint even on misses (only meaningful on
    # hits per the contract, but must stay finite).
    assert np.isfinite(result.hit_point).all()


def test_hit_exactly_at_max_distance_is_a_miss(caster) -> None:
    # The kernel accepts t < max_distance strictly; a surface exactly at the
    # cutoff reads as a miss (pinned boundary convention).
    result = caster.trace(
        np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]),
        np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        max_distance=0.5,
    )
    assert not result.hit[0, 0]
    assert result.distance[0, 0] == np.float32(0.5)


def test_batch_shape_validation(caster) -> None:
    origins = np.zeros((2, 3))
    directions = np.tile(np.array([[0.0, 0.0, -1.0]]), (2, 1))
    with pytest.raises(ValueError, match="does not match"):
        caster.trace(origins, directions[:1], max_distance=1.0)
    with pytest.raises(ValueError, match="ray batches"):
        caster.trace(np.zeros((7, 3)), np.zeros((7, 3)), max_distance=1.0)
    with pytest.raises(ValueError, match="body_pos"):
        caster.update_pose(np.zeros((3, 5, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (3, 1, 1)))
    with pytest.raises(ValueError, match="body_quat"):
        caster.update_pose(np.zeros((3, 1, 3)), np.zeros((3, 1, 5)))
    with pytest.raises(ValueError, match="finite"):
        caster.update_pose(np.full((3, 1, 3), np.nan), np.tile([1.0, 0.0, 0.0, 0.0], (3, 1, 1)))
    # num_envs rows mismatching the env_ids selection length is rejected.
    with pytest.raises(ValueError, match="body_pos"):
        caster.update_pose(
            np.zeros((3, 1, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (3, 1, 1)), env_ids=[1]
        )
    with pytest.raises(TypeError, match="RayTraceOutputs"):
        caster.trace(origins, directions, max_distance=1.0, outputs="hit_point")


def test_empty_selection_returns_empty_results(caster) -> None:
    origins = np.zeros((0, 2, 3))
    directions = np.zeros((0, 2, 3))
    result = caster.trace(origins, directions, max_distance=1.0, env_ids=[])
    assert result.distance.shape == (0, 2)
    assert result.hit.shape == (0, 2)


def test_zero_geom_scene_traces_as_all_miss() -> None:
    empty_scene = RaySceneDescription(
        num_bodies=1,
        geom_types=(),
        geom_sizes=np.zeros((0, 3)),
        geom_local_pos=np.zeros((0, 3)),
        geom_local_quat=np.zeros((0, 4)),
        geom_body_ids=np.zeros((0,), dtype=np.intp),
    )
    caster = uni_ray.create_ray_caster(num_envs=2, num_rays=4)
    caster.materialize(empty_scene)
    try:
        caster.update_pose(np.zeros((2, 1, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (2, 1, 1)))
        result = caster.trace(
            np.zeros((4, 3)), np.tile(np.array([[0.0, 0.0, -1.0]]), (4, 1)), max_distance=2.0
        )
        assert result.distance.shape == (2, 4)
        assert not result.hit.any()
        assert np.all(result.distance == np.float32(2.0))
    finally:
        caster.close()


def test_materialize_rejects_wrong_types_and_sizes() -> None:
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=1)
    with pytest.raises(TypeError, match="RaySceneDescription"):
        caster.materialize("not a scene")
    zero_sized = RaySceneDescription(
        num_bodies=1,
        geom_types=("sphere",),
        geom_sizes=np.array([[0.0, 0.0, 0.0]]),
        geom_local_pos=np.zeros((1, 3)),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.array([0]),
    )
    with pytest.raises(ValueError, match="positive sizes"):
        caster.materialize(zero_sized)
    caster.close()


def test_collision_descriptor_mismatch_fails(caster, scene: RaySceneDescription) -> None:
    from uni_ray.mjbatch import MjBatchCollision

    other_scene = RaySceneDescription(
        num_bodies=1,
        geom_types=("sphere", "sphere"),
        geom_sizes=np.tile([0.5, 0.0, 0.0], (2, 1)),
        geom_local_pos=np.zeros((2, 3)),
        geom_local_quat=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        geom_body_ids=np.zeros(2, dtype=np.intp),
    )
    mismatch = MjBatchCollision(
        scene=other_scene, meshes=(), geom_mesh_ids=np.full(2, -1, dtype=np.intp)
    )
    fresh = uni_ray.create_ray_caster(num_envs=1, num_rays=1, collision=mismatch)
    with pytest.raises(ValueError, match="does not match"):
        fresh.materialize(scene)
    fresh.close()


def test_close_releases_and_blocks_everything(caster, scene: RaySceneDescription) -> None:
    caster.close()
    caster.close()  # idempotent
    with pytest.raises(BackendError, match="closed"):
        caster.trace(np.zeros((2, 3)), np.zeros((2, 3)), max_distance=1.0)
    with pytest.raises(BackendError, match="closed"):
        caster.update_pose(np.zeros((3, 1, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (3, 1, 1)))
    with pytest.raises(BackendError, match="closed"):
        caster.materialize(scene)


def test_mesh_geom_with_collision_but_no_meshes_fails_closed() -> None:
    mesh_scene = RaySceneDescription(
        num_bodies=1,
        geom_types=("mesh",),
        geom_sizes=np.zeros((1, 3)),
        geom_local_pos=np.zeros((1, 3)),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.array([0]),
    )
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=1)
    with pytest.raises(UnsupportedCapabilityError):
        caster.materialize(mesh_scene)
    caster.close()
