"""Lifecycle smoke tests for the uni_ray plugin skeleton (#297).

These use a fake in-memory descriptor and pose inputs; no MuJoCo model is
involved. Warp itself is required to create the caster and is skipped
appropriately.
"""

from __future__ import annotations

import numpy as np
import pytest
from unisim.errors import BackendError, UnsupportedCapabilityError
from unisim.ray_query import (
    RayCaster,
    RaySceneDescription,
    RayTraceOutputs,
    RayTraceResult,
)

wp = pytest.importorskip("warp", reason="warp-lang is required to create the caster")

import uni_ray  # noqa: E402


@pytest.fixture()
def scene() -> RaySceneDescription:
    return RaySceneDescription(
        num_bodies=2,
        geom_types=("plane", "sphere"),
        geom_sizes=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        geom_local_pos=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.array([0, 1]),
    )


@pytest.fixture()
def caster(scene: RaySceneDescription):
    caster = uni_ray.create_ray_caster(num_envs=2, num_rays=3)
    caster.materialize(scene)
    yield caster
    caster.close()


def _down_rays() -> tuple[np.ndarray, np.ndarray]:
    origins = np.array([[0.0, 0.0, 2.0], [4.0, 4.0, 2.0], [0.0, 0.0, 2.0]])
    directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    return origins, directions


def test_factory_entry_point_returns_ray_caster() -> None:
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=1)
    assert isinstance(caster, RayCaster)
    assert caster.caster_type == "uni_ray"
    assert caster.num_envs == 1 and caster.num_rays == 1
    caster.close()


def test_unisim_factory_discovers_uni_ray() -> None:
    from unisim.factory import create_ray_caster

    caster = create_ray_caster("uni_ray", num_envs=1, num_rays=1)
    assert isinstance(caster, RayCaster)
    caster.close()


def test_capabilities_are_declared_honestly() -> None:
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=1)
    capabilities = caster.get_ray_capabilities()
    assert capabilities.supports_pose_sync
    assert capabilities.supports_per_env_rays
    assert capabilities.supports_host_readback
    assert not capabilities.supports_device_output
    caster.close()


def test_lifecycle_smoke(caster, scene: RaySceneDescription) -> None:
    origins, directions = _down_rays()
    result = caster.trace(
        origins,
        directions,
        max_distance=10.0,
        outputs=RayTraceOutputs(hit_point=True, geom_id=True, body_id=True),
    )
    assert isinstance(result, RayTraceResult)
    # Ray 0 hits the sphere at z=1.5 (dist 0.5); ray 1 hits the plane
    # (dist 2.0); ray 2 points up and misses.
    np.testing.assert_allclose(result.distance[0], [0.5, 2.0, 10.0], atol=1e-5)
    assert result.hit[0].tolist() == [True, True, False]
    assert result.geom_id[0].tolist() == [1, 0, -1]
    assert result.body_id[0].tolist() == [1, 0, -1]
    np.testing.assert_allclose(result.hit_point[0, 0], [0.0, 0.0, 1.5], atol=1e-5)

    # Pose sync: lift the sphere's body in env row 1 only; env 0 is unchanged.
    body_pos = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]])
    body_quat = np.array([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    caster.update_pose(body_pos, body_quat, env_ids=[1])
    moved = caster.trace(origins, directions, max_distance=10.0)
    np.testing.assert_allclose(moved.distance[0], [0.5, 2.0, 10.0], atol=1e-5)
    # Env 1: the sphere moved up to z=6, so the down rays now hit the plane
    # (dist 2.0) and the up ray hits the lifted sphere (dist 3.5).
    np.testing.assert_allclose(moved.distance[1], [2.0, 2.0, 3.5], atol=1e-5)

    # Per-environment ray profile and selected rows.
    per_env_origins = np.broadcast_to(origins, (2, 3, 3))
    per_env_directions = np.broadcast_to(directions, (2, 3, 3))
    both = caster.trace(per_env_origins, per_env_directions, max_distance=10.0)
    np.testing.assert_allclose(both.distance[1], [2.0, 2.0, 3.5], atol=1e-5)
    selected = caster.trace(
        per_env_origins[1:], per_env_directions[1:], max_distance=10.0, env_ids=[1]
    )
    assert selected.distance.shape == (1, 3)
    np.testing.assert_allclose(selected.distance[0], [2.0, 2.0, 3.5], atol=1e-5)


def test_unisim_conformance_helper(caster) -> None:
    from unisim.conformance import assert_ray_caster_conformance

    fresh = uni_ray.create_ray_caster(num_envs=2, num_rays=4)
    try:
        assert_ray_caster_conformance(fresh)
    finally:
        fresh.close()


def test_materialize_twice_fails(caster, scene: RaySceneDescription) -> None:
    with pytest.raises(BackendError, match="already materialized"):
        caster.materialize(scene)


def test_trace_before_materialize_fails(scene: RaySceneDescription) -> None:
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=1)
    origins, directions = _down_rays()
    with pytest.raises(BackendError, match="materialized"):
        caster.trace(origins[:1], directions[:1], max_distance=1.0)
    with pytest.raises(BackendError, match="materialized"):
        caster.update_pose(np.zeros((1, 2, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (1, 2, 1)))
    caster.close()


def test_close_is_idempotent_and_later_queries_fail(
    caster, scene: RaySceneDescription
) -> None:
    caster.close()
    caster.close()
    origins, directions = _down_rays()
    with pytest.raises(BackendError, match="closed"):
        caster.trace(origins, directions, max_distance=1.0)
    with pytest.raises(BackendError, match="closed"):
        caster.materialize(scene)


def test_mesh_geom_without_collision_descriptor_fails_closed() -> None:
    mesh_scene = RaySceneDescription(
        num_bodies=1,
        geom_types=("mesh",),
        geom_sizes=np.zeros((1, 3)),
        geom_local_pos=np.zeros((1, 3)),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.array([0]),
    )
    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=1)
    with pytest.raises(UnsupportedCapabilityError, match="collision descriptor"):
        caster.materialize(mesh_scene)
    caster.close()


def test_undeclared_output_request_fails_closed(caster) -> None:
    origins, directions = _down_rays()
    with pytest.raises(UnsupportedCapabilityError):
        caster.trace(origins, directions, max_distance=10.0, outputs=RayTraceOutputs(normal=True))


def test_invalid_batch_arguments_fail_closed(caster) -> None:
    origins, directions = _down_rays()
    with pytest.raises(ValueError, match="max_distance"):
        caster.trace(origins, directions, max_distance=-1.0)
    with pytest.raises(ValueError, match="unit vectors"):
        caster.trace(origins, directions * 2.0, max_distance=10.0)
    with pytest.raises(ValueError, match="environment IDs"):
        caster.trace(origins, directions, max_distance=10.0, env_ids=[5])
    with pytest.raises(ValueError, match="unit wxyz"):
        caster.update_pose(
            np.zeros((2, 2, 3)), np.zeros((2, 2, 4))
        )


def test_result_buffers_are_reused_views(caster) -> None:
    origins, directions = _down_rays()
    first = caster.trace(origins, directions, max_distance=10.0)
    second = caster.trace(origins, directions, max_distance=5.0)
    # distance/hit/geom_id are views into caster-owned buffers reused by the
    # next trace call; callers retaining results across calls must copy.
    assert first.distance is second.distance or np.shares_memory(first.distance, second.distance)
    np.testing.assert_allclose(second.distance[0], [0.5, 2.0, 5.0], atol=1e-5)


def test_create_validates_batch_shape() -> None:
    with pytest.raises(ValueError, match="positive"):
        uni_ray.create_ray_caster(num_envs=0, num_rays=1)
    with pytest.raises(TypeError, match="integer"):
        uni_ray.create_ray_caster(num_envs=1.5, num_rays=1)
