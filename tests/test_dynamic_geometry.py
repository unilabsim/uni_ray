"""Dynamic per-row geometry state tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("warp", reason="Warp is required for dynamic geometry tests")
from unisim.ray_query import (
    RayCollisionDescription,
    RayGeometryState,
    RayGeometryTensorView,
    RayMeshDescription,
    RayPoseTensorView,
    RaySceneDescription,
)

import uni_ray


def _scene() -> RaySceneDescription:
    return RaySceneDescription(
        num_bodies=1,
        geom_types=("sphere",),
        geom_sizes=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        geom_local_pos=np.zeros((1, 3), dtype=np.float64),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        geom_body_ids=np.array([0], dtype=np.intp),
        body_names=("body",),
    )


def _geometry(
    *, enabled: np.ndarray | None = None, radius: np.ndarray | None = None
) -> RayGeometryState:
    return RayGeometryState(
        num_envs=2,
        geom_names=("geom_0",),
        geom_sizes=(np.array([[[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]) if radius is None else radius),
        geom_local_pos=np.zeros((2, 1, 3)),
        geom_local_quat=np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1, 4)).copy(),
        geom_enabled=(
            np.ones((2, 1), dtype=np.bool_) if enabled is None else enabled.astype(np.bool_)
        ),
        geom_mesh_ids=np.full((2, 1), -1, dtype=np.int32),
    )


def _prepare(caster):
    scene = _scene()
    caster.materialize(scene)
    caster.bind_pose_layout(scene.body_names)
    caster.update_pose_tensor(
        RayPoseTensorView(
            body_names=scene.body_names,
            pos_w=torch.zeros((2, 1, 3)),
            quat_w=torch.broadcast_to(
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                (2, 1, 4),
            ).contiguous(),
        )
    )


def test_host_geometry_updates_primitive_sizes_and_optional_slots() -> None:
    caster = uni_ray.create_ray_caster(num_envs=2, num_rays=1, device="cpu")
    scene = _scene()
    caster.materialize(scene)
    caster.update_geometry(_geometry())
    result = caster.trace(
        np.array([[[0.0, 0.0, 3.0]]] * 2),
        np.array([[[0.0, 0.0, -1.0]]] * 2),
        10.0,
    )
    np.testing.assert_allclose(result.distance, [[2.5], [2.0]], atol=1e-5)

    caster.update_geometry(_geometry(enabled=np.array([[True], [False]])), env_ids=np.array([1]))
    result = caster.trace(
        np.array([[[0.0, 0.0, 3.0]]] * 2),
        np.array([[[0.0, 0.0, -1.0]]] * 2),
        10.0,
    )
    np.testing.assert_allclose(result.distance, [[2.5], [10.0]], atol=1e-5)
    caster.close()


def test_tensor_geometry_has_host_parity_without_hidden_copy() -> None:
    caster = uni_ray.create_ray_caster(num_envs=2, num_rays=1, device="cpu")
    _prepare(caster)
    caster.update_geometry_tensor(
        RayGeometryTensorView(
            geom_sizes=torch.tensor([[[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
            geom_local_pos=torch.zeros((2, 1, 3)),
            geom_local_quat=torch.broadcast_to(
                torch.tensor([1.0, 0.0, 0.0, 0.0]), (2, 1, 4)
            ).contiguous(),
            geom_enabled=torch.ones((2, 1), dtype=torch.bool),
            geom_mesh_ids=torch.full((2, 1), -1, dtype=torch.int32),
        )
    )
    result = caster.trace_tensor(
        torch.tensor([[[0.0, 0.0, 3.0]]] * 2),
        torch.tensor([[[0.0, 0.0, -1.0]]] * 2),
        10.0,
    )
    torch.testing.assert_close(result.distance, torch.tensor([[2.5], [2.0]]))
    caster.close()


def test_dynamic_geometry_selects_immutable_mesh_assets() -> None:
    points = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    meshes = (
        RayMeshDescription(asset_id="mesh:low", points=points + [0, 0, 1], indices=indices),
        RayMeshDescription(asset_id="mesh:high", points=points + [0, 0, 2], indices=indices),
    )
    scene = RaySceneDescription(
        num_bodies=1,
        geom_types=("mesh",),
        geom_sizes=np.ones((1, 3)),
        geom_local_pos=np.zeros((1, 3)),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.array([0], dtype=np.intp),
        body_names=("body",),
    )
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=("geom_0",),
        meshes=meshes,
        geom_mesh_ids=np.array([0], dtype=np.intp),
    )
    caster = uni_ray.create_ray_caster(num_envs=2, num_rays=1, collision=collision)
    caster.materialize(scene)
    caster.update_geometry(
        RayGeometryState(
            num_envs=2,
            geom_names=("geom_0",),
            geom_sizes=np.ones((2, 1, 3)),
            geom_local_pos=np.zeros((2, 1, 3)),
            geom_local_quat=np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1, 4)).copy(),
            geom_enabled=np.ones((2, 1), dtype=np.bool_),
            geom_mesh_ids=np.array([[0], [1]], dtype=np.int32),
        )
    )
    result = caster.trace(
        np.array([[[0.0, 0.0, 3.0]]] * 2),
        np.array([[[0.0, 0.0, -1.0]]] * 2),
        10.0,
    )
    np.testing.assert_allclose(result.distance, [[2.0], [1.0]], atol=1e-5)
    caster.close()
