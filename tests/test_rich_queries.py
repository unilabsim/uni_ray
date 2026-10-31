"""Public-descriptor, static-filter, minimum-distance, and normal tests (#472)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from unisim.ray_query import (
    RayCollisionDescription,
    RayMeshDescription,
    RayPoseTensorView,
    RaySceneDescription,
    RayTraceFilter,
    RayTraceOutputs,
)

wp = pytest.importorskip("warp", reason="warp-lang is required to create the caster")
torch = pytest.importorskip("torch", reason="torch is required for the tensor profile")

import uniray  # noqa: E402


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _primitive_collision() -> RayCollisionDescription:
    scene = RaySceneDescription(
        num_bodies=2,
        geom_types=("plane", "sphere"),
        geom_sizes=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        geom_local_pos=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.array([0, 1], dtype=np.intp),
        body_names=("world", "ball"),
    )
    return RayCollisionDescription(
        scene=scene,
        geom_names=("ground", "ball"),
        geom_tags=(("terrain",), ("robot",)),
        body_parent_ids=np.array([-1, -1], dtype=np.intp),
    )


def _mesh_collision() -> RayCollisionDescription:
    points = np.array(
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    scene = RaySceneDescription(
        num_bodies=1,
        geom_types=("mesh",),
        geom_sizes=np.zeros((1, 3)),
        geom_local_pos=np.zeros((1, 3)),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.array([0], dtype=np.intp),
        body_names=("world",),
    )
    return RayCollisionDescription(
        scene=scene,
        geom_names=("double_plane",),
        geom_tags=(("terrain",),),
        meshes=(RayMeshDescription("mesh:double_plane", points, indices),),
        geom_mesh_ids=np.array([0], dtype=np.intp),
    )


def _bind(collision: RayCollisionDescription) -> Any:
    caster = uniray.create_ray_caster(
        num_envs=1,
        num_rays=1,
        collision=collision,
        device=_device(),
    )
    caster.materialize(collision.scene)
    caster.bind_pose_layout(collision.scene.body_names)
    names = collision.scene.body_names
    pos = torch.zeros((1, len(names), 3), dtype=torch.float32, device=_device())
    quat = torch.zeros((1, len(names), 4), dtype=torch.float32, device=_device())
    quat[..., 0] = 1.0
    caster.update_pose_tensor(RayPoseTensorView(names, pos, quat))
    return caster


def _down_ray(device: str) -> tuple[Any, Any]:
    return (
        torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=device),
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device),
    )


def test_public_descriptor_replaces_backend_specific_builder() -> None:
    collision = _primitive_collision()
    caster = _bind(collision)
    try:
        origins, directions = _down_ray(caster._required_tensor_device())
        result = caster.trace_tensor(origins, directions, 10.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        assert result.hit.item()
        assert result.distance.item() == pytest.approx(0.5, abs=1e-6)
    finally:
        caster.close()


def test_static_filter_and_min_distance_select_nearest_valid_hit() -> None:
    collision = _primitive_collision()
    caster = _bind(collision)
    try:
        origins, directions = _down_ray(caster._required_tensor_device())
        plane_only = RayTraceFilter(allowed_tags=("terrain",))
        result = caster.trace_tensor(
            origins,
            directions,
            10.0,
            ray_filter=plane_only,
            min_distance=1.5,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        assert result.hit.item()
        assert result.distance.item() == pytest.approx(2.0, abs=1e-6)

        host = caster.trace(
            origins.detach().cpu().numpy(),
            directions.detach().cpu().numpy(),
            10.0,
            ray_filter=plane_only,
            min_distance=1.5,
        )
        assert host.distance.item() == pytest.approx(float(result.distance.item()), abs=1e-6)

        boundary = caster.trace_tensor(origins, directions, 10.0, min_distance=2.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        assert boundary.hit.item()
        assert boundary.distance.item() == pytest.approx(2.0, abs=1e-6)
    finally:
        caster.close()


def test_optional_tensor_normal_has_host_parity() -> None:
    collision = _primitive_collision()
    caster = _bind(collision)
    try:
        origins, directions = _down_ray(caster._required_tensor_device())
        outputs = RayTraceOutputs(normal=True)
        tensor_result = caster.trace_tensor(origins, directions, 10.0, outputs=outputs)
        host_result = caster.trace(
            origins.detach().cpu().numpy(),
            directions.detach().cpu().numpy(),
            10.0,
            outputs=outputs,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.testing.assert_close(
            tensor_result.normal,
            torch.as_tensor(host_result.normal, dtype=torch.float32).to(origins.device),
            rtol=2e-6,
            atol=2e-6,
        )
        assert tensor_result.normal is not None
        torch.testing.assert_close(
            tensor_result.normal[0], torch.tensor([[0.0, 0.0, 1.0]], device=origins.device)
        )
    finally:
        caster.close()


def test_primitive_side_and_cap_normals_are_correct() -> None:
    cases = (
        (
            "plane",
            np.zeros(3),
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        (
            "box",
            np.array([0.5, 0.4, 0.3]),
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        (
            "cylinder",
            np.array([0.5, 0.5, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ),
        (
            "cylinder",
            np.array([0.5, 0.5, 0.0]),
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        (
            "capsule",
            np.array([0.5, 0.5, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ),
        (
            "capsule",
            np.array([0.5, 0.5, 0.0]),
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        (
            "ellipsoid",
            np.array([0.5, 0.4, 0.3]),
            np.array([0.0, 0.0, 2.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
    )
    for geom_type, size, origin, direction, expected in cases:
        scene = RaySceneDescription(
            num_bodies=1,
            geom_types=(geom_type,),
            geom_sizes=size[None, :],
            geom_local_pos=np.zeros((1, 3)),
            geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
            geom_body_ids=np.array([0], dtype=np.intp),
            body_names=("world",),
        )
        collision = RayCollisionDescription(scene=scene)
        caster = _bind(collision)
        try:
            result = caster.trace(
                origin[None, :],
                direction[None, :],
                10.0,
                outputs=RayTraceOutputs(normal=True),
            )
            assert result.hit.item(), geom_type
            np.testing.assert_allclose(result.normal[0, 0], expected, atol=2e-6, err_msg=geom_type)
        finally:
            caster.close()


def test_mesh_min_distance_returns_next_valid_hit_in_one_traversal() -> None:
    collision = _mesh_collision()
    caster = _bind(collision)
    try:
        origins, directions = _down_ray(caster._required_tensor_device())
        near = caster.trace_tensor(origins, directions, 10.0)
        near_distance = near.distance.clone()
        boundary = caster.trace_tensor(origins, directions, 10.0, min_distance=1.0)
        boundary_distance = boundary.distance.clone()
        far = caster.trace_tensor(origins, directions, 10.0, min_distance=1.1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        assert near.hit.item()
        assert near_distance.item() == pytest.approx(1.0, abs=1e-6)
        assert boundary.hit.item()
        assert boundary_distance.item() == pytest.approx(1.0, abs=1e-6)
        assert far.hit.item()
        assert far.distance.item() == pytest.approx(2.0, abs=1e-6)

        normal = caster.trace_tensor(
            origins,
            directions,
            10.0,
            outputs=RayTraceOutputs(normal=True),
            min_distance=1.1,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        assert normal.normal is not None
        torch.testing.assert_close(
            normal.normal[0], torch.tensor([[0.0, 0.0, 1.0]], device=origins.device)
        )
    finally:
        caster.close()
