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
from unisim.backend.ray_scene import ray_collision_from_mujoco_model
from unisim.ray_query import RayTraceFilter, RayTraceOutputs

torch = pytest.importorskip("torch", reason="torch is required for the tensor hfield path")

from unisim.ray_query import RayPoseTensorView  # noqa: E402

mujoco = pytest.importorskip("mujoco", reason="mujoco is required for the hfield tests")
wp = pytest.importorskip("warp", reason="warp-lang is required to create the caster")

import uniray  # noqa: E402
from uniray.backends.mjbatch import build_collision_description  # noqa: E402

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
    caster.update_pose(np.array(data.xpos, copy=True)[None], np.array(data.xquat, copy=True)[None])


def _packed_pose_tensors(
    body_pos: np.ndarray, body_quat: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the public full-batch views from a packed ``(pos, quat)`` carrier."""
    carrier = torch.empty(
        (body_pos.shape[0], body_pos.shape[1], 7), dtype=torch.float32, device=device
    )
    carrier[..., :3].copy_(torch.from_numpy(np.asarray(body_pos, dtype=np.float32)))
    carrier[..., 3:7].copy_(torch.from_numpy(np.asarray(body_quat, dtype=np.float32)))
    return carrier[..., :3], carrier[..., 3:7]


def test_hfield_lifecycle_and_correctness_vs_mj_ray() -> None:
    model = mujoco.MjModel.from_xml_string(TERRAIN_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    collision = build_collision_description(model)
    caster = uniray.create_ray_caster(num_envs=1, num_rays=NUM_RAYS, collision=collision)
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


def test_public_hfield_descriptor_supports_rich_tensor_query_parity() -> None:
    model = mujoco.MjModel.from_xml_string(TERRAIN_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    collision = ray_collision_from_mujoco_model(model)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    caster = uniray.create_ray_caster(
        num_envs=1, num_rays=NUM_RAYS, collision=collision, device=device
    )
    caster.materialize(collision.scene)
    caster.bind_pose_layout(collision.scene.body_names)
    pos, quat = _packed_pose_tensors(
        np.asarray(data.xpos)[None], np.asarray(data.xquat)[None], device
    )
    caster.update_pose_tensor(RayPoseTensorView(collision.scene.body_names, pos, quat))

    origins, directions = _rays()
    ray_filter = RayTraceFilter(allowed_geom_ids=(0,))
    outputs = RayTraceOutputs(normal=True, geom_id=True)
    tensor_result = caster.trace_tensor(
        torch.tensor(origins, dtype=torch.float32, device=device),
        torch.tensor(directions, dtype=torch.float32, device=device),
        10.0,
        outputs=outputs,
        min_distance=0.05,
        ray_filter=ray_filter,
    )
    host_result = caster.trace(
        origins,
        directions,
        10.0,
        outputs=outputs,
        min_distance=0.05,
        ray_filter=ray_filter,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    torch.testing.assert_close(
        tensor_result.distance,
        torch.as_tensor(host_result.distance, dtype=torch.float32, device=device),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(tensor_result.hit, torch.as_tensor(host_result.hit, device=device))
    torch.testing.assert_close(
        tensor_result.geom_id,
        torch.as_tensor(host_result.geom_id, dtype=torch.int32, device=device),
    )
    assert tensor_result.normal is not None
    torch.testing.assert_close(
        tensor_result.normal,
        torch.as_tensor(host_result.normal, dtype=torch.float32, device=device),
        rtol=2e-5,
        atol=2e-5,
    )
    assert torch.isfinite(tensor_result.normal[tensor_result.hit]).all()
    normal_lengths = torch.linalg.vector_norm(tensor_result.normal, dim=-1)
    torch.testing.assert_close(
        normal_lengths[tensor_result.hit],
        torch.ones_like(normal_lengths[tensor_result.hit]),
        rtol=2e-4,
        atol=2e-5,
    )
    caster.close()


def test_public_hfield_descriptor_returns_native_face_normal() -> None:
    xml = """
    <mujoco>
      <asset>
        <hfield name='terrain' nrow='2' ncol='2' size='1 1 0.5 0.2'
                elevation='0 1  1 0'/>
      </asset>
      <worldbody><geom type='hfield' hfield='terrain'/></worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    collision = ray_collision_from_mujoco_model(model)
    caster = uniray.create_ray_caster(num_envs=1, num_rays=1, collision=collision)
    caster.materialize(collision.scene)
    try:
        origin = np.array([[-0.75, -0.875, 2.0]])
        direction = np.array([[0.0, 0.0, -1.0]])
        result = caster.trace(
            origin,
            direction,
            10.0,
            outputs=RayTraceOutputs(normal=True, geom_id=True),
        )
        expected_normal = np.array([0.25, -0.25, 1.0])
        expected_normal /= np.linalg.norm(expected_normal)
        assert result.hit.item()
        assert result.geom_id.item() == 0
        assert result.distance.item() == pytest.approx(1.53125, abs=1e-5)
        np.testing.assert_allclose(result.normal[0, 0], expected_normal, atol=1e-5)
    finally:
        caster.close()
    with pytest.raises(Exception, match="closed"):
        caster.trace(origin, direction, max_distance=10.0)


def test_hfield_geom_pose_follows_body_pose() -> None:
    model = mujoco.MjModel.from_xml_string(MOVING_TERRAIN_XML)
    data = mujoco.MjData(model)
    collision = build_collision_description(model)
    caster = uniray.create_ray_caster(num_envs=2, num_rays=NUM_RAYS, collision=collision)
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


def test_hfield_tensor_path_follows_body_pose() -> None:
    model = mujoco.MjModel.from_xml_string(MOVING_TERRAIN_XML)
    data = mujoco.MjData(model)
    collision = build_collision_description(model)
    caster = uniray.create_ray_caster(num_envs=2, num_rays=NUM_RAYS, collision=collision)
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

    caster_device = caster._device_obj.alias
    pos_w, quat_w = _packed_pose_tensors(np.stack(body_pos), np.stack(body_quat), caster_device)
    caster.bind_pose_layout(collision.scene.body_names)
    caster.update_pose_tensor(
        RayPoseTensorView(body_names=collision.scene.body_names, pos_w=pos_w, quat_w=quat_w)
    )
    ray_origins = torch.tensor(origins, dtype=torch.float32).to(caster_device)
    ray_directions = torch.tensor(directions, dtype=torch.float32).to(caster_device)
    try:
        result = caster.trace_tensor(
            ray_origins,
            ray_directions,
            max_distance=10.0,
            outputs=RayTraceOutputs(geom_id=True),
        )
        assert result.distance.device == ray_origins.device
        assert result.hit.device == ray_origins.device
        assert result.geom_id is not None
        assert result.geom_id.device == ray_origins.device
        for env_id, (ref_distances, ref_geom_ids) in enumerate(references):
            ref_hit = torch.from_numpy(ref_geom_ids >= 0)
            ref_hit_device = ref_hit.to(caster_device)
            torch.testing.assert_close(result.hit[env_id], ref_hit_device)
            torch.testing.assert_close(
                result.distance[env_id][ref_hit_device],
                torch.from_numpy(ref_distances[ref_hit.numpy()].astype(np.float32)).to(
                    caster_device
                ),
                atol=ATOL,
                rtol=0.0,
            )
            torch.testing.assert_close(
                result.geom_id[env_id][ref_hit_device],
                torch.from_numpy(ref_geom_ids[ref_hit.numpy()]).to(caster_device),
            )
            assert bool((~ref_hit_device).any())
    finally:
        caster.close()
