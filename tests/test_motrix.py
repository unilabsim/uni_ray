"""MotrixSim-profile adapter tests (#301).

The MotrixSim adapter (``uni_ray.motrix``) sources the collision descriptor
from a ``motrixsim.SceneModel`` and feeds the shared profile-neutral
descriptor build, so the caster, pose-sync contract, and kernels are
identical to the other profiles. These tests pin: descriptor extraction
(including the MotrixSim xyzw -> wxyz quat convention and the link -> body
numbering), the minimal trace through the UniSim plugin contract, hfield
triangulation against the SDK's compiled ``height_matrix``, the consumer
link-pose -> update_pose mapping, and the fail-closed surface (mesh geoms,
finite planes/SDF, device output, dynamic geometry).

The whole module skips when motrixsim-core is absent (it ships cp310-only
wheels; the default CI env stays green).
"""

from __future__ import annotations

import numpy as np
import pytest
from unisim.errors import BackendError, UnsupportedCapabilityError
from unisim.ray_query import RayCaster, RayGeomType, RayTraceResult

ms = pytest.importorskip("motrixsim", reason="motrixsim-core is required for the motrix profile")
wp = pytest.importorskip("warp", reason="warp-lang is required to create the caster")

import uni_ray  # noqa: E402
from uni_ray.motrix import (  # noqa: E402
    build_collision_description,
    create_ray_caster,
    rebuild_caster_from_motrix,
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

# Non-symmetric hfield (orientation-sensitive), every primitive type, a
# rotated geom (quat-convention check), and a nested body (link numbering).
SCENE_XML = """
<mujoco>
  <asset>
    <hfield name="hf" nrow="4" ncol="4" size="1 1 0.5 0.1"
            elevation="0.0 0.1 0.2 0.3
                       0.1 0.4 0.3 0.1
                       0.2 0.3 0.6 0.2
                       0.0 0.1 0.2 0.4"/>
  </asset>
  <worldbody>
    <geom type="plane" size="0 0 0.1"/>
    <geom type="hfield" hfield="hf" pos="0 5 0"/>
    <body name="b1" pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.3" pos="0.4 0 0"/>
      <geom type="box" size="0.1 0.2 0.15"/>
      <geom type="capsule" size="0.05 0.2" pos="0 0.5 0" quat="0.9238795 0.3826834 0 0"/>
      <geom type="cylinder" size="0.06 0.25" pos="0 -0.5 0"/>
      <geom type="ellipsoid" size="0.1 0.2 0.07" pos="0 0 0.5"/>
      <body name="b2" pos="0.5 0 0">
        <geom type="sphere" size="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

MESH_XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
  </asset>
  <worldbody>
    <geom type="plane" size="0 0 0.1"/>
    <geom type="mesh" mesh="tet" pos="3 0 0"/>
  </worldbody>
</mujoco>
"""


def _down_rays() -> tuple[np.ndarray, np.ndarray]:
    origins = np.array([[0.0, 0.0, 2.0], [4.0, 4.0, 2.0], [0.0, 0.0, 2.0]])
    directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    return origins, directions


def test_descriptor_extraction() -> None:
    model = ms.load_mjcf_str(SCENE_XML)
    collision = build_collision_description(model)
    scene = collision.scene

    assert scene.num_bodies == model.num_links + 1 == 3
    assert scene.geom_types == (
        RayGeomType.PLANE,
        RayGeomType.MESH,  # hfield, triangulated
        RayGeomType.SPHERE,
        RayGeomType.BOX,
        RayGeomType.CAPSULE,
        RayGeomType.CYLINDER,
        RayGeomType.ELLIPSOID,
        RayGeomType.SPHERE,
    )
    # World geoms bind to body 0; link i binds to body i + 1.
    np.testing.assert_array_equal(scene.geom_body_ids, [0, 0, 1, 1, 1, 1, 1, 2])
    # MuJoCo-style size conventions are carried through (capsule keeps the
    # (radius, half_length, 0) layout); the plane size is zeroed.
    np.testing.assert_array_equal(scene.geom_sizes[0], np.zeros(3))
    np.testing.assert_allclose(scene.geom_sizes[2], [0.3, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(scene.geom_sizes[3], [0.1, 0.2, 0.15], atol=1e-7)
    np.testing.assert_allclose(scene.geom_sizes[4], [0.05, 0.2, 0.0], atol=1e-7)
    np.testing.assert_allclose(scene.geom_sizes[6], [0.1, 0.2, 0.07], atol=1e-7)
    # MotrixSim local_pose quats are xyzw; the descriptor is wxyz (the XML
    # quat is already wxyz, so the capsule row must round-trip).
    np.testing.assert_allclose(
        scene.geom_local_quat[4], [0.9238795, 0.3826834, 0.0, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(scene.geom_local_pos[1], [0.0, 5.0, 0.0], atol=1e-7)

    # The hfield is the only mesh, triangulated from the compiled heights.
    assert len(collision.meshes) == 1
    np.testing.assert_array_equal(collision.geom_mesh_ids, [-1, 0, -1, -1, -1, -1, -1, -1])
    hfield = model.geoms[1].hfield
    heights = np.asarray(hfield.height_matrix)
    mesh = collision.meshes[0]
    grid = mesh.points[:16]
    xs = np.linspace(-1.0, 1.0, 4)
    ys = np.linspace(-1.0, 1.0, 4)
    np.testing.assert_allclose(grid[:, 0], np.tile(xs, 4), atol=1e-6)
    np.testing.assert_allclose(grid[:, 1], np.repeat(ys, 4), atol=1e-6)
    np.testing.assert_allclose(grid[:, 2], heights.reshape(-1), atol=1e-6)


def test_minimal_trace_through_plugin_contract() -> None:
    model = ms.load_mjcf_str(MINI_XML)
    collision = build_collision_description(model)

    from unisim.factory import create_ray_caster as factory_create

    caster = factory_create("uni_ray", num_envs=1, num_rays=3, collision=collision)
    assert isinstance(caster, RayCaster)  # no motrixsim types leak out
    caster.materialize(collision.scene)
    origins, directions = _down_rays()
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


def test_hfield_trace_matches_compiled_heights() -> None:
    model = ms.load_mjcf_str(SCENE_XML)
    collision = build_collision_description(model)
    hfield = model.geoms[1].hfield
    heights = np.asarray(hfield.height_matrix)
    nrow, ncol = heights.shape
    xs = np.linspace(-1.0, 1.0, ncol)
    ys = np.linspace(-1.0, 1.0, nrow)
    # Grid vertices and cell-edge midpoints are diagonal-independent.
    points = [(x, y, heights[r, c]) for r, y in enumerate(ys) for c, x in enumerate(xs)]
    points += [
        (0.5 * (xs[c] + xs[c + 1]), y, 0.5 * (heights[r, c] + heights[r, c + 1]))
        for r, y in enumerate(ys)
        for c in range(ncol - 1)
    ]
    origins = np.array([[x, 5.0 + y, 2.0] for x, y, _ in points])
    directions = np.tile([[0.0, 0.0, -1.0]], (len(points), 1))
    expected = np.array([2.0 - h for _, _, h in points])

    caster = uni_ray.create_ray_caster(
        num_envs=1, num_rays=len(points), collision=collision
    )
    caster.materialize(collision.scene)
    result = caster.trace(origins, directions, max_distance=10.0)
    assert np.array(result.hit)[0].all()
    np.testing.assert_allclose(np.array(result.distance)[0], expected, atol=ATOL)
    caster.close()


def test_pose_sync_from_link_poses() -> None:
    """The documented consumer mapping: get_link_poses (xyz+xyzw) -> world
    row + wxyz -> update_pose."""
    model = ms.load_mjcf_str(MINI_XML)
    data = ms.SceneData(model)
    caster = create_ray_caster(model, num_envs=1, num_rays=3)
    origins, directions = _down_rays()

    link_poses = np.asarray(model.get_link_poses(data), dtype=np.float64)
    assert link_poses.shape == (model.num_links, 7)
    body_pos = np.concatenate([np.zeros((1, 3)), link_poses[:, :3]])[None]
    link_quat_wxyz = link_poses[:, 3:7][:, [3, 0, 1, 2]]
    body_quat = np.concatenate([[[1.0, 0.0, 0.0, 0.0]], link_quat_wxyz])[None]
    caster.update_pose(body_pos, body_quat)
    result = caster.trace(origins, directions, max_distance=10.0)
    # MotrixSim does not seed the free joint from the XML body pos, so the
    # link pose is the origin and the sphere spans z -0.5..0.5.
    np.testing.assert_allclose(result.distance[0], [1.5, 2.0, 10.0], atol=1e-5)
    caster.close()


def test_mesh_geom_fails_closed() -> None:
    model = ms.load_mjcf_str(MESH_XML)
    with pytest.raises(UnsupportedCapabilityError, match="vertex/face"):
        build_collision_description(model)


def test_non_scene_model_input_rejected() -> None:
    with pytest.raises(TypeError, match="motrixsim.SceneModel"):
        build_collision_description(object())


def test_adapter_create_ray_caster_lifecycle() -> None:
    model = ms.load_mjcf_str(MINI_XML)
    caster = create_ray_caster(model, num_envs=1, num_rays=3)
    assert isinstance(caster, RayCaster)

    # Bound already (materialized by the convenience): a second materialize
    # fails closed, per the contract.
    with pytest.raises(BackendError, match="materialized"):
        caster.materialize(build_collision_description(model).scene)

    origins, directions = _down_rays()
    result = caster.trace(origins, directions, max_distance=10.0)
    # Identity body poses on a fresh caster: the sphere sits at the origin.
    np.testing.assert_allclose(result.distance[0], [1.5, 2.0, 10.0], atol=1e-5)
    caster.close()
    with pytest.raises(BackendError, match="closed"):
        caster.trace(origins, directions, max_distance=10.0)


def test_rebuild_caster_from_motrix() -> None:
    model = ms.load_mjcf_str(MINI_XML)
    caster = create_ray_caster(model, num_envs=1, num_rays=3)
    origins, directions = _down_rays()
    before = np.array(caster.trace(origins, directions, max_distance=10.0).distance, copy=True)

    grown = ms.load_mjcf_str(MINI_XML.replace('size="0.5"', 'size="0.9"'))
    collision = rebuild_caster_from_motrix(caster, grown)
    assert caster._collision is collision
    after = np.array(caster.trace(origins, directions, max_distance=10.0).distance, copy=True)
    # Ray 0 now hits the larger sphere (radius 0.9 -> top at z=0.9).
    np.testing.assert_allclose(before[0], [1.5, 2.0, 10.0], atol=1e-5)
    np.testing.assert_allclose(after[0], [1.1, 2.0, 10.0], atol=1e-5)
    caster.close()


def test_device_output_and_dynamic_geometry_fail_closed() -> None:
    model = ms.load_mjcf_str(MINI_XML)
    collision = build_collision_description(model)

    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=1, collision=collision)
    assert not caster.get_ray_capabilities().supports_device_output
    caster.close()
    with pytest.raises(TypeError, match="device_output"):
        uni_ray.create_ray_caster(
            num_envs=1, num_rays=1, collision=collision, device_output=True
        )

    # Dynamic geometry outside the explicit cold-path rebuild has no API:
    # re-materializing a bound caster (geometry change without rebuild) fails.
    bound = create_ray_caster(model, num_envs=1, num_rays=1)
    with pytest.raises(BackendError, match="materialized"):
        bound.materialize(collision.scene)
    bound.close()
