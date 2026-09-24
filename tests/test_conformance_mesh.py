"""Static-mesh conformance sweep: Warp wp.Mesh vs MuJoCo CPU mj_ray (#299).

The scene carries a triangulated UV-sphere mesh and a tetrahedron mesh, both
static. Distances are compared against ``mujoco.mj_ray`` with the fixed
documented tolerance ``ATOL = 1e-4`` (float32 device math vs float64 CPU).
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco is required for the mesh sweep")
wp = pytest.importorskip("warp", reason="warp-lang is required for the mesh sweep")

from unisim.ray_query import RayTraceOutputs  # noqa: E402

import uni_ray  # noqa: E402
from uni_ray.mjbatch import build_collision_description  # noqa: E402

ATOL = 1e-4


def _uv_sphere_mesh_xml(name: str, radius: float, nlat: int = 6, nlon: int = 12) -> str:
    vertices = [(0.0, 0.0, radius)]
    for i in range(1, nlat):
        phi = np.pi * i / nlat
        for j in range(nlon):
            theta = 2.0 * np.pi * j / nlon
            vertices.append(
                (
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi),
                )
            )
    vertices.append((0.0, 0.0, -radius))
    bottom = len(vertices) - 1
    faces = []
    for j in range(nlon):
        faces.append((0, 1 + j, 1 + (j + 1) % nlon))
    for i in range(nlat - 2):
        ring0 = 1 + i * nlon
        ring1 = ring0 + nlon
        for j in range(nlon):
            a, b = ring0 + j, ring0 + (j + 1) % nlon
            c, d = ring1 + j, ring1 + (j + 1) % nlon
            faces.append((a, c, b))
            faces.append((b, c, d))
    last_ring = 1 + (nlat - 2) * nlon
    for j in range(nlon):
        faces.append((last_ring + j, bottom, last_ring + (j + 1) % nlon))
    vertex_attr = " ".join(f"{x} {y} {z}" for x, y, z in vertices)
    face_attr = " ".join(f"{a} {b} {c}" for a, b, c in faces)
    return f'<mesh name="{name}" vertex="{vertex_attr}" face="{face_attr}"/>'


def _mj_ray_all(
    model, data, origins: np.ndarray, directions: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full(origins.shape[0], -1.0)
    geom_ids = np.full(origins.shape[0], -1, dtype=np.int32)
    for index in range(origins.shape[0]):
        geom_id = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            model, data, origins[index], directions[index], None, 1, -1, geom_id
        )
        if 0.0 <= distance <= max_distance:
            distances[index] = distance
            geom_ids[index] = geom_id[0]
    return distances, geom_ids


def test_static_mesh_scene_matches_mj_ray() -> None:
    xml = f"""
    <mujoco>
      <asset>
        {_uv_sphere_mesh_xml("ball", 0.5)}
        <mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="ball" pos="3 0 0.5"/>
        <geom type="mesh" mesh="tet" pos="-3 0 0" quat="0.9238795 0 0.3826834 0"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    collision = build_collision_description(model)
    assert len(collision.meshes) == 2

    rng = np.random.default_rng(31)
    num_rays = 1024
    halves = num_rays // 2
    origins_list = []
    directions_list = []
    for center in (np.array([3.0, 0.0, 0.5]), np.array([-2.5, 0.5, 0.5])):
        count = halves
        on_sphere = rng.normal(size=(count, 3))
        on_sphere /= np.linalg.norm(on_sphere, axis=1, keepdims=True)
        shell_origins = center + on_sphere * rng.uniform(1.2, 2.5, size=(count, 1))
        targets = center + rng.uniform(-0.4, 0.4, size=(count, 3))
        dirs = targets - shell_origins
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        origins_list.append(shell_origins)
        directions_list.append(dirs)
    origins = np.concatenate(origins_list)
    directions = np.concatenate(directions_list)

    caster = uni_ray.create_ray_caster(num_envs=1, num_rays=num_rays, collision=collision)
    caster.materialize(collision.scene)
    try:
        caster.update_pose(data.xpos[None], data.xquat[None])
        cpu_distances, cpu_geom_ids = _mj_ray_all(model, data, origins, directions, 20.0)
        result = caster.trace(
            origins, directions, max_distance=20.0, outputs=RayTraceOutputs(geom_id=True)
        )
        cpu_hit = cpu_distances >= 0.0
        np.testing.assert_array_equal(result.hit[0], cpu_hit)
        assert np.all(result.distance[0][~cpu_hit] == np.float32(20.0))
        assert np.all(result.geom_id[0][~cpu_hit] == -1)
        np.testing.assert_allclose(
            result.distance[0][cpu_hit], cpu_distances[cpu_hit], atol=ATOL
        )
        np.testing.assert_array_equal(result.geom_id[0][cpu_hit], cpu_geom_ids[cpu_hit])
        # Both meshes are exercised by the sweep.
        assert set(np.unique(result.geom_id[0][cpu_hit]).tolist()) == {0, 1}
    finally:
        caster.close()
