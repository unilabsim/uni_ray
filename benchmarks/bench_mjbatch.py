"""Segmented timing benchmark for the mjbatch Warp ray caster (#299).

Records four hot-path segments separately — pose upload (host staging + H2D
copy + scatter kernel), geom pose + AABB update + BVH refit, the intersection
kernel, and the host readback (D2H copy) — plus end-to-end ``update_pose`` and
``trace`` totals. Rays are uploaded once before the timed loop, so the
intersection segment is kernel-only; everything else is measured per call.

This is an explicit host-readback caster: there is intentionally no zero-copy
metric here. Results are printed as a Markdown table for docs/conformance.md.

Usage:

    uv run python benchmarks/bench_mjbatch.py [--device cuda:0] [--iters 100]

Run twice (e.g. ``--device cuda:0`` and ``--device cpu``) to compare devices.
The segment timings call the caster's internals directly; they are a
profiling harness, not public API.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import warp as wp

import uni_ray
from uni_ray.kernels import trace_rays_batch_bvh_kernel, write_body_poses_kernel
from uni_ray.mjbatch import build_collision_description


def _uv_sphere_mesh_xml(name: str, radius: float, nlat: int = 12, nlon: int = 24) -> str:
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
    vertex_attr = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    face_attr = " ".join(f"{a} {b} {c}" for a, b, c in faces)
    return f'<mesh name="{name}" vertex="{vertex_attr}" face="{face_attr}"/>'


def _build_scene_xml() -> str:
    return f"""
    <mujoco>
      <asset>
        {_uv_sphere_mesh_xml("ball", 0.5)}
      </asset>
      <worldbody>
        <geom type="plane" size="0 0 0.1"/>
        <geom type="mesh" mesh="ball" pos="3 0 0.5"/>
        <body pos="0 0 1.2">
          <freejoint/>
          <geom type="sphere" size="0.25" pos="0.45 0 0"/>
          <geom type="box" size="0.15 0.2 0.1" pos="-0.45 0 0"/>
          <geom type="capsule" size="0.08 0.25" pos="0 0.55 0" quat="0.7071068 0.7071068 0 0"/>
          <geom type="cylinder" size="0.1 0.2" pos="0 -0.55 0"/>
          <geom type="ellipsoid" size="0.12 0.2 0.07" pos="0 0 0.45"/>
        </body>
      </worldbody>
    </mujoco>
    """


def _segment_timer(device: str) -> Callable[[Callable[[], None]], float]:
    def time_segment(fn: Callable[[], None]) -> float:
        start = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        return time.perf_counter() - start

    return time_segment


def _run_profile(caster, num_envs: int, num_rays: int, iterations: int) -> dict[str, float]:
    """Time the four hot-path segments over ``iterations`` calls (mean ms)."""
    device = caster._device
    rng = np.random.default_rng(7)
    body_pos = rng.normal(size=(num_envs, caster._num_bodies, 3)) * 0.2
    body_pos[..., 2] += 1.2
    body_quat = rng.normal(size=(num_envs, caster._num_bodies, 4))
    body_quat /= np.linalg.norm(body_quat, axis=-1, keepdims=True)
    origins = rng.normal(size=(num_rays, 3)) * 1.5
    origins[:, 2] += 2.0
    directions = -origins / np.linalg.norm(origins, axis=1, keepdims=True)

    time_segment = _segment_timer(device)
    quat_xyzw = body_quat[..., (1, 2, 3, 0)].astype(np.float32)
    rows = np.arange(num_envs, dtype=np.intp)

    def pose_upload() -> None:
        caster._pose_pos_host[:num_envs] = body_pos.astype(np.float32)
        caster._pose_quat_host[:num_envs] = quat_xyzw
        wp.copy(caster._pose_pos_dev, caster._pose_pos_host_wp)
        wp.copy(caster._pose_quat_dev, caster._pose_quat_host_wp)
        wp.launch(
            write_body_poses_kernel,
            dim=(num_envs, caster._num_bodies),
            inputs=[
                caster._env_map_for(rows),
                caster._pose_pos_dev,
                caster._pose_quat_dev,
                caster._body_xpos_dev,
                caster._body_xquat_dev,
            ],
            device=device,
        )

    def bvh_refit() -> None:
        caster._refresh_poses(rows)
        caster._bvh.refit()

    # Stage the rays once; the intersection segment is kernel-only.
    caster._ray_origins_host[:num_envs] = origins.astype(np.float32)
    caster._ray_directions_host[:num_envs] = directions.astype(np.float32)
    wp.copy(caster._ray_origins_dev, caster._ray_origins_host_wp)
    wp.copy(caster._ray_directions_dev, caster._ray_directions_host_wp)

    def intersection() -> None:
        wp.launch(
            trace_rays_batch_bvh_kernel,
            dim=(num_envs, num_rays),
            inputs=[
                caster._bvh.id,
                caster._geom_types_dev,
                caster._geom_sizes_dev,
                caster._geom_mesh_ids_dev,
                caster._mesh_ids_dev,
                caster._geom_xpos_dev,
                caster._geom_xmat_dev,
                caster._env_map_for(rows),
                caster._ray_origins_dev,
                caster._ray_directions_dev,
                10.0,
                caster._distances_dev,
                caster._hits_dev,
                caster._hit_geoms_dev,
            ],
            device=device,
        )

    def readback() -> None:
        wp.copy(caster._distances_host_wp, caster._distances_dev)
        wp.copy(caster._hits_host_wp, caster._hits_dev)
        wp.copy(caster._hit_geoms_host_wp, caster._hit_geoms_dev)

    segments = {
        "pose_upload": pose_upload,
        "bvh_refit": bvh_refit,
        "intersection": intersection,
        "readback": readback,
    }
    # Warmup: full lifecycle calls (covers lazy kernel compilation).
    for _ in range(3):
        caster.update_pose(body_pos, body_quat)
        caster.trace(origins, directions, max_distance=10.0)

    totals = {name: 0.0 for name in segments}
    for _ in range(iterations):
        for name, segment in segments.items():
            totals[name] += time_segment(segment)
    e2e_pose = 0.0
    e2e_trace = 0.0
    for _ in range(iterations):
        e2e_pose += time_segment(lambda: caster.update_pose(body_pos, body_quat))
        e2e_trace += time_segment(
            lambda: caster.trace(origins, directions, max_distance=10.0)
        )
    results = {name: total / iterations * 1e3 for name, total in totals.items()}
    results["update_pose_total"] = e2e_pose / iterations * 1e3
    results["trace_total"] = e2e_trace / iterations * 1e3
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="warp device (default: cuda:0 if present)")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_string(_build_scene_xml())
    collision = build_collision_description(model)

    print(f"device: {args.device or 'default'}  iters: {args.iters}")
    print(f"scene: {model.ngeom} geoms, {model.nbody} bodies, {len(collision.meshes)} mesh(es)")
    print()
    print("| num_envs | num_rays | pose upload (ms) | pose+AABB+refit (ms) | "
          "intersection (ms) | host readback (ms) | update_pose total (ms) | trace total (ms) |")
    print("|---|---|---|---|---|---|---|---|")
    for num_envs, num_rays in ((64, 128), (256, 512), (1024, 512)):
        caster = uni_ray.create_ray_caster(
            num_envs=num_envs, num_rays=num_rays, collision=collision, device=args.device
        )
        caster.materialize(collision.scene)
        results = _run_profile(caster, num_envs, num_rays, args.iters)
        caster.close()
        print(
            f"| {num_envs} | {num_rays} | {results['pose_upload']:.4f} | "
            f"{results['bvh_refit']:.4f} | {results['intersection']:.4f} | "
            f"{results['readback']:.4f} | {results['update_pose_total']:.4f} | "
            f"{results['trace_total']:.4f} |"
        )


if __name__ == "__main__":
    main()
