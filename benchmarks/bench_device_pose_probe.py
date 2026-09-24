"""Device-pose boundary probe (#300): mjwarp Data.xpos/xquat D2H vs pose-sync.

Lightweight, one-off measurement backing the host pose-sync vs backend-owned
device-pose evaluation in docs/backends.md. For an mjwarp profile deployment,
the pose question per frame is:

- uni_ray's phase-1 contract path: host NumPy body poses -> ``update_pose``
  (validation + staging + H2D + pose scatter + AABB + BVH refit), measured
  here end-to-end on the dense scene.
- the device-pose alternative: ``mujoco_warp.Data.xpos``/``xquat`` already
  live on device after FK. Consuming them without a contract device-pose
  capability means reading them back (D2H, measured here) and feeding the
  same host path — a device->host->device round trip that pays a sync point
  per frame and still pays the full ``update_pose`` cost on top.

Sweeping num_envs on the dense scene (13 bodies). No new benchmark framework:
both sides are timed with the same perf-counter + synchronize pattern as
bench_vs_mjwarp.py.

Usage:

    uv run python benchmarks/bench_device_pose_probe.py [--device cuda:0] [--iters 100]
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from bench_vs_mjwarp import _build_dense_scene_xml, _poses

import uni_ray
from uni_ray.mjbatch import build_collision_description


def _mean_ms(fn, device: str, iterations: int) -> float:
    total = 0.0
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        total += time.perf_counter() - start
    return total / iterations * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="warp device (default: cuda:0 if present)")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    wp.init()

    mjm = mujoco.MjModel.from_xml_string(_build_dense_scene_xml())
    mjd = mujoco.MjData(mjm)
    collision = build_collision_description(mjm)
    print(f"device: {device}  iters: {args.iters}  scene: dense ({mjm.nbody} bodies)")
    print()
    print("| num_envs | mjw xpos+xquat D2H (ms) | uni_ray update_pose (ms) | D2H / update_pose |")
    print("|---|---|---|---|")
    for num_envs in (64, 256, 1024):
        with wp.ScopedDevice(device):
            d = mjw.put_data(mjm, mjd, nworld=num_envs)
            xpos_host = wp.empty((num_envs, mjm.nbody), dtype=wp.vec3, device="cpu")
            xquat_host = wp.empty((num_envs, mjm.nbody), dtype=wp.quat, device="cpu")

            def d2h_poses() -> None:
                wp.copy(xpos_host, d.xpos)
                wp.copy(xquat_host, d.xquat)

            caster = uni_ray.create_ray_caster(
                num_envs=num_envs, num_rays=8, collision=collision, device=device
            )
            caster.materialize(collision.scene)
            rng = np.random.default_rng(7)
            body_pos, body_quat = _poses(rng, num_envs, mjm.nbody)

            # Warmup both sides.
            for _ in range(3):
                d2h_poses()
                caster.update_pose(body_pos, body_quat)
            wp.synchronize_device(device)

            d2h_ms = _mean_ms(d2h_poses, device, args.iters)
            update_ms = _mean_ms(
                lambda: caster.update_pose(body_pos, body_quat), device, args.iters
            )
            caster.close()
        print(f"| {num_envs} | {d2h_ms:.4f} | {update_ms:.4f} | {d2h_ms / update_ms:.2f}x |")


if __name__ == "__main__":
    main()
