"""Rich tensor-query benchmark for the public UniSim collision descriptor (#472).

The profile is deliberately explicit so results are comparable across runs:

- fixed ``--num-envs`` x ``--num-rays`` batch;
- ``max_distance=10.0`` and inclusive ``min_distance=0.1``;
- primitive, shared-mesh, and hfield geometry profiles;
- normal output on and off;
- static-filter densities 1.0 and 0.5;
- nearest-valid-hit semantics only (literal continuation is not implemented).

Each row reports sync-scoped launch time for ``trace_rays_batch_bvh_kernel``
only (not a pure CUDA-event kernel measurement) and end-to-end
``trace_tensor`` time on the caller's current stream. Pose update, descriptor
materialization, output allocation, and filter-mask upload are cold/warmup
work and are not included.

Usage:

    uv run python benchmarks/bench_rich_queries.py [--device cuda:0] \
        [--num-envs 256] [--num-rays 512] [--iters 20]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from bench_mjbatch import _hfield_xml, _uv_sphere_mesh_xml
from unisim.backend.ray_scene import ray_collision_from_mujoco_model
from unisim.ray_query import RayPoseTensorView, RayTraceFilter, RayTraceOutputs

import uni_ray
from uni_ray.kernels import trace_rays_batch_bvh_kernel

MAX_DISTANCE = 10.0
MIN_DISTANCE = 0.1


def _primitive_xml() -> str:
    return """
    <mujoco>
      <worldbody>
        <geom type="plane" size="0 0 0.1"/>
        <geom type="sphere" size="0.35" pos="0 0 0.35"/>
        <geom type="box" size="0.2 0.2 0.2" pos="1 0 0.2"/>
        <geom type="capsule" size="0.1 0.3" pos="-1 0 0.3"/>
        <geom type="cylinder" size="0.12 0.25" pos="0 1 0.25"/>
        <geom type="ellipsoid" size="0.2 0.15 0.1" pos="0 -1 0.1"/>
      </worldbody>
    </mujoco>
    """


def _mesh_xml() -> str:
    positions = ((0, 0, 0.5), (1.2, 0, 0.5), (-1.2, 0, 0.5), (0, 1.2, 0.5))
    geoms = "\n".join(f'<geom type="mesh" mesh="ball" pos="{x} {y} {z}"/>' for x, y, z in positions)
    return f"""
    <mujoco>
      <asset>{_uv_sphere_mesh_xml("ball", 0.5, nlat=24, nlon=48)}</asset>
      <worldbody>
        <geom type="plane" size="0 0 0.1"/>
        {geoms}
      </worldbody>
    </mujoco>
    """


def _hfield_profile_xml() -> str:
    return f"""
    <mujoco>
      <asset>{_hfield_xml("terrain", 33, 33, "4 4 0.8 0.2")}</asset>
      <worldbody>
        <geom type="hfield" hfield="terrain"/>
      </worldbody>
    </mujoco>
    """


def _rays(num_rays: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(472)
    origins = np.zeros((num_rays, 3), dtype=np.float32)
    origins[:, 0] = rng.uniform(-4.0, 4.0, num_rays)
    origins[:, 1] = rng.uniform(-4.0, 4.0, num_rays)
    origins[:, 2] = rng.uniform(2.0, 4.0, num_rays)
    targets = rng.uniform(-1.0, 1.0, (num_rays, 3)).astype(np.float32)
    targets[:, 2] = rng.uniform(0.0, 0.8, num_rays)
    directions = targets - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return (
        torch.tensor(origins, dtype=torch.float32, device=device),
        torch.tensor(directions, dtype=torch.float32, device=device),
    )


def _mean_ms(fn: Callable[[], object], device: str, iterations: int) -> float:
    total = 0.0
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        wp.synchronize_device(device)
        total += time.perf_counter() - start
    return total / iterations * 1e3


def _trace_kernel_ms(call: Callable[[], object], device: str, iterations: int) -> float:
    original_launch = wp.launch
    samples: list[float] = []

    def timed_launch(kernel, *args, **kwargs):
        if kernel is not trace_rays_batch_bvh_kernel:
            return original_launch(kernel, *args, **kwargs)
        start = time.perf_counter()
        result = original_launch(kernel, *args, **kwargs)
        wp.synchronize_device(device)
        samples.append(time.perf_counter() - start)
        return result

    wp.launch = timed_launch
    try:
        for _ in range(iterations):
            call()
    finally:
        wp.launch = original_launch
    return float(np.mean(samples) * 1e3) if samples else float("nan")


def _profile(
    name: str,
    xml: str,
    num_envs: int,
    num_rays: int,
    device: str,
    normal: bool,
    density: float,
    iterations: int,
) -> tuple[object, ...]:
    model = mujoco.MjModel.from_xml_string(xml)
    collision = ray_collision_from_mujoco_model(model)
    caster = uni_ray.create_ray_caster(
        num_envs=num_envs,
        num_rays=num_rays,
        collision=collision,
        device=device,
    )
    caster.materialize(collision.scene)
    caster.bind_pose_layout(collision.scene.body_names)
    pos = torch.zeros((num_envs, len(collision.scene.body_names), 3), device=device)
    quat = torch.zeros((num_envs, len(collision.scene.body_names), 4), device=device)
    quat[..., 0] = 1.0
    caster.update_pose_tensor(RayPoseTensorView(collision.scene.body_names, pos, quat))

    origins, directions = _rays(num_rays, device)
    count = max(1, int(np.ceil(density * collision.num_geoms)))
    ray_filter = RayTraceFilter(allowed_geom_ids=tuple(range(count)))
    outputs = RayTraceOutputs(normal=normal)

    def trace() -> object:
        return caster.trace_tensor(
            origins,
            directions,
            MAX_DISTANCE,
            outputs=outputs,
            min_distance=MIN_DISTANCE,
            ray_filter=ray_filter,
        )

    for _ in range(3):
        trace()
    wp.synchronize_device(device)
    kernel_ms = _trace_kernel_ms(trace, device, iterations)
    end_to_end_ms = _mean_ms(trace, device, iterations)
    result = trace()
    wp.synchronize_device(device)
    hit_fraction = float(result.hit.detach().float().mean().item())
    caster.close()
    return name, collision.num_geoms, normal, density, kernel_ms, end_to_end_ms, hit_fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Torch/Warp device")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-rays", type=int, default=512)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wp.init()

    print(f"device: {device}")
    print(f"warp: {wp.__version__}")
    print(f"batch: {args.num_envs} x {args.num_rays}")
    print(f"max_distance: {MAX_DISTANCE}  min_distance: {MIN_DISTANCE}")
    print("continuation: disabled (nearest-valid-hit)")
    print()
    print(
        "| geometry | geoms | normal | filter density | "
        "intersection kernel (ms) | trace_tensor e2e (ms) | hit fraction |"
    )
    print("|---|---:|---|---:|---:|---:|---:|")
    profiles = (
        ("primitives", _primitive_xml()),
        ("shared-mesh", _mesh_xml()),
        ("hfield", _hfield_profile_xml()),
    )
    for name, xml in profiles:
        for normal in (False, True):
            for density in (1.0, 0.5):
                row = _profile(
                    name,
                    xml,
                    args.num_envs,
                    args.num_rays,
                    device,
                    normal,
                    density,
                    args.iters,
                )
                print(
                    f"| {row[0]} | {row[1]} | {str(row[2]).lower()} | {row[3]:.1f} | "
                    f"{row[4]:.3f} | {row[5]:.3f} | {row[6]:.3f} |"
                )


if __name__ == "__main__":
    if __package__ in (None, ""):
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
    main()
