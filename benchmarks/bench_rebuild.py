"""Cold-path rebuild benchmark for the mjbatch Warp ray caster (#2).

uni_ray-only (the rebuild API has no mujoco-warp counterpart): times the
explicit scene-replacement path added for runtime geometry randomization,
on a fixed (num_envs, num_rays) batch over a mesh-heavy scene (24 mesh geoms
sharing one ~2208-triangle asset + plane):

- ``build_collision_description(model)`` — host-side descriptor build
  (geom extraction + mesh triangulation/packing), no caster involved.
- ``caster.rebuild(collision=...)`` — full rebind with a changed mesh asset:
  new wp.Mesh objects (incl. warp's per-mesh BVH build) + scene device arrays
  + scene wp.Bvh build. An instrumented split (timed wrapper around wp.Mesh,
  same spy pattern as tests/test_hotpath.py) separates the wp.Mesh builds
  from the rest of the rebind; the remainder is derived by subtraction.
- ``caster.rebuild(scene=...)`` — geometry randomization (geom sizes/local
  poses) that reuses the mesh data already bound to the caster.
- ``trace`` before vs after a rebuild (same poses re-applied): the rebuild is
  a cold path and must not change hot-path cost; outputs must be identical.
- amortized: (rebuild + 100 traces) / 100, i.e. the per-trace cost of
  rebuilding the scene every 100 traces (e.g. periodic domain randomization).

The rebuild path reallocates scene-dependent state on every call, so this is
a cold-path measurement: ~15 iterations after a short warmup.

Usage:

    uv run python benchmarks/bench_rebuild.py [--device cuda:0] [--iters 15]
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import mujoco
import numpy as np
import warp as wp
from bench_vs_mjwarp import _build_dense_mesh_scene_xml, _poses, _rays

import uni_ray
from uni_ray.mjbatch import build_collision_description

TRACE_BATCH = (256, 512)
MAX_DISTANCE = 10.0


class _TimedCall:
    """Sync-scoped wall-time accumulator wrapping a callable (spy pattern)."""

    def __init__(self, wrapped, device: str) -> None:
        self._wrapped = wrapped
        self._device = device
        self.total = 0.0

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        out = self._wrapped(*args, **kwargs)
        wp.synchronize_device(self._device)
        self.total += time.perf_counter() - start
        return out


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
    parser.add_argument("--iters", type=int, default=15, help="cold-path iterations")
    parser.add_argument("--trace-iters", type=int, default=50, help="trace timing iterations")
    args = parser.parse_args()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    wp.init()

    num_envs, num_rays = TRACE_BATCH
    model_v1 = mujoco.MjModel.from_xml_string(_build_dense_mesh_scene_xml(radius=0.4))
    # Same scene with a scaled mesh asset: the rebuild replaces mesh data.
    model_v2 = mujoco.MjModel.from_xml_string(_build_dense_mesh_scene_xml(radius=0.55))
    collision_v1 = build_collision_description(model_v1)
    collision_v2 = build_collision_description(model_v2)
    num_tris = collision_v1.meshes[0].indices.size // 3

    rng = np.random.default_rng(7)
    body_pos, body_quat = _poses(rng, num_envs, model_v1.nbody)
    origins, directions = _rays(rng, num_rays)

    with wp.ScopedDevice(device):
        caster = uni_ray.create_ray_caster(
            num_envs=num_envs, num_rays=num_rays, collision=collision_v1, device=device
        )
        caster.materialize(collision_v1.scene)

        # Warmup: full lifecycle incl. one rebuild of each kind (kernel compiles).
        caster.update_pose(body_pos, body_quat)
        caster.trace(origins, directions, max_distance=MAX_DISTANCE)
        caster.rebuild(collision_v2)
        caster.rebuild(collision_v1)
        caster.rebuild(scene=collision_v1.scene)

        descriptor_ms = _mean_ms(
            lambda: build_collision_description(model_v2), device, args.iters
        )

        # rebuild(collision=...) with alternating descriptors so every call
        # rebinds changed mesh data; wp.Mesh builds are timed via the spy.
        mesh_spy = _TimedCall(wp.Mesh, device)
        wp.Mesh = mesh_spy  # type: ignore[assignment]
        try:
            rebuild_mesh_ms = _mean_ms(
                lambda: caster.rebuild(
                    collision_v2 if caster._collision is collision_v1 else collision_v1
                ),
                device,
                args.iters,
            )
        finally:
            wp.Mesh = mesh_spy._wrapped  # type: ignore[assignment]
        wp_mesh_ms = mesh_spy.total / args.iters * 1e3

        # rebuild(scene=...): geometry randomization reusing bound mesh data.
        sizes_a = np.array(collision_v1.scene.geom_sizes)
        sizes_b = sizes_a.copy()
        sizes_b[1:] *= 1.2
        state = {"a": True}

        def rebuild_scene() -> None:
            sizes = sizes_a if state["a"] else sizes_b
            state["a"] = not state["a"]
            caster.rebuild(scene=dataclasses.replace(collision_v1.scene, geom_sizes=sizes))

        rebuild_scene_ms = _mean_ms(rebuild_scene, device, args.iters)

        # Trace before vs after a rebuild (same descriptor, poses re-applied):
        # timing must be unchanged and outputs bitwise identical.
        caster.rebuild(collision_v1)
        caster.update_pose(body_pos, body_quat)
        trace_before_ms = _mean_ms(
            lambda: caster.trace(origins, directions, max_distance=MAX_DISTANCE),
            device,
            args.trace_iters,
        )
        before = caster.trace(origins, directions, max_distance=MAX_DISTANCE)
        dist_before = np.array(before.distance, copy=True)

        caster.rebuild(collision_v1)
        caster.update_pose(body_pos, body_quat)
        trace_after_ms = _mean_ms(
            lambda: caster.trace(origins, directions, max_distance=MAX_DISTANCE),
            device,
            args.trace_iters,
        )
        after = caster.trace(origins, directions, max_distance=MAX_DISTANCE)
        dist_after = np.array(after.distance, copy=True)
        max_diff = float(np.max(np.abs(dist_after - dist_before)))
        caster.close()

    amortized_ms = (rebuild_mesh_ms + 100.0 * trace_before_ms) / 100.0

    print(f"device: {device}  iters: {args.iters}  trace iters: {args.trace_iters}")
    print(
        f"scene: dense mesh ({model_v1.ngeom} geoms, {model_v1.nbody} bodies, "
        f"1 shared {num_tris}-tri mesh asset), batch {num_envs} x {num_rays}"
    )
    print()
    print("| segment | mean (ms) |")
    print("|---|---|")
    print(f"| build_collision_description (host, changed mesh) | {descriptor_ms:.4f} |")
    print(f"| rebuild(collision=...) total | {rebuild_mesh_ms:.4f} |")
    print(f"| &nbsp;&nbsp;of which wp.Mesh builds (instrumented) | {wp_mesh_ms:.4f} |")
    print(
        f"| &nbsp;&nbsp;of which scene arrays + scene BVH (remainder) "
        f"| {rebuild_mesh_ms - wp_mesh_ms:.4f} |"
    )
    print(f"| rebuild(scene=...) total (geom sizes, mesh data reused) | {rebuild_scene_ms:.4f} |")
    print(f"| trace before rebuild | {trace_before_ms:.4f} |")
    print(f"| trace after rebuild | {trace_after_ms:.4f} |")
    print(f"| amortized (rebuild + 100 traces) / 100 | {amortized_ms:.4f} |")
    print()
    print(
        f"post-rebuild trace check: max |dt| vs pre-rebuild (same descriptor, same poses) "
        f"= {max_diff:.3e}"
    )


if __name__ == "__main__":
    main()
