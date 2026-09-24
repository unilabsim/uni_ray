"""Batch ray-tracing benchmark: uni_ray vs mujoco-warp (#300 input).

Compares uni_ray's mjbatch Warp ray caster against mujoco-warp's public batch
ray API (``mujoco_warp.rays``) on identical scenes, geom poses, and ray
batches, sweeping (num_envs x num_rays). mujoco-warp is measured in two modes:

- brute force: ``rays(..., rc=None)`` — the default public path; each ray
  tests every geom (and every mesh triangle).
- BVH: ``rays(..., rc=rc)`` with a ``RenderContext`` from
  ``create_render_context``; the scene BVH is rebuilt per iteration with the
  public ``refit_bvh`` so dynamic poses are reflected (same lifecycle as
  uni_ray's update_pose refit).

What is measured per iteration on each side (device work only, warp
synchronize-scoped wall time, mean over --iters):

- uni_ray: ``update_pose`` (public API: validation + staging + H2D + pose
  scatter + AABB + BVH refit), ``trace`` (public API: validation + staging +
  H2D rays + kernel + D2H readback), and the intersection kernel alone (same
  internals segment as bench_mjbatch).
- mujoco-warp: pose update (qpos H2D + ``kinematics`` FK, plus ``refit_bvh``
  in BVH mode) and trace (H2D ray upload + ``rays`` kernel + D2H readback of
  distance/geom_id), plus the ``rays`` kernel alone.

Fairness notes (also in docs/benchmark-mjwarp.md):

- mujoco-warp's ``rays`` kernel always computes hit normals; uni_ray does not
  compute normals at all. This inflates the mujoco-warp kernel slightly.
- mujoco-warp has no max_distance cutoff; uni_ray prunes at max_distance
  (10.0 here) inside its kernel.
- uni_ray's public ``trace`` performs contract-mandated host validation and
  staging that the raw mujoco-warp calls skip; the kernel-only columns are the
  like-for-like ray-throughput comparison.
- mujoco-warp planes are single-sided (mj_ray semantics); the contract plane
  uni_ray implements is double-sided. Rays approaching the plane from below
  therefore hit on uni_ray and miss (or hit a geom beyond) on mujoco-warp;
  the sanity check reports those as expected mismatches.

A correctness sanity check (8 envs x 64 rays, same inputs both sides) runs
before any timing and prints hit-mask agreement and worst distance error.

Usage:

    uv run python benchmarks/bench_vs_mjwarp.py [--device cuda:0] [--iters 100]
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from bench_mjbatch import _build_scene_xml, _run_profile, _uv_sphere_mesh_xml
from mujoco_warp._src.types import vec6
from unisim.ray_query import RayTraceOutputs

import uni_ray
from uni_ray.mjbatch import build_collision_description

MAX_DISTANCE = 10.0
SHAPES = ((64, 128), (256, 512), (1024, 512))

GEOM_TYPES = ("sphere", "box", "capsule", "cylinder", "ellipsoid")
GEOM_SIZES = {
    "sphere": "0.18",
    "box": "0.14 0.11 0.16",
    "capsule": "0.07 0.18",
    "cylinder": "0.09 0.15",
    "ellipsoid": "0.1 0.16 0.07",
}


def _build_dense_scene_xml(nx: int = 4, ny: int = 3) -> str:
    """Plane + static mesh + a grid of freejoint bodies with mixed primitives."""
    bodies = []
    k = 0
    for i in range(nx):
        for j in range(ny):
            x, y = 1.2 * (i - (nx - 1) / 2), 1.2 * (j - (ny - 1) / 2)
            geoms = []
            for n in range(4):
                gtype = GEOM_TYPES[(k + n) % len(GEOM_TYPES)]
                angle = (k + n) * 0.7
                quat = (np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0)
                geoms.append(
                    f'<geom type="{gtype}" size="{GEOM_SIZES[gtype]}" '
                    f'pos="{0.35 * (n - 1.5):.2f} 0 0" quat="{" ".join(map(str, quat))}"/>'
                )
            bodies.append(
                f'<body pos="{x:.2f} {y:.2f} 1.2"><freejoint/>{"".join(geoms)}</body>'
            )
            k += 1
    return f"""
    <mujoco>
      <asset>
        {_uv_sphere_mesh_xml("ball", 0.5)}
      </asset>
      <worldbody>
        <geom type="plane" size="0 0 0.1"/>
        <geom type="mesh" mesh="ball" pos="3 0 0.5"/>
        {"".join(bodies)}
      </worldbody>
    </mujoco>
    """


def _poses(rng: np.random.Generator, num_envs: int, num_bodies: int) -> tuple:
    body_pos = rng.normal(size=(num_envs, num_bodies, 3)) * 0.2
    body_pos[..., 2] += 1.2
    body_quat = rng.normal(size=(num_envs, num_bodies, 4))
    body_quat /= np.linalg.norm(body_quat, axis=-1, keepdims=True)
    # The world body (row 0) is fixed at identity, matching mujoco.
    body_pos[:, 0] = 0.0
    body_quat[:, 0] = [1.0, 0.0, 0.0, 0.0]
    return body_pos, body_quat


def _rays(rng: np.random.Generator, num_rays: int) -> tuple:
    origins = rng.normal(size=(num_rays, 3)) * 1.5
    origins[:, 2] += 2.0
    directions = -origins / np.linalg.norm(origins, axis=1, keepdims=True)
    return origins, directions


def _qpos_from_body_poses(body_pos: np.ndarray, body_quat: np.ndarray, nq: int) -> np.ndarray:
    """Map (envs, nbody, *) wxyz body poses to mujoco freejoint qpos rows."""
    num_envs, num_bodies = body_pos.shape[:2]
    qpos = np.zeros((num_envs, nq), dtype=np.float32)
    for body in range(1, num_bodies):
        adr = 7 * (body - 1)
        qpos[:, adr : adr + 3] = body_pos[:, body]
        qpos[:, adr + 3 : adr + 7] = body_quat[:, body]
    return qpos


class _MjwarpBench:
    """mujoco-warp batch ray harness: qpos upload + FK (+ refit) + rays."""

    def __init__(self, mjm: mujoco.MjModel, num_envs: int, num_rays: int, device: str) -> None:
        mjd = mujoco.MjData(mjm)
        self.m = mjw.put_model(mjm)
        self.d = mjw.put_data(mjm, mjd, nworld=num_envs)
        self.device = device
        self.num_envs = num_envs
        self.num_rays = num_rays
        self.geomgroup = vec6(-1, -1, -1, -1, -1, -1)
        self.bodyexclude = wp.full(num_rays, -1, dtype=int, device=device)
        self.pnt_host = wp.empty((num_envs, num_rays), dtype=wp.vec3, device="cpu")
        self.vec_host = wp.empty((num_envs, num_rays), dtype=wp.vec3, device="cpu")
        self.qpos_host = wp.empty((num_envs, mjm.nq), dtype=float, device="cpu")
        self.pnt = wp.empty((num_envs, num_rays), dtype=wp.vec3, device=device)
        self.vec = wp.empty((num_envs, num_rays), dtype=wp.vec3, device=device)
        self.dist = wp.empty((num_envs, num_rays), dtype=float, device=device)
        self.geomid = wp.empty((num_envs, num_rays), dtype=int, device=device)
        self.normal = wp.empty((num_envs, num_rays), dtype=wp.vec3, device=device)
        self.dist_host = wp.empty((num_envs, num_rays), dtype=float, device="cpu")
        self.geomid_host = wp.empty((num_envs, num_rays), dtype=int, device="cpu")
        self.rc = mjw.create_render_context(mjm, nworld=num_envs)

    def set_inputs(self, qpos: np.ndarray, origins: np.ndarray, directions: np.ndarray) -> None:
        self.qpos_host.numpy()[:] = qpos
        origins32 = np.broadcast_to(origins.astype(np.float32), (self.num_envs, *origins.shape))
        directions32 = np.broadcast_to(
            directions.astype(np.float32), (self.num_envs, *directions.shape)
        )
        self.pnt_host.numpy()[:] = origins32
        self.vec_host.numpy()[:] = directions32

    def pose_update(self, refit: bool) -> None:
        wp.copy(self.d.qpos, self.qpos_host)
        mjw.kinematics(self.m, self.d)
        if refit:
            mjw.refit_bvh(self.m, self.d, self.rc)

    def kernel(self, use_bvh: bool) -> None:
        mjw.rays(
            self.m,
            self.d,
            self.pnt,
            self.vec,
            self.geomgroup,
            True,
            self.bodyexclude,
            self.dist,
            self.geomid,
            self.normal,
            rc=self.rc if use_bvh else None,
        )

    def trace(self, use_bvh: bool) -> None:
        wp.copy(self.pnt, self.pnt_host)
        wp.copy(self.vec, self.vec_host)
        self.kernel(use_bvh)
        wp.copy(self.dist_host, self.dist)
        wp.copy(self.geomid_host, self.geomid)

    def sync(self) -> None:
        wp.synchronize_device(self.device)


def _time(fn, device: str) -> float:
    start = time.perf_counter()
    fn()
    wp.synchronize_device(device)
    return time.perf_counter() - start


def _mean_ms(fn, device: str, iterations: int) -> float:
    return sum(_time(fn, device) for _ in range(iterations)) / iterations * 1e3


def _sanity_check(scene_xml: str, device: str) -> None:
    """Verify uni_ray and mujoco-warp agree on identical inputs (small config)."""
    num_envs, num_rays = 8, 64
    mjm = mujoco.MjModel.from_xml_string(scene_xml)
    collision = build_collision_description(mjm)
    rng = np.random.default_rng(7)
    body_pos, body_quat = _poses(rng, num_envs, mjm.nbody)
    origins, directions = _rays(rng, num_rays)

    caster = uni_ray.create_ray_caster(
        num_envs=num_envs, num_rays=num_rays, collision=collision, device=device
    )
    caster.materialize(collision.scene)
    caster.update_pose(body_pos, body_quat)
    result = caster.trace(
        origins,
        directions,
        max_distance=MAX_DISTANCE,
        outputs=RayTraceOutputs(geom_id=True),
    )
    uni_hit = np.array(result.hit, copy=True)
    uni_dist = np.array(result.distance, copy=True)
    uni_geom = np.array(result.geom_id, copy=True)
    caster.close()

    bench = _MjwarpBench(mjm, num_envs, num_rays, device)
    bench.set_inputs(_qpos_from_body_poses(body_pos, body_quat, mjm.nq), origins, directions)
    for use_bvh in (False, True):
        bench.pose_update(refit=use_bvh)
        bench.trace(use_bvh)
        bench.sync()
        mjw_dist = bench.dist_host.numpy()
        mjw_geom = bench.geomid_host.numpy()
        # mujoco-warp has no max_distance cutoff; clip to the contract window.
        mjw_hit = (mjw_geom >= 0) & (mjw_dist < MAX_DISTANCE)
        agree = uni_hit & mjw_hit
        same_geom = agree & (uni_geom == mjw_geom)
        worst = (
            float(np.max(np.abs(uni_dist[same_geom] - mjw_dist[same_geom])))
            if np.any(same_geom)
            else 0.0
        )
        mask_mismatch = int(np.count_nonzero(uni_hit != mjw_hit))
        geom_mismatch = int(np.count_nonzero(agree & ~same_geom))
        mode = "bvh  " if use_bvh else "brute"
        print(
            f"  sanity [{mode}]: agreed hits {int(np.count_nonzero(agree))}/{uni_hit.size}, "
            f"hit-mask mismatches {mask_mismatch}, geom-id mismatches {geom_mismatch}, "
            f"worst |dt| on agreed hits {worst:.3e}"
        )
        if mask_mismatch or geom_mismatch:
            plane = 0
            only_uni = uni_hit & ~mjw_hit
            only_mjw = mjw_hit & ~uni_hit
            diff_geom = agree & ~same_geom
            print(
                f"    mismatch detail: uni-hit-only involving plane geom "
                f"{int(np.count_nonzero(only_uni & (uni_geom == plane)))}/"
                f"{int(np.count_nonzero(only_uni))}; mjw-hit-only "
                f"{int(np.count_nonzero(only_mjw))}; both-hit different geom "
                f"{int(np.count_nonzero(diff_geom))} "
                f"(uni->plane: {int(np.count_nonzero(diff_geom & (uni_geom == plane)))})"
            )
        assert worst < 1e-3, f"mujoco-warp [{mode}] distance mismatch {worst}"


def _run_shape(
    mjm, collision, num_envs: int, num_rays: int, iterations: int, device: str
) -> dict[str, float]:
    rng = np.random.default_rng(7)
    body_pos, body_quat = _poses(rng, num_envs, mjm.nbody)
    origins, directions = _rays(rng, num_rays)

    caster = uni_ray.create_ray_caster(
        num_envs=num_envs, num_rays=num_rays, collision=collision, device=device
    )
    caster.materialize(collision.scene)
    uni = _run_profile(caster, num_envs, num_rays, iterations)
    caster.close()

    bench = _MjwarpBench(mjm, num_envs, num_rays, device)
    bench.set_inputs(_qpos_from_body_poses(body_pos, body_quat, mjm.nq), origins, directions)
    # Warmup (covers lazy kernel compilation), then timed segments.
    for use_bvh in (False, True):
        for _ in range(3):
            bench.pose_update(refit=use_bvh)
            bench.trace(use_bvh)
    out = dict(uni)
    out["mjw_brute_pose"] = _mean_ms(lambda: bench.pose_update(False), device, iterations)
    out["mjw_brute_trace"] = _mean_ms(lambda: bench.trace(False), device, iterations)
    out["mjw_brute_kernel"] = _mean_ms(lambda: bench.kernel(False), device, iterations)
    out["mjw_bvh_pose"] = _mean_ms(lambda: bench.pose_update(True), device, iterations)
    out["mjw_bvh_trace"] = _mean_ms(lambda: bench.trace(True), device, iterations)
    out["mjw_bvh_kernel"] = _mean_ms(lambda: bench.kernel(True), device, iterations)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="warp device (default: cuda:0 if present)")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--skip-dense", action="store_true", help="only run the standard 7-geom scene"
    )
    args = parser.parse_args()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    wp.init()

    scenes = [("standard (7 geoms)", _build_scene_xml())]
    if not args.skip_dense:
        scenes.append(("dense (50 geoms)", _build_dense_scene_xml()))

    print(f"device: {device}  iters: {args.iters}  max_distance: {MAX_DISTANCE}")
    print("mujoco-warp mode: qpos H2D + kinematics (+ refit_bvh in BVH mode) + rays")
    for scene_name, scene_xml in scenes:
        print(f"\nscene: {scene_name}")
        print("correctness sanity check (8 envs x 64 rays, identical inputs):")
        with wp.ScopedDevice(device):
            _sanity_check(scene_xml, device)
            mjm = mujoco.MjModel.from_xml_string(scene_xml)
            collision = build_collision_description(mjm)
            rows = []
            for num_envs, num_rays in SHAPES:
                rows.append(
                    (
                        num_envs,
                        num_rays,
                        _run_shape(mjm, collision, num_envs, num_rays, args.iters, device),
                    )
                )
        print(
            "\n| num_envs | num_rays | uni_ray update_pose (ms) | uni_ray trace (ms) | "
            "mjw brute pose (ms) | mjw brute trace (ms) | mjw bvh pose+refit (ms) | "
            "mjw bvh trace (ms) | speedup vs brute | speedup vs bvh |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for num_envs, num_rays, r in rows:
            print(
                f"| {num_envs} | {num_rays} | {r['update_pose_total']:.4f} | "
                f"{r['trace_total']:.4f} | {r['mjw_brute_pose']:.4f} | "
                f"{r['mjw_brute_trace']:.4f} | {r['mjw_bvh_pose']:.4f} | "
                f"{r['mjw_bvh_trace']:.4f} | {r['mjw_brute_trace'] / r['trace_total']:.2f}x | "
                f"{r['mjw_bvh_trace'] / r['trace_total']:.2f}x |"
            )
        print(
            "\n| num_envs | num_rays | uni_ray kernel (ms) | mjw brute kernel (ms) | "
            "mjw bvh kernel (ms) | kernel ratio brute/uni_ray | kernel ratio bvh/uni_ray |"
        )
        print("|---|---|---|---|---|---|---|")
        for num_envs, num_rays, r in rows:
            print(
                f"| {num_envs} | {num_rays} | {r['intersection']:.4f} | "
                f"{r['mjw_brute_kernel']:.4f} | {r['mjw_bvh_kernel']:.4f} | "
                f"{r['mjw_brute_kernel'] / r['intersection']:.2f}x | "
                f"{r['mjw_bvh_kernel'] / r['intersection']:.2f}x |"
            )


if __name__ == "__main__":
    main()
