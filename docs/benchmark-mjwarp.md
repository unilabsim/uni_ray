# Batch ray tracing: uni_ray vs mujoco-warp (#300 input)

Measured with `benchmarks/bench_vs_mjwarp.py` on an RTX 4090 (`--device cuda:0`,
100 iterations, warp-lang 1.17.0 / mujoco 3.14.0 / mujoco-warp 3.14.0). This
comparison informs the planned mujoco-warp ray-query adapter (unisim#300).

## What is compared

Both sides trace the **same rays** against the **same geom poses** (one NumPy
seed feeds both). Two scenes:

- **standard** (7 geoms): infinite plane, static UV-sphere mesh (530
  verts / 1056 faces), one freejoint body with sphere/box/capsule/cylinder/
  ellipsoid — the `bench_mjbatch.py` scene.
- **dense** (50 geoms): the same plane and mesh plus a 4x3 grid of freejoint
  bodies carrying 48 mixed primitives, to make broadphase differences visible.

Measured entry points (device work, warp-synchronize-scoped wall time, mean):

| Side | Pose update | Ray query |
|---|---|---|
| uni_ray | `update_pose` (public API: host validation + staging + H2D + pose scatter + AABB + BVH refit) | `trace` (public API: host validation + staging + H2D rays + BVH kernel + D2H readback of distance/hit) |
| mujoco-warp brute | qpos H2D + `mjw.kinematics` (on-device FK from the same body poses) | H2D rays + `mjw.rays(..., rc=None)` (brute-force kernel: every ray tests every geom and every mesh triangle) + D2H distance/geom_id |
| mujoco-warp BVH | qpos H2D + `mjw.kinematics` + `mjw.refit_bvh` (scene-BVH refit, public API) | same, with `rc=rc` from `create_render_context` (BVH-accelerated kernel) |

`mjw.rays` is mujoco-warp's public batch ray entry point (shape
`(nworld, nray)`), chosen because it accepts explicit origins/directions and
reads `d.geom_xpos`/`d.geom_xmat`, so no physics stepping is involved — the
direct equivalent of uni_ray's `update_pose` + `trace`. The BVH mode requires
`create_render_context` (its scene BVH is built once from the model's default
pose), so the public `mjw.refit_bvh` is called per iteration to reflect the
updated poses, mirroring uni_ray's per-`update_pose` refit. The rangefinder
sensor path was not used: it routes through the same kernels but couples ray
counts to sensor declarations.

## Correctness sanity check

Runs before timing (8 envs x 64 rays, identical inputs; hard-asserted worst
distance error < 1e-3 on agreed hits):

- standard: agreed hits 448/512, worst |dt| **4.41e-05**, 0 geom-id
  mismatches; 64 hit-mask mismatches, all uni_ray-hit-only on the plane.
- dense: agreed hits 456/512, worst |dt| **5.72e-05**; 56 uni_ray-hit-only
  plane hits plus 16 rays where uni_ray stops at the plane and mujoco-warp
  passes through to a geom above — the same plane difference.
- brute and BVH modes agree with each other exactly.

### Semantic differences (documented, not bugs)

1. **Plane sidedness**: mujoco-warp implements `mj_ray` semantics — planes
   are single-sided (approached from the +normal side only). The contract
   plane uni_ray implements is double-sided infinite, so rays from below the
   plane hit on uni_ray and miss (or hit geometry beyond) on mujoco-warp.
   Every observed mismatch is of this kind.
2. **Miss sentinel / cutoff**: mujoco-warp reports misses as `dist = -1`,
   `geom_id = -1` and has no `max_distance` cutoff; the comparison clips
   mujoco-warp hits at the contract's `max_distance` (10.0).
3. **Normals**: the `mjw.rays` kernel always computes hit normals; uni_ray
   does not compute normals at all. This inflates the mujoco-warp kernel
   time slightly (conservative for the ratios below).
4. **Inside-origin** rays were not exercised by these ray distributions;
   uni_ray matches `mj_ray` exit-distance semantics there (see
   docs/conformance.md).

## Results

### standard scene (7 geoms)

| num_envs | num_rays | uni_ray update_pose (ms) | uni_ray trace (ms) | mjw brute pose (ms) | mjw brute trace (ms) | mjw bvh pose+refit (ms) | mjw bvh trace (ms) | speedup vs brute | speedup vs bvh |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0942 | 0.2544 | 0.0532 | 0.1002 | 0.0975 | 0.0618 | 0.39x | 0.24x |
| 256 | 512 | 0.1071 | 3.0115 | 0.0525 | 1.2441 | 0.0803 | 0.3617 | 0.41x | 0.12x |
| 1024 | 512 | 0.1501 | 21.2322 | 0.0618 | 4.2651 | 0.0822 | 1.0687 | 0.20x | 0.05x |

| num_envs | num_rays | uni_ray kernel (ms) | mjw brute kernel (ms) | mjw bvh kernel (ms) | kernel ratio brute/uni_ray | kernel ratio bvh/uni_ray |
|---|---|---|---|---|---|---|
| 64 | 128 | 0.0537 | 0.0674 | 0.0477 | 1.26x | 0.89x |
| 256 | 512 | 0.0836 | 0.9804 | 0.0959 | 11.73x | 1.15x |
| 1024 | 512 | 0.1991 | 3.3414 | 0.2491 | 16.78x | 1.25x |

### dense scene (50 geoms)

| num_envs | num_rays | uni_ray update_pose (ms) | uni_ray trace (ms) | mjw brute pose (ms) | mjw brute trace (ms) | mjw bvh pose+refit (ms) | mjw bvh trace (ms) | speedup vs brute | speedup vs bvh |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.1213 | 0.3004 | 0.0548 | 0.2339 | 0.0864 | 0.1111 | 0.78x | 0.37x |
| 256 | 512 | 0.1763 | 3.2219 | 0.0561 | 1.1658 | 0.0842 | 0.5030 | 0.36x | 0.16x |
| 1024 | 512 | 0.3537 | 20.5515 | 0.0700 | 4.5684 | 0.1162 | 1.5401 | 0.22x | 0.07x |

| num_envs | num_rays | uni_ray kernel (ms) | mjw brute kernel (ms) | mjw bvh kernel (ms) | kernel ratio brute/uni_ray | kernel ratio bvh/uni_ray |
|---|---|---|---|---|---|---|
| 64 | 128 | 0.0930 | 0.1963 | 0.0807 | 2.11x | 0.87x |
| 256 | 512 | 0.1847 | 0.9159 | 0.2316 | 4.96x | 1.25x |
| 1024 | 512 | 0.5486 | 3.6606 | 0.7421 | 6.67x | 1.35x |

"Speedup" columns are mujoco-warp trace time / uni_ray trace time: **>1 means
uni_ray is faster**. Kernel ratios are likewise mujoco-warp kernel / uni_ray
intersection kernel.

## Reading the results

- **Kernels**: uni_ray's BVH kernel is 11.7-16.8x (standard) / 2.1-6.7x
  (dense) faster than mujoco-warp's default brute-force kernel at batch
  sizes, and is at rough parity with mujoco-warp's BVH kernel (0.87-1.35x —
  mujoco-warp ~10-15% faster at small shapes, uni_ray ~25-35% faster at
  1024x512). Note mujoco-warp's BVH path is tied to the render context and
  needs the separate `create_render_context` + `refit_bvh` lifecycle.
- **End-to-end**: uni_ray's public `trace` loses to mujoco-warp's raw calls
  (0.05-0.78x) because its contract-mandated host validation and ray staging
  dominate at large batch (21.2 ms total vs 0.20 ms kernel at 1024x512 — see
  docs/conformance.md for the segment breakdown). The mujoco-warp side does
  no equivalent host work. An mjwarp-based adapter for #300 would therefore
  win on host overhead but not on kernel throughput; closing uni_ray's host
  staging gap is the larger lever regardless of backend.
- **Pose update**: mujoco-warp's qpos-upload + FK is cheaper (0.05-0.12 ms)
  than uni_ray's public `update_pose` (0.09-0.35 ms), again mostly uni_ray
  host-side validation; uni_ray's device-side pose+refit segments alone are
  comparable (see bench_mjbatch segments).

## Fairness caveats

- mujoco-warp kernels compute hit normals; uni_ray does not.
- mujoco-warp has no `max_distance` pruning inside its kernels; uni_ray
  prunes at 10.0.
- uni_ray's timed `trace` returns distance+hit; mujoco-warp's timed readback
  returns distance+geom_id (geom_id materialization on uni_ray is one extra
  D2H copy of an int array, costed separately in bench_mjbatch's readback
  segment).
- mujoco-warp's brute kernel uses `wp.launch_tiled`; the BVH kernel uses
  `wp.launch`. Both were warmed up identically (3 full iterations) before
  timing.
- The mujoco-warp side performs no Python-level input validation in the
  timed loop; uni_ray's public API does. Compare kernels for throughput,
  totals for API cost.
