# Batch ray tracing: uni_ray vs mujoco-warp (#300 input)

Measured with `benchmarks/bench_vs_mjwarp.py` on an RTX 4090 (`--device cuda:0`,
100 iterations, warp-lang 1.17.0 / mujoco 3.14.0 / mujoco-warp 3.14.0). This
comparison informs the planned mujoco-warp ray-query adapter (unisim#300).
The hfield, dense-mesh, and rebuild-path sections were added in a later run
under the same conditions (idle GPU, verified via nvidia-smi).

## What is compared

Both sides trace the **same rays** against the **same geom poses** (one NumPy
seed feeds both). Four scenes (select with `--scenes`; all run by default):

- **standard** (7 geoms): infinite plane, static UV-sphere mesh (530
  verts / 1056 faces), one freejoint body with sphere/box/capsule/cylinder/
  ellipsoid — the `bench_mjbatch.py` scene.
- **dense** (50 geoms): the same plane and mesh plus a 4x3 grid of freejoint
  bodies carrying 48 mixed primitives, to make broadphase differences visible.
- **hfield** (7 geoms): a 33x33 elevation-grid hfield terrain (uni_ray
  triangulates it into the closed `mj_rayHfield` solid: ~2.2k triangles) +
  static mesh + the standard primitive body. See the capability finding
  below: mujoco-warp's BVH mode is excluded from timing on this scene.
- **densemesh** (25 geoms): 24 mesh geoms sharing one ~2208-triangle
  UV-sphere asset (8 freejoint bodies x 3 geoms) + plane — the mesh-heavy
  case, with rays aimed at the body ring so the meshes are actually
  exercised.

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
- hfield, brute mode: agreed hits **512/512, zero mismatches**, worst |dt|
  **4.41e-05** — including the skirt-wall and below-base ray families.
- hfield, BVH mode: **capability finding** — mujoco-warp's RenderContext
  scene BVH only covers the hfield top surface: 64/64 side/below rays that
  uni_ray and the brute mode resolve against the closed solid (skirt walls,
  base box) are missed by the BVH mode, and below-base rays that do register
  hit the top surface from underneath instead (agreed-hit |dt| up to 4.84).
  The hfield scene therefore times mujoco-warp in brute mode only; BVH-mode
  hfield rays would be semantically wrong, not merely slower.
- densemesh: agreed hits 512/512, zero mismatches in both modes, worst |dt|
  **9.54e-07** (brute) / **1.07e-06** (BVH).
- on primitive scenes, brute and BVH modes agree with each other exactly.

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

### hfield scene (33x33 terrain, ~2.2k triangulated triangles; mjwarp brute only)

mujoco-warp BVH mode excluded — its RenderContext BVH covers only the hfield
top surface (see the capability finding in the sanity check above).

| num_envs | num_rays | uni_ray update_pose (ms) | uni_ray trace (ms) | mjw brute pose (ms) | mjw brute trace (ms) | mjw bvh pose+refit (ms) | mjw bvh trace (ms) | speedup vs brute | speedup vs bvh |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0923 | 0.3429 | 0.0522 | 0.2637 | - | - | 0.77x | - |
| 256 | 512 | 0.1346 | 3.4619 | 0.0527 | 3.4710 | - | - | 1.00x | - |
| 1024 | 512 | 0.1851 | 21.0971 | 0.0547 | 12.7924 | - | - | 0.61x | - |

| num_envs | num_rays | uni_ray kernel (ms) | mjw brute kernel (ms) | mjw bvh kernel (ms) | kernel ratio brute/uni_ray | kernel ratio bvh/uni_ray |
|---|---|---|---|---|---|---|
| 64 | 128 | 0.1329 | 0.2322 | - | 1.75x | - |
| 256 | 512 | 0.3124 | 3.0818 | - | 9.87x | - |
| 1024 | 512 | 1.0112 | 11.3149 | - | 11.19x | - |

### dense-mesh scene (24 x 2208-tri mesh geoms + plane)

| num_envs | num_rays | uni_ray update_pose (ms) | uni_ray trace (ms) | mjw brute pose (ms) | mjw brute trace (ms) | mjw bvh pose+refit (ms) | mjw bvh trace (ms) | speedup vs brute | speedup vs bvh |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0974 | 0.5682 | 0.0529 | 6.0996 | 0.0800 | 0.3310 | 10.74x | 0.58x |
| 256 | 512 | 0.3112 | 4.2556 | 0.0574 | 111.0185 | 0.0874 | 1.1938 | 26.09x | 0.28x |
| 1024 | 512 | 0.3493 | 25.3768 | 0.0639 | 456.3015 | 0.1014 | 4.1022 | 17.98x | 0.16x |

| num_envs | num_rays | uni_ray kernel (ms) | mjw brute kernel (ms) | mjw bvh kernel (ms) | kernel ratio brute/uni_ray | kernel ratio bvh/uni_ray |
|---|---|---|---|---|---|---|
| 64 | 128 | 0.3637 | 6.4739 | 0.3082 | 17.80x | 0.85x |
| 256 | 512 | 0.9300 | 110.3361 | 0.8948 | 118.65x | 0.96x |
| 1024 | 512 | 3.2651 | 444.0060 | 3.2473 | 135.98x | 0.99x |

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
- **Hfield**: against the only semantically correct mujoco-warp mode
  (brute), uni_ray's kernel is 1.75x faster at 64x128 and 9.9-11.2x faster
  at batch sizes — uni_ray traces the triangulated terrain through its
  per-mesh BVH while the brute kernel walks every terrain triangle per ray.
  mujoco-warp's BVH mode cannot be used for hfields at all (top surface
  only; see the sanity-check finding).
- **Dense mesh**: the brute kernel collapses under 24 x 2208-triangle geoms
  (444 ms at 1024x512, 136x uni_ray's kernel); both BVH kernels stay flat
  (3.2-3.3 ms) at rough parity (0.85-0.99x). Even end-to-end, uni_ray's
  staging-dominated public `trace` beats brute by 10.7-26.1x here.
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

## Rebuild-path benchmark (uni_ray-only)

`benchmarks/bench_rebuild.py` times the explicit cold-path scene replacement
(`WarpRayCaster.rebuild`, added for runtime geometry randomization) on the
dense-mesh scene above at a fixed 256x512 batch — there is no mujoco-warp
counterpart to compare against. Measured on the same idle RTX 4090 (15
cold-path iterations after warmup, 50 trace iterations):

| segment | mean (ms) |
|---|---|
| build_collision_description (host, changed mesh) | 0.0573 |
| rebuild(collision=...) total | 1.6615 |
| &nbsp;&nbsp;of which wp.Mesh builds (instrumented) | 0.1190 |
| &nbsp;&nbsp;of which scene arrays + scene BVH (remainder) | 1.5426 |
| rebuild(scene=...) total (geom sizes, mesh data reused) | 1.5934 |
| trace before rebuild | 4.2443 |
| trace after rebuild | 3.9308 |
| amortized (rebuild + 100 traces) / 100 | 4.2610 |

Post-rebuild trace check: with the same descriptor rebuilt and the same
poses re-applied, trace outputs are bitwise identical to the pre-rebuild
trace (max |dt| = 0.0), and trace cost is unchanged (3.93 ms vs 4.24 ms,
within run-to-run noise) — the rebuild is strictly a cold path.

Reading the table:

- A full `rebuild(collision=...)` with a changed 2208-triangle mesh asset
  costs ~1.7 ms; the wp.Mesh rebuilds (incl. warp's per-mesh BVH builds) are
  only ~0.12 ms of it — the rest is scene device-array reallocation, local
  AABB recompute, and the scene wp.Bvh build. `rebuild(scene=...)` (geom
  size/pose randomization reusing bound mesh data) costs about the same,
  confirming the mesh rebuild is not the dominant term at this scene size.
- The host-side descriptor build (`build_collision_description`) is
  negligible (~0.06 ms) for mesh-only changes.
- Amortized over a "rebuild every 100 traces" domain-randomization cycle,
  the rebuild adds ~0.017 ms per trace (~0.4% of the staging-dominated
  trace cost at this batch shape).
