# Conformance and correctness evidence (#299)

This page records the correctness sweeps, contract semantics, hot-path
audits, and segmented timings for the mjbatch Warp ray caster
(`WarpRayCaster`). Everything listed here is enforced by tests under
`tests/` or reproducible via `benchmarks/bench_mjbatch.py`.

## Numeric tolerances

The caster computes in float32 on device; the CPU reference
(`mujoco.mj_ray`) is float64. All sweeps use the fixed tolerance
**`ATOL = 1e-4`** on hit distances, with strict hit/miss and nearest-geom-id
agreement (a geom-id mismatch is only accepted when both backends report tied
distances within tolerance).

| Sweep | Rays | Worst observed \|Δt\| | Tolerance |
|---|---|---|---|
| Primitive scene, 6 random freejoint poses × 2 envs (`tests/test_conformance_primitives.py`) | 6144 | 5.16e-05 | 1e-4 |
| Capsule/cylinder rotated axes, 2000 random rays (mirrors MuJoCo-LiDAR's `test_warp_backend_capsule_cylinder_match_cpu`) | 2000 (73 hits, misses agree) | 4.31e-06 | 1e-4 |
| Static meshes (triangulated UV sphere + tetrahedron, `tests/test_conformance_mesh.py`) | 1024 (654 hits, misses agree) | 1.72e-06 | 1e-4 |

Measured on warp-lang 1.17.0 / mujoco 3.14.0 / RTX 4090.

## CPU reference adjustments

`mujoco.mj_ray` is not exactly the contract reference, so the sweeps compare
against a contract-faithful CPU reference (`_cpu_reference` in
`tests/test_conformance_primitives.py`):

- `mj_ray` has no cutoff; hits beyond `max_distance` are misses under the
  contract and are clipped accordingly in the reference.
- `mj_ray` only reports plane hits approached from the upper (+normal) side
  and treats a nonzero plane size as a finite grid. The contract plane is the
  **infinite, double-sided** local `z = 0` plane (as in `FakeRayCaster`), so
  the reference adds an analytic double-sided plane for world-attached plane
  geoms.

## Known-semantics resolutions

Pinned by `tests/test_known_semantics.py`:

1. **Rays originating inside a primitive** report the forward *exit*
   distance, matching `mujoco.mj_ray`. uni_ray deliberately deviates from the
   upstream MuJoCo-LiDAR kernels here (they reported 0.0): sphere/box/
   ellipsoid use the far root, and the capsule is intersected as the cylinder
   side plus cap spheres restricted to their exposed hemispheres
   (`|z| >= half_length`) — the upstream flat cylinder caps are interior
   planes of a capsule, and unclamped cap-sphere roots report interior points
   for inside-origin rays.
2. **Planes are infinite and double-sided** per the contract, even when a
   `RaySceneDescription` carries a nonzero plane size (sizes are zeroed when
   the device descriptor is built). `mj_ray`'s finite-grid, single-sided
   plane behavior is a documented difference, not a bug.
3. A surface exactly at `max_distance` reads as a miss: the trace kernel
   accepts `t < max_distance` strictly (pinned boundary convention).

## Lifecycle and semantics coverage

`tests/test_lifecycle.py` and `tests/test_semantics.py` pin: the exact miss
convention (`distance == max_distance`, `hit == False`, `geom_id`/`body_id`
`-1`), cutoff behavior, selected rows (`env_ids`), shared and per-env ray
profiles, batch shape validation, double-materialize/closed-caster failures,
`close()` idempotence, result-buffer view-reuse (`distance`/`hit`/`geom_id`
are views into reused caster-owned buffers; `hit_point`/`body_id` are fresh
per call), empty selections, and zero-geom scenes (all-miss).

## Hot-path audits

`tests/test_hotpath.py` proves, as executable tests:

- **No MuJoCo on the hot path**: in a subprocess, after materialize the
  `mujoco` module is removed from `sys.modules`, blocked on the meta path,
  and `uni_ray.mjbatch.build_collision_description` is poisoned to raise;
  a 10-iteration `update_pose`/`trace` loop still succeeds. No `MjData`
  access and no XML parsing can occur.
- **No descriptor/BVH/mesh rebuilds, no unbounded allocation**: `wp.Bvh` is
  constructed exactly once at materialize (and `wp.Mesh` once per unique
  mesh); spies on `wp.array`/`wp.zeros`/`wp.empty`/`wp.full`/`wp.Bvh`/
  `wp.Mesh` count **zero** calls across 25 `update_pose` + `trace`
  iterations after warmup. Bounded per-call host-side NumPy work (input
  validation/staging, `hit_point`/`body_id` computation) is by design.

## Segmented timings

`benchmarks/bench_mjbatch.py` records pose upload (host staging + H2D copy +
scatter kernel), geom pose + AABB update + BVH refit, the intersection
kernel, and host readback (D2H) separately, plus end-to-end totals. Scene:
infinite plane, static UV-sphere mesh (530 verts / 1056 faces), freejoint
body with sphere/box/capsule/cylinder/ellipsoid. Mean of 100 iterations
(GPU) / 50 iterations (CPU), rays pre-uploaded so the intersection column is
kernel-only. **This caster returns results through explicit host readback;
zero-copy throughput is not a milestone metric and is not measured.**

RTX 4090 (`--device cuda:0`, warp-lang 1.17.0):

| num_envs | num_rays | pose upload (ms) | pose+AABB+refit (ms) | intersection (ms) | host readback (ms) | update_pose total (ms) | trace total (ms) |
|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0240 | 0.0424 | 0.0429 | 0.0249 | 0.1061 | 0.2824 |
| 256 | 512 | 0.0281 | 0.0461 | 0.0804 | 0.1130 | 0.1160 | 3.1778 |
| 1024 | 512 | 0.0363 | 0.0450 | 0.1940 | 0.3545 | 0.1871 | 21.6006 |

CPU device (`--device cpu`, 50 iterations):

| num_envs | num_rays | pose upload (ms) | pose+AABB+refit (ms) | intersection (ms) | host readback (ms) | update_pose total (ms) | trace total (ms) |
|---|---|---|---|---|---|---|---|
| 64 | 128 | 0.0170 | 0.0369 | 0.9950 | 0.0068 | 0.0817 | 1.1818 |
| 256 | 512 | 0.0299 | 0.0936 | 18.6863 | 0.0310 | 0.1720 | 20.8785 |
| 1024 | 512 | 0.0771 | 0.3042 | 74.0797 | 0.1626 | 0.5105 | 94.1390 |

Reading the GPU table: the intersection kernel stays sub-millisecond even at
1024×512 rays (0.194 ms); `trace total` is dominated by host-side contract
validation and staging of the ray batch (float64 checks, broadcast
materialization), which is the contract-mandated input checking cost, not
kernel time. BVH refit is essentially free at this scene size (0.04–0.05 ms).

## CI-skip discipline

GPU/runtime tests skip cleanly when warp, mujoco, or CUDA are absent:
`tests/test_ci_skip.py` runs the whole suite in a subprocess with `warp` and
`mujoco` blocked on the meta path (raising `ModuleNotFoundError`, what a
genuinely missing package produces and what `pytest.importorskip` keys on)
and asserts the run passes with skips instead of failures. The pure contract
and import-boundary tests always run.

## Known limitations

- float32 device math; worst observed distance error 5.2e-05 on the
  primitive sweep (tolerance 1e-4).
- Normals are not computed (`supports_normal = False`); requesting them
  fails closed.
- Hfield geoms, runtime geometry randomization, and dynamic meshes fail
  closed at descriptor build / materialize time.
- Meshes are static for the caster's lifetime (wp.Mesh internal BVH built
  once at materialize).
- The default device is `cuda:0` when a CUDA device is present, else `cpu`;
  both paths are exercised (GPU for the full suite, CPU verified separately).
