# Backend profiles: mjbatch, mjwarp, and motrix (#300, #301)

uni_ray ships three cold-path profiles that produce the **same** immutable
collision descriptor (`MjBatchCollision`) and bind the **same** caster,
pose-sync contract, and kernels:

| Profile | Builder | Input |
|---|---|---|
| mjbatch | `uni_ray.mjbatch.build_collision_description` | `mujoco.MjModel` |
| mjwarp | `uni_ray.mjwarp.build_collision_description` | `mujoco_warp.Model` (e.g. from `mujoco_warp.put_model`) |
| motrix | `uni_ray.motrix.build_collision_description` | `motrixsim.SceneModel` (e.g. from `motrixsim.load_model` / `load_mjcf_str`) |

All builders snapshot their model's fields into a shared plain-NumPy
extraction (`uni_ray.mjbatch._ModelSceneData`) and run one profile-neutral
descriptor build (`_build_collision_description`), so geom typing, plane-size
zeroing, mesh dedup, and hfield triangulation behave identically. Each
profile's runtime import (`mujoco` / `mujoco_warp` / `motrixsim`) is lazy and
confined to its module; no engine types appear in the returned objects
(plain NumPy descriptor, plain `unisim.ray_query.RayCaster`).

## mjwarp profile usage

```python
import mujoco
import mujoco_warp as mjw

mj_model = mujoco.MjModel.from_xml_string(SCENE_XML)
mjw_model = mjw.put_model(mj_model)

# Option A: descriptor + the UniSim plugin factory (same as mjbatch).
from uni_ray.mjwarp import build_collision_description
from unisim.factory import create_ray_caster

collision = build_collision_description(mjw_model)
caster = create_ray_caster("uni_ray", num_envs=256, num_rays=512, collision=collision)
caster.materialize(collision.scene)

# Option B: one-call convenience (build + create + materialize).
from uni_ray.mjwarp import create_ray_caster as create_mjwarp_caster

caster = create_mjwarp_caster(mjw_model, num_envs=256, num_rays=512)

# Geometry randomization re-put_model's the mutated MjModel and goes through
# the explicit cold-path rebuild.
from uni_ray.mjwarp import rebuild_caster_from_mjwarp

rebuild_caster_from_mjwarp(caster, mjw.put_model(mutated_mj_model))
```

`update_pose`/`trace`/`rebuild` semantics are exactly the mjbatch ones; the
hot path receives validated host NumPy body pos/quat (wxyz) and never sees an
`MjModel`, `mjwarp.Model`, or `mjwarp.Data`.

## Parity with mjbatch

`tests/test_mjwarp.py` builds the same static scene (plane + hfield + two
mesh geoms + primitive body) through both profiles and traces identical poses
and rays: hit masks and geom ids are identical and distances agree within
**ATOL = 1e-4** — worst observed |Δt| **0.0** (bitwise identical on the test
scene). The only descriptor difference between profiles is float32 rounding
of the model fields the mjwarp side sources (sizes/local poses ~3e-8), which
the caster's own float32 device conversion absorbs. Mesh data, including the
triangulated hfield, is bitwise identical across profiles.

## Pose-sync vs backend-owned device pose (phase-1 boundary)

Phase 1 implements **host pose-sync only**: poses cross the contract as host
NumPy through `update_pose`, identical to mjbatch. An mjwarp deployment also
has `mujoco_warp.Data.xpos`/`xquat` living on device after FK; consuming
those directly would avoid the host round trip, but the contract has no
device-pose capability today (`supports_device_output` is reserved;
`update_pose` accepts host arrays only, and requesting `device_output=True`
fails closed on the capability surface).

Measured probe (`benchmarks/bench_device_pose_probe.py`, dense 13-body scene,
RTX 4090, 100 iterations): reading `xpos`+`xquat` back from an mjwarp `Data`
(D2H into preallocated host buffers, sync-scoped) costs

| num_envs | mjw xpos+xquat D2H (ms) | uni_ray update_pose (ms) | D2H / update_pose |
|---|---|---|---|
| 64 | 0.0144 | 0.1120 | 0.13x |
| 256 | 0.0218 | 0.1551 | 0.14x |
| 1024 | 0.0410 | 0.3745 | 0.11x |

Evaluation:

- The D2H read itself is cheap (14-41 µs at 64-1024 envs) but **additive**:
  without a device-pose contract capability, device poses must make a
  device→host→device round trip (D2H read + full `update_pose` staging/H2D
  on top), adding a sync point per frame for zero kernel-side benefit.
- The real win of backend-owned device poses is skipping the host entirely,
  which requires contract work (a pose-source / device-pose capability and
  ADR) before any backend can expose it. That is future ADR work; the
  phase-1 boundary is documented here so the adapter does not grow a
  backend-private side channel.
- Meanwhile the host path is not the bottleneck it appears: `update_pose`
  cost is dominated by contract-mandated host validation/staging, not by the
  H2D copy itself (see the segmented timings in docs/conformance.md).

## MotrixSim profile (#301)

`motrixsim-core==0.10.1` publishes **cp310-only wheels**, so the `motrix`
extra is marker-restricted to Python 3.10 and kept out of the dev group. Use
a dedicated 3.10 environment (the default `.venv` is untouched):

```bash
uv venv .venv-motrix --python 3.10
UV_PROJECT_ENVIRONMENT=.venv-motrix uv sync --extra motrix
UV_PROJECT_ENVIRONMENT=.venv-motrix uv run --no-sync pytest tests/ -q
```

Usage mirrors the mjwarp profile; the input is a loaded `motrixsim.SceneModel`:

```python
import motrixsim as ms
from uni_ray.motrix import build_collision_description, create_ray_caster

model = ms.load_mjcf_str(SCENE_XML)          # or ms.load_model(path)
collision = build_collision_description(model)  # cold path, plain NumPy out
caster = create_ray_caster(model, num_envs=256, num_rays=512)  # build + bind
```

### Consumer pose mapping (pose-sync profile)

The hot path is host pose-sync only, exactly like the other profiles —
nothing Motrix-side runs per frame. Consumers read link poses from their
`SceneData` and adapt them once:

```python
data = ms.SceneData(model)
link_poses = np.asarray(model.get_link_poses(data))   # (num_links, 7), xyz + xyzw
body_pos = np.concatenate([np.zeros((1, 3)), link_poses[:, :3]])[None]
link_quat_wxyz = link_poses[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
body_quat = np.concatenate([[[1.0, 0.0, 0.0, 0.0]], link_quat_wxyz])[None]
caster.update_pose(body_pos, body_quat)               # world body = row 0
```

MotrixSim links are the rigid bodies; the world is not a link, so the
descriptor numbers the world body 0 and link `i` as body `i + 1`. Note
`get_link_poses` at init returns the origin for a free joint — MotrixSim
does not seed the joint from the XML body `pos` the way MuJoCo's `qpos0`
does (verified on 0.10.1).

### Capability audit

Each row is justified by a test in `tests/test_motrix.py` or a probed SDK
limit (motrixsim-core 0.10.1, probed on RTX 4090 / Linux x86_64):

| Geometry kind | Support | Evidence / notes |
|---|---|---|
| Infinite plane (`Shape.InfinitePlane`) | **exact** | Maps to the contract's infinite double-sided plane (size zeroed); trace test |
| Sphere / cuboid / capsule / cylinder / ellipsoid | **exact** | Same size conventions as MuJoCo (radius, half extents, `(radius, half_length)`, radii); capsule z-axis alignment verified via `get_local_aabb`; descriptor + trace tests |
| Hfield (`Shape.HField`) | **approximate** | Triangulated from the compiled `HField.height_matrix` (verified exact at grid vertices and cell edges end-to-end). Deviations: per-cell triangulation diagonal is unverifiable (no public batch raycast in the SDK; `sample_height` interpolates bilinearly, so cell interiors are smooth there); MotrixSim's solid extends to a very deep z bound (`bound` z low = −1000) vs our no-base-box mapping — rays entering the side walls below z = 0 can differ; above-surface rays match |
| Mesh (`Shape.Mesh`) | **unsupported, fails closed** | motrixsim-core 0.10.1 exposes no mesh vertex/face introspection (`GeomMesh` offers only `mesh_name`/scale/transform/AABB; `SceneModel` has no mesh data getter). Building from the AABB would silently change the collision world, so the descriptor build raises `UnsupportedCapabilityError` (test-pinned) |
| Finite plane (`Shape.Plane`) | **unsupported, fails closed** | No contract equivalent (the contract plane is infinite) |
| SDF (`Shape.Sdf`) | **unsupported, fails closed** | No contract equivalent |
| Profile: pose-sync | **implemented** | Host NumPy body pos/quat through `update_pose`, shared caster; link-pose mapping above is test-pinned |
| Profile: static-map | covered by pose-sync | A materialized caster without `update_pose` traces the identity-pose static scene |
| Profile: native | **unsupported** | No general batch raycast API in motrixsim-core 0.10.1 (`TerrainScanner` is terrain-only) |
| Outputs | as uni_ray core | distance/hit/geom_id/body_id/hit_point exact; normals and device output unsupported (fail closed), same as every profile |

### MuJoCo-vs-MotrixSim semantic differences found

1. **Hfield height normalization**: MuJoCo computes heights as
   `elevation * z_scale` raw; MotrixSim renormalizes to
   `(elevation − min) / (max − min) * z_scale` (probed: elevation range
   1..4 with z scale 0.5 yields heights 0..0.5). The adapter consumes the
   compiled `height_matrix` directly, so it matches the **MotrixSim**
   collision world — scenes sharing one MJCF produce *different* terrains in
   the two engines, by the engines' own choice.
2. **Hfield solid depth**: MuJoCo's solid has a finite base box
   (`−base_depth`); MotrixSim's bound reaches z = −1000 (effectively
   infinite). Our mapping builds the top surface + skirt walls down to
   z = 0 and no base box (marked approximate).
3. **Quaternion convention**: MotrixSim poses are `(x, y, z, qx, qy, qz,
   qw)` (xyzw); MuJoCo and the contract are wxyz. Converted on the cold
   path (descriptor) and required in the consumer pose mapping above.
4. **Bodies vs links**: MotrixSim's rigid bodies are links (the world is
   not one); `num_bodies = num_links + 1` with world at index 0.
5. **Free-joint init**: `get_link_poses` at init returns the origin for a
   free joint; MotrixSim does not seed from the XML body `pos` (MuJoCo's
   `qpos0` does).
6. **Geom naming**: MJCF geom names are optional and may be `None`; the
   fail-closed diagnostics fall back to the geom index.

### MotrixSim environment notes

- `motrixsim-core==0.10.1` initializes fine on this machine (RTX 4090,
  driver 13.0); scene loading and descriptor extraction are host-side, and
  uni_ray's caster runs on warp as usual — no MotrixSim GPU requirement was
  hit.
- The adapter never touches `SceneData` on the hot path; `SceneData` appears
  only in the *consumer* pose mapping (and in tests).

## Fail-closed surface (shared with mjbatch)
- Unknown geom type codes raise `UnsupportedCapabilityError` at descriptor
  build (hfields are supported via cold-path triangulation).
- `device_output=True` requests fail closed (`supports_device_output=False`;
  the caster constructor rejects the kwarg).
- Dynamic geometry has no hot-path API: a second `materialize` fails, and
  mesh/hfield/geom-data changes require the explicit cold-path `rebuild`
  (`rebuild_caster_from_mjwarp` for the mjwarp profile).
- `mujoco_warp` missing → `OptionalDependencyError` naming the extra;
  `import uni_ray`, `uni_ray.mjwarp`, and factory discovery are unaffected.

## Known limitations

- The mjwarp profile reads the **static model only**; it never touches
  `mjwarp.Data` — per-frame physics state enters exclusively through
  `update_pose` host poses.
- `mujoco_warp.Model` carries no per-mesh face count; the adapter derives it
  from `mesh_faceadr`/`nmeshface` (verified against `MjModel` in tests).
- Model fields are sourced in float32 on the mjwarp profile (vs float64 from
  `MjModel`); observable descriptor difference is ≤ ~3e-8 and below the
  caster's own float32 device precision.
