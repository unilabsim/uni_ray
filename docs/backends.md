# Backend profiles: mjbatch and mjwarp (#300)

uni_ray ships two cold-path profiles that produce the **same** immutable
collision descriptor (`MjBatchCollision`) and bind the **same** caster,
pose-sync contract, and kernels:

| Profile | Builder | Input |
|---|---|---|
| mjbatch | `uni_ray.mjbatch.build_collision_description` | `mujoco.MjModel` |
| mjwarp | `uni_ray.mjwarp.build_collision_description` | `mujoco_warp.Model` (e.g. from `mujoco_warp.put_model`) |

Both builders snapshot their model's fields into a shared plain-NumPy
extraction (`uni_ray.mjbatch._ModelSceneData`) and run one profile-neutral
descriptor build (`_build_collision_description`), so geom typing, plane-size
zeroing, mesh dedup, and hfield triangulation behave identically. Each
profile's runtime import (`mujoco` / `mujoco_warp`) is lazy and confined to
its module; neither type appears in the returned objects (plain NumPy
descriptor, plain `unisim.ray_query.RayCaster`).

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
