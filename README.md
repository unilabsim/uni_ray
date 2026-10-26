# uni_ray

`uni_ray` is the Warp-accelerated ray caster plugin for the
[UniSim](https://github.com/unilabsim/unisim) ray-query contract
(`unisim.ray_query`). It is discovered and constructed lazily through
`unisim.factory.create_ray_caster("uni_ray", ...)` and implements the
`RayCaster` lifecycle: `materialize` → `update_pose` → `trace` → `close`.

## Features

- Batched tracing over a fixed `(num_envs, num_rays)` shape with per-env BVH
  groups and persistent device/host buffers (no unbounded per-call allocation).
- Pose sync: a dedicated Warp kernel rebuilds geom world poses from body
  positions and unit `wxyz` quaternions (`mj_kinematics` semantics), then
  refits the BVH; the hot path never parses XML, resolves names, or touches
  `mujoco.MjData`.
- Geometry: analytic primitives (plane, sphere, box, cylinder, capsule,
  ellipsoid), static meshes (`wp.Mesh` with its internal BVH), and height
  fields — hfield geoms are triangulated once on the cold path into the
  closed solid `mujoco.mj_ray` intersects (top surface, skirt walls, base
  box) and bound as static meshes. Per-frame dynamic meshes fail closed with
  a clear unsupported-capability error.
- Public collision descriptor (#472): `create_ray_caster(..., collision=RayCollisionDescription)`
  consumes UniSim's detached mesh topology, geom/mesh identities, tags, and
  body ancestry. Portable callers never need a backend-specific builder.
- Rich queries (#472): one nearest-valid-hit traversal supports call-static
  geom/body/tag filters, an inclusive `min_distance`, and optional world-space
  against-ray normals on both host and tensor profiles.
- Explicit cold-path rebuild: `WarpRayCaster.rebuild(...)` (or the
  `uni_ray.mjbatch.rebuild_caster_from_model` convenience) re-binds geom
  sizes/local poses and mesh/hfield data from an updated descriptor or
  MjModel without recreating the caster, so runtime geometry randomization
  is usable; the hot path keeps its body-pose-only contract.
- The host profile reads back into persistent NumPy buffers; the tensor profile
  returns persistent caster-owned Torch views on the caller's device and stream.
- `uni_ray.mjbatch.build_collision_description(mj_model)` reads a
  `mujoco.MjModel` exactly once on the cold path and produces the immutable
  public geometry-only `RayCollisionDescription`. It remains a profile-specific
  convenience for direct model access; portable perception callers use the
  backend's tag/ancestry-rich public collision descriptor instead.
- mjwarp profile (#300): `uni_ray.mjwarp.build_collision_description` sources
  the same descriptor from a compiled `mujoco_warp.Model` through the shared
  profile-neutral build, plus `uni_ray.mjwarp.create_ray_caster`
  (build + bind) and `rebuild_caster_from_mjwarp` conveniences — see
  [docs/backends.md](docs/backends.md) for usage and the pose-sync vs
  device-pose boundary evaluation.
- motrix profile (#301): `uni_ray.motrix.build_collision_description` sources
  the same descriptor from a `motrixsim.SceneModel` (pose-sync profile), with
  a full capability audit (exact/approximate/unsupported per geometry kind)
  in [docs/backends.md](docs/backends.md). `motrixsim-core` ships cp310-only
  wheels — install the `motrix` extra in a Python 3.10 environment.

## Correctness and performance evidence

[docs/conformance.md](docs/conformance.md) records the conformance sweep against
`mujoco.mj_ray` (primitives and static meshes), the pinned known-semantics
resolutions, the hot-path audit approach, and segmented CPU/GPU timings from
`benchmarks/bench_mjbatch.py`.
[docs/benchmark-mjwarp.md](docs/benchmark-mjwarp.md) compares batch ray
tracing against mujoco-warp's public `rays` API (brute-force and BVH modes)
from `benchmarks/bench_vs_mjwarp.py`, and records the uni_ray-only
rebuild-path timings from `benchmarks/bench_rebuild.py`.
`benchmarks/bench_rich_queries.py` records fixed-profile kernel and end-to-end
tensor timings for primitives, shared meshes, and hfields with normal output and
filter density varied.

## Installation

The package depends on `unisim-core>=2.2.0` (the ray-query contract) and
NumPy; the public ray-query contract is introduced by that UniSim release.
Warp and MuJoCo are optional extras, imported lazily:

```bash
uv sync --extra warp --extra mujoco   # or just `uv sync` (dev group includes both)
```

For local development against a sibling UniSim checkout, `pyproject.toml`
carries `[tool.uv.sources] unisim-core = { path = "../unisim", editable = true }`;
that checkout may temporarily lead the published UniSim release.

## Usage

```python
import numpy as np
import uni_ray
from unisim.ray_query import RayTraceFilter, RayTraceOutputs

collision = backend.get_ray_collision_description()  # public detached descriptor
ray_filter = RayTraceFilter(excluded_body_ids=(attachment_body_id,))

caster = uni_ray.create_ray_caster(num_envs=4, num_rays=64, collision=collision)
caster.materialize(collision.scene)
caster.update_pose(body_pos, body_quat)  # (rows, nbody, 3) / (rows, nbody, 4) wxyz
result = caster.trace(
    origins,
    directions,
    max_distance=10.0,
    min_distance=0.1,
    ray_filter=ray_filter,
    outputs=RayTraceOutputs(normal=True),
)
print(result.distance, result.hit)
caster.close()
```

The same caster is reachable from UniSim:

```python
from unisim.factory import create_ray_caster

caster = create_ray_caster("uni_ray", num_envs=4, num_rays=64, collision=collision)
```

`distance`, `hit`, and `geom_id` result arrays are views into caster-owned
host buffers reused by the next `trace` call; copy them if you retain results
across calls. Normals are world-space face/side normals oriented against the
ray. `min_distance` is inclusive, `max_distance` is exclusive, and a filtered
or too-near hit is a normal miss.

The same caster also serves the UniSim Torch-native tensor profile (#462):
`bind_pose_layout` fixes the public body-name order on the cold path,
`update_pose_tensor` aliases caller-owned float32 packet-strided pose views and
launches pose/AABB/BVH work on the caller's current stream, and `trace_tensor`
returns persistent Torch result views with device-side `hit_point`, `normal`,
`geom_id`, and `body_id`, plus static filters and `min_distance`. Torch is a
test/development dependency here and is imported lazily
inside the tensor methods; plugin imports remain Torch-free. The tensor profile
requires the caster device (`cpu` or `cuda:<index>`), float32 pose/ray tensors,
and fully contiguous ray and `env_ids` tensors. Pose packet views may have
outer-row strides, provided all strides are positive and non-overlapping and
the component axis has stride one.

The shared `(num_rays, 3)` ray profile is live: every call re-stages the values
currently in the caller's tensors with a device-side fan-out on the caller's
stream. Staging storage is allocated once. Per-environment `(rows, num_rays, 3)`
inputs likewise read the caller's tensors directly on every call. Geometry is
the only ray-scene input that requires the explicit cold-path `rebuild()`.

On CUDA, invalid `env_ids` are validated by a bounded device kernel rather than
a per-call readback. The offending trace may therefore return before its
`ValueError` is raised; the error is surfaced at the next cold boundary
(`close()` or `rebuild()`), together with a message identifying the prior trace.
Duplicate and out-of-range IDs still fail closed and reset the validation
marker; valid traces do not synchronize.

To randomize geometry between episodes without recreating the caster, mutate
the model and rebuild on the cold path (poses reset to identity, so call
`update_pose` again before tracing):

```python
from uni_ray.mjbatch import rebuild_caster_from_model

model.geom_size[geom_id] = [0.5, 0.4, 0.3]  # geom size / local pose / mesh edits
rebuild_caster_from_model(caster, model)  # re-triangulates, re-uploads, rebuilds BVH
```

## Examples

`examples/visualize_pointcloud.py` renders the caster's world-frame hit-point
cloud with [viser](https://github.com/nerfstudio-project/viser): a small scene
(ground, hfield terrain, static mesh, and a freejoint body carrying
sphere/box/capsule geoms) is traced by a spinning LiDAR-like ring fan per env,
and each frame's hit points stream to the browser:

```bash
uv sync --extra warp --extra mujoco --extra viz
uv run python examples/visualize_pointcloud.py --num-envs 4 --num-rays 512
```

Open the printed URL (http://localhost:8080 by default). `hit_point` is already
world-frame (rays go in world-frame, `hit_point = origin + distance *
direction`), so the demo only masks out misses.

## Development

```bash
make sync        # uv sync --locked
make lint        # uv run ruff check .
make typecheck   # uv run mypy src/uni_ray
make test        # uv run pytest -q
make check       # all of the above
```

## Attribution

The analytic ray-primitive intersection functions and the BVH/AABB trace
kernels in `src/uni_ray/geometry.py` and `src/uni_ray/kernels.py` are copied
(with adaptations) from
[MuJoCo-LiDAR](https://github.com/discoverse-dev/MuJoCo-LiDAR),
Copyright (c) 2025 Yufei Jia, MIT licensed. See [NOTICE](NOTICE).
