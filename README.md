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
  ellipsoid) and static meshes (`wp.Mesh` with its internal BVH). Height
  fields, runtime geometry randomization, and dynamic meshes fail closed with
  a clear unsupported-capability error.
- Results come back through explicit host readback into persistent NumPy
  buffers; there is intentionally no zero-copy/device-output claim.
- `uni_ray.mjbatch.build_collision_description(mj_model)` reads a
  `mujoco.MjModel` exactly once on the cold path and produces the immutable
  `RaySceneDescription` plus mesh collision descriptor.

## Correctness and performance evidence

[docs/conformance.md](docs/conformance.md) records the conformance sweep against
`mujoco.mj_ray` (primitives and static meshes), the pinned known-semantics
resolutions, the hot-path audit approach, and segmented CPU/GPU timings from
`benchmarks/bench_mjbatch.py`.

## Installation

The package depends on `unisim-core` (the ray-query contract) and NumPy.
Warp and MuJoCo are optional extras, imported lazily:

```bash
uv sync --extra warp --extra mujoco   # or just `uv sync` (dev group includes both)
```

For local development against a sibling UniSim checkout, `pyproject.toml`
carries `[tool.uv.sources] unisim-core = { path = "../unisim", editable = true }`.

## Usage

```python
import mujoco
import numpy as np
import uni_ray
from uni_ray.mjbatch import build_collision_description

model = mujoco.MjModel.from_xml_string(xml)
collision = build_collision_description(model)   # cold path, reads MjModel once

caster = uni_ray.create_ray_caster(num_envs=4, num_rays=64, collision=collision)
caster.materialize(collision.scene)
caster.update_pose(body_pos, body_quat)          # (rows, nbody, 3) / (rows, nbody, 4) wxyz
result = caster.trace(origins, directions, max_distance=10.0)
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
across calls. `hit_point` and `body_id` are freshly computed per call.

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
