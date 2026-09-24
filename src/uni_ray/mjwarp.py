"""Cold-path ``mujoco_warp.Model`` → collision descriptor builder (#300).

This is the mjwarp profile of the uni_ray adapter: it reads the same geom,
mesh, and hfield fields the mjbatch profile reads from a ``mujoco.MjModel``,
but out of the warp arrays of a compiled ``mujoco_warp.Model`` (e.g. from
``mujoco_warp.put_model``), converts them to plain NumPy on the cold path, and
feeds the shared, profile-neutral descriptor build in ``uni_ray.mjbatch``.
The result is the same immutable :class:`MjBatchCollision` — the caster, the
pose-sync contract, and the hot path are identical for both profiles.

The ``mujoco_warp`` import is lazy and confined to this module, exactly like
``mujoco`` in ``uni_ray.mjbatch``; nothing here leaks warp or mujoco_warp
types into the returned objects (the descriptor is plain NumPy and the caster
is a plain :class:`unisim.ray_query.RayCaster`).

Pose boundary (phase 1): the hot path receives validated host NumPy body
pos/quat through ``update_pose`` — the same contract as mjbatch. Consuming
``mujoco_warp.Data.xpos``/``xquat`` directly on device would need a
device-pose capability the contract does not have yet; see docs/backends.md
for the evaluation (including the measured D2H-vs-upload numbers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from unisim.optional import OptionalDependencyError

from .mjbatch import MjBatchCollision, _build_collision_description, _ModelSceneData

if TYPE_CHECKING:
    from unisim.ray_query import RayCaster


def _import_mujoco_warp() -> Any:
    try:
        import mujoco_warp
    except ImportError as error:
        raise OptionalDependencyError(
            "uni_ray.mjwarp requires the optional dependency 'mujoco-warp>=3.2.0', "
            "which is not installed; install it to build collision descriptors "
            "from a mujoco_warp.Model"
        ) from error
    return mujoco_warp


def build_collision_description(mjw_model: Any) -> MjBatchCollision:
    """Read a ``mujoco_warp.Model`` once and freeze it into a collision descriptor.

    Same semantics as ``uni_ray.mjbatch.build_collision_description``: plane
    sizes are zeroed (contract infinite-plane semantics), hfield geoms are
    triangulated once in the geom-local frame and bound as static meshes, and
    unknown geom types fail closed with
    :class:`unisim.errors.UnsupportedCapabilityError`. The model's warp arrays
    are copied to host NumPy here and never touched again; no
    ``mujoco_warp.Data`` is involved at any point.
    """
    mujoco_warp = _import_mujoco_warp()
    if not isinstance(mjw_model, mujoco_warp.Model):
        raise TypeError(
            f"mjw_model must be a mujoco_warp.Model, got {type(mjw_model).__name__}"
        )
    return _build_collision_description(_scene_data_from_mjwarp(mjw_model))


def _scene_data_from_mjwarp(mjw_model: Any) -> _ModelSceneData:
    """Snapshot the descriptor inputs out of a ``mujoco_warp.Model``.

    Static model fields with a leading (singleton) world dim are indexed at 0.
    ``mujoco_warp.Model`` has no per-mesh face count, so it is derived from
    ``mesh_faceadr`` and the total ``nmeshface``. Geom quaternions are wxyz on
    both profiles. Values are float32 on the mjwarp side (the mjbatch profile
    reads float64 from the MjModel), which is the only source of descriptor
    difference between the two profiles.
    """
    mesh_face_adr = np.asarray(mjw_model.mesh_faceadr.numpy(), dtype=np.int64)
    mesh_face_num = np.diff(np.append(mesh_face_adr, int(mjw_model.nmeshface)))
    return _ModelSceneData(
        num_bodies=int(mjw_model.nbody),
        geom_type_codes=np.asarray(mjw_model.geom_type.numpy(), dtype=np.int64),
        geom_sizes=np.asarray(mjw_model.geom_size.numpy()[0], dtype=np.float64),
        geom_pos=np.asarray(mjw_model.geom_pos.numpy()[0], dtype=np.float64),
        geom_quat=np.asarray(mjw_model.geom_quat.numpy()[0], dtype=np.float64),
        geom_body_ids=np.asarray(mjw_model.geom_bodyid.numpy(), dtype=np.intp),
        geom_data_ids=np.asarray(mjw_model.geom_dataid.numpy()[0], dtype=np.int64),
        mesh_vert_adr=np.asarray(mjw_model.mesh_vertadr.numpy(), dtype=np.int64),
        mesh_vert_num=np.asarray(mjw_model.mesh_vertnum.numpy(), dtype=np.int64),
        mesh_face_adr=mesh_face_adr,
        mesh_face_num=mesh_face_num,
        mesh_vert=np.asarray(mjw_model.mesh_vert.numpy(), dtype=np.float32),
        mesh_face=np.asarray(mjw_model.mesh_face.numpy(), dtype=np.int32).reshape(-1),
        hfield_nrow=np.asarray(mjw_model.hfield_nrow.numpy(), dtype=np.int64),
        hfield_ncol=np.asarray(mjw_model.hfield_ncol.numpy(), dtype=np.int64),
        hfield_adr=np.asarray(mjw_model.hfield_adr.numpy(), dtype=np.int64),
        hfield_size=np.asarray(mjw_model.hfield_size.numpy(), dtype=np.float64),
        hfield_data=np.asarray(mjw_model.hfield_data.numpy(), dtype=np.float64),
    )


def create_ray_caster(
    mjw_model: Any,
    num_envs: int = 1,
    num_rays: int = 1,
    **kwargs: Any,
) -> RayCaster:
    """Build the descriptor from ``mjw_model`` and return a bound caster.

    Convenience for the mjwarp profile: equivalent to
    ``uni_ray.create_ray_caster(num_envs, num_rays, collision=
    build_collision_description(mjw_model), **kwargs)`` followed by
    ``materialize``. The returned object is a plain
    :class:`unisim.ray_query.RayCaster`; body poses are the contract-specified
    identity until the first ``update_pose``.
    """
    import uni_ray

    collision = build_collision_description(mjw_model)
    caster = uni_ray.create_ray_caster(
        num_envs=num_envs, num_rays=num_rays, collision=collision, **kwargs
    )
    caster.materialize(collision.scene)
    return caster


def rebuild_caster_from_mjwarp(caster: Any, mjw_model: Any) -> MjBatchCollision:
    """Re-read an updated ``mujoco_warp.Model`` and rebuild ``caster``'s scene.

    The mjwarp-profile counterpart of
    ``uni_ray.mjbatch.rebuild_caster_from_model``: geometry randomization
    produces a fresh ``mujoco_warp.Model`` (``mujoco_warp.put_model`` on the
    mutated MjModel), which this binds through the explicit cold-path
    ``rebuild``. Never invoked by ``update_pose``/``trace``.
    """
    from .warp_caster import WarpRayCaster

    if not isinstance(caster, WarpRayCaster):
        raise TypeError(f"caster must be a WarpRayCaster, got {type(caster).__name__}")
    collision = build_collision_description(mjw_model)
    caster.rebuild(collision)
    return collision


__all__ = [
    "build_collision_description",
    "create_ray_caster",
    "rebuild_caster_from_mjwarp",
]
