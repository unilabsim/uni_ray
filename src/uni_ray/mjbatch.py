"""Cold-path ``mujoco.MjModel`` → collision descriptor builder.

This module is the only place uni_ray touches MuJoCo, and it does so exactly
once per scene: it reads geom types, sizes, local poses, body ids, and static
mesh vertex/face data out of an ``MjModel`` and freezes them into the
backend-neutral :class:`unisim.ray_query.RaySceneDescription` plus a mesh
collision descriptor (:class:`MjBatchCollision`) consumed by
``uni_ray.create_ray_caster(..., collision=...)``. The ``mujoco`` import is
lazy and confined to this module; the hot path (``update_pose``/``trace``)
never sees an ``MjModel`` or ``MjData``.

The MjModel extraction mirrors the init-time cold path of MuJoCo-LiDAR's
``MjLidarWarp`` (https://github.com/discoverse-dev/MuJoCo-LiDAR, MIT License,
Copyright (c) 2025 Yufei Jia; see NOTICE): geom arrays are sliced out of the
model and meshes are deduplicated by ``(geom_type, data_id)``. Unlike that
wrapper, hfield geoms fail closed instead of being triangulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from unisim.errors import UnsupportedCapabilityError
from unisim.optional import OptionalDependencyError
from unisim.ray_query import RayGeomType, RaySceneDescription

# MuJoCo mjtGeom numbering (stable across the mujoco>=3.2 line) mapped onto the
# contract geometry kinds. HFIELD (1) is deliberately absent: it fails closed.
_MJ_TO_CONTRACT_TYPES = {
    0: RayGeomType.PLANE,
    2: RayGeomType.SPHERE,
    3: RayGeomType.CAPSULE,
    4: RayGeomType.ELLIPSOID,
    5: RayGeomType.CYLINDER,
    6: RayGeomType.BOX,
    7: RayGeomType.MESH,
}
_MJ_GEOM_HFIELD = 1
_MJ_GEOM_MESH = 7


@dataclass(frozen=True)
class MeshData:
    """One deduplicated static mesh in its geom-local frame."""

    points: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=np.float32, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("mesh points must be a finite array of shape (V, 3)")
        indices = np.array(self.indices, dtype=np.int32, copy=True)
        if indices.ndim != 1 or indices.size % 3 != 0:
            raise ValueError("mesh indices must be a flat triangle array of shape (3F,)")
        if indices.size and (indices.min() < 0 or indices.max() >= points.shape[0]):
            raise ValueError("mesh indices are outside [0, V)")
        points.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True)
class MjBatchCollision:
    """Immutable collision descriptor consumed by ``WarpRayCaster``.

    ``scene`` is the backend-neutral descriptor handed to
    :meth:`unisim.ray_query.RayCaster.materialize`; ``meshes`` and
    ``geom_mesh_ids`` carry the static mesh geometry the scene descriptor
    cannot express (``geom_mesh_ids[g]`` indexes ``meshes`` for mesh geoms and
    is ``-1`` otherwise).
    """

    scene: RaySceneDescription
    meshes: tuple[MeshData, ...]
    geom_mesh_ids: np.ndarray

    def __post_init__(self) -> None:
        geom_mesh_ids = np.array(self.geom_mesh_ids, dtype=np.intp, copy=True)
        if geom_mesh_ids.shape != (self.scene.num_geoms,):
            raise ValueError(
                f"geom_mesh_ids must have shape ({self.scene.num_geoms},), "
                f"got {geom_mesh_ids.shape}"
            )
        for geom_id, (geom_type, mesh_id) in enumerate(
            zip(self.scene.geom_types, geom_mesh_ids)
        ):
            if geom_type == RayGeomType.MESH:
                if mesh_id < 0 or mesh_id >= len(self.meshes):
                    raise ValueError(f"mesh geom {geom_id} has invalid mesh id {mesh_id}")
            elif mesh_id != -1:
                raise ValueError(f"non-mesh geom {geom_id} must have mesh id -1")
        geom_mesh_ids.setflags(write=False)
        object.__setattr__(self, "geom_mesh_ids", geom_mesh_ids)


def build_collision_description(mj_model: Any) -> MjBatchCollision:
    """Read a ``mujoco.MjModel`` once and freeze it into a collision descriptor.

    Plane sizes are zeroed so the caster honors the contract's infinite-plane
    semantics. Hfield geoms, unknown geom types, and any geometry randomization
    fail closed with :class:`UnsupportedCapabilityError`.
    """
    try:
        import mujoco
    except ImportError as error:
        raise OptionalDependencyError(
            "uni_ray.mjbatch requires the optional dependency 'mujoco>=3.2.0', "
            "which is not installed; install the 'uni-ray[mujoco]' extra to build "
            "collision descriptors from an MjModel"
        ) from error
    if not isinstance(mj_model, mujoco.MjModel):
        raise TypeError(f"mj_model must be a mujoco.MjModel, got {type(mj_model).__name__}")

    geom_type_codes = np.asarray(mj_model.geom_type, dtype=np.int64)
    if np.any(geom_type_codes == _MJ_GEOM_HFIELD):
        raise UnsupportedCapabilityError(
            "uni_ray mjbatch does not support hfield geoms; triangulate the height field "
            "into a static mesh asset on the cold path or remove the hfield geom"
        )
    unknown = sorted(set(geom_type_codes.tolist()) - set(_MJ_TO_CONTRACT_TYPES))
    if unknown:
        raise UnsupportedCapabilityError(
            f"uni_ray mjbatch does not support MuJoCo geom type codes {unknown}; "
            "supported geoms are plane, sphere, capsule, ellipsoid, cylinder, box, "
            "and static mesh"
        )

    geom_types = tuple(_MJ_TO_CONTRACT_TYPES[int(code)] for code in geom_type_codes)
    geom_sizes = np.array(mj_model.geom_size, dtype=np.float64, copy=True)
    # The contract plane is the infinite local z = 0 plane and ignores sizes;
    # MuJoCo's plane size is a rendering grid, so zero it out.
    geom_sizes[geom_type_codes == 0] = 0.0

    meshes, geom_mesh_ids = _extract_meshes(mj_model, geom_type_codes)

    scene = RaySceneDescription(
        num_bodies=int(mj_model.nbody),
        geom_types=geom_types,
        geom_sizes=geom_sizes,
        geom_local_pos=np.array(mj_model.geom_pos, dtype=np.float64, copy=True),
        geom_local_quat=np.array(mj_model.geom_quat, dtype=np.float64, copy=True),
        geom_body_ids=np.array(mj_model.geom_bodyid, dtype=np.intp, copy=True),
    )
    return MjBatchCollision(scene=scene, meshes=meshes, geom_mesh_ids=geom_mesh_ids)


def _extract_meshes(
    mj_model: Any, geom_type_codes: np.ndarray
) -> tuple[tuple[MeshData, ...], np.ndarray]:
    """Slice static mesh vertex/face data out of the model, deduplicated by data id."""
    geom_mesh_ids = np.full(mj_model.ngeom, -1, dtype=np.intp)
    mesh_index_by_data_id: dict[int, int] = {}
    meshes: list[MeshData] = []
    for geom_id, data_id in enumerate(np.asarray(mj_model.geom_dataid, dtype=np.int64)):
        if geom_type_codes[geom_id] != _MJ_GEOM_MESH or data_id < 0:
            continue
        if int(data_id) not in mesh_index_by_data_id:
            vert_adr = int(mj_model.mesh_vertadr[data_id])
            vert_num = int(mj_model.mesh_vertnum[data_id])
            face_adr = int(mj_model.mesh_faceadr[data_id])
            face_num = int(mj_model.mesh_facenum[data_id])
            points = mj_model.mesh_vert[vert_adr : vert_adr + vert_num].astype(np.float32)
            faces = mj_model.mesh_face[face_adr : face_adr + face_num].reshape(-1)
            mesh_index_by_data_id[int(data_id)] = len(meshes)
            meshes.append(MeshData(points=points, indices=faces.astype(np.int32)))
        geom_mesh_ids[geom_id] = mesh_index_by_data_id[int(data_id)]
    return tuple(meshes), geom_mesh_ids


__all__ = ["MeshData", "MjBatchCollision", "build_collision_description"]
