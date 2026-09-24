"""Cold-path ``mujoco.MjModel`` → collision descriptor builder.

This module is the only place uni_ray touches MuJoCo, and it does so exactly
once per scene: it reads geom types, sizes, local poses, body ids, and static
mesh vertex/face data out of an ``MjModel`` and freezes them into the
backend-neutral :class:`unisim.ray_query.RaySceneDescription` plus a mesh
collision descriptor (:class:`MjBatchCollision`) consumed by
``uni_ray.create_ray_caster(..., collision=...)``. The ``mujoco`` import is
lazy and confined to this module; the hot path (``update_pose``/``trace``)
never sees an ``MjModel`` or ``MjData``.

The extraction is split into a plain-NumPy snapshot (:class:`_ModelSceneData`)
and a profile-neutral descriptor build (``_build_collision_description``) so
``uni_ray.mjwarp`` can feed the same build from a ``mujoco_warp.Model``
without going through an ``MjModel``.

The MjModel extraction mirrors the init-time cold path of MuJoCo-LiDAR's
``MjLidarWarp`` (https://github.com/discoverse-dev/MuJoCo-LiDAR, MIT License,
Copyright (c) 2025 Yufei Jia; see NOTICE): geom arrays are sliced out of the
model, meshes are deduplicated by ``(geom_type, data_id)``, and hfield geoms
are triangulated once in their geom-local frame (``_triangulate_hfield``,
ported from ``_build_hfield_mesh`` and extended to the full closed solid
``mj_rayHfield`` intersects: top surface, clipped skirt walls, and base box)
and registered as static meshes, matching ``mujoco.mj_ray`` hit distances up
to float32 rounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from unisim.errors import UnsupportedCapabilityError
from unisim.optional import OptionalDependencyError
from unisim.ray_query import RayGeomType, RaySceneDescription

# MuJoCo mjtGeom numbering (stable across the mujoco>=3.2 line) mapped onto the
# contract geometry kinds. HFIELD (1) is triangulated on the cold path and
# bound as a static MESH.
_MJ_TO_CONTRACT_TYPES = {
    0: RayGeomType.PLANE,
    1: RayGeomType.MESH,
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


@dataclass(frozen=True)
class _ModelSceneData:
    """Plain-NumPy snapshot of the model fields the descriptor build reads.

    Shared extraction target: ``_scene_data_from_mjmodel`` fills it from a
    ``mujoco.MjModel`` and ``uni_ray.mjwarp`` fills it from a
    ``mujoco_warp.Model``, so the descriptor build below is profile-neutral.
    """

    num_bodies: int
    geom_type_codes: np.ndarray  # (G,) int64, MuJoCo mjtGeom numbering
    geom_sizes: np.ndarray  # (G, 3) float64
    geom_pos: np.ndarray  # (G, 3) float64
    geom_quat: np.ndarray  # (G, 4) float64, wxyz
    geom_body_ids: np.ndarray  # (G,) intp
    geom_data_ids: np.ndarray  # (G,) int64
    mesh_vert_adr: np.ndarray  # (M,) int64
    mesh_vert_num: np.ndarray  # (M,) int64
    mesh_face_adr: np.ndarray  # (M,) int64
    mesh_face_num: np.ndarray  # (M,) int64
    mesh_vert: np.ndarray  # (V, 3) float32
    mesh_face: np.ndarray  # (3F,) int32, flat triangle array
    hfield_nrow: np.ndarray  # (H,) int64
    hfield_ncol: np.ndarray  # (H,) int64
    hfield_adr: np.ndarray  # (H,) int64
    hfield_size: np.ndarray  # (H, 4) float64
    hfield_data: np.ndarray  # (nhfielddata,) float64


def build_collision_description(mj_model: Any) -> MjBatchCollision:
    """Read a ``mujoco.MjModel`` once and freeze it into a collision descriptor.

    Plane sizes are zeroed so the caster honors the contract's infinite-plane
    semantics. Hfield geoms are triangulated once here (geom-local frame) and
    bound as static meshes. Unknown geom types fail closed with
    :class:`UnsupportedCapabilityError`.
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
    return _build_collision_description(_scene_data_from_mjmodel(mj_model))


def _scene_data_from_mjmodel(mj_model: Any) -> _ModelSceneData:
    """Snapshot the descriptor inputs out of a ``mujoco.MjModel``."""
    return _ModelSceneData(
        num_bodies=int(mj_model.nbody),
        geom_type_codes=np.asarray(mj_model.geom_type, dtype=np.int64),
        geom_sizes=np.array(mj_model.geom_size, dtype=np.float64, copy=True),
        geom_pos=np.array(mj_model.geom_pos, dtype=np.float64, copy=True),
        geom_quat=np.array(mj_model.geom_quat, dtype=np.float64, copy=True),
        geom_body_ids=np.array(mj_model.geom_bodyid, dtype=np.intp, copy=True),
        geom_data_ids=np.asarray(mj_model.geom_dataid, dtype=np.int64),
        mesh_vert_adr=np.asarray(mj_model.mesh_vertadr, dtype=np.int64),
        mesh_vert_num=np.asarray(mj_model.mesh_vertnum, dtype=np.int64),
        mesh_face_adr=np.asarray(mj_model.mesh_faceadr, dtype=np.int64),
        mesh_face_num=np.asarray(mj_model.mesh_facenum, dtype=np.int64),
        mesh_vert=np.asarray(mj_model.mesh_vert, dtype=np.float32),
        mesh_face=np.asarray(mj_model.mesh_face, dtype=np.int32).reshape(-1),
        hfield_nrow=np.asarray(mj_model.hfield_nrow, dtype=np.int64),
        hfield_ncol=np.asarray(mj_model.hfield_ncol, dtype=np.int64),
        hfield_adr=np.asarray(mj_model.hfield_adr, dtype=np.int64),
        hfield_size=np.asarray(mj_model.hfield_size, dtype=np.float64),
        hfield_data=np.asarray(mj_model.hfield_data, dtype=np.float64),
    )


def _build_collision_description(data: _ModelSceneData) -> MjBatchCollision:
    """Descriptor build shared by the mjbatch and mjwarp cold-path profiles."""
    unknown = sorted(set(data.geom_type_codes.tolist()) - set(_MJ_TO_CONTRACT_TYPES))
    if unknown:
        raise UnsupportedCapabilityError(
            f"uni_ray does not support MuJoCo geom type codes {unknown}; "
            "supported geoms are plane, hfield (triangulated), sphere, capsule, "
            "ellipsoid, cylinder, box, and static mesh"
        )

    geom_types = tuple(_MJ_TO_CONTRACT_TYPES[int(code)] for code in data.geom_type_codes)
    geom_sizes = np.array(data.geom_sizes, dtype=np.float64, copy=True)
    # The contract plane is the infinite local z = 0 plane and ignores sizes;
    # MuJoCo's plane size is a rendering grid, so zero it out.
    geom_sizes[data.geom_type_codes == 0] = 0.0

    meshes, geom_mesh_ids = _extract_meshes(data)

    scene = RaySceneDescription(
        num_bodies=data.num_bodies,
        geom_types=geom_types,
        geom_sizes=geom_sizes,
        geom_local_pos=np.array(data.geom_pos, dtype=np.float64, copy=True),
        geom_local_quat=np.array(data.geom_quat, dtype=np.float64, copy=True),
        geom_body_ids=np.array(data.geom_body_ids, dtype=np.intp, copy=True),
    )
    return MjBatchCollision(scene=scene, meshes=meshes, geom_mesh_ids=geom_mesh_ids)


def _extract_meshes(data: _ModelSceneData) -> tuple[tuple[MeshData, ...], np.ndarray]:
    """Slice static mesh data out of the model, deduplicated by (geom type, data id).

    Mesh geoms carry their compiled ``mesh_vert``/``mesh_face`` slice; hfield
    geoms are triangulated once in the geom-local frame. The dedup key carries
    the geom type because hfield and mesh data ids are separate index spaces.
    """
    geom_mesh_ids = np.full(data.geom_type_codes.shape[0], -1, dtype=np.intp)
    mesh_index_by_key: dict[tuple[int, int], int] = {}
    meshes: list[MeshData] = []
    for geom_id, data_id in enumerate(data.geom_data_ids):
        type_code = int(data.geom_type_codes[geom_id])
        if type_code not in (_MJ_GEOM_HFIELD, _MJ_GEOM_MESH) or data_id < 0:
            continue
        key = (type_code, int(data_id))
        if key not in mesh_index_by_key:
            if type_code == _MJ_GEOM_HFIELD:
                points, faces = _triangulate_hfield(data, int(data_id))
            else:
                vert_adr = int(data.mesh_vert_adr[data_id])
                vert_num = int(data.mesh_vert_num[data_id])
                face_adr = int(data.mesh_face_adr[data_id])
                face_num = int(data.mesh_face_num[data_id])
                points = data.mesh_vert[vert_adr : vert_adr + vert_num]
                faces = data.mesh_face[face_adr * 3 : (face_adr + face_num) * 3]
            mesh_index_by_key[key] = len(meshes)
            meshes.append(MeshData(points=points, indices=faces.astype(np.int32)))
        geom_mesh_ids[geom_id] = mesh_index_by_key[key]
    return tuple(meshes), geom_mesh_ids


def _triangulate_hfield(data: _ModelSceneData, hfield_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate an hfield into a static closed-solid mesh in the geom-local frame.

    The top surface grid is ported from MuJoCo-LiDAR's
    ``MjLidarWarp._build_hfield_mesh`` (MIT License; see NOTICE): vertices on
    the [-x, x] x [-y, y] grid with ``z = elevation * z_scale``, each cell
    split along the same diagonal ``mujoco.mj_ray`` uses.

    The port is extended into the closed solid ``mj_rayHfield`` actually
    intersects (mujoco/src/engine/engine_ray.c): MuJoCo's hfield size is
    (x half extent, y half extent, elevation scale, base depth), and the ray
    query tests, in addition to the top surface triangles, the four side
    walls of the elevation box clipped below the boundary surface edge (from
    z = 0 up to the edge heights) and the full base box (z in
    [-base_depth, 0], including its interior-facing top at z = 0). Building
    the same solid keeps hit distances identical to mj_ray for rays from any
    direction, including side and below-base approaches; MuJoCo-LiDAR only
    builds the top surface.
    """
    nrow = int(data.hfield_nrow[hfield_id])
    ncol = int(data.hfield_ncol[hfield_id])
    adr = int(data.hfield_adr[hfield_id])
    hfield = data.hfield_data[adr : adr + nrow * ncol].reshape(nrow, ncol)
    size = data.hfield_size[hfield_id]
    rx, ry, ez = float(size[0]), float(size[1]), float(size[2])
    base_z = float(size[3])

    x = np.linspace(-rx, rx, ncol, dtype=np.float32)
    y = np.linspace(-ry, ry, nrow, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heights = (hfield * ez).astype(np.float32)
    points = np.stack((xx, yy, heights), axis=-1).reshape(-1, 3).astype(np.float32)

    cells = np.arange((nrow - 1) * (ncol - 1), dtype=np.int32).reshape(nrow - 1, ncol - 1)
    row = cells // (ncol - 1)
    col = cells % (ncol - 1)
    v00 = row * ncol + col
    v10 = v00 + 1
    v01 = v00 + ncol
    v11 = v01 + 1
    faces = np.stack(
        (
            np.stack((v00, v10, v11), axis=-1),
            np.stack((v00, v11, v01), axis=-1),
        ),
        axis=2,
    ).reshape(-1)

    extra_points: list[tuple[float, float, float]] = []
    extra_faces: list[tuple[int, int, int]] = []

    def add_vertex(px: float, py: float, pz: float) -> int:
        extra_points.append((px, py, pz))
        return len(points) + len(extra_points) - 1

    # Skirt walls: z = 0 up to each boundary edge, matching the elevation-box
    # side faces mj_ray clips against the boundary surface line.
    boundary_edges: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for c in range(ncol - 1):  # y = -ry and y = +ry rows
        boundary_edges.append(
            (
                (float(x[c]), -ry, float(heights[0, c])),
                (float(x[c + 1]), -ry, float(heights[0, c + 1])),
            )
        )
        boundary_edges.append(
            (
                (float(x[c]), ry, float(heights[nrow - 1, c])),
                (float(x[c + 1]), ry, float(heights[nrow - 1, c + 1])),
            )
        )
    for r in range(nrow - 1):  # x = -rx and x = +rx columns
        boundary_edges.append(
            (
                (-rx, float(y[r]), float(heights[r, 0])),
                (-rx, float(y[r + 1]), float(heights[r + 1, 0])),
            )
        )
        boundary_edges.append(
            (
                (rx, float(y[r]), float(heights[r, ncol - 1])),
                (rx, float(y[r + 1]), float(heights[r + 1, ncol - 1])),
            )
        )
    for (ax, ay, ah), (bx, by, bh) in boundary_edges:
        if ah == 0.0 and bh == 0.0:
            continue  # zero-height wall: mj_ray's strict z < edge rejects it too
        a_top = add_vertex(ax, ay, ah)
        b_top = add_vertex(bx, by, bh)
        a_zero = add_vertex(ax, ay, 0.0)
        b_zero = add_vertex(bx, by, 0.0)
        extra_faces.append((a_zero, b_zero, b_top))
        extra_faces.append((a_zero, b_top, a_top))

    # Base box: the full box z in [-base_z, 0] mj_ray intersects first.
    if base_z > 0.0:
        top_ring = [
            add_vertex(px, py, 0.0)
            for px, py in ((-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry))
        ]
        bottom_ring = [
            add_vertex(px, py, -base_z)
            for px, py in ((-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry))
        ]
        t0, t1, t2, t3 = top_ring
        b0, b1, b2, b3 = bottom_ring
        extra_faces.extend(
            [
                (t0, t1, t2), (t0, t2, t3),  # top (z = 0), as mj_ray reports it
                (b0, b2, b1), (b0, b3, b2),  # bottom
                (b0, b1, t1), (b0, t1, t0),  # y = -ry side
                (b1, b2, t2), (b1, t2, t1),  # x = +rx side
                (b2, b3, t3), (b2, t3, t2),  # y = +ry side
                (b3, b0, t0), (b3, t0, t3),  # x = -rx side
            ]
        )

    if extra_points:
        all_points = np.concatenate(
            [points, np.array(extra_points, dtype=np.float32).reshape(-1, 3)]
        )
        all_faces = np.concatenate(
            [faces, np.array(extra_faces, dtype=np.int32).reshape(-1)]
        )
    else:
        all_points, all_faces = points, faces
    return all_points, all_faces.astype(np.int32)


def rebuild_caster_from_model(caster: Any, mj_model: Any) -> MjBatchCollision:
    """Re-read an updated ``MjModel`` and rebuild ``caster``'s scene (cold path).

    Convenience for geometry randomization: mutate geom sizes/local poses,
    hfield or mesh data in the model, then call this instead of recreating
    the caster. Equivalent to ``caster.rebuild(build_collision_description(
    mj_model))``; returns the rebuilt descriptor. Never invoked by
    ``update_pose``/``trace`` — the hot path only accepts body poses.
    """
    from .warp_caster import WarpRayCaster

    if not isinstance(caster, WarpRayCaster):
        raise TypeError(f"caster must be a WarpRayCaster, got {type(caster).__name__}")
    collision = build_collision_description(mj_model)
    caster.rebuild(collision)
    return collision


__all__ = [
    "MeshData",
    "MjBatchCollision",
    "build_collision_description",
    "rebuild_caster_from_model",
]
