"""Warp-accelerated batched ray caster implementing the UniSim ray-query contract.

The caster owns persistent device and host buffers sized by the fixed batch
shape ``(num_envs, num_rays)`` and the materialized scene; ``update_pose`` and
``trace`` only copy validated NumPy inputs into those buffers and launch
fixed-shape kernels, so the hot path performs no unbounded allocation, no XML
parsing, no name resolution, and no ``mujoco.MjData`` access.

Results are returned through an explicit host readback. ``distance``, ``hit``,
and ``geom_id`` are views into caster-owned host buffers that the next
``trace`` call reuses (callers retaining results across calls must copy them);
``hit_point`` and ``body_id`` are computed fresh on the host per call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from unisim.errors import BackendError, UnsupportedCapabilityError
from unisim.ray_query import (
    RayCaster,
    RayCasterCapabilities,
    RayGeomType,
    RaySceneDescription,
    RayTraceOutputs,
    RayTraceResult,
)

from .kernels import (
    compute_geom_poses_kernel,
    trace_rays_batch_bvh_kernel,
    update_aabbs_batch_kernel,
    write_body_poses_kernel,
)

if TYPE_CHECKING:
    from .mjbatch import MjBatchCollision

# Kernel dispatch codes follow the MuJoCo mjtGeom numbering used by the copied
# MuJoCo-LiDAR kernels (0=plane, 2=sphere, 3=capsule, 4=ellipsoid, 5=cylinder,
# 6=box, 7=mesh).
_KERNEL_GEOM_TYPES = {
    RayGeomType.PLANE: 0,
    RayGeomType.SPHERE: 2,
    RayGeomType.CAPSULE: 3,
    RayGeomType.ELLIPSOID: 4,
    RayGeomType.CYLINDER: 5,
    RayGeomType.BOX: 6,
    RayGeomType.MESH: 7,
}

_POSITIVE_SIZE_DIMS = {
    RayGeomType.SPHERE: (0,),
    RayGeomType.BOX: (0, 1, 2),
    RayGeomType.CYLINDER: (0, 1),
    RayGeomType.CAPSULE: (0, 1),
    RayGeomType.ELLIPSOID: (0, 1, 2),
}

# Warp quaternions are stored (x, y, z, w); the contract uses wxyz.
_WXYZ_TO_XYZW = (1, 2, 3, 0)


def _default_device() -> str:
    return "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


class WarpRayCaster(RayCaster):
    """Batched Warp ray caster with per-env BVH groups and host readback."""

    caster_type = "uni_ray"

    _ray_capabilities = RayCasterCapabilities(
        supports_pose_sync=True,
        supports_per_env_rays=True,
        supports_host_readback=True,
        supports_device_output=False,
        supports_hit_point=True,
        supports_normal=False,
        supports_geom_id=True,
        supports_body_id=True,
    )

    def __init__(
        self,
        num_envs: int = 1,
        num_rays: int = 1,
        *,
        collision: MjBatchCollision | None = None,
        device: str | None = None,
    ) -> None:
        for name, value in (("num_envs", num_envs), ("num_rays", num_rays)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self._num_envs = num_envs
        self._num_rays = num_rays
        self._collision = collision
        wp.init()
        self._device = device if device is not None else _default_device()

        self._scene: RaySceneDescription | None = None
        self._closed = False

        # Persistent ray staging and result buffers (fixed batch shape).
        self._ray_origins_host = np.zeros((num_envs, num_rays, 3), dtype=np.float32)
        self._ray_directions_host = np.zeros((num_envs, num_rays, 3), dtype=np.float32)
        self._env_map_host = np.zeros(num_envs, dtype=np.int32)
        self._ray_origins_host_wp = wp.array(
            self._ray_origins_host, dtype=wp.vec3, device="cpu", copy=False
        )
        self._ray_directions_host_wp = wp.array(
            self._ray_directions_host, dtype=wp.vec3, device="cpu", copy=False
        )
        self._env_map_host_wp = wp.array(
            self._env_map_host, dtype=wp.int32, device="cpu", copy=False
        )
        self._ray_origins_dev = wp.zeros((num_envs, num_rays), dtype=wp.vec3, device=self._device)
        self._ray_directions_dev = wp.zeros(
            (num_envs, num_rays), dtype=wp.vec3, device=self._device
        )
        self._env_map_dev = wp.zeros(num_envs, dtype=wp.int32, device=self._device)
        self._distances_dev = wp.zeros((num_envs, num_rays), dtype=wp.float32, device=self._device)
        self._hits_dev = wp.zeros((num_envs, num_rays), dtype=wp.bool, device=self._device)
        self._hit_geoms_dev = wp.zeros((num_envs, num_rays), dtype=wp.int32, device=self._device)
        self._distances_host = np.zeros((num_envs, num_rays), dtype=np.float32)
        self._hits_host = np.zeros((num_envs, num_rays), dtype=np.bool_)
        self._hit_geoms_host = np.zeros((num_envs, num_rays), dtype=np.int32)
        self._distances_host_wp = wp.array(
            self._distances_host, dtype=wp.float32, device="cpu", copy=False
        )
        self._hits_host_wp = wp.array(self._hits_host, dtype=wp.bool, device="cpu", copy=False)
        self._hit_geoms_host_wp = wp.array(
            self._hit_geoms_host, dtype=wp.int32, device="cpu", copy=False
        )

        # Scene-dependent state, allocated by materialize().
        self._num_bodies = 0
        self._num_geoms = 0
        self._geom_types_dev: wp.array | None = None
        self._geom_sizes_dev: wp.array | None = None
        self._geom_aabb_center_dev: wp.array | None = None
        self._geom_aabb_size_dev: wp.array | None = None
        self._geom_mesh_ids_dev: wp.array | None = None
        self._mesh_ids_dev: wp.array | None = None
        self._geom_body_ids_dev: wp.array | None = None
        self._geom_local_pos_dev: wp.array | None = None
        self._geom_local_quat_dev: wp.array | None = None
        self._body_xpos_dev: wp.array | None = None
        self._body_xquat_dev: wp.array | None = None
        self._geom_xpos_dev: wp.array | None = None
        self._geom_xmat_dev: wp.array | None = None
        self._aabb_lowers_dev: wp.array | None = None
        self._aabb_uppers_dev: wp.array | None = None
        self._bvh: wp.Bvh | None = None
        self._pose_pos_host: np.ndarray | None = None
        self._pose_quat_host: np.ndarray | None = None
        self._pose_pos_host_wp: wp.array | None = None
        self._pose_quat_host_wp: wp.array | None = None
        self._pose_pos_dev: wp.array | None = None
        self._pose_quat_dev: wp.array | None = None
        self._meshes: list[wp.Mesh] = []

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_rays(self) -> int:
        return self._num_rays

    def materialize(self, scene: RaySceneDescription) -> None:
        self._require_open()
        if self._scene is not None:
            raise BackendError("uni_ray caster is already materialized")
        if not isinstance(scene, RaySceneDescription):
            raise TypeError("scene must be a RaySceneDescription")
        collision = self._collision
        if RayGeomType.MESH in scene.geom_types and collision is None:
            raise UnsupportedCapabilityError(
                "uni_ray mesh support requires a collision descriptor built by "
                "uni_ray.mjbatch.build_collision_description and passed as "
                "create_ray_caster(..., collision=...)"
            )
        if collision is not None:
            if (
                collision.scene.num_geoms != scene.num_geoms
                or collision.scene.num_bodies != scene.num_bodies
            ):
                raise ValueError(
                    "collision descriptor does not match the materialized scene; "
                    "materialize the collision.scene of the descriptor passed at creation"
                )
        for geom_type, size in zip(scene.geom_types, scene.geom_sizes):
            required = _POSITIVE_SIZE_DIMS.get(geom_type, ())
            if any(float(size[dim]) <= 0.0 for dim in required):
                raise ValueError(
                    f"uni_ray requires positive sizes for geom type "
                    f"'{geom_type.value}' dims {required}, got {tuple(size)}"
                )

        num_geoms = scene.num_geoms
        self._scene = scene
        self._num_bodies = scene.num_bodies
        self._num_geoms = num_geoms
        device = self._device

        geom_types = np.array(
            [_KERNEL_GEOM_TYPES[geom_type] for geom_type in scene.geom_types], dtype=np.int32
        )
        geom_sizes = scene.geom_sizes.astype(np.float32).copy()
        # The copied kernels read the cylinder/capsule half height at size[2];
        # swap it in from the MuJoCo-style (radius, half_length, 0) layout.
        cylindrical = (geom_types == 3) | (geom_types == 5)
        geom_sizes[cylindrical, 2] = geom_sizes[cylindrical, 1]
        geom_sizes[cylindrical, 1] = geom_sizes[cylindrical, 0]
        aabb_center, aabb_size = self._local_aabbs(scene, collision)
        geom_local_quat_xyzw = scene.geom_local_quat[:, _WXYZ_TO_XYZW].astype(np.float32)

        self._geom_types_dev = wp.array(geom_types, dtype=wp.int32, device=device)
        self._geom_sizes_dev = wp.array(geom_sizes, dtype=wp.vec3, device=device)
        self._geom_aabb_center_dev = wp.array(aabb_center, dtype=wp.vec3, device=device)
        self._geom_aabb_size_dev = wp.array(aabb_size, dtype=wp.vec3, device=device)
        if collision is not None and collision.meshes:
            # wp.Mesh builds its internal BVH once here; meshes are static for
            # the caster's lifetime (dynamic meshes are unsupported by design).
            self._meshes = [
                wp.Mesh(
                    points=wp.array(mesh.points, dtype=wp.vec3, device=device),
                    indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
                )
                for mesh in collision.meshes
            ]
            mesh_ids = np.array([mesh.id for mesh in self._meshes], dtype=np.uint64)
            geom_mesh_ids = collision.geom_mesh_ids.astype(np.int32)
        else:
            mesh_ids = np.zeros(0, dtype=np.uint64)
            geom_mesh_ids = np.full(num_geoms, -1, dtype=np.int32)
        self._geom_mesh_ids_dev = wp.array(geom_mesh_ids, dtype=wp.int32, device=device)
        self._mesh_ids_dev = wp.array(mesh_ids, dtype=wp.uint64, device=device)
        self._geom_body_ids_dev = wp.array(
            scene.geom_body_ids.astype(np.int32), dtype=wp.int32, device=device
        )
        self._geom_local_pos_dev = wp.array(
            scene.geom_local_pos.astype(np.float32), dtype=wp.vec3, device=device
        )
        self._geom_local_quat_dev = wp.array(
            geom_local_quat_xyzw, dtype=wp.quat, device=device
        )

        num_envs = self._num_envs
        num_bodies = self._num_bodies
        identity_quat_xyzw = np.zeros((num_envs, num_bodies, 4), dtype=np.float32)
        identity_quat_xyzw[..., 3] = 1.0
        self._body_xpos_dev = wp.zeros((num_envs, num_bodies), dtype=wp.vec3, device=device)
        self._body_xquat_dev = wp.array(identity_quat_xyzw, dtype=wp.quat, device=device)
        self._geom_xpos_dev = wp.zeros((num_envs, num_geoms), dtype=wp.vec3, device=device)
        self._geom_xmat_dev = wp.zeros((num_envs, num_geoms, 9), dtype=wp.float32, device=device)

        self._pose_pos_host = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        self._pose_quat_host = np.zeros((num_envs, num_bodies, 4), dtype=np.float32)
        self._pose_pos_host_wp = wp.array(
            self._pose_pos_host, dtype=wp.vec3, device="cpu", copy=False
        )
        self._pose_quat_host_wp = wp.array(
            self._pose_quat_host, dtype=wp.quat, device="cpu", copy=False
        )
        self._pose_pos_dev = wp.zeros((num_envs, num_bodies), dtype=wp.vec3, device=device)
        self._pose_quat_dev = wp.zeros((num_envs, num_bodies), dtype=wp.quat, device=device)

        if num_geoms > 0:
            self._aabb_lowers_dev = wp.zeros(num_envs * num_geoms, dtype=wp.vec3, device=device)
            self._aabb_uppers_dev = wp.zeros(num_envs * num_geoms, dtype=wp.vec3, device=device)
            groups = np.repeat(np.arange(num_envs, dtype=np.int32), num_geoms)
            groups_dev = wp.array(groups, dtype=wp.int32, device=device)
            self._bvh = wp.Bvh(self._aabb_lowers_dev, self._aabb_uppers_dev, groups=groups_dev)
            # Bind the identity poses the contract specifies for a freshly
            # materialized scene, then build the BVH once.
            self._refresh_poses(np.arange(num_envs, dtype=np.intp))
            self._bvh.rebuild()

    def update_pose(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self._require_open()
        scene = self._require_materialized()
        rows = self._resolve_selected_rows(env_ids)
        pos = np.asarray(body_pos, dtype=np.float64)
        quat = np.asarray(body_quat, dtype=np.float64)
        for name, value in (("body_pos", pos), ("body_quat", quat)):
            expected = (rows.size, scene.num_bodies, 3 if name == "body_pos" else 4)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("body_quat requires unit wxyz quaternions")
        if rows.size == 0:
            return
        assert self._pose_pos_host is not None and self._pose_quat_host is not None
        self._pose_pos_host[: rows.size] = pos.astype(np.float32)
        self._pose_quat_host[: rows.size] = quat[..., _WXYZ_TO_XYZW].astype(np.float32)
        wp.copy(self._pose_pos_dev, self._pose_pos_host_wp)
        wp.copy(self._pose_quat_dev, self._pose_quat_host_wp)
        wp.launch(
            write_body_poses_kernel,
            dim=(rows.size, self._num_bodies),
            inputs=[
                self._env_map_for(rows),
                self._pose_pos_dev,
                self._pose_quat_dev,
                self._body_xpos_dev,
                self._body_xquat_dev,
            ],
            device=self._device,
        )
        self._refresh_poses(rows)
        if self._bvh is not None:
            self._bvh.refit()

    def trace(
        self,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
        max_distance: float,
        env_ids: Sequence[int] | np.ndarray | None = None,
        outputs: RayTraceOutputs | None = None,
    ) -> RayTraceResult:
        self._require_open()
        scene = self._require_materialized()
        request = self._check_trace_outputs(outputs)
        rows = self._resolve_selected_rows(env_ids)
        limit = self._resolve_max_distance(max_distance)
        origins, directions = self._resolve_ray_batch(ray_origins, ray_directions, rows.size)

        count = rows.size
        if count and self._num_geoms:
            self._ray_origins_host[:count] = origins.astype(np.float32)
            self._ray_directions_host[:count] = directions.astype(np.float32)
            wp.copy(self._ray_origins_dev, self._ray_origins_host_wp)
            wp.copy(self._ray_directions_dev, self._ray_directions_host_wp)
            assert self._bvh is not None
            wp.launch(
                trace_rays_batch_bvh_kernel,
                dim=(count, self._num_rays),
                inputs=[
                    self._bvh.id,
                    self._geom_types_dev,
                    self._geom_sizes_dev,
                    self._geom_mesh_ids_dev,
                    self._mesh_ids_dev,
                    self._geom_xpos_dev,
                    self._geom_xmat_dev,
                    self._env_map_for(rows),
                    self._ray_origins_dev,
                    self._ray_directions_dev,
                    float(limit),
                    self._distances_dev,
                    self._hits_dev,
                    self._hit_geoms_dev,
                ],
                device=self._device,
            )
            wp.copy(self._distances_host_wp, self._distances_dev)
            wp.copy(self._hits_host_wp, self._hits_dev)
            wp.copy(self._hit_geoms_host_wp, self._hit_geoms_dev)
            wp.synchronize_device(self._device)
            distance = self._distances_host[:count]
            hit = self._hits_host[:count]
            geom_id = self._hit_geoms_host[:count]
        else:
            distance = np.full((count, self._num_rays), limit, dtype=np.float32)
            hit = np.zeros((count, self._num_rays), dtype=np.bool_)
            geom_id = np.full((count, self._num_rays), -1, dtype=np.int32)

        hit_point = None
        body_id = None
        if request.hit_point:
            hit_point = origins + distance.astype(np.float64)[..., None] * directions
        if request.body_id:
            body_id = np.where(hit, scene.geom_body_ids[np.maximum(geom_id, 0)], -1)
        return RayTraceResult(
            distance=distance,
            hit=hit,
            hit_point=hit_point,
            geom_id=geom_id if request.geom_id else None,
            body_id=body_id,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._meshes = []
        self._bvh = None

    def _env_map_for(self, rows: np.ndarray) -> wp.array:
        self._env_map_host[: rows.size] = rows.astype(np.int32)
        wp.copy(self._env_map_dev, self._env_map_host_wp)
        return self._env_map_dev

    def _refresh_poses(self, rows: np.ndarray) -> None:
        """Recompute geom world poses and AABBs for the selected rows on device."""
        if not rows.size or not self._num_geoms:
            return
        env_map = self._env_map_for(rows)
        wp.launch(
            compute_geom_poses_kernel,
            dim=(rows.size, self._num_geoms),
            inputs=[
                env_map,
                self._geom_body_ids_dev,
                self._geom_local_pos_dev,
                self._geom_local_quat_dev,
                self._body_xpos_dev,
                self._body_xquat_dev,
                self._geom_xpos_dev,
                self._geom_xmat_dev,
            ],
            device=self._device,
        )
        wp.launch(
            update_aabbs_batch_kernel,
            dim=(rows.size, self._num_geoms),
            inputs=[
                self._geom_types_dev,
                self._geom_sizes_dev,
                self._geom_aabb_center_dev,
                self._geom_aabb_size_dev,
                self._geom_xpos_dev,
                self._geom_xmat_dev,
                env_map,
                self._aabb_lowers_dev,
                self._aabb_uppers_dev,
            ],
            device=self._device,
        )

    @staticmethod
    def _local_aabbs(
        scene: RaySceneDescription, collision: MjBatchCollision | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Local-frame AABB center/half extents per geom for the broad phase."""
        center = np.zeros((scene.num_geoms, 3), dtype=np.float32)
        size = np.zeros((scene.num_geoms, 3), dtype=np.float32)
        for index, (geom_type, geom_size) in enumerate(zip(scene.geom_types, scene.geom_sizes)):
            radius = float(geom_size[0])
            if geom_type == RayGeomType.SPHERE:
                size[index] = radius
            elif geom_type == RayGeomType.BOX or geom_type == RayGeomType.ELLIPSOID:
                size[index] = geom_size
            elif geom_type == RayGeomType.CYLINDER:
                size[index] = (radius, radius, float(geom_size[1]))
            elif geom_type == RayGeomType.CAPSULE:
                size[index] = (radius, radius, float(geom_size[1]) + radius)
            elif geom_type == RayGeomType.MESH:
                assert collision is not None
                points = collision.meshes[int(collision.geom_mesh_ids[index])].points
                lower = points.min(axis=0)
                upper = points.max(axis=0)
                center[index] = (lower + upper) * 0.5
                size[index] = (upper - lower) * 0.5
            # Planes stay zero; update_aabbs_batch_kernel substitutes the
            # infinite-plane half extent from geom_sizes directly.
        return center, size

    def _require_open(self) -> None:
        if self._closed:
            raise BackendError("uni_ray caster is closed")

    def _require_materialized(self) -> RaySceneDescription:
        if self._scene is None:
            raise BackendError("uni_ray caster must be materialized before querying")
        return self._scene


__all__ = ["WarpRayCaster"]
