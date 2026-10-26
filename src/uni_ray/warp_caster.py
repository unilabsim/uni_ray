"""Warp-accelerated batched ray caster implementing the UniSim ray-query contract.

The caster owns persistent device and host buffers sized by the fixed batch
shape ``(num_envs, num_rays)`` and the materialized scene; ``update_pose`` and
``trace`` only copy validated NumPy inputs into those buffers and launch
fixed-shape kernels, so the hot path performs no unbounded allocation, no XML
parsing, no name resolution, and no ``mujoco.MjData`` access.

The host profile returns results through an explicit readback into persistent
buffers. The tensor profile returns persistent caster-owned Torch views on the
caller's device and stream. Both profiles support static geom/body/tag filters,
an inclusive ``min_distance``, and optional world-space against-ray normals.
The public ``RayCollisionDescription`` companion supplies mesh topology and
public identities.

The contract requires a second ``materialize`` to fail, so scene replacement
(runtime geometry randomization, mesh/hfield replacement) goes through the
explicit cold-path :meth:`WarpRayCaster.rebuild`, which re-binds an updated
descriptor without recreating the caster and is never invoked by the hot
path.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import warp as wp
from unisim.errors import BackendError, UnsupportedCapabilityError
from unisim.ray_query import (
    RayCaster,
    RayCasterCapabilities,
    RayCollisionDescription,
    RayGeometryState,
    RayGeometryTensorView,
    RayGeomType,
    RayPoseTensorView,
    RaySceneDescription,
    RayTraceFilter,
    RayTraceOutputs,
    RayTraceResult,
    RayTraceTensorResult,
)

from .kernels import (
    compute_geom_poses_kernel,
    fill_empty_trace_kernel,
    fill_ray_pattern_kernel,
    trace_rays_batch_bvh_kernel,
    update_aabbs_batch_kernel,
    write_body_poses_from_tensors_kernel,
    write_body_poses_kernel,
    write_env_map_and_validate_ids_kernel,
    write_geometry_kernel,
    write_tensor_trace_outputs_kernel,
    zero_int32_kernel,
)

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
_LITTLE_ENDIAN = sys.byteorder == "little"
_TensorResultKey = tuple[int, bool, bool, bool, bool]


def _default_device() -> str:
    return "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _torch_int64_view(tensor: Any) -> wp.array:
    """Alias a contiguous public int64 ID tensor without copying."""
    if not _LITTLE_ENDIAN:
        raise RuntimeError("uni_ray tensor environment IDs require a little-endian host")
    view = wp.array(
        ptr=tensor.data_ptr(),
        dtype=wp.int64,
        shape=(tensor.numel(),),
        strides=(8,),
        device=tensor.device.type if tensor.device.type == "cpu" else f"cuda:{tensor.device.index}",
        copy=False,
    )
    view._tensor = tensor
    return view


@contextmanager
def _torch_caller_stream_scope(torch: Any, device: str, warp_device: wp.Device) -> Any:
    """Launch Warp work on Torch's current stream without host synchronization."""
    if device == "cpu":

        @contextmanager
        def cpu_scope() -> Any:
            yield

        with cpu_scope():
            yield
        return

    caller = torch.cuda.current_stream(device)
    stream = wp.Stream(device=warp_device, cuda_stream=int(caller.cuda_stream))
    try:
        with wp.ScopedStream(stream, sync_enter=False):
            yield
    finally:
        # The caller owns the underlying CUDA stream; this only unregisters Warp's
        # temporary external-stream wrapper after all launches have been enqueued.
        stream.__del__()


def _has_non_overlapping_strides(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    """Return whether positive axis strides map logical indices injectively."""
    active = sorted((stride, size) for size, stride in zip(shape, strides, strict=True) if size > 1)
    extent = 1
    for stride, size in active:
        if stride < extent:
            return False
        extent = (size - 1) * stride + 1
    return True


class WarpRayCaster(RayCaster):
    """Batched Warp ray caster with NumPy and Torch-native query profiles."""

    caster_type = "uni_ray"

    _ray_capabilities = RayCasterCapabilities(
        supports_pose_sync=True,
        supports_geometry_sync=True,
        supports_per_env_rays=True,
        supports_host_readback=True,
        supports_device_output=True,
        supports_tensor_pose_sync=True,
        supports_tensor_geometry_sync=True,
        supports_tensor_rays=True,
        torch_devices=("cpu", "cuda"),
        stream_event_ownership="caller",
        supports_hit_point=True,
        supports_host_normal=True,
        supports_tensor_normal=True,
        supports_geom_id=True,
        supports_body_id=True,
        supports_static_filter=True,
        supports_min_distance=True,
    )

    def __init__(
        self,
        num_envs: int = 1,
        num_rays: int = 1,
        *,
        collision: RayCollisionDescription | None = None,
        device: str | Any | None = None,
    ) -> None:
        for name, value in (("num_envs", num_envs), ("num_rays", num_rays)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if collision is not None and not isinstance(collision, RayCollisionDescription):
            raise TypeError("collision must be a RayCollisionDescription or None")
        self._num_envs = num_envs
        self._num_rays = num_rays
        self._collision = collision
        wp.init()
        self._device = str(device) if device is not None else _default_device()

        self._scene: RaySceneDescription | None = None
        self._closed = False
        self._filter_masks: dict[RayTraceFilter, wp.array] = {}
        self._all_geoms_allowed: wp.array | None = None
        self._tensor_layout: tuple[str, ...] | None = None
        self._tensor_pose_device: str | None = None
        self._tensor_pos: Any | None = None
        self._tensor_pos_wp: wp.array | None = None
        self._tensor_quat: Any | None = None
        self._tensor_quat_wp: wp.array | None = None
        self._tensor_geometry: dict[str, Any] | None = None
        self._tensor_env_ids_wp: wp.array | None = None
        self._tensor_static_pattern: tuple[Any, Any] | None = None
        self._tensor_static_origins: wp.array | None = None
        self._tensor_static_directions: wp.array | None = None
        self._tensor_results: dict[_TensorResultKey, RayTraceTensorResult] = {}
        # Minimal outputs always use a caster-owned geom-id scratch tensor; optional
        # profiles own their returned geom_id views.
        self._tensor_hit_geoms: Any | None = None
        self._env_id_error = wp.zeros(1, dtype=wp.int32, device=self._device)
        self._env_id_seen = wp.zeros(self._num_envs, dtype=wp.int32, device=self._device)
        self._all_env_map_dev = wp.array(
            np.arange(self._num_envs, dtype=np.int32),
            dtype=wp.int32,
            device=self._device,
        )
        self._empty3d_float32 = wp.zeros((0, 0, 0), dtype=wp.float32, device=self._device)
        self._empty2d_int32 = wp.zeros((0, 0), dtype=wp.int32, device=self._device)

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
        self._normals_dev = wp.zeros((num_envs, num_rays, 3), dtype=wp.float32, device=self._device)
        self._normals_host = np.zeros((num_envs, num_rays, 3), dtype=np.float32)
        self._normals_host_wp = wp.array(
            self._normals_host, dtype=wp.float32, device="cpu", copy=False
        )
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
        self._geom_sizes_dev: wp.array2d | None = None
        self._mesh_aabb_center_dev: wp.array | None = None
        self._mesh_aabb_size_dev: wp.array | None = None
        self._geom_mesh_ids_dev: wp.array2d | None = None
        self._mesh_ids_dev: wp.array | None = None
        self._geom_body_ids_dev: wp.array | None = None
        self._geom_local_pos_dev: wp.array2d | None = None
        self._geom_local_quat_dev: wp.array2d | None = None
        self._geom_enabled_dev: wp.array2d | None = None
        self._body_xpos_dev: wp.array | None = None
        self._body_xquat_dev: wp.array | None = None
        self._geom_xpos_dev: wp.array | None = None
        self._geom_xmat_dev: wp.array | None = None
        self._aabb_lowers_dev: wp.array | None = None
        self._aabb_uppers_dev: wp.array | None = None
        self._bvh: wp.Bvh | None = None
        self._device_obj = wp.get_device(self._device)
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
            raise BackendError(
                "uni_ray caster is already materialized; use rebuild() to replace the scene"
            )
        if not isinstance(scene, RaySceneDescription):
            raise TypeError("scene must be a RaySceneDescription")
        if self._collision is not None and self._collision.scene is not scene:
            raise ValueError(
                "collision descriptor does not match the bound scene; materialize collision.scene"
            )
        self._bind_scene(scene, self._collision)

    def rebuild(
        self,
        collision: RayCollisionDescription | None = None,
        *,
        scene: RaySceneDescription | None = None,
    ) -> None:
        """Re-bind the scene geometry from an updated descriptor (cold path).

        This is the explicit runtime-geometry-randomization path: pass a
        freshly built ``collision`` descriptor (e.g. from
        ``uni_ray.mjbatch.build_collision_description`` on an updated MjModel,
        or the ``uni_ray.mjbatch.rebuild_caster_from_model`` convenience) to
        replace geom sizes/local poses and mesh/hfield data, or pass a bare
        ``scene`` to change geom sizes/local poses while reusing the mesh
        data of the descriptor already bound to this caster. Exactly one of
        ``collision``/``scene`` must be given.

        Cold-path semantics: all scene-dependent device arrays, wp.Mesh
        objects, and the BVH are reallocated/rebuilt (``num_bodies`` and
        ``num_geoms`` may change); buffers keyed to the fixed
        ``(num_envs, num_rays)`` batch are untouched. As after
        ``materialize``, every body pose is the identity until
        ``update_pose`` writes it. ``update_pose``/``trace`` never call this
        method; the hot path only accepts body poses.
        """
        self._require_open()
        self._require_materialized()
        self._check_pending_env_id_error()
        if (collision is None) == (scene is None):
            raise ValueError("rebuild takes exactly one of collision= or scene=")
        if collision is not None:
            if not isinstance(collision, RayCollisionDescription):
                raise TypeError("collision must be a RayCollisionDescription")
            self._collision = collision
            self._bind_scene(collision.scene, collision)
        else:
            assert scene is not None
            if not isinstance(scene, RaySceneDescription):
                raise TypeError("scene must be a RaySceneDescription")
            self._bind_scene(scene, self._collision)

    def _bind_scene(
        self, scene: RaySceneDescription, collision: RayCollisionDescription | None
    ) -> None:
        """(Re)allocate all scene-dependent state; cold path only."""
        if RayGeomType.MESH in scene.geom_types and collision is None:
            raise UnsupportedCapabilityError(
                "uni_ray mesh support requires a public RayCollisionDescription "
                "passed as "
                "create_ray_caster(..., collision=...) or rebuild(collision=...)"
            )
        if collision is not None:
            if (
                collision.scene.num_geoms != scene.num_geoms
                or collision.scene.num_bodies != scene.num_bodies
            ):
                raise ValueError(
                    "collision descriptor does not match the bound scene; bind the "
                    "collision.scene of the descriptor the caster was created or "
                    "rebuilt with"
                )
            if (
                collision.scene.geom_types != scene.geom_types
                or collision.scene.body_names != scene.body_names
                or not np.array_equal(collision.scene.geom_body_ids, scene.geom_body_ids)
            ):
                raise ValueError(
                    "scene rebuild may change only geom sizes/local poses; geom types, "
                    "body mapping, body names, and mesh mapping must match the collision"
                )
        else:
            collision = RayCollisionDescription(scene=scene)
        self._collision = collision
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
        self._filter_masks.clear()
        self._all_geoms_allowed = wp.array(
            np.ones(num_geoms, dtype=np.int32), dtype=wp.int32, device=self._device
        )
        self._tensor_layout = None
        self._tensor_pose_device = None
        self._tensor_pos = None
        self._tensor_pos_wp = None
        self._tensor_quat = None
        self._tensor_quat_wp = None
        self._tensor_geometry = None
        self._tensor_env_ids_wp = None
        self._tensor_static_pattern = None
        self._tensor_static_origins = None
        self._tensor_static_directions = None
        self._tensor_results.clear()
        self._tensor_hit_geoms = None
        self._env_id_error.zero_()
        self._env_id_seen.zero_()
        device = self._device
        num_envs = self._num_envs

        geom_types = np.array(
            [_KERNEL_GEOM_TYPES[geom_type] for geom_type in scene.geom_types], dtype=np.int32
        )
        geom_sizes = scene.geom_sizes.astype(np.float32).copy()
        geom_sizes = np.broadcast_to(geom_sizes, (num_envs, num_geoms, 3)).copy()
        mesh_aabb_center, mesh_aabb_size = self._mesh_aabbs(collision)
        geom_local_pos = np.broadcast_to(
            scene.geom_local_pos.astype(np.float32), (num_envs, num_geoms, 3)
        ).copy()
        geom_local_quat = np.broadcast_to(
            scene.geom_local_quat.astype(np.float32), (num_envs, num_geoms, 4)
        ).copy()

        self._geom_types_dev = wp.array(geom_types, dtype=wp.int32, device=device)
        self._geom_sizes_dev = wp.array(geom_sizes, dtype=wp.vec3, device=device)
        self._mesh_aabb_center_dev = wp.array(mesh_aabb_center, dtype=wp.vec3, device=device)
        self._mesh_aabb_size_dev = wp.array(mesh_aabb_size, dtype=wp.vec3, device=device)
        if collision is not None and collision.meshes:
            # wp.Mesh builds its internal BVH once here; meshes are static
            # until the next materialize/rebuild rebinds them (per-frame
            # vertex streaming is unsupported by design).
            self._meshes = [
                wp.Mesh(
                    points=wp.array(mesh.points, dtype=wp.vec3, device=device),
                    indices=wp.array(mesh.indices, dtype=wp.int32, device=device),
                )
                for mesh in collision.meshes
            ]
            mesh_ids = np.array([mesh.id for mesh in self._meshes], dtype=np.uint64)
            geom_mesh_ids = np.broadcast_to(
                collision.geom_mesh_ids.astype(np.int32), (num_envs, num_geoms)
            ).copy()
        else:
            mesh_ids = np.zeros(0, dtype=np.uint64)
            geom_mesh_ids = np.full((num_envs, num_geoms), -1, dtype=np.int32)
        self._geom_mesh_ids_dev = wp.array(geom_mesh_ids, dtype=wp.int32, device=device)
        self._mesh_ids_dev = wp.array(mesh_ids, dtype=wp.uint64, device=device)
        self._geom_body_ids_dev = wp.array(
            scene.geom_body_ids.astype(np.int32), dtype=wp.int32, device=device
        )
        self._geom_local_pos_dev = wp.array(geom_local_pos, dtype=wp.vec3, device=device)
        self._geom_local_quat_dev = wp.array(geom_local_quat, dtype=wp.vec4, device=device)
        self._geom_enabled_dev = wp.array(
            np.ones((num_envs, num_geoms), dtype=np.bool_), dtype=wp.bool, device=device
        )

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

    def update_geometry(
        self,
        geometry: RayGeometryState,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """Write per-row sizes, local frames, masks, and mesh selections."""
        self._require_open()
        self._require_materialized()
        rows = self._resolve_selected_rows(env_ids)
        self._validate_geometry_state(geometry)
        if not rows.size or not self._num_geoms:
            return
        sources = self._geometry_wp_arrays(
            sizes=geometry.geom_sizes[rows],
            local_pos=geometry.geom_local_pos[rows],
            local_quat=geometry.geom_local_quat[rows],
            enabled=geometry.geom_enabled[rows],
            mesh_ids=geometry.geom_mesh_ids[rows],
        )
        wp.launch(
            write_geometry_kernel,
            dim=(rows.size, self._num_geoms),
            inputs=[
                self._env_map_for(rows),
                *sources,
                self._geom_sizes_dev,
                self._geom_local_pos_dev,
                self._geom_local_quat_dev,
                self._geom_enabled_dev,
                self._geom_mesh_ids_dev,
            ],
            device=self._device,
        )
        self._refresh_poses(rows)
        self._refit_bvh()

    def bind_pose_layout(self, body_names: Sequence[str]) -> None:
        """Bind the exact public dynamic-body order on the cold path."""
        self._require_open()
        scene = self._require_materialized()
        if self._tensor_layout is not None:
            raise BackendError("uni_ray caster tensor pose layout is already bound")
        if isinstance(body_names, (str, bytes)) or not isinstance(body_names, Sequence):
            raise TypeError("ray tensor body_names must be a sequence of strings")
        names = tuple(body_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise TypeError("ray tensor body names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"ray tensor body names must be unique: {names}")
        if names != scene.body_names:
            raise ValueError(
                f"ray tensor body_names must equal the scene body layout "
                f"{scene.body_names}, got {names}"
            )
        self._tensor_layout = names
        caster_device = "cpu" if self._device_obj.is_cpu else f"cuda:{self._device_obj.ordinal}"
        self._tensor_pose_device = caster_device

    def update_pose_tensor(self, pose: RayPoseTensorView) -> None:
        """Alias a full-batch caller-owned Torch pose view without copying."""
        import torch

        self._require_open()
        scene = self._require_materialized()
        layout = self._require_tensor_layout()
        if not isinstance(pose, RayPoseTensorView):
            raise TypeError("pose must be a RayPoseTensorView")
        if pose.body_names != layout:
            raise ValueError(
                f"ray tensor pose body order {pose.body_names} does not match bound layout {layout}"
            )
        bound_device = self._required_tensor_device()
        expected = (self._num_envs, len(layout), 3)
        pos = self._validated_pose_tensor(
            pose.pos_w, "pos_w", torch, device=bound_device, expected_shape=expected
        )
        quat = self._validated_pose_tensor(
            pose.quat_w,
            "quat_w",
            torch,
            device=bound_device,
            expected_shape=(self._num_envs, len(layout), 4),
        )
        self._tensor_pos = pos
        self._tensor_quat = quat
        self._tensor_pos_wp = wp.from_torch(pos, dtype=wp.float32)
        self._tensor_quat_wp = wp.from_torch(quat, dtype=wp.float32)
        with _torch_caller_stream_scope(torch, bound_device, self._device_obj):
            wp.launch(
                write_body_poses_from_tensors_kernel,
                dim=(self._num_envs, len(layout)),
                inputs=[
                    self._body_xpos_dev,
                    self._body_xquat_dev,
                    self._tensor_pos_wp,
                    self._tensor_quat_wp,
                ],
                device=self._device,
            )
            self._refresh_tensor_poses_all_rows()
            self._refit_bvh()
        # Keep these aliases only for owner diagnostics and close; Warp arrays
        # above retain the public tensors themselves.
        del self._tensor_pos, self._tensor_quat
        self._tensor_pos = self._tensor_pos_wp._tensor
        self._tensor_quat = self._tensor_quat_wp._tensor
        assert scene is not None

    def update_geometry_tensor(self, geometry: RayGeometryTensorView) -> None:
        """Alias a full-batch Torch geometry view and refresh broad-phase state."""
        import torch

        self._require_open()
        self._require_materialized()
        self._require_tensor_layout()
        device = self._required_tensor_device()
        if not isinstance(geometry, RayGeometryTensorView):
            raise TypeError("geometry must be a RayGeometryTensorView")
        values = {
            "sizes": self._validated_tensor(
                geometry.geom_sizes,
                "geometry_sizes",
                torch,
                device=device,
                expected_shape=(self._num_envs, self._num_geoms, 3),
                require_contiguous=False,
            ),
            "local_pos": self._validated_tensor(
                geometry.geom_local_pos,
                "geometry_local_pos",
                torch,
                device=device,
                expected_shape=(self._num_envs, self._num_geoms, 3),
                require_contiguous=False,
            ),
            "local_quat": self._validated_tensor(
                geometry.geom_local_quat,
                "geometry_local_quat",
                torch,
                device=device,
                expected_shape=(self._num_envs, self._num_geoms, 4),
                require_contiguous=False,
            ),
        }
        for name, dtype, shape in (
            ("enabled", torch.bool, (self._num_envs, self._num_geoms)),
            ("mesh_ids", torch.int32, (self._num_envs, self._num_geoms)),
        ):
            value = getattr(geometry, f"geom_{name}")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"ray tensor geometry_{name} must be a torch.Tensor")
            if value.dtype != dtype:
                raise TypeError(f"ray tensor geometry_{name} must have dtype {dtype}")
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"ray tensor geometry_{name} must have shape {shape}, got {tuple(value.shape)}"
                )
            if not bool(value.is_contiguous()):
                raise ValueError(f"ray tensor geometry_{name} must be contiguous")
            if self._resolve_torch_device(value.device, f"geometry_{name}") != device:
                raise ValueError(f"ray tensor geometry_{name} must live on {device}")
            values[name] = value

        sources = self._geometry_wp_arrays(
            sizes=values["sizes"],
            local_pos=values["local_pos"],
            local_quat=values["local_quat"],
            enabled=values["enabled"],
            mesh_ids=values["mesh_ids"],
        )
        with _torch_caller_stream_scope(torch, device, self._device_obj):
            wp.launch(
                write_geometry_kernel,
                dim=(self._num_envs, self._num_geoms),
                inputs=[
                    self._all_env_map_dev,
                    *sources,
                    self._geom_sizes_dev,
                    self._geom_local_pos_dev,
                    self._geom_local_quat_dev,
                    self._geom_enabled_dev,
                    self._geom_mesh_ids_dev,
                ],
                device=self._device,
            )
            self._refresh_tensor_poses_all_rows()
            self._refit_bvh()
        self._tensor_geometry = values

    def trace(
        self,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
        max_distance: float,
        env_ids: Sequence[int] | np.ndarray | None = None,
        outputs: RayTraceOutputs | None = None,
        *,
        min_distance: float = 0.0,
        ray_filter: RayTraceFilter | None = None,
    ) -> RayTraceResult:
        self._require_open()
        scene = self._require_materialized()
        request = self._check_trace_outputs(outputs)
        rows = self._resolve_selected_rows(env_ids)
        limit = self._resolve_max_distance(max_distance)
        minimum = self._check_trace_query(max_distance, min_distance, ray_filter)
        origins, directions = self._resolve_ray_batch(ray_origins, ray_directions, rows.size)
        geom_filter = self._geom_filter_array(ray_filter)

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
                    self._geom_enabled_dev,
                    self._geom_xpos_dev,
                    self._geom_xmat_dev,
                    geom_filter,
                    self._env_map_for(rows),
                    self._ray_origins_dev,
                    self._ray_directions_dev,
                    float(limit),
                    float(minimum),
                    self._distances_dev,
                    self._hits_dev,
                    self._hit_geoms_dev,
                    self._normals_dev,
                    int(request.normal),
                ],
                device=self._device,
            )
            wp.copy(self._distances_host_wp, self._distances_dev)
            wp.copy(self._hits_host_wp, self._hits_dev)
            wp.copy(self._hit_geoms_host_wp, self._hit_geoms_dev)
            if request.normal:
                wp.copy(self._normals_host_wp, self._normals_dev)
            wp.synchronize_device(self._device)
            distance = self._distances_host[:count]
            hit = self._hits_host[:count]
            geom_id = self._hit_geoms_host[:count]
            normal = self._normals_host[:count].copy() if request.normal else None
        else:
            distance = np.full((count, self._num_rays), limit, dtype=np.float32)
            hit = np.zeros((count, self._num_rays), dtype=np.bool_)
            geom_id = np.full((count, self._num_rays), -1, dtype=np.int32)
            if request.normal:
                normal = np.zeros((count, self._num_rays, 3), dtype=np.float32)
            else:
                normal = None

        hit_point = None
        body_id = None
        if request.hit_point:
            hit_point = origins + distance.astype(np.float64)[..., None] * directions
        if request.body_id:
            if scene.num_geoms:
                body_id = np.where(hit, scene.geom_body_ids[np.maximum(geom_id, 0)], -1)
            else:
                body_id = np.full((count, self._num_rays), -1, dtype=np.int32)
        return RayTraceResult(
            distance=distance,
            hit=hit,
            hit_point=hit_point,
            geom_id=geom_id if request.geom_id else None,
            body_id=body_id,
            normal=normal,
        )

    def trace_tensor(
        self,
        ray_origins: Any,
        ray_directions: Any,
        max_distance: float,
        env_ids: Any | None = None,
        outputs: RayTraceOutputs | None = None,
        *,
        min_distance: float = 0.0,
        ray_filter: RayTraceFilter | None = None,
    ) -> RayTraceTensorResult:
        """Launch a Torch-native query on the caller's current stream."""
        import torch

        self._require_open()
        self._require_materialized()
        self._require_tensor_layout()
        request = self._check_trace_outputs(outputs, tensor=True)
        minimum = self._check_trace_query(max_distance, min_distance, ray_filter)
        limit = self._resolve_max_distance(max_distance)
        device = self._required_tensor_device()
        rows = self._resolve_tensor_rows(env_ids, torch, device)
        origins, directions, shared = self._resolve_tensor_ray_batch(
            ray_origins, ray_directions, torch, device, rows.shape[0]
        )
        result = self._tensor_result(
            torch,
            device,
            rows.shape[0],
            request.hit_point,
            request.normal,
            request.geom_id,
            request.body_id,
        )
        with _torch_caller_stream_scope(torch, device, self._device_obj):
            geom_filter = self._geom_filter_array(ray_filter)
            origins_wp = wp.from_torch(origins, dtype=wp.vec3)
            directions_wp = wp.from_torch(directions, dtype=wp.vec3)
            rows_wp = _torch_int64_view(rows)
            distances_wp = wp.from_torch(result.distance, dtype=wp.float32)
            hits_wp = wp.from_torch(result.hit, dtype=wp.bool)
            if result.geom_id is None:
                if self._tensor_hit_geoms is None:
                    self._tensor_hit_geoms = torch.empty(
                        (self._num_envs, self._num_rays), dtype=torch.int32, device=device
                    )
                hit_geoms_tensor = self._tensor_hit_geoms[: rows.shape[0]]
            else:
                hit_geoms_tensor = result.geom_id
            hit_geoms_wp = wp.from_torch(hit_geoms_tensor, dtype=wp.int32)
            hit_points_wp = (
                wp.from_torch(result.hit_point, dtype=wp.float32)
                if result.hit_point is not None
                else self._empty3d_float32
            )
            body_ids_wp = (
                wp.from_torch(result.body_id, dtype=wp.int32)
                if result.body_id is not None
                else self._empty2d_int32
            )
            normals_wp = (
                wp.from_torch(result.normal, dtype=wp.float32)
                if result.normal is not None
                else self._empty3d_float32
            )

            if rows.shape[0]:
                wp.launch(
                    zero_int32_kernel,
                    dim=[self._num_envs],
                    inputs=[self._env_id_seen],
                    device=self._device,
                )
                wp.launch(
                    write_env_map_and_validate_ids_kernel,
                    dim=[rows.shape[0]],
                    inputs=[
                        rows_wp,
                        self._num_envs,
                        self._env_id_seen,
                        self._env_map_dev,
                        self._env_id_error,
                    ],
                    device=self._device,
                )
            self._raise_on_env_id_error()
            self._tensor_env_ids_wp = rows_wp
            self._tensor_static_pattern = (
                (origins_wp._tensor, directions_wp._tensor) if shared else None
            )
            if shared:
                ray_origins, ray_directions = self._static_tensor_rays()
                wp.launch(
                    fill_ray_pattern_kernel,
                    dim=(self._num_envs, self._num_rays),
                    inputs=[ray_origins, ray_directions, origins_wp, directions_wp],
                    device=self._device,
                )
                origins_wp = ray_origins
                directions_wp = ray_directions
            elif origins.ndim == 2:
                ray_origins, ray_directions = self._static_tensor_rays()
                wp.launch(
                    fill_ray_pattern_kernel,
                    dim=(rows.shape[0], self._num_rays),
                    inputs=[
                        ray_origins,
                        ray_directions,
                        wp.from_torch(origins, dtype=wp.vec3),
                        wp.from_torch(directions, dtype=wp.vec3),
                    ],
                    device=self._device,
                )
                origins_wp = ray_origins
                directions_wp = ray_directions

            if rows.shape[0] == 0 or self._num_geoms == 0:
                wp.launch(
                    fill_empty_trace_kernel,
                    dim=(rows.shape[0], self._num_rays),
                    inputs=[
                        distances_wp,
                        hits_wp,
                        hit_geoms_wp,
                        hit_points_wp,
                        normals_wp,
                        body_ids_wp,
                        float(limit),
                        int(result.hit_point is not None),
                        int(result.normal is not None),
                        int(result.body_id is not None),
                    ],
                    device=self._device,
                )
            else:
                assert self._bvh is not None
                wp.launch(
                    trace_rays_batch_bvh_kernel,
                    dim=(rows.shape[0], self._num_rays),
                    inputs=[
                        self._bvh.id,
                        self._geom_types_dev,
                        self._geom_sizes_dev,
                        self._geom_mesh_ids_dev,
                        self._mesh_ids_dev,
                        self._geom_enabled_dev,
                        self._geom_xpos_dev,
                        self._geom_xmat_dev,
                        geom_filter,
                        self._env_map_dev,
                        origins_wp,
                        directions_wp,
                        float(limit),
                        float(minimum),
                        distances_wp,
                        hits_wp,
                        hit_geoms_wp,
                        normals_wp,
                        int(request.normal),
                    ],
                    device=self._device,
                )
                if request.hit_point or request.body_id:
                    assert self._geom_body_ids_dev is not None
                    wp.launch(
                        write_tensor_trace_outputs_kernel,
                        dim=(rows.shape[0], self._num_rays),
                        inputs=[
                            origins_wp,
                            directions_wp,
                            distances_wp,
                            hits_wp,
                            hit_geoms_wp,
                            self._geom_body_ids_dev,
                            hit_points_wp,
                            body_ids_wp,
                            int(request.hit_point),
                            int(request.body_id),
                        ],
                        device=self._device,
                    )
        return result

    def _static_tensor_rays(self) -> tuple[wp.array, wp.array]:
        """Return persistent staging for the current shared/per-row ray values."""
        if self._tensor_static_origins is None:
            self._tensor_static_origins = wp.zeros(
                (self._num_envs, self._num_rays), dtype=wp.vec3, device=self._device
            )
            self._tensor_static_directions = wp.zeros(
                (self._num_envs, self._num_rays), dtype=wp.vec3, device=self._device
            )
        return self._tensor_static_origins, self._tensor_static_directions

    def close(self) -> None:
        if self._closed:
            return
        self._check_pending_env_id_error()
        self._closed = True
        self._meshes = []
        self._bvh = None
        self._filter_masks.clear()
        self._all_geoms_allowed = None
        self._tensor_layout = None
        self._tensor_pose_device = None
        self._tensor_pos = None
        self._tensor_pos_wp = None
        self._tensor_quat = None
        self._tensor_quat_wp = None
        self._tensor_geometry = None
        self._tensor_env_ids_wp = None
        self._tensor_static_pattern = None
        self._tensor_static_origins = None
        self._tensor_static_directions = None
        self._tensor_results.clear()
        self._tensor_hit_geoms = None

    def _env_map_for(self, rows: np.ndarray) -> wp.array:
        self._env_map_host[: rows.size] = rows.astype(np.int32)
        wp.copy(self._env_map_dev, self._env_map_host_wp)
        return self._env_map_dev

    def _refresh_poses(self, rows: np.ndarray) -> None:
        """Recompute geom world poses for the selected rows on device."""
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
        self._launch_update_aabbs(rows)

    def _refresh_tensor_poses_all_rows(self) -> None:
        """Refresh every tensor-path pose without a host-to-device row map."""
        if not self._num_geoms:
            return
        wp.launch(
            compute_geom_poses_kernel,
            dim=(self._num_envs, self._num_geoms),
            inputs=[
                self._all_env_map_dev,
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
        self._launch_update_aabbs(np.arange(self._num_envs, dtype=np.intp))

    def _refit_bvh(self) -> None:
        """Refit the BVH after an already-enqueued pose/AABB update."""
        if self._bvh is None:
            return
        if self._device_obj.is_cpu:
            self._bvh.refit()
            return
        self._bvh.runtime.core.wp_bvh_refit_device(self._bvh.id)

    def _raise_on_env_id_error(self) -> None:
        """Complete device-side selected-environment validation."""
        if self._device_obj.is_cpu:
            marker = self._env_id_error.numpy()[0]
            self._env_id_error.zero_()
            if marker:
                self._env_id_seen.zero_()
                raise ValueError(
                    "ray tensor environment IDs must be distinct and in range "
                    f"[0, {self._num_envs})"
                )
            return

    def _check_pending_env_id_error(self) -> None:
        """Check a CUDA validation marker on an explicit cold-path boundary."""
        if self._device_obj.is_cpu:
            return
        marker = self._env_id_error.numpy()[0]
        self._env_id_error.zero_()
        if marker:
            self._env_id_seen.zero_()
            raise ValueError(
                "ray tensor environment IDs must be distinct and in range "
                f"[0, {self._num_envs}); prior tensor trace detected an invalid ID"
            )

    def _require_tensor_layout(self) -> tuple[str, ...]:
        if self._tensor_layout is None:
            raise BackendError("uni_ray caster requires bind_pose_layout before tensor queries")
        return self._tensor_layout

    def _resolve_torch_device(self, device: Any, name: str) -> str:
        if device.type == "cpu":
            return "cpu"
        if device.type != "cuda":
            raise ValueError(f"ray tensor {name} must live on cpu or cuda, got {str(device)!r}")
        if device.index is None:
            raise ValueError(f"ray tensor {name} CUDA device index must be explicit")
        label = f"{device.type}:{device.index}"
        expected = self._device_obj.alias
        if label != expected:
            raise ValueError(
                f"ray tensor {name} must live on caster device {expected}, got {label}"
            )
        return label

    def _required_tensor_device(self) -> str:
        label = self._tensor_pose_device
        if label is None:
            raise BackendError("uni_ray caster requires update_pose_tensor before tensor queries")
        return label

    def _validated_tensor(
        self,
        value: Any,
        name: str,
        torch: Any,
        *,
        device: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
        require_contiguous: bool = True,
    ) -> Any:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"ray tensor {name} must be a torch.Tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"ray tensor {name} must have dtype float32")
        if require_contiguous and not bool(value.is_contiguous()):
            raise ValueError(
                f"ray tensor {name} must be contiguous; caller materialization required"
            )
        if device is None:
            device = self._resolve_torch_device(value.device, name)
        elif self._resolve_torch_device(value.device, name) != device:
            raise ValueError(f"ray tensor {name} must live on {device}, got {value.device}")
        if expected_shape is not None and tuple(value.shape) != expected_shape:
            raise ValueError(
                f"ray tensor {name} must have shape {expected_shape}, got {tuple(value.shape)}"
            )
        return value

    def _validated_pose_tensor(
        self,
        value: Any,
        name: str,
        torch: Any,
        *,
        device: str,
        expected_shape: tuple[int, ...],
    ) -> Any:
        """Validate one packet-strided pose view without materializing a copy."""
        value = self._validated_tensor(
            value,
            name,
            torch,
            device=device,
            expected_shape=expected_shape,
            require_contiguous=False,
        )
        shape = tuple(value.shape)
        strides = tuple(value.stride())
        if any(stride <= 0 for stride in strides):
            raise ValueError(
                f"ray tensor {name} must have positive non-overlapping strides, "
                f"got shape {shape} with strides {strides}"
            )
        if strides[-1] != 1:
            raise ValueError(f"ray tensor {name} requires an innermost stride of 1")
        if not _has_non_overlapping_strides(shape, strides):
            raise ValueError(
                f"ray tensor {name} must have positive non-overlapping strides, "
                f"got shape {shape} with strides {strides}"
            )
        return value

    def _resolve_tensor_rows(self, env_ids: Any, torch: Any, device: str) -> Any:
        if env_ids is None:
            return torch.arange(self._num_envs, dtype=torch.int64, device=device)
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("ray tensor env_ids must be a torch.Tensor or None")
        if env_ids.dtype != torch.int64 or env_ids.ndim != 1 or not bool(env_ids.is_contiguous()):
            raise TypeError("ray tensor env_ids must be a contiguous 1-D int64 tensor")
        if self._resolve_torch_device(env_ids.device, "env_ids") != device:
            raise ValueError(f"ray tensor env_ids must live on {device}, got {env_ids.device}")
        return env_ids

    def _resolve_tensor_ray_batch(
        self,
        ray_origins: Any,
        ray_directions: Any,
        torch: Any,
        device: str,
        rows: int,
    ) -> tuple[Any, Any, bool]:
        row_count = self._num_envs if rows is None else rows
        origins = self._validated_tensor(ray_origins, "ray_origins", torch, device=device)
        directions = self._validated_tensor(ray_directions, "ray_directions", torch, device=device)
        origin_shape = tuple(origins.shape)
        if origin_shape != tuple(directions.shape):
            raise ValueError(
                f"ray origins shape {origin_shape} does not match directions shape "
                f"{tuple(directions.shape)}"
            )
        allowed = ((self._num_rays, 3), (row_count, self._num_rays, 3))
        if origin_shape not in allowed:
            raise ValueError(
                f"ray tensor batches must have one of shapes {allowed}, got {origin_shape}"
            )
        return origins, directions, origin_shape == (self._num_rays, 3)

    def _tensor_result(
        self,
        torch: Any,
        device: str,
        rows: int,
        hit_point: bool,
        normal: bool,
        geom_id: bool,
        body_id: bool,
    ) -> RayTraceTensorResult:
        key: _TensorResultKey = (int(rows), hit_point, normal, geom_id, body_id)
        result = self._tensor_results.get(key)
        if result is None:
            distance = torch.empty((rows, self._num_rays), dtype=torch.float32, device=device)
            hit = torch.empty(distance.shape, dtype=torch.bool, device=device)
            result = RayTraceTensorResult(
                distance=distance,
                hit=hit,
                hit_point=(
                    torch.empty((*distance.shape, 3), dtype=torch.float32, device=device)
                    if hit_point
                    else None
                ),
                normal=(
                    torch.empty((*distance.shape, 3), dtype=torch.float32, device=device)
                    if normal
                    else None
                ),
                geom_id=(
                    torch.empty(distance.shape, dtype=torch.int32, device=device)
                    if geom_id
                    else None
                ),
                body_id=(
                    torch.empty(distance.shape, dtype=torch.int32, device=device)
                    if body_id
                    else None
                ),
            )
            self._tensor_results[key] = result
        return result

    def _launch_update_aabbs(self, rows: np.ndarray) -> None:
        """Launch the AABB refresh after pose kernels are already enqueued."""
        if not rows.size or not self._num_geoms:
            return
        assert self._env_map_dev is not None
        wp.launch(
            update_aabbs_batch_kernel,
            dim=(rows.size, self._num_geoms),
            inputs=[
                self._geom_types_dev,
                self._geom_sizes_dev,
                self._geom_mesh_ids_dev,
                self._mesh_aabb_center_dev,
                self._mesh_aabb_size_dev,
                self._geom_enabled_dev,
                self._geom_xpos_dev,
                self._geom_xmat_dev,
                self._env_map_dev,
                self._aabb_lowers_dev,
                self._aabb_uppers_dev,
            ],
            device=self._device,
        )

    @staticmethod
    def _mesh_aabbs(
        collision: RayCollisionDescription | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Local-frame AABB center/half extents for each public mesh asset."""

        count = len(collision.meshes) if collision is not None else 0
        center = np.zeros((count, 3), dtype=np.float32)
        size = np.zeros((count, 3), dtype=np.float32)
        if collision is None:
            return center, size
        for index, mesh in enumerate(collision.meshes):
            lower = mesh.points.min(axis=0)
            upper = mesh.points.max(axis=0)
            center[index] = (lower + upper) * 0.5
            size[index] = (upper - lower) * 0.5
        return center, size

    def _require_open(self) -> None:
        if self._closed:
            raise BackendError("uni_ray caster is closed")

    def _require_materialized(self) -> RaySceneDescription:
        if self._scene is None:
            raise BackendError("uni_ray caster must be materialized before querying")
        return self._scene

    def _geom_filter_array(self, ray_filter: RayTraceFilter | None) -> wp.array:
        """Return a persistent device mask for one immutable static filter."""

        if ray_filter is None or not ray_filter.has_constraints():
            assert self._all_geoms_allowed is not None
            return self._all_geoms_allowed
        cached = self._filter_masks.get(ray_filter)
        if cached is not None:
            return cached
        assert self._collision is not None
        mask = ray_filter.geom_mask(self._collision).astype(np.int32)
        result = wp.array(mask, dtype=wp.int32, device=self._device)
        self._filter_masks[ray_filter] = result
        return result

    def _validate_geometry_state(self, geometry: RayGeometryState) -> None:
        collision = self._collision
        if collision is None:
            raise BackendError("uni_ray dynamic geometry requires a collision descriptor")
        if geometry.num_envs != self._num_envs or geometry.geom_names != tuple(
            collision.geom_names
        ):
            raise ValueError("geometry state must match the materialized environment and geom axes")

    def _geometry_wp_arrays(
        self,
        *,
        sizes: Any,
        local_pos: Any,
        local_quat: Any,
        enabled: Any,
        mesh_ids: Any,
    ) -> tuple[wp.array, wp.array, wp.array, wp.array, wp.array]:
        if isinstance(sizes, np.ndarray):
            return (
                wp.array(np.ascontiguousarray(sizes), dtype=wp.vec3, device=self._device),
                wp.array(np.ascontiguousarray(local_pos), dtype=wp.vec3, device=self._device),
                wp.array(np.ascontiguousarray(local_quat), dtype=wp.vec4, device=self._device),
                wp.array(np.ascontiguousarray(enabled), dtype=wp.bool, device=self._device),
                wp.array(np.ascontiguousarray(mesh_ids), dtype=wp.int32, device=self._device),
            )
        return (
            wp.from_torch(sizes, dtype=wp.vec3),
            wp.from_torch(local_pos, dtype=wp.vec3),
            wp.from_torch(local_quat, dtype=wp.vec4),
            wp.from_torch(enabled, dtype=wp.bool),
            wp.from_torch(mesh_ids, dtype=wp.int32),
        )


__all__ = ["WarpRayCaster"]
