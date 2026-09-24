"""Visualize uni_ray's ray hit-point cloud with viser.

Self-contained demo: builds a small scene (ground plane, hfield terrain,
static tetrahedron mesh, and a freejoint body carrying sphere/box/capsule
geoms) through ``uni_ray.mjbatch.build_collision_description``, then animates
a spinning LiDAR-like ring pattern from one sensor origin per env while the
body oscillates. Each frame's world-frame hit points are pushed to a per-env
viser point cloud.

The caster's ``hit_point`` output is already world-frame: rays are supplied
in world coordinates and ``hit_point = origin + distance * direction`` in
that same frame (verified in tests/test_lifecycle.py), so no extra transform
is needed — misses are simply masked out.

Run:

    uv run python examples/visualize_pointcloud.py [--num-envs 4] [--num-rays 512]
        [--frames 0] [--port 8080] [--device cuda:0]

Then open the printed URL (http://localhost:8080 by default). Requires the
optional viser dependency: install the 'uni-ray[viz]' extra.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from unisim.ray_query import RayTraceOutputs

import uni_ray
from uni_ray.mjbatch import build_collision_description

SCENE_XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
    <hfield name="terrain" nrow="5" ncol="5" size="2 2 0.5 0.1"
            elevation="0.0 0.1 0.2 0.3 0.4
                       0.1 0.3 0.5 0.35 0.1
                       0.2 0.5 0.9 0.5 0.2
                       0.15 0.3 0.55 0.3 0.1
                       0.0 0.05 0.15 0.25 0.35"/>
  </asset>
  <worldbody>
    <geom type="plane" size="0 0 0.1"/>
    <geom type="hfield" hfield="terrain" pos="0 5 0"/>
    <geom type="mesh" mesh="tet" pos="3 -1 0"/>
    <body pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.3" pos="0.4 0 0"/>
      <geom type="box" size="0.12 0.18 0.1" pos="-0.4 0 0"/>
      <geom type="capsule" size="0.06 0.22" pos="0 0.5 0"/>
    </body>
  </worldbody>
</mujoco>
"""

ENV_COLORS = (
    (90, 200, 250),
    (250, 160, 60),
    (120, 230, 120),
    (240, 110, 170),
    (200, 160, 250),
    (250, 230, 90),
    (120, 200, 190),
    (250, 130, 100),
    (150, 150, 250),
    (170, 230, 160),
    (230, 190, 140),
    (160, 220, 240),
    (220, 140, 220),
    (190, 240, 120),
    (240, 180, 200),
    (140, 190, 230),
)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions a * b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec by the wxyz quaternion."""
    w, x, y, z = quat
    uv = np.array([y * vec[2] - z * vec[1], z * vec[0] - x * vec[2], x * vec[1] - y * vec[0]])
    uuv = np.array(
        [y * uv[2] - z * uv[1], z * uv[0] - x * uv[2], x * uv[1] - y * uv[0]]
    )
    return vec + 2.0 * (w * uv + uuv)


def _compose(
    body_pos: np.ndarray, body_quat: np.ndarray, local_pos: np.ndarray, local_quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World pose of a geom: body pose ∘ geom-local pose (wxyz)."""
    return (
        body_pos + _quat_rotate(body_quat, local_pos),
        _quat_mul(body_quat, local_quat),
    )


def _capsule_mesh(
    radius: float, half_length: float, segments: int = 16, rings: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Small z-aligned capsule mesh for viser context rendering."""
    rings_z = [-half_length, half_length]
    vertices = []
    # Cylinder rings.
    for z in rings_z:
        for s in range(segments):
            angle = 2.0 * np.pi * s / segments
            vertices.append((radius * np.cos(angle), radius * np.sin(angle), z))
    # Hemisphere rings (excluding the shared cylinder rings), then poles.
    for sign, z0 in ((-1.0, -half_length), (1.0, half_length)):
        for r in range(1, rings):
            phi = 0.5 * np.pi * r / rings
            z = z0 + sign * radius * np.sin(phi)
            rr = radius * np.cos(phi)
            for s in range(segments):
                angle = 2.0 * np.pi * s / segments
                vertices.append((rr * np.cos(angle), rr * np.sin(angle), z))
    south = len(vertices)
    vertices.append((0.0, 0.0, -half_length - radius))
    north = len(vertices)
    vertices.append((0.0, 0.0, half_length + radius))

    ring_count = 2 + 2 * (rings - 1)

    def vid(ring: int, s: int) -> int:
        return ring * segments + (s % segments)

    faces = []
    for ring in range(ring_count - 1):
        for s in range(segments):
            a, b = vid(ring, s), vid(ring, s + 1)
            c, d = vid(ring + 1, s), vid(ring + 1, s + 1)
            faces.extend([(a, c, b), (b, c, d)])
    last_ring = ring_count - 1
    for s in range(segments):
        faces.append((south, vid(0, s + 1), vid(0, s)))
        faces.append((north, vid(last_ring, s), vid(last_ring, s + 1)))
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def _lidar_directions(num_rays: int, phase: float) -> np.ndarray:
    """Spinning ring fan: ``num_rays`` world-frame unit directions.

    Eight fixed elevation rings cycled across the fan; the whole pattern
    rotates by ``phase`` around z.
    """
    elevations = np.deg2rad(np.linspace(-15.0, 12.0, 8))
    azimuths = 2.0 * np.pi * np.arange(num_rays) / num_rays + phase
    el = elevations[np.arange(num_rays) % elevations.size]
    return np.stack(
        [np.cos(el) * np.cos(azimuths), np.cos(el) * np.sin(azimuths), np.sin(el)], axis=1
    )


def _sensor_origins(num_envs: int) -> np.ndarray:
    """One LiDAR origin per env on a circle around the scene."""
    angles = 2.0 * np.pi * np.arange(num_envs) / num_envs
    return np.stack(
        [2.5 * np.cos(angles), 2.5 * np.sin(angles), np.full(num_envs, 1.6)], axis=1
    )


def _body_pose(t: float, num_bodies: int) -> tuple[np.ndarray, np.ndarray]:
    """Oscillating freejoint body (translation + slow yaw); world row fixed."""
    body_pos = np.zeros((num_bodies, 3))
    body_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_bodies, 1))
    yaw = 0.4 * np.sin(0.5 * t)
    body_pos[1] = [0.3 * np.sin(0.9 * t), 0.0, 1.0 + 0.35 * np.sin(1.3 * t)]
    body_quat[1] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    return body_pos, body_quat


def _import_viser():
    try:
        import viser
    except ImportError as error:
        from unisim.optional import OptionalDependencyError

        raise OptionalDependencyError(
            "the point-cloud example requires the optional dependency 'viser', "
            "which is not installed; install the 'uni-ray[viz]' extra to run "
            "examples/visualize_pointcloud.py"
        ) from error
    return viser


def run_demo(
    num_envs: int = 4,
    num_rays: int = 512,
    frames: int = 0,
    port: int = 8080,
    device: str | None = None,
    frame_dt: float = 1.0 / 30.0,
) -> dict:
    """Run the demo loop; returns a summary (frames rendered, last hit counts).

    ``frames = 0`` runs forever. ``frame_dt`` paces the loop and drives the
    animation time base; pass 0.0 for headless smoke runs.
    """
    viser = _import_viser()

    import mujoco

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    collision = build_collision_description(model)
    scene = collision.scene
    caster = uni_ray.create_ray_caster(
        num_envs=num_envs, num_rays=num_rays, collision=collision, device=device
    )
    caster.materialize(scene)

    server = viser.ViserServer(port=port)
    actual_port = server.get_port() if hasattr(server, "get_port") else port
    print(f"viser serving at http://localhost:{actual_port} (Ctrl+C to stop)")

    origins_per_env = _sensor_origins(num_envs)
    outputs = RayTraceOutputs(hit_point=True)
    clouds = []
    geom_handles: list[tuple[int, object]] = []  # (geom index, handle) for body geoms
    try:
        # Static scene context, added once.
        server.scene.add_box(
            "/scene/ground", dimensions=(16.0, 16.0, 0.02), color=(120, 120, 120), opacity=0.4
        )
        for geom_id, mesh_id in enumerate(collision.geom_mesh_ids):
            if mesh_id < 0:
                continue  # primitives below are handled individually
            mesh = collision.meshes[int(mesh_id)]
            handle = server.scene.add_mesh_simple(
                f"/scene/mesh_{geom_id}",
                vertices=mesh.points,
                faces=mesh.indices.reshape(-1, 3),
                color=(150, 170, 200),
                opacity=0.85,
            )
            if scene.geom_body_ids[geom_id] == 0:
                pos = scene.geom_local_pos[geom_id]
                quat = scene.geom_local_quat[geom_id]
                handle.position = np.asarray(pos, dtype=np.float32)
                handle.wxyz = np.asarray(quat, dtype=np.float32)
        # Primitive geoms: boxes/spheres as viser primitives, capsules as
        # small meshes (viser has no capsule primitive).
        from unisim.ray_query import RayGeomType

        for geom_id, geom_type in enumerate(scene.geom_types):
            if geom_type not in (
                RayGeomType.SPHERE,
                RayGeomType.BOX,
                RayGeomType.CAPSULE,
            ):
                continue
            size = scene.geom_sizes[geom_id]
            name = f"/scene/geom_{geom_id}"
            if geom_type == RayGeomType.SPHERE:
                handle = server.scene.add_icosphere(
                    name, radius=float(size[0]), color=(230, 190, 90)
                )
            elif geom_type == RayGeomType.BOX:
                handle = server.scene.add_box(
                    name, dimensions=tuple(2.0 * float(s) for s in size), color=(230, 190, 90)
                )
            else:
                vertices, faces = _capsule_mesh(float(size[0]), float(size[1]))
                handle = server.scene.add_mesh_simple(
                    name, vertices=vertices, faces=faces, color=(230, 190, 90)
                )
            geom_handles.append((geom_id, handle))

        clouds = [
            server.scene.add_point_cloud(
                f"/cloud/env_{env}",
                points=np.zeros((1, 3), dtype=np.float32),
                point_size=0.02,
                point_shape="circle",
                colors=ENV_COLORS[env % len(ENV_COLORS)],
            )
            for env in range(num_envs)
        ]

        last_hits = np.zeros(num_envs, dtype=np.int64)
        frame = 0
        while frames == 0 or frame < frames:
            start = time.perf_counter()
            t = frame * frame_dt
            body_pos, body_quat = _body_pose(t, scene.num_bodies)
            caster.update_pose(
                np.broadcast_to(body_pos, (num_envs, *body_pos.shape)),
                np.broadcast_to(body_quat, (num_envs, *body_quat.shape)),
            )
            # Moving body geoms: compose the body pose with the local pose.
            for geom_id, handle in geom_handles:
                body = int(scene.geom_body_ids[geom_id])
                pos, quat = _compose(
                    body_pos[body],
                    body_quat[body],
                    scene.geom_local_pos[geom_id],
                    scene.geom_local_quat[geom_id],
                )
                handle.position = np.asarray(pos, dtype=np.float32)
                handle.wxyz = np.asarray(quat, dtype=np.float32)

            for env in range(num_envs):
                directions = _lidar_directions(num_rays, phase=0.7 * t + env * 0.4)
                origins = np.broadcast_to(origins_per_env[env], directions.shape)
                result = caster.trace(
                    origins,
                    directions,
                    max_distance=30.0,
                    env_ids=[env],
                    outputs=outputs,
                )
                hit = np.asarray(result.hit)[0]
                # hit_point is world-frame already (world-frame rays in,
                # origin + distance * direction out); just drop the misses.
                points = np.asarray(result.hit_point)[0][hit]
                last_hits[env] = points.shape[0]
                clouds[env].points = (
                    points if points.size else np.zeros((1, 3), dtype=np.float32)
                ).astype(np.float32)
            frame += 1
            if frame_dt > 0.0:
                time.sleep(max(0.0, frame_dt - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        caster.close()
        server.stop()
    return {"frames": frame, "hits": last_hits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-rays", type=int, default=512)
    parser.add_argument("--frames", type=int, default=0, help="0 = run forever")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default=None, help="warp device (default: cuda:0 if present)")
    args = parser.parse_args()
    run_demo(
        num_envs=args.num_envs,
        num_rays=args.num_rays,
        frames=args.frames,
        port=args.port,
        device=args.device,
    )


if __name__ == "__main__":
    main()
