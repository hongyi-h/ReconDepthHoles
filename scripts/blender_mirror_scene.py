"""
Blender Mirror-Scene Synthesis Script
Generates training data for layered pointmap reconstruction.

Per scene outputs:
  - rgb_{view_id:03d}.png          — RGB image (H×W×3)
  - depth_first_{view_id:03d}.exr  — First-surface depth (ray's first hit)
  - depth_second_{view_id:03d}.exr — Secondary-path depth (reflected ray hit, 0 where no mirror)
  - mask_{view_id:03d}.png         — Material mask (0=diffuse, 1=mirror, 2=glass, 3=glossy)
  - camera_{view_id:03d}.npz       — Camera intrinsics (3×3) + extrinsics (4×4)
  - pointmap_first_{view_id:03d}.npy  — First-surface 3D points (H×W×3, world coords)
  - pointmap_second_{view_id:03d}.npy — Secondary-path 3D points (H×W×3, world coords)
  - mirror_planes.npy              — Mirror plane params (N_mirrors × 4: nx,ny,nz,d)
  - scene_meta.json                — Scene metadata

Usage:
  blender --background --python scripts/blender_mirror_scene.py -- \
    --output_dir data/synthetic/scene_0001 \
    --num_views 10 \
    --resolution 480 640 \
    --seed 42

Requirements:
  Blender >= 3.6 with Cycles renderer
"""

import bpy
import bmesh
import numpy as np
import os
import sys
import json
import argparse
import mathutils
from pathlib import Path


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--resolution", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_mirrors", type=int, default=1)
    parser.add_argument("--num_objects", type=int, default=5)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)


def setup_renderer(resolution):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'  # CPU for local test; change to 'GPU' on server
    scene.cycles.samples = 128
    scene.render.resolution_x = resolution[1]
    scene.render.resolution_y = resolution[0]
    scene.render.image_settings.file_format = 'PNG'
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_normal = True
    scene.view_layers[0].use_pass_object_index = True


def create_room(rng, size=5.0):
    """Create a simple room (floor + 3 walls, one side open for variety)."""
    # Floor
    bpy.ops.mesh.primitive_plane_add(size=size * 2, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    mat = create_diffuse_material("FloorMat", rng)
    floor.data.materials.append(mat)
    floor["material_class"] = 0  # diffuse

    # Back wall
    bpy.ops.mesh.primitive_plane_add(size=size * 2, location=(0, -size, size))
    wall = bpy.context.active_object
    wall.rotation_euler = (np.pi / 2, 0, 0)
    wall.name = "BackWall"
    mat = create_diffuse_material("WallMat", rng)
    wall.data.materials.append(mat)
    wall["material_class"] = 0

    # Left wall
    bpy.ops.mesh.primitive_plane_add(size=size * 2, location=(-size, 0, size))
    wall = bpy.context.active_object
    wall.rotation_euler = (0, -np.pi / 2, 0)
    wall.name = "LeftWall"
    wall.data.materials.append(mat)
    wall["material_class"] = 0

    # Right wall
    bpy.ops.mesh.primitive_plane_add(size=size * 2, location=(size, 0, size))
    wall = bpy.context.active_object
    wall.rotation_euler = (0, np.pi / 2, 0)
    wall.name = "RightWall"
    wall.data.materials.append(mat)
    wall["material_class"] = 0

    return size


def create_diffuse_material(name, rng):
    mat = bpy.data.materials.new(name=name)
    if not mat.use_nodes:
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    color = rng.uniform(0.1, 0.9, size=3).tolist() + [1.0]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.7, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    return mat


def create_mirror_material(name):
    mat = bpy.data.materials.new(name=name)
    if not mat.use_nodes:
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.0
    bsdf.inputs["Metallic"].default_value = 1.0
    return mat


def create_mirror(rng, room_size, mirror_id=0):
    """Place a mirror on one of the walls."""
    # Mirror dimensions
    width = rng.uniform(1.0, 2.5)
    height = rng.uniform(1.5, 3.0)

    # Choose wall: back=0, left=1, right=2
    wall_choice = rng.integers(0, 3)
    wall_names = {0: "BackWall", 1: "LeftWall", 2: "RightWall"}

    if wall_choice == 0:  # back wall
        loc = (rng.uniform(-room_size * 0.5, room_size * 0.5), -room_size + 0.01, rng.uniform(1.0, 2.5))
        rot = (np.pi / 2, 0, 0)
        normal = np.array([0.0, 1.0, 0.0])
    elif wall_choice == 1:  # left wall
        loc = (-room_size + 0.01, rng.uniform(-room_size * 0.5, room_size * 0.5), rng.uniform(1.0, 2.5))
        rot = (0, -np.pi / 2, 0)
        normal = np.array([1.0, 0.0, 0.0])
    else:  # right wall
        loc = (room_size - 0.01, rng.uniform(-room_size * 0.5, room_size * 0.5), rng.uniform(1.0, 2.5))
        rot = (0, np.pi / 2, 0)
        normal = np.array([-1.0, 0.0, 0.0])

    bpy.ops.mesh.primitive_plane_add(size=1, location=loc)
    mirror = bpy.context.active_object
    mirror.scale = (width, height, 1)
    mirror.rotation_euler = rot
    mirror.name = f"Mirror_{mirror_id}"

    mat = create_mirror_material(f"MirrorMat_{mirror_id}")
    mirror.data.materials.append(mat)
    mirror["material_class"] = 1  # mirror
    mirror["host_wall"] = wall_names[wall_choice]  # track which wall this mirror is on
    mirror.pass_index = 1  # for object index pass

    # Compute plane equation: n·x + d = 0
    d = -np.dot(normal, np.array(loc))
    plane_params = np.append(normal, d)

    return plane_params


def create_random_objects(rng, num_objects, room_size):
    """Place random primitive objects in the room."""
    for i in range(num_objects):
        loc = (
            rng.uniform(-room_size * 0.6, room_size * 0.6),
            rng.uniform(-room_size * 0.6, room_size * 0.3),
            rng.uniform(0.3, 2.0),
        )
        scale = rng.uniform(0.2, 0.8)

        prim_choice = rng.integers(0, 4)
        if prim_choice == 0:
            bpy.ops.mesh.primitive_cube_add(location=loc)
        elif prim_choice == 1:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=loc)
        elif prim_choice == 2:
            bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, location=loc)
        else:
            bpy.ops.mesh.primitive_cone_add(radius1=0.5, depth=1.0, location=loc)

        obj = bpy.context.active_object
        obj.name = f"Object_{i}"
        obj.scale = (scale, scale, scale)
        obj.rotation_euler = tuple(rng.uniform(0, np.pi, size=3))

        mat = create_diffuse_material(f"ObjMat_{i}", rng)
        obj.data.materials.append(mat)
        obj["material_class"] = 0
        obj.pass_index = 0


def create_lighting(rng):
    """Add area lights for realistic illumination."""
    # Main area light (ceiling)
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 4.5))
    light = bpy.context.active_object
    light.data.energy = rng.uniform(200, 500)
    light.data.size = 3.0

    # Secondary fill light
    bpy.ops.object.light_add(
        type='AREA',
        location=(rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(2, 4))
    )
    light2 = bpy.context.active_object
    light2.data.energy = rng.uniform(50, 150)
    light2.data.size = 1.5


def generate_camera_poses(rng, num_views, room_size, mirror_locations=None):
    """Generate camera poses that look toward the mirror area.

    If mirror_locations is provided, cameras will look toward/through the mirrors
    to ensure the reflected virtual scene is visible.
    """
    poses = []
    for i in range(num_views):
        # Camera position: in front of mirror area, at moderate distance
        cam_loc = mathutils.Vector((
            rng.uniform(-room_size * 0.3, room_size * 0.3),
            rng.uniform(room_size * 0.1, room_size * 0.6),
            rng.uniform(1.2, 2.2),
        ))

        # Look toward the mirror (if known) with some randomness
        if mirror_locations is not None and len(mirror_locations) > 0:
            # Pick a mirror to look at
            mirror_loc = mirror_locations[i % len(mirror_locations)]
            # Add some jitter so camera doesn't stare dead-center every time
            target = mathutils.Vector((
                mirror_loc[0] + rng.uniform(-0.5, 0.5),
                mirror_loc[1] + rng.uniform(-0.5, 0.5),
                mirror_loc[2] + rng.uniform(-0.3, 0.3),
            ))
        else:
            # Fallback: look toward center-back of room
            target = mathutils.Vector((
                rng.uniform(-1, 1),
                rng.uniform(-room_size * 0.5, -room_size * 0.2),
                rng.uniform(0.8, 2.0),
            ))

        direction = target - cam_loc
        rot_quat = direction.to_track_quat('-Z', 'Y')

        poses.append((cam_loc, rot_quat))

    return poses


def get_camera_matrices(camera, scene):
    """Extract intrinsic and extrinsic matrices from Blender camera."""
    render = scene.render
    width = render.resolution_x
    height = render.resolution_y

    # Intrinsics
    focal_length = camera.data.lens
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height

    fx = focal_length * width / sensor_width
    fy = focal_length * height / sensor_height
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Extrinsics (world-to-camera)
    # Blender camera looks down -Z, with Y up
    cam_matrix_world = np.array(camera.matrix_world)

    # Convert Blender convention to OpenCV convention
    # Blender: X-right, Y-up, Z-back
    # OpenCV: X-right, Y-down, Z-forward
    blender_to_cv = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    extrinsic = blender_to_cv @ np.linalg.inv(cam_matrix_world)

    return K, extrinsic


def depth_to_pointmap(depth, K, extrinsic):
    """Convert depth map to world-space pointmap."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth

    points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(depth)], axis=-1)  # (H, W, 4)

    # Camera-to-world transform
    cam_to_world = np.linalg.inv(extrinsic)
    points_world = (cam_to_world @ points_cam.reshape(-1, 4).T).T.reshape(H, W, 4)

    return points_world[:, :, :3]


def reflect_matrix_about_plane(normal, d):
    """
    Compute 4x4 reflection matrix about plane n·x + d = 0.
    normal: unit normal (3,), d: offset scalar
    Reflects point p as: p' = p - 2*(n·p + d)*n
    """
    n = np.array(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    # For affine reflection: [I - 2*n*n^T | -2*d*n]
    R = np.eye(4, dtype=np.float64)
    R[:3, :3] -= 2.0 * np.outer(n, n)
    R[:3, 3] = -2.0 * d * n
    return R


def _get_depth_from_raycast(scene, cam_obj, resolution):
    """Get per-pixel depth via scene.ray_cast (C++ accelerated, ~2s for 480x640).

    Returns depth as distance from camera origin to hit point (euclidean, not Z-buffer).
    """
    H, W = resolution
    depsgraph = bpy.context.evaluated_depsgraph_get()

    cam_matrix = cam_obj.matrix_world
    cam_loc = cam_matrix.translation
    cam_rot = cam_matrix.to_3x3()

    focal = cam_obj.data.lens
    sensor_w = cam_obj.data.sensor_width
    fx = focal * W / sensor_w
    fy = fx  # square pixels assumption (close enough for pilot)
    cx, cy = W / 2.0, H / 2.0

    depth = np.zeros((H, W), dtype=np.float32)

    for v in range(H):
        for u in range(W):
            dx = (u - cx) / fx
            dy = -(v - cy) / fy  # image Y is down, camera Y is up
            direction = cam_rot @ mathutils.Vector((dx, dy, -1.0))
            direction.normalize()

            result, location, normal, index, obj, matrix = scene.ray_cast(
                depsgraph, cam_loc, direction
            )
            if result:
                depth[v, u] = (location - cam_loc).length

    return depth


def render_scene(output_dir, camera_poses, mirror_planes, resolution):
    """Render all views and save outputs including secondary-path GT."""
    scene = bpy.context.scene
    os.makedirs(output_dir, exist_ok=True)

    # Setup camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 35  # 35mm focal length
    cam_data.sensor_width = 36
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Collect mirror objects for toggling visibility
    mirror_objs = [obj for obj in bpy.data.objects if obj.get("material_class") == 1]

    for view_id, (loc, rot) in enumerate(camera_poses):
        cam_obj.location = loc
        cam_obj.rotation_mode = 'QUATERNION'
        cam_obj.rotation_quaternion = rot
        bpy.context.view_layer.update()

        # Get camera matrices
        K, extrinsic = get_camera_matrices(cam_obj, scene)

        # --- First-surface render ---
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = os.path.join(output_dir, f"rgb_{view_id:03d}")
        bpy.ops.render.render(write_still=True)

        # Get first-surface depth via ray casting
        depth_first = _get_depth_from_raycast(scene, cam_obj, resolution)
        np.save(os.path.join(output_dir, f"depth_first_{view_id:03d}.npy"), depth_first)

        # --- Secondary-path depth (virtual camera + raycast) ---
        for plane_id, plane_params in enumerate(mirror_planes):
            normal = plane_params[:3]
            d = plane_params[3]

            # Reflect camera position and orientation
            cam_world = np.array(cam_obj.matrix_world)
            reflect_mat = reflect_matrix_about_plane(normal, d)
            cam_pos = np.array(cam_obj.location)
            virtual_pos = (reflect_mat @ np.append(cam_pos, 1.0))[:3]

            cam_forward = -np.array(cam_world[:3, 2])
            cam_up = np.array(cam_world[:3, 1])
            reflect_linear = reflect_mat[:3, :3]
            virtual_forward = reflect_linear @ cam_forward
            virtual_up = reflect_linear @ cam_up

            virtual_forward_norm = virtual_forward / np.linalg.norm(virtual_forward)
            virtual_up_norm = virtual_up / np.linalg.norm(virtual_up)
            virtual_right = np.cross(virtual_forward_norm, virtual_up_norm)
            virtual_right /= np.linalg.norm(virtual_right)
            virtual_up_norm = np.cross(virtual_right, virtual_forward_norm)

            rot_mat = np.eye(3)
            rot_mat[:, 0] = virtual_right
            rot_mat[:, 1] = virtual_up_norm
            rot_mat[:, 2] = -virtual_forward_norm

            # Set virtual camera for raycast
            orig_matrix = cam_obj.matrix_world.copy()
            virtual_world = np.eye(4)
            virtual_world[:3, :3] = rot_mat
            virtual_world[:3, 3] = virtual_pos
            cam_obj.matrix_world = mathutils.Matrix(virtual_world.tolist())
            bpy.context.view_layer.update()

            # Hide mirror + room for raycast (virtual cam is outside room)
            room_parts = ["Floor", "BackWall", "LeftWall", "RightWall"]
            for m_obj in mirror_objs:
                m_obj.hide_render = True
            for part_name in room_parts:
                part_obj = bpy.data.objects.get(part_name)
                if part_obj:
                    part_obj.hide_render = True

            # Raycast depth from virtual camera
            depth_secondary = _get_depth_from_raycast(scene, cam_obj, resolution)
            np.save(
                os.path.join(output_dir, f"depth_secondary_{view_id:03d}_mirror{plane_id:02d}.npy"),
                depth_secondary,
            )

            # Restore visibility
            for m_obj in mirror_objs:
                m_obj.hide_render = False
            for part_name in room_parts:
                part_obj = bpy.data.objects.get(part_name)
                if part_obj:
                    part_obj.hide_render = False

            # Restore camera
            cam_obj.matrix_world = orig_matrix
            bpy.context.view_layer.update()

        # Save camera params
        np.savez(
            os.path.join(output_dir, f"camera_{view_id:03d}.npz"),
            intrinsic=K,
            extrinsic=extrinsic,
            location=np.array(loc),
            rotation=np.array(rot),
        )

    # Save mirror planes
    if len(mirror_planes) > 0:
        np.save(os.path.join(output_dir, "mirror_planes.npy"), np.array(mirror_planes))

    # Save scene metadata
    meta = {
        "num_views": len(camera_poses),
        "resolution": resolution,
        "num_mirrors": len(mirror_planes),
        "mirror_planes": [p.tolist() for p in mirror_planes],
    }
    with open(os.path.join(output_dir, "scene_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Generating mirror scene: seed={args.seed}, views={args.num_views}")
    print(f"Output: {args.output_dir}")

    # Build scene
    clear_scene()
    setup_renderer(args.resolution)
    room_size = create_room(rng)

    # Add mirrors
    mirror_planes = []
    for i in range(args.num_mirrors):
        plane = create_mirror(rng, room_size, mirror_id=i)
        mirror_planes.append(plane)

    # Add objects
    create_random_objects(rng, args.num_objects, room_size)

    # Add lighting
    create_lighting(rng)

    # Generate camera poses (looking toward mirrors)
    mirror_locations = [
        list(bpy.data.objects[f"Mirror_{i}"].location)
        for i in range(args.num_mirrors)
    ]
    camera_poses = generate_camera_poses(rng, args.num_views, room_size, mirror_locations)

    # Render
    render_scene(args.output_dir, camera_poses, mirror_planes, args.resolution)

    print(f"Done! Scene saved to {args.output_dir}")


if __name__ == "__main__":
    main()
