I'll start by getting an observation to understand the scene.

```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment the red and green cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

print("red box", red_best["box"], "score", red_best["score"])
print("green box", green_best["box"], "score", green_best["score"])
```

Now let me compute the 3D positions of both cubes using depth and camera extrinsics.

```python
def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```

Now I'll plan a grasp on the red cube (the one currently at the bottom, which we need to move on top). Actually, the goal is to place the red cube on top of the green cube. The green cube is currently on top of the red. So I need to pick up the red cube and place it on top of the green cube. But the green is on top of red... Let me first pick up the green cube to clear the way, or pick the red cube directly.

Since green is on top of red, I should first move the green cube away, then pick the red cube and place it on the green. Let me plan grasps.

```python
# Plan grasp for green cube first (to move it aside)
green_seg = green_mask.astype(np.int32)
green_grasps, green_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_seg)
print("green grasps", green_grasps.shape, "scores", green_scores[:5])
```

Let me think about the strategy more carefully. The green cube is on top of the red cube. To place red on top of green, I need to:
1. Move green cube off the red cube (to a clear spot on the table)
2. Pick up red cube
3. Place red cube on top of green cube

Let me get grasp poses for the green cube and move it aside.

```python
# Get best green grasp in world frame
best_g_idx = green_scores.argmax()
green_T_cam = green_grasps[best_g_idx]
green_T_world = cam_ext @ green_T_cam
print("green grasp world pos", green_T_world[:3, 3])
print("green grasp world rot", green_T_world[:3, :3])
```

Let me approach this step by step. First, I'll move the green cube to a clear location on the table.

```python
from scipy.spatial.transform import Rotation as R

def grasp_to_quat_wxyz(T):
    rot = T[:3, :3]
    quat_xyzw = R.from_matrix(rot).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

# Approach green cube from above
green_pos = green_T_world[:3, 3]
# Use downward-facing orientation for top grasp
down_quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper pointing down

# Pre-grasp above green
pre_green = green_pos + np.array([0, 0, 0.12])
joints = solve_ik(pre_green, down_quat)
move_to_joints(joints)
open_gripper()

# Move down to grasp
grasp_green = green_pos + np.array([0, 0, 0.005])
joints = solve_ik(grasp_green, down_quat)
move_to_joints(joints)
close_gripper()

# Lift
lift_green = green_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_green, down_quat)
move_to_joints(joints)
```

Now move green aside to a clear spot on the table (away from red cube).

```python
# Move green to a clear spot - offset in y direction
green_target = green_pos + np.array([0.0, 0.20, 0.0])
green_target[2] = green_pos[2]  # keep same height initially

# Move above target
above_target = green_target + np.array([0, 0, 0.15])
joints = solve_ik(above_target, down_quat)
move_to_joints(joints)

# Lower to place
place_green = green_target + np.array([0, 0, 0.02])
joints = solve_ik(place_green, down_quat)
move_to_joints(joints)

open_gripper()

# Retreat
joints = solve_ik(above_target, down_quat)
move_to_joints(joints)
```

Now pick up the red cube and place it on top of the green cube.

```python
# Re-observe to get updated positions
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```

Now grasp the red cube and place it on top of the green cube.

```python
# Grasp red cube
red_pos = red_center
pre_red = red_pos + np.array([0, 0, 0.12])
joints = solve_ik(pre_red, down_quat)
move_to_joints(joints)
open_gripper()

grasp_red = red_pos + np.array([0, 0, 0.005])
joints = solve_ik(grasp_red, down_quat)
move_to_joints(joints)
close_gripper()

lift_red = red_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_red, down_quat)
move_to_joints(joints)

# Place red on top of green
# Green cube top is at green_center[2] + green_extent[2]/2
green_top_z = green_center[2] + green_extent[2] / 2.0
red_half_height = red_extent[2] / 2.0
place_red_z = green_top_z + red_half_height + 0.005
place_red = np.array([green_center[0], green_center[1], place_red_z])

above_place = place_red + np.array([0, 0, 0.12])
joints = solve_ik(above_place, down_quat)
move_to_joints(joints)

joints = solve_ik(place_red, down_quat)
move_to_joints(joints)

open_gripper()

# Retreat
joints = solve_ik(above_place, down_quat)
move_to_joints(joints)
```