# Code block 0
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

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    pts_cam = np.stack([x, y, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```
Now I'll plan a grasp on the red cube. Since the green cube sits on top of the red cube, I need to grasp the red cube from the side. Let me plan grasps and pick one that approaches from the side.
```python
# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))

# Transform to world frame
grasp_world = np.array([cam_ext @ T for T in grasp_poses])

# We want a side grasp (approach roughly horizontal) to avoid the green cube on top.
# The grasp z-axis is the approach direction (typically the gripper's forward).
best_idx = None
best_score = -1
for i, T in enumerate(grasp_world):
    approach = T[:3, 2]  # approach axis
    # prefer horizontal approach (small z component)
    if abs(approach[2]) < 0.5:
        if grasp_scores[i] > best_score:
            best_score = grasp_scores[i]
            best_idx = i

if best_idx is None:
    best_idx = int(np.argmax(grasp_scores))

grasp_T = grasp_world[best_idx]
print("chosen grasp score", grasp_scores[best_idx])
print("grasp position", grasp_T[:3, 3])
print("approach", grasp_T[:3, 2])
```
Let me compute the grasp target and execute a pre-grasp, grasp, and lift sequence.
```python
from scipy.spatial.transform import Rotation as R

grasp_pos = grasp_T[:3, 3]
grasp_rot = grasp_T[:3, :3]

# Convert rotation matrix to quaternion wxyz
quat_xyzw = R.from_matrix(grasp_rot).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

# Pre-grasp: back off along approach axis
approach = grasp_T[:3, 2]
pre_grasp_pos = grasp_pos - approach * 0.10

open_gripper()
joints_pre = solve_ik(pre_grasp_pos, quat_wxyz)
move_to_joints(joints_pre)

# Move to grasp
joints_grasp = solve_ik(grasp_pos, quat_wxyz)
move_to_joints(joints_grasp)

close_gripper()

# Lift straight up
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, quat_wxyz)
move_to_joints(joints_lift)
```
Now I need to place the red cube on top of the green cube. Let me get a fresh observation to locate the green cube precisely.
```python
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("green center now", green_center, "extent", green_extent)
```
Now place the red cube on top of the green cube. The red cube is held in the gripper. I'll position it above the green cube center and lower it.
```python
# Place red cube on top of green cube.
# The red cube is grasped; its center is roughly at the gripper TCP.
# Target: red cube center should be at green_center + (0,0, green_half + red_half)
green_half_z = green_extent[2] / 2.0
red_half_z = red_extent[2] / 2.0

place_pos = green_center + np.array([0, 0, green_half_z + red_half_z + 0.005])

# Keep current grasp orientation
quat_xyzw = R.from_matrix(grasp_rot).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

# Move above place position first
above_pos = place_pos + np.array([0, 0, 0.10])
joints_above = solve_ik(above_pos, quat_wxyz)
move_to_joints(joints_above)

# Lower to place
joints_place = solve_ik(place_pos, quat_wxyz)
move_to_joints(joints_place)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos + np.array([0, 0, 0.12])
joints_retreat = solve_ik(retreat_pos, quat_wxyz)
move_to_joints(joints_retreat)

# Code block 1
import numpy as np
from scipy.spatial.transform import Rotation as R

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    pts_cam = np.stack([x, y, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
grasp_world = np.array([cam_ext @ T for T in grasp_poses])

best_idx = None
best_score = -1
for i, T in enumerate(grasp_world):
    approach = T[:3, 2]
    if abs(approach[2]) < 0.5:
        if grasp_scores[i] > best_score:
            best_score = grasp_scores[i]
            best_idx = i
if best_idx is None:
    best_idx = int(np.argmax(grasp_scores))

grasp_T = grasp_world[best_idx]
grasp_pos = grasp_T[:3, 3]
grasp_rot = grasp_T[:3, :3]
print("chosen grasp score", grasp_scores[best_idx])
print("grasp position", grasp_pos)
print("approach", grasp_T[:3, 2])

quat_xyzw = R.from_matrix(grasp_rot).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

approach = grasp_T[:3, 2]
pre_grasp_pos = grasp_pos - approach * 0.10

open_gripper()
joints_pre = solve_ik(pre_grasp_pos, quat_wxyz)
move_to_joints(joints_pre)

joints_grasp = solve_ik(grasp_pos, quat_wxyz)
move_to_joints(joints_grasp)

close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, quat_wxyz)
move_to_joints(joints_lift)

# Fresh observation to locate green cube
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("green center now", green_center, "extent", green_extent)

green_half_z = green_extent[2] / 2.0
red_half_z = red_extent[2] / 2.0
place_pos = green_center + np.array([0, 0, green_half_z + red_half_z + 0.005])

above_pos = place_pos + np.array([0, 0, 0.10])
joints_above = solve_ik(above_pos, quat_wxyz)
move_to_joints(joints_above)

joints_place = solve_ik(place_pos, quat_wxyz)
move_to_joints(joints_place)

open_gripper()

retreat_pos = place_pos + np.array([0, 0, 0.12])
joints_retreat = solve_ik(retreat_pos, quat_wxyz)
move_to_joints(joints_retreat)

# Code block 2
import numpy as np
from scipy.spatial.transform import Rotation as R

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    pts_cam = np.stack([x, y, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

# Segment both cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

# The green cube is on top of the red cube. We need to remove the green cube first,
# then grasp the red cube and place it on top of the green cube.
# Step 1: Grasp the green cube (top one) and move it aside.

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
grasp_world = np.array([cam_ext @ T for T in grasp_poses])

# Prefer top-down grasp for the green cube (it's on top, nothing above it)
best_idx = None
best_score = -1
for i, T in enumerate(grasp_world):
    approach = T[:3, 2]
    if approach[2] < -0.5:  # approaching from above (z-axis pointing down)
        if grasp_scores[i] > best_score:
            best_score = grasp_scores[i]
            best_idx = i
if best_idx is None:
    best_idx = int(np.argmax(grasp_scores))

grasp_T = grasp_world[best_idx]
grasp_pos = grasp_T[:3, 3]
grasp_rot = grasp_T[:3, :3]
print("green grasp score", grasp_scores[best_idx], "pos", grasp_pos, "approach", grasp_T[:3, 2])

quat_xyzw = R.from_matrix(grasp_rot).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

approach = grasp_T[:3, 2]
pre_grasp_pos = grasp_pos - approach * 0.10

open_gripper()
joints_pre = solve_ik(pre_grasp_pos, quat_wxyz)
move_to_joints(joints_pre)

joints_grasp = solve_ik(grasp_pos, quat_wxyz)
move_to_joints(joints_grasp)

close_gripper()

# Lift green cube up and move it aside (away from red cube)
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, quat_wxyz)
move_to_joints(joints_lift)

# Move green cube to the side (e.g., +y direction) to clear the red cube
aside_pos = lift_pos + np.array([0.0, 0.20, 0.0])
joints_aside = solve_ik(aside_pos, quat_wxyz)
move_to_joints(joints_aside)

# Lower and release green cube
place_green_pos = aside_pos - np.array([0, 0, 0.10])
joints_place_green = solve_ik(place_green_pos, quat_wxyz)
move_to_joints(joints_place_green)
open_gripper()

# Retreat
retreat_pos = place_green_pos + np.array([0, 0, 0.15])
joints_retreat = solve_ik(retreat_pos, quat_wxyz)
move_to_joints(joints_retreat)
print("green cube moved aside")

# Code block 3
import numpy as np
from scipy.spatial.transform import Rotation as R

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    pts_cam = np.stack([x, y, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

# Segment both cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

# Grasp the red cube (now free on the table)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
grasp_world = np.array([cam_ext @ T for T in grasp_poses])

# Prefer top-down grasp since red cube is now free on the table
best_idx = None
best_score = -1
for i, T in enumerate(grasp_world):
    approach = T[:3, 2]
    if approach[2] < -0.5:
        if grasp_scores[i] > best_score:
            best_score = grasp_scores[i]
            best_idx = i
if best_idx is None:
    best_idx = int(np.argmax(grasp_scores))

grasp_T = grasp_world[best_idx]
grasp_pos = grasp_T[:3, 3]
grasp_rot = grasp_T[:3, :3]
print("red grasp score", grasp_scores[best_idx], "pos", grasp_pos, "approach", grasp_T[:3, 2])

quat_xyzw = R.from_matrix(grasp_rot).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

approach = grasp_T[:3, 2]
pre_grasp_pos = grasp_pos - approach * 0.10

open_gripper()
joints_pre = solve_ik(pre_grasp_pos, quat_wxyz)
move_to_joints(joints_pre)

joints_grasp = solve_ik(grasp_pos, quat_wxyz)
move_to_joints(joints_grasp)

close_gripper()

# Lift red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, quat_wxyz)
move_to_joints(joints_lift)

# Fresh observation to locate green cube precisely
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("green center now", green_center, "extent", green_extent)

# Place red cube on top of green cube
green_half_z = green_extent[2] / 2.0
red_half_z = red_extent[2] / 2.0
place_pos = green_center + np.array([0, 0, green_half_z + red_half_z + 0.005])

above_pos = place_pos + np.array([0, 0, 0.10])
joints_above = solve_ik(above_pos, quat_wxyz)
move_to_joints(joints_above)

joints_place = solve_ik(place_pos, quat_wxyz)
move_to_joints(joints_place)

open_gripper()

retreat_pos = place_pos + np.array([0, 0, 0.12])
joints_retreat = solve_ik(retreat_pos, quat_wxyz)
move_to_joints(joints_retreat)