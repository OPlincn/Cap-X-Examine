# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_res, key=lambda d: d["score"])
green_best = max(green_res, key=lambda d: d["score"])
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
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```

Now I'll plan a grasp on the red cube (which is currently at the bottom). Since the green cube sits on top of the red cube, I need to grasp the red cube from the side to avoid the green cube.

```python
# Plan grasp on red cube
seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=seg)

# Transform to world frame
grasp_world = np.array([cam_ext @ T for T in grasp_poses])
scores = np.array(grasp_scores)

# Prefer grasps that approach from the side (gripper z-axis roughly horizontal)
# and are near the red cube center height
best_idx = None
best_score = -1e9
for i, T in enumerate(grasp_world):
    approach = T[:3, 2]  # gripper z axis
    # penalize downward/upward approaches
    horiz = np.linalg.norm(approach[:2])
    pos = T[:3, 3]
    # prefer grasp position near red center height
    height_pen = abs(pos[2] - red_center[2])
    s = scores[i] + 0.5 * horiz - 0.3 * height_pen
    if s > best_score:
        best_score = s
        best_idx = i

best_T = grasp_world[best_idx]
print("best grasp pos", best_T[:3, 3])
print("approach", best_T[:3, 2])
```

Now I'll move to a pre-grasp position, then grasp the red cube.

```python
from scipy.spatial.transform import Rotation as R

def mat_to_quat_wxyz(T):
    rot = R.from_matrix(T[:3, :3])
    q = rot.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])

grasp_pos = best_T[:3, 3]
grasp_quat = mat_to_quat_wxyz(best_T)

# Pre-grasp: back off along approach axis
approach = best_T[:3, 2]
pre_pos = grasp_pos - approach * 0.08

open_gripper()
joints = solve_ik(pre_pos, grasp_quat)
move_to_joints(joints)

# Move to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()
```

Now lift the red cube up and out, then place it on top of the green cube.

```python
# Lift red cube
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Compute placement position: on top of green cube
# Green cube top surface height
green_top_z = green_center[2] + green_extent[2] / 2.0
# Red cube half height (in its grasp orientation, use extent)
red_half_h = red_extent[2] / 2.0

place_pos = np.array([green_center[0], green_center[1], green_top_z + red_half_h + 0.01])
print("place pos", place_pos)

# Move above placement
above_pos = place_pos + np.array([0.0, 0.0, 0.12])
joints = solve_ik(above_pos, grasp_quat)
move_to_joints(joints)

# Descend to place
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos + np.array([0.0, 0.0, 0.12])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)

# Code block 1
import numpy as np
from scipy.spatial.transform import Rotation as R

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_res, key=lambda d: d["score"])
green_best = max(green_res, key=lambda d: d["score"])
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
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=seg)

grasp_world = np.array([cam_ext @ T for T in grasp_poses])
scores = np.array(grasp_scores)

best_idx = None
best_score = -1e9
for i, T in enumerate(grasp_world):
    approach = T[:3, 2]
    horiz = np.linalg.norm(approach[:2])
    pos = T[:3, 3]
    height_pen = abs(pos[2] - red_center[2])
    s = scores[i] + 0.5 * horiz - 0.3 * height_pen
    if s > best_score:
        best_score = s
        best_idx = i

best_T = grasp_world[best_idx]

def mat_to_quat_wxyz(T):
    rot = R.from_matrix(T[:3, :3])
    q = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

grasp_pos = best_T[:3, 3]
grasp_quat = mat_to_quat_wxyz(best_T)

approach = best_T[:3, 2]
pre_pos = grasp_pos - approach * 0.08

open_gripper()
joints = solve_ik(pre_pos, grasp_quat)
move_to_joints(joints)

joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()

lift_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

green_top_z = green_center[2] + green_extent[2] / 2.0
red_half_h = red_extent[2] / 2.0

place_pos = np.array([green_center[0], green_center[1], green_top_z + red_half_h + 0.01])

above_pos = place_pos + np.array([0.0, 0.0, 0.12])
joints = solve_ik(above_pos, grasp_quat)
move_to_joints(joints)

joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

open_gripper()

retreat_pos = place_pos + np.array([0.0, 0.0, 0.12])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)

# Code block 2
import numpy as np
from scipy.spatial.transform import Rotation as R

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_res, key=lambda d: d["score"])
green_best = max(green_res, key=lambda d: d["score"])
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
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

# Plan grasp on red cube
seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=seg)

grasp_world = np.array([cam_ext @ T for T in grasp_poses])
scores = np.array(grasp_scores)

# Prefer top-down grasps (approach axis pointing down) so we can place on top of green cube
best_idx = None
best_score = -1e9
for i, T in enumerate(grasp_world):
    approach = T[:3, 2]
    # top-down: approach z should be negative (pointing down)
    down = -approach[2]
    pos = T[:3, 3]
    height_pen = abs(pos[2] - red_center[2])
    s = scores[i] + 0.8 * down - 0.2 * height_pen
    if s > best_score:
        best_score = s
        best_idx = i

best_T = grasp_world[best_idx]
print("best grasp pos", best_T[:3, 3])
print("approach", best_T[:3, 2])

def mat_to_quat_wxyz(T):
    rot = R.from_matrix(T[:3, :3])
    q = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

grasp_pos = best_T[:3, 3]
grasp_quat = mat_to_quat_wxyz(best_T)

approach = best_T[:3, 2]
pre_pos = grasp_pos - approach * 0.08

open_gripper()
joints = solve_ik(pre_pos, grasp_quat)
move_to_joints(joints)

joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()

# Lift red cube
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Compute placement position: on top of green cube
green_top_z = green_center[2] + green_extent[2] / 2.0
red_half_h = red_extent[2] / 2.0

place_pos = np.array([green_center[0], green_center[1], green_top_z + red_half_h + 0.01])
print("place pos", place_pos)

above_pos = place_pos + np.array([0.0, 0.0, 0.12])
joints = solve_ik(above_pos, grasp_quat)
move_to_joints(joints)

joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

open_gripper()

retreat_pos = place_pos + np.array([0.0, 0.0, 0.12])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)