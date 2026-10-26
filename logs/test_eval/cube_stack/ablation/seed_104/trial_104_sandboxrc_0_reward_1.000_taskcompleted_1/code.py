# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

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

```python
def mask_to_world_points(mask, depth, K, cam_pose):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_pose @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_pose)
green_pts = mask_to_world_points(green_mask, depth, K, cam_pose)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_obb["extent"])
print("green center", green_center, "extent", green_extent)
```

```python
# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam
grasp_pos = best_T_world[:3, 3]
print("grasp pos world", grasp_pos, "score", grasp_scores[best_idx])
```

```python
# Approach above the red cube, then descend to grasp
approach_pos = grasp_pos + np.array([0.0, 0.0, 0.10])
# Use downward-facing orientation (gripper pointing down)
quat_down = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, 180 deg about x -> gripper down

open_gripper()
joints = solve_ik(approach_pos, quat_down)
move_to_joints(joints)

joints = solve_ik(grasp_pos, quat_down)
move_to_joints(joints)

close_gripper()
```

```python
# Lift the red cube
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(lift_pos, quat_down)
move_to_joints(joints)
```

```python
# Move above the green cube
green_top_z = green_center[2] + green_extent[2] / 2.0
place_above = np.array([green_center[0], green_center[1], green_top_z + 0.12])
joints = solve_ik(place_above, quat_down)
move_to_joints(joints)
```

```python
# Lower onto the green cube
place_pos = np.array([green_center[0], green_center[1], green_top_z + 0.03])
joints = solve_ik(place_pos, quat_down)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retract upward
retract_pos = place_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(retract_pos, quat_down)
move_to_joints(joints)

# Code block 1
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

def mask_to_world_points(mask, depth, K, cam_pose):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_pose @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_pose)
green_pts = mask_to_world_points(green_mask, depth, K, cam_pose)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_obb["extent"])
print("green center", green_center, "extent", green_extent)

# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam
grasp_pos = best_T_world[:3, 3]
print("grasp pos world", grasp_pos, "score", grasp_scores[best_idx])

quat_down = np.array([0.0, 1.0, 0.0, 0.0])

# Approach above the red cube, then descend to grasp
approach_pos = grasp_pos + np.array([0.0, 0.0, 0.10])
open_gripper()
joints = solve_ik(approach_pos, quat_down)
move_to_joints(joints)

joints = solve_ik(grasp_pos, quat_down)
move_to_joints(joints)

close_gripper()

# Lift the red cube
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(lift_pos, quat_down)
move_to_joints(joints)

# Move above the green cube
green_top_z = green_center[2] + green_extent[2] / 2.0
place_above = np.array([green_center[0], green_center[1], green_top_z + 0.12])
joints = solve_ik(place_above, quat_down)
move_to_joints(joints)

# Lower onto the green cube
place_pos = np.array([green_center[0], green_center[1], green_top_z + 0.03])
joints = solve_ik(place_pos, quat_down)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retract upward
retract_pos = place_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(retract_pos, quat_down)
move_to_joints(joints)