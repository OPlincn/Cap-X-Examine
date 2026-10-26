I'll start by getting an observation to understand the scene and locate the cubes.
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

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center, "green center", green_center, "green extent", green_extent)
```
Now I'll plan a grasp on the red cube and pick it up.
```python
# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
# Approach from above: use downward orientation
grasp_quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper pointing down

# Pre-grasp position above the red cube
pre_grasp_pos = grasp_pos + np.array([0.0, 0.0, 0.12])
open_gripper()
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift the red cube
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```
Now I'll move the red cube above the green cube and place it on top.
```python
# Place red cube on top of green cube
place_pos = green_center + np.array([0.0, 0.0, green_extent[2] / 2.0 + 0.03])

# Move above the green cube
above_place = place_pos + np.array([0.0, 0.0, 0.12])
joints = solve_ik(above_place, grasp_quat)
move_to_joints(joints)

# Lower to place position
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)
```