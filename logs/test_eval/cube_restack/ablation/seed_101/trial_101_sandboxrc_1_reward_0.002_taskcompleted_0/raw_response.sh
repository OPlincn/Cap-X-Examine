I'll start by getting an observation to understand the scene.

```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment the green and red cubes
green_masks = segment_sam3_text_prompt(rgb, "green cube")
red_masks = segment_sam3_text_prompt(rgb, "red cube")

green = max(green_masks, key=lambda d: d["score"])
red = max(red_masks, key=lambda d: d["score"])

def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xc = (xs - cx) * zs / fx
    yc = (ys - cy) * zs / fy
    pts_cam = np.stack([xc, yc, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

green_pts = mask_to_world_points(green["mask"], depth, K, cam_ext)
red_pts = mask_to_world_points(red["mask"], depth, K, cam_ext)

green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)

green_center = green_obb["center"]
red_center = red_obb["center"]
print("green center", green_center, "extent", green_obb["extent"])
print("red center", red_center, "extent", red_obb["extent"])
```

Now I'll plan grasps for the green cube (top one) and lift it aside.

```python
# Plan grasp for green cube
green_seg = green["mask"].astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

# Approach from above: use position, keep gripper pointing down
grasp_pos = best_T_world[:3, 3]
# Orientation: gripper pointing down (identity-ish with z down)
# Use quaternion for downward-facing gripper: rotate 180 about x
quat_down = np.array([0.0, 1.0, 0.0, 0.0])  # w,x,y,z

# Move above green cube
pre_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(pre_pos, quat_down)
move_to_joints(joints)

open_gripper()

# Descend to grasp
joints = solve_ik(grasp_pos, quat_down)
move_to_joints(joints)
close_gripper()

# Lift green cube
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.20])
joints = solve_ik(lift_pos, quat_down)
move_to_joints(joints)
```

Now move the green cube aside (to the side of the workspace).

```python
# Move green cube to the side
side_pos = lift_pos + np.array([0.0, 0.25, 0.0])
joints = solve_ik(side_pos, quat_down)
move_to_joints(joints)

# Lower and release green cube
place_pos = side_pos - np.array([0.0, 0.0, 0.12])
joints = solve_ik(place_pos, quat_down)
move_to_joints(joints)
open_gripper()

# Retreat
joints = solve_ik(place_pos + np.array([0.0, 0.0, 0.15]), quat_down)
move_to_joints(joints)
```

Now grasp the red cube and place it on top of the green cube.

```python
# Re-observe to get updated positions
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, "red cube")
red = max(red_masks, key=lambda d: d["score"])
red_pts = mask_to_world_points(red["mask"], depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
red_center = red_obb["center"]

green_masks = segment_sam3_text_prompt(rgb, "green cube")
green = max(green_masks, key=lambda d: d["score"])
green_pts = mask_to_world_points(green["mask"], depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_top_z = green_center[2] + green_obb["extent"][2] / 2.0

# Plan grasp for red cube
red_seg = red["mask"].astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
red_grasp_pos = best_T_world[:3, 3]

# Approach red cube
pre_pos = red_grasp_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(pre_pos, quat_down)
move_to_joints(joints)
open_gripper()

joints = solve_ik(red_grasp_pos, quat_down)
move_to_joints(joints)
close_gripper()

# Lift red cube
lift_pos = red_grasp_pos + np.array([0.0, 0.0, 0.20])
joints = solve_ik(lift_pos, quat_down)
move_to_joints(joints)

# Move above green cube
above_green = np.array([green_center[0], green_center[1], green_top_z + 0.15])
joints = solve_ik(above_green, quat_down)
move_to_joints(joints)

# Lower red cube onto green cube
place_pos = np.array([green_center[0], green_center[1], green_top_z + 0.02])
joints = solve_ik(place_pos, quat_down)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat
joints = solve_ik(place_pos + np.array([0.0, 0.0, 0.15]), quat_down)
move_to_joints(joints)
```