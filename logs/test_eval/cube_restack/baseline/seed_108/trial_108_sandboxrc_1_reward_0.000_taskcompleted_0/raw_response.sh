I'll start by getting an observation to understand the scene, then segment the cubes to locate them.

```python
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

def mask_to_world_points(mask, depth, K, cam_pose):
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
    pts_world = (cam_pose @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_pose)
green_pts = mask_to_world_points(green_mask, depth, K, cam_pose)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```

Now I'll plan a grasp on the red cube (the bottom one) and lift it off the green cube.

```python
# Plan grasp on red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_quat = None
# Convert rotation matrix to quaternion (wxyz)
R = best_T_world[:3, :3]
def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

grasp_quat = rot_to_quat(R)

# Approach from above: pre-grasp position
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.10])
joints_pre = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints_pre)

open_gripper()

# Move down to grasp
joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)
close_gripper()

# Lift red cube up
lift_pos = grasp_pos + np.array([0, 0, 0.20])
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)
```

Now I need to place the red cube on top of the green cube. Let me get a new observation to locate the green cube's current position (it may have shifted).

```python
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_world_points(green_mask, depth, K, cam_pose)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# Place red cube on top of green cube
place_pos = green_center + np.array([0, 0, green_extent[2]/2 + red_extent[2]/2 + 0.005])

# Keep the grasp orientation for placement
joints_place_above = solve_ik(place_pos + np.array([0, 0, 0.10]), grasp_quat)
move_to_joints(joints_place_above)

joints_place = solve_ik(place_pos, grasp_quat)
move_to_joints(joints_place)

open_gripper()

# Retreat upward
joints_retreat = solve_ik(place_pos + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints_retreat)
```