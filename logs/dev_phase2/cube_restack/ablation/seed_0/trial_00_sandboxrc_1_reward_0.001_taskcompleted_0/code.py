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

def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
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

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
print("red center", red_center, "extent", red_obb["extent"])
print("green center", green_center, "extent", green_obb["extent"])
```
Now I'll plan a grasp on the red cube, lift it, move it above the green cube, and place it.
```python
# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_R = best_T_world[:3, :3]

# Convert rotation to quaternion (wxyz)
def rot_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
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
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

grasp_quat = rot_to_quat(grasp_R)

# Approach from above: pre-grasp position
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.10])
joints_pre = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints_pre)

open_gripper()

# Move down to grasp
joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)
close_gripper()

# Lift the red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)
```
Now I'll move the red cube above the green cube and place it on top.
```python
# Determine placement position: above green cube center
# Green cube top surface height
green_top_z = green_center[2] + green_obb["extent"][2] / 2.0
red_half_z = red_obb["extent"][2] / 2.0

# Target: red cube center sits on top of green cube
place_center = np.array([green_center[0], green_center[1], green_top_z + red_half_z])

# Keep the same grasp orientation; offset so grasped cube center aligns with place_center
# The grasp offset between grasp_pos and red_center
grasp_offset = red_center - grasp_pos
place_grasp_pos = place_center - grasp_offset

# Move above placement
above_place = place_grasp_pos + np.array([0, 0, 0.12])
joints_above = solve_ik(above_place, grasp_quat)
move_to_joints(joints_above)

# Lower to place
joints_place = solve_ik(place_grasp_pos, grasp_quat)
move_to_joints(joints_place)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_grasp_pos + np.array([0, 0, 0.15])
joints_retreat = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints_retreat)

# Code block 1
import numpy as np

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

def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
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

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]

# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_R = best_T_world[:3, :3]

def rot_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
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
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

grasp_quat = rot_to_quat(grasp_R)

pre_grasp_pos = grasp_pos + np.array([0, 0, 0.10])
joints_pre = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints_pre)

open_gripper()

joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)

# Determine placement position: above green cube center
green_top_z = green_center[2] + green_obb["extent"][2] / 2.0
red_half_z = red_obb["extent"][2] / 2.0

place_center = np.array([green_center[0], green_center[1], green_top_z + red_half_z])

grasp_offset = red_center - grasp_pos
place_grasp_pos = place_center - grasp_offset

above_place = place_grasp_pos + np.array([0, 0, 0.12])
joints_above = solve_ik(above_place, grasp_quat)
move_to_joints(joints_above)

joints_place = solve_ik(place_grasp_pos, grasp_quat)
move_to_joints(joints_place)

open_gripper()

retreat_pos = place_grasp_pos + np.array([0, 0, 0.15])
joints_retreat = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints_retreat)