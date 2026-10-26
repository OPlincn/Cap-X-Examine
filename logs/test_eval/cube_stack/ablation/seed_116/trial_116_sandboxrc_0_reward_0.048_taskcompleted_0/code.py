# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment red and green cubes
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

# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])

# Approach from above
approach_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)

open_gripper()

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()

# Lift
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.2])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Place above green cube
place_pos = green_center + np.array([0.0, 0.0, green_extent[2] / 2 + 0.05])
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

# Lower onto green cube
place_final = green_center + np.array([0.0, 0.0, green_extent[2] / 2 + 0.02])
joints = solve_ik(place_final, grasp_quat)
move_to_joints(joints)

open_gripper()

# Retreat
retreat_pos = place_final + np.array([0.0, 0.0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)

# Code block 1
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Re-segment green cube to get its current position
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
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
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# Also locate red cube to know where it currently is (held by gripper)
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
red_best = max(red_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
red_pts = mask_to_points(red_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
red_center = red_obb["center"]
red_extent = red_obb["extent"]

grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])

# Target: place red cube on top of green cube
# Red cube bottom should sit at green top
green_top_z = green_center[2] + green_extent[2] / 2
# Current red cube center z; we want red center at green_top + red_extent/2
target_red_center = np.array([
    green_center[0],
    green_center[1],
    green_top_z + red_extent[2] / 2 + 0.005,
])

# The gripper holds the red cube; move gripper so red cube center reaches target.
# Offset between gripper position and red cube center (assume gripper holds near red center).
# We don't know exact gripper pos, but we can command gripper to target_red_center directly
# since the grasp was near the cube center.
target_pos = target_red_center

# Move above target first
above_pos = target_pos + np.array([0.0, 0.0, 0.1])
joints = solve_ik(above_pos, grasp_quat)
move_to_joints(joints)

# Lower to place position
joints = solve_ik(target_pos, grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = target_pos + np.array([0.0, 0.0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)

# Code block 2
import numpy as np

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
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

# Locate green cube
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# Locate red cube (currently held by gripper)
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
red_best = max(red_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
red_pts = mask_to_points(red_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
red_center = red_obb["center"]
red_extent = red_obb["extent"]

grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])

# Compute offset between gripper TCP and red cube center.
# We don't have direct TCP readout, but we can use the red cube center as proxy.
# Move red cube so its center is directly above green cube center, at proper height.
green_top_z = green_center[2] + green_extent[2] / 2
target_red_center = np.array([
    green_center[0],
    green_center[1],
    green_top_z + red_extent[2] / 2 + 0.01,
])

# Current red center is red_center. Compute delta and apply to gripper.
delta = target_red_center - red_center

# Move up first to clear green cube
lift_pos = red_center + np.array([0.0, 0.0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move horizontally above green cube
above_target = target_red_center + np.array([0.0, 0.0, 0.1])
joints = solve_ik(above_target, grasp_quat)
move_to_joints(joints)

# Lower to place position
joints = solve_ik(target_red_center, grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = target_red_center + np.array([0.0, 0.0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)