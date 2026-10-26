# Code block 0
I'll start by getting an observation to understand the scene and locate the cubes.

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
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

# The green cube is on top of red. We need to move red on top of green.
# Strategy: grasp red cube, lift it, move it above green cube, place it down, open gripper.

# Plan grasp on red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_quat = None
# Convert rotation matrix to quaternion (wxyz)
R = best_T_world[:3, :3]
def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

grasp_quat = rot_to_quat(R)

# Pre-grasp approach: move above the grasp position
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.1])
open_gripper()
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift the red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Now move above the green cube. Green cube top is at green_center + extent/2 in z
green_top_z = green_center[2] + green_extent[2] / 2.0
# Place red cube so its bottom sits on green top. Red cube half-height ~ red_extent/2
# The grasp point is roughly at red center; offset from grasp to red bottom
red_half_height = red_extent[2] / 2.0
# Target: red center should be at green_top_z + red_half_height
target_red_center_z = green_top_z + red_half_height
# Grasp offset: grasp_pos z relative to red_center z
grasp_offset_z = grasp_pos[2] - red_center[2]
target_grasp_z = target_red_center_z + grasp_offset_z

place_pos = np.array([green_center[0], green_center[1], target_grasp_z + 0.1])
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

# Lower to place
place_pos_final = np.array([green_center[0], green_center[1], target_grasp_z])
joints = solve_ik(place_pos_final, grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos_final + np.array([0, 0, 0.15])
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
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Green is on top of red. We must first remove green, then move red on top of green.
# Strategy: grasp green, lift and set aside; grasp red, place on green; then place green? 
# Simpler: grasp green, lift it, move red under? Hard. Instead:
# 1) Grasp green, lift high, set it aside on table.
# 2) Grasp red, place it where green was (on table).
# 3) Grasp green, place it on top of red? No - goal is red on top of green.
# Correct: 1) grasp green, set aside. 2) grasp red, place at green's original location.
# 3) grasp green, place on top of red? That puts green on red. Wrong.
# We need red on top of green. So: 1) grasp green, set aside. 2) grasp red, set aside.
# 3) place green at original spot. 4) place red on top of green.

def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

def plan_and_grasp(mask, lift_height=0.15):
    seg = mask.astype(np.int32)
    grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=seg)
    best_idx = grasp_scores.argmax()
    best_T_world = cam_ext @ grasp_poses[best_idx]
    grasp_pos = best_T_world[:3, 3]
    grasp_quat = rot_to_quat(best_T_world[:3, :3])
    pre = grasp_pos + np.array([0, 0, 0.1])
    open_gripper()
    move_to_joints(solve_ik(pre, grasp_quat))
    move_to_joints(solve_ik(grasp_pos, grasp_quat))
    close_gripper()
    lift = grasp_pos + np.array([0, 0, lift_height])
    move_to_joints(solve_ik(lift, grasp_quat))
    return grasp_pos, grasp_quat

# Step 1: grasp green and set aside
green_grasp_pos, green_grasp_quat = plan_and_grasp(green_mask)
setaside_green = np.array([green_center[0] + 0.15, green_center[1] + 0.15, green_grasp_pos[2]])
move_to_joints(solve_ik(setaside_green, green_grasp_quat))
move_to_joints(solve_ik(setaside_green - np.array([0, 0, 0.1]), green_grasp_quat))
open_gripper()
move_to_joints(solve_ik(setaside_green + np.array([0, 0, 0.1]), green_grasp_quat))

# Step 2: grasp red and set aside
red_grasp_pos, red_grasp_quat = plan_and_grasp(red_mask)
setaside_red = np.array([red_center[0] - 0.15, red_center[1] + 0.15, red_grasp_pos[2]])
move_to_joints(solve_ik(setaside_red, red_grasp_quat))
move_to_joints(solve_ik(setaside_red - np.array([0, 0, 0.1]), red_grasp_quat))
open_gripper()
move_to_joints(solve_ik(setaside_red + np.array([0, 0, 0.1]), red_grasp_quat))

# Step 3: place green back at original location (on table)
green_table_z = green_center[2] - green_extent[2] / 2.0
green_target_center_z = green_table_z + green_extent[2] / 2.0
green_grasp_offset_z = green_grasp_pos[2] - green_center[2]
green_place_z = green_target_center_z + green_grasp_offset_z
green_place = np.array([green_center[0], green_center[1], green_place_z + 0.1])
move_to_joints(solve_ik(green_place, green_grasp_quat))
move_to_joints(solve_ik(np.array([green_center[0], green_center[1], green_place_z]), green_grasp_quat))
open_gripper()
move_to_joints(solve_ik(green_place, green_grasp_quat))

# Step 4: place red on top of green
green_top_z = green_target_center_z + green_extent[2] / 2.0
red_half_height = red_extent[2] / 2.0
red_target_center_z = green_top_z + red_half_height
red_grasp_offset_z = red_grasp_pos[2] - red_center[2]
red_place_z = red_target_center_z + red_grasp_offset_z
red_place = np.array([green_center[0], green_center[1], red_place_z + 0.1])
move_to_joints(solve_ik(red_place, red_grasp_quat))
move_to_joints(solve_ik(np.array([green_center[0], green_center[1], red_place_z]), red_grasp_quat))
open_gripper()
move_to_joints(solve_ik(red_place, red_grasp_quat))