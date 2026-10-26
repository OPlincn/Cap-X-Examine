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
    ys, xs = np.where(mask)
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
Now I'll plan a grasp on the green cube (top), lift it off, then grasp the red cube and place it on the green cube.
```python
# Plan grasp on green cube (top cube) to remove it
green_seg = green_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

# Grasp pose in world frame
grasp_pos = best_T_world[:3, 3]
grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])  # will refine below

# Extract orientation from rotation matrix -> quaternion wxyz
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

# Approach from above: move to pre-grasp position
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift green cube up
lift_pos = grasp_pos + np.array([0, 0, 0.2])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```
Now I'll set the green cube aside, then grasp the red cube and place it on top of the green cube.
```python
# Move green cube to a side location (set aside)
side_pos = green_center + np.array([0.15, 0.0, 0.15])
joints = solve_ik(side_pos, grasp_quat)
move_to_joints(joints)

# Lower and release green cube
place_green_pos = green_center + np.array([0.15, 0.0, 0.02])
joints = solve_ik(place_green_pos, grasp_quat)
move_to_joints(joints)
open_gripper()

# Lift up to clear
joints = solve_ik(place_green_pos + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints)
```
Now grasp the red cube and place it on top of the green cube.
```python
# Plan grasp on red cube
red_seg = red_mask.astype(np.int32)
red_grasp_poses, red_grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
red_best_idx = red_grasp_scores.argmax()
red_T_world = cam_ext @ red_grasp_poses[red_best_idx]

red_grasp_pos = red_T_world[:3, 3]
red_R = red_T_world[:3, :3]
red_grasp_quat = rot_to_quat(red_R)

# Pre-grasp
joints = solve_ik(red_grasp_pos + np.array([0, 0, 0.1]), red_grasp_quat)
move_to_joints(joints)

# Grasp
joints = solve_ik(red_grasp_pos, red_grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
joints = solve_ik(red_grasp_pos + np.array([0, 0, 0.2]), red_grasp_quat)
move_to_joints(joints)

# Place red cube on top of green cube (green is now at side_pos location)
# Green cube top is at side_pos + half green height
green_top_z = place_green_pos[2] + green_extent[2] / 2.0
red_place_pos = np.array([side_pos[0], side_pos[1], green_top_z + red_extent[2] / 2.0 + 0.01])

joints = solve_ik(red_place_pos + np.array([0, 0, 0.1]), red_grasp_quat)
move_to_joints(joints)

joints = solve_ik(red_place_pos, red_grasp_quat)
move_to_joints(joints)

# Open gripper to release red cube
open_gripper()

# Lift up to clear
joints = solve_ik(red_place_pos + np.array([0, 0, 0.15]), red_grasp_quat)
move_to_joints(joints)

# Code block 1
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
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

# Segment cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Green is on top of red. Grasp green and set it aside.
green_seg = green_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_seg)
best_T_world = cam_ext @ grasp_poses[grasp_scores.argmax()]
grasp_pos = best_T_world[:3, 3]
grasp_quat = rot_to_quat(best_T_world[:3, :3])

# Approach, grasp, lift green
move_to_joints(solve_ik(grasp_pos + np.array([0, 0, 0.1]), grasp_quat))
move_to_joints(solve_ik(grasp_pos, grasp_quat))
close_gripper()
move_to_joints(solve_ik(grasp_pos + np.array([0, 0, 0.2]), grasp_quat))

# Set green aside
side_pos = green_center + np.array([0.15, 0.0, 0.0])
move_to_joints(solve_ik(side_pos + np.array([0, 0, 0.15]), grasp_quat))
move_to_joints(solve_ik(side_pos, grasp_quat))
open_gripper()
move_to_joints(solve_ik(side_pos + np.array([0, 0, 0.15]), grasp_quat))

# Grasp red cube
red_seg = red_mask.astype(np.int32)
red_grasp_poses, red_grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
red_T_world = cam_ext @ red_grasp_poses[red_grasp_scores.argmax()]
red_grasp_pos = red_T_world[:3, 3]
red_grasp_quat = rot_to_quat(red_T_world[:3, :3])

move_to_joints(solve_ik(red_grasp_pos + np.array([0, 0, 0.1]), red_grasp_quat))
move_to_joints(solve_ik(red_grasp_pos, red_grasp_quat))
close_gripper()
move_to_joints(solve_ik(red_grasp_pos + np.array([0, 0, 0.2]), red_grasp_quat))

# Place red on top of green (green now at side_pos)
green_top_z = side_pos[2] + green_extent[2] / 2.0
red_place_pos = np.array([side_pos[0], side_pos[1], green_top_z + red_extent[2] / 2.0 + 0.01])

move_to_joints(solve_ik(red_place_pos + np.array([0, 0, 0.1]), red_grasp_quat))
move_to_joints(solve_ik(red_place_pos, red_grasp_quat))
open_gripper()
move_to_joints(solve_ik(red_place_pos + np.array([0, 0, 0.15]), red_grasp_quat))

# Code block 2
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
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

# Segment cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Grasp red cube
red_seg = red_mask.astype(np.int32)
red_grasp_poses, red_grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
red_T_world = cam_ext @ red_grasp_poses[red_grasp_scores.argmax()]
red_grasp_pos = red_T_world[:3, 3]
red_grasp_quat = rot_to_quat(red_T_world[:3, :3])

move_to_joints(solve_ik(red_grasp_pos + np.array([0, 0, 0.1]), red_grasp_quat))
move_to_joints(solve_ik(red_grasp_pos, red_grasp_quat))
close_gripper()
move_to_joints(solve_ik(red_grasp_pos + np.array([0, 0, 0.2]), red_grasp_quat))

# Place red on top of green (green is on table)
green_top_z = green_center[2] + green_extent[2] / 2.0
red_place_pos = np.array([green_center[0], green_center[1], green_top_z + red_extent[2] / 2.0 + 0.01])

move_to_joints(solve_ik(red_place_pos + np.array([0, 0, 0.1]), red_grasp_quat))
move_to_joints(solve_ik(red_place_pos, red_grasp_quat))
open_gripper()
move_to_joints(solve_ik(red_place_pos + np.array([0, 0, 0.15]), red_grasp_quat))

# Code block 3
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
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

# Segment cubes to get green cube location
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# The red cube is currently held in the gripper. Move it above the green cube and lower it.
# Use a downward-facing gripper orientation (identity quaternion) for placement.
place_quat = np.array([1.0, 0.0, 0.0, 0.0])

# Target: red cube bottom should rest on green cube top.
# Estimate red cube half-height from a typical small cube; use green extent as proxy.
red_half = green_extent[2] / 2.0
green_top_z = green_center[2] + green_extent[2] / 2.0
# Gripper holds cube below the hand; place hand so cube sits on green top.
# Move to a position above green center, then lower.
target_xy = np.array([green_center[0], green_center[1]])

# First move above the green cube
above_pos = np.array([target_xy[0], target_xy[1], green_top_z + 0.15])
move_to_joints(solve_ik(above_pos, place_quat))

# Lower to place the red cube on top of green cube
place_pos = np.array([target_xy[0], target_xy[1], green_top_z + red_half + 0.02])
move_to_joints(solve_ik(place_pos, place_quat))

# Open gripper to release red cube
open_gripper()

# Lift up to clear
move_to_joints(solve_ik(place_pos + np.array([0, 0, 0.15]), place_quat))

# Code block 4
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
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

# Segment cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Grasp red cube
red_seg = red_mask.astype(np.int32)
red_grasp_poses, red_grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
red_T_world = cam_ext @ red_grasp_poses[red_grasp_scores.argmax()]
red_grasp_pos = red_T_world[:3, 3]
red_grasp_quat = rot_to_quat(red_T_world[:3, :3])

move_to_joints(solve_ik(red_grasp_pos + np.array([0, 0, 0.1]), red_grasp_quat))
move_to_joints(solve_ik(red_grasp_pos, red_grasp_quat))
close_gripper()
move_to_joints(solve_ik(red_grasp_pos + np.array([0, 0, 0.2]), red_grasp_quat))

# Place red on top of green using downward orientation
place_quat = np.array([1.0, 0.0, 0.0, 0.0])
green_top_z = green_center[2] + green_extent[2] / 2.0
red_half = red_extent[2] / 2.0
place_pos = np.array([green_center[0], green_center[1], green_top_z + red_half + 0.02])

move_to_joints(solve_ik(place_pos + np.array([0, 0, 0.1]), place_quat))
move_to_joints(solve_ik(place_pos, place_quat))
open_gripper()
move_to_joints(solve_ik(place_pos + np.array([0, 0, 0.15]), place_quat))