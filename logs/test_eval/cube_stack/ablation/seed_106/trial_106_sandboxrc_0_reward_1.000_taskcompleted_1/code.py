# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment the red cube and green block
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green block")

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
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_obb["extent"])
print("green center", green_center, "extent", green_extent)
```

Now I'll plan a grasp on the red cube and pick it up.

```python
# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])  # will refine below

# Extract orientation from rotation matrix (wxyz)
R = best_T_world[:3, :3]
def rot_to_quat_wxyz(R):
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

grasp_quat = rot_to_quat_wxyz(R)

# Approach from above
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.10])
joints_pre = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints_pre)

open_gripper()

joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)
close_gripper()

# Lift
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)
```

Now I'll move the red cube above the green block and place it on top.

```python
# Compute target placement position: on top of green block
green_top_z = green_center[2] + green_extent[2] / 2.0
red_half_height = red_obb["extent"][2] / 2.0
place_pos = np.array([green_center[0], green_center[1], green_top_z + red_half_height + 0.005])

# Move above placement
above_place = place_pos + np.array([0, 0, 0.12])
joints_above = solve_ik(above_place, grasp_quat)
move_to_joints(joints_above)

# Lower to place
joints_place = solve_ik(place_pos, grasp_quat)
move_to_joints(joints_place)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos + np.array([0, 0, 0.15])
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
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green block")

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
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]

def rot_to_quat_wxyz(R):
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

grasp_quat = rot_to_quat_wxyz(R)

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

green_top_z = green_center[2] + green_extent[2] / 2.0
red_half_height = red_extent[2] / 2.0
place_pos = np.array([green_center[0], green_center[1], green_top_z + red_half_height + 0.005])

above_place = place_pos + np.array([0, 0, 0.12])
joints_above = solve_ik(above_place, grasp_quat)
move_to_joints(joints_above)

joints_place = solve_ik(place_pos, grasp_quat)
move_to_joints(joints_place)

open_gripper()

retreat_pos = place_pos + np.array([0, 0, 0.15])
joints_retreat = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints_retreat)

# Code block 2
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green block")

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
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

# Use downward-facing orientation for placement
place_quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper pointing down

green_top_z = green_center[2] + green_extent[2] / 2.0
red_half_height = red_extent[2] / 2.0
place_pos = np.array([green_center[0], green_center[1], green_top_z + red_half_height + 0.005])

# Move above placement
above_place = place_pos + np.array([0, 0, 0.12])
joints_above = solve_ik(above_place, place_quat)
move_to_joints(joints_above)

# Lower to place
joints_place = solve_ik(place_pos, place_quat)
move_to_joints(joints_place)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos + np.array([0, 0, 0.15])
joints_retreat = solve_ik(retreat_pos, place_quat)
move_to_joints(joints_retreat)

# Code block 3
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green block")

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
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]

# Use a fixed downward-facing orientation for a top grasp
down_quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper pointing down

# Approach from above
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.12])
joints_pre = solve_ik(pre_grasp_pos, down_quat)
move_to_joints(joints_pre)

open_gripper()

# Descend to grasp
joints_grasp = solve_ik(grasp_pos, down_quat)
move_to_joints(joints_grasp)
close_gripper()

# Lift
lift_pos = grasp_pos + np.array([0, 0, 0.18])
joints_lift = solve_ik(lift_pos, down_quat)
move_to_joints(joints_lift)

# Compute placement position on top of green block
green_top_z = green_center[2] + green_extent[2] / 2.0
red_half_height = red_extent[2] / 2.0
place_pos = np.array([green_center[0], green_center[1], green_top_z + red_half_height + 0.01])

# Move above placement
above_place = place_pos + np.array([0, 0, 0.12])
joints_above = solve_ik(above_place, down_quat)
move_to_joints(joints_above)

# Lower to place
joints_place = solve_ik(place_pos, down_quat)
move_to_joints(joints_place)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos + np.array([0, 0, 0.15])
joints_retreat = solve_ik(retreat_pos, down_quat)
move_to_joints(joints_retreat)