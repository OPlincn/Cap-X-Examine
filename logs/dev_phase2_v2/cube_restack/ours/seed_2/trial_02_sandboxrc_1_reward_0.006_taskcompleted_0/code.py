# Code block 0
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
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

print("red center", red_obb["center"], "extent", red_obb["extent"])
print("green center", green_obb["center"], "extent", green_obb["extent"])
```

Now I'll plan a grasp on the red cube (the bottom one) and lift it, then place it on the green cube.

```python
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
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.1])
joints_pre = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints_pre)

open_gripper()
joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)
close_gripper()

# Lift red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)

print("red grasp pos", grasp_pos)
print("green center", green_obb["center"])
```

Now I'll place the red cube on top of the green cube. The green cube's top is at green_center_z + extent_z/2, and the red cube's half-height is red_extent_z/2.

```python
green_top_z = green_obb["center"][2] + green_obb["extent"][2] / 2.0
red_half_z = red_obb["extent"][2] / 2.0
place_z = green_top_z + red_half_z + 0.005

place_xy = green_obb["center"][:2]
place_pos = np.array([place_xy[0], place_xy[1], place_z + 0.1])

# Keep same orientation as grasp
joints_place_pre = solve_ik(place_pos, grasp_quat)
move_to_joints(joints_place_pre)

place_pos_final = np.array([place_xy[0], place_xy[1], place_z])
joints_place = solve_ik(place_pos_final, grasp_quat)
move_to_joints(joints_place)

open_gripper()

# Retreat upward
retreat_pos = place_pos_final + np.array([0, 0, 0.15])
joints_retreat = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints_retreat)

print("done")

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
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
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

pre_grasp_pos = grasp_pos + np.array([0, 0, 0.1])
joints_pre = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints_pre)

open_gripper()
joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)

green_top_z = green_obb["center"][2] + green_obb["extent"][2] / 2.0
red_half_z = red_obb["extent"][2] / 2.0
place_z = green_top_z + red_half_z + 0.005

place_xy = green_obb["center"][:2]
place_pos = np.array([place_xy[0], place_xy[1], place_z + 0.1])

joints_place_pre = solve_ik(place_pos, grasp_quat)
move_to_joints(joints_place_pre)

place_pos_final = np.array([place_xy[0], place_xy[1], place_z])
joints_place = solve_ik(place_pos_final, grasp_quat)
move_to_joints(joints_place)

open_gripper()

retreat_pos = place_pos_final + np.array([0, 0, 0.15])
joints_retreat = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints_retreat)