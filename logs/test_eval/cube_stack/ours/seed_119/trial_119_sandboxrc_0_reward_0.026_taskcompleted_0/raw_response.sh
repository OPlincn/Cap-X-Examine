import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extrinsics = cam["pose_mat"]

# Segment red and green cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

def mask_to_world_points(mask, depth, K, extrinsics):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (extrinsics @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, extrinsics)
green_pts = mask_to_world_points(green_mask, depth, K, extrinsics)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

# Grasp red cube from above
red_grasp_poses, red_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = red_scores.argmax()
best_T_cam = red_grasp_poses[best_idx]
best_T_world = extrinsics @ best_T_cam

# Approach above red cube
approach_pos = best_T_world[:3, 3].copy()
approach_pos[2] += 0.10
quat = np.array([1.0, 0.0, 0.0, 0.0])

open_gripper()
joints = solve_ik(approach_pos, quat)
move_to_joints(joints)

# Move down to grasp
grasp_pos = best_T_world[:3, 3].copy()
joints = solve_ik(grasp_pos, quat)
move_to_joints(joints)

close_gripper()

# Lift red cube
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, quat)
move_to_joints(joints)

# Place on top of green cube
place_pos = green_center.copy()
place_pos[2] = green_center[2] + green_extent[2] / 2.0 + red_extent[2] / 2.0 + 0.02
joints = solve_ik(place_pos, quat)
move_to_joints(joints)

open_gripper()

# Retreat upward
retreat_pos = place_pos.copy()
retreat_pos[2] += 0.10
joints = solve_ik(retreat_pos, quat)
move_to_joints(joints)