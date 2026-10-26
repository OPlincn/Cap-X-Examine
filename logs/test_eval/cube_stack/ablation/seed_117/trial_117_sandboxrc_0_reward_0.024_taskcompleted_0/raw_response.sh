```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment red and green cubes
red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_res, key=lambda d: d["score"])
green_best = max(green_res, key=lambda d: d["score"])
red_mask = red_best["mask"].astype(np.int32)
green_mask = green_best["mask"].astype(np.int32)

# Get 3D points for green cube to find its top
def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask > 0)
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

green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
green_top_z = green_center[2] + green_extent[2] / 2.0

# Plan grasp for red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])  # top-down grasp

# Approach above red cube
approach_pos = grasp_pos.copy()
approach_pos[2] += 0.10
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move above green cube
place_pos = np.array([green_center[0], green_center[1], green_top_z + 0.10])
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

# Lower to place
place_pos[2] = green_top_z + 0.02
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

open_gripper()

# Retreat upward
retreat_pos = place_pos.copy()
retreat_pos[2] += 0.15
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)
```