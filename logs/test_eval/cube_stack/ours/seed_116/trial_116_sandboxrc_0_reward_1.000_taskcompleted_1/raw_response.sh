```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
ext = cam["pose_mat"]

# Segment red and green cubes
red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red = max(red_res, key=lambda d: d["score"])
green = max(green_res, key=lambda d: d["score"])
red_mask = red["mask"].astype(np.int32)
green_mask = green["mask"].astype(np.int32)

def mask_to_world_points(mask, depth, K, ext):
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
    pts_world = (ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, ext)
green_pts = mask_to_world_points(green_mask, depth, K, ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = ext @ best_T_cam

# Approach pose: above red cube, gripper pointing down
approach_pos = red_center + np.array([0, 0, 0.12])
down_quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper pointing down

open_gripper()
joints = solve_ik(approach_pos, down_quat)
move_to_joints(joints)

# Move to grasp pose
grasp_pos = best_T_world[:3, 3]
grasp_quat = np.array([0.0, 1.0, 0.0, 0.0])
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()

# Lift
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Place on top of green cube
place_pos = green_center + np.array([0, 0, green_extent[2] / 2 + 0.03])
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

open_gripper()

# Retreat
retreat_pos = place_pos + np.array([0, 0, 0.12])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)
```