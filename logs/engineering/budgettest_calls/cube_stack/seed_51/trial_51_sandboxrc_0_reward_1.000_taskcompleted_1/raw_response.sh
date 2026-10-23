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
red_mask = red_best["mask"]
green_mask = green_best["mask"]

def mask_to_points(mask, depth, K):
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

red_pts = mask_to_points(red_mask, depth, K)
green_pts = mask_to_points(green_mask, depth, K)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# Target: red cube on top of green cube
target_pos = green_center.copy()
target_pos[2] = green_center[2] + green_extent[2] / 2.0 + red_obb["extent"][2] / 2.0 + 0.005

# Grasp red cube from above
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T = grasp_poses[best_idx]
grasp_world = cam_ext @ best_T

# Approach above red cube
approach_pos = grasp_world[:3, 3].copy()
approach_pos[2] += 0.1

# Orientation: gripper pointing down (identity quaternion wxyz)
quat_down = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x -> gripper down

open_gripper()
joints = solve_ik(approach_pos, quat_down)
move_to_joints(joints)

# Move to grasp pose
grasp_pos = grasp_world[:3, 3].copy()
joints = solve_ik(grasp_pos, quat_down)
move_to_joints(joints)

close_gripper()

# Lift
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, quat_down)
move_to_joints(joints)

# Move above green cube target
above_target = target_pos.copy()
above_target[2] += 0.1
joints = solve_ik(above_target, quat_down)
move_to_joints(joints)

# Lower to place
joints = solve_ik(target_pos, quat_down)
move_to_joints(joints)

open_gripper()

# Retreat
retreat_pos = target_pos.copy()
retreat_pos[2] += 0.15
joints = solve_ik(retreat_pos, quat_down)
move_to_joints(joints)
```