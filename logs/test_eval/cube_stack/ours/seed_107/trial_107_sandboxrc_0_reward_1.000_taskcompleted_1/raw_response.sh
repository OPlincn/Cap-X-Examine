```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
ext = cam["pose_mat"]

# Segment red and green cubes
red_res = segment_sam3_text_prompt(rgb, "red cube")
green_res = segment_sam3_text_prompt(rgb, "green cube")
red = max(red_res, key=lambda d: d["score"])
green = max(green_res, key=lambda d: d["score"])
red_mask = red["mask"]
green_mask = green["mask"]

# Get 3D points for green cube to find its top
def mask_to_points(mask, depth, K, ext):
    ys, xs = np.where(mask)
    z = depth[ys, xs]
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (ext @ pts_h.T).T[:, :3]
    return pts_world

green_pts = mask_to_points(green_mask, depth, K, ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
green_top_z = green_center[2] + green_extent[2] / 2.0

# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = ext @ best_T_cam

# Approach from above: use red cube center xy, grasp orientation from planner
red_pts = mask_to_points(red_mask, depth, K, ext)
red_center = red_pts.mean(axis=0)

# Extract rotation from grasp pose (camera frame -> world)
R_grasp = best_T_world[:3, :3]
# Convert rotation to quaternion wxyz
def rot_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

quat = rot_to_quat(R_grasp)

# Pre-grasp above red cube
pre_pos = red_center.copy()
pre_pos[2] = red_center[2] + 0.10
joints = solve_ik(pre_pos, quat)
move_to_joints(joints)
open_gripper()

# Move down to grasp
grasp_pos = red_center.copy()
grasp_pos[2] = red_center[2]
joints = solve_ik(grasp_pos, quat)
move_to_joints(joints)
close_gripper()

# Lift
lift_pos = grasp_pos.copy()
lift_pos[2] = green_top_z + 0.15
joints = solve_ik(lift_pos, quat)
move_to_joints(joints)

# Move above green cube
place_pos = np.array([green_center[0], green_center[1], green_top_z + 0.10])
joints = solve_ik(place_pos, quat)
move_to_joints(joints)

# Lower to place
place_pos[2] = green_top_z + 0.02
joints = solve_ik(place_pos, quat)
move_to_joints(joints)

open_gripper()

# Retreat upward
retreat_pos = place_pos.copy()
retreat_pos[2] = green_top_z + 0.15
joints = solve_ik(retreat_pos, quat)
move_to_joints(joints)
```