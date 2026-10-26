```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment red cube
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
red_best = max(red_masks, key=lambda d: d["score"])
red_mask = red_best["mask"].astype(np.int32)

# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask)
best_idx = int(np.argmax(grasp_scores))
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

# Approach: move above the grasp pose first
grasp_pos = best_T_world[:3, 3]
grasp_R = best_T_world[:3, :3]

# Convert rotation to quaternion (wxyz)
def rot_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
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

quat = rot_to_quat(grasp_R)

# Pre-grasp above
pre_pos = grasp_pos + np.array([0, 0, 0.1])
open_gripper()
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)

# Move down to grasp
joints_grasp = solve_ik(grasp_pos, quat)
move_to_joints(joints_grasp)
close_gripper()

# Lift
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)

# Segment green cube to find placement location
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]

# Get 3D points of green cube
ys, xs = np.where(green_mask)
zs = depth[ys, xs]
valid = zs > 0
xs, ys, zs = xs[valid], ys[valid], zs[valid]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
X = (xs - cx) * zs / fx
Y = (ys - cy) * zs / fy
Z = zs
pts_cam = np.stack([X, Y, Z], axis=1)
pts_world = (cam_ext[:3,:3] @ pts_cam.T).T + cam_ext[:3,3]

obb = get_oriented_bounding_box_from_3d_points(pts_world)
green_center = obb["center"]
green_extent = obb["extent"]
green_top_z = green_center[2] + green_extent[2] / 2.0

# Place red cube on top of green cube
place_pos = np.array([green_center[0], green_center[1], green_top_z + 0.03])
joints_place_above = solve_ik(place_pos + np.array([0,0,0.1]), quat)
move_to_joints(joints_place_above)
joints_place = solve_ik(place_pos, quat)
move_to_joints(joints_place)

open_gripper()

# Retreat
joints_retreat = solve_ik(place_pos + np.array([0,0,0.15]), quat)
move_to_joints(joints_retreat)
```