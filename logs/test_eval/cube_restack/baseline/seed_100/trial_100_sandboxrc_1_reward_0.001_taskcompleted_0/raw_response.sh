I'll start by getting an observation to understand the scene geometry.

```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

# Segment the red and green cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

def mask_to_points(mask, depth, K, cam_pose):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    pts_cam = np.stack([x, y, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_pose @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_points(red_mask, depth, K, cam_pose)
green_pts = mask_to_points(green_mask, depth, K, cam_pose)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
print("red center", red_center, "extent", red_obb["extent"])
print("green center", green_center, "extent", green_obb["extent"])
```

Now I'll plan a grasp on the red cube, lift it, and place it on the green cube.

```python
# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_R = best_T_world[:3, :3]

# Convert rotation matrix to quaternion (wxyz)
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

grasp_quat = rot_to_quat(grasp_R)

# Approach from above
approach_pos = grasp_pos + np.array([0, 0, 0.1])
open_gripper()
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Now place the red cube on top of the green cube. I need to compute the target position above the green cube.

```python
# Determine green cube top height and place red cube above it
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]
# Green cube top z
green_top_z = green_center[2] + green_extent[2] / 2.0
# Red cube half height
red_half_h = red_extent[2] / 2.0

# Target: red cube center above green cube center
place_xy = green_center[:2]
place_z = green_top_z + red_half_h + 0.01
place_pos = np.array([place_xy[0], place_xy[1], place_z])

# Keep grasp orientation but approach from above
place_quat = grasp_quat

# Move above placement
above_place = place_pos + np.array([0, 0, 0.1])
joints = solve_ik(above_place, place_quat)
move_to_joints(joints)

# Move down to place
joints = solve_ik(place_pos, place_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, place_quat)
move_to_joints(joints)
```