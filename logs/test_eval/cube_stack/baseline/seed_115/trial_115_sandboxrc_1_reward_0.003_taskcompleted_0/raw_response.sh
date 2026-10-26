I'll start by getting an observation to understand the scene and locate the cubes.

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

print("red box:", red_best["box"], "score:", red_best["score"])
print("green box:", green_best["box"], "score:", green_best["score"])
```

```python
# Helper to back-project a mask to 3D world points
def mask_to_world_points(mask, depth, K, cam_pose):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_pose @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_pose)
green_pts = mask_to_world_points(green_mask, depth, K, cam_pose)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
print("red center:", red_center, "extent:", red_obb["extent"])
print("green center:", green_center, "extent:", green_obb["extent"])
```

```python
# Plan a grasp on the red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam
print("best grasp score:", grasp_scores[best_idx])
print("grasp world pos:", best_T_world[:3, 3])
```

```python
# Move to a pre-grasp pose above the red cube, then descend and grasp
grasp_pos = best_T_world[:3, 3]
grasp_R = best_T_world[:3, :3]

# Convert rotation matrix to quaternion (wxyz)
def rot_to_quat_wxyz(R):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])

grasp_quat = rot_to_quat_wxyz(grasp_R)

# Pre-grasp: approach from above
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.10])
open_gripper()
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

# Descend to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()
```

```python
# Lift the red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

```python
# Place red cube on top of green cube
green_top = green_center + np.array([0, 0, green_obb["extent"][2] / 2.0])
place_pos = green_top + np.array([0, 0, 0.12])  # hover above green top

joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

# Lower to place
place_final = green_top + np.array([0, 0, 0.02])
joints = solve_ik(place_final, grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()
```

```python
# Retreat upward after releasing
retreat_pos = place_final + np.array([0, 0, 0.12])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)
```