I'll start by getting an observation to understand the scene and locate the cubes.

```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment the red and green cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

print("red box", red_best["box"], "score", red_best["score"])
print("green box", green_best["box"], "score", green_best["score"])
```

```python
def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xc = (xs - cx) * zs / fx
    yc = (ys - cy) * zs / fy
    pts_cam = np.stack([xc, yc, zs], axis=1)
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

```python
# Plan grasp for the green cube (top object) first
green_seg = green_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
print("green grasp world pos", best_T_world[:3, 3])
print("green grasp score", grasp_scores[best_idx])
```

```python
from scipy.spatial.transform import Rotation as R

def quat_wxyz_from_mat(mat):
    rot = R.from_matrix(mat[:3, :3])
    q = rot.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])

# Approach green cube from above
green_center = green_obb["center"]
grasp_pos = best_T_world[:3, 3]
grasp_quat = quat_wxyz_from_mat(best_T_world)

# Move above the green cube
pre_pos = grasp_pos.copy()
pre_pos[2] += 0.10
joints = solve_ik(pre_pos, grasp_quat)
move_to_joints(joints)

# Descend to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift green cube
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

```python
# Set green cube aside (to the side of the workspace)
setdown_pos = lift_pos.copy()
setdown_pos[0] += 0.15  # move sideways
joints = solve_ik(setdown_pos, grasp_quat)
move_to_joints(joints)

# Lower and release green cube
place_pos = setdown_pos.copy()
place_pos[2] -= 0.12
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)
open_gripper()

# Lift up away from green
retreat_pos = place_pos.copy()
retreat_pos[2] += 0.15
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)
```

```python
# Now grasp the red cube
red_seg = red_mask.astype(np.int32)
red_grasp_poses, red_grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
red_best_idx = red_grasp_scores.argmax()
red_T_world = cam_ext @ red_grasp_poses[red_best_idx]
red_grasp_pos = red_T_world[:3, 3]
red_grasp_quat = quat_wxyz_from_mat(red_T_world)
print("red grasp world pos", red_grasp_pos, "score", red_grasp_scores[red_best_idx])

# Approach red cube from above
red_pre = red_grasp_pos.copy()
red_pre[2] += 0.10
joints = solve_ik(red_pre, red_grasp_quat)
move_to_joints(joints)

joints = solve_ik(red_grasp_pos, red_grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
red_lift = red_grasp_pos.copy()
red_lift[2] += 0.15
joints = solve_ik(red_lift, red_grasp_quat)
move_to_joints(joints)
```

```python
# Place red cube on top of green cube (green is now at setdown location)
# Green cube top height: place_pos z + half green extent
green_top_z = place_pos[2] + green_obb["extent"][2] / 2.0
red_half_z = red_obb["extent"][2] / 2.0

target_xy = place_pos[:2]  # above green cube location
target_z = green_top_z + red_half_z + 0.005

above_target = np.array([target_xy[0], target_xy[1], target_z + 0.12])
joints = solve_ik(above_target, red_grasp_quat)
move_to_joints(joints)

place_target = np.array([target_xy[0], target_xy[1], target_z])
joints = solve_ik(place_target, red_grasp_quat)
move_to_joints(joints)

open_gripper()

# Retreat upward
retreat = place_target.copy()
retreat[2] += 0.15
joints = solve_ik(retreat, red_grasp_quat)
move_to_joints(joints)
```