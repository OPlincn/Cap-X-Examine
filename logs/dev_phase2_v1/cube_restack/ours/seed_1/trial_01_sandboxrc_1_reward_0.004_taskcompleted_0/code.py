# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
intrinsics = cam["intrinsics"]
pose_mat = cam["pose_mat"]

# Segment the red and green cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

print("red masks:", len(red_masks))
print("green masks:", len(green_masks))
for m in red_masks:
    print("red score", m["score"], "box", m.get("box"))
for m in green_masks:
    print("green score", m["score"], "box", m.get("box"))
```

Let me analyze the segmentation results to determine the positions of both cubes.

```python
# Pick best masks
red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])

red_mask = red_best["mask"]
green_mask = green_best["mask"]

def mask_to_world_points(mask, depth, intrinsics, pose_mat):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx = intrinsics[0, 0]; fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]; cy = intrinsics[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (pose_mat @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, intrinsics, pose_mat)
green_pts = mask_to_world_points(green_mask, depth, intrinsics, pose_mat)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

print("red center", red_obb["center"], "extent", red_obb["extent"])
print("green center", green_obb["center"], "extent", green_obb["extent"])
```

Now I'll plan a grasp on the red cube, lift it, and place it on top of the green cube.

```python
# Plan grasp on red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=intrinsics, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = pose_mat @ best_T_cam

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
print("grasp pos", grasp_pos, "quat", grasp_quat)
```

Now I'll execute the pick-and-place motion: approach, grasp, lift, move above green cube, lower, release.

```python
# Approach above the grasp pose
approach_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

# Close gripper
close_gripper()

# Lift the red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Now I'll move the red cube above the green cube and place it.

```python
# Target: place red cube on top of green cube
green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

# Height of green cube top
green_top_z = green_center[2] + green_extent[2] / 2
# Red cube half height
red_half_z = red_extent[2] / 2

# Target position for red cube center when placed on green
target_red_center = np.array([green_center[0], green_center[1], green_top_z + red_half_z])

# The grasp offset: red cube center relative to grasp position
# Grasp position is roughly at red cube center; compute offset
red_center = red_obb["center"]
offset = red_center - grasp_pos  # world offset from grasp to cube center

# Desired grasp position so that cube center lands at target
target_grasp_pos = target_red_center - offset

# Move above target
above_target = target_grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(above_target, grasp_quat)
move_to_joints(joints)

# Lower to place
joints = solve_ik(target_grasp_pos, grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = target_grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)

# Code block 1
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
intrinsics = cam["intrinsics"]
pose_mat = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

def mask_to_world_points(mask, depth, intrinsics, pose_mat):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx = intrinsics[0, 0]; fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]; cy = intrinsics[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (pose_mat @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, intrinsics, pose_mat)
green_pts = mask_to_world_points(green_mask, depth, intrinsics, pose_mat)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=intrinsics, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = pose_mat @ best_T_cam

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

approach_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)

joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

green_top_z = green_center[2] + green_extent[2] / 2
red_half_z = red_extent[2] / 2

target_red_center = np.array([green_center[0], green_center[1], green_top_z + red_half_z])

red_center = red_obb["center"]
offset = red_center - grasp_pos
target_grasp_pos = target_red_center - offset

above_target = target_grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(above_target, grasp_quat)
move_to_joints(joints)

joints = solve_ik(target_grasp_pos, grasp_quat)
move_to_joints(joints)

open_gripper()

retreat_pos = target_grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)