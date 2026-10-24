# Code block 0
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

print("red box", red_best["box"], "score", red_best["score"])
print("green box", green_best["box"], "score", green_best["score"])
```

```python
def mask_to_world_points(mask, depth, K, cam_pose):
    ys, xs = np.nonzero(mask)
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

print("red center", red_obb["center"], "extent", red_obb["extent"])
print("green center", green_obb["center"], "extent", green_obb["extent"])
```

```python
# Determine which cube is on top (higher z) and their centers
red_c = red_obb["center"]
green_c = green_obb["center"]
red_ext = red_obb["extent"]
green_ext = green_obb["extent"]

print("red z", red_c[2], "green z", green_c[2])
```

```python
# The green cube is on top of red. We need to move red on top of green.
# Strategy: grasp red cube, lift it, place it on top of green cube, open gripper.

# First, plan a grasp on the red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_quat = None
# Convert rotation matrix to quaternion (wxyz)
R = best_T_world[:3, :3]
def rot_to_quat_wxyz(R):
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

grasp_quat = rot_to_quat_wxyz(R)
print("grasp pos", grasp_pos, "quat", grasp_quat)
```

```python
# Approach from above the grasp pose
approach_pos = grasp_pos.copy()
approach_pos[2] += 0.10

open_gripper()
joints_approach = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints_approach)

# Move down to grasp
joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)

close_gripper()
```

```python
# Lift the red cube
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)
```

```python
# Now place red cube on top of green cube.
# Target: above green cube center, at height green_top + red_half_height
green_top_z = green_c[2] + green_ext[2] / 2.0
red_half_z = red_ext[2] / 2.0
place_z = green_top_z + red_half_z + 0.005

place_pos = np.array([green_c[0], green_c[1], place_z + 0.10])
joints_place_above = solve_ik(place_pos, grasp_quat)
move_to_joints(joints_place_above)

place_pos_final = np.array([green_c[0], green_c[1], place_z])
joints_place = solve_ik(place_pos_final, grasp_quat)
move_to_joints(joints_place)

open_gripper()
```

```python
# Retreat upward
retreat_pos = place_pos_final.copy()
retreat_pos[2] += 0.15
joints_retreat = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints_retreat)

# Code block 1
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

def mask_to_world_points(mask, depth, K, cam_pose):
    ys, xs = np.nonzero(mask)
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

red_c = red_obb["center"]
green_c = green_obb["center"]
red_ext = red_obb["extent"]
green_ext = green_obb["extent"]

# Plan grasp on red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam

grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]

def rot_to_quat_wxyz(R):
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

grasp_quat = rot_to_quat_wxyz(R)

# Approach from above
approach_pos = grasp_pos.copy()
approach_pos[2] += 0.10

open_gripper()
joints_approach = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints_approach)

joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)

close_gripper()

# Lift
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)

# Place on top of green cube
green_top_z = green_c[2] + green_ext[2] / 2.0
red_half_z = red_ext[2] / 2.0
place_z = green_top_z + red_half_z + 0.005

place_pos_above = np.array([green_c[0], green_c[1], place_z + 0.10])
joints_place_above = solve_ik(place_pos_above, grasp_quat)
move_to_joints(joints_place_above)

place_pos_final = np.array([green_c[0], green_c[1], place_z])
joints_place = solve_ik(place_pos_final, grasp_quat)
move_to_joints(joints_place)

open_gripper()

# Retreat
retreat_pos = place_pos_final.copy()
retreat_pos[2] += 0.15
joints_retreat = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints_retreat)

# Code block 2
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

def mask_to_world_points(mask, depth, K, cam_pose):
    ys, xs = np.nonzero(mask)
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

red_c = red_obb["center"]
green_c = green_obb["center"]
red_ext = red_obb["extent"]
green_ext = green_obb["extent"]

print("red center", red_c, "extent", red_ext)
print("green center", green_c, "extent", green_ext)

# Plan grasp on red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam

grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]

def rot_to_quat_wxyz(R):
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

grasp_quat = rot_to_quat_wxyz(R)
print("grasp pos", grasp_pos, "quat", grasp_quat)

# Approach from above
approach_pos = grasp_pos.copy()
approach_pos[2] += 0.10

open_gripper()
joints_approach = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints_approach)

joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)

close_gripper()

# Lift
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.20
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)

# Place on top of green cube. Use green top surface z.
green_top_z = green_c[2] + green_ext[2] / 2.0
red_half_z = red_ext[2] / 2.0
place_z = green_top_z + red_half_z + 0.01

place_xy = np.array([green_c[0], green_c[1]])

place_pos_above = np.array([place_xy[0], place_xy[1], place_z + 0.12])
joints_place_above = solve_ik(place_pos_above, grasp_quat)
move_to_joints(joints_place_above)

place_pos_final = np.array([place_xy[0], place_xy[1], place_z])
joints_place = solve_ik(place_pos_final, grasp_quat)
move_to_joints(joints_place)

open_gripper()

# Retreat
retreat_pos = place_pos_final.copy()
retreat_pos[2] += 0.15
joints_retreat = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints_retreat)

# Code block 3
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

def mask_to_world_points(mask, depth, K, cam_pose):
    ys, xs = np.nonzero(mask)
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

red_c = red_obb["center"]
green_c = green_obb["center"]
red_ext = red_obb["extent"]
green_ext = green_obb["extent"]

print("red center", red_c, "extent", red_ext)
print("green center", green_c, "extent", green_ext)

# Plan grasp on red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam

grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]

def rot_to_quat_wxyz(R):
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

grasp_quat = rot_to_quat_wxyz(R)
print("grasp pos", grasp_pos, "quat", grasp_quat)

# Approach from above
approach_pos = grasp_pos.copy()
approach_pos[2] += 0.10

open_gripper()
joints_approach = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints_approach)

joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)

close_gripper()

# Lift
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.20
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)

# Place on top of green cube. Use green top surface z.
green_top_z = green_c[2] + green_ext[2] / 2.0
red_half_z = red_ext[2] / 2.0
place_z = green_top_z + red_half_z + 0.01

place_xy = np.array([green_c[0], green_c[1]])

place_pos_above = np.array([place_xy[0], place_xy[1], place_z + 0.12])
joints_place_above = solve_ik(place_pos_above, grasp_quat)
move_to_joints(joints_place_above)

place_pos_final = np.array([place_xy[0], place_xy[1], place_z])
joints_place = solve_ik(place_pos_final, grasp_quat)
move_to_joints(joints_place)

open_gripper()

# Retreat
retreat_pos = place_pos_final.copy()
retreat_pos[2] += 0.15
joints_retreat = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints_retreat)