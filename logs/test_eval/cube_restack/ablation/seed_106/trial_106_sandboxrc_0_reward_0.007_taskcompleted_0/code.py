# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment the green and red cubes
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")

green_best = max(green_masks, key=lambda d: d["score"])
red_best = max(red_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
red_mask = red_best["mask"]

def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)

green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)

green_center = green_obb["center"]
red_center = red_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

print("green center", green_center, "extent", green_extent)
print("red center", red_center, "extent", red_extent)
```

Now I'll pick up the green cube (top one), move it aside, then pick up the red cube and place it on the green cube.

```python
# Grasp the green cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_quat = None
# Convert rotation matrix to quaternion (wxyz)
R = best_T_world[:3, :3]
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

grasp_quat = rot_to_quat(R)

# Approach from above
approach_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift green cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move green cube aside (to the side of the workspace)
setdown_pos = green_center + np.array([0.15, 0.0, 0.15])
joints = solve_ik(setdown_pos, grasp_quat)
move_to_joints(joints)

# Lower and release green cube
place_pos = green_center + np.array([0.15, 0.0, 0.02])
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)
open_gripper()

# Lift away
joints = solve_ik(place_pos + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints)
```

Now I'll pick up the red cube and place it on top of the green cube.

```python
# Re-observe to get updated positions
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# Grasp red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]
grasp_quat = rot_to_quat(R)

# Approach and grasp
joints = solve_ik(grasp_pos + np.array([0, 0, 0.1]), grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move above green cube
target_above = green_center + np.array([0, 0, green_extent[2] / 2 + 0.08])
joints = solve_ik(target_above, grasp_quat)
move_to_joints(joints)

# Lower red cube onto green cube
target_place = green_center + np.array([0, 0, green_extent[2] / 2 + 0.02])
joints = solve_ik(target_place, grasp_quat)
move_to_joints(joints)
open_gripper()

# Lift away
joints = solve_ik(target_place + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints)

# Code block 1
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

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

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

# Grasp red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]
grasp_quat = rot_to_quat(R)

# Approach and grasp
joints = solve_ik(grasp_pos + np.array([0, 0, 0.1]), grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move above green cube
target_above = green_center + np.array([0, 0, green_extent[2] / 2 + red_extent[2] / 2 + 0.1])
joints = solve_ik(target_above, grasp_quat)
move_to_joints(joints)

# Lower red cube onto green cube
target_place = green_center + np.array([0, 0, green_extent[2] / 2 + red_extent[2] / 2 + 0.005])
joints = solve_ik(target_place, grasp_quat)
move_to_joints(joints)
open_gripper()

# Lift away
joints = solve_ik(target_place + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints)

# Code block 2
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

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

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

# The green cube is currently on top of the red cube. We need to swap them.
# Step 1: Grasp the green cube (top) and move it aside.
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]
grasp_quat = rot_to_quat(R)

joints = solve_ik(grasp_pos + np.array([0, 0, 0.1]), grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.2])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move green cube aside
aside_pos = green_center + np.array([0.2, 0.0, 0.15])
joints = solve_ik(aside_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(aside_pos - np.array([0, 0, 0.1]), grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(aside_pos + np.array([0, 0, 0.1]), grasp_quat)
move_to_joints(joints)

# Step 2: Re-observe to find red cube (now exposed on bottom)
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
red_best = max(red_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
red_center = red_obb["center"]
red_extent = red_obb["extent"]

# Grasp red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]
grasp_quat = rot_to_quat(R)

joints = solve_ik(grasp_pos + np.array([0, 0, 0.1]), grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.2])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Step 3: Move red cube to where green cube is now (aside position)
# Place red cube at the aside location where green cube sits
target_above = aside_pos + np.array([0, 0, 0.1])
joints = solve_ik(target_above, grasp_quat)
move_to_joints(joints)
target_place = aside_pos - np.array([0, 0, 0.02])
joints = solve_ik(target_place, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(target_place + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints)

# Step 4: Re-observe, grasp green cube again and place it on top of red cube
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
red_best = max(red_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
red_center = red_obb["center"]
red_extent = red_obb["extent"]

# Grasp green cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]
grasp_quat = rot_to_quat(R)

joints = solve_ik(grasp_pos + np.array([0, 0, 0.1]), grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.2])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Place green cube on top of red cube
target_above = red_center + np.array([0, 0, red_extent[2] / 2 + green_extent[2] / 2 + 0.1])
joints = solve_ik(target_above, grasp_quat)
move_to_joints(joints)
target_place = red_center + np.array([0, 0, red_extent[2] / 2 + green_extent[2] / 2 + 0.005])
joints = solve_ik(target_place, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(target_place + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints)

# Code block 3
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

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

def get_center_extent(mask):
    pts = mask_to_world_points(mask, depth, K, cam_ext)
    obb = get_oriented_bounding_box_from_3d_points(pts)
    return obb["center"], obb["extent"]

def grasp_and_lift(mask, lift_height=0.2):
    grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask.astype(np.int32))
    best_idx = grasp_scores.argmax()
    best_T_world = cam_ext @ grasp_poses[best_idx]
    grasp_pos = best_T_world[:3, 3]
    grasp_quat = rot_to_quat(best_T_world[:3, :3])
    joints = solve_ik(grasp_pos + np.array([0, 0, 0.1]), grasp_quat)
    move_to_joints(joints)
    joints = solve_ik(grasp_pos, grasp_quat)
    move_to_joints(joints)
    close_gripper()
    lift_pos = grasp_pos + np.array([0, 0, lift_height])
    joints = solve_ik(lift_pos, grasp_quat)
    move_to_joints(joints)
    return grasp_quat

# Current state: green on top of red. Need red on top of green.
# Step 1: Grasp green (top) and move it aside.
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]
green_center, green_extent = get_center_extent(green_mask)

grasp_quat = grasp_and_lift(green_mask)

# Move green aside
aside_pos = green_center + np.array([0.2, 0.0, 0.0])
joints = solve_ik(aside_pos + np.array([0, 0, 0.1]), grasp_quat)
move_to_joints(joints)
joints = solve_ik(aside_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(aside_pos + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints)

# Step 2: Re-observe, grasp red cube
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
red_center, red_extent = get_center_extent(red_mask)

grasp_quat = grasp_and_lift(red_mask)

# Step 3: Re-observe green cube (now aside), place red on top of green
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]
green_center, green_extent = get_center_extent(green_mask)

target_above = green_center + np.array([0, 0, green_extent[2] / 2 + red_extent[2] / 2 + 0.1])
joints = solve_ik(target_above, grasp_quat)
move_to_joints(joints)
target_place = green_center + np.array([0, 0, green_extent[2] / 2 + red_extent[2] / 2 + 0.005])
joints = solve_ik(target_place, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(target_place + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints)