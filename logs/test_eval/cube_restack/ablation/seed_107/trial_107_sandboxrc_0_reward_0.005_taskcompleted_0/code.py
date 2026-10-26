# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment the red cube (currently on top)
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

print("red box", red_best.get("box"), "score", red_best["score"])
print("green box", green_best.get("box"), "score", green_best["score"])
```

Now I need to figure out the geometry. The red cube is on top of the green cube. To place the red cube on top of the green cube, I need to:
1. Grasp the red cube
2. Lift it
3. Move it above the green cube
4. Lower it onto the green cube
5. Open the gripper

Let me compute the 3D positions of both cubes using the depth image.

```python
def mask_to_3d_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_3d_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_3d_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```

Now plan the grasp on the red cube. I'll use plan_grasp with the red mask.

```python
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
print("best grasp score", grasp_scores[best_idx])
print("grasp world pos", best_T_world[:3, 3])
```

Now I'll execute the grasp: move to a pre-grasp pose above the red cube, open gripper, descend, close gripper, lift.

```python
# Grasp orientation from planned pose
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

# Pre-grasp: above the red cube
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.10])
open_gripper()
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

# Descend to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Now move above the green cube and place the red cube on top of it. The green cube's top surface is at green_center[2] + green_extent[2]/2. The red cube's bottom is at red_center[2] - red_extent[2]/2. So the target red center z = green_top + red_extent[2]/2.

```python
green_top_z = green_center[2] + green_extent[2] / 2.0
red_half_h = red_extent[2] / 2.0
target_red_center = np.array([green_center[0], green_center[1], green_top_z + red_half_h])

# The grasp point is offset from red center; compute offset
grasp_offset = grasp_pos - red_center
target_grasp_pos = target_red_center + grasp_offset

# Move above target
above_target = target_grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(above_target, grasp_quat)
move_to_joints(joints)

# Lower to place
joints = solve_ik(target_grasp_pos, grasp_quat)
move_to_joints(joints)

# Open gripper
open_gripper()

# Retreat upward
retreat_pos = target_grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)

# Code block 1
import numpy as np

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

def mask_to_3d_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_3d_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_3d_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
print("best grasp score", grasp_scores[best_idx])
print("grasp world pos", best_T_world[:3, 3])

grasp_pos = best_T_world[:3, 3]
grasp_R = best_T_world[:3, :3]

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

pre_grasp_pos = grasp_pos + np.array([0, 0, 0.10])
open_gripper()
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

green_top_z = green_center[2] + green_extent[2] / 2.0
red_half_h = red_extent[2] / 2.0
target_red_center = np.array([green_center[0], green_center[1], green_top_z + red_half_h])

grasp_offset = grasp_pos - red_center
target_grasp_pos = target_red_center + grasp_offset

above_target = target_grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(above_target, grasp_quat)
move_to_joints(joints)

joints = solve_ik(target_grasp_pos, grasp_quat)
move_to_joints(joints)

open_gripper()

retreat_pos = target_grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)

# Code block 2
import numpy as np

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

def mask_to_3d_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_3d_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_3d_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
print("best grasp score", grasp_scores[best_idx])
print("grasp world pos", best_T_world[:3, 3])

grasp_pos = best_T_world[:3, 3]
grasp_R = best_T_world[:3, :3]

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

# Use a top-down grasp orientation for reliability
top_down_R = np.array([[1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0]])
top_down_quat = rot_to_quat(top_down_R)

# Grasp at red cube center using top-down orientation
grasp_target = red_center.copy()
grasp_target[2] = red_center[2]

pre_grasp_pos = grasp_target + np.array([0, 0, 0.12])
open_gripper()
joints = solve_ik(pre_grasp_pos, top_down_quat)
move_to_joints(joints)

joints = solve_ik(grasp_target, top_down_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_target + np.array([0, 0, 0.18])
joints = solve_ik(lift_pos, top_down_quat)
move_to_joints(joints)

# Compute target: red cube center on top of green cube
green_top_z = green_center[2] + green_extent[2] / 2.0
red_half_h = red_extent[2] / 2.0
target_red_center = np.array([green_center[0], green_center[1], green_top_z + red_half_h + 0.005])

above_target = target_red_center + np.array([0, 0, 0.15])
joints = solve_ik(above_target, top_down_quat)
move_to_joints(joints)

joints = solve_ik(target_red_center, top_down_quat)
move_to_joints(joints)

open_gripper()

retreat_pos = target_red_center + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, top_down_quat)
move_to_joints(joints)