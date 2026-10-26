I'll start by getting an observation to understand the scene, then segment the cubes to locate them.

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

print("red box:", red_best["box"], "score:", red_best["score"])
print("green box:", green_best["box"], "score:", green_best["score"])
```

Now let me compute 3D positions of both cubes from the depth image to understand their spatial arrangement.

```python
def mask_to_points(mask, depth, K, cam_ext):
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
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

print("red center:", red_obb["center"], "extent:", red_obb["extent"])
print("green center:", green_obb["center"], "extent:", green_obb["extent"])
```

The green cube is on top of the red cube. To swap them, I need to:
1. Grasp the green cube and move it aside
2. Grasp the red cube and place it on top of the green cube
3. Open gripper

Let me plan grasps for the green cube first to move it out of the way.

```python
# Plan grasp for green cube (topmost object) to move it aside
green_seg = green_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_seg)
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
print("green grasp pos:", grasp_pos, "quat:", grasp_quat)
```

Now I'll move to grasp the green cube, lift it, and place it aside.

```python
# Approach green cube from above
approach_pos = grasp_pos + np.array([0, 0, 0.10])
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

# Move green cube aside (away from red cube)
green_center = green_obb["center"]
red_center = red_obb["center"]
# Move in direction away from red cube horizontally
offset_dir = green_center - red_center
offset_dir[2] = 0
norm = np.linalg.norm(offset_dir)
if norm < 1e-3:
    offset_dir = np.array([0.15, 0.0, 0.0])
else:
    offset_dir = offset_dir / norm * 0.15
aside_pos = lift_pos + offset_dir
joints = solve_ik(aside_pos, grasp_quat)
move_to_joints(joints)

# Lower green cube to table
place_green_pos = aside_pos - np.array([0, 0, 0.13])
joints = solve_ik(place_green_pos, grasp_quat)
move_to_joints(joints)
open_gripper()

# Lift away
joints = solve_ik(place_green_pos + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints)
```

Now the green cube is moved aside. Let me re-observe to get the red cube's current position and plan to grasp it.

```python
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
red_best = max(red_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]

red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
red_grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]
red_grasp_quat = rot_to_quat(R)
print("red grasp pos:", red_grasp_pos, "quat:", red_grasp_quat)
```

Now grasp the red cube, lift it, and place it on top of the green cube.

```python
# Approach red cube
joints = solve_ik(red_grasp_pos + np.array([0, 0, 0.10]), red_grasp_quat)
move_to_joints(joints)
joints = solve_ik(red_grasp_pos, red_grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
red_lift = red_grasp_pos + np.array([0, 0, 0.20])
joints = solve_ik(red_lift, red_grasp_quat)
move_to_joints(joints)
```

Now I need to find the green cube's new position to place the red cube on top of it.

```python
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("green center now:", green_center, "extent:", green_extent)
```

Now place the red cube on top of the green cube. The red cube's grasp point is offset from its center, so I need to account for that offset when placing.

```python
# Red cube extent to know its height
red_pts = mask_to_points(red_mask, depth, K, cam_ext)  # from earlier obs, but red is now lifted
# Use red_obb from earlier for extent
red_extent = red_obb["extent"]
red_height = red_extent[2] if red_extent[2] < red_extent[0] else min(red_extent)
green_height = green_extent[2] if green_extent[2] < green_extent[0] else min(green_extent)

# Target: red cube center should be at green_center + [0,0, green_height/2 + red_height/2]
target_red_center = green_center + np.array([0, 0, green_height/2 + red_height/2 + 0.005])

# Offset between grasp point and red cube center (in world frame at grasp time)
# grasp point is roughly at cube center height; compute offset
red_center_orig = red_obb["center"]
grasp_offset = red_grasp_pos - red_center_orig
grasp_offset[2] = 0  # keep horizontal offset only; vertical handled by target

target_grasp_pos = target_red_center + grasp_offset
target_grasp_pos[2] = target_red_center[2]  # grasp at center height

# Move above target
joints = solve_ik(target_grasp_pos + np.array([0, 0, 0.15]), red_grasp_quat)
move_to_joints(joints)

# Lower to place
joints = solve_ik(target_grasp_pos, red_grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Lift away
joints = solve_ik(target_grasp_pos + np.array([0, 0, 0.15]), red_grasp_quat)
move_to_joints(joints)
```