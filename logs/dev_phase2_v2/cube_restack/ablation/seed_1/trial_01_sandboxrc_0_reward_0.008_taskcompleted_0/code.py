# Code block 0
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

Now let me compute the 3D positions of both cubes to understand the geometry.

```python
def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
print("red center:", red_center, "extent:", red_obb["extent"])
print("green center:", green_center, "extent:", green_obb["extent"])
```

The green cube is on top of the red cube. I need to grasp the red cube (which is underneath), lift it, move it above the green cube, and place it down. Let me plan a grasp on the red cube.

```python
# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
print("best grasp score:", grasp_scores[best_idx])
print("grasp pos world:", best_T_world[:3, 3])
```

Let me approach the red cube from the side since it's underneath the green cube. First, let me open the gripper and move to a pre-grasp pose.

```python
open_gripper()

grasp_pos = best_T_world[:3, 3]
grasp_R = best_T_world[:3, :3]

# Convert rotation matrix to quaternion (wxyz)
def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

grasp_quat = rot_to_quat(grasp_R)

# Pre-grasp: approach from above the grasp position
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)
```

Now move down to the grasp position and close the gripper.

```python
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()
```

Now lift the red cube up. Since the green cube was on top of the red cube, lifting the red cube will remove it from under the green cube (green will fall to the table). Let me lift up.

```python
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Let me get a new observation to see the current state.

```python
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_pts = mask_to_points(green_best["mask"], depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("green center now:", green_center, "extent:", green_extent)
```

Now I need to place the red cube on top of the green cube. The red cube is held in the gripper. I'll move it above the green cube and lower it down.

```python
# Place red cube on top of green cube
# Green cube top is at green_center[2] + green_extent[2]/2
# Red cube half-height ~ red_extent[2]/2
red_half_h = red_obb["extent"][2] / 2.0
green_top = green_center[2] + green_extent[2] / 2.0

# Target position for the red cube center when placed
target_red_center = np.array([green_center[0], green_center[1], green_top + red_half_h + 0.005])

# The grasp offset: red cube center relative to grasp position
# grasp_pos was at the red cube center roughly; compute offset
grasp_offset = red_center - grasp_pos  # in world frame at grasp time

# We want red cube center at target; so gripper pos = target - offset
place_pos = target_red_center - grasp_offset

# Move above the place position first
above_place = place_pos + np.array([0, 0, 0.1])
joints = solve_ik(above_place, grasp_quat)
move_to_joints(joints)

# Lower to place position
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)
```

Now open the gripper to release the red cube.

```python
open_gripper()

# Move up to clear
retreat_pos = place_pos + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)
```

Let me verify the final state.

```python
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
red_pts = mask_to_points(red_best["mask"], depth, K, cam_ext)
green_pts = mask_to_points(green_best["mask"], depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
print("red center:", red_obb["center"], "extent:", red_obb["extent"])
print("green center:", green_obb["center"], "extent:", green_obb["extent"])
print("red above green:", red_obb["center"][2] > green_obb["center"][2])

# Code block 1
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

# Segment cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Plan grasp on green cube (top cube) since it's accessible
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
grasp_quat = rot_to_quat(best_T_world[:3, :3])

open_gripper()

# Pre-grasp above green cube
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

# Move to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift green cube
lift_pos = grasp_pos + np.array([0, 0, 0.2])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Now place green cube on top of red cube
# Red cube top
red_top = red_center[2] + red_extent[2] / 2.0
green_half_h = green_extent[2] / 2.0
target_green_center = np.array([red_center[0], red_center[1], red_top + green_half_h + 0.005])

# Offset between green center and grasp pos
grasp_offset = green_center - grasp_pos
place_pos = target_green_center - grasp_offset

above_place = place_pos + np.array([0, 0, 0.1])
joints = solve_ik(above_place, grasp_quat)
move_to_joints(joints)

joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

open_gripper()

retreat_pos = place_pos + np.array([0, 0, 0.15])
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

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

# Segment cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Grasp the red cube (the one we need to place on top)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
grasp_quat = rot_to_quat(best_T_world[:3, :3])

open_gripper()

# Pre-grasp above red cube
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
lift_pos = grasp_pos + np.array([0, 0, 0.2])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Re-observe green cube position (it may have shifted)
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_pts = mask_to_points(green_best["mask"], depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# Place red cube on top of green cube
green_top = green_center[2] + green_extent[2] / 2.0
red_half_h = red_extent[2] / 2.0
target_red_center = np.array([green_center[0], green_center[1], green_top + red_half_h + 0.005])

# Offset between red center and grasp pos (in world frame at grasp time)
grasp_offset = red_center - grasp_pos
place_pos = target_red_center - grasp_offset

above_place = place_pos + np.array([0, 0, 0.12])
joints = solve_ik(above_place, grasp_quat)
move_to_joints(joints)

joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

open_gripper()

retreat_pos = place_pos + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)