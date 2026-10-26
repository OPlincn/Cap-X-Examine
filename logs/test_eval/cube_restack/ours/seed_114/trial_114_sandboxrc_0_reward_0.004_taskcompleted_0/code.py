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

print("red box", red_best["box"], "score", red_best["score"])
print("green box", green_best["box"], "score", green_best["score"])
```

Now let me compute the 3D positions of both cubes to understand the geometry.

```python
def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
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

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

print("red center", red_obb["center"], "extent", red_obb["extent"])
print("green center", green_obb["center"], "extent", green_obb["extent"])
```

Now I'll plan a grasp for the red cube. Since the red cube is on the bottom with green on top, I need to grasp the red cube from the side. Let me plan grasps.

```python
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
print("best grasp world pos", best_T_world[:3, 3])
print("best grasp score", grasp_scores[best_idx])
```

Let me approach this more carefully. The red cube is at the bottom, green on top. I need to grasp the red cube (likely from the side since green blocks the top), lift it, move it aside, then place it on top of green. Let me first move the green cube out of the way, or grasp red from the side.

Let me get the grasp and execute a pick of the red cube from the side.

```python
# Use the best grasp pose for the red cube
grasp_pos = best_T_world[:3, 3]
grasp_rot = best_T_world[:3, :3]

# Convert rotation matrix to quaternion (wxyz)
def rot_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

grasp_quat = rot_to_quat(grasp_rot)

# Pre-grasp: back off along approach direction (z axis of grasp frame)
approach = grasp_rot[:, 2]
pre_grasp_pos = grasp_pos - approach * 0.08

open_gripper()
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()

# Lift the red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Now I need to place the red cube on top of the green cube. Let me compute the target position above the green cube.

```python
green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

# Target: red cube resting on top of green cube
# Green top surface z = green_center[2] + green_extent[2]/2
# Red cube half height = red_extent[2]/2
target_z = green_center[2] + green_extent[2]/2 + red_extent[2]/2
target_pos = np.array([green_center[0], green_center[1], target_z])

# Keep the same orientation as grasp, approach from above
# Use downward-facing orientation: gripper pointing down
down_quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, 180 about x -> gripper down

# Move above target
above_target = target_pos + np.array([0, 0, 0.12])
joints = solve_ik(above_target, down_quat)
move_to_joints(joints)

# Lower to place
joints = solve_ik(target_pos, down_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = target_pos + np.array([0, 0, 0.12])
joints = solve_ik(retreat_pos, down_quat)
move_to_joints(joints)
```

Let me verify the result with a new observation.

```python
obs2 = get_observation()
rgb2 = obs2["robot0_robotview"]["images"]["rgb"]
red_masks2 = segment_sam3_text_prompt(rgb2, text_prompt="red cube")
green_masks2 = segment_sam3_text_prompt(rgb2, text_prompt="green cube")
print("red box after:", max(red_masks2, key=lambda d: d["score"])["box"])
print("green box after:", max(green_masks2, key=lambda d: d["score"])["box"])

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

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
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

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]

grasp_pos = best_T_world[:3, 3]
grasp_rot = best_T_world[:3, :3]

def rot_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

grasp_quat = rot_to_quat(grasp_rot)
approach = grasp_rot[:, 2]
pre_grasp_pos = grasp_pos - approach * 0.08

open_gripper()
move_to_joints(solve_ik(pre_grasp_pos, grasp_quat))
move_to_joints(solve_ik(grasp_pos, grasp_quat))
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
move_to_joints(solve_ik(lift_pos, grasp_quat))

green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]
target_z = green_center[2] + green_extent[2]/2 + red_extent[2]/2
target_pos = np.array([green_center[0], green_center[1], target_z])

down_quat = np.array([0.0, 1.0, 0.0, 0.0])
above_target = target_pos + np.array([0, 0, 0.12])
move_to_joints(solve_ik(above_target, down_quat))
move_to_joints(solve_ik(target_pos, down_quat))
open_gripper()
retreat_pos = target_pos + np.array([0, 0, 0.12])
move_to_joints(solve_ik(retreat_pos, down_quat))

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

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
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

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

# The green cube is currently on top of the red cube. We need to grasp the
# green cube (top one) and move it aside, then grasp the red cube and place it
# on top of the green cube.

# First, grasp the green cube (which is on top, easy to grasp from above).
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
grasp_rot = best_T_world[:3, :3]

def rot_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

grasp_quat = rot_to_quat(grasp_rot)
approach = grasp_rot[:, 2]
pre_grasp_pos = grasp_pos - approach * 0.08

open_gripper()
move_to_joints(solve_ik(pre_grasp_pos, grasp_quat))
move_to_joints(solve_ik(grasp_pos, grasp_quat))
close_gripper()

# Lift green cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
move_to_joints(solve_ik(lift_pos, grasp_quat))

# Move green cube aside (to the side, away from red cube location)
green_center = green_obb["center"]
red_center = red_obb["center"]
# Place green cube to the side of the red cube
side_offset = np.array([0.0, 0.15, 0.0])
green_place_pos = np.array([red_center[0], red_center[1] + 0.15, green_center[2]])
above_green_place = green_place_pos + np.array([0, 0, 0.12])
move_to_joints(solve_ik(above_green_place, grasp_quat))
move_to_joints(solve_ik(green_place_pos, grasp_quat))
open_gripper()
move_to_joints(solve_ik(above_green_place, grasp_quat))

# Now grasp the red cube (now exposed on top)
obs2 = get_observation()
cam2 = obs2["robot0_robotview"]
rgb2 = cam2["images"]["rgb"]
depth2 = cam2["images"]["depth"]
K2 = cam2["intrinsics"]
cam_ext2 = cam2["pose_mat"]

red_masks2 = segment_sam3_text_prompt(rgb2, text_prompt="red cube")
red_best2 = max(red_masks2, key=lambda d: d["score"])
red_mask2 = red_best2["mask"]

red_pts2 = mask_to_points(red_mask2, depth2, K2, cam_ext2)
red_obb2 = get_oriented_bounding_box_from_3d_points(red_pts2)

grasp_poses2, grasp_scores2 = plan_grasp(depth=depth2, intrinsics=K2, segmentation=red_mask2.astype(np.int32))
best_idx2 = grasp_scores2.argmax()
best_T_world2 = cam_ext2 @ grasp_poses2[best_idx2]
grasp_pos2 = best_T_world2[:3, 3]
grasp_rot2 = best_T_world2[:3, :3]
grasp_quat2 = rot_to_quat(grasp_rot2)
approach2 = grasp_rot2[:, 2]
pre_grasp_pos2 = grasp_pos2 - approach2 * 0.08

open_gripper()
move_to_joints(solve_ik(pre_grasp_pos2, grasp_quat2))
move_to_joints(solve_ik(grasp_pos2, grasp_quat2))
close_gripper()

lift_pos2 = grasp_pos2 + np.array([0, 0, 0.15])
move_to_joints(solve_ik(lift_pos2, grasp_quat2))

# Now place red cube on top of green cube
# Re-observe green cube position
green_masks2 = segment_sam3_text_prompt(rgb2, text_prompt="green cube")
green_best2 = max(green_masks2, key=lambda d: d["score"])
green_mask2 = green_best2["mask"]
green_pts2 = mask_to_points(green_mask2, depth2, K2, cam_ext2)
green_obb2 = get_oriented_bounding_box_from_3d_points(green_pts2)

green_center2 = green_obb2["center"]
green_extent2 = green_obb2["extent"]
red_extent2 = red_obb2["extent"]
target_z = green_center2[2] + green_extent2[2]/2 + red_extent2[2]/2
target_pos = np.array([green_center2[0], green_center2[1], target_z])

down_quat = np.array([0.0, 1.0, 0.0, 0.0])
above_target = target_pos + np.array([0, 0, 0.12])
move_to_joints(solve_ik(above_target, down_quat))
move_to_joints(solve_ik(target_pos, down_quat))
open_gripper()
retreat_pos = target_pos + np.array([0, 0, 0.12])
move_to_joints(solve_ik(retreat_pos, down_quat))