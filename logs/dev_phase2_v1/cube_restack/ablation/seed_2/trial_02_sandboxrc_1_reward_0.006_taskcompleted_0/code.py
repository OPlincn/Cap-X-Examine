# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment the red and green cubes
red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_res, key=lambda d: d["score"])
green_best = max(green_res, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

# Compute 3D points for each cube to get their centers in world frame
def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    pts_cam = np.stack([X, Y, Z], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```

Now I'll plan a grasp on the red cube (the top-most object is green, so I need to first remove green, or grasp red from the side). Since green is on top of red, I'll grasp the green cube first and move it aside, then grasp red and place it on green.

```python
# Plan grasp on green cube (top object) to move it aside
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
open_gripper()
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift green cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move green aside (to the side of the table)
aside_pos = np.array([green_center[0] + 0.15, green_center[1] + 0.15, green_center[2] + 0.15])
joints = solve_ik(aside_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
```

Now grasp the red cube and place it on top of the green cube.

```python
# Re-observe to get updated positions
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_res, key=lambda d: d["score"])["mask"]
green_mask = max(green_res, key=lambda d: d["score"])["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Plan grasp on red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
grasp_quat = rot_to_quat(best_T_world[:3, :3])

approach_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Place red on top of green: target = green center + half heights
place_z = green_center[2] + green_extent[2] / 2 + red_extent[2] / 2 + 0.01
place_pos = np.array([green_center[0], green_center[1], place_z + 0.1])
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

# Lower to place
place_pos_low = np.array([green_center[0], green_center[1], place_z])
joints = solve_ik(place_pos_low, grasp_quat)
move_to_joints(joints)

open_gripper()

# Retreat
retreat_pos = place_pos_low + np.array([0, 0, 0.15])
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

def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    pts_cam = np.stack([X, Y, Z], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
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

red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_res, key=lambda d: d["score"])["mask"]
green_mask = max(green_res, key=lambda d: d["score"])["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Green is on top. Grasp green and move it aside.
green_seg = green_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
grasp_quat = rot_to_quat(best_T_world[:3, :3])

approach_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move green aside
aside_pos = np.array([green_center[0] + 0.2, green_center[1] + 0.2, green_center[2] + 0.15])
joints = solve_ik(aside_pos, grasp_quat)
move_to_joints(joints)
open_gripper()

# Retreat up
retreat_pos = aside_pos + np.array([0, 0, 0.1])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)
```

```python
# Re-observe to grasp red cube and place on green
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_res, key=lambda d: d["score"])["mask"]
green_mask = max(green_res, key=lambda d: d["score"])["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
grasp_quat = rot_to_quat(best_T_world[:3, :3])

approach_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Place red on top of green
place_z = green_center[2] + green_extent[2] / 2 + red_extent[2] / 2 + 0.01
place_pos = np.array([green_center[0], green_center[1], place_z + 0.1])
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

place_pos_low = np.array([green_center[0], green_center[1], place_z])
joints = solve_ik(place_pos_low, grasp_quat)
move_to_joints(joints)

open_gripper()

retreat_pos = place_pos_low + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)