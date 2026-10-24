I'll start by getting an observation to understand the scene, then segment the cubes and plan grasps.

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

# Compute 3D centers of each cube using depth + intrinsics
def mask_to_3d_center(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    pts_cam = np.stack([X, Y, Z], axis=1)
    pts_world = (cam_ext[:3, :3] @ pts_cam.T).T + cam_ext[:3, 3]
    return pts_world.mean(axis=0), pts_world

red_center, red_pts = mask_to_3d_center(red_mask, depth, K, cam_ext)
green_center, green_pts = mask_to_3d_center(green_mask, depth, K, cam_ext)
print("red center:", red_center)
print("green center:", green_center)

# Get OBBs to know cube sizes
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
print("red extent:", red_obb["extent"])
print("green extent:", green_obb["extent"])
```

Now I'll plan a grasp on the green cube (top of stack), lift it off, then grasp the red cube and place it on the green cube.

```python
# Plan grasp on green cube (top of stack) to remove it
green_seg = green_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

# Grasp pose in world frame
grasp_pos = best_T_world[:3, 3]
grasp_R = best_T_world[:3, :3]

# Convert rotation to quaternion (wxyz)
def rot_to_quat_wxyz(R):
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
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

grasp_quat = rot_to_quat_wxyz(grasp_R)
print("grasp pos:", grasp_pos, "quat:", grasp_quat)

# Approach from above: pre-grasp position
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.10])

open_gripper()
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()

# Lift green cube up
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Now I'll move the green cube aside, then grasp the red cube and place it on the green cube.

```python
# Move green cube to the side (away from red cube)
# Determine a side offset direction based on relative positions
side_offset = np.array([0.15, 0.0, 0.0])
green_place_pos = green_center + side_offset
green_place_pos[2] = green_center[2] + 0.05  # keep lifted a bit

# Move to place green cube aside
joints = solve_ik(green_place_pos + np.array([0, 0, 0.10]), grasp_quat)
move_to_joints(joints)
joints = solve_ik(green_place_pos, grasp_quat)
move_to_joints(joints)
open_gripper()

# Retreat
joints = solve_ik(green_place_pos + np.array([0, 0, 0.15]), grasp_quat)
move_to_joints(joints)

# Now grasp the red cube
red_seg = red_mask.astype(np.int32)
red_grasp_poses, red_grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
red_best_idx = red_grasp_scores.argmax()
red_T_world = cam_ext @ red_grasp_poses[red_best_idx]
red_grasp_pos = red_T_world[:3, 3]
red_grasp_R = red_T_world[:3, :3]
red_grasp_quat = rot_to_quat_wxyz(red_grasp_R)

print("red grasp pos:", red_grasp_pos)

# Approach red cube
joints = solve_ik(red_grasp_pos + np.array([0, 0, 0.10]), red_grasp_quat)
move_to_joints(joints)
joints = solve_ik(red_grasp_pos, red_grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
joints = solve_ik(red_grasp_pos + np.array([0, 0, 0.15]), red_grasp_quat)
move_to_joints(joints)
```

Now place the red cube on top of the green cube (which is now at green_place_pos).

```python
# Place red cube on top of green cube
# Green cube top height: green_place_pos z + half cube height
green_half_h = green_obb["extent"][2] / 2.0
red_half_h = red_obb["extent"][2] / 2.0

# Target: red cube center should be above green cube center
place_pos = np.array([green_place_pos[0], green_place_pos[1], green_place_pos[2] + green_half_h + red_half_h + 0.01])

# Move above placement
joints = solve_ik(place_pos + np.array([0, 0, 0.10]), red_grasp_quat)
move_to_joints(joints)
joints = solve_ik(place_pos, red_grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
joints = solve_ik(place_pos + np.array([0, 0, 0.15]), red_grasp_quat)
move_to_joints(joints)
```