# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment the two cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_masks, key=lambda d: d["score"])
green_best = max(green_masks, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    pts_cam = np.stack([x, y, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```

Now I'll plan a grasp on the green cube (the top one), lift it off, set it aside, then grasp the red cube and place it on the green cube.

```python
# Grasp the green cube (top) first to remove it
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_quat = None
# Convert rotation to quaternion wxyz
R = best_T_world[:3, :3]
def rot_to_quat(R):
    w = np.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2
    x = np.sqrt(max(0, 1 + R[0,0] - R[1,1] - R[2,2])) / 2
    y = np.sqrt(max(0, 1 - R[0,0] + R[1,1] - R[2,2])) / 2
    z = np.sqrt(max(0, 1 - R[0,0] - R[1,1] + R[2,2])) / 2
    x = np.copysign(x, R[2,1] - R[1,2])
    y = np.copysign(y, R[0,2] - R[2,0])
    z = np.copysign(z, R[1,0] - R[0,1])
    return np.array([w, x, y, z])
grasp_quat = rot_to_quat(R)

# Approach from above
approach_pos = grasp_pos + np.array([0, 0, 0.1])
open_gripper()
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift green cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move green cube aside (to the side, away from red)
set_aside_pos = np.array([green_center[0] + 0.15, green_center[1] + 0.15, green_center[2] + 0.15])
joints = solve_ik(set_aside_pos, grasp_quat)
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

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Grasp red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]
grasp_quat = rot_to_quat(R)

open_gripper()
approach_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Place red cube on top of green cube
place_z = green_center[2] + green_extent[2] / 2 + red_extent[2] / 2 + 0.01
place_pos = np.array([green_center[0], green_center[1], place_z + 0.1])
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

place_pos_final = np.array([green_center[0], green_center[1], place_z])
joints = solve_ik(place_pos_final, grasp_quat)
move_to_joints(joints)

open_gripper()

# Retreat
retreat_pos = place_pos_final + np.array([0, 0, 0.15])
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

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    pts_cam = np.stack([x, y, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

def rot_to_quat(R):
    w = np.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2
    x = np.sqrt(max(0, 1 + R[0,0] - R[1,1] - R[2,2])) / 2
    y = np.sqrt(max(0, 1 - R[0,0] + R[1,1] - R[2,2])) / 2
    z = np.sqrt(max(0, 1 - R[0,0] - R[1,1] + R[2,2])) / 2
    x = np.copysign(x, R[2,1] - R[1,2])
    y = np.copysign(y, R[0,2] - R[2,0])
    z = np.copysign(z, R[1,0] - R[0,1])
    return np.array([w, x, y, z])

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Grasp red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]
grasp_quat = rot_to_quat(R)

open_gripper()
approach_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Place red cube on top of green cube
place_z = green_center[2] + green_extent[2] / 2 + red_extent[2] / 2 + 0.01
place_pos = np.array([green_center[0], green_center[1], place_z + 0.1])
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

place_pos_final = np.array([green_center[0], green_center[1], place_z])
joints = solve_ik(place_pos_final, grasp_quat)
move_to_joints(joints)

open_gripper()

# Retreat
retreat_pos = place_pos_final + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)