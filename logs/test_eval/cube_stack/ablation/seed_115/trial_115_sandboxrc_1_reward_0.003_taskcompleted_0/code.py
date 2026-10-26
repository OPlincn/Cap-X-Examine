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

def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xc = (xs - cx) * zs / fx
    yc = (ys - cy) * zs / fy
    pts_cam = np.stack([xc, yc, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_obb["extent"])
print("green center", green_center, "extent", green_extent)
```

Now I'll plan a grasp on the red cube using Contact-GraspNet.

```python
grasp_poses, grasp_scores = plan_grasp(
    depth=depth,
    intrinsics=K,
    segmentation=red_mask.astype(np.int32),
)
best_idx = int(np.argmax(grasp_scores))
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
print("best grasp world:\n", best_T_world)
print("score", grasp_scores[best_idx])
```

I'll approach the red cube from above with a downward-facing gripper for a reliable top grasp.

```python
# Pre-grasp above red cube
red_top = red_center.copy()
red_top[2] = red_center[2] + 0.15

# Downward-facing orientation (gripper pointing down): rotate 180 deg about x
# quaternion wxyz for rotation of pi about x-axis: (0, 1, 0, 0)
down_quat = np.array([0.0, 1.0, 0.0, 0.0])

open_gripper()
joints = solve_ik(red_top, down_quat)
move_to_joints(joints)

# Move down to grasp height
grasp_pos = red_center.copy()
grasp_pos[2] = red_center[2] + 0.005
joints = solve_ik(grasp_pos, down_quat)
move_to_joints(joints)

close_gripper()

# Lift the red cube
lift_pos = red_center.copy()
lift_pos[2] = red_center[2] + 0.20
joints = solve_ik(lift_pos, down_quat)
move_to_joints(joints)
```

Now I'll move the red cube above the green cube and place it on top.

```python
# Target position: above green cube center, at green top + red half-height
red_half_height = red_obb["extent"][2] / 2.0
green_top_z = green_center[2] + green_extent[2] / 2.0

place_above = np.array([green_center[0], green_center[1], green_top_z + red_half_height + 0.15])
joints = solve_ik(place_above, down_quat)
move_to_joints(joints)

# Lower to place
place_pos = np.array([green_center[0], green_center[1], green_top_z + red_half_height + 0.005])
joints = solve_ik(place_pos, down_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos.copy()
retreat_pos[2] += 0.15
joints = solve_ik(retreat_pos, down_quat)
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

def mask_to_world_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xc = (xs - cx) * zs / fx
    yc = (ys - cy) * zs / fy
    pts_cam = np.stack([xc, yc, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

down_quat = np.array([0.0, 1.0, 0.0, 0.0])

open_gripper()

# Pre-grasp above red cube
red_top = red_center.copy()
red_top[2] = red_center[2] + 0.15
joints = solve_ik(red_top, down_quat)
move_to_joints(joints)

# Move down to grasp height
grasp_pos = red_center.copy()
grasp_pos[2] = red_center[2] + 0.005
joints = solve_ik(grasp_pos, down_quat)
move_to_joints(joints)

close_gripper()

# Lift the red cube
lift_pos = red_center.copy()
lift_pos[2] = red_center[2] + 0.20
joints = solve_ik(lift_pos, down_quat)
move_to_joints(joints)

# Target position: above green cube center, at green top + red half-height
red_half_height = red_extent[2] / 2.0
green_top_z = green_center[2] + green_extent[2] / 2.0

place_above = np.array([green_center[0], green_center[1], green_top_z + red_half_height + 0.15])
joints = solve_ik(place_above, down_quat)
move_to_joints(joints)

# Lower to place
place_pos = np.array([green_center[0], green_center[1], green_top_z + red_half_height + 0.005])
joints = solve_ik(place_pos, down_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos.copy()
retreat_pos[2] += 0.15
joints = solve_ik(retreat_pos, down_quat)
move_to_joints(joints)