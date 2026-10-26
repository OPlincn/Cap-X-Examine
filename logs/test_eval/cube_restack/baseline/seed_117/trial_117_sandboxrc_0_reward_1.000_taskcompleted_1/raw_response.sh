I'll start by observing the scene and locating the cubes.

```python
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
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
print("red center", red_center, "extent", red_obb["extent"])
print("green center", green_center, "extent", green_obb["extent"])
```

Now I'll grasp the green cube (on top), lift it, place it on the table, then grasp the red cube and place it on the green cube.

```python
# Grasp the green cube (top one) first
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

# Approach from above: use position of grasp, orientation pointing down
grasp_pos = best_T_world[:3, 3]
# Downward orientation quaternion (wxyz): rotate 180 about x
down_quat = np.array([0.0, 1.0, 0.0, 0.0])

# Move above green cube
pre_pos = grasp_pos.copy()
pre_pos[2] += 0.10
joints = solve_ik(pre_pos, down_quat)
move_to_joints(joints)

# Descend to grasp
joints = solve_ik(grasp_pos, down_quat)
move_to_joints(joints)
close_gripper()

# Lift green cube
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, down_quat)
move_to_joints(joints)
```

Now place the green cube on the table beside the red cube, then release it.

```python
# Place green cube on table to the side (offset in y)
place_pos = green_center.copy()
place_pos[2] = red_center[2] - 0.02  # table level near red cube base
place_pos[1] += 0.12  # move aside

# Move above placement
above_place = place_pos.copy()
above_place[2] += 0.15
joints = solve_ik(above_place, down_quat)
move_to_joints(joints)

joints = solve_ik(place_pos, down_quat)
move_to_joints(joints)
open_gripper()

# Retreat
joints = solve_ik(above_place, down_quat)
move_to_joints(joints)
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

green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_top_z = green_center[2] + green_obb["extent"][2] / 2.0

# Grasp red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos = best_T_world[:3, 3]

pre_pos = grasp_pos.copy()
pre_pos[2] += 0.10
joints = solve_ik(pre_pos, down_quat)
move_to_joints(joints)

joints = solve_ik(grasp_pos, down_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, down_quat)
move_to_joints(joints)
```

Now place the red cube on top of the green cube.

```python
# Place red cube on top of green cube
red_extent_z = red_obb["extent"][2] if 'red_obb' in dir() else 0.04
place_pos = np.array([green_center[0], green_center[1], green_top_z + 0.02 + 0.02])

above_place = place_pos.copy()
above_place[2] += 0.15
joints = solve_ik(above_place, down_quat)
move_to_joints(joints)

joints = solve_ik(place_pos, down_quat)
move_to_joints(joints)
open_gripper()

# Retreat
joints = solve_ik(above_place, down_quat)
move_to_joints(joints)
```