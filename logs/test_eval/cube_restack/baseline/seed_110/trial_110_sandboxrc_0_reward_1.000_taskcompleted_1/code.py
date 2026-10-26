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

print("green masks:", [(m["score"], m["box"]) for m in green_masks])
print("red masks:", [(m["score"], m["box"]) for m in red_masks])
```

```python
import numpy as np

def mask_to_world_points(mask, depth, K, cam_ext):
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

green_mask = max(green_masks, key=lambda d: d["score"])["mask"]
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]

green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)

green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)

print("green center:", green_obb["center"], "extent:", green_obb["extent"])
print("red center:", red_obb["center"], "extent:", red_obb["extent"])
```

```python
import numpy as np

# Plan grasps for green cube (top object) to move it aside
green_grasps, green_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_g = green_scores.argmax()
green_T_world = cam_ext @ green_grasps[best_g]
print("green grasp world pos:", green_T_world[:3, 3])
print("green grasp score:", green_scores[best_g])
```

```python
import numpy as np

# Approach green cube from above and grasp it
grasp_pos = green_T_world[:3, 3]
# Use downward-facing orientation (gripper pointing down)
# quaternion wxyz for 180deg about x: gripper pointing down
quat_down = np.array([0.0, 1.0, 0.0, 0.0])

# Pre-grasp above
pre = grasp_pos.copy()
pre[2] += 0.10
joints = solve_ik(pre, quat_down)
move_to_joints(joints)

open_gripper()

# Move down to grasp
joints = solve_ik(grasp_pos, quat_down)
move_to_joints(joints)
close_gripper()

# Lift
lift = grasp_pos.copy()
lift[2] += 0.15
joints = solve_ik(lift, quat_down)
move_to_joints(joints)
```

```python
import numpy as np

# Place green cube aside (offset in y away from red cube)
place_pos = green_obb["center"].copy()
place_pos[1] += 0.12  # move aside
place_pos[2] = green_obb["center"][2] + 0.02  # near table level

# Move above place location
above = place_pos.copy()
above[2] += 0.12
joints = solve_ik(above, quat_down)
move_to_joints(joints)

# Lower
joints = solve_ik(place_pos, quat_down)
move_to_joints(joints)
open_gripper()

# Retract
joints = solve_ik(above, quat_down)
move_to_joints(joints)
```

```python
import numpy as np

# Now grasp the red cube
red_grasps, red_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_r = red_scores.argmax()
red_T_world = cam_ext @ red_grasps[best_r]
red_grasp_pos = red_T_world[:3, 3]
print("red grasp world pos:", red_grasp_pos, "score:", red_scores[best_r])

# Approach from above
pre = red_grasp_pos.copy()
pre[2] += 0.10
joints = solve_ik(pre, quat_down)
move_to_joints(joints)

open_gripper()
joints = solve_ik(red_grasp_pos, quat_down)
move_to_joints(joints)
close_gripper()

# Lift
lift = red_grasp_pos.copy()
lift[2] += 0.15
joints = solve_ik(lift, quat_down)
move_to_joints(joints)
```

```python
import numpy as np

# Place red cube on top of green cube
# Green cube is now at place_pos; its top is at place_pos[2] + half height
green_half_h = green_obb["extent"][2] / 2.0
red_half_h = red_obb["extent"][2] / 2.0
target = place_pos.copy()
target[2] = place_pos[2] + green_half_h + red_half_h + 0.005

above = target.copy()
above[2] += 0.12
joints = solve_ik(above, quat_down)
move_to_joints(joints)

joints = solve_ik(target, quat_down)
move_to_joints(joints)

open_gripper()

# Retract
joints = solve_ik(above, quat_down)
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

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]

green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)

quat_down = np.array([0.0, 1.0, 0.0, 0.0])

# Grasp the green cube (currently on top) and move it aside
green_grasps, green_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_g = green_scores.argmax()
green_T_world = cam_ext @ green_grasps[best_g]
green_grasp_pos = green_T_world[:3, 3]

pre = green_grasp_pos.copy(); pre[2] += 0.10
move_to_joints(solve_ik(pre, quat_down))
open_gripper()
move_to_joints(solve_ik(green_grasp_pos, quat_down))
close_gripper()
lift = green_grasp_pos.copy(); lift[2] += 0.15
move_to_joints(solve_ik(lift, quat_down))

# Place green cube aside
green_place = green_obb["center"].copy()
green_place[1] += 0.15
green_place[2] = green_obb["center"][2]
above = green_place.copy(); above[2] += 0.12
move_to_joints(solve_ik(above, quat_down))
move_to_joints(solve_ik(green_place, quat_down))
open_gripper()
move_to_joints(solve_ik(above, quat_down))

# Grasp the red cube (now exposed)
red_grasps, red_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_r = red_scores.argmax()
red_T_world = cam_ext @ red_grasps[best_r]
red_grasp_pos = red_T_world[:3, 3]

pre = red_grasp_pos.copy(); pre[2] += 0.10
move_to_joints(solve_ik(pre, quat_down))
open_gripper()
move_to_joints(solve_ik(red_grasp_pos, quat_down))
close_gripper()
lift = red_grasp_pos.copy(); lift[2] += 0.15
move_to_joints(solve_ik(lift, quat_down))

# Place red cube on top of green cube
green_half_h = green_obb["extent"][2] / 2.0
red_half_h = red_obb["extent"][2] / 2.0
target = green_place.copy()
target[2] = green_place[2] + green_half_h + red_half_h + 0.005
above = target.copy(); above[2] += 0.12
move_to_joints(solve_ik(above, quat_down))
move_to_joints(solve_ik(target, quat_down))
open_gripper()
move_to_joints(solve_ik(above, quat_down))

# Code block 2
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_world_points(mask, depth, K, cam_ext):
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

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]

green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)

print("green center:", green_obb["center"], "extent:", green_obb["extent"])
print("red center:", red_obb["center"], "extent:", red_obb["extent"])

quat_down = np.array([0.0, 1.0, 0.0, 0.0])

# Grasp the red cube
red_grasps, red_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_r = red_scores.argmax()
red_T_world = cam_ext @ red_grasps[best_r]
red_grasp_pos = red_T_world[:3, 3]
print("red grasp pos:", red_grasp_pos)

pre = red_grasp_pos.copy(); pre[2] += 0.10
move_to_joints(solve_ik(pre, quat_down))
open_gripper()
move_to_joints(solve_ik(red_grasp_pos, quat_down))
close_gripper()
lift = red_grasp_pos.copy(); lift[2] += 0.15
move_to_joints(solve_ik(lift, quat_down))

# Place red cube on top of green cube (green stays where it is)
green_half_h = green_obb["extent"][2] / 2.0
red_half_h = red_obb["extent"][2] / 2.0
target = green_obb["center"].copy()
target[2] = green_obb["center"][2] + green_half_h + red_half_h + 0.005

above = target.copy(); above[2] += 0.12
move_to_joints(solve_ik(above, quat_down))
move_to_joints(solve_ik(target, quat_down))
open_gripper()
move_to_joints(solve_ik(above, quat_down))