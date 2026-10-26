# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
print("red:", [(m["score"], m.get("box")) for m in red_masks])
print("green:", [(m["score"], m.get("box")) for m in green_masks])
```

```python
red = max(red_masks, key=lambda d: d["score"])
green = max(green_masks, key=lambda d: d["score"])
red_mask = red["mask"]
green_mask = green["mask"]

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
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
print("red center", red_obb["center"], "extent", red_obb["extent"])
print("green center", green_obb["center"], "extent", green_obb["extent"])
```

```python
# Plan grasp for green cube (top object) to move it aside
green_seg = green_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
print("green grasp world pos:", best_T_world[:3, 3])
print("score:", grasp_scores[best_idx])
```

```python
from scipy.spatial.transform import Rotation as R

def T_to_pos_quat(T):
    pos = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    q = rot.as_quat()  # xyzw
    return pos, np.array([q[3], q[0], q[1], q[2]])  # wxyz

# Approach green cube from above
grasp_pos, grasp_quat = T_to_pos_quat(best_T_world)
approach_pos = grasp_pos + np.array([0, 0, 0.10])

open_gripper()
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()
```

```python
# Lift green cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move green cube aside (away from red cube)
red_center = red_obb["center"]
green_center = green_obb["center"]
# Place green to the side, offset in x/y away from red
offset_dir = green_center[:2] - red_center[:2]
norm = np.linalg.norm(offset_dir)
if norm < 1e-3:
    offset_dir = np.array([1.0, 0.0])
else:
    offset_dir = offset_dir / norm
place_xy = red_center[:2] + offset_dir * 0.20
place_pos = np.array([place_xy[0], place_xy[1], green_center[2]])
place_approach = place_pos + np.array([0, 0, 0.10])

joints = solve_ik(place_approach, grasp_quat)
move_to_joints(joints)
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(place_approach, grasp_quat)
move_to_joints(joints)
```

```python
# Now grasp red cube
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
red = max(red_masks, key=lambda d: d["score"])
red_mask = red["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
print("red center", red_obb["center"], "extent", red_obb["extent"])

red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
print("red grasp world pos:", best_T_world[:3, 3], "score:", grasp_scores[best_idx])
```

```python
grasp_pos, grasp_quat = T_to_pos_quat(best_T_world)
approach_pos = grasp_pos + np.array([0, 0, 0.10])

open_gripper()
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

```python
# Now place red cube on top of green cube
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green = max(green_masks, key=lambda d: d["score"])
green_mask = green["mask"]
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
print("green center", green_obb["center"], "extent", green_obb["extent"])

# Red cube extent for stacking height
red_extent = red_obb["extent"]
green_top_z = green_obb["center"][2] + green_obb["extent"][2] / 2.0
# Target: red cube center at green top + half red height
target_z = green_top_z + red_extent[2] / 2.0
target_pos = np.array([green_obb["center"][0], green_obb["center"][1], target_z])
print("target red center:", target_pos)

place_approach = target_pos + np.array([0, 0, 0.10])
joints = solve_ik(place_approach, grasp_quat)
move_to_joints(joints)
joints = solve_ik(target_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(place_approach, grasp_quat)
move_to_joints(joints)

# Code block 1
import numpy as np
from scipy.spatial.transform import Rotation as R

def T_to_pos_quat(T):
    pos = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    q = rot.as_quat()  # xyzw
    return pos, np.array([q[3], q[0], q[1], q[2]])  # wxyz

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
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

# Observe current state
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red = max(red_masks, key=lambda d: d["score"])
green = max(green_masks, key=lambda d: d["score"])
red_mask = red["mask"]
green_mask = green["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
print("red center", red_obb["center"], "extent", red_obb["extent"])
print("green center", green_obb["center"], "extent", green_obb["extent"])

# Determine which is on top by z center
if green_obb["center"][2] > red_obb["center"][2]:
    top_mask, top_obb = green_mask, green_obb
    bottom_obb = red_obb
    top_name = "green"
else:
    top_mask, top_obb = red_mask, red_obb
    bottom_obb = green_obb
    top_name = "red"
print("top object:", top_name)

# Grasp the top object and move it aside
top_seg = top_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=top_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos, grasp_quat = T_to_pos_quat(best_T_world)
print("top grasp pos:", grasp_pos, "score:", grasp_scores[best_idx])

approach_pos = grasp_pos + np.array([0, 0, 0.10])
open_gripper()
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move top object aside
bottom_center = bottom_obb["center"]
top_center = top_obb["center"]
offset_dir = top_center[:2] - bottom_center[:2]
norm = np.linalg.norm(offset_dir)
if norm < 1e-3:
    offset_dir = np.array([1.0, 0.0])
else:
    offset_dir = offset_dir / norm
place_xy = bottom_center[:2] + offset_dir * 0.20
place_pos = np.array([place_xy[0], place_xy[1], top_center[2]])
place_approach = place_pos + np.array([0, 0, 0.10])

joints = solve_ik(place_approach, grasp_quat)
move_to_joints(joints)
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(place_approach, grasp_quat)
move_to_joints(joints)
```

```python
# Now grasp the bottom object (which is now exposed) and place it on top of the moved object
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red = max(red_masks, key=lambda d: d["score"])
green = max(green_masks, key=lambda d: d["score"])
red_mask = red["mask"]
green_mask = green["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
print("red center", red_obb["center"], "extent", red_obb["extent"])
print("green center", green_obb["center"], "extent", green_obb["extent"])

# Grasp red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
grasp_pos, grasp_quat = T_to_pos_quat(best_T_world)
print("red grasp pos:", grasp_pos, "score:", grasp_scores[best_idx])

approach_pos = grasp_pos + np.array([0, 0, 0.10])
open_gripper()
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Place red cube on top of green cube
green_top_z = green_obb["center"][2] + green_obb["extent"][2] / 2.0
target_z = green_top_z + red_obb["extent"][2] / 2.0
target_pos = np.array([green_obb["center"][0], green_obb["center"][1], target_z])
print("target red center:", target_pos)

place_approach = target_pos + np.array([0, 0, 0.10])
joints = solve_ik(place_approach, grasp_quat)
move_to_joints(joints)
joints = solve_ik(target_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
joints = solve_ik(place_approach, grasp_quat)
move_to_joints(joints)