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
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
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

Now I'll grasp the red cube (which is on the bottom) — actually the red is under the green. I need to move the green cube off first, then place red on green. Let me grasp the green cube first and move it aside.

```python
# Grasp the green cube (top) first
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

# Approach from above: use position of green center, top-down orientation
grasp_pos = best_T_world[:3, 3]
# top-down quaternion (gripper pointing down): w=0, x=1, y=0, z=0
top_down_quat = np.array([0.0, 1.0, 0.0, 0.0])

# Move above green cube
above = grasp_pos.copy()
above[2] += 0.10
joints = solve_ik(above, top_down_quat)
move_to_joints(joints)

# Descend to grasp
joints = solve_ik(grasp_pos, top_down_quat)
move_to_joints(joints)
close_gripper()

# Lift green cube
lift = grasp_pos.copy()
lift[2] += 0.15
joints = solve_ik(lift, top_down_quat)
move_to_joints(joints)
```

Now move the green cube aside to reveal the red cube.

```python
# Move green cube to the side (offset in x)
side_pos = lift.copy()
side_pos[0] += 0.20
joints = solve_ik(side_pos, top_down_quat)
move_to_joints(joints)

# Place green cube down
place_green = side_pos.copy()
place_green[2] -= 0.12
joints = solve_ik(place_green, top_down_quat)
move_to_joints(joints)
open_gripper()

# Lift away
retreat = place_green.copy()
retreat[2] += 0.15
joints = solve_ik(retreat, top_down_quat)
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

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# Grasp red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_T_world = cam_ext @ grasp_poses[grasp_scores.argmax()]
red_grasp_pos = best_T_world[:3, 3]

above = red_grasp_pos.copy()
above[2] += 0.10
joints = solve_ik(above, top_down_quat)
move_to_joints(joints)
joints = solve_ik(red_grasp_pos, top_down_quat)
move_to_joints(joints)
close_gripper()

lift = red_grasp_pos.copy()
lift[2] += 0.15
joints = solve_ik(lift, top_down_quat)
move_to_joints(joints)

# Place red on top of green
place_red = green_center.copy()
place_red[2] = green_center[2] + green_extent[2] / 2.0 + 0.06
joints = solve_ik(place_red, top_down_quat)
move_to_joints(joints)

# Lower slightly and release
place_red_low = place_red.copy()
place_red_low[2] -= 0.02
joints = solve_ik(place_red_low, top_down_quat)
move_to_joints(joints)
open_gripper()

# Retreat
retreat = place_red_low.copy()
retreat[2] += 0.15
joints = solve_ik(retreat, top_down_quat)
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
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

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
green_extent = green_obb["extent"]

top_down_quat = np.array([0.0, 1.0, 0.0, 0.0])

# Grasp green cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
best_T_world = cam_ext @ grasp_poses[grasp_scores.argmax()]
green_grasp_pos = best_T_world[:3, 3]

above = green_grasp_pos.copy()
above[2] += 0.10
move_to_joints(solve_ik(above, top_down_quat))
move_to_joints(solve_ik(green_grasp_pos, top_down_quat))
close_gripper()

lift = green_grasp_pos.copy()
lift[2] += 0.15
move_to_joints(solve_ik(lift, top_down_quat))

# Move green aside
side_pos = lift.copy()
side_pos[0] += 0.20
move_to_joints(solve_ik(side_pos, top_down_quat))

place_green = side_pos.copy()
place_green[2] -= 0.12
move_to_joints(solve_ik(place_green, top_down_quat))
open_gripper()

retreat = place_green.copy()
retreat[2] += 0.15
move_to_joints(solve_ik(retreat, top_down_quat))

# Re-observe
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
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# Grasp red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_T_world = cam_ext @ grasp_poses[grasp_scores.argmax()]
red_grasp_pos = best_T_world[:3, 3]

above = red_grasp_pos.copy()
above[2] += 0.10
move_to_joints(solve_ik(above, top_down_quat))
move_to_joints(solve_ik(red_grasp_pos, top_down_quat))
close_gripper()

lift = red_grasp_pos.copy()
lift[2] += 0.15
move_to_joints(solve_ik(lift, top_down_quat))

# Place red on top of green
place_red = green_center.copy()
place_red[2] = green_center[2] + green_extent[2] / 2.0 + 0.06
move_to_joints(solve_ik(place_red, top_down_quat))

place_red_low = place_red.copy()
place_red_low[2] -= 0.02
move_to_joints(solve_ik(place_red_low, top_down_quat))
open_gripper()

retreat = place_red_low.copy()
retreat[2] += 0.15
move_to_joints(solve_ik(retreat, top_down_quat))

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
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    pts_cam = np.stack([x, y, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

top_down_quat = np.array([0.0, 1.0, 0.0, 0.0])

# Locate both cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

# Grasp the red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_T_world = cam_ext @ grasp_poses[grasp_scores.argmax()]
red_grasp_pos = best_T_world[:3, 3]

above = red_grasp_pos.copy()
above[2] += 0.10
move_to_joints(solve_ik(above, top_down_quat))
move_to_joints(solve_ik(red_grasp_pos, top_down_quat))
close_gripper()

lift = red_grasp_pos.copy()
lift[2] += 0.15
move_to_joints(solve_ik(lift, top_down_quat))

# Place red on top of green
place_red = green_center.copy()
place_red[2] = green_center[2] + green_extent[2] / 2.0 + red_extent[2] / 2.0 + 0.01
move_to_joints(solve_ik(place_red, top_down_quat))

place_red_low = place_red.copy()
place_red_low[2] -= 0.01
move_to_joints(solve_ik(place_red_low, top_down_quat))
open_gripper()

retreat = place_red_low.copy()
retreat[2] += 0.15
move_to_joints(solve_ik(retreat, top_down_quat))

# Code block 3
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
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

top_down_quat = np.array([0.0, 1.0, 0.0, 0.0])

# Locate both cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

# Grasp the red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_T_world = cam_ext @ grasp_poses[grasp_scores.argmax()]
red_grasp_pos = best_T_world[:3, 3]

above = red_grasp_pos.copy()
above[2] += 0.10
move_to_joints(solve_ik(above, top_down_quat))
move_to_joints(solve_ik(red_grasp_pos, top_down_quat))
close_gripper()

lift = red_grasp_pos.copy()
lift[2] += 0.15
move_to_joints(solve_ik(lift, top_down_quat))

# Place red on top of green: target xy = green center, z = green top + red half height
place_red = green_center.copy()
place_red[2] = green_center[2] + green_extent[2] / 2.0 + red_extent[2] / 2.0 + 0.005
move_to_joints(solve_ik(place_red, top_down_quat))

# Release
open_gripper()

# Retreat straight up
retreat = place_red.copy()
retreat[2] += 0.15
move_to_joints(solve_ik(retreat, top_down_quat))

# Code block 4
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
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

top_down_quat = np.array([0.0, 1.0, 0.0, 0.0])

# Locate both cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

# Grasp the red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_T_world = cam_ext @ grasp_poses[grasp_scores.argmax()]
red_grasp_pos = best_T_world[:3, 3]

above = red_grasp_pos.copy()
above[2] += 0.10
move_to_joints(solve_ik(above, top_down_quat))
move_to_joints(solve_ik(red_grasp_pos, top_down_quat))
close_gripper()

lift = red_grasp_pos.copy()
lift[2] += 0.15
move_to_joints(solve_ik(lift, top_down_quat))

# Place red on top of green: target xy = green center, z = green top + red half height
place_red = green_center.copy()
place_red[2] = green_center[2] + green_extent[2] / 2.0 + red_extent[2] / 2.0 + 0.005
move_to_joints(solve_ik(place_red, top_down_quat))

# Release
open_gripper()

# Retreat straight up
retreat = place_red.copy()
retreat[2] += 0.15
move_to_joints(solve_ik(retreat, top_down_quat))

# Code block 5
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
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

top_down_quat = np.array([0.0, 1.0, 0.0, 0.0])

# Locate both cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

# Grasp the red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_T_world = cam_ext @ grasp_poses[grasp_scores.argmax()]
red_grasp_pos = best_T_world[:3, 3]

above = red_grasp_pos.copy()
above[2] += 0.10
move_to_joints(solve_ik(above, top_down_quat))
move_to_joints(solve_ik(red_grasp_pos, top_down_quat))
close_gripper()

lift = red_grasp_pos.copy()
lift[2] += 0.15
move_to_joints(solve_ik(lift, top_down_quat))

# Place red on top of green: target xy = green center, z = green top + red half height
place_red = green_center.copy()
place_red[2] = green_center[2] + green_extent[2] / 2.0 + red_extent[2] / 2.0 + 0.005
move_to_joints(solve_ik(place_red, top_down_quat))

# Release
open_gripper()

# Retreat straight up
retreat = place_red.copy()
retreat[2] += 0.15
move_to_joints(solve_ik(retreat, top_down_quat))