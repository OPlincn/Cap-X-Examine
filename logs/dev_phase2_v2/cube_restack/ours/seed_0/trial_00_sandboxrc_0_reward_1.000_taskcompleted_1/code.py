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

print("red box", red_best.get("box"), "score", red_best["score"])
print("green box", green_best.get("box"), "score", green_best["score"])
```

Now let me compute 3D positions of both cubes to understand the geometry.

```python
def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
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

The green cube is on top of the red cube. I need to grasp the red cube (which is underneath), lift it, and place it on top of the green cube. But the green cube is on top of the red cube, so I first need to move the green cube out of the way, or grasp the red cube from the side.

Let me think about the strategy. The green cube sits on the red cube. To place the red cube on top of the green cube, I should:
1. First move the green cube aside (grasp green, place it on the table).
2. Then grasp the red cube and place it on top of the green cube.

Let me first grasp the green cube and move it to the side.

```python
# Plan grasp for green cube to move it aside
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
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

grasp_quat = rot_to_quat(R)
print("green grasp pos", grasp_pos, "quat", grasp_quat)
```

Let me approach the green cube from above and grasp it, then move it aside.

```python
# Approach green cube from above
approach_pos = grasp_pos.copy()
approach_pos[2] += 0.10
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)
open_gripper()

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Now move the green cube to a side location on the table.

```python
# Place green cube to the side (away from red cube)
side_pos = green_center.copy()
side_pos[0] += 0.15  # move in x direction
side_pos[2] = green_center[2]  # keep at table height roughly

# Move above side location
above_side = side_pos.copy()
above_side[2] += 0.15
joints = solve_ik(above_side, grasp_quat)
move_to_joints(joints)

# Lower to place
joints = solve_ik(side_pos, grasp_quat)
move_to_joints(joints)
open_gripper()

# Retreat
joints = solve_ik(above_side, grasp_quat)
move_to_joints(joints)
```

Now I need to grasp the red cube and place it on top of the green cube. Let me get a new observation to find the green cube's new position.

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
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center)
print("green center", green_center, "extent", green_extent)
```

Now grasp the red cube and place it on top of the green cube.

```python
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
red_grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]
red_grasp_quat = rot_to_quat(R)
print("red grasp pos", red_grasp_pos, "quat", red_grasp_quat)

# Approach red cube
approach_pos = red_grasp_pos.copy()
approach_pos[2] += 0.10
joints = solve_ik(approach_pos, red_grasp_quat)
move_to_joints(joints)
open_gripper()

# Grasp
joints = solve_ik(red_grasp_pos, red_grasp_quat)
move_to_joints(joints)
close_gripper()

# Lift
lift_pos = red_grasp_pos.copy()
lift_pos[2] += 0.20
joints = solve_ik(lift_pos, red_grasp_quat)
move_to_joints(joints)
```

Now place the red cube on top of the green cube. The target position is above the green cube center by half the red cube height plus half the green cube height.

```python
# Target: on top of green cube
red_height = red_obb["extent"][2]
green_height = green_extent[2]
place_pos = green_center.copy()
place_pos[2] = green_center[2] + green_height/2 + red_height/2

# Move above place position
above_place = place_pos.copy()
above_place[2] += 0.15
joints = solve_ik(above_place, red_grasp_quat)
move_to_joints(joints)

# Lower to place
joints = solve_ik(place_pos, red_grasp_quat)
move_to_joints(joints)

# Open gripper
open_gripper()

# Retreat
joints = solve_ik(above_place, red_grasp_quat)
move_to_joints(joints)

# Code block 1
import numpy as np

def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

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

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center)
print("green center", green_center, "extent", green_extent)

# Grasp green cube
green_seg = green_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
green_grasp_pos = best_T_world[:3, 3]
green_grasp_quat = rot_to_quat(best_T_world[:3, :3])
print("green grasp pos", green_grasp_pos, "quat", green_grasp_quat)

open_gripper()
approach_pos = green_grasp_pos.copy()
approach_pos[2] += 0.10
move_to_joints(solve_ik(approach_pos, green_grasp_quat))
move_to_joints(solve_ik(green_grasp_pos, green_grasp_quat))
close_gripper()
lift_pos = green_grasp_pos.copy()
lift_pos[2] += 0.15
move_to_joints(solve_ik(lift_pos, green_grasp_quat))

# Move green aside
side_pos = green_center.copy()
side_pos[0] += 0.15
above_side = side_pos.copy()
above_side[2] += 0.15
move_to_joints(solve_ik(above_side, green_grasp_quat))
move_to_joints(solve_ik(side_pos, green_grasp_quat))
open_gripper()
move_to_joints(solve_ik(above_side, green_grasp_quat))

# Re-observe to find new green position
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

# Grasp red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
red_grasp_pos = best_T_world[:3, 3]
red_grasp_quat = rot_to_quat(best_T_world[:3, :3])
print("red grasp pos", red_grasp_pos, "quat", red_grasp_quat)

open_gripper()
approach_pos = red_grasp_pos.copy()
approach_pos[2] += 0.10
move_to_joints(solve_ik(approach_pos, red_grasp_quat))
move_to_joints(solve_ik(red_grasp_pos, red_grasp_quat))
close_gripper()
lift_pos = red_grasp_pos.copy()
lift_pos[2] += 0.20
move_to_joints(solve_ik(lift_pos, red_grasp_quat))

# Place red on top of green
place_pos = green_center.copy()
place_pos[2] = green_center[2] + green_extent[2]/2 + red_extent[2]/2
above_place = place_pos.copy()
above_place[2] += 0.15
move_to_joints(solve_ik(above_place, red_grasp_quat))
move_to_joints(solve_ik(place_pos, red_grasp_quat))
open_gripper()
move_to_joints(solve_ik(above_place, red_grasp_quat))

# Code block 2
import numpy as np

def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts_cam = np.stack([x_cam, y_cam, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

# Move arm away to get a clear view
home_joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
move_to_joints(home_joints)
open_gripper()

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
print("num red masks", len(red_masks), "num green masks", len(green_masks))
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
print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)

# Grasp red cube
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = grasp_scores.argmax()
best_T_world = cam_ext @ grasp_poses[best_idx]
red_grasp_pos = best_T_world[:3, 3]
red_grasp_quat = rot_to_quat(best_T_world[:3, :3])
print("red grasp pos", red_grasp_pos, "quat", red_grasp_quat)

open_gripper()
approach_pos = red_grasp_pos.copy()
approach_pos[2] += 0.10
move_to_joints(solve_ik(approach_pos, red_grasp_quat))
move_to_joints(solve_ik(red_grasp_pos, red_grasp_quat))
close_gripper()
lift_pos = red_grasp_pos.copy()
lift_pos[2] += 0.20
move_to_joints(solve_ik(lift_pos, red_grasp_quat))

# Place red on top of green
place_pos = green_center.copy()
place_pos[2] = green_center[2] + green_extent[2]/2 + red_extent[2]/2
above_place = place_pos.copy()
above_place[2] += 0.15
move_to_joints(solve_ik(above_place, red_grasp_quat))
move_to_joints(solve_ik(place_pos, red_grasp_quat))
open_gripper()
move_to_joints(solve_ik(above_place, red_grasp_quat))