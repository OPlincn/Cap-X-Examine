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

print("red box:", red_best["box"], "score:", red_best["score"])
print("green box:", green_best["box"], "score:", green_best["score"])
```

Now let me get the 3D positions of both cubes to understand the geometry.

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

print("red center:", red_obb["center"], "extent:", red_obb["extent"])
print("green center:", green_obb["center"], "extent:", green_obb["extent"])
```

The green cube is on top of the red cube. I need to grasp the red cube, lift it, and place it on top of the green cube. Let me plan a grasp on the red cube.

```python
# Plan grasp on red cube
seg_red = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=seg_red)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_quat = None
# Extract quaternion from rotation matrix (wxyz)
R = best_T_world[:3, :3]
def rot_to_quat_wxyz(R):
    tr = R[0,0] + R[1,1] + R[2,2]
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

grasp_quat = rot_to_quat_wxyz(R)
print("grasp pos:", grasp_pos, "quat:", grasp_quat)
```

Now let me execute the grasp: approach, close gripper, lift, move above green cube, lower, open.

```python
# Approach the red cube from above
approach_pos = grasp_pos + np.array([0, 0, 0.10])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp pose
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

# Close gripper
close_gripper()

# Lift the red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Now I need to place the red cube on top of the green cube. The green cube's top is at green_center + extent_z/2. The red cube should sit on top of it.

```python
# Compute placement position above green cube
green_top_z = green_obb["center"][2] + green_obb["extent"][2] / 2.0
red_half_z = red_obb["extent"][2] / 2.0
# Target: red cube center at green top + red half height
place_center = np.array([green_obb["center"][0], green_obb["center"][1], green_top_z + red_half_z])

# The grasp offset: gripper holds red cube at grasp_pos, red center is red_obb center
# Offset from grasp point to red center
offset = red_obb["center"] - grasp_pos
# We want red center at place_center, so gripper pos = place_center - offset
place_grasp_pos = place_center - offset
# Add clearance for approach
place_above = place_grasp_pos + np.array([0, 0, 0.10])

joints = solve_ik(place_above, grasp_quat)
move_to_joints(joints)

joints = solve_ik(place_grasp_pos, grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_grasp_pos + np.array([0, 0, 0.15])
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

print("red center:", red_obb["center"], "extent:", red_obb["extent"])
print("green center:", green_obb["center"], "extent:", green_obb["extent"])

# Plan grasp on red cube
seg_red = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=seg_red)
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam

grasp_pos = best_T_world[:3, 3]
R = best_T_world[:3, :3]

def rot_to_quat_wxyz(R):
    tr = R[0,0] + R[1,1] + R[2,2]
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

grasp_quat = rot_to_quat_wxyz(R)
print("grasp pos:", grasp_pos, "quat:", grasp_quat)

# Approach the red cube from above
approach_pos = grasp_pos + np.array([0, 0, 0.10])
joints = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp pose
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

# Close gripper
close_gripper()

# Lift the red cube
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Compute placement position above green cube
green_top_z = green_obb["center"][2] + green_obb["extent"][2] / 2.0
red_half_z = red_obb["extent"][2] / 2.0
place_center = np.array([green_obb["center"][0], green_obb["center"][1], green_top_z + red_half_z])

offset = red_obb["center"] - grasp_pos
place_grasp_pos = place_center - offset
place_above = place_grasp_pos + np.array([0, 0, 0.10])

joints = solve_ik(place_above, grasp_quat)
move_to_joints(joints)

joints = solve_ik(place_grasp_pos, grasp_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)