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

# Compute 3D centers of each cube from depth
def mask_to_points(mask, depth, K):
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    pts_cam = np.stack([X, Y, z], axis=1)
    # transform to world
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_points(red_mask, depth, K)
green_pts = mask_to_points(green_mask, depth, K)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```

Now I'll plan a grasp on the red cube. Since the green cube sits on top of the red cube, I need to grasp the red cube from the side (horizontal approach) to avoid colliding with the green cube on top.

```python
# Plan grasp on red cube
seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=seg)

# Transform grasps to world frame
grasps_world = np.array([cam_ext @ T for T in grasp_poses])
scores = grasp_scores

# Prefer grasps that approach horizontally (gripper z-axis roughly horizontal)
# and are near the red cube center height
best_idx = None
best_val = -1e9
for i, T in enumerate(grasps_world):
    approach = T[:3, 2]  # gripper z axis
    # want approach roughly horizontal (small |z| component)
    horiz = 1.0 - abs(approach[2])
    pos = T[:3, 3]
    # prefer grasp near red cube center height (avoid top where green is)
    height_pen = abs(pos[2] - red_center[2])
    val = scores[i] + 2.0 * horiz - 1.0 * height_pen
    if val > best_val:
        best_val = val
        best_idx = i

best_T = grasps_world[best_idx]
grasp_pos = best_T[:3, 3]
print("grasp pos", grasp_pos)
print("approach", best_T[:3, 2])
```

Now I'll execute the grasp: open gripper, move to a pre-grasp pose, move to grasp, close, lift.

```python
from scipy.spatial.transform import Rotation as R

# Build target orientation from grasp rotation matrix
grasp_R = best_T[:3, :3]
quat_xyzw = R.from_matrix(grasp_R).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

approach = best_T[:3, 2]
pre_grasp_pos = grasp_pos - approach * 0.08

open_gripper()

# Move to pre-grasp
joints = solve_ik(pre_grasp_pos, quat_wxyz)
move_to_joints(joints)

# Move to grasp
joints = solve_ik(grasp_pos, quat_wxyz)
move_to_joints(joints)

close_gripper()

# Lift straight up
lift_pos = grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(lift_pos, quat_wxyz)
move_to_joints(joints)
```

Now I need to place the red cube on top of the green cube. The green cube is currently on top of the red cube's original position. After lifting the red cube, the green cube should be resting on the table. I'll place the red cube on top of the green cube.

```python
# Re-observe to find green cube's current position
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_points(green_mask, depth, K)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("green center now", green_center, "extent", green_extent)

# Place red cube on top of green cube
# Target: red cube center should be above green cube center
place_pos = green_center + np.array([0, 0, green_extent[2]/2 + red_extent[2]/2 + 0.005])

# Keep the same orientation as grasp
# Move above place position first
above_place = place_pos + np.array([0, 0, 0.10])
joints = solve_ik(above_place, quat_wxyz)
move_to_joints(joints)

# Move down to place
joints = solve_ik(place_pos, quat_wxyz)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Retreat upward
retreat_pos = place_pos + np.array([0, 0, 0.12])
joints = solve_ik(retreat_pos, quat_wxyz)
move_to_joints(joints)

# Code block 1
import numpy as np
from scipy.spatial.transform import Rotation as R

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    pts_cam = np.stack([X, Y, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

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

# Plan grasp on red cube
seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=seg)
grasps_world = np.array([cam_ext @ T for T in grasp_poses])

best_idx = None
best_val = -1e9
for i, T in enumerate(grasps_world):
    approach = T[:3, 2]
    horiz = 1.0 - abs(approach[2])
    pos = T[:3, 3]
    height_pen = abs(pos[2] - red_center[2])
    val = grasp_scores[i] + 2.0 * horiz - 1.0 * height_pen
    if val > best_val:
        best_val = val
        best_idx = i

best_T = grasps_world[best_idx]
grasp_pos = best_T[:3, 3]
grasp_R = best_T[:3, :3]
quat_xyzw = R.from_matrix(grasp_R).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
approach = best_T[:3, 2]

open_gripper()
pre_grasp_pos = grasp_pos - approach * 0.08
joints = solve_ik(pre_grasp_pos, quat_wxyz)
move_to_joints(joints)
joints = solve_ik(grasp_pos, quat_wxyz)
move_to_joints(joints)
close_gripper()
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, quat_wxyz)
move_to_joints(joints)

# Re-observe to find green cube's current position
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]

# Place red cube on top of green cube
place_pos = green_center + np.array([0, 0, green_extent[2]/2 + red_extent[2]/2 + 0.01])
above_place = place_pos + np.array([0, 0, 0.12])
joints = solve_ik(above_place, quat_wxyz)
move_to_joints(joints)
joints = solve_ik(place_pos, quat_wxyz)
move_to_joints(joints)
open_gripper()
retreat_pos = place_pos + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, quat_wxyz)
move_to_joints(joints)

# Code block 2
import numpy as np
from scipy.spatial.transform import Rotation as R

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    pts_cam = np.stack([X, Y, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

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

# Plan grasp on red cube
seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=seg)
grasps_world = np.array([cam_ext @ T for T in grasp_poses])

best_idx = None
best_val = -1e9
for i, T in enumerate(grasps_world):
    approach = T[:3, 2]
    horiz = 1.0 - abs(approach[2])
    pos = T[:3, 3]
    height_pen = abs(pos[2] - red_center[2])
    val = grasp_scores[i] + 2.0 * horiz - 1.0 * height_pen
    if val > best_val:
        best_val = val
        best_idx = i

best_T = grasps_world[best_idx]
grasp_pos = best_T[:3, 3]
grasp_R = best_T[:3, :3]
quat_xyzw = R.from_matrix(grasp_R).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
approach = best_T[:3, 2]
print("grasp pos", grasp_pos, "approach", approach)

open_gripper()
pre_grasp_pos = grasp_pos - approach * 0.08
joints = solve_ik(pre_grasp_pos, quat_wxyz)
move_to_joints(joints)
joints = solve_ik(grasp_pos, quat_wxyz)
move_to_joints(joints)
close_gripper()
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, quat_wxyz)
move_to_joints(joints)

# Re-observe to find green cube's current position
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("green center now", green_center, "extent", green_extent)

# Place red cube on top of green cube
place_pos = green_center + np.array([0, 0, green_extent[2]/2 + red_extent[2]/2 + 0.01])
above_place = place_pos + np.array([0, 0, 0.12])
joints = solve_ik(above_place, quat_wxyz)
move_to_joints(joints)
joints = solve_ik(place_pos, quat_wxyz)
move_to_joints(joints)
open_gripper()
retreat_pos = place_pos + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, quat_wxyz)
move_to_joints(joints)

# Code block 3
import numpy as np
from scipy.spatial.transform import Rotation as R

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    pts_cam = np.stack([X, Y, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_h.T).T[:, :3]
    return pts_world

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

# Plan grasp on red cube
seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=seg)
grasps_world = np.array([cam_ext @ T for T in grasp_poses])

best_idx = None
best_val = -1e9
for i, T in enumerate(grasps_world):
    approach = T[:3, 2]
    horiz = 1.0 - abs(approach[2])
    pos = T[:3, 3]
    height_pen = abs(pos[2] - red_center[2])
    val = grasp_scores[i] + 2.0 * horiz - 1.0 * height_pen
    if val > best_val:
        best_val = val
        best_idx = i

best_T = grasps_world[best_idx]
grasp_pos = best_T[:3, 3]
grasp_R = best_T[:3, :3]
quat_xyzw = R.from_matrix(grasp_R).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
approach = best_T[:3, 2]
print("grasp pos", grasp_pos, "approach", approach)

open_gripper()
pre_grasp_pos = grasp_pos - approach * 0.10
joints = solve_ik(pre_grasp_pos, quat_wxyz)
move_to_joints(joints)
joints = solve_ik(grasp_pos, quat_wxyz)
move_to_joints(joints)
close_gripper()

# Lift straight up
lift_pos = grasp_pos + np.array([0, 0, 0.20])
joints = solve_ik(lift_pos, quat_wxyz)
move_to_joints(joints)

# Verify grasp by re-observing
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
if len(red_masks) > 0:
    red_best = max(red_masks, key=lambda d: d["score"])
    red_mask = red_best["mask"]
    red_pts = mask_to_points(red_mask, depth, K, cam_ext)
    if red_pts.shape[0] > 10:
        red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
        red_center_now = red_obb["center"]
        print("red center after lift", red_center_now)

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("green center now", green_center, "extent", green_extent)

# Place red cube on top of green cube
place_pos = green_center + np.array([0, 0, green_extent[2]/2 + red_extent[2]/2 + 0.01])
above_place = place_pos + np.array([0, 0, 0.15])
joints = solve_ik(above_place, quat_wxyz)
move_to_joints(joints)
joints = solve_ik(place_pos, quat_wxyz)
move_to_joints(joints)
open_gripper()
retreat_pos = place_pos + np.array([0, 0, 0.20])
joints = solve_ik(retreat_pos, quat_wxyz)
move_to_joints(joints)