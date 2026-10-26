I'll start by getting an observation to understand the scene and locate the cubes.

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

def mask_to_world_points(mask, depth, K, cam_ext):
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

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```

Now I'll plan a grasp on the red cube. Since the green cube sits on top of the red cube, I need to grasp the red cube from the side. Let me plan grasps and pick one that approaches horizontally.

```python
# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))

# Transform to world frame
grasp_world = np.array([cam_ext @ T for T in grasp_poses])

# We want a grasp that approaches from the side (gripper z-axis roughly horizontal)
# and is near the red cube center height
best_idx = None
best_score = -1
for i, T in enumerate(grasp_world):
    approach = T[:3, 2]  # z axis of grasp frame
    # prefer horizontal approach (small |z| component)
    horiz = 1.0 - abs(approach[2])
    pos = T[:3, 3]
    # prefer grasp near red cube center
    dist = np.linalg.norm(pos - red_center)
    score = grasp_scores[i] * horiz - 0.1 * dist
    if score > best_score:
        best_score = score
        best_idx = i

best_T = grasp_world[best_idx]
print("best grasp pos", best_T[:3, 3], "approach", best_T[:3, 2])
```

Now I'll execute the grasp: move to a pre-grasp pose, descend, close gripper, lift.

```python
from scipy.spatial.transform import Rotation as R

def mat_to_quat_wxyz(T):
    rot = R.from_matrix(T[:3, :3])
    q = rot.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])

grasp_pos = best_T[:3, 3]
grasp_quat = mat_to_quat_wxyz(best_T)

# Pre-grasp: back off along approach axis
approach = best_T[:3, 2]
pre_pos = grasp_pos - approach * 0.10

open_gripper()
joints = solve_ik(pre_pos, grasp_quat)
move_to_joints(joints)

# Move to grasp pose
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()

# Lift straight up
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Now I need to place the red cube on top of the green cube. The green cube is currently on top of the red cube, so I need to move the red cube to above the green cube's position. But wait — the green cube is on top of the red cube. If I lift the red cube, the green cube will fall. Let me reconsider.

Actually, the task is to swap: place red on top of green. Since green is on red, I should first move the red cube away, then... but the green will drop to the table. Let me re-observe to see the current state.

```python
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
green_best = max(green_masks, key=lambda d: d["score"])
green_mask = green_best["mask"]
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("green center now", green_center, "extent", green_extent)
```

Now I'll place the red cube on top of the green cube. The target position is above the green cube center, at height green_top + red_half_height.

```python
# Target: red cube center should be at green_center + (0,0, green_half + red_half)
red_half = red_extent[2] / 2.0
green_half = green_extent[2] / 2.0
target_center = green_center + np.array([0, 0, green_half + red_half + 0.005])

# Current grasp holds red cube; the grasp offset from red center to grasp pos
# We keep same orientation. Compute offset between grasp pos and red center at grasp time.
# Approximate: red cube center relative to grasp pos
# We'll place so that red center goes to target_center
# grasp_pos currently at lifted location; offset = grasp_pos - red_center (from earlier)
offset = grasp_pos - red_center  # in world frame, roughly
place_pos = target_center + offset

# Approach from above for placement
place_quat = grasp_quat

# Move above target
above_pos = place_pos + np.array([0, 0, 0.10])
joints = solve_ik(above_pos, place_quat)
move_to_joints(joints)

# Descend to place
joints = solve_ik(place_pos, place_quat)
move_to_joints(joints)

# Open gripper
open_gripper()

# Retreat upward
retreat_pos = place_pos + np.array([0, 0, 0.12])
joints = solve_ik(retreat_pos, place_quat)
move_to_joints(joints)
```

Let me verify the result with a new observation.

```python
obs = get_observation()
rgb = obs["robot0_robotview"]["images"]["rgb"]
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")
print("red found:", len(red_masks) > 0, "green found:", len(green_masks) > 0)
```