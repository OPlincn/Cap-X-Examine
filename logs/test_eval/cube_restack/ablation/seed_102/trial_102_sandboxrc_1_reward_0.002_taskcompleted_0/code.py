# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

# Segment the red and green cubes
red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")

red_best = max(red_res, key=lambda d: d["score"])
green_best = max(green_res, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

print("red box", red_best["box"], "score", red_best["score"])
print("green box", green_best["box"], "score", green_best["score"])
```

Now I'll compute 3D positions of both cubes from the depth image to know where they are in the world.

```python
def mask_to_world_points(mask, depth, K, cam_ext):
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

Now I'll plan a grasp on the red cube. Since the red cube is beneath the green cube, I need to grasp it from the side. Let me plan grasps and pick one that approaches from the side.

```python
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
print("num grasps", len(grasp_scores))

# Transform grasps to world frame
grasps_world = np.array([cam_ext @ T for T in grasp_poses])

# Choose a grasp approaching from the side (gripper z-axis roughly horizontal)
best_idx = None
best_score = -1
for i, T in enumerate(grasps_world):
    approach = T[:3, 2]  # gripper z axis
    # prefer horizontal approach (small vertical component)
    if abs(approach[2]) < 0.5:
        if grasp_scores[i] > best_score:
            best_score = grasp_scores[i]
            best_idx = i

if best_idx is None:
    best_idx = int(np.argmax(grasp_scores))

best_T = grasps_world[best_idx]
print("chosen grasp score", grasp_scores[best_idx])
print("approach", best_T[:3, 2])
print("pos", best_T[:3, 3])
```

Now I'll execute the grasp: move to a pre-grasp pose, approach, close gripper, lift.

```python
from scipy.spatial.transform import Rotation as R

def mat_to_quat_wxyz(T):
    rot = R.from_matrix(T[:3, :3])
    q = rot.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])

grasp_pos = best_T[:3, 3]
grasp_quat = mat_to_quat_wxyz(best_T)
approach_dir = best_T[:3, 2]

# Pre-grasp: back off along approach axis
pre_pos = grasp_pos - approach_dir * 0.08
joints = solve_ik(pre_pos, grasp_quat)
move_to_joints(joints)

# Move to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()

# Lift straight up
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Now I'll move the red cube over the green cube and place it on top. The green cube's top is at green_center[2] + green_extent[2]/2. The red cube's center should be placed so its bottom rests on the green top.

```python
# Target: red cube center above green cube center
red_half_height = red_extent[2] / 2.0
green_top = green_center[2] + green_extent[2] / 2.0
target_red_center = np.array([green_center[0], green_center[1], green_top + red_half_height])

# The grasp offset: grasp_pos relative to red_center
offset = grasp_pos - red_center
# Keep same orientation, so target grasp pos = target_red_center + offset
target_grasp_pos = target_red_center + offset

# Move above target first
above_pos = target_grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(above_pos, grasp_quat)
move_to_joints(joints)

# Lower to place
joints = solve_ik(target_grasp_pos, grasp_quat)
move_to_joints(joints)

open_gripper()

# Retreat upward
retreat_pos = target_grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)
```

Let me verify the result with a new observation.

```python
obs2 = get_observation()
rgb2 = obs2["robot0_robotview"]["images"]["rgb"]
red_res2 = segment_sam3_text_prompt(rgb2, text_prompt="red cube")
green_res2 = segment_sam3_text_prompt(rgb2, text_prompt="green cube")
print("red boxes", [r["box"] for r in red_res2])
print("green boxes", [r["box"] for r in green_res2])

# Code block 1
import numpy as np
from scipy.spatial.transform import Rotation as R

def mask_to_world_points(mask, depth, K, cam_ext):
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

def mat_to_quat_wxyz(T):
    rot = R.from_matrix(T[:3, :3])
    q = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_res, key=lambda d: d["score"])
green_best = max(green_res, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# The red cube is currently on the table (bottom), green is on top of it.
# We need to grasp the RED cube and place it on top of the GREEN cube.
# But green is on top of red, so we must first move green aside, or grasp red from the side.
# Simplest: grasp green, lift it, set it down elsewhere; then grasp red, place on green.
# Actually goal: red on top of green. So we need green on the table and red on top.

# Step 1: grasp the green cube (currently on top) and move it to the side.
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
grasps_world = np.array([cam_ext @ T for T in grasp_poses])
best_idx = int(np.argmax(grasp_scores))
best_T = grasps_world[best_idx]
grasp_pos = best_T[:3, 3]
grasp_quat = mat_to_quat_wxyz(best_T)
approach_dir = best_T[:3, 2]

pre_pos = grasp_pos - approach_dir * 0.08
joints = solve_ik(pre_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Move green to a side location (offset in x)
side_target = np.array([green_center[0] + 0.15, green_center[1], green_center[2]])
offset = grasp_pos - green_center
side_grasp_pos = side_target + offset
above_side = side_grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(above_side, grasp_quat)
move_to_joints(joints)
joints = solve_ik(side_grasp_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
retreat = side_grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(retreat, grasp_quat)
move_to_joints(joints)

# Step 2: re-observe to get updated positions
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

red_res = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_res = segment_sam3_text_prompt(rgb, text_prompt="green cube")
red_best = max(red_res, key=lambda d: d["score"])
green_best = max(green_res, key=lambda d: d["score"])
red_mask = red_best["mask"]
green_mask = green_best["mask"]

red_pts = mask_to_world_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_world_points(green_mask, depth, K, cam_ext)
red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

# Step 3: grasp red cube and place on top of green cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
grasps_world = np.array([cam_ext @ T for T in grasp_poses])
best_idx = int(np.argmax(grasp_scores))
best_T = grasps_world[best_idx]
grasp_pos = best_T[:3, 3]
grasp_quat = mat_to_quat_wxyz(best_T)
approach_dir = best_T[:3, 2]

pre_pos = grasp_pos - approach_dir * 0.08
joints = solve_ik(pre_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)
close_gripper()

lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)

# Target: red cube center above green cube center
red_half_height = red_extent[2] / 2.0
green_top = green_center[2] + green_extent[2] / 2.0
target_red_center = np.array([green_center[0], green_center[1], green_top + red_half_height])
offset = grasp_pos - red_center
target_grasp_pos = target_red_center + offset

above_pos = target_grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(above_pos, grasp_quat)
move_to_joints(joints)
joints = solve_ik(target_grasp_pos, grasp_quat)
move_to_joints(joints)
open_gripper()
retreat_pos = target_grasp_pos + np.array([0, 0, 0.12])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)