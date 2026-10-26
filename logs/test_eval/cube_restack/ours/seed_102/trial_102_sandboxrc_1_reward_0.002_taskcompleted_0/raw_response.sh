I'll start by getting an observation to understand the scene.

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

print("red box", red_best["box"], "score", red_best["score"])
print("green box", green_best["box"], "score", green_best["score"])
```

Now I'll compute the 3D positions of both cubes to know where to place the red cube.

```python
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

The green cube is on top of the red cube. I need to grasp the red cube, lift it, and place it on top of the green cube. Let me plan a grasp on the red cube.

```python
# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
print("best grasp score", grasp_scores[best_idx])
print("grasp pos world", best_T_world[:3, 3])
```

Let me approach the red cube from above. Since the green cube sits on top of the red cube, I need to grasp the red cube from the side or lift the green first. Actually, the goal is to place the red cube on top of the green cube. But green is currently on top of red. So I need to first move the green cube off, or grasp the red cube and place it on the green.

Wait - the goal is red on top of green. Currently green is on top of red. So I need to swap them. The simplest: grasp the red cube (which is at the bottom), lift it, and place it on top of the green cube. But the green cube is on top of the red cube, so grasping the red cube directly is hard.

Better approach: grasp the green cube, move it aside, then grasp the red cube and place it where green was... but then green needs to be on top. Actually goal is red on top of green. So: move green aside, place red on table, then place green, then put red on green. That's complex.

Simplest: grasp the green cube, lift it, move it aside to the table. Then grasp the red cube, lift it, and place it on top of the green cube.

Let me first grasp the green cube and move it aside.

```python
# Grasp green cube and move it aside
green_grasp_poses, green_grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=green_mask.astype(np.int32))
g_idx = green_grasp_scores.argmax()
g_T_cam = green_grasp_poses[g_idx]
g_T_world = cam_ext @ g_T_cam
grasp_pos = g_T_world[:3, 3]
grasp_quat = None
# Convert rotation matrix to quaternion (wxyz)
R = g_T_world[:3, :3]
# use standard conversion
def rot_to_quat(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        S = np.sqrt(tr+1.0)*2
        w = 0.25*S
        x = (R[2,1]-R[1,2])/S
        y = (R[0,2]-R[2,0])/S
        z = (R[1,0]-R[0,1])/S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        w = (R[2,1]-R[1,2])/S
        x = 0.25*S
        y = (R[0,1]+R[1,0])/S
        z = (R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        w = (R[0,2]-R[2,0])/S
        x = (R[0,1]+R[1,0])/S
        y = 0.25*S
        z = (R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        w = (R[1,0]-R[0,1])/S
        x = (R[0,2]+R[2,0])/S
        y = (R[1,2]+R[2,1])/S
        z = 0.25*S
    return np.array([w,x,y,z])

grasp_quat = rot_to_quat(R)
print("green grasp pos", grasp_pos, "quat", grasp_quat)
```

Let me use a top-down grasp approach for reliability. I'll approach from above with gripper pointing down.

```python
# Top-down grasp orientation: gripper pointing down (z-axis down)
# quaternion for rotation of 180 deg about x: w=0, x=1, y=0, z=0
down_quat = np.array([0.0, 1.0, 0.0, 0.0])

# Approach green cube from above
green_top = green_center.copy()
green_top[2] = green_center[2] + green_extent[2]/2 + 0.10
pre_joints = solve_ik(green_top, down_quat)
move_to_joints(pre_joints)
open_gripper()

# Move down to grasp
grasp_target = green_center.copy()
grasp_target[2] = green_center[2] + green_extent[2]/2 - 0.005
grasp_joints = solve_ik(grasp_target, down_quat)
move_to_joints(grasp_joints)
close_gripper()

# Lift green cube
lift_target = green_center.copy()
lift_target[2] = green_center[2] + 0.15
lift_joints = solve_ik(lift_target, down_quat)
move_to_joints(lift_joints)
```

Now move green cube aside to a free spot on the table.

```python
# Move green aside - to the left of the cubes
aside_target = green_center.copy()
aside_target[0] = green_center[0] - 0.15
aside_target[2] = green_center[2] + 0.15
aside_joints = solve_ik(aside_target, down_quat)
move_to_joints(aside_joints)

# Lower green to table
place_green = aside_target.copy()
place_green[2] = green_center[2] - green_extent[2]/2 + 0.02
place_green_joints = solve_ik(place_green, down_quat)
move_to_joints(place_green_joints)
open_gripper()

# Lift up
up_joints = solve_ik(aside_target, down_quat)
move_to_joints(up_joints)
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
red_extent = red_obb["extent"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
print("red center", red_center, "green center", green_center)
```

Now grasp red cube from above and place on green.

```python
# Approach red cube from above
red_top = red_center.copy()
red_top[2] = red_center[2] + red_extent[2]/2 + 0.10
pre_joints = solve_ik(red_top, down_quat)
move_to_joints(pre_joints)
open_gripper()

grasp_target = red_center.copy()
grasp_target[2] = red_center[2] + red_extent[2]/2 - 0.005
grasp_joints = solve_ik(grasp_target, down_quat)
move_to_joints(grasp_joints)
close_gripper()

# Lift red cube
lift_target = red_center.copy()
lift_target[2] = red_center[2] + 0.15
lift_joints = solve_ik(lift_target, down_quat)
move_to_joints(lift_joints)

# Move above green cube
above_green = green_center.copy()
above_green[2] = green_center[2] + green_extent[2]/2 + red_extent[2]/2 + 0.10
above_green_joints = solve_ik(above_green, down_quat)
move_to_joints(above_green_joints)

# Lower red cube onto green cube
place_target = green_center.copy()
place_target[2] = green_center[2] + green_extent[2]/2 + red_extent[2]/2 + 0.005
place_joints = solve_ik(place_target, down_quat)
move_to_joints(place_joints)

# Open gripper to release
open_gripper()

# Lift up
final_lift = place_target.copy()
final_lift[2] = place_target[2] + 0.15
final_lift_joints = solve_ik(final_lift, down_quat)
move_to_joints(final_lift_joints)
```

The task should now be complete: red cube placed on top of green cube with gripper open.