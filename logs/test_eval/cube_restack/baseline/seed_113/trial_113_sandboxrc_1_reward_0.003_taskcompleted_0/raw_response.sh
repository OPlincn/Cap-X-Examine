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

print("red box", red_best["box"], "score", red_best["score"])
print("green box", green_best["box"], "score", green_best["score"])
```

```python
def mask_to_points(mask, depth, K, cam_ext):
    ys, xs = np.where(mask)
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

```python
# The green cube sits on top of the red cube. We need to remove the green cube first,
# then move the red cube on top of the green cube's location.

# Determine cube size (use max horizontal extent)
cube_size = float(np.max(red_extent))
print("cube size", cube_size)

# Grasp orientation: top-down grasp (gripper pointing down)
# quaternion for gripper pointing down (rotate 180 about x): w=0, x=1, y=0, z=0
down_quat = np.array([0.0, 1.0, 0.0, 0.0])

# Step 1: Grasp the green cube (top cube) and lift it away
green_grasp_pos = green_center.copy()
green_grasp_pos[2] = green_center[2]  # center height

# Approach from above
approach_z = green_center[2] + 0.10
pre_grasp = np.array([green_center[0], green_center[1], approach_z])
joints = solve_ik(pre_grasp, down_quat)
move_to_joints(joints)
open_gripper()

# Move down to grasp
grasp_pos = np.array([green_center[0], green_center[1], green_center[2]])
joints = solve_ik(grasp_pos, down_quat)
move_to_joints(joints)
close_gripper()

# Lift green cube
lift_pos = np.array([green_center[0], green_center[1], green_center[2] + 0.15])
joints = solve_ik(lift_pos, down_quat)
move_to_joints(joints)
```

```python
# Place green cube to the side (away from red cube) on the table
# Move to a free spot beside the stack
side_pos = np.array([green_center[0] + 0.15, green_center[1], green_center[2] + 0.15])
joints = solve_ik(side_pos, down_quat)
move_to_joints(joints)

# Lower to table
place_green = np.array([green_center[0] + 0.15, green_center[1], green_center[2] - 0.02])
joints = solve_ik(place_green, down_quat)
move_to_joints(joints)
open_gripper()

# Lift up
joints = solve_ik(side_pos, down_quat)
move_to_joints(joints)
```

```python
# Step 2: Grasp the red cube and place it on top of the green cube
red_approach = np.array([red_center[0], red_center[1], red_center[2] + 0.10])
joints = solve_ik(red_approach, down_quat)
move_to_joints(joints)
open_gripper()

red_grasp = np.array([red_center[0], red_center[1], red_center[2]])
joints = solve_ik(red_grasp, down_quat)
move_to_joints(joints)
close_gripper()

# Lift red cube
red_lift = np.array([red_center[0], red_center[1], red_center[2] + 0.15])
joints = solve_ik(red_lift, down_quat)
move_to_joints(joints)
```

```python
# Move red cube above the green cube's new location
green_new_center = np.array([green_center[0] + 0.15, green_center[1], green_center[2]])
above_green = np.array([green_new_center[0], green_new_center[1], green_new_center[2] + cube_size + 0.10])
joints = solve_ik(above_green, down_quat)
move_to_joints(joints)

# Lower red cube onto green cube
place_red = np.array([green_new_center[0], green_new_center[1], green_new_center[2] + cube_size + 0.01])
joints = solve_ik(place_red, down_quat)
move_to_joints(joints)

# Open gripper to release
open_gripper()

# Lift up
joints = solve_ik(above_green, down_quat)
move_to_joints(joints)
```