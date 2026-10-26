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

print("red box", red_best["box"], "score", red_best["score"])
print("green box", green_best["box"], "score", green_best["score"])
```

Now I need to figure out the geometry. The green cube is on top of the red cube. I need to grasp the red cube (bottom), lift it, and place it on top of the green cube. But the green cube is sitting on the red cube, so I can't grasp the red cube directly from above.

Let me think about the strategy: I should first move the green cube out of the way, then grasp the red cube, place it, then put the green cube back on top. Actually the goal is red on top of green. So:
1. Move green cube aside
2. Grasp red cube, place it where green was (or just place red on table)
3. Place green cube, then red on top

Actually simpler: grasp green cube, lift it, set it aside. Then grasp red cube, place it on the table. Then place green cube, then red on top of green.

Let me get 3D positions first.

```python
def mask_to_3d(mask):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    xc = (xs - cx) * zs / fx
    yc = (ys - cy) * zs / fy
    pts_cam = np.stack([xc, yc, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_3d(red_mask)
green_pts = mask_to_3d(green_mask)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)
print("red center", red_obb["center"], "extent", red_obb["extent"])
print("green center", green_obb["center"], "extent", green_obb["extent"])
```

Let me plan the full sequence. I'll write the complete manipulation code.

```python
red_center = red_obb["center"]
green_center = green_obb["center"]
red_ext = red_obb["extent"]
green_ext = green_obb["extent"]

# Cube half-heights (z extent / 2)
red_h = red_ext[2] / 2.0
green_h = green_ext[2] / 2.0

# Table height approx = red bottom
table_z = red_center[2] - red_h

# Grasp orientation: top-down, gripper pointing down
# quaternion wxyz for rotation 180 about x (gripper down)
quat_down = np.array([0.0, 1.0, 0.0, 0.0])

# Step 1: Move green cube aside
# Grasp green from above
green_grasp_z = green_center[2] + green_h + 0.02
green_grasp_pos = np.array([green_center[0], green_center[1], green_grasp_z])
open_gripper()
joints = solve_ik(green_grasp_pos, quat_down)
move_to_joints(joints)

# lower to grasp
green_grasp_pos2 = np.array([green_center[0], green_center[1], green_center[2]])
joints = solve_ik(green_grasp_pos2, quat_down)
move_to_joints(joints)
close_gripper()

# lift green
green_lift = np.array([green_center[0], green_center[1], green_center[2] + 0.15])
joints = solve_ik(green_lift, quat_down)
move_to_joints(joints)

# move green aside (to the side)
green_aside = np.array([green_center[0] + 0.15, green_center[1], green_center[2] + 0.15])
joints = solve_ik(green_aside, quat_down)
move_to_joints(joints)

# lower green to table
green_place = np.array([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.005])
joints = solve_ik(green_place, quat_down)
move_to_joints(joints)
open_gripper()

# lift up
joints = solve_ik(green_aside, quat_down)
move_to_joints(joints)
```

Now grasp the red cube and place it on top of the green cube (which is now aside). Wait — the goal is red on top of green. So I should place green back at its original spot, then place red on top. Let me reconsider.

Actually, let me place green back at original location first, then place red on top of green. But red is currently at original location. Let me instead:
- Move green aside
- Grasp red, move it aside too (or lift it)
- Put green back at original spot
- Place red on top of green

Let me redo more carefully. After moving green aside, red is at original spot. I'll grasp red, lift it, move green back to original spot, then place red on top of green.

```python
# Step 2: Grasp red cube (now exposed at original location)
red_grasp_z = red_center[2] + red_h + 0.02
red_grasp_pos = np.array([red_center[0], red_center[1], red_grasp_z])
joints = solve_ik(red_grasp_pos, quat_down)
move_to_joints(joints)

red_grasp_pos2 = np.array([red_center[0], red_center[1], red_center[2]])
joints = solve_ik(red_grasp_pos2, quat_down)
move_to_joints(joints)
close_gripper()

# lift red
red_lift = np.array([red_center[0], red_center[1], red_center[2] + 0.2])
joints = solve_ik(red_lift, quat_down)
move_to_joints(joints)
```

Now move green back to original spot. Green is currently aside at green_center + 0.15. I need to grasp it again and move it back.

```python
# Step 3: Move green back to original location
green_aside_pos = np.array([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.005])
# approach green from above
green_approach = np.array([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.15])
joints = solve_ik(green_approach, quat_down)
move_to_joints(joints)
joints = solve_ik(green_aside_pos, quat_down)
move_to_joints(joints)
close_gripper()

# lift green
joints = solve_ik(green_approach, quat_down)
move_to_joints(joints)

# move to original location
green_orig_above = np.array([green_center[0], green_center[1], table_z + green_h + 0.15])
joints = solve_ik(green_orig_above, quat_down)
move_to_joints(joints)

# lower green to table
green_orig_place = np.array([green_center[0], green_center[1], table_z + green_h + 0.005])
joints = solve_ik(green_orig_place, quat_down)
move_to_joints(joints)
open_gripper()

# lift up
joints = solve_ik(green_orig_above, quat_down)
move_to_joints(joints)
```

Now place red on top of green. Red is currently held at red_lift position. Green is at original location with top at table_z + 2*green_h.

```python
# Step 4: Place red on top of green
red_place_z = table_z + 2*green_h + red_h + 0.005
red_place_above = np.array([green_center[0], green_center[1], red_place_z + 0.15])
joints = solve_ik(red_place_above, quat_down)
move_to_joints(joints)

red_place = np.array([green_center[0], green_center[1], red_place_z])
joints = solve_ik(red_place, quat_down)
move_to_joints(joints)

# open gripper to release
open_gripper()

# lift up
joints = solve_ik(red_place_above, quat_down)
move_to_joints(joints)
```

Let me combine all of this into one executable block.

```python
import numpy as np

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

def mask_to_3d(mask):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    xc = (xs - cx) * zs / fx
    yc = (ys - cy) * zs / fy
    pts_cam = np.stack([xc, yc, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
    return (cam_ext @ pts_cam_h.T).T[:, :3]

red_obb = get_oriented_bounding_box_from_3d_points(mask_to_3d(red_mask))
green_obb = get_oriented_bounding_box_from_3d_points(mask_to_3d(green_mask))

red_center = red_obb["center"]
green_center = green_obb["center"]
red_h = red_obb["extent"][2] / 2.0
green_h = green_obb["extent"][2] / 2.0
table_z = red_center[2] - red_h

quat_down = np.array([0.0, 1.0, 0.0, 0.0])

def move_to(pos):
    joints = solve_ik(np.array(pos, dtype=np.float64), quat_down)
    move_to_joints(joints)

# --- Move green aside ---
open_gripper()
move_to([green_center[0], green_center[1], green_center[2] + green_h + 0.02])
move_to([green_center[0], green_center[1], green_center[2]])
close_gripper()
move_to([green_center[0], green_center[1], green_center[2] + 0.15])
move_to([green_center[0] + 0.15, green_center[1], green_center[2] + 0.15])
move_to([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.005])
open_gripper()
move_to([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.15])

# --- Grasp red and lift ---
move_to([red_center[0], red_center[1], red_center[2] + red_h + 0.02])
move_to([red_center[0], red_center[1], red_center[2]])
close_gripper()
move_to([red_center[0], red_center[1], red_center[2] + 0.2])

# --- Move green back to original location ---
move_to([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.15])
move_to([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.005])
close_gripper()
move_to([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.15])
move_to([green_center[0], green_center[1], table_z + green_h + 0.15])
move_to([green_center[0], green_center[1], table_z + green_h + 0.005])
open_gripper()
move_to([green_center[0], green_center[1], table_z + green_h + 0.15])

# --- Place red on top of green ---
red_place_z = table_z + 2*green_h + red_h + 0.005
move_to([green_center[0], green_center[1], red_place_z + 0.15])
move_to([green_center[0], green_center[1], red_place_z])
open_gripper()
move_to([green_center[0], green_center[1], red_place_z + 0.15])

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
red_mask = max(red_masks, key=lambda d: d["score"])["mask"]
green_mask = max(green_masks, key=lambda d: d["score"])["mask"]

def mask_to_3d(mask):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    xc = (xs - cx) * zs / fx
    yc = (ys - cy) * zs / fy
    pts_cam = np.stack([xc, yc, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
    return (cam_ext @ pts_cam_h.T).T[:, :3]

red_obb = get_oriented_bounding_box_from_3d_points(mask_to_3d(red_mask))
green_obb = get_oriented_bounding_box_from_3d_points(mask_to_3d(green_mask))

red_center = red_obb["center"]
green_center = green_obb["center"]
red_h = red_obb["extent"][2] / 2.0
green_h = green_obb["extent"][2] / 2.0
table_z = red_center[2] - red_h

quat_down = np.array([0.0, 1.0, 0.0, 0.0])

def move_to(pos):
    joints = solve_ik(np.array(pos, dtype=np.float64), quat_down)
    move_to_joints(joints)

# --- Move green aside ---
open_gripper()
move_to([green_center[0], green_center[1], green_center[2] + green_h + 0.02])
move_to([green_center[0], green_center[1], green_center[2]])
close_gripper()
move_to([green_center[0], green_center[1], green_center[2] + 0.15])
move_to([green_center[0] + 0.15, green_center[1], green_center[2] + 0.15])
move_to([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.005])
open_gripper()
move_to([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.15])

# --- Grasp red and lift ---
move_to([red_center[0], red_center[1], red_center[2] + red_h + 0.02])
move_to([red_center[0], red_center[1], red_center[2]])
close_gripper()
move_to([red_center[0], red_center[1], red_center[2] + 0.2])

# --- Move green back to original location ---
move_to([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.15])
move_to([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.005])
close_gripper()
move_to([green_center[0] + 0.15, green_center[1], table_z + green_h + 0.15])
move_to([green_center[0], green_center[1], table_z + green_h + 0.15])
move_to([green_center[0], green_center[1], table_z + green_h + 0.005])
open_gripper()
move_to([green_center[0], green_center[1], table_z + green_h + 0.15])

# --- Place red on top of green ---
red_place_z = table_z + 2*green_h + red_h + 0.005
move_to([green_center[0], green_center[1], red_place_z + 0.15])
move_to([green_center[0], green_center[1], red_place_z])
open_gripper()
move_to([green_center[0], green_center[1], red_place_z + 0.15])

# Code block 2
import numpy as np

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

def mask_to_3d(mask):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    xc = (xs - cx) * zs / fx
    yc = (ys - cy) * zs / fy
    pts_cam = np.stack([xc, yc, zs], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
    return (cam_ext @ pts_cam_h.T).T[:, :3]

red_obb = get_oriented_bounding_box_from_3d_points(mask_to_3d(red_mask))
green_obb = get_oriented_bounding_box_from_3d_points(mask_to_3d(green_mask))

red_center = red_obb["center"]
green_center = green_obb["center"]
red_h = red_obb["extent"][2] / 2.0
green_h = green_obb["extent"][2] / 2.0

# Green is on the table now; table height from green bottom
table_z = green_center[2] - green_h

quat_down = np.array([0.0, 1.0, 0.0, 0.0])

def move_to(pos):
    joints = solve_ik(np.array(pos, dtype=np.float64), quat_down)
    move_to_joints(joints)

# Red is currently held by gripper (open per report, so it may have dropped).
# Re-grasp red wherever it is, then place it on top of green.
open_gripper()
move_to([red_center[0], red_center[1], red_center[2] + red_h + 0.05])
move_to([red_center[0], red_center[1], red_center[2]])
close_gripper()
move_to([red_center[0], red_center[1], red_center[2] + 0.2])

# Place red on top of green
red_place_z = table_z + 2*green_h + red_h + 0.005
move_to([green_center[0], green_center[1], red_place_z + 0.15])
move_to([green_center[0], green_center[1], red_place_z])
open_gripper()
move_to([green_center[0], green_center[1], red_place_z + 0.15])