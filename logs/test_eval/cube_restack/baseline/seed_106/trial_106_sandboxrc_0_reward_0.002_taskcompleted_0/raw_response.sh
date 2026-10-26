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

Now I need to figure out the 3D positions of the cubes. The green cube is on top of the red cube. The goal is to place the red cube on top of the green cube. So I need to: pick up the red cube, move it above the green cube, and place it there.

Let me compute 3D positions from the masks.

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
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_ext @ pts_cam_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_points(red_mask, depth, K, cam_ext)
green_pts = mask_to_points(green_mask, depth, K, cam_ext)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
print("red center", red_center, "extent", red_obb["extent"])
print("green center", green_center, "extent", green_obb["extent"])
```

Now let me plan a grasp on the red cube.

```python
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_ext @ best_T_cam
print("grasp world pos", best_T_world[:3, 3])
print("grasp score", grasp_scores[best_idx])
```

Now I'll execute the pick and place. First move to a pre-grasp pose above the red cube, then grasp.

```python
from scipy.spatial.transform import Rotation as R

grasp_pos = best_T_world[:3, 3]
grasp_rot = best_T_world[:3, :3]
quat = R.from_matrix(grasp_rot).as_quat()  # xyzw
quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])

# Pre-grasp: approach from above
pre_grasp_pos = grasp_pos + np.array([0, 0, 0.1])
joints = solve_ik(pre_grasp_pos, quat_wxyz)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, quat_wxyz)
move_to_joints(joints)

close_gripper()

# Lift
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, quat_wxyz)
move_to_joints(joints)
```

Now place the red cube on top of the green cube. The green cube top is at green_center + half extent in z. The red cube should sit on top.

```python
green_top_z = green_center[2] + green_obb["extent"][2] / 2.0
red_half_z = red_obb["extent"][2] / 2.0
place_z = green_top_z + red_half_z + 0.005

place_pos = np.array([green_center[0], green_center[1], place_z + 0.1])
joints = solve_ik(place_pos, quat_wxyz)
move_to_joints(joints)

# Lower to place
place_pos_low = np.array([green_center[0], green_center[1], place_z])
joints = solve_ik(place_pos_low, quat_wxyz)
move_to_joints(joints)

open_gripper()

# Retreat upward
retreat_pos = place_pos_low + np.array([0, 0, 0.15])
joints = solve_ik(retreat_pos, quat_wxyz)
move_to_joints(joints)
```

Let me verify the result with a new observation.

```python
obs2 = get_observation()
rgb2 = obs2["robot0_robotview"]["images"]["rgb"]
red_masks2 = segment_sam3_text_prompt(rgb2, text_prompt="red cube")
green_masks2 = segment_sam3_text_prompt(rgb2, text_prompt="green cube")
print("red found:", len(red_masks2) > 0, "green found:", len(green_masks2) > 0)
```