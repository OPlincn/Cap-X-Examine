I'll start by getting an observation to understand the scene, then segment the cubes to find their 3D positions.
```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

# Segment red and green cubes
red_masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
green_masks = segment_sam3_text_prompt(rgb, text_prompt="green cube")

def best_mask(masks):
    return max(masks, key=lambda d: d["score"])["mask"]

red_mask = best_mask(red_masks)
green_mask = best_mask(green_masks)

def mask_to_world_points(mask, depth, K, cam_pose):
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
    pts_world = (cam_pose @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_pose)
green_pts = mask_to_world_points(green_mask, depth, K, cam_pose)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
red_extent = red_obb["extent"]
green_extent = green_obb["extent"]

print("red center", red_center, "extent", red_extent)
print("green center", green_center, "extent", green_extent)
```

Now I'll plan a grasp on the red cube, lift it, and place it on top of the green cube.
```python
# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam

grasp_pos = best_T_world[:3, 3]
grasp_R = best_T_world[:3, :3]

# Convert rotation to quaternion (wxyz)
def rot_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

grasp_quat = rot_to_quat(grasp_R)

# Approach from above: pre-grasp position
approach_offset = 0.1
pre_grasp_pos = grasp_pos + np.array([0, 0, approach_offset])

open_gripper()
joints = solve_ik(pre_grasp_pos, grasp_quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints)

close_gripper()

# Lift
lift_pos = grasp_pos + np.array([0, 0, 0.15])
joints = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints)
```

Now I'll move the red cube above the green cube and lower it to place it on top.
```python
# Compute placement position: on top of green cube
# Green cube top surface z = green_center[2] + green_extent[2]/2
# Red cube half-height = red_extent[2]/2
place_z = green_center[2] + green_extent[2]/2 + red_extent[2]/2 + 0.005
place_pos = np.array([green_center[0], green_center[1], place_z])

# Keep same orientation as grasp
# Move above placement position first
above_place = place_pos + np.array([0, 0, 0.1])
joints = solve_ik(above_place, grasp_quat)
move_to_joints(joints)

# Lower to place
joints = solve_ik(place_pos, grasp_quat)
move_to_joints(joints)

# Open gripper
open_gripper()

# Retreat upward
retreat_pos = place_pos + np.array([0, 0, 0.1])
joints = solve_ik(retreat_pos, grasp_quat)
move_to_joints(joints)
```