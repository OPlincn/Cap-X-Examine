```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

def get_best_mask(rgb, prompt):
    res = segment_sam3_text_prompt(rgb, text_prompt=prompt)
    if not res:
        return None
    best = max(res, key=lambda d: d["score"])
    return best["mask"]

red_mask = get_best_mask(rgb, "red cube")
green_mask = get_best_mask(rgb, "green cube")

def mask_to_world_points(mask, depth, K, cam_pose):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    pts_cam = np.stack([x, y, zs], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
    pts_world = (cam_pose @ pts_h.T).T[:, :3]
    return pts_world

red_pts = mask_to_world_points(red_mask, depth, K, cam_pose)
green_pts = mask_to_world_points(green_mask, depth, K, cam_pose)

red_obb = get_oriented_bounding_box_from_3d_points(red_pts)
green_obb = get_oriented_bounding_box_from_3d_points(green_pts)

red_center = red_obb["center"]
green_center = green_obb["center"]
green_extent = green_obb["extent"]
red_extent = red_obb["extent"]

# Grasp red cube from above
grasp_pos = red_center.copy()
grasp_pos[2] = red_center[2] + red_extent[2] * 0.5 + 0.10
quat = np.array([0.0, 1.0, 0.0, 0.0])  # pointing down

open_gripper()
joints = solve_ik(grasp_pos, quat)
move_to_joints(joints)

# Lower to grasp
grasp_pos2 = red_center.copy()
grasp_pos2[2] = red_center[2] + red_extent[2] * 0.5 + 0.005
joints = solve_ik(grasp_pos2, quat)
move_to_joints(joints)
close_gripper()

# Lift
lift_pos = grasp_pos2.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, quat)
move_to_joints(joints)

# Move above green cube
place_pos = green_center.copy()
place_pos[2] = green_center[2] + green_extent[2] * 0.5 + red_extent[2] * 0.5 + 0.10
joints = solve_ik(place_pos, quat)
move_to_joints(joints)

# Lower to place
place_pos2 = green_center.copy()
place_pos2[2] = green_center[2] + green_extent[2] * 0.5 + red_extent[2] * 0.5 + 0.005
joints = solve_ik(place_pos2, quat)
move_to_joints(joints)

open_gripper()

# Retreat
retreat_pos = place_pos2.copy()
retreat_pos[2] += 0.15
joints = solve_ik(retreat_pos, quat)
move_to_joints(joints)
```