# Code block 0
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

# Code block 1
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

print("red_center", red_center, "red_extent", red_extent)
print("green_center", green_center, "green_extent", green_extent)

# Use grasp planner on red cube to get a valid grasp pose
red_seg = red_mask.astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_seg)
best_idx = int(np.argmax(grasp_scores))
best_T_cam = grasp_poses[best_idx]
best_T_world = cam_pose @ best_T_cam
grasp_pos = best_T_world[:3, 3].copy()

# Extract orientation quaternion from rotation matrix
R = best_T_world[:3, :3]
def rot_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
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
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

quat = rot_to_quat_wxyz(R)

# Approach from above the grasp
open_gripper()
approach_pos = grasp_pos.copy()
approach_pos[2] += 0.12
joints = solve_ik(approach_pos, quat)
move_to_joints(joints)

# Lower to grasp
joints = solve_ik(grasp_pos, quat)
move_to_joints(joints)
close_gripper()

# Lift
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.20
joints = solve_ik(lift_pos, quat)
move_to_joints(joints)

# Move above green cube
place_pos = green_center.copy()
place_pos[2] = green_center[2] + green_extent[2] * 0.5 + red_extent[2] * 0.5 + 0.12
joints = solve_ik(place_pos, quat)
move_to_joints(joints)

# Lower to place
place_pos2 = green_center.copy()
place_pos2[2] = green_center[2] + green_extent[2] * 0.5 + red_extent[2] * 0.5 + 0.01
joints = solve_ik(place_pos2, quat)
move_to_joints(joints)

open_gripper()

# Retreat
retreat_pos = place_pos2.copy()
retreat_pos[2] += 0.15
joints = solve_ik(retreat_pos, quat)
move_to_joints(joints)