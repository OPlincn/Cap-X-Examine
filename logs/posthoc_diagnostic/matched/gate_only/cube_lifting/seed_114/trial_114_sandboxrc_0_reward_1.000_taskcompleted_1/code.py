# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extrinsics = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]

grasp_world = extrinsics @ best_T
pos = grasp_world[:3, 3]
R = grasp_world[:3, :3]

# Convert rotation matrix to quaternion (wxyz)
def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

quat = rot_to_quat(R)

# Approach from above
approach_pos = pos.copy()
approach_pos[2] += 0.10

open_gripper()
joints = solve_ik(approach_pos, quat)
move_to_joints(joints)

joints = solve_ik(pos, quat)
move_to_joints(joints)

close_gripper()

lift_pos = pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, quat)
move_to_joints(joints)

print("Grasp position:", pos)
print("Lift complete")

# Code block 1
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extrinsics = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

# Get 3D points of the cube to find its center
ys, xs = np.where(mask > 0)
zs = depth[ys, xs]
valid = zs > 0
xs, ys, zs = xs[valid], ys[valid], zs[valid]

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
X = (xs - cx) * zs / fx
Y = (ys - cy) * zs / fy
Z = zs
pts_cam = np.stack([X, Y, Z], axis=1)

# Transform to world frame
R_ext = extrinsics[:3, :3]
t_ext = extrinsics[:3, 3]
pts_world = (R_ext @ pts_cam.T).T + t_ext

cube_center = pts_world.mean(axis=0)
print("Cube center world:", cube_center)

# Plan grasp for orientation
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
grasp_world = extrinsics @ best_T
R = grasp_world[:3, :3]

def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

quat = rot_to_quat(R)

# Use cube center for grasp position, slightly above to grip the body
grasp_pos = cube_center.copy()
grasp_pos[2] += 0.005

# Approach from above
approach_pos = grasp_pos.copy()
approach_pos[2] += 0.12

open_gripper()
joints = solve_ik(approach_pos, quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, quat)
move_to_joints(joints)

close_gripper()

# Lift
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.20
joints = solve_ik(lift_pos, quat)
move_to_joints(joints)

print("Grasp position:", grasp_pos)
print("Lift complete")