# Code block 0
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
ext = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(grasp_scores.argmax())
best_T = grasp_poses[best_idx]
grasp_world = ext @ best_T

pos = grasp_world[:3, 3]
R = grasp_world[:3, :3]

# Convert rotation matrix to quaternion (wxyz)
def rot_to_quat(R):
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

quat = rot_to_quat(R)

# Approach from above
pre = pos.copy()
pre[2] += 0.10
open_gripper()
joints_pre = solve_ik(pre, quat)
move_to_joints(joints_pre)

joints_grasp = solve_ik(pos, quat)
move_to_joints(joints_grasp)

close_gripper()

lift = pos.copy()
lift[2] += 0.15
joints_lift = solve_ik(lift, quat)
move_to_joints(joints_lift)

print("done", pos)

# Code block 1
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
ext = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

# Get 3D points of the cube to find its center
ys, xs = np.where(mask > 0)
zs = depth[ys, xs]
valid = zs > 0
xs, ys, zs = xs[valid], ys[valid], zs[valid]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
X = (xs - cx) * zs / fx
Y = (ys - cy) * zs / fy
Z = zs
pts_cam = np.stack([X, Y, Z], axis=1)
pts_world = (ext[:3,:3] @ pts_cam.T).T + ext[:3,3]
cube_center = pts_world.mean(axis=0)
print("cube center", cube_center)

# Use grasp pose orientation but target cube center
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(grasp_scores.argmax())
best_T = grasp_poses[best_idx]
grasp_world = ext @ best_T
R = grasp_world[:3, :3]

def rot_to_quat(R):
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

quat = rot_to_quat(R)

# Target position: cube center, slightly above to grasp
pos = cube_center.copy()
pos[2] += 0.01

pre = pos.copy()
pre[2] += 0.12
open_gripper()
joints_pre = solve_ik(pre, quat)
move_to_joints(joints_pre)

joints_grasp = solve_ik(pos, quat)
move_to_joints(joints_grasp)

close_gripper()

lift = pos.copy()
lift[2] += 0.20
joints_lift = solve_ik(lift, quat)
move_to_joints(joints_lift)

print("done lift", lift)

# Code block 2
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
ext = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

# Get 3D points of the cube to find its center
ys, xs = np.where(mask > 0)
zs = depth[ys, xs]
valid = zs > 0
xs, ys, zs = xs[valid], ys[valid], zs[valid]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
X = (xs - cx) * zs / fx
Y = (ys - cy) * zs / fy
Z = zs
pts_cam = np.stack([X, Y, Z], axis=1)
pts_world = (ext[:3,:3] @ pts_cam.T).T + ext[:3,3]
cube_center = pts_world.mean(axis=0)
print("cube center", cube_center)

# Use a top-down grasp orientation (gripper pointing down)
# Quaternion for gripper pointing down (rotate 180 deg about x): w=0, x=1, y=0, z=0
quat = np.array([0.0, 1.0, 0.0, 0.0])

# Target position: cube center
pos = cube_center.copy()

pre = pos.copy()
pre[2] += 0.12
open_gripper()
joints_pre = solve_ik(pre, quat)
move_to_joints(joints_pre)

joints_grasp = solve_ik(pos, quat)
move_to_joints(joints_grasp)

close_gripper()

lift = pos.copy()
lift[2] += 0.20
joints_lift = solve_ik(lift, quat)
move_to_joints(joints_lift)

print("done lift", lift)

# Code block 3
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
ext = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

# Get 3D points of the cube to find its center and top
ys, xs = np.where(mask > 0)
zs = depth[ys, xs]
valid = zs > 0
xs, ys, zs = xs[valid], ys[valid], zs[valid]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
X = (xs - cx) * zs / fx
Y = (ys - cy) * zs / fy
Z = zs
pts_cam = np.stack([X, Y, Z], axis=1)
pts_world = (ext[:3,:3] @ pts_cam.T).T + ext[:3,3]
cube_center = pts_world.mean(axis=0)
cube_top = pts_world[:,2].max()
print("cube center", cube_center, "top", cube_top)

# Use grasp pose from planner for orientation
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(grasp_scores.argmax())
best_T = grasp_poses[best_idx]
grasp_world = ext @ best_T
R = grasp_world[:3, :3]

def rot_to_quat(R):
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

quat = rot_to_quat(R)

# Grasp at cube center height (mid-body) so fingers close on the sides
pos = cube_center.copy()

pre = pos.copy()
pre[2] += 0.12
open_gripper()
joints_pre = solve_ik(pre, quat)
move_to_joints(joints_pre)

joints_grasp = solve_ik(pos, quat)
move_to_joints(joints_grasp)

close_gripper()

lift = pos.copy()
lift[2] += 0.20
joints_lift = solve_ik(lift, quat)
move_to_joints(joints_lift)

print("done lift", lift)