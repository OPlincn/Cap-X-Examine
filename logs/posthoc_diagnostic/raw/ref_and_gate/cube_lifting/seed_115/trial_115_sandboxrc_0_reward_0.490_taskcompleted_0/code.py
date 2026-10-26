# Code block 0
I'll start by getting the observation and segmenting the red cube to find its 3D position.

import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
ext = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
print("num masks", len(masks))
for m in masks:
    print(m.get("score"), m.get("box"))
best = max(masks, key=lambda d: d["score"])
mask = best["mask"]
print("mask sum", mask.sum())

# backproject to 3D
ys, xs = np.nonzero(mask)
z = depth[ys, xs]
valid = z > 0
xs, ys, z = xs[valid], ys[valid], z[valid]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
X = (xs - cx) * z / fx
Y = (ys - cy) * z / fy
pts_cam = np.stack([X, Y, z], axis=1)
# transform to world
R = ext[:3,:3]; t = ext[:3,3]
pts_world = (R @ pts_cam.T).T + t
print("world mean", pts_world.mean(axis=0))
print("world min", pts_world.min(axis=0))
print("world max", pts_world.max(axis=0))
obb = get_oriented_bounding_box_from_3d_points(pts_world)
print("obb center", obb["center"])
print("obb extent", obb["extent"])
print("obb R", obb["R"])

# Code block 1
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
ext = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
print("num masks", len(masks))
for m in masks:
    print(m.get("score"), m.get("box"))
best = max(masks, key=lambda d: d["score"])
mask = best["mask"]
print("mask sum", mask.sum())

ys, xs = np.nonzero(mask)
z = depth[ys, xs]
valid = z > 0
xs, ys, z = xs[valid], ys[valid], z[valid]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
X = (xs - cx) * z / fx
Y = (ys - cy) * z / fy
pts_cam = np.stack([X, Y, z], axis=1)
R = ext[:3,:3]; t = ext[:3,3]
pts_world = (R @ pts_cam.T).T + t
print("world mean", pts_world.mean(axis=0))
print("world min", pts_world.min(axis=0))
print("world max", pts_world.max(axis=0))
obb = get_oriented_bounding_box_from_3d_points(pts_world)
print("obb center", obb["center"])
print("obb extent", obb["extent"])
print("obb R", obb["R"])

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

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
print("num grasps", len(grasp_poses))
best_idx = int(grasp_scores.argmax())
best_T = grasp_poses[best_idx]
print("best score", grasp_scores[best_idx])

# transform to world frame
T_world = ext @ best_T
pos = T_world[:3, 3]
R = T_world[:3, :3]
print("grasp world pos", pos)

# convert rotation to quaternion wxyz
def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

quat = rot_to_quat(R)
quat = quat / np.linalg.norm(quat)
print("quat", quat)

# approach from above: pre-grasp position
pre_pos = pos.copy()
pre_pos[2] += 0.10

open_gripper()
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)

joints_grasp = solve_ik(pos, quat)
move_to_joints(joints_grasp)

close_gripper()

# lift
lift_pos = pos.copy()
lift_pos[2] += 0.15
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)
print("done")