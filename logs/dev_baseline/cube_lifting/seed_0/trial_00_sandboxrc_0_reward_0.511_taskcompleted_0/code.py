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
mask = best["mask"]

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
grasp_world = extrinsics @ best_T

pos = grasp_world[:3, :3]
# approach from above: use grasp orientation but position above
target_pos = grasp_world[:3, 3].copy()

# quaternion from rotation matrix
R = grasp_world[:3, :3]
def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1]-R[1,2])/s
        y = (R[0,2]-R[2,0])/s
        z = (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        w = (R[2,1]-R[1,2])/s
        x = 0.25*s
        y = (R[0,1]+R[1,0])/s
        z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        w = (R[0,2]-R[2,0])/s
        x = (R[0,1]+R[1,0])/s
        y = 0.25*s
        z = (R[1,2]+R[2,1])/s
    else:
        s = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        w = (R[1,0]-R[0,1])/s
        x = (R[0,2]+R[2,0])/s
        y = (R[1,2]+R[2,1])/s
        z = 0.25*s
    return np.array([w,x,y,z])

quat = rot_to_quat(R)
quat = quat / np.linalg.norm(quat)

# pre-grasp above
pre_pos = target_pos + np.array([0,0,0.1])
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)
open_gripper()

# descend
joints_grasp = solve_ik(target_pos, quat)
move_to_joints(joints_grasp)
close_gripper()

# lift
lift_pos = target_pos + np.array([0,0,0.15])
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)
print("done")

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
mask = best["mask"]

# Get 3D points of the cube in world frame
ys, xs = np.where(mask)
z = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
x_cam = (xs - cx) * z / fx
y_cam = (ys - cy) * z / fy
pts_cam = np.stack([x_cam, y_cam, z], axis=1)
pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
pts_world = (extrinsics @ pts_cam_h.T).T[:, :3]

obb = get_oriented_bounding_box_from_3d_points(pts_world)
center = obb["center"]
extent = obb["extent"]
print("center", center, "extent", extent)

# Grasp from above: position at cube center, orientation pointing down
target_pos = center.copy()
# gripper pointing down: rotate 180 about x from identity -> quat (0,1,0,0)
quat = np.array([0.0, 1.0, 0.0, 0.0])

pre_pos = target_pos + np.array([0,0,0.12])
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)
open_gripper()

joints_grasp = solve_ik(target_pos, quat)
move_to_joints(joints_grasp)
close_gripper()

lift_pos = target_pos + np.array([0,0,0.15])
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)
print("done")

# Code block 2
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extrinsics = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"]

ys, xs = np.where(mask)
z = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
x_cam = (xs - cx) * z / fx
y_cam = (ys - cy) * z / fy
pts_cam = np.stack([x_cam, y_cam, z], axis=1)
pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
pts_world = (extrinsics @ pts_cam_h.T).T[:, :3]

obb = get_oriented_bounding_box_from_3d_points(pts_world)
center = obb["center"]
extent = obb["extent"]
print("center", center, "extent", extent)

# Grasp slightly below center to ensure fingers wrap around cube
target_pos = center.copy()
target_pos[2] = center[2] - 0.01
quat = np.array([0.0, 1.0, 0.0, 0.0])

pre_pos = target_pos + np.array([0,0,0.12])
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)
open_gripper()

joints_grasp = solve_ik(target_pos, quat)
move_to_joints(joints_grasp)
close_gripper()

lift_pos = target_pos + np.array([0,0,0.2])
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)
print("done")

# Code block 3
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extrinsics = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"]

# Get 3D points of the cube in world frame
ys, xs = np.where(mask)
z = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
x_cam = (xs - cx) * z / fx
y_cam = (ys - cy) * z / fy
pts_cam = np.stack([x_cam, y_cam, z], axis=1)
pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
pts_world = (extrinsics @ pts_cam_h.T).T[:, :3]

obb = get_oriented_bounding_box_from_3d_points(pts_world)
center = obb["center"]
extent = obb["extent"]
print("center", center, "extent", extent)

# Grasp at cube center height, gripper pointing down
target_pos = center.copy()
quat = np.array([0.0, 1.0, 0.0, 0.0])

# Open gripper first, then approach from above
open_gripper()

pre_pos = target_pos + np.array([0,0,0.15])
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)

# Descend to grasp position
joints_grasp = solve_ik(target_pos, quat)
move_to_joints(joints_grasp)

# Close gripper and wait
close_gripper()

# Lift straight up
lift_pos = target_pos + np.array([0,0,0.2])
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)
print("done")

# Code block 4
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extrinsics = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"]

ys, xs = np.where(mask)
z = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
x_cam = (xs - cx) * z / fx
y_cam = (ys - cy) * z / fy
pts_cam = np.stack([x_cam, y_cam, z], axis=1)
pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
pts_world = (extrinsics @ pts_cam_h.T).T[:, :3]

obb = get_oriented_bounding_box_from_3d_points(pts_world)
center = obb["center"]
extent = obb["extent"]
print("center", center, "extent", extent)

# Grasp at cube center height, gripper pointing down
target_pos = center.copy()
quat = np.array([0.0, 1.0, 0.0, 0.0])

# Open gripper first, then approach from above
open_gripper()

pre_pos = target_pos + np.array([0,0,0.15])
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)

# Descend to grasp position
joints_grasp = solve_ik(target_pos, quat)
move_to_joints(joints_grasp)

# Close gripper and wait
close_gripper()

# Lift straight up
lift_pos = target_pos + np.array([0,0,0.2])
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)
print("done")

# Code block 5
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extrinsics = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"]

ys, xs = np.where(mask)
z = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
x_cam = (xs - cx) * z / fx
y_cam = (ys - cy) * z / fy
pts_cam = np.stack([x_cam, y_cam, z], axis=1)
pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
pts_world = (extrinsics @ pts_cam_h.T).T[:, :3]

obb = get_oriented_bounding_box_from_3d_points(pts_world)
center = obb["center"]
extent = obb["extent"]
print("center", center, "extent", extent)

# Use Contact-GraspNet for a proper grasp pose
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask.astype(np.int32))
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
grasp_world = extrinsics @ best_T

target_pos = grasp_world[:3, 3].copy()
R = grasp_world[:3, :3]

def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1]-R[1,2])/s
        y = (R[0,2]-R[2,0])/s
        z = (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        w = (R[2,1]-R[1,2])/s
        x = 0.25*s
        y = (R[0,1]+R[1,0])/s
        z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        w = (R[0,2]-R[2,0])/s
        x = (R[0,1]+R[1,0])/s
        y = 0.25*s
        z = (R[1,2]+R[2,1])/s
    else:
        s = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        w = (R[1,0]-R[0,1])/s
        x = (R[0,2]+R[2,0])/s
        y = (R[1,2]+R[2,1])/s
        z = 0.25*s
    return np.array([w,x,y,z])

quat = rot_to_quat(R)
quat = quat / np.linalg.norm(quat)
print("target_pos", target_pos, "quat", quat)

open_gripper()

# Pre-grasp: back off along the grasp approach direction (z-axis of grasp frame)
approach = grasp_world[:3, 2]
pre_pos = target_pos - approach * 0.1
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)

# Move to grasp
joints_grasp = solve_ik(target_pos, quat)
move_to_joints(joints_grasp)

close_gripper()

# Lift straight up
lift_pos = target_pos + np.array([0, 0, 0.15])
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)
print("done")

# Code block 6
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extrinsics = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"]

ys, xs = np.where(mask)
z = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
x_cam = (xs - cx) * z / fx
y_cam = (ys - cy) * z / fy
pts_cam = np.stack([x_cam, y_cam, z], axis=1)
pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
pts_world = (extrinsics @ pts_cam_h.T).T[:, :3]

obb = get_oriented_bounding_box_from_3d_points(pts_world)
center = obb["center"]
extent = obb["extent"]
print("center", center, "extent", extent)

# Grasp from above with gripper pointing straight down
target_pos = center.copy()
quat = np.array([0.0, 1.0, 0.0, 0.0])

open_gripper()

pre_pos = target_pos + np.array([0, 0, 0.12])
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)

joints_grasp = solve_ik(target_pos, quat)
move_to_joints(joints_grasp)

close_gripper()

lift_pos = target_pos + np.array([0, 0, 0.2])
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)

# Verify lift by checking cube position
obs2 = get_observation()
cam2 = obs2["robot0_robotview"]
masks2 = segment_sam3_text_prompt(cam2["images"]["rgb"], text_prompt="red cube")
if masks2:
    best2 = max(masks2, key=lambda d: d["score"])
    m2 = best2["mask"]
    ys2, xs2 = np.where(m2)
    z2 = cam2["images"]["depth"][ys2, xs2]
    K2 = cam2["intrinsics"]
    fx2, fy2 = K2[0,0], K2[1,1]
    cx2, cy2 = K2[0,2], K2[1,2]
    xc2 = (xs2 - cx2) * z2 / fx2
    yc2 = (ys2 - cy2) * z2 / fy2
    pc2 = np.stack([xc2, yc2, z2], axis=1)
    pch2 = np.concatenate([pc2, np.ones((pc2.shape[0],1))], axis=1)
    pw2 = (cam2["pose_mat"] @ pch2.T).T[:, :3]
    obb2 = get_oriented_bounding_box_from_3d_points(pw2)
    print("new center", obb2["center"])
print("done")

# Code block 7
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extrinsics = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"]

ys, xs = np.where(mask)
z = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
x_cam = (xs - cx) * z / fx
y_cam = (ys - cy) * z / fy
pts_cam = np.stack([x_cam, y_cam, z], axis=1)
pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
pts_world = (extrinsics @ pts_cam_h.T).T[:, :3]

obb = get_oriented_bounding_box_from_3d_points(pts_world)
center = obb["center"]
extent = obb["extent"]
print("center", center, "extent", extent)

# Grasp at cube center height, gripper pointing down
target_pos = center.copy()
quat = np.array([0.0, 1.0, 0.0, 0.0])

open_gripper()

pre_pos = target_pos + np.array([0, 0, 0.12])
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)

joints_grasp = solve_ik(target_pos, quat)
move_to_joints(joints_grasp)

close_gripper()

lift_pos = target_pos + np.array([0, 0, 0.2])
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)

# Verify lift by checking cube position
obs2 = get_observation()
cam2 = obs2["robot0_robotview"]
masks2 = segment_sam3_text_prompt(cam2["images"]["rgb"], text_prompt="red cube")
if masks2:
    best2 = max(masks2, key=lambda d: d["score"])
    m2 = best2["mask"]
    ys2, xs2 = np.where(m2)
    z2 = cam2["images"]["depth"][ys2, xs2]
    K2 = cam2["intrinsics"]
    fx2, fy2 = K2[0,0], K2[1,1]
    cx2, cy2 = K2[0,2], K2[1,2]
    xc2 = (xs2 - cx2) * z2 / fx2
    yc2 = (ys2 - cy2) * z2 / fy2
    pc2 = np.stack([xc2, yc2, z2], axis=1)
    pch2 = np.concatenate([pc2, np.ones((pc2.shape[0],1))], axis=1)
    pw2 = (cam2["pose_mat"] @ pch2.T).T[:, :3]
    obb2 = get_oriented_bounding_box_from_3d_points(pw2)
    print("new center", obb2["center"])
print("done")