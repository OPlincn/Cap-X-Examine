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
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
grasp_world = ext @ best_T
print("grasp_world:\n", grasp_world)
print("score:", grasp_scores[best_idx])

# compute cube center from mask points
ys, xs = np.where(mask)
z = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
X = (xs - cx) * z / fx
Y = (ys - cy) * z / fy
pts_cam = np.stack([X, Y, z], axis=1)
pts_cam = pts_cam[np.isfinite(pts_cam[:,2]) & (pts_cam[:,2] > 0)]
R = ext[:3,:3]; t = ext[:3,3]
pts_world = (R @ pts_cam.T).T + t
obb = get_oriented_bounding_box_from_3d_points(pts_world)
print("obb center:", obb["center"], "extent:", obb["extent"])
np.save("/tmp/grasp_world.npy", grasp_world)
np.save("/tmp/obb_center.npy", obb["center"])

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

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
grasp_world = ext @ best_T

# Approach pose: back off along grasp z-axis (approach direction)
approach = grasp_world.copy()
approach[:3, 3] = grasp_world[:3, 3] - grasp_world[:3, 2] * 0.10

def quat_from_R(Rm):
    # convert rotation matrix to quaternion wxyz
    tr = Rm[0,0] + Rm[1,1] + Rm[2,2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (Rm[2,1] - Rm[1,2]) / S
        y = (Rm[0,2] - Rm[2,0]) / S
        z = (Rm[1,0] - Rm[0,1]) / S
    elif (Rm[0,0] > Rm[1,1]) and (Rm[0,0] > Rm[2,2]):
        S = np.sqrt(1.0 + Rm[0,0] - Rm[1,1] - Rm[2,2]) * 2
        w = (Rm[2,1] - Rm[1,2]) / S
        x = 0.25 * S
        y = (Rm[0,1] + Rm[1,0]) / S
        z = (Rm[0,2] + Rm[2,0]) / S
    elif Rm[1,1] > Rm[2,2]:
        S = np.sqrt(1.0 + Rm[1,1] - Rm[0,0] - Rm[2,2]) * 2
        w = (Rm[0,2] - Rm[2,0]) / S
        x = (Rm[0,1] + Rm[1,0]) / S
        y = 0.25 * S
        z = (Rm[1,2] + Rm[2,1]) / S
    else:
        S = np.sqrt(1.0 + Rm[2,2] - Rm[0,0] - Rm[1,1]) * 2
        w = (Rm[1,0] - Rm[0,1]) / S
        x = (Rm[0,2] + Rm[2,0]) / S
        y = (Rm[1,2] + Rm[2,1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

q_grasp = quat_from_R(grasp_world[:3, :3])

open_gripper()

# Move to approach pose
j_app = solve_ik(approach[:3, 3], q_grasp)
move_to_joints(j_app)

# Move to grasp pose
j_grasp = solve_ik(grasp_world[:3, 3], q_grasp)
move_to_joints(j_grasp)

# Close gripper
close_gripper()

# Lift
lift = grasp_world.copy()
lift[:3, 3] = grasp_world[:3, 3] + np.array([0.0, 0.0, 0.15])
j_lift = solve_ik(lift[:3, 3], q_grasp)
move_to_joints(j_lift)

print("done lifting")