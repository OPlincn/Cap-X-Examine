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

# Compute object center in world frame from mask points
ys, xs = np.where(mask)
z = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
X = (xs - cx) * z / fx
Y = (ys - cy) * z / fy
pts_cam = np.stack([X, Y, z], axis=1)
pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
pts_world = (ext @ pts_cam_h.T).T[:, :3]
center = pts_world.mean(axis=0)
print("center", center)

# Approach from above
target_pos = center + np.array([0,0,0.15])
quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper pointing down
open_gripper()
joints = solve_ik(target_pos, quat)
move_to_joints(joints)

# Move down to grasp
target_pos2 = center + np.array([0,0,0.02])
joints2 = solve_ik(target_pos2, quat)
move_to_joints(joints2)
close_gripper()

# Lift
target_pos3 = center + np.array([0,0,0.2])
joints3 = solve_ik(target_pos3, quat)
move_to_joints(joints3)
print("done")