import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extr = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(grasp_scores.argmax())
best_T = grasp_poses[best_idx]
grasp_world = extr @ best_T

# compute object center in world via mask points
ys, xs = np.where(mask)
zs = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
X = (xs - cx) * zs / fx
Y = (ys - cy) * zs / fy
pts_cam = np.stack([X, Y, zs], axis=1)
pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1))], axis=1)
pts_world = (extr @ pts_cam_h.T).T[:, :3]
center = pts_world.mean(axis=0)

# approach from above
target_pos = center + np.array([0,0,0.15])
quat = np.array([0.0, 1.0, 0.0, 0.0])  # pointing down
joints = solve_ik(target_pos, quat)
move_to_joints(joints)
open_gripper()

# descend to grasp
grasp_pos = center + np.array([0,0,0.02])
joints = solve_ik(grasp_pos, quat)
move_to_joints(joints)
close_gripper()

# lift
lift_pos = center + np.array([0,0,0.2])
joints = solve_ik(lift_pos, quat)
move_to_joints(joints)
print("done", center)