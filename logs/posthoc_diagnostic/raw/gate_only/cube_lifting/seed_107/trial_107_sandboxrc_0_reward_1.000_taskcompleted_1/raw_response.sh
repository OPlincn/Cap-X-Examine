import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]

grasp_world = cam_ext @ best_T
pos = grasp_world[:3, :3]
# approach from above: use identity-ish orientation pointing down
quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper pointing down

# pre-grasp above
pre = grasp_world[:3, 3].copy()
pre[2] += 0.1
joints = solve_ik(pre, quat)
move_to_joints(joints)

open_gripper()

# move to grasp
joints = solve_ik(grasp_world[:3, 3], quat)
move_to_joints(joints)

close_gripper()

# lift
lift = grasp_world[:3, 3].copy()
lift[2] += 0.2
joints = solve_ik(lift, quat)
move_to_joints(joints)

print("done", grasp_world[:3, 3], grasp_scores[best_idx])