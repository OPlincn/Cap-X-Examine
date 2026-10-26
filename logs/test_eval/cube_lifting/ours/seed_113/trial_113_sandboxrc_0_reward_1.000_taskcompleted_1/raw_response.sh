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

pos = grasp_world[:3, 3]
R = grasp_world[:3, :3]

# Approach from above: use gripper pointing down
# Build quaternion for downward grasp (rotate 180 about x from identity)
# We'll use the grasp orientation but ensure approach from above
quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper down

# Pre-grasp position above
pre = pos.copy()
pre[2] += 0.10

open_gripper()
joints = solve_ik(pre, quat)
move_to_joints(joints)

joints = solve_ik(pos, quat)
move_to_joints(joints)

close_gripper()

lift = pos.copy()
lift[2] += 0.15
joints = solve_ik(lift, quat)
move_to_joints(joints)

print("grasp pos", pos, "score", grasp_scores[best_idx])