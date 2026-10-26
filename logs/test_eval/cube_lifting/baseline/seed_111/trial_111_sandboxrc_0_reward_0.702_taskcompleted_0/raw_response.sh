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

pos = grasp_world[:3, :3]
# approach from above: use grasp orientation but position above
quat = np.array([1.0, 0.0, 0.0, 0.0])
target = grasp_world[:3, 3].copy()
target[2] += 0.1
joints = solve_ik(target, quat)
move_to_joints(joints)
open_gripper()

# move down to grasp
target2 = grasp_world[:3, 3].copy()
joints2 = solve_ik(target2, quat)
move_to_joints(joints2)
close_gripper()

# lift
target3 = target2.copy()
target3[2] += 0.15
joints3 = solve_ik(target3, quat)
move_to_joints(joints3)
print("done", grasp_world[:3,3])