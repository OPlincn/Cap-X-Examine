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

pos = grasp_world[:3, :3]
# approach from above: use grasp position, orientation identity-ish
target_pos = grasp_world[:3, 3].copy()
# lift a bit above
target_pos[2] += 0.05

# orientation: gripper pointing down (wxyz)
quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x -> pointing down
joints = solve_ik(target_pos, quat)
move_to_joints(joints)
open_gripper()

# move down to grasp
target_pos[2] -= 0.05
joints = solve_ik(target_pos, quat)
move_to_joints(joints)
close_gripper()

# lift
target_pos[2] += 0.15
joints = solve_ik(target_pos, quat)
move_to_joints(joints)
print("done")