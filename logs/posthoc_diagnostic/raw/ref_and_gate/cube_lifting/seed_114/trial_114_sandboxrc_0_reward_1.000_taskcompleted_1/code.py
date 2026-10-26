# Code block 0
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
pos = grasp_world[:3, 3]
R = grasp_world[:3, :3]

# Approach from above: use a top-down orientation (gripper z pointing down)
# Build quaternion for rotation with z-axis pointing down (180 deg about x)
quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, 180 deg about x

# Pre-grasp above the cube
pre_pos = pos.copy()
pre_pos[2] += 0.10

open_gripper()
joints = solve_ik(pre_pos, quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(pos, quat)
move_to_joints(joints)

close_gripper()

# Lift
lift_pos = pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, quat)
move_to_joints(joints)

print("Grasp position:", pos)
print("Done")