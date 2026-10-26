import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(np.argmax(grasp_scores))
best_T_cam = grasp_poses[best_idx]

best_T_world = cam_pose @ best_T_cam
grasp_pos = best_T_world[:3, 3]
grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])

# Approach from above
approach_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
joints_approach = solve_ik(approach_pos, grasp_quat)
move_to_joints(joints_approach)

open_gripper()

# Move down to grasp
joints_grasp = solve_ik(grasp_pos, grasp_quat)
move_to_joints(joints_grasp)

close_gripper()

# Lift
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.2])
joints_lift = solve_ik(lift_pos, grasp_quat)
move_to_joints(joints_lift)

print("Grasp position (world):", grasp_pos)
print("Done lifting red cube.")