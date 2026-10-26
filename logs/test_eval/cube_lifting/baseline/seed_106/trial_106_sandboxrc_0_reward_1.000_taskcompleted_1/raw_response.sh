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
# approach from above: keep orientation but ensure gripper points down
# use rotation with z-axis pointing down (gripper approach along -z)
quat_wxyz = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x -> z down
pre = pos + np.array([0.0, 0.0, 0.12])
joints_pre = solve_ik(pre, quat_wxyz)
move_to_joints(joints_pre)
open_gripper()
joints_grasp = solve_ik(pos, quat_wxyz)
move_to_joints(joints_grasp)
close_gripper()
lift = pos + np.array([0.0, 0.0, 0.15])
joints_lift = solve_ik(lift, quat_wxyz)
move_to_joints(joints_lift)
print("done", pos)