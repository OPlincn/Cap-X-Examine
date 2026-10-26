# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

results = segment_sam3_text_prompt(rgb, text_prompt="red cube")
print("num results:", len(results))
for r in results:
    print(r.get("score"), r.get("box"))

# Code block 1
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

results = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(results, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
print("num grasps:", len(grasp_poses))
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
print("best score:", grasp_scores[best_idx])

grasp_world = cam_ext @ best_T
print("grasp world:\n", grasp_world)

# Approach from above: use grasp position, but orient gripper pointing down
pos = grasp_world[:3, 3]
print("grasp pos:", pos)

# Pre-grasp pose above the cube
pre_pos = pos.copy()
pre_pos[2] += 0.10
down_quat = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper pointing down

joints_pre = solve_ik(pre_pos, down_quat)
move_to_joints(joints_pre)
open_gripper()

# Move down to grasp
joints_grasp = solve_ik(pos, down_quat)
move_to_joints(joints_grasp)
close_gripper()

# Lift
lift_pos = pos.copy()
lift_pos[2] += 0.15
joints_lift = solve_ik(lift_pos, down_quat)
move_to_joints(joints_lift)
print("done")