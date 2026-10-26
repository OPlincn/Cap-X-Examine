```python
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
best_T = grasp_poses[best_idx]

grasp_world = cam_pose @ best_T
pos = grasp_world[:3, :3]
# approach from above: use grasp position, orientation pointing down
target_pos = grasp_world[:3, 3]

# pre-grasp above
pre_pos = target_pos.copy()
pre_pos[2] += 0.10

# downward orientation quaternion (wxyz): rotate 180 about x -> gripper pointing down
quat_down = np.array([0.0, 1.0, 0.0, 0.0])

open_gripper()
joints = solve_ik(pre_pos, quat_down)
move_to_joints(joints)

joints = solve_ik(target_pos, quat_down)
move_to_joints(joints)

close_gripper()

lift_pos = target_pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, quat_down)
move_to_joints(joints)
```