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
# approach from above: use grasp orientation but position above
target_pos = grasp_world[:3, 3].copy()
target_pos[2] += 0.1

# convert rotation to quaternion wxyz
R = grasp_world[:3, :3]
# ensure proper rotation
tr = np.trace(R)
if tr > 0:
    S = np.sqrt(tr + 1.0) * 2
    w = 0.25 * S
    x = (R[2,1] - R[1,2]) / S
    y = (R[0,2] - R[2,0]) / S
    z = (R[1,0] - R[0,1]) / S
else:
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
quat = np.array([w, x, y, z])
quat = quat / np.linalg.norm(quat)

open_gripper()
joints = solve_ik(target_pos, quat)
move_to_joints(joints)

# move down to grasp
grasp_pos = grasp_world[:3, 3].copy()
joints = solve_ik(grasp_pos, quat)
move_to_joints(joints)

close_gripper()

# lift
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.15
joints = solve_ik(lift_pos, quat)
move_to_joints(joints)
print("done")