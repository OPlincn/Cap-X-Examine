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

# TCP offset: approach from above, align gripper z-axis downward
pos = grasp_world[:3, 3]
R = grasp_world[:3, :3]

# Build a downward-facing orientation (gripper z pointing down)
z_axis = np.array([0.0, 0.0, -1.0])
# use grasp rotation's x/y projected
x_axis = R[:, 0]
x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
if np.linalg.norm(x_axis) < 1e-6:
    x_axis = np.array([1.0, 0.0, 0.0])
x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)
R_down = np.column_stack([x_axis, y_axis, z_axis])

def mat_to_quat_wxyz(Rm):
    t = np.trace(Rm)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (Rm[2,1] - Rm[1,2]) / s
        y = (Rm[0,2] - Rm[2,0]) / s
        z = (Rm[1,0] - Rm[0,1]) / s
    elif Rm[0,0] > Rm[1,1] and Rm[0,0] > Rm[2,2]:
        s = np.sqrt(1.0 + Rm[0,0] - Rm[1,1] - Rm[2,2]) * 2
        w = (Rm[2,1] - Rm[1,2]) / s
        x = 0.25 * s
        y = (Rm[0,1] + Rm[1,0]) / s
        z = (Rm[0,2] + Rm[2,0]) / s
    elif Rm[1,1] > Rm[2,2]:
        s = np.sqrt(1.0 + Rm[1,1] - Rm[0,0] - Rm[2,2]) * 2
        w = (Rm[0,2] - Rm[2,0]) / s
        x = (Rm[0,1] + Rm[1,0]) / s
        y = 0.25 * s
        z = (Rm[1,2] + Rm[2,1]) / s
    else:
        s = np.sqrt(1.0 + Rm[2,2] - Rm[0,0] - Rm[1,1]) * 2
        w = (Rm[1,0] - Rm[0,1]) / s
        x = (Rm[0,2] + Rm[2,0]) / s
        y = (Rm[1,2] + Rm[2,1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

quat = mat_to_quat_wxyz(R_down)

# Pre-grasp above
pre_pos = pos + np.array([0.0, 0.0, 0.10])
joints_pre = solve_ik(pre_pos, quat)
move_to_joints(joints_pre)
open_gripper()

# Move down to grasp
joints_grasp = solve_ik(pos, quat)
move_to_joints(joints_grasp)
close_gripper()

# Lift
lift_pos = pos + np.array([0.0, 0.0, 0.15])
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)
print("done", pos)