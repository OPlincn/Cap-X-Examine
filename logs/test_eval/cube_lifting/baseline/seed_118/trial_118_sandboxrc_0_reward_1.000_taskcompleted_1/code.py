# Code block 0
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
extr = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)

grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
grasp_world = extr @ best_T

# TCP offset: approach along grasp z-axis, back off a bit
pos = grasp_world[:3, 3]
R = grasp_world[:3, :3]
z_axis = R[:, 2]
approach_pos = pos - 0.08 * z_axis

# quaternion from rotation matrix (wxyz)
def rot_to_quat(Rm):
    t = np.trace(Rm)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (Rm[2,1]-Rm[1,2])/s
        y = (Rm[0,2]-Rm[2,0])/s
        z = (Rm[1,0]-Rm[0,1])/s
    else:
        i = np.argmax(np.diag(Rm))
        if i == 0:
            s = np.sqrt(1+Rm[0,0]-Rm[1,1]-Rm[2,2])*2
            w = (Rm[2,1]-Rm[1,2])/s
            x = 0.25*s
            y = (Rm[0,1]+Rm[1,0])/s
            z = (Rm[0,2]+Rm[2,0])/s
        elif i == 1:
            s = np.sqrt(1+Rm[1,1]-Rm[0,0]-Rm[2,2])*2
            w = (Rm[0,2]-Rm[2,0])/s
            x = (Rm[0,1]+Rm[1,0])/s
            y = 0.25*s
            z = (Rm[1,2]+Rm[2,1])/s
        else:
            s = np.sqrt(1+Rm[2,2]-Rm[0,0]-Rm[1,1])*2
            w = (Rm[1,0]-Rm[0,1])/s
            x = (Rm[0,2]+Rm[2,0])/s
            y = (Rm[1,2]+Rm[2,1])/s
            z = 0.25*s
    q = np.array([w,x,y,z])
    return q/np.linalg.norm(q)

quat = rot_to_quat(R)

open_gripper()
j = solve_ik(approach_pos, quat)
move_to_joints(j)

j2 = solve_ik(pos, quat)
move_to_joints(j2)

close_gripper()

lift_pos = pos + np.array([0.0, 0.0, 0.15])
j3 = solve_ik(lift_pos, quat)
move_to_joints(j3)
print("done", pos, grasp_scores[best_idx])