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
T_cam = grasp_poses[best_idx]
T_world = ext @ T_cam

pos = T_world[:3, 3]
R = T_world[:3, :3]
# convert rotation to quaternion wxyz
def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1]-R[1,2])/s
        y = (R[0,2]-R[2,0])/s
        z = (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        w = (R[2,1]-R[1,2])/s
        x = 0.25*s
        y = (R[0,1]+R[1,0])/s
        z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        w = (R[0,2]-R[2,0])/s
        x = (R[0,1]+R[1,0])/s
        y = 0.25*s
        z = (R[1,2]+R[2,1])/s
    else:
        s = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        w = (R[1,0]-R[0,1])/s
        x = (R[0,2]+R[2,0])/s
        y = (R[1,2]+R[2,1])/s
        z = 0.25*s
    q = np.array([w,x,y,z])
    return q/np.linalg.norm(q)

quat = rot_to_quat(R)

# approach from above
pre = pos.copy()
pre[2] += 0.10
open_gripper()
j = solve_ik(pre, quat)
move_to_joints(j)
j = solve_ik(pos, quat)
move_to_joints(j)
close_gripper()
lift = pos.copy()
lift[2] += 0.15
j = solve_ik(lift, quat)
move_to_joints(j)
print("done", pos, quat)