```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
ext = cam["pose_mat"]

def get_center_world(text):
    res = segment_sam3_text_prompt(rgb, text_prompt=text)
    best = max(res, key=lambda d: d["score"])
    mask = best["mask"]
    ys, xs = np.where(mask)
    u, v = xs.mean(), ys.mean()
    z = depth[int(v), int(u)]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    p_cam = np.array([x, y, z, 1.0])
    p_world = ext @ p_cam
    return p_world[:3], mask

red_pos, red_mask = get_center_world("red cube")
green_pos, green_mask = get_center_world("green cube")

# Plan grasp on red cube
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=red_mask.astype(np.int32))
best_idx = grasp_scores.argmax()
best_T = grasp_poses[best_idx]
grasp_world = ext @ best_T

# Approach from above: use grasp position but orient downward
grasp_pos = grasp_world[:3, 3]
# Downward orientation (gripper pointing down): quaternion wxyz
down_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x -> pointing down

# Move above red cube
above = grasp_pos.copy()
above[2] += 0.10
joints = solve_ik(above, down_quat)
move_to_joints(joints)

# Move down to grasp
joints = solve_ik(grasp_pos, down_quat)
move_to_joints(joints)
close_gripper()

# Lift
lift = grasp_pos.copy()
lift[2] += 0.15
joints = solve_ik(lift, down_quat)
move_to_joints(joints)

# Move above green cube
place_above = green_pos.copy()
place_above[2] += 0.15
joints = solve_ik(place_above, down_quat)
move_to_joints(joints)

# Lower to place
place_pos = green_pos.copy()
place_pos[2] += 0.05
joints = solve_ik(place_pos, down_quat)
move_to_joints(joints)

open_gripper()

# Retreat
retreat = place_pos.copy()
retreat[2] += 0.15
joints = solve_ik(retreat, down_quat)
move_to_joints(joints)
```