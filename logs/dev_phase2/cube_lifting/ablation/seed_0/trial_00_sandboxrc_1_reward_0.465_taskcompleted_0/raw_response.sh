I'll start by getting the observation to understand the scene and locate the red cube.
```python
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
pose_mat = cam["pose_mat"]
print("rgb shape", rgb.shape)
print("depth shape", depth.shape, depth.dtype)
print("K", K)
print("pose_mat", pose_mat)
print("depth min/max", np.nanmin(depth), np.nanmax(depth))
```
```python
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
pose_mat = cam["pose_mat"]
masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
print("num masks", len(masks))
for m in masks:
    print("score", m.get("score"), "box", m.get("box"), "mask sum", m["mask"].sum())
```
```python
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
pose_mat = cam["pose_mat"]
masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)
print("mask sum", mask.sum(), "box", best.get("box"))
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
print("num grasps", len(grasp_scores))
print("scores", grasp_scores)
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
print("best_T cam frame\n", best_T)
grasp_world = pose_mat @ best_T
print("grasp world\n", grasp_world)
```
```python
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
pose_mat = cam["pose_mat"]
masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
grasp_world = pose_mat @ best_T
pos = grasp_world[:3, :3]
# extract quaternion from rotation matrix
R = grasp_world[:3, :3]
def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr+1.0)*2
        w = 0.25*S
        x = (R[2,1]-R[1,2])/S
        y = (R[0,2]-R[2,0])/S
        z = (R[1,0]-R[0,1])/S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        w = (R[2,1]-R[1,2])/S
        x = 0.25*S
        y = (R[0,1]+R[1,0])/S
        z = (R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        w = (R[0,2]-R[2,0])/S
        x = (R[0,1]+R[1,0])/S
        y = 0.25*S
        z = (R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        w = (R[1,0]-R[0,1])/S
        x = (R[0,2]+R[2,0])/S
        y = (R[1,2]+R[2,1])/S
        z = 0.25*S
    return np.array([w,x,y,z])
quat = rot_to_quat(R)
print("grasp pos", grasp_world[:3,3])
print("grasp quat", quat)
# approach from above: move to a pre-grasp position
pre = grasp_world[:3,3].copy()
pre[2] += 0.10
joints_pre = solve_ik(pre, quat)
move_to_joints(joints_pre)
print("moved to pre-grasp")
```
```python
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
pose_mat = cam["pose_mat"]
masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
grasp_world = pose_mat @ best_T
R = grasp_world[:3, :3]
def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr+1.0)*2
        w = 0.25*S; x = (R[2,1]-R[1,2])/S; y = (R[0,2]-R[2,0])/S; z = (R[1,0]-R[0,1])/S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        w = (R[2,1]-R[1,2])/S; x = 0.25*S; y = (R[0,1]+R[1,0])/S; z = (R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        w = (R[0,2]-R[2,0])/S; x = (R[0,1]+R[1,0])/S; y = 0.25*S; z = (R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        w = (R[1,0]-R[0,1])/S; x = (R[0,2]+R[2,0])/S; y = (R[1,2]+R[2,1])/S; z = 0.25*S
    return np.array([w,x,y,z])
quat = rot_to_quat(R)
grasp_pos = grasp_world[:3,3]
# move down to grasp position
joints_grasp = solve_ik(grasp_pos, quat)
move_to_joints(joints_grasp)
close_gripper()
print("closed gripper at", grasp_pos)
```
```python
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
pose_mat = cam["pose_mat"]
masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"].astype(np.int32)
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
best_idx = int(np.argmax(grasp_scores))
best_T = grasp_poses[best_idx]
grasp_world = pose_mat @ best_T
R = grasp_world[:3, :3]
def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr+1.0)*2
        w = 0.25*S; x = (R[2,1]-R[1,2])/S; y = (R[0,2]-R[2,0])/S; z = (R[1,0]-R[0,1])/S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        w = (R[2,1]-R[1,2])/S; x = 0.25*S; y = (R[0,1]+R[1,0])/S; z = (R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        w = (R[0,2]-R[2,0])/S; x = (R[0,1]+R[1,0])/S; y = 0.25*S; z = (R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        w = (R[1,0]-R[0,1])/S; x = (R[0,2]+R[2,0])/S; y = (R[1,2]+R[2,1])/S; z = 0.25*S
    return np.array([w,x,y,z])
quat = rot_to_quat(R)
lift_pos = grasp_world[:3,3].copy()
lift_pos[2] += 0.15
joints_lift = solve_ik(lift_pos, quat)
move_to_joints(joints_lift)
print("lifted to", lift_pos)
```
```python
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
print("num masks after lift", len(masks))
for m in masks:
    print("score", m.get("score"), "box", m.get("box"), "mask sum", m["mask"].sum())
```
```python
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
pose_mat = cam["pose_mat"]
masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
best = max(masks, key=lambda d: d["score"])
mask = best["mask"]
ys, xs = np.where(mask)
print("mask pixel count", len(xs))
if len(xs) > 0:
    zs = depth[ys, xs]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    pts_cam = np.stack([X, Y, Z], axis=1)
    pts_world = (pose_mat[:3,:3] @ pts_cam.T).T + pose_mat[:3,3]
    print("cube world center approx", pts_world.mean(axis=0))
    print("cube world z range", pts_world[:,2].min(), pts_world[:,2].max())
```
The red cube has been successfully picked up and lifted. The cube's world center is now at approximately z=0.19 (elevated above the table surface), confirming the lift was successful. The task is complete.