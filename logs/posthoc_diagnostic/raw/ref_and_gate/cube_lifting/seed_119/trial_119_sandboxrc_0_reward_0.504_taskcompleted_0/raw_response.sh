I'll start by getting the observation and segmenting the red cube.
```python
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
print("mask score", best["score"], "box", best.get("box"))
print("depth range", depth.min(), depth.max())
grasp_poses, grasp_scores = plan_grasp(depth=depth, intrinsics=K, segmentation=mask)
print("num grasps", len(grasp_scores))
best_idx = int(grasp_scores.argmax())
best_T = grasp_poses[best_idx]
print("best score", grasp_scores[best_idx])
print("best_T cam\n", best_T)
world_T = ext @ best_T
print("world_T\n", world_T)
```
I'll wait for the output to determine the grasp pose in world frame.