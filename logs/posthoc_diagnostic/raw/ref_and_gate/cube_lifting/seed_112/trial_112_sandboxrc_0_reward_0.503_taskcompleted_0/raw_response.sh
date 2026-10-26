I'll start by getting an observation to understand the scene and locate the red cube.
```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_pose = cam["pose_mat"]

print("rgb shape:", rgb.shape)
print("depth shape:", depth.shape, "dtype:", depth.dtype)
print("intrinsics:\n", K)
print("cam_pose:\n", cam_pose)

masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
print("num masks:", len(masks))
for i, m in enumerate(masks):
    print(i, "score:", m.get("score"), "box:", m.get("box"), "mask sum:", m["mask"].sum())
```