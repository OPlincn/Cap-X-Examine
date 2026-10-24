I'll start by getting an observation to understand the scene.
```python
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
ext = cam["pose_mat"]
print("rgb", rgb.shape, "depth", depth.shape)
print("K", K)
print("ext", ext)
masks = segment_sam3_text_prompt(rgb, text_prompt="red cube")
print("num masks", len(masks))
for m in masks:
    print("score", m.get("score"), "box", m.get("box"))
```