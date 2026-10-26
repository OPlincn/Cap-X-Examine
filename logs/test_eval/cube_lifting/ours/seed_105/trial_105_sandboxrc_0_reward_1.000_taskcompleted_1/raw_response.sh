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
print("num masks", len(masks))
for m in masks:
    print(m.get("score"), m.get("box"))
```
I'll wait for the output to see the segmentation results.