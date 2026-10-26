I'll start by getting the observation and segmenting the red cube.
```python
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

results = segment_sam3_text_prompt(rgb, text_prompt="red cube")
print("num results:", len(results))
for r in results:
    print(r.get("score"), r.get("box"))
```
I'll wait for the output to see the segmentation results.