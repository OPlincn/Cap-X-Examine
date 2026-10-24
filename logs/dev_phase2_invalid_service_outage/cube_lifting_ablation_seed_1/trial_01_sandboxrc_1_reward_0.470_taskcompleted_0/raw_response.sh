I'll start by getting the observation to understand the scene.

import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
pose_mat = cam["pose_mat"]
print("rgb", rgb.shape, "depth", depth.shape)
print("K", K)
print("pose_mat", pose_mat)
print("depth range", np.nanmin(depth), np.nanmax(depth))