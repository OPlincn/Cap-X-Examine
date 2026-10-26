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
best_T = grasp_poses[best_idx]
grasp_world = ext @ best_T
print("grasp_world:\n", grasp_world)
print("score:", grasp_scores[best_idx])

# compute cube center from mask points
ys, xs = np.where(mask)
z = depth[ys, xs]
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
X = (xs - cx) * z / fx
Y = (ys - cy) * z / fy
pts_cam = np.stack([X, Y, z], axis=1)
pts_cam = pts_cam[np.isfinite(pts_cam[:,2]) & (pts_cam[:,2] > 0)]
R = ext[:3,:3]; t = ext[:3,3]
pts_world = (R @ pts_cam.T).T + t
obb = get_oriented_bounding_box_from_3d_points(pts_world)
print("obb center:", obb["center"], "extent:", obb["extent"])
np.save("/tmp/grasp_world.npy", grasp_world)
np.save("/tmp/obb_center.npy", obb["center"])