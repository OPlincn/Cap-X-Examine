# Code block 0
import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
cam_ext = cam["pose_mat"]

masks = segment_sam3_text_prompt(rgb, text_prompt="brown spill")
if len(masks) == 0:
    print("No spill found")
else:
    best = max(masks, key=lambda d: d["score"])
    mask = best["mask"]
    ys, xs = np.where(mask)
    # backproject masked pixels to world
    zs = depth[ys, xs]
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xc = (xs - cx) * zs / fx
    yc = (ys - cy) * zs / fy
    pts_cam = np.stack([xc, yc, zs, np.ones_like(zs)], axis=1)
    pts_world = (cam_ext @ pts_cam.T).T[:, :3]
    # filter near table
    pts_world = pts_world[pts_world[:, 2] < 0.05]
    xmin, xmax = pts_world[:, 0].min(), pts_world[:, 0].max()
    ymin, ymax = pts_world[:, 1].min(), pts_world[:, 1].max()
    print("bounds", xmin, xmax, ymin, ymax)

    quat = np.array([0.0, 0.0, 1.0, 0.0])
    z = 0.0

    # small margin inward
    mx = (xmax - xmin) * 0.1
    my = (ymax - ymin) * 0.1
    xmin += mx; xmax -= mx; ymin += my; ymax -= my

    # build zigzag path with small steps
    n_strips = 5
    ys_path = np.linspace(ymin, ymax, n_strips)
    step = 0.02
    path = []
    for i, y in enumerate(ys_path):
        xs_line = np.arange(xmin, xmax + step, step)
        if i % 2 == 1:
            xs_line = xs_line[::-1]
        for x in xs_line:
            path.append((x, y))

    # move to start
    start = np.array([path[0][0], path[0][1], z])
    joints = solve_ik(start, quat)
    move_to_joints(joints)

    for (x, y) in path:
        pos = np.array([x, y, z])
        joints = solve_ik(pos, quat)
        move_to_joints(joints)
    print("wiping done")