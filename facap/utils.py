import numpy as np
import torch
import open3d as o3d

from matplotlib import pyplot as plt
from facap.geometry.numpy import unproject_points_rotvec
from facap.geometry.open3d import make_pcd_from_numpy, make_line_set


def read_npy(path):
    paths = np.load(path, allow_pickle=True)

    edges = []

    for path in paths:
        ed = list(zip(path[1:].tolist(), path[:-1].tolist()))
        ed = [(tuple(i[0]), tuple(i[1])) for i in ed if i[0] != i[1]]
        edges.extend(ed)
    corners = [tuple(elem) for edge in edges for elem in edge]
    corners = set(corners)

    return corners, edges


def plot_graph(edges, color, name, markersize=20):
    for j, (st, end) in enumerate(edges):
        if j == 0:
            plt.plot([st[0], end[0]], [st[1], end[1]], f"{color}.-", label=name, markersize=markersize)
        else:
            plt.plot([st[0], end[0]], [st[1], end[1]], f"{color}.-", markersize=markersize)


def dicts_to_torch(dict_list, device):
    for dct in dict_list:
        for key in dct:
            if key != "camera_idxs":
                dct[key] = torch.from_numpy(dct[key]).to(device).float()


def dicts_to_numpy(dict_list):
    for dct in dict_list:
        for key in dct:
            if key != "camera_idxs":
                dct[key] = dct[key].cpu().detach().numpy()


def visualize_aligned_walls(wall_data, save_path):
    segments = [tuple(i.tolist()) for i in wall_data["segments"]]
    unique_segments = list(set(segments))

    plt.figure(figsize=(8, 8))

    segment_to_color = { i: np.random.rand(3) for i in unique_segments}
    points_colors = [segment_to_color[i] for i in segments]

    pcd = unproject_points_rotvec(wall_data["depths"], wall_data["points"],
                                  wall_data["f"], wall_data["pp"],
                                  wall_data["rotvecs"], wall_data["translations"])

    for s, c in segment_to_color.items():
        plt.plot([s[0], s[2]], [s[1], s[3]], c=c)
    plt.scatter(pcd[:, 0], pcd[:, 2], c=points_colors)
    plt.savefig(f"{save_path}/alligned_walls.png")


def visualize_data(data, save_path):
    left, right, wall, floor = data
    left_pcd = unproject_points_rotvec(left["depths"], left["points"], left["f"],
                                       left["pp"], left["rotvecs"], left["translations"])

    right_pcd = unproject_points_rotvec(right["depths"], right["points"], right["f"],
                                        right["pp"], right["rotvecs"], right["translations"])

    floor_pcd = unproject_points_rotvec(floor["depths"], floor["points"], floor["f"],
                                        floor["pp"], floor["rotvecs"], floor["translations"])

    wall_pcd = unproject_points_rotvec(wall["depths"], wall["points"], wall["f"],
                                       wall["pp"], wall["rotvecs"], wall["translations"])

    if "segments" in wall:
        visualize_aligned_walls(wall, save_path)

    red = np.array([1, 0, 0])
    green = np.array([0, 1, 0])
    blue = np.array([0, 0, 1])
    left_pcd = make_pcd_from_numpy(left_pcd, red)
    right_pcd = make_pcd_from_numpy(right_pcd, green)
    wall_pcd = make_pcd_from_numpy(wall_pcd, red)
    floor_pcd = make_pcd_from_numpy(floor_pcd, blue)
    line_set = make_line_set(left_pcd, right_pcd, green)

    o3d.io.write_line_set(f"{save_path}/keypoints.ply", line_set)
    o3d.io.write_point_cloud(f"{save_path}/wall.ply", wall_pcd)
    o3d.io.write_point_cloud(f"{save_path}/floor.ply", floor_pcd)
