import argparse
import torch
import os
import open3d as o3d
import numpy as np

import yaml
from torch import nn
from torch import optim

from facap import feature_errors
from facap.data.scan import Scan
from facap.optimization import Project, Unproject, CameraParameters, FloorTerm, WallTerm
from facap.utils import dicts_to_torch

if __name__ == '__main__':
    print("anton")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--device", default='cuda:0', help="Device to run")
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    scan = Scan(cfg["paths"]["scan_path"], wall_sparsity=cfg["data"]["wall_sparsity"],
                floor_sparsity=cfg["data"]["floor_sparsity"], scale=cfg["data"]["depths_scale"], cut_frames=100)
    save_path = cfg["paths"]["save_path"]
    os.makedirs(save_path, exist_ok=True)
    o3d.io.write_triangle_mesh(f"{save_path}/source_mesh.ply", scan.make_mesh())

    data = scan.generate_ba_data(min_frame_difference=cfg["data"]["min_frame_difference"],
                                 max_initial_distance=cfg["data"]["max_initial_distance"],
                                 floor_percentiles=cfg["data"]["floor_percentiles"]
                                 )

    dicts_to_torch(data, args.device)
    left, right, wall, floor = data

    camera_parameters = CameraParameters(scan.cameras).to(args.device).float()
    unproject = Unproject(camera_parameters, scale=scan.scale)
    project = Project(camera_parameters)
    cost_function = nn.MSELoss()

    if cfg["error"]["floor_term"]:
        floor_function = FloorTerm(floor, unproject, cost_function)

    if cfg["error"]["wall_term"]:
        scan_path = cfg["paths"]["scan_path"]
        floorplan = torch.from_numpy(np.load(f"{scan_path}/floorplan.npy"))
        wall_function = WallTerm(wall, unproject, cost_function, floorplan).to(args.device).float()

    params = []

    fixed_cameras = [scan._frames[i] for i in cfg["optimization"]["fixed_cameras_idx"]]
    for name, param in camera_parameters.named_parameters():
        if name.split(".")[-1] not in fixed_cameras:
            params.append(param)

    optimizer = optim.SGD(params, lr=cfg["optimization"]["lr"], momentum=cfg["optimization"]["momentum"])

    for epoch in range(cfg["optimization"]["num_epoches"]):
        optimizer.zero_grad()
        error_args = {"unproject": unproject,
                      "project": project,
                      "scale": scan.scale,
                      "distance_function": cost_function,
                      **cfg["error"]}
        ba_function = getattr(feature_errors, cfg["error"]["error_type"])
        ba_term = ba_function(left, right, **error_args)

        floor_term = 0.
        wall_term = 0.
        print(f"The value of the loss function on the {epoch}-iteration")
        print(f"\t\t feature-based BA term - {float(ba_term)}")

        if cfg["error"]["floor_term"]:
            floor_term = floor_function() * cfg["error"]["floor_weight"]
            print(f"\t\t floor term - {float(floor_term)}")

        if cfg["error"]["wall_term"]:
            floor_term = wall_function() * cfg["error"]["wall_weight"]
            print(f"\t\t wall term - {float(floor_term)}")

        loss = ba_term + wall_term + floor_term
        loss.backward()
        optimizer.step()

    torch.save(camera_parameters.state_dict(), f"{save_path}/cameras.pth")
    cameras = camera_parameters.get_cameras()
    scan.set_cameras(cameras)
    o3d.io.write_triangle_mesh(f"{save_path}/processed_mesh.ply", scan.make_mesh())

# import pytorch3d
#
# from pytorch3d.ops import knn_points
# from os.path import join
# import os
# from matplotlib import pyplot as plt
# import cv2
# from tqdm import tqdm
# import open3d as o3d
# import numpy as np
# from pytorch3d.ops import knn_points
# from scipy.spatial.transform import Rotation as R
# import torch
# from torch import optim
#
# from third_party.colmap_scripts.read_write_model import (
#     qvec2rotmat,
#     read_images_binary,
#     read_points3D_binary,
#     read_cameras_binary,
#     read_model, write_model,
#     Point3D, BaseImage)
#
# from facap.geometry.open3d import make_pcd_from_numpy, make_line_set
# from facap.geometry.numpy import unproject_points_rotvec
#
# from facap.data.scan import Scan
#
# scene_path = "./data/htkr"
#
# scan = Scan(scene_path, wall_sparsity=30, floor_sparsity=30)
#
# left, right, wall, floor = scan.generate_ba_data()


# left_pcd = unproject_points_rotvec(left["depths"], left["points"], left["f"],
#                                    left["pp"], left["rotvecs"], left["translations"])
#
# right_pcd = unproject_points_rotvec(right["depths"], right["points"], right["f"],
#                                     right["pp"], right["rotvecs"], right["translations"])
#
# floor_pcd = unproject_points_rotvec(floor["depths"], floor["points"], floor["f"],
#                                     floor["pp"], floor["rotvecs"], floor["translations"])
#
# wall_pcd = unproject_points_rotvec(wall["depths"], wall["points"], wall["f"],
#                                    wall["pp"], wall["rotvecs"], wall["translations"])
#
# red = np.array([1, 0, 0])
# green = np.array([0, 1, 0])
# blue = np.array([0, 0, 1])
# left_pcd = make_pcd_from_numpy(left_pcd, red)
# right_pcd = make_pcd_from_numpy(right_pcd, green)
# wall_pcd = make_pcd_from_numpy(wall_pcd, red)
# floor_pcd = make_pcd_from_numpy(floor_pcd, blue)
# line_set = make_line_set(left_pcd, right_pcd, green)
#
# o3d.io.write_line_set("keypoints.ply", line_set)
# o3d.io.write_point_cloud("wall.ply", wall_pcd)
# o3d.io.write_point_cloud("floor.ply", floor_pcd)
