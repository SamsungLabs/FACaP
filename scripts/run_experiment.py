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
from facap.optimization import Project, Unproject, CameraParameters, FloorTerm, WallTerm, WallSegmentTerm
from facap.utils import dicts_to_torch, visualize_data
from facap.geometry.allign_walls import align_walls

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--device", default='cuda:0', help="Device to run")
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    scan_path = cfg["paths"]["scan_path"]
    scan = Scan(scan_path, scale=cfg["data"]["depths_scale"])
    save_path = cfg["paths"]["save_path"]
    os.makedirs(save_path, exist_ok=True)
    o3d.io.write_triangle_mesh(f"{save_path}/source_mesh.ply", scan.make_mesh())

    data = scan.generate_ba_data(min_frame_difference=cfg["data"]["min_frame_difference"],
                                 max_initial_distance=cfg["data"]["max_initial_distance"],
                                 floor_percentiles=cfg["data"]["floor_percentiles"],
                                 wall_sparsity=cfg["data"]["wall_sparsity"],
                                 floor_sparsity=cfg["data"]["floor_sparsity"]
                                 )

    if "wall_term_type" in cfg["error"] and cfg["error"]["wall_term_type"] == "segment":
        floorplan = torch.from_numpy(np.load(f"{scan_path}/floorplan.npy"))
        alligned_walls = align_walls(data[2], floorplan)
        data = (data[0], data[1], alligned_walls, data[3])
    visualize_data(data, save_path=save_path)
    dicts_to_torch(data, args.device)
    left, right, wall, floor = data

    camera_parameters = CameraParameters(scan.cameras).to(args.device).float()
    unproject = Unproject(camera_parameters, scale=scan.scale)
    project = Project(camera_parameters)
    cost_function = nn.MSELoss()

    if cfg["error"]["floor_term"]:
        floor_function = FloorTerm(floor, unproject, cost_function)

    if cfg["error"]["wall_term"]:
        floorplan = torch.from_numpy(np.load(f"{scan_path}/floorplan.npy"))
        if cfg["error"]["wall_term_type"] == "point":
            wall_function = WallTerm(wall, unproject, cost_function, floorplan).to(args.device).float()
        else:
            wall_function = WallSegmentTerm(wall, unproject, cost_function, floorplan).to(args.device).float()

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
