import pytorch3d

from pytorch3d.ops import knn_points
from os.path import join
import os
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
import open3d as o3d
import numpy as np
from pytorch3d.ops import knn_points
from scipy.spatial.transform import Rotation as R
import torch
from torch import optim

from third_party.colmap_scripts.read_write_model import (
    qvec2rotmat,
    read_images_binary,
    read_points3D_binary,
    read_cameras_binary,
    read_model, write_model,
    Point3D, BaseImage)

from facap.geometry.open3d import make_pcd_from_numpy, make_line_set
from facap.geometry.numpy import unproject_points_rotvec

from facap.data.scan import Scan

scene_path = "./data/htkr"

scan = Scan(scene_path, wall_sparsity=30, floor_sparsity=30)

left, right, wall, floor = scan.generate_ba_data()


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
