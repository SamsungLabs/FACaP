import numpy as np
import torch

import open3d as o3d
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import torch

from torch import nn


def nearest_segment_distance(floorplan, pcd_2d):
    biases = torch.stack([i[:2] for i in floorplan])
    directions = torch.stack([i[2:] - i[:2] for i in floorplan])

    distances = []

    for a, b in zip(directions, biases):
        c = pcd_2d - b
        dot_product = torch.sum(c * a, dim=-1) / a.dot(a)

        par_dist = torch.zeros_like(dot_product)
        par_dist[dot_product < 0] = - dot_product[dot_product < 0] * torch.norm(a)
        par_dist[dot_product > 1.] = (dot_product[dot_product > 1.] - 1) * torch.norm(a)
        ort_dst = torch.abs(c[:, 0] * a[1] - c[:, 1] * a[0]) / torch.norm(a)
        distances.append(torch.sqrt(par_dist ** 2 + ort_dst ** 2))

    distances = torch.stack(distances)
    distances = torch.min(distances, dim=0).values
    return distances


class WallSegmentTerm(nn.Module):
    def __init__(self, wall, unproject, distance_function, floorplan, device):
        super(WallTerm, self).__init__()
        self.wall = wall
        self.unproject = unproject
        self.distance_function = distance_function
        self.register_buffer("floorplan", floorplan)
        self.alligned_walls

    def forward(self):
        wall_pcd = self.unproject(self.wall["depths"], self.wall["points"], self.wall["camera_idxs"])
        distances = nearest_segment_distance(self.floorplan, wall_pcd[:, [0, 2]])
        zeros = torch.zeros_like(distances)

        return self.distance_function(zeros, distances)


def visualize_pcd(pcd, color='g', size=2, label=None):
    if isinstance(pcd, o3d.open3d.geometry.TriangleMesh):
        points = np.array(pcd.vertices)
    elif isinstance(pcd, o3d.open3d.geometry.PointCloud):
        points = np.array(pcd.points)
    elif isinstance(pcd, np.ndarray):
        shape = pcd.shape[-2:]
        points = pcd.reshape(shape)
    elif isinstance(pcd, torch.tensor):
        shape = pcd.shape[-2:]
        points = pcd.detach().cpu().numpy().reshape(shape)

    plt.plot(points[..., 2], points[..., 0], '.', c=color, ms=size, label=label)
    if label:
        plt.legend()


class WallAligner:
    def __init__(self, floorplan, walls_pcd, transformations=None):
        self.floorplan = floorplan
        self.walls_pcd = walls_pcd
        self.walls_pcd.estimate_normals()
        self.walls = []
        self.reper = []
        if transformations is None:
            self.transformations = []
        else:
            self.transformations = transformations
        self.fp_walls = []

    def align_walls(self):

        for t in self.transformations:
            self.floorplan.rotate(t, center=False)
            self.walls_pcd.rotate(t, center=False)
        norm_proj = np.asarray(self.walls_pcd.normals)[..., [0, 2]][
            abs(np.asarray(self.walls_pcd.normals)[..., 1]) < 0.3]
        counts, bins, _ = (plt.hist((np.arctan(norm_proj[..., 1] / (1e-10 + norm_proj[..., 0]))), bins=360))
        plt.show()
        peaks = find_peaks(counts, height=300, distance=50, prominence=30)
        print(peaks[0])
        for peak in peaks[0]:
            print(peak)
            alpha = (bins[peak] + bins[peak + 1]) / 2
            filtered_by_normals = self.walls_pcd.select_down_sample(
                np.where(abs(np.asarray(self.walls_pcd.normals)[..., 1]) < 0.3)[0]
            ).select_down_sample(
                np.where(abs((np.arctan(norm_proj[..., 1] / (norm_proj[..., 0] + 1e-10))) - alpha) < .1)[0]
            )
            self.get_distinct_walls(filtered_by_normals, alpha)

        self.walls = self.unite_walls(self.walls, iou_thr=0.7)
        self.fp_walls = self.get_fp_walls()
        self.fp_walls = self.unite_walls(self.fp_walls, colin_thr=0.9)

        print('aligning')
        points2planes = []
        for i, wall in enumerate(self.walls):
            plane_dists = [np.array(wall.compute_point_cloud_distance(fp_wall))
                           for fp_wall in self.fp_walls]
            close_planes = [np.mean(dist) for dist in plane_dists]
            closest_plane = np.argmin(close_planes)
            wall_dists = [np.array(self.fp_walls[closest_plane].compute_point_cloud_distance(wall))
                          for wall in self.walls]
            close_walls = [np.mean(dist) for dist in wall_dists]

            if i in np.argsort(close_walls)[:3]:

                plane1, _ = wall.segment_plane(0.01, 4, 1000)
                plane2, _ = self.fp_walls[closest_plane].segment_plane(0.01, 4, 1000)
                if plane1[:3].dot(plane2[:3]) > .8:
                    corresp_plane = self.fp_walls[closest_plane]
                    for t in self.transformations[::-1]:
                        wall.rotate(np.linalg.inv(t), center=False)
                        corresp_plane.rotate(np.linalg.inv(t), center=False)
                    real_plane, _ = corresp_plane.segment_plane(0.01, 4, 1000)
                    points2planes.append((wall, corresp_plane, real_plane))
                else:
                    print('non-parallel')
        return points2planes

    def get_distinct_walls(self, pcd, alpha):
        print('getting distinct walls')
        cur_pcd = pcd
        for i in range(20000):
            if len(cur_pcd.points) < 500:
                break
            plane_model, inliers = cur_pcd.segment_plane(distance_threshold=0.01,
                                                         ransac_n=10,
                                                         num_iterations=100)
            [a, b, c, d] = plane_model
            cur_dir = plane_model[:3]
            if b > 0.3:
                continue

            if abs(alpha - np.arctan(c / a)) > .2:
                continue

            self.reper.append((plane_model, len(inliers)))
            self.walls.append(cur_pcd.select_down_sample(inliers))

            cur_pcd = cur_pcd.select_down_sample(inliers, invert=True)

    #     plt.show()

    def unite_walls(self, walls, iou_thr=0.8, colin_thr=0.8):
        print('uniting walls')
        united_walls = {}
        for i, wall in tqdm(enumerate(walls)):
            new_id = -1
            for wall_id in united_walls:
                for w in united_walls[wall_id]:
                    dist_1 = np.array(wall.compute_point_cloud_distance(w))
                    dist_2 = np.array(w.compute_point_cloud_distance(wall))
                    plane_model_1, _ = wall.segment_plane(distance_threshold=0.1, ransac_n=10, num_iterations=100)
                    plane_model_2, _ = w.segment_plane(distance_threshold=0.1, ransac_n=10, num_iterations=100)
                    if ((dist_1 < .2).mean() > iou_thr or (dist_2 < .2).mean() > iou_thr) and abs(
                            plane_model_1[:3].dot(plane_model_2[:3])) > colin_thr:
                        new_wall = False
                        new_id = wall_id

            if len(united_walls) == 0:
                new_id = 0
            elif new_id == -1:
                new_id = max(list(united_walls.keys())) + 1
            if new_id not in united_walls:
                united_walls[new_id] = []
            united_walls[new_id].append(wall)

        walls = []
        for wall_id in united_walls:

            all_points = []
            all_normals = []
            for w in united_walls[wall_id]:
                all_points += np.array(w.points).tolist()
                all_normals += np.array(w.normals).tolist()
            new_wall = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
            new_wall.normals = o3d.utility.Vector3dVector(all_normals)
            walls.append(new_wall)
        return walls

    def get_fp_walls(self):
        print('getting floorplan walls')
        reper = []
        colored_pcd = []

        fp_walls = []

        cur_pcd = self.floorplan
        while len(cur_pcd.points) > 200:
            plane_model, inliers = cur_pcd.segment_plane(distance_threshold=0.02,
                                                         ransac_n=10,
                                                         num_iterations=100)

            [a, b, c, d] = plane_model
            cur_dir = plane_model[:3]
            new_dir = True
            if len(reper) == 0:
                new_dir = True
            if b > 0.90:
                continue

            if new_dir:
                if len(inliers) > 150:
                    color = np.random.rand(3)
                    colored_pcd += list((np.hstack((np.array(cur_pcd.points)[inliers],
                                                    plane_model.repeat(len(inliers)).reshape(4, -1).T,
                                                    color.repeat(len(inliers)).reshape(3, -1).T))))

                    reper.append(plane_model)
                    fp_walls.append(cur_pcd.select_down_sample(inliers))

                cur_pcd = cur_pcd.select_down_sample(inliers, invert=True)
        print(len(reper))
        return fp_walls
