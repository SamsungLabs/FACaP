import numpy as np

from copy import deepcopy

import open3d as o3d

from scipy.signal import find_peaks
from tqdm import tqdm

from facap.geometry.open3d import make_pcd_from_numpy
from facap.geometry.numpy import unproject_points_rotvec


def apply_mask(data, mask):
    for key in data:
        data[key] = data[key][mask]


def wall_floorplan_distance(floorplan, wall):
    biases = np.stack([i[:2] for i in floorplan])
    directions = np.stack([i[2:] - i[:2] for i in floorplan])

    distances = []

    for a, b in zip(directions, biases):
        c = wall - b
        dot_product = np.sum(c * a, axis=-1) / a.dot(a)

        par_dist = np.zeros_like(dot_product)
        par_dist[dot_product < 0] = - dot_product[dot_product < 0] * np.linalg.norm(a)
        par_dist[dot_product > 1.] = (dot_product[dot_product > 1.] - 1) * np.linalg.norm(a)
        ort_dst = np.abs(c[:, 0] * a[1] - c[:, 1] * a[0]) / np.linalg.norm(a)
        distances.append(np.sqrt(par_dist ** 2 + ort_dst ** 2))

    distances = np.stack(distances)
    distances = np.mean(distances, axis=1)
    segment_idx = np.argmin(distances, axis=0)

    return segment_idx, distances[segment_idx]


def get_distinct_walls(pcd, indexes, alpha,
                       max_vertical_component=.3,
                       min_num_points=500,
                       num_ransac_tries=20000,
                       max_collinearity_rate=.2):
    walls = []
    cur_pcd = deepcopy(pcd)
    for i in range(num_ransac_tries):
        if len(cur_pcd.points) >= min_num_points:
            plane_model, inliers = cur_pcd.segment_plane(distance_threshold=0.01,
                                                         ransac_n=10,
                                                         num_iterations=100)
            a, b, c, _ = plane_model

            if (b <= max_vertical_component) and (abs(alpha - np.arctan(c / a)) <= max_collinearity_rate):
                wall_pcd = cur_pcd.select_by_index(inliers)
                wall = {"plane_model": plane_model,
                        "pcd": wall_pcd,
                        "indexes": indexes[inliers]}

                cur_pcd = cur_pcd.select_by_index(inliers, invert=True)
                indexes = np.delete(indexes, inliers)
                walls.append(wall)
    return walls


def join_walls(walls):
    wall = o3d.geometry.PointCloud()
    plane_models = []
    indexes = []
    for w in walls:
        wall += w["pcd"]
        plane_models.append(w["plane_model"])
        indexes.append(w["indexes"])
    plane_model = np.array(plane_models).mean(axis=-1).tolist()
    indexes = np.concatenate(indexes)
    wall = {"pcd": wall, "plane_model": plane_model, "indexes": indexes}
    return wall


def unite_walls(walls, iou_thr=0.8, colin_thr=0.8):
    united_walls = {}
    for i, wall in tqdm(enumerate(walls)):
        new_id = -1
        for wall_id in united_walls:
            for w in united_walls[wall_id]:
                dist_1 = np.array(wall["pcd"].compute_point_cloud_distance(w["pcd"]))
                dist_2 = np.array(w["pcd"].compute_point_cloud_distance(wall["pcd"]))
                plane_model_1 = wall["plane_model"]
                plane_model_2 = w["plane_model"]
                if ((dist_1 < .2).mean() > iou_thr or (dist_2 < .2).mean() > iou_thr) and abs(
                        plane_model_1[:3].dot(plane_model_2[:3])) > colin_thr:
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
        wall = join_walls(united_walls[wall_id])
        walls.append(wall)
    return walls


def reorder_wall(wall, segment2wall, floorplan):
    indexes = []
    segments = []
    for segment_id, w in segment2wall.items():
        indexes.append(w["indexes"])
        segments.append(np.stack([floorplan[segment_id]] * len(w["indexes"])))
    indexes = np.concatenate(indexes)
    segments = np.concatenate(segments)

    for key in wall:
        wall[key] = wall[key][indexes]
    wall["segments"] = segments
    return wall


def align_walls(wall, floorplan):
    walls_pcd = unproject_points_rotvec(wall["depths"], wall["points"], wall["f"], wall["pp"],
                                        wall["rotvecs"], wall["translations"])
    walls_pcd = make_pcd_from_numpy(walls_pcd, np.array([0, 1, 0]))
    indexes = np.arange(len(walls_pcd.points))
    normals = np.asarray(walls_pcd.normals)
    vertical_component_condition = np.abs(normals[..., 1]) < 0.3
    norm_proj = normals[..., [0, 2]][vertical_component_condition]
    counts, bins = np.histogram(np.arctan(norm_proj[..., 1] / (1e-10 + norm_proj[..., 0])), bins=360)
    peaks = find_peaks(counts, height=300, distance=50, prominence=30)

    walls = []
    for peak in peaks[0]:
        alpha = (bins[peak] + bins[peak + 1]) / 2

        angle_condition = abs((np.arctan(normals[..., 1] / (normals[..., 0] + 1e-10))) - alpha) < .1
        condition = vertical_component_condition & angle_condition
        mask = np.where(condition)[0]
        filtered_by_normals = walls_pcd.select_by_index(mask)
        filtered_indexes = indexes[mask]

        walls.extend(get_distinct_walls(filtered_by_normals, filtered_indexes, alpha))

    walls = unite_walls(walls, iou_thr=0.7)

    print('aligning')
    plane2walls = {}
    for i, w in enumerate(walls):
        wall_pcd = np.asarray(w["pcd"].points)[:, [0, 2]]
        idx, _ = wall_floorplan_distance(floorplan, wall_pcd)
        if idx not in plane2walls:
            plane2walls[idx] = [w, ]
        else:
            plane2walls[idx].append(w)

    for segment_id in plane2walls:
        plane2walls[segment_id] = join_walls(plane2walls[segment_id])
    wall = reorder_wall(wall, plane2walls, floorplan)
    return wall
