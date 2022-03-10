from copy import deepcopy
from glob import glob
from open3d import pipelines

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from facap.geometry.open3d import unproject_points, sample_points_from_pcd
from facap.geometry.numpy import unproject_points_rotvec
from third_party.colmap_scripts.read_write_model import read_model


def read_data(scan_path, frame_id):
    color = cv2.imread(f'{scan_path}/frame-{frame_id}.png')
    wall = cv2.imread(f'{scan_path}/frame-{frame_id}_wall.png', cv2.IMREAD_GRAYSCALE)
    wall = np.rot90(wall > wall.min())
    floor = cv2.imread(f'{scan_path}/frame-{frame_id}_floor.png', cv2.IMREAD_GRAYSCALE)
    floor = np.rot90(floor > floor.min())
    depth = cv2.imread(f'{scan_path}/depth-{frame_id}.png', -1)

    pose = np.loadtxt(f'{scan_path}/pose-{frame_id}.txt')
    camera_params = np.loadtxt(f'{scan_path}/cam_params-{frame_id}.txt')

    return color, wall, floor, depth, pose, camera_params


def read_features(scan_path, xyds, frame_ids, min_freq=2):
    scan_path = glob(f'{scan_path}/db/1/triangulated/feat*/')[0]
    cameras, images, points_3d = read_model(f'{scan_path}/sparse/models/triangulated')

    result = {}
    for point in points_3d:
        result[point] = {}

        for img_id in points_3d[point].image_ids:
            img = images[img_id]
            xy = tuple((img.xys[img.point3D_ids == point] + 0.5).astype(int)[0])
            yx = (xy[1], xy[0])
            frame_id = img.name[6:-4]
            if frame_id in frame_ids:
                if yx in xyds[frame_id]:
                    d = xyds[frame_id][yx]
                    result[point][img.name[6:-4]] = (yx, d)
    filtered_result = {}

    for point in result:
        if len(result[point]) > min_freq:
            filtered_result[point] = result[point]
    return filtered_result


def get_index_value_dict(array_2d, mask, sparsity=1):
    y, x = np.array(mask).astype(int)
    d = array_2d[mask].astype(float)
    x, y, d = x[::sparsity], y[::sparsity], d[::sparsity]
    yxd = dict(zip(list(zip(y, x)), d))
    return yxd


class Camera:
    def __init__(self, f, pp, rotvec, translation):
        self.f = f
        self.pp = pp
        self.rotvec = rotvec.astype(float)
        self.translation = translation.astype(float)


class Scan:
    def __init__(self, scan_path, sparsity=1, cut_frames=None,
                 wall_sparsity=30, floor_sparsity=30, store_frames=False, scale=1000):
        frames = sorted(glob(f"{scan_path}/frame*_floor*"))
        frame_ids = [i.split("/")[-1][6:-10] for i in frames]
        frame_ids = frame_ids[::sparsity]

        if cut_frames is not None:
            frame_ids = frame_ids[:cut_frames]
        self.scan_path = scan_path
        self.wall_sparsity = wall_sparsity
        self.floor_sparsity = floor_sparsity
        self._frames = frame_ids
        self.store_frames = store_frames
        self.scale = scale

        if self.store_frames:
            self.rgbs = {}
            self.depth_maps = {}
        self.yxds = {}
        self.cameras = {}
        self.floor = {}
        self.wall = {}
        self.read_data(scan_path, frame_ids)
        self.features = read_features(scan_path, self.yxds, frame_ids)

        self.frame_points = self.get_frame_points()

    def read_data(self, scan_path, frame_ids, max_depth=3000, min_depth=0):
        for frame_id in frame_ids:
            color, wall, floor, depth, pose, camera_params = read_data(scan_path, frame_id)
            if self.store_frames:
                self.rgbs[frame_id] = color
                self.depth_maps[frame_id] = depth

            depth_mask = np.where((depth > min_depth) * (depth < max_depth))
            self.yxds[frame_id] = get_index_value_dict(depth, depth_mask)

            wall_mask = np.where((depth > min_depth) * (depth < max_depth) * (wall > 0))
            wall_dict = get_index_value_dict(depth, wall_mask, sparsity=self.wall_sparsity)
            self.wall[frame_id] = list(wall_dict.items())

            floor_mask = np.where((depth > min_depth) * (depth < max_depth) * (floor > 0))
            floor_dict = get_index_value_dict(depth, floor_mask, sparsity=self.floor_sparsity)
            self.floor[frame_id] = list(floor_dict.items())

            rotvec = R.from_matrix(pose[:3, :3]).as_rotvec()
            translation = pose[:3, 3]
            f = (camera_params[2], camera_params[3])
            pp = (camera_params[4], camera_params[5])
            camera = Camera(f, pp, rotvec, translation)
            self.cameras[frame_id] = camera

    def get_data(self, cam_id):
        color, wall, floor, depth, pose, camera_params = read_data(self.scan_path, cam_id)
        return color, wall, floor, depth, pose, camera_params

    def get_frame_points(self):
        frame_points = {}
        for point in self.features:
            for img_id in self.features[point]:
                if img_id not in frame_points:
                    frame_points[img_id] = {}
                yx = self.features[point][img_id][0]
                frame_points[img_id][point] = yx
        return frame_points

    def make_pcd(self, num_points=None):
        pcds = []
        for frame_id in self._frames:
            color_map, _, _, depth_map, _, _ = self.get_data(frame_id)
            camera = self.cameras[frame_id]
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R.from_rotvec(camera.rotvec).as_matrix()
            extrinsic[:3, 3] = camera.translation

            pcd = unproject_points(depth_map, color_map, np.linalg.inv(extrinsic),
                                   camera.f, camera.pp, *depth_map.shape)
            pcds.append(pcd)

        pcd_combined = o3d.geometry.PointCloud()

        for pcd in pcds:
            pcd_combined += pcd

        if num_points is not None:
            pcd_combined = sample_points_from_pcd(pcd_combined, num_points)

        return pcd_combined

    def make_mesh(self, vox_length=0.05):

        volume = pipelines.integration.ScalableTSDFVolume(
            voxel_length=vox_length,
            sdf_trunc=vox_length * 4,
            color_type=pipelines.integration.TSDFVolumeColorType.RGB8)

        camera = o3d.camera.PinholeCameraIntrinsic()

        for cam_id in zip(self._frames):
            color_map, wall, floor, depth_map, pose, camera_params = self.get_data(cam_id)
            depth_map = depth_map.astype(np.float32)
            camera.set_intrinsics(*camera_params)
            color = o3d.geometry.Image(color_map)
            depth = o3d.geometry.Image(depth_map)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_trunc=6., convert_rgb_to_intensity=False)

            volume.integrate(rgbd, camera, np.linalg.inv(pose))
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def set_cameras(self, cameras):
        self.cameras = cameras

    def generate_ba_data(self, min_frame_difference=3,
                         floor_percentiles=(2, 90),
                         max_initial_distance=0.4):
        left = {"points": [],
                "depths": [],
                "camera_idxs": [],
                "rotvecs": [],
                "translations": [],
                "f": [],
                "pp": []}
        right = deepcopy(left)
        floor = deepcopy(left)
        wall = deepcopy(left)

        for point_id in self.features:
            point = self.features[point_id]
            cams = list(point.keys())
            cam_params = [self.cameras[i] for i in cams]
            rotvecs = [i.rotvec for i in cam_params]
            translations = [i.translation for i in cam_params]

            for i, cam_i in enumerate(cams):
                for j in range(i + 1, len(cams)):
                    if int(cam_i) - int(cams[j]) >= min_frame_difference:
                        for part, idx, cam_idx in zip([left, right], [i, j], [cam_i, cams[j]]):
                            part["points"].append(point[cam_idx][0])
                            part["depths"].append(point[cam_idx][1])
                            part["rotvecs"].append(rotvecs[idx])
                            part["translations"].append(translations[idx])
                            part["f"].append(cam_params[idx].f)
                            part["pp"].append(cam_params[idx].pp)
                            part["camera_idxs"].append(cam_idx)

        for source, target in zip([self.wall, self.floor], [wall, floor]):
            for cam_id in source:
                camera = self.cameras[cam_id]
                rotvec = camera.rotvec
                translation = camera.translation
                f = camera.f
                pp = camera.pp
                for point in source[cam_id]:
                    target["points"].append(point[0])
                    target["depths"].append(point[1])
                target["rotvecs"].extend([rotvec] * len(source[cam_id]))
                target["translations"].extend([translation] * len(source[cam_id]))
                target["f"].extend([f] * len(source[cam_id]))
                target["pp"].extend([pp] * len(source[cam_id]))
                target["camera_idxs"].extend([cam_id]*len(source[cam_id]))

        for part in [left, right, wall, floor]:
            for key in part:
                part[key] = np.array(part[key])

        def apply_mask(dct, mask):
            for key in dct:
                dct[key] = dct[key][mask]

        def unproject(part):
            return unproject_points_rotvec(part["depths"], part["points"], part["f"],
                                           part["pp"], part["rotvecs"], part["translations"], scale=self.scale)
        keypoint_mask = np.linalg.norm(unproject(left) - unproject(right), axis=-1) < max_initial_distance
        apply_mask(left, keypoint_mask)
        apply_mask(right, keypoint_mask)
        floor_pcd_vert = unproject(floor)[:, 1]
        floor_mask = (floor_pcd_vert > np.percentile(floor_pcd_vert, floor_percentiles[0])) & \
                     (floor_pcd_vert < np.percentile(floor_pcd_vert, floor_percentiles[1]))
        apply_mask(floor, floor_mask)
        return left, right, wall, floor
