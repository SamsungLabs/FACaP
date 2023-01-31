from copy import deepcopy
from glob import glob
from open3d import integration

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from facap.geometry.open3d import unproject_points, sample_points_from_pcd
from facap.geometry.numpy import unproject_points_rotvec
from facap.colmap_scripts.read_write_model import read_model


def read_data(scan_path, frame_id):
    color = cv2.imread(f'{scan_path}/arcore/frame-{frame_id}.png')
    wall = cv2.imread(f'{scan_path}/segmentation/frame-{frame_id}_wall.png', cv2.IMREAD_GRAYSCALE)
    wall = np.rot90(wall > wall.min())
    floor = cv2.imread(f'{scan_path}/segmentation/frame-{frame_id}_floor.png', cv2.IMREAD_GRAYSCALE)
    floor = np.rot90(floor > floor.min())
    depth = cv2.imread(f'{scan_path}/arcore/depth-{frame_id}.png', -1)

    pose = np.loadtxt(f'{scan_path}/arcore/pose-{frame_id}.txt')
    camera_params = np.loadtxt(f'{scan_path}/arcore/cam_params-{frame_id}.txt')

    return color, wall, floor, depth, pose, camera_params


def get_yxds(scan_path, frame_ids, max_depth=3000, min_depth=0):
    yxds = {}
    for frame_id in frame_ids:
        color, _, _, depth, _, _ = read_data(scan_path, frame_id)
        depth_mask = np.where((depth > min_depth) * (depth < max_depth))
        yxds[frame_id] = get_index_value_dict(depth, depth_mask)
    return yxds


def get_segmentation(scan_path, frame_ids, part="floor", sparsity=30, max_depth=3000, min_depth=0):
    result = {}
    for frame_id in frame_ids:
        color, wall, floor, depth, _, _ = read_data(scan_path, frame_id)

        if part == "floor":
            part = floor
        else:
            part = wall

        part_mask = np.where((depth > min_depth) * (depth < max_depth) * (part > 0))
        part_dict = get_index_value_dict(depth, part_mask, sparsity=sparsity)
        result[frame_id] = list(part_dict.items())
    return result


def read_features(scan_path, xyds, frame_ids, min_freq=2):
    scan_path = f"{scan_path}/db" #glob(f'{scan_path}/db/1/triangulated/feat*/')[0]
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

    @classmethod
    def read_camera(cls, scan_path, frame_id):
        pose = np.loadtxt(f'{scan_path}/arcore/pose-{frame_id}.txt')
        camera_params = np.loadtxt(f'{scan_path}/arcore/cam_params-{frame_id}.txt')
        rotvec = R.from_matrix(pose[:3, :3]).as_rotvec()
        translation = pose[:3, 3]
        f = (camera_params[2], camera_params[3])
        pp = (camera_params[4], camera_params[5])
        camera = cls(f, pp, rotvec, translation)
        return camera


class Scan:
    def __init__(self, scan_path, sparsity=1, cut_frames=None, scale=1000):
        frames = sorted(glob(f"{scan_path}/segmentation/frame*_floor*"))
        frame_ids = [i.split("/")[-1][6:-10] for i in frames]
        frame_ids = frame_ids[::sparsity]

        if cut_frames is not None:
            frame_ids = frame_ids[:cut_frames]
        self.scan_path = scan_path
        self._frames = frame_ids
        self.scale = scale
        self.cameras = {frame_id: Camera.read_camera(scan_path, frame_id) for frame_id in frame_ids}

    def get_data(self, cam_id):
        color, wall, floor, depth, pose, camera_params = read_data(self.scan_path, cam_id)
        return color, wall, floor, depth, pose, camera_params

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

        volume = integration.ScalableTSDFVolume(
            voxel_length=vox_length,
            sdf_trunc=vox_length * 4,
            color_type=integration.TSDFVolumeColorType.RGB8)

        camera = o3d.camera.PinholeCameraIntrinsic()

        for cam_id in self._frames:
            color_map, wall, floor, depth_map, pose, camera_params = self.get_data(cam_id)
            depth_map = depth_map.astype(np.float32)
            camera.set_intrinsics(int(camera_params[0]), int(camera_params[1]), *camera_params[2:])
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
                         max_initial_distance=0.4,
                         wall_sparsity=30,
                         floor_sparsity=30):
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

        yxds = get_yxds(self.scan_path, self._frames)
        features = read_features(self.scan_path, yxds, self._frames)

        for point_id in features:
            point = features[point_id]
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
        del yxds
        wall_data = get_segmentation(self.scan_path, self._frames, part="wall", sparsity=wall_sparsity)
        floor_data = get_segmentation(self.scan_path, self._frames, part="floor", sparsity=floor_sparsity)

        for source, target in zip([wall_data, floor_data], [wall, floor]):
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
                target["camera_idxs"].extend([cam_id] * len(source[cam_id]))

        del wall_data, floor_data

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
