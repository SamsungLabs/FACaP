import numpy as np

import open3d as o3d
from open3d import visualization


def project_points(pcd, extrinsic, f, pp, width, height):
    """
    Projects a given depth map into a depth map with Open3D.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Source point cloud.
    f : tuple
        Focus (fx, fy).
    pp : tuple
        The principal point (cx, cy).
    extrinsic : numpy.ndarray
        The 4x4  extrinsic matrix.
    width : int
        Width of the map.
    height : int
        Height of the map.

    Returns
    -------
    numpy.ndarray
        Depth map.

    """
    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic.set_intrinsics(width, height, f[0], f[1], pp[0] - 0.5, pp[1] - 0.5)
    camera.extrinsic = extrinsic
    viewer = visualization.Visualizer()
    viewer.create_window(width=width, height=height)
    viewer.add_geometry(pcd)
    viewer.get_view_control().convert_from_pinhole_camera_parameters(camera)
    viewer.poll_events()
    viewer.update_renderer()
    depth = viewer.capture_depth_float_buffer()
    viewer.destroy_window()
    depth = (np.asarray(depth).astype(np.float32) * 1000).astype(np.uint16)
    return depth


def unproject_points(depth_map, color_map, extrinsic, f, pp, width, height):
    """
    Unprojects a given depth map into a point cloud using Open3D.

    Parameters
    ----------
    depth_map : numpy.ndarray
        Source depth map.
    color_map : numpy.ndarray
        Source rgb image.
    f : tuple
        Focus (fx, fy).
    pp : tuple
        The principal point (cx, cy).
    extrinsic : numpy.ndarray
        The 4x4  extrinsic matrix.
    width : int
        Width of the map.
    height : int
        Height of the map.

    Returns
    -------
    open3d.geometry.PointCloud
        The point cloud constructed from the given data.

    """

    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.set_intrinsics(width, height, f[0], f[1], pp[0], pp[1])
    color = o3d.geometry.Image(color_map)
    depth = o3d.geometry.Image(depth_map)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                                                              convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera, extrinsic)
    return pcd


def sample_points_from_pcd(pcd, num_points):
    new_pcd = o3d.geometry.PointCloud()
    xyz = np.asarray(pcd.points)
    color = np.asarray(pcd.colors)

    indexes = np.random.randint(0, len(xyz), num_points)

    xyz = xyz[indexes]
    color = color[indexes]

    new_pcd.points = o3d.utility.Vector3dVector(xyz)
    new_pcd.colors = o3d.utility.Vector3dVector(color)
    return new_pcd


def make_pcd_from_numpy(xyz, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if len(colors.shape) == 1:
        colors = np.ones_like(xyz) * colors
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    return pcd


def make_line_set(pcd1, pcd2, colors=np.array([1, 0, 0]), num_points=None):
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    if num_points is not None:
        indexes = np.random.randint(0, len(points1), num_points)

        points1 = points1[indexes]
        points2 = points2[indexes]

    lines = o3d.geometry.LineSet()
    points = np.r_[points1, points2]
    lines.points = o3d.utility.Vector3dVector(points)

    size = len(points) // 2
    l = np.c_[np.arange(size), np.arange(size) + size]
    lines.lines = o3d.utility.Vector2iVector(l)
    if len(colors.shape) == 1:
        colors = np.ones((size, 3)) * colors
    lines.colors = o3d.utility.Vector3dVector(colors)

    return lines
