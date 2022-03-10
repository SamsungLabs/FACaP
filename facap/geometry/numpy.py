import numpy as np


def project_points(pcd, f, pp, extrinsic, height, width, scale=1000):
    """
    Projects a given point cloud to 2d according to camera's pose.

    Parameters
    ----------
    pcd : numpy.ndarray
        Source point cloud.
    f : tuple
        Focus (fx, fy).
    pp : tuple
        The principal point (cx, cy).
    extrinsic : numpy.ndarray
        The 4x4 extrinsic matrix.
    scale : float, optional (default=1000)
        The scale for the map.
    width : int
        Width of the map.
    height : int
        Height of the map.

    Returns
    -------
    numpy.ndarray
        2d depth map.

    """

    depth_map = np.zeros((height, width))
    intrinsic = np.array([[f[0], 0, pp[0]],
                          [0, f[1], pp[1]],
                          [0, 0, 1]])
    proj_mat = np.dot(intrinsic, extrinsic[:3])
    pcd_xyz1 = np.pad(pcd, [[0, 0], [0, 1]], 'constant', constant_values=1).T
    p_uv_1 = np.dot(proj_mat, pcd_xyz1)
    p_uv = np.divide(p_uv_1, p_uv_1[2])[:2]
    pos_x = np.array(list(map(lambda x: round(x), p_uv[0])))
    pos_y = np.array(list(map(lambda y: round(y), p_uv[1])))
    valid_ids = []
    for i, (x, y) in enumerate(zip(pos_x, pos_y)):
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        if p_uv_1[2][i] < 0 or p_uv_1[2][i] > 3.:
            continue
        valid_ids.append(i)
    valid_ids = np.array(valid_ids)
    if len(valid_ids) > 0:
        depth_map[pos_y[valid_ids], pos_x[valid_ids]] = np.maximum(0, p_uv_1[2][valid_ids]) * (
                p_uv_1[2][valid_ids] < 3.)
    depth_map *= scale
    return depth_map, valid_ids


def unproject_points(depth_map, coords_2d, f, pp, inv_extrinsic, scale=1000):
    """
    Unprojects a given depth map into a point cloud with inverse extrinsics.

    Parameters
    ----------
    depth_map : numpy.ndarray
        Source depth map.
    coords_2d : numpy.ndarray
        Coordinates of using points on the map.
    f : tuple
        Focus (fx, fy).
    pp : tuple
        The principal point (cx, cy).
    inv_extrinsic : numpy.ndarray
        The 4x4 matrix with the pose of the camera.
    scale : float, optional (default=1000)
        The scale for the map.

    Returns
    -------
    numpy.ndarray
        3d coordinates of the given points.

    """
    u, v = np.array(list(zip(coords_2d)))[:, 0, :]
    z_cam = depth_map / scale
    if isinstance(f, tuple) or len(f.shape) == 1:
        x_cam = z_cam * (v - pp[0]) / f[0]
        y_cam = z_cam * (u - pp[1]) / f[1]
    else:
        x_cam = z_cam * (v - pp[:, 0]) / f[:, 0]
        y_cam = z_cam * (u - pp[:, 1]) / f[:, 1]
    # x_cam = z_cam * (v - pp[0]) / f[0]
    # y_cam = z_cam * (u - pp[1]) / f[1]

    coords_cam = np.stack((x_cam, y_cam, z_cam, np.ones_like(z_cam)))
    coords_world = inv_extrinsic @ coords_cam.T[..., None]

    return coords_world[:, :3, 0]


def unproject_points_rotvec(depths, yxs, f, pp, rotvec, translation, scale=1000):
    """
    Unprojects a given depth map into a point cloud with inverse extrinsics.

    Parameters
    ----------
    depths : numpy.ndarray
         Array of source depths.
    yxs : numpy.ndarray
        Coordinates of points which are corresponding to depths.
    f : tuple
        Focus (fx, fy).
    pp : tuole
        The principal point (cx, cy).
    rotvec : numpy.ndarray
        Rotation vectors.
    translation : numpy.ndarray
        Translations.
    scale : float, optional (default=1000)
        The scale for the map.

    Returns
    -------
    numpy.ndarray
        3d coordinates of the given points.

    """

    u, v = yxs[:, 0], yxs[:, 1]
    z_cam = depths / scale
    if isinstance(f, tuple) or len(f.shape) == 1:
        x_cam = z_cam * (v - pp[0]) / f[0]
        y_cam = z_cam * (u - pp[1]) / f[1]
    else:
        x_cam = z_cam * (v - pp[:, 0]) / f[:, 0]
        y_cam = z_cam * (u - pp[:, 1]) / f[:, 1]
    # x_cam = z_cam * (v - pp[0]) / f[0]
    # y_cam = z_cam * (u - pp[1]) / f[1]

    coords_cam = np.stack((x_cam, y_cam, z_cam)).T
    if len(rotvec.shape) == 1:
        coords_world = rotate(coords_cam, rotvec[None]) + translation
    else:
        coords_world = rotate(coords_cam, rotvec) + translation

    return coords_world


def rotate(points, rotvec):
    """
    Rotates the given point cloud.

    Parameters
    ----------

    rotvec : numpy.ndarray
        Rotation vectors.
    points : numpy.ndarray
        3d point cloud.

    Returns
    -------
    numpy.ndarray
        Rotated point cloud.

    """

    theta = np.linalg.norm(rotvec, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rotvec / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
