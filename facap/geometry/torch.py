import torch


def unproject_points_rotvec(depths, yxs, f, pp, rotvec, translation, scale=1000):
    """
    Unprojects a given depth map into a point cloud with inverse extrinsics.

    Parameters
    ----------
    depths : torch.Tensor
        Array of source depths.
    yxs : torch.Tensor
        Coordinates of points which are corresponding to depths.
    f : torch.Tensor
        Focus with shape (2,) and (N, 2).
    pp : torch.Tensor
        The principal point with shape (2,) and (N, 2).
    rotvec : torch.Tensor
        Rotation vectors.
    translation : torch.Tensor
        Translations.
    scale : float, optional (default=1000)
        The scale for the map.

    Returns
    -------
    torch.Tensor
        3d coordinates of the given points.

    """
    u, v = yxs[:, 0], yxs[:, 1]
    z_cam = torch.divide(depths, scale)
    if isinstance(f, tuple) or len(f.shape) == 1:
        x_cam = z_cam * (v - pp[0]) / f[0]
        y_cam = z_cam * (u - pp[1]) / f[1]
    else:
        x_cam = z_cam * (v - pp[:, 0]) / f[:, 0]
        y_cam = z_cam * (u - pp[:, 1]) / f[:, 1]

    coords_cam = torch.stack((x_cam, y_cam, z_cam)).T
    coords_world = rotate(coords_cam, rotvec) + translation
    return coords_world


def project_points_rotvec(coords_world, f, pp, rotvec, translation):
    """
    Unprojects a given depth map into a point cloud with inverse extrinsics.

    Parameters
    ----------
    depths : torch.Tensor
        Array of source depths.
    yxs : torch.Tensor
        Coordinates of points which are corresponding to depths.
    f : torch.Tensor
        Focus with shape (2,) and (N, 2).
    pp : torch.Tensor
        The principal point with shape (2,) and (N, 2).
    rotvec : torch.Tensor
        Rotation vectors.
    translation : torch.Tensor
        Translations.
    scale : float, optional (default=1000)
        The scale for the map.

    Returns
    -------
    torch.Tensor
        3d coordinates of the given points.

    """
    coords_cam = rotate(coords_world - translation, -rotvec)
    x_cam, y_cam, z_cam = coords_cam[:, 0], coords_cam[:, 1], coords_cam[:, 2]
    if isinstance(f, tuple) or len(f.shape) == 1:
        v = torch.divide(x_cam, z_cam) * f[0] + pp[0]
        u = torch.divide(y_cam, z_cam) * f[1] + pp[1]
    else:
        v = torch.divide(x_cam, z_cam) * f[:, 0] + pp[:, 0]
        u = torch.divide(y_cam, z_cam) * f[:, 1] + pp[:, 1]
    yxs = torch.stack((u, v)).T
    return yxs


def rotate(points, rotvec):
    """
    Rotates the given point cloud.

    Parameters
    ----------

    rotvec : torch.Tensor
        Rotation vectors.
    points : torch.Tensor
        3d point cloud.

    Returns
    -------
    torch.Tensor
        Rotated point cloud.

    """

    if len(rotvec.shape) == 1:
        rotvec = rotvec.repeat(len(points)).reshape(-1, 3)
    theta = torch.linalg.norm(rotvec, dim=1).unsqueeze(1)
    v = torch.divide(rotvec, theta)
    v = torch.where(torch.isnan(v), torch.zeros_like(v), v)
    dot = torch.sum(points * v, dim=1).unsqueeze(1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return cos_theta * points + sin_theta * torch.cross(v, points) + dot * (1 - cos_theta) * v
