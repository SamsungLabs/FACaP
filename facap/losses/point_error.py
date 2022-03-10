from facap.geometry.torch import unproject_points_rotvec


def point_error(left, right, depth_scale=1000):
    unproject = lambda part: unproject_points_rotvec(part["depth"], part["points"], part["f"],
                                                     part["pp"], part["rotvecs"], part["translations"],
                                                     scale=depth_scale)
    left_pcd = unproject(left)
    right_pcd = unproject(right)

    return left_pcd - right_pcd
