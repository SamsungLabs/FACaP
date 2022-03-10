from facap.geometry.torch import unproject_points_rotvec, project_points_rotvec


def reprojection_error(left, right, depth_scale=1000):
    unprojected_left = unproject_points_rotvec(left["depth"], left["points"], left["f"],
                                               left["pp"], left["rotvecs"], left["translations"],
                                               scale=depth_scale)
    projected_left_on_right = project_points_rotvec(unprojected_left, right["f"], right["pp"],
                                                    right["rotvec"], right["translation"])

    r = (right["points"] - projected_left_on_right)

    return r
