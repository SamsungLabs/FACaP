
def reprojection_error(left, right, unproject, project, distance_function, **kwargs):
    unprojected_left = unproject(left["depths"], left["points"], left["camera_idxs"])
    projected_left_on_right = project(unprojected_left, right["camera_idxs"])

    return distance_function(right["points"], projected_left_on_right)
