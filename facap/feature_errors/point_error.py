def point_error(left, right, unproject, distance_function, **kwargs):
    processed_left = unproject(left["depths"], left["points"], left["camera_idxs"])
    processed_right = unproject(right["depths"], right["points"], right["camera_idxs"])

    return distance_function(processed_left, processed_right)
