import numpy as np


def nearest_line_distance(florplan_edges, pcd_2d, max_dist=10., border_margin=.1, percentile=.9):
    biases = np.array([np.array(i[0]) for i in florplan_edges])
    directions = np.array([np.array(i[1]) - np.array(i[0]) for i in florplan_edges])

    distances = []

    for a, b in zip(directions, biases):
        c = pcd_2d - b
        dot_product = np.sum(c * a, axis=-1) / a.dot(a)

        mask = np.where((dot_product < -border_margin) | (dot_product > 1 + border_margin))

        dst = np.abs(c[:, 0] * a[1] - c[:, 1] * a[0]) / np.linalg.norm(a)

        dst[mask] = max_dist
        dst[dst > max_dist] = max_dist
        distances.append(dst)

    distances = np.stack(distances)
    distances = np.min(distances, axis=0)
    return distances, np.mean(distances), np.percentile(distances, percentile)
