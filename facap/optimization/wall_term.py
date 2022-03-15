import torch

from torch import nn


def nearest_segment_distance(floorplan, pcd_2d):
    biases = torch.stack([i[:2] for i in floorplan])
    directions = torch.stack([i[2:] - i[:2] for i in floorplan])

    distances = []

    for a, b in zip(directions, biases):
        c = pcd_2d - b
        dot_product = torch.sum(c * a, dim=-1) / a.dot(a)

        par_dist = torch.zeros_like(dot_product)
        par_dist[dot_product < 0] = - dot_product[dot_product < 0] * torch.norm(a)
        par_dist[dot_product > 1.] = (dot_product[dot_product > 1.] - 1) * torch.norm(a)
        ort_dst = torch.abs(c[:, 0] * a[1] - c[:, 1] * a[0]) / torch.norm(a)
        distances.append(torch.sqrt(par_dist ** 2 + ort_dst ** 2))

    distances = torch.stack(distances)
    distances = torch.min(distances, dim=0).values
    return distances


class WallTerm(nn.Module):
    def __init__(self, wall, unproject, distance_function, floorplan):
        super(WallTerm, self).__init__()
        self.wall = wall
        self.unproject = unproject
        self.distance_function = distance_function
        self.register_buffer("floorplan", floorplan)

    def forward(self):
        wall_pcd = self.unproject(self.wall["depths"], self.wall["points"], self.wall["camera_idxs"])
        distances = nearest_segment_distance(self.floorplan, wall_pcd[:, [0, 2]])
        zeros = torch.zeros_like(distances)

        return self.distance_function(zeros, distances)
