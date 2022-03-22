import torch

from torch import nn


def point_segment_distance(pcd_2d, segments):
    biases = torch.stack([i[:2] for i in segments])
    directions = torch.stack([i[2:] - i[:2] for i in segments])

    c = pcd_2d - biases
    dot_product = torch.sum(c * directions, dim=-1) / torch.sum(directions * directions, dim=-1)

    par_dist = torch.zeros_like(dot_product)
    mask = dot_product < 0
    par_dist[mask] = - dot_product[mask] * torch.norm(directions, dim=-1)[mask]
    mask = dot_product > 1.
    par_dist[dot_product > 1.] = (dot_product[mask] - 1) * torch.norm(directions, dim=-1)[mask]
    ort_dst = torch.abs(c[:, 0] * directions[:, 1] - c[:, 1] * directions[:, 0]) / torch.norm(directions, dim=-1)

    return par_dist, ort_dst


class WallSegmentTerm(nn.Module):
    def __init__(self, wall, unproject, distance_function, floorplan):
        super(WallSegmentTerm, self).__init__()
        self.wall = wall
        self.unproject = unproject
        self.distance_function = distance_function
        self.register_buffer("floorplan", floorplan)

    def forward(self):
        wall_pcd = self.unproject(self.wall["depths"], self.wall["points"], self.wall["camera_idxs"])
        par_dist, ort_dst = point_segment_distance(wall_pcd[:, [0, 2]], self.wall["segments"])
        zeros = torch.zeros_like(par_dist)

        return self.distance_function(zeros, par_dist) + self.distance_function(zeros, ort_dst)
