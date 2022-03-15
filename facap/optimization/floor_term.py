import torch
from torch import nn


def fit_plane(pcd):
    mean = torch.mean(pcd, dim=0)
    pcd_c = pcd - mean
    x = torch.matmul(pcd_c.T, pcd_c)
    u, s, v = torch.svd(x)
    abc = v[:, -1]
    d = -torch.dot(abc, mean)
    coefs = torch.cat([abc, d.view(1, )])
    return coefs


class FloorTerm(nn.Module):
    def __init__(self, floor, unproject, distance_function):
        super(FloorTerm, self).__init__()
        self.floor = floor
        self.unproject = unproject

        self.plane = self.get_initial_plane()
        self.distance_function = distance_function

    def get_initial_plane(self):
        with torch.no_grad():
            floor_pcd = self.unproject(self.floor["depths"], self.floor["points"], self.floor["camera_idxs"])
            normed_coefs = fit_plane(floor_pcd)
            return normed_coefs

    def forward(self):
        floor_pcd = self.unproject(self.floor["depths"], self.floor["points"], self.floor["camera_idxs"])
        floor_distances = (torch.sum(floor_pcd * self.plane[:3], dim=-1) + self.plane[3])
        zeros = torch.zeros_like(floor_distances)
        floor_term = self.distance_function(zeros, floor_distances)
        return floor_term
