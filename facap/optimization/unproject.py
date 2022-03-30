import torch
from torch import nn

from facap.geometry.torch import unproject_points_rotvec


class Unproject(nn.Module):
    def __init__(self, camera_parameters, scale):
        super(Unproject, self).__init__()
        self.camera_parameters = camera_parameters
        self.scale = scale

    def forward(self, depths, points, cam_ids):
        if len(cam_ids) == 1:
            cam_ids = list(cam_ids) * len(depths)
        rotvecs, translations, f, pp = self.camera_parameters.get_tensors(cam_ids)
        pcd = unproject_points_rotvec(depths, points, f, pp, rotvecs, translations, scale=self.scale)

        return pcd
