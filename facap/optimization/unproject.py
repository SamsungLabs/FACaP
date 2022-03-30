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
        rotvecs = torch.stack([self.camera_parameters.rotvecs[i] for i in cam_ids])
        translations = torch.stack([self.camera_parameters.translations[i] for i in cam_ids])
        f = torch.stack([self.camera_parameters.f[self.camera_parameters.cam2id[i]] for i in cam_ids])
        pp = torch.stack([self.camera_parameters.pp[self.camera_parameters.cam2id[i]] for i in cam_ids])
        pcd = unproject_points_rotvec(depths, points, f, pp, rotvecs, translations, scale=self.scale)

        return pcd
