import torch
from torch import nn

from facap.geometry.torch import project_points_rotvec


class Project(nn.Module):
    def __init__(self, camera_parameters):
        super(Project, self).__init__()
        self.camera_parameters = camera_parameters

    def forward(self, pcd, cam_ids):
        if len(cam_ids) == 1:
            cam_ids = list(cam_ids) * len(pcd)
        rotvecs = torch.stack([self.camera_parameters.rotvecs[i] for i in cam_ids])
        translations = torch.stack([self.camera_parameters.translations[i] for i in cam_ids])
        f = torch.stack([self.camera_parameters.f[self.camera_parameters.cam2id[i]] for i in cam_ids])
        pp = torch.stack([self.camera_parameters.pp[self.camera_parameters.cam2id[i]] for i in cam_ids])

        yxs = project_points_rotvec(pcd, f, pp, rotvecs, translations)

        return yxs
