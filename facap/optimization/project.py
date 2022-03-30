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
        rotvecs, translations, f, pp = self.camera_parameters.get_tensors(cam_ids)

        yxs = project_points_rotvec(pcd, f, pp, rotvecs, translations)

        return yxs
