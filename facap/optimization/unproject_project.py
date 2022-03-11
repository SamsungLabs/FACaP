import torch
from torch import nn

from facap.data.scan import Camera
from facap.geometry.torch import unproject_points_rotvec, project_points_rotvec


class UnprojectProject(nn.Module):
    def __init__(self, cameras):
        super(UnprojectProject, self).__init__()
        self.cameras = sorted(list(cameras.keys()))
        self.cam2id = dict(zip(self.cameras, range(len(self.cameras))))
        self.rotvecs = nn.ParameterDict(
            {id: nn.Parameter(torch.from_numpy(cameras[id].rotvec)) for id in cameras})
        self.translations = nn.ParameterDict(
            {id: nn.Parameter(torch.from_numpy(cameras[id].translation)) for id in cameras})
        f = torch.stack([torch.tensor(cameras[i].f) for i in cameras])
        pp = torch.stack([torch.tensor(cameras[i].pp) for i in cameras])
        self.register_buffer("f", f)
        self.register_buffer("pp", pp)

    def forward(self, depths, points, left_cam_ids, right_cam_ids):
        if len(left_cam_ids) == 1:
            left_cam_ids = list(left_cam_ids) * len(depths)
            right_cam_ids = list(right_cam_ids) * len(depths)
        left_camera = []
        right_camera = []

        for part, cam_ids in zip([left_camera, right_camera], [left_cam_ids, right_cam_ids]):
            for attr in ["f", "pp"]:
                data = torch.stack([getattr(self, attr)[self.cam2id[i]] for i in cam_ids])
                part.append(data)
            for attr in ["rotvecs", "translations"]:
                data = torch.stack([getattr(self, attr)[i] for i in cam_ids])
                part.append(data)
        print(depths.shape, points.shape, left_camera[0].shape)
        pcd = unproject_points_rotvec(depths, points, *left_camera)

        right_points = project_points_rotvec(pcd, *right_camera)

        return right_points

    def get_cameras(self):
        cameras = {}

        for cam_id in self.cameras:
            rotvec = self.rotvecs[cam_id].cpu().detach().numpy()
            translation = self.translations[cam_id].cpu().detach().numpy()
            f = self.f[self.cam2id[cam_id]].cpu().detach().numpy()
            pp = self.pp[self.cam2id[cam_id]].cpu().detach().numpy()
            cameras[cam_id] = Camera(f, pp, rotvec, translation)

        return cameras
