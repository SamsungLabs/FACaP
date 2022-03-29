import torch

from torch import nn

from facap.data.scan import Camera


class CameraParameters(nn.Module):
    def __init__(self, cameras):
        super(CameraParameters, self).__init__()
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

    def get_cameras(self):
        cameras = {}

        for cam_id in self.cameras:
            rotvec = self.rotvecs[cam_id].cpu().detach().numpy()
            translation = self.translations[cam_id].cpu().detach().numpy()
            f = self.f[self.cam2id[cam_id]].cpu().detach().numpy()
            pp = self.pp[self.cam2id[cam_id]].cpu().detach().numpy()
            cameras[cam_id] = Camera(f, pp, rotvec, translation)

        return cameras
