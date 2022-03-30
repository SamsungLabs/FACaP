import torch

from torch import nn

from facap.data.scan import Camera


class CameraParameters(nn.Module):
    def __init__(self, cameras):
        super(CameraParameters, self).__init__()
        self.cameras = sorted(list(cameras.keys()))
        self.cam2id = dict(zip(self.cameras, range(len(self.cameras))))
        self.rotvecs = nn.Parameter(torch.stack([torch.from_numpy(cameras[i].rotvec) for i in cameras]))
        self.translations = nn.Parameter(torch.stack([torch.from_numpy(cameras[i].translation) for i in cameras]))
        f = torch.stack([torch.tensor(cameras[i].f) for i in cameras])
        pp = torch.stack([torch.tensor(cameras[i].pp) for i in cameras])
        self.register_buffer("f", f)
        self.register_buffer("pp", pp)

    def get_cameras(self):
        cameras = {}

        for cam_id in self.cameras:
            idx = self.cam2id[cam_id]
            rotvec = self.rotvecs[idx].cpu().detach().numpy()
            translation = self.translations[idx].cpu().detach().numpy()
            f = self.f[idx].cpu().detach().numpy()
            pp = self.pp[idx].cpu().detach().numpy()
            cameras[cam_id] = Camera(f, pp, rotvec, translation)

        return cameras

    def get_tensors(self, camera_idxs):
        idxs = [self.cam2id[i] for i in camera_idxs]
        rotvecs = self.rotvecs[idxs]
        translations = self.translations[idxs]
        f = self.f[idxs]
        pp = self.pp[idxs]
        return rotvecs, translations, f, pp
