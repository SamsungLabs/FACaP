import torch


def ray_distance(a0, a, b0, b):
    c = torch.cross(a, b)
    c = c / torch.norm(c, dim=-1, keepdim=True)
    dr = torch.unsqueeze(torch.sum(c * (b0 - a0), dim=-1), dim=-1)
    a0 += dr * c

    def get_coef(a, b, a0, b0):
        return ((b0[:, 1] - a0[:, 1]) * a[:, 0] - (b0[:, 0] - a0[:, 0]) * a[:, 1]) / \
               (b[:, 0] * a[:, 1] - a[:, 0] * b[:, 1])

    beta = get_coef(a, b, a0, b0)
    alpha = get_coef(b, a, b0, a0)

    beta = torch.unsqueeze(beta, dim=-1)
    alpha = torch.unsqueeze(alpha, dim=-1)

    return (dr.squeeze(dim=-1),
            alpha.squeeze(dim=-1),
            beta.squeeze(dim=-1))


def ray_error(left, right, unproject, project, distance_function, max_depth=3., parallel_eps=1e-2,
              depths_weight=0.1, depths_scale=1000, **kwargs):
    zero_depth = torch.zeros_like(left["depths"])
    ones_depth = torch.ones_like(left["depths"])

    a0 = unproject(zero_depth, left["points"], left["camera_idxs"])
    b0 = unproject(zero_depth, right["points"], right["camera_idxs"])

    a1 = unproject(ones_depth, left["points"], left["camera_idxs"])
    b1 = unproject(ones_depth, right["points"], right["camera_idxs"])

    a = a1 - a0
    b = b1 - b0
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)

    dr, left_est_depth, right_est_depth = ray_distance(a0, a, b0, b)

    c = torch.cross(a, b)
    norm_c = torch.norm(c, dim=-1)

    mask = (left_est_depth > 0) & (right_est_depth > 0) & (left_est_depth < max_depth) \
           & (right_est_depth < max_depth) & (norm_c > parallel_eps)

    dr = dr[mask]
    scaled_right_depth = right["depth"] / depths_scale
    scaled_left_depth = left["depth"] / depths_scale

    zeros = torch.zeros_like(dr)
    ray_term = distance_function(zeros, dr)
    depths_distance = distance_function(scaled_right_depth[mask], right_est_depth[mask])
    depths_distance += distance_function(scaled_left_depth[mask], left_est_depth[mask])

    return ray_term + depths_weight * depths_distance
