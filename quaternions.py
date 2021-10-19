import torch


def quaternion_to_R(q):
    """
    Works on batch of quaternions.
    `q` must be a batch of unit quaternions with shape (4, n).
    Quaternions must be in the "real part first" configuration (qw, qx, qy, qz).
    Returns (n, 3, 3) rotations matrices.
    """
    qw, qx, qy, qz = q
    return torch.stack([
        1 - 2 * (torch.square(qy) + torch.square(qz)), 2 * (qx * qy - qw * qz), 2 * (qw * qy + qx * qz),
        2 * (qx * qy + qw * qz), 1 - 2 * (torch.square(qx) + torch.square(qz)), 2 * (qy * qz - qw * qx),
        2 * (qx * qz - qw * qy), 2 * (qw * qx + qy * qz), 1 - 2 * (torch.square(qx) + torch.square(qy))
    ]).reshape(3, 3, -1).permute(2, 0, 1)


def angle_between_quaternions(q, r):
    """
    Works on batchs of quaternions only.
    `q` and `r` must be batchs of unit quaternions with shape (n, 4).
    """
    return 2 * torch.sum(q * r, dim=1).abs().clip(0, 1).arccos()
