import torch

from quaternions import angle_between_quaternions
from utils import l1_loss, l2_loss, compute_ABC, project


class LocalHomographyLoss(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()

        # `c_n` is the normal vector of the plane inducing the homographies in the ground-truth camera frame
        self.c_n = torch.tensor([0, 0, -1], dtype=torch.float32, device=device).view(3, 1)

        # `eye` is the (3, 3) identity matrix
        self.eye = torch.eye(3, device=device)

    def __call__(self, batch):
        A, B, C = compute_ABC(batch['w_t_c'], batch['c_R_w'], batch['w_t_chat'], batch['chat_R_w'], self.c_n, self.eye)

        xmin = batch['xmin'].view(-1, 1, 1)
        xmax = batch['xmax'].view(-1, 1, 1)
        B_weight = torch.log(xmax / xmin) / (xmax - xmin)
        C_weight = xmin * xmax

        error = A + B * B_weight + C / C_weight
        error = error.diagonal(dim1=1, dim2=2).sum(dim=1).mean()
        return error


class GlobalHomographyLoss(torch.nn.Module):
    def __init__(self, xmin, xmax, device='cuda'):
        """
        `xmin` is the minimum distance of observations across all frames.
        `xmax` is the maximum distance of observations across all frames.
        """
        super().__init__()

        # `xmin` is the minimum distance of observations in all frames
        xmin = torch.tensor(xmin, dtype=torch.float32, device=device)

        # `xmax` is the maximum distance of observations in all frames
        xmax = torch.tensor(xmax, dtype=torch.float32, device=device)

        # `B_weight` and `C_weight` are the weigths of matrices A and B computed from `xmin` and `xmax`
        self.B_weight = torch.log(xmin / xmax) / (xmax - xmin)
        self.C_weight = xmin * xmax

        # `c_n` is the normal vector of the plane inducing the homographies in the ground-truth camera frame
        self.c_n = torch.tensor([0, 0, -1], dtype=torch.float32, device=device).view(3, 1)

        # `eye` is the (3, 3) identity matrix
        self.eye = torch.eye(3, device=device)

    def __call__(self, batch):
        A, B, C = compute_ABC(batch['w_t_c'], batch['c_R_w'], batch['w_t_chat'], batch['chat_R_w'], self.c_n, self.eye)

        error = A + B * self.B_weight + C / self.C_weight
        error = error.diagonal(dim1=1, dim2=2).sum(dim=1).mean()
        return error


class PoseNetLoss(torch.nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def __call__(self, batch):
        t_error = l2_loss(batch['w_t_chat'], batch['w_t_c'])
        q_error = l2_loss(batch['chat_q_w'], batch['c_q_w'])
        error = t_error + self.beta * q_error
        return error


class HomoscedasticLoss(torch.nn.Module):
    def __init__(self, s_hat_t, s_hat_q, device='cuda'):
        super().__init__()
        self.s_hat_t = torch.nn.Parameter(torch.tensor(s_hat_t, dtype=torch.float32, device=device))
        self.s_hat_q = torch.nn.Parameter(torch.tensor(s_hat_q, dtype=torch.float32, device=device))

    def __call__(self, batch):
        LtI = l1_loss(batch['w_t_chat'], batch['w_t_c'])
        LqI = l1_loss(batch['normalized_chat_q_w'], batch['c_q_w'])
        error = LtI * torch.exp(-self.s_hat_t) + self.s_hat_t + LqI * torch.exp(-self.s_hat_q) + self.s_hat_q
        return error


class GeometricLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, batch):
        error = 0
        for w_t_c, c_R_w, w_t_chat, chat_R_w, w_P in zip(batch['w_t_c'], batch['c_R_w'], batch['w_t_chat'],
                                                         batch['chat_R_w'], batch['w_P']):
            c_p = project(w_t_c, c_R_w, w_P)
            chat_p = project(w_t_chat, chat_R_w, w_P)
            error += l1_loss(chat_p.T, c_p.T, reduce='none').clip(0, 100).mean()
        error = error / batch['w_t_c'].shape[0]
        return error


class DSACLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, batch):
        t_error = 100 * l2_loss(batch['w_t_chat'], batch['w_t_c'], reduce='none')
        q_error = angle_between_quaternions(batch['normalized_chat_q_w'], batch['c_q_w']).rad2deg()
        error = torch.max(
            t_error.view(-1),
            q_error
        ).mean()
        return error
